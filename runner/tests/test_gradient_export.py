import copy,importlib.util,json
import pytest,torch
from runner import engine
from runner.tests.test_loss_engine import prepared
from runner.tests.test_gradient_engine import policy,optimizer,saved
from runner.tests.test_trainability import recipe as flags,owner
from runner.tests.test_requested_gradients import request

def load(path):
 spec=importlib.util.spec_from_file_location(path.stem,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
@pytest.mark.parametrize('device',['cpu','mps'])
def test_real_saved_final_derivatives_and_portable_differentiable_forward(tmp_path,device):
 if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
 graph,r,m,data=prepared(tmp_path/'source','loss_mse');q=request('sum',output=dict(node='output',port='out'));selected=policy(requests=[q]);freeze=flags(owner('learn','weight'))
 engine.run(graph,tmp_path/'run',epochs=2,seed=41,batch_size=5,device=device,prepared=data,objective_recipe=r,target_mapping=m,optimizer_recipe=optimizer(),gradient_policy=selected,trainability=freeze,derivative_capture=dict(schemaVersion=1,values=True))
 state=saved(tmp_path/'run');artifact=json.loads((tmp_path/'run/derivative-results.json').read_text());weights=state['modelState']['layers.learn.weight'].tolist()
 expected=[sum(row[i] for row in weights) for i in range(2)]*2
 assert artifact['requests'][0]['targets'][0]['values']==pytest.approx(expected,abs=2e-6)
 assert artifact['source']['checkpointSHA256']==__import__('hashlib').sha256((tmp_path/'run/checkpoint.pt').read_bytes()).hexdigest()
 model=load(tmp_path/'run/model.py').Model().to(device);model.load_state_dict(state['modelState']);assert not model.layers['learn'].weight.requires_grad
 helper=load(tmp_path/'run/gradient_policy.py');assert not hasattr(helper,'GRADIENT_POLICY_RECIPE')
 assert 'input-effect' not in (tmp_path/'run/gradient_policy.py').read_text() and 'upstream' not in helper.ACCUMULATION_CONFIG
 measured=helper.analyze(model,data[0][2][:2].to(device),q,device=device);assert measured['targets'][0]['values']==pytest.approx(expected,abs=2e-6)
 x=data[0][2][:2].to(device).detach().requires_grad_(True);model(x).sum().backward();assert x.grad is not None
 other=load(tmp_path/'run/trainability_recipe.py');other.apply_trainability(model);assert not model.layers['learn'].weight.requires_grad
 assert (tmp_path/'run/training_recipe.py').stat().st_size<=8192
@pytest.mark.parametrize('device',['cpu','mps'])
def test_portable_actual_window_first_update_analytic_and_state_guards(tmp_path,device):
 if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
 graph,r,m,data=prepared(tmp_path/'source','loss_mse');engine.run(graph,tmp_path/'run',epochs=1,batch_size=5,device=device,prepared=data,objective_recipe=r,target_mapping=m,optimizer_recipe=optimizer(),gradient_policy=policy())
 mod=load(tmp_path/'run/model.py');model=mod.Model().to(device);model.load_state_dict(saved(tmp_path/'run')['modelState']);helper=load(tmp_path/'run/gradient_policy.py');training=load(tmp_path/'run/training_recipe.py');opt=training.create_optimizer(model)
 before={k:v.clone() for k,v in model.state_dict().items()};x=torch.tensor([[1.,2.],[3.,4.],[5.,6.]],device=device);target=torch.zeros(3,2,device=device);state=helper.create_accumulation_state();helper.begin_window(model,state,denominator=6.,expected_microbatches=2)
 with pytest.raises(ValueError):helper.finish_window(model,opt,state)
 expected_w=before['layers.learn.weight'].cpu().tolist();expected_b=before['layers.learn.bias'].cpu().tolist()
 for out in range(2):
  errors=[sum(a*z for a,z in zip(expected_w[out],row))+expected_b[out] for row in x.cpu().tolist()]
  for i in range(2):expected_w[out][i]-=.02*sum(2*e*row[i]/6 for e,row in zip(errors,x.cpu().tolist()))
  expected_b[out]-=.02*sum(2*e/6 for e in errors)
 helper.backward_contribution(((model(x[:2])-target[:2])**2).sum(),state);helper.backward_contribution(((model(x[2:])-target[2:])**2).sum(),state);helper.finish_window(model,opt,state)
 assert model.layers['learn'].weight.detach().cpu().flatten().tolist()==pytest.approx([v for row in expected_w for v in row],abs=3e-6,rel=3e-5)
 assert model.layers['learn'].bias.detach().cpu().tolist()==pytest.approx(expected_b,abs=3e-6,rel=3e-5)
 assert all(p.grad is None for p in model.parameters()) and not state['active']
 with pytest.raises(ValueError):helper.finish_window(model,opt,state)

@pytest.mark.parametrize('predictions',[False,True])
@pytest.mark.parametrize('activations',[False,True])
def test_independent_derivative_permission_never_widens_existing_sample_captures(tmp_path,predictions,activations):
 graph,r,m,data=prepared(tmp_path/'source','loss_mse');q=request('sum',output=dict(node='output',port='out'))
 engine.run(graph,tmp_path/'run',epochs=1,batch_size=5,prepared=data,objective_recipe=r,target_mapping=m,gradient_policy=policy(requests=[q]),derivative_capture=dict(schemaVersion=1,values=True),sample_capture=dict(schemaVersion=1,predictionRows=predictions,activationValues=activations))
 artifact=json.loads((tmp_path/'run/derivative-results.json').read_text());assert set(artifact['requests'][0]['output'])=={'node','port','shape','dtype'}
 pred=json.loads((tmp_path/'run/predictions.json').read_text());assert bool(pred['predictionRows'])==predictions
 capture=json.loads((tmp_path/'run/activations.json').read_text())
 assert bool(capture['previews']['input']['values'])==activations

@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('reduction,scalarization',[('mean','mean'),('sum','mean'),('none','mean'),('none','sum')])
@pytest.mark.parametrize('op',['loss_cross_entropy','loss_bce_with_logits','loss_mse','loss_l1','loss_nll','loss_kl_div','loss_ctc','loss_triplet_margin'])
def test_portable_objective_owned_window_arithmetic(op,device,reduction,scalarization):
 from runner.tests.test_loss_partition import case,recipe
 from runner.gradient_portable import gradient_policy_code
 rows,targets,config,x,roles=case(op,device);config['reduction']=reduction;r=recipe(op,config)
 namespace={'__name__':'portable_window'}
 scale='sum' if reduction=='sum' or reduction=='none' and scalarization=='sum' else 'mean'
 exec(gradient_policy_code(policy(scale=scale),objective_recipe=r,scalarization=scalarization),namespace)
 denominator=namespace['window_denominator'](roles,list(x.shape))
 first=namespace['objective_contribution'](x[:2],{k:v[:2] for k,v in roles.items()})
 second=namespace['objective_contribution'](x[2:],{k:v[2:] for k,v in roles.items()})
 actual=(first+second)/denominator
 # Native whole-window reduction is independent of the exported accumulation arithmetic.
 from runner.loss_objectives import create_objective,evaluate_objective
 native=evaluate_objective(create_objective(r,device),x,roles,scalarization)['scalar']
 assert actual.item()==pytest.approx(native.item(),abs=3e-6,rel=3e-6)
 assert torch.allclose(torch.autograd.grad(actual,x,retain_graph=True)[0],torch.autograd.grad(native,x)[0],atol=3e-6,rtol=3e-6)
