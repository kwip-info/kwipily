"""Standalone genuine native hierarchy, scalar predictions/influence and helpers."""
import copy,hashlib,importlib.util,io,json,subprocess,sys
from pathlib import Path
import pytest,torch
from runner import engine,initialization
from runner.native_ownership import native_owners
from runner.tests.test_module_composition_execution import fixture,assign,scalar
from runner.tests.test_module_composition_lifecycle import composed,setup,DEVICES
from runner.tests.test_gradient_engine import saved,policy
from runner.tests.test_trainability import recipe as flags,owner
from runner.tests.test_initialization import recipe,rule
from runner.tests.test_gradient_export import load
from runner.tests.test_requested_gradients import request

@pytest.fixture(autouse=True)
def preserve():
    state=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled()
    yield
    torch.use_deterministic_algorithms(state,warn_only=warn)

@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('kind,branch',[('graph',True),('sequential',False),('orderedList',True),('namedMap',True)])
def test_isolated_portable_actual_native_hierarchy_math_influences_and_state(device,kind,branch,tmp_path):
    graph=fixture(kind,branch);native=engine.GraphModel(graph,bundled_dataset=False).to(device);parameters=assign(native)
    path=tmp_path/'model.py';path.write_text(engine.export_source(native));text=path.read_text()
    assert 'runner' not in text and 'kwipml' not in text.lower().replace('exported kwipml','')
    module=load(path);model=module.Model().to(device);model.load_state_dict(native.state_dict(),strict=True)
    assert list(model.state_dict())==list(native.state_dict())
    assert all(a is not b for a,b in zip(native.parameters(),model.parameters()))
    cls={'graph':torch.nn.Module,'sequential':torch.nn.Sequential,'orderedList':torch.nn.ModuleList,'namedMap':torch.nn.ModuleDict}[kind]
    assert isinstance(model.get_submodule('composition_root.blocks.0'),cls)
    assert 'layers' not in model._modules
    rows=[[-2.,.5],[1.25,3.]];upstream=[[.25,-1.],[2.,.5]];expected,dx,dp=scalar(graph,parameters,rows,upstream)
    x=torch.tensor(rows,device=device,requires_grad=True);y,values=model(x,inspect=True)
    assert set(values)==set(native.nodes)
    torch.testing.assert_close(y,torch.tensor(expected,device=device),rtol=2e-5,atol=2e-5)
    (y*torch.tensor(upstream,device=device)).sum().backward()
    torch.testing.assert_close(x.grad,torch.tensor(dx,device=device),rtol=2e-5,atol=2e-5)
    for m,p in native_owners(model):assert p.grad.flatten().tolist()==pytest.approx(dp[(m['node'],m['parameter'])],abs=2e-5)
    if kind=='sequential':
        calls=[];hook=model.get_submodule('composition_root.blocks.0').register_forward_hook(lambda *args:calls.append(1));model(x.detach());hook.remove();assert calls==[1]
    torch.testing.assert_close(torch.cat([model(row[None]) for row in x.detach()]),y.detach(),rtol=1e-6,atol=1e-7)
    # A clean subprocess outside the repository imports only the saved source.
    script="import importlib.util,torch,sys; s=importlib.util.spec_from_file_location('saved',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);x=m.Model();assert not any(k.startswith('runner') for k in sys.modules);assert list(x.state_dict())=="+repr(list(native.state_dict()))
    result=subprocess.run([sys.executable,'-I','-c',script,str(path)],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0,result.stderr

@pytest.mark.parametrize('device',DEVICES)
def test_actual_training_saved_portable_helpers_and_gradients(tmp_path,device):
    graph,kw=setup(tmp_path,device=device);selected=recipe(rule('learn',algorithm='normal',mean=.25,std=0));q=request('sum',output=dict(node='output',port='out'))
    kw.update(initialization_recipe=selected,trainability=flags(owner('learn','weight')),gradient_policy=policy(requests=[q]),derivative_capture=dict(schemaVersion=1,values=True))
    engine.run(graph,tmp_path/'run',epochs=2,**kw);state=saved(tmp_path/'run')
    model=load(tmp_path/'run/model.py').Model().to(device);model.load_state_dict(state['modelState']);model.eval()
    x=kw['prepared'][0][2].to(device);w,b=[v.tolist() for v in state['modelState'].values()]
    expected=[[sum(a*z for a,z in zip(row,weights))+offset for weights,offset in zip(w,b)] for row in x.tolist()]
    torch.testing.assert_close(model(x),torch.tensor(expected,device=device),rtol=2e-6,atol=2e-6)
    initializer=load(tmp_path/'run/initialization_recipe.py');before={k:v.clone() for k,v in model.state_dict().items()}
    # Imported helper does not apply fresh starting values to loaded inference.
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    initializer.apply_initialization(model,initializer.INITIALIZATION_RECIPE)
    assert torch.equal(model.layers[model.layer_keys['learn']].weight,torch.full((2,2),.25,device=device))
    trainable=load(tmp_path/'run/trainability_recipe.py');trainable.apply_trainability(model);assert not model.layers[model.layer_keys['learn']].weight.requires_grad
    helper=load(tmp_path/'run/gradient_policy.py');effect=helper.analyze(model,x[:2],q,device=device)
    assert effect['targets'][0]['values']==pytest.approx([.5,.5]*2)
    assert all(p.grad is None for p in model.parameters())
    reported=copy.deepcopy(q);reported['targets'].append(dict(kind='parameter',node='learn',parameter='weight'));reported['unavailable']='reportUnavailable'
    unavailable=helper.analyze(model,x[:2],reported,device=device)['targets'][1]
    assert unavailable['cause']=='frozenParameter' and 'values' not in unavailable
    trainer=load(tmp_path/'run/training_recipe.py');opt=trainer.create_optimizer(model)
    rows=x[:3].tolist();bias=model.layers[model.layer_keys['learn']].bias.detach().cpu().tolist()
    expected_bias=[offset-.02*sum(2*(.25*sum(row)+offset)/6 for row in rows) for offset in bias]
    window=helper.create_accumulation_state();helper.begin_window(model,window,denominator=6.,expected_microbatches=2)
    helper.backward_contribution(model(x[:2]).square().sum(),window);helper.backward_contribution(model(x[2:3]).square().sum(),window);helper.finish_window(model,opt,window)
    assert model.layers[model.layer_keys['learn']].bias.tolist()==pytest.approx(expected_bias,abs=3e-6,rel=3e-5)
    assert all(p.grad is None for p in model.parameters()) and not window['active']


def test_native_normalization_buffer_state_portable(tmp_path):
    graph=fixture();graph['nodes'][2]={'id':'b','type':'batch_norm1d','config':{'num_features':2}}
    model=engine.GraphModel(graph,bundled_dataset=False);model(torch.tensor([[1.,2.],[3.,-1.]]));model.eval()
    path=tmp_path/'model.py';path.write_text(engine.export_source(model));portable=load(path).Model();portable.load_state_dict(model.state_dict());portable.eval()
    assert list(portable.state_dict())==list(model.state_dict())
    assert list(portable.named_buffers())[-1][1].dtype==torch.int64
    x=torch.tensor([[.5,1.],[2.,3.]]);torch.testing.assert_close(portable(x),model(x),rtol=0,atol=0)

@pytest.mark.parametrize('device',DEVICES)
def test_saved_package_production_inference_full_micro_and_helper_nonexecution(tmp_path,device):
    import shutil
    from runner.tests.test_objective_portable_inference import HELPER,seal
    graph,kw=setup(tmp_path,kind='namedMap',device=device);selected=recipe(rule('learn',algorithm='normal',mean=.2,std=.1));kw['initialization_recipe']=selected
    engine.run(graph,tmp_path/'origin',epochs=2,**kw);state=saved(tmp_path/'origin');x=kw['prepared'][0][2].cpu()
    actual=list(state['modelState'].values());w,b=[v.tolist() for v in actual]
    expected=torch.tensor([[sum(a*z for a,z in zip(row,weights))+offset for weights,offset in zip(w,b)] for row in x.tolist()])
    target=tmp_path/'package';target.mkdir()
    for name in ('model.py','checkpoint.pt','objective_recipe.py','training_recipe.py','initialization_recipe.py'):shutil.copyfile(tmp_path/'origin'/name,target/name)
    shutil.copyfile(HELPER,target/'inference.py')
    contract=dict(schemaVersion=2,input=dict(dtype='float32',shape=['N',2],featureColumns=kw['target_mapping']['inputColumns'],preprocessing='alreadyPrepared'),output=dict(kind='modelOutput',dtype='float32',shape=['N',2]),objective=dict(recipe=kw['objective_recipe'],source=dict(domain='control',localID='objective'),targetMapping=kw['target_mapping'],scalarization='mean'))
    (target/'input-contract.json').write_text(json.dumps(contract))
    # All optional training helpers are immutable inventory items, never executed.
    for name in ('objective_recipe.py','training_recipe.py','initialization_recipe.py'):(target/name).write_text("raise AssertionError('Training helper executed during inference')\n")
    seal(target);shutil.rmtree(tmp_path/'origin');shutil.rmtree(tmp_path/'source')
    predictor=load(target/'inference.py').load_model()
    for batch in (1,len(x)):torch.testing.assert_close(predictor.predict(x,batch_size=batch),expected,rtol=2e-6,atol=2e-6)
    (target/'input.json').write_text(json.dumps(x.tolist()))
    result=subprocess.run([sys.executable,'-I',str(target/'inference.py'),'--input',str(target/'input.json'),'--batch-size','1'],cwd=target,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    measured=json.loads(result.stdout);assert set(measured)=={'inputKind','modelOutput'}
    torch.testing.assert_close(torch.tensor(measured['modelOutput']),expected,rtol=2e-6,atol=2e-6)


def test_fresh_portable_constructor_rng_and_owner_values_match_reordered_runtime(tmp_path):
    graph=fixture('orderedList',True);children=graph['compositionPlan']['containers'][1]['children'];children.reverse()
    for index,child in enumerate(children):child['key']=str(index)
    torch.manual_seed(123);native=engine.GraphModel(graph,bundled_dataset=False);expected_rng=torch.get_rng_state().clone()
    path=tmp_path/'model.py';path.write_text(engine.export_source(native));module=load(path)
    torch.manual_seed(123);portable=module.Model()
    assert torch.equal(expected_rng,torch.get_rng_state())
    assert list(native.state_dict())==list(portable.state_dict())
    for name,value in native.state_dict().items():assert torch.equal(value,portable.state_dict()[name]),name
