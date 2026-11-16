"""Independent scalar activation values, influence, state and parent-mode contracts."""
import math
import copy
import pytest
import torch
from runner.engine import GraphModel,REGISTRY,export_source
from runner.graph_semantics import effective_config
CONFIG={'prelu':{'num_parameters':2,'init':.25},'log_softmax':{'dim':1},'dropout':{'p':.5}}

def graph(op,config=None):
 rows=[('input','input',{'features':2}),('projection','linear',{'in_features':2,'out_features':6,'bias':True}),('array','reshape',{'shape':[2,3]}),('activation',op,CONFIG[op] if config is None else config),('flat','flatten',{'start_dim':1,'end_dim':-1}),('head','linear',{'in_features':6,'out_features':2,'bias':True}),('output','output',{})]
 return {'schemaVersion':2,'name':'Activation controls','nodes':[{'id':i,'type':t,'config':copy.deepcopy(c)} for i,t,c in rows],'edges':[{'id':str(j),'source':rows[j][0],'sourcePort':'out','target':rows[j+1][0],'targetPort':'in'} for j in range(len(rows)-1)]}

@pytest.mark.parametrize('op',CONFIG)
@pytest.mark.parametrize('device',['cpu','mps'])
def test_new_real_graph_portable_state_and_parent_modes(op,device):
 model=GraphModel(graph(op),REGISTRY).to(device);scope={};exec(export_source(model),scope);portable=scope['Model']().to(device);portable.load_state_dict(model.state_dict(),strict=True)
 x=torch.tensor([[1.,-2.],[-.5,3.]],device=device)
 for training in (False,True):
  model.train(training);portable.train(training);torch.manual_seed(41);actual=model(x);torch.manual_seed(41);expected=portable(x);torch.testing.assert_close(actual,expected,rtol=0,atol=0)
 assert set(model.state_dict())==set(portable.state_dict())

@pytest.mark.parametrize('op',CONFIG)
@pytest.mark.parametrize('config',[None,{'unknown':1},{'inplace':True}])
def test_strict_new_configuration_refusals(op,config):
 with pytest.raises(ValueError):effective_config({'type':op,'config':config},2)

@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('count',[1,2])
def test_prelu_scalar_channel_and_zero_convention_weighted_influence(device,count):
 model=GraphModel(graph('prelu',{'num_parameters':count,'init':-.25}),REGISTRY).to(device)
 slopes=[-.25] if count==1 else [-.25,.5]
 with torch.no_grad():model.layers[model.layer_keys['activation']].weight.copy_(torch.tensor(slopes,device=device))
 values=[-3.,0.,2.,-1.,4.,0.,2.,-5.,1.,0.,-2.,3.];g=[(i%5-2)/3 for i in range(12)]
 storage=torch.zeros(2,2,6,device=device);storage[...,::2]=torch.tensor(values,device=device).reshape(2,2,3);x=storage[...,::2].detach().requires_grad_()
 actual=model.execute_node('activation',{'array':x},None);expected=[];dx=[];dw=[0.]*count
 for i,(v,u) in enumerate(zip(values,g)):
  k=0 if count==1 else (i//3)%2;s=slopes[k];expected.append(v if v>0 else s*v);dx.append(u*(1 if v>0 else s));dw[k]+=u*v if v<=0 else 0
 torch.testing.assert_close(actual.cpu(),torch.tensor(expected).reshape(2,2,3),atol=1e-6,rtol=1e-6)
 a,b=torch.autograd.grad(actual,(x,model.layers[model.layer_keys['activation']].weight),torch.tensor(g,device=device).reshape(2,2,3));torch.testing.assert_close(a.cpu(),torch.tensor(dx).reshape(2,2,3),atol=1e-6,rtol=1e-6);torch.testing.assert_close(b.cpu(),torch.tensor(dw),atol=1e-6,rtol=1e-6)

@pytest.mark.parametrize('op,bad',[
 ('prelu',{'num_parameters':True}),('prelu',{'num_parameters':0}),('prelu',{'num_parameters':4097}),('prelu',{'num_parameters':1.5}),('prelu',{'init':True}),('prelu',{'init':float('inf')}),('prelu',{'init':float('nan')}),('prelu',{'init':3.5e38}),
 ('log_softmax',{'dim':0}),('log_softmax',{'dim':True}),('log_softmax',{'dim':1.5}),('log_softmax',{'dim':8}),('log_softmax',{'dim':None}),
 ('dropout',{'p':True}),('dropout',{'p':-0.1}),('dropout',{'p':1.1}),('dropout',{'p':None}),('dropout',{'p':float('nan')}),('dropout',{'p':float('inf')})])
def test_bounded_configuration_negatives(op,bad):
 from runner.remaining_activation_configuration import normalize_remaining_activation
 with pytest.raises(ValueError):normalize_remaining_activation(op,bad)

@pytest.mark.parametrize('op',CONFIG)
def test_shape_schema_numeric_aliases_and_defaults(op):
 from runner.remaining_activation_configuration import normalize_remaining_activation,resolve_remaining_activation
 from runner.tensor_layout import BatchedLayout
 assert normalize_remaining_activation(op,{})
 with pytest.raises(ValueError):effective_config({'type':op,'config':{}},1)
 if op=='prelu':
  assert normalize_remaining_activation(op,{'num_parameters':2.0})['num_parameters']==2
  assert resolve_remaining_activation(op,{'num_parameters':4096,'init':-3.4028234663852886e38},BatchedLayout((4096,)))['parameterCount']==4096
  with pytest.raises(ValueError):resolve_remaining_activation(op,{'num_parameters':3},BatchedLayout((2,3)))
 if op=='log_softmax':
  with pytest.raises(ValueError):resolve_remaining_activation(op,{'dim':-3},BatchedLayout((2,3)))
  with pytest.raises(ValueError):resolve_remaining_activation(op,{'dim':4},BatchedLayout((2,3)))

@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('dim',[1,2,-1,-2])
def test_logsoftmax_scalar_stability_and_weighted_input_influence(device,dim):
 model=GraphModel(graph('log_softmax',{'dim':dim}),REGISTRY).to(device)
 vals=[10000.,9998.,10002.,10001.,9997.,10000.];weights=[-.5,.25,1.,-.75,.5,-.25]
 storage=torch.zeros(1,2,6,device=device);storage[...,::2]=torch.tensor(vals,device=device).reshape(1,2,3);x=storage[...,::2].detach().requires_grad_();axis=dim%3;out=[0.]*6;gradient=[0.]*6
 groups=[[r*3+c for r in range(2)] for c in range(3)] if axis==1 else [[r*3+c for c in range(3)] for r in range(2)]
 for indices in groups:
  maximum=max(vals[i] for i in indices);exp=[math.exp(vals[i]-maximum) for i in indices];den=math.fsum(exp);total=math.fsum(weights[i] for i in indices)
  for i,e in zip(indices,exp):out[i]=vals[i]-maximum-math.log(den);gradient[i]=weights[i]-e/den*total
 actual=model.execute_node('activation',{'array':x},None);torch.testing.assert_close(actual.cpu(),torch.tensor(out).reshape(1,2,3),atol=2e-6,rtol=2e-6)
 g=torch.autograd.grad(actual,x,torch.tensor(weights,device=device).reshape(1,2,3))[0];torch.testing.assert_close(g.cpu(),torch.tensor(gradient).reshape(1,2,3),atol=2e-6,rtol=2e-6)

@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('p',[0,.5,1])
def test_dropout_parent_mode_successive_seed_schedule_and_endpoints(device,p):
 model=GraphModel(graph('dropout',{'p':p}),REGISTRY).to(device);x=torch.linspace(-2,3,600,device=device).reshape(100,2,3).requires_grad_();g=torch.linspace(-1,1,600,device=device).reshape(x.shape)
 def seed():torch.manual_seed(173)
 model.train();seed();gold=torch.nn.Dropout(p)(torch.ones_like(x));seed();a=model.execute_node('activation',{'array':x},None);b=model.execute_node('activation',{'array':x},None);seed();again=model.execute_node('activation',{'array':x},None);assert torch.equal(a,again)
 if p==.5:assert not torch.equal(a,b)
 scale=0 if p==1 else 1/(1-p);mask=(gold!=0).to(x.dtype);expected=x*mask*scale
 torch.testing.assert_close(a,expected,rtol=0,atol=0);torch.testing.assert_close(torch.autograd.grad(a,x,g)[0],g*mask*scale,rtol=0,atol=0)
 if p==0:assert torch.equal(a,x)
 if p==1:assert torch.count_nonzero(a)==0
 model.eval();seed();before=torch.get_rng_state().clone();actual=model.execute_node('activation',{'array':x},None);after=torch.get_rng_state();assert torch.equal(actual,x) and torch.equal(before,after)
 model.train();after_eval=model.execute_node('activation',{'array':x},None);seed();direct=model.execute_node('activation',{'array':x},None);assert torch.equal(after_eval,direct)

@pytest.mark.parametrize('op',CONFIG)
@pytest.mark.parametrize('problem',['dtype','sparse','empty','shape','nonfinite','budget'])
def test_actual_activation_boundary_before_allocation(op,problem):
 model=GraphModel(graph(op),REGISTRY);x=torch.ones(2,2,3)
 if problem=='dtype':x=x.double()
 elif problem=='sparse':x=x.to_sparse()
 elif problem=='empty':x=x[:0]
 elif problem=='shape':x=x[...,0]
 elif problem=='nonfinite':x[0,0,0]=float('nan')
 else:x=x[:1].expand(100_000_001,2,3)
 with pytest.raises(ValueError):model.execute_node('activation',{'array':x},None)

@pytest.mark.parametrize('op',['prelu','dropout'])
def test_cpu_exact_resume_new_state_and_rng_and_weights_only_finetune(op,tmp_path):
 from runner.engine import run
 from runner.tests.test_checkpoint_lifecycle import assert_nested_equal
 g=graph(op);options={'sample_limit':32,'batch_size':8,'device':'cpu','registry':REGISTRY,'emit':lambda event:None}
 for directory,epochs,resume in [('full',2,None),('first',1,None),('resume',2,tmp_path/'first/checkpoint.pt')]:run(g,tmp_path/directory,epochs=epochs,resume_checkpoint=resume,**options)
 full=torch.load(tmp_path/'full/checkpoint.pt',weights_only=True);resumed=torch.load(tmp_path/'resume/checkpoint.pt',weights_only=True)
 for key in ('modelState','optimizerState','rngState'):assert_nested_equal(full[key],resumed[key])
 assert (tmp_path/'full/metrics.json').read_bytes()==(tmp_path/'resume/metrics.json').read_bytes()
 if op=='prelu':
  assert full['contract']['schedule']['parameterNodeOrder']==['projection','activation','head']
  assert not torch.equal(full['modelState']['layers.activation.weight'],torch.full((2,),.25))
 else:assert full['contract']['schedule']['stochasticNodeOrder']==['activation']
 run(g,tmp_path/'finetune',epochs=1,finetune_checkpoint=tmp_path/'first/checkpoint.pt',**options)
 import json
 manifest=json.loads((tmp_path/'finetune/manifest.json').read_text());assert manifest['checkpoint']['contract']['guarantee']=='epochBoundaryExactCPU'

@pytest.mark.parametrize('op',['prelu','dropout'])
def test_mps_training_saved_state_and_honest_weights_only_contract(op,tmp_path):
 from runner.engine import run
 run(graph(op),tmp_path,epochs=1,sample_limit=32,batch_size=8,device='mps',registry=REGISTRY,emit=lambda event:None)
 checkpoint=torch.load(tmp_path/'checkpoint.pt',weights_only=True,map_location='cpu');assert checkpoint['contract']['guarantee']=='weightsOnly'


def test_prelu_distinct_modules_collision_safe_keys_and_fanout_parameter_ownership():
 g=graph('prelu');second=copy.deepcopy(next(n for n in g['nodes'] if n['id']=='activation'));second['id']='training';g['nodes'].insert(4,second)
 # Two separately authored nodes in a chain retain independent slopes/state.
 edge=next(e for e in g['edges'] if e['source']=='activation');edge['source']='training';g['edges'].append({'id':'other','source':'activation','sourcePort':'out','target':'training','targetPort':'in'})
 model=GraphModel(g,REGISTRY);left=model.layers[model.layer_keys['activation']];right=model.layers[model.layer_keys['training']];assert model.layer_keys['training']!='training';assert left.weight.data_ptr()!=right.weight.data_ptr()
 with torch.no_grad():left.weight.fill_(-.5)
 assert right.weight.tolist()==[.25,.25]
 scope={};exec(export_source(model),scope);portable=scope['Model']();portable.load_state_dict(model.state_dict(),strict=True);model.eval();portable.eval();torch.testing.assert_close(model(torch.ones(2,2)),portable(torch.ones(2,2)),rtol=0,atol=0)


def test_new_checkpoint_order_only_changes_new_graphs_and_binds_stochastic_execution():
 from runner.checkpoint_lifecycle import contract,digest
 old=graph('prelu');old['nodes']=[{**n,'type':'relu','config':{}} if n['id']=='activation' else n for n in old['nodes']]
 args=([torch.ones(2,2),torch.ones(2,dtype=torch.int64)],{'sourceHash':'source','splitHash':'split'},42,8,.03,'cpu',1)
 before=contract(old,*args,[n['id'] for n in old['nodes']]);assert 'parameterNodeOrder' not in before['schedule'] and 'stochasticNodeOrder' not in before['schedule']
 new=graph('dropout');extra=copy.deepcopy(next(n for n in new['nodes'] if n['id']=='activation'));extra['id']='second';new['nodes'].append(extra)
 a=contract(new,*args,['input','projection','array','activation','second','flat','head','output']);b=contract(new,*args,['input','projection','array','second','activation','flat','head','output']);assert a['semanticGraphDigest']==b['semanticGraphDigest'] and a['schedule']['stochasticNodeOrder']!=b['schedule']['stochasticNodeOrder']

@pytest.mark.parametrize('op',['prelu','log_softmax'])
def test_actual_nonfinite_result_refuses_even_finite_input(op):
 c={'num_parameters':2,'init':3.4028234663852886e38} if op=='prelu' else {'dim':1};model=GraphModel(graph(op,c),REGISTRY)
 x=torch.tensor([[-2.,1.,0.],[2.,-1.,0.]]).unsqueeze(0)
 if op=='log_softmax':x[0,0,:]=-3.4028234663852886e38;x[0,1,:]=3.4028234663852886e38
 with pytest.raises(ValueError,match='nonfinite'):model.execute_node('activation',{'array':x},None)


def test_cpu_dropout_frozen_upstream_mask_bank_float32_parity_and_weighted_influence():
 import json
 from pathlib import Path
 bank=json.loads((Path(__file__).resolve().parents[1]/'fixtures/remaining_activation_dropout_masks_v1.json').read_text());assert bank['torchVersion']==str(torch.__version__)
 model=GraphModel(graph('dropout'),REGISTRY).train()
 for entry in bank['entries']:
  shape=tuple(entry['shape']);count=math.prod(shape);values=[(i%11-5)/3 for i in range(count)];upstream=[(i%7-3)/4 for i in range(count)];expected=[v*m*2 for v,m in zip(values,entry['mask'])];influence=[g*m*2 for g,m in zip(upstream,entry['mask'])]
  x=torch.tensor(values).reshape(shape).requires_grad_();torch.manual_seed(entry['seed']);actual=model.execute_node('activation',{'array':x},None)
  torch.testing.assert_close(actual,torch.tensor(expected).reshape(shape),rtol=0,atol=0);torch.testing.assert_close(torch.autograd.grad(actual,x,torch.tensor(upstream).reshape(shape))[0],torch.tensor(influence).reshape(shape),rtol=0,atol=0)


def test_actual_prelu_parameters_cross_total_budget_before_constructor(monkeypatch):
 from runner.engine import validate
 widths=[2,512,*([512]*7),310,2];rows=[('input','input',{'features':2})]
 for i,(a,b) in enumerate(zip(widths,widths[1:])):
  if b==2:rows.append(('activation','prelu',{'num_parameters':310,'init':.25}))
  rows.append((f'linear{i}','linear',{'in_features':a,'out_features':b,'bias':True}))
 rows.append(('output','output',{}));g={'schemaVersion':2,'name':'Slope budget','nodes':[{'id':i,'type':t,'config':c} for i,t,c in rows],'edges':[{'id':str(i),'source':a[0],'sourcePort':'out','target':b[0],'targetPort':'in'} for i,(a,b) in enumerate(zip(rows,rows[1:]))]}
 plain=copy.deepcopy(g);next(n for n in plain['nodes'] if n['id']=='activation').update(type='relu',config={});validate(plain,REGISTRY)
 calls=[];monkeypatch.setattr(torch.nn,'PReLU',lambda *a,**k:calls.append(True))
 with pytest.raises(ValueError,match='2 million'):GraphModel(g,REGISTRY)
 assert calls==[]

@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('init',[0,-.5,.25])
def test_prelu_rank_eight_sample_ownership_and_reinitialization(device,init):
 g=graph('prelu',{'num_parameters':2,'init':init});next(n for n in g['nodes'] if n['id']=='array')['config']['shape']=[2,1,1,1,1,1,3]
 model=GraphModel(g,REGISTRY).to(device);layer=model.layers[model.layer_keys['activation']];assert layer.weight.tolist()==[init,init]
 x=torch.tensor([-2.,0.,3.,-1.,2.,0.],device=device).reshape(1,2,1,1,1,1,1,3).requires_grad_();actual=model.execute_node('activation',{'array':x},None);expected=torch.tensor([init*-2,0,3,init*-1,2,0],device=device,dtype=torch.float32).reshape(x.shape);torch.testing.assert_close(actual,expected,rtol=0,atol=0)
 gradient=torch.autograd.grad(actual.sum(),x)[0];torch.testing.assert_close(gradient,torch.tensor([init,init,1,init,1,init],device=device,dtype=torch.float32).reshape(x.shape),rtol=0,atol=0)
 with torch.no_grad():layer.weight.fill_(.75)
 fresh=GraphModel(g,REGISTRY).to(device);assert fresh.layers[fresh.layer_keys['activation']].weight.tolist()==[init,init]
 wrong=copy.deepcopy(model.state_dict());wrong['layers.activation.weight']=torch.ones(3,device=device)
 with pytest.raises(RuntimeError):fresh.load_state_dict(wrong,strict=True)


def test_prelu_fanout_reuses_one_learned_slope_set_with_accumulated_influence():
 g=graph('prelu');g['nodes'].insert(4,{'id':'combine','type':'add','config':{}})
 edge=next(e for e in g['edges'] if e['source']=='activation');edge['source']='combine'
 for p in ('a','b'):g['edges'].append({'id':'fanout-'+p,'source':'activation','sourcePort':'out','target':'combine','targetPort':p})
 model=GraphModel(g,REGISTRY);assert sum(isinstance(layer,torch.nn.PReLU) for layer in model.layers.values())==1
 x=torch.tensor([-2.,0.,3.,-1.,2.,0.]).reshape(1,2,3).requires_grad_();values={'array':x};values['activation']=model.execute_node('activation',values,None);out=model.execute_node('combine',values,None)
 gradient=torch.autograd.grad(out.sum(),model.layers[model.layer_keys['activation']].weight)[0];torch.testing.assert_close(gradient,torch.tensor([-4.,-2.]),rtol=0,atol=0)

@pytest.mark.parametrize('op',CONFIG)
def test_actual_authored_generated_discovery_descriptor_equality(op,monkeypatch):
 import importlib,json
 from pathlib import Path
 from runner.engine import capabilities
 from runner.remaining_activation_configuration import capability
 root=Path(__file__).resolve().parents[2];bundle=json.loads((root/'components/activations/bundle.json').read_text());authored=next(c['runtimeContract'] for c in bundle['components'] if c['definition']['id']==op)
 monkeypatch.syspath_prepend(str(root/'runner'));generated=importlib.import_module('runner.remaining_activation_adapters_generated');assert authored==capability(op)==generated.CAPABILITIES[op]()==capabilities()['componentCapabilities'][op]

@pytest.mark.parametrize('op',['prelu','dropout'])
def test_actual_multi_node_order_changes_refuse_exact_resume_even_equal_semantic_graph(op,tmp_path):
 from runner.engine import run
 from runner.graph_semantics import semantic_graph
 g=graph(op);g['nodes'].insert(4,{'id':'second','type':op,'config':copy.deepcopy(CONFIG[op])});g['nodes'].insert(5,{'id':'combine','type':'add','config':{}})
 for edge in g['edges']:
  if edge['source']=='activation':edge['source']='combine'
 for source,target,port in [('array','second','in'),('activation','combine','a'),('second','combine','b')]:g['edges'].append({'id':f'{source}-{target}','source':source,'sourcePort':'out','target':target,'targetPort':port})
 changed=copy.deepcopy(g);a=next(i for i,n in enumerate(changed['nodes']) if n['id']=='activation');b=next(i for i,n in enumerate(changed['nodes']) if n['id']=='second');changed['nodes'][a],changed['nodes'][b]=changed['nodes'][b],changed['nodes'][a]
 # Edge order owns ready-node insertion in the actual graph executor too.
 changed['edges'].reverse()
 assert semantic_graph(g)==semantic_graph(changed)
 model=GraphModel(g,REGISTRY);other=GraphModel(changed,REGISTRY)
 relevant=lambda m:[i for i in m.order if m.nodes[i]['type']==op]
 assert relevant(model)!=relevant(other)
 options={'sample_limit':32,'batch_size':8,'device':'cpu','registry':REGISTRY,'emit':lambda event:None};run(g,tmp_path/'first',epochs=1,**options)
 with pytest.raises(ValueError,match='Exact resume'):run(changed,tmp_path/'changed',epochs=2,resume_checkpoint=tmp_path/'first/checkpoint.pt',**options)

@pytest.mark.parametrize('device',['cpu','mps'])
def test_prelu_actual_gradient_capture_tracks_learned_slopes_and_finetune_reset(device,tmp_path):
 import json
 from runner.engine import run
 options={'sample_limit':32,'batch_size':8,'device':device,'registry':REGISTRY,'emit':lambda event:None};run(graph('prelu'),tmp_path/'first',epochs=1,gradient_capture={'schemaVersion':1,'everyEpochs':1,'maxBatchesPerEpoch':1,'maxCaptures':1},**options)
 report=json.loads((tmp_path/'first/gradient-summary.json').read_text());slopes=[row for row in report['inventory']['parameters'] if row['nodeID']=='activation'];assert len(slopes)==1 and slopes[0]['shape']==[2] and slopes[0]['elementCount']==2
 run(graph('prelu'),tmp_path/'child',epochs=1,finetune_checkpoint=tmp_path/'first/checkpoint.pt',learning_rate=.01,seed=173,**options)
 manifest=json.loads((tmp_path/'child/manifest.json').read_text());assert manifest['lineage']['mode']=='weightsOnlyFineTune';checkpoint=torch.load(tmp_path/'child/checkpoint.pt',weights_only=True,map_location='cpu');assert checkpoint['epoch']==1 and checkpoint['seed']==173 and all(value['step'].item()==4 for value in checkpoint['optimizerState']['state'].values())
