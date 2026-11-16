import copy
import pytest
from runner import gradient_policy as g

def recipe(**changes):return dict(schemaVersion=1,configurationVersion=1,componentID='gradient_policy',config=dict(accumulation_steps=2,gradient_scale='mean',requests=[],**changes))

def test_nondefault_window_not_silently_reset():assert g.canonical_configuration({'accumulation_steps':2})['accumulation_steps']==2
@pytest.mark.parametrize('bad',[True,1.,0,33,None])
def test_strict_window(bad):
 with pytest.raises(ValueError):g.canonical_configuration({'accumulation_steps':bad})
def test_unknown_config_refused():
 with pytest.raises(ValueError):g.canonical_configuration({'extra':1})

import torch
from runner.tests.test_requested_gradients import request

def wire(config):return dict(schemaVersion=1,configurationVersion=1,componentID='gradient_policy',config=config)
@pytest.mark.parametrize('change',[{'schemaVersion':True},{'configurationVersion':2},{'componentID':'other'},{'extra':1}])
def test_exact_recipe_identity(change):
 value=wire(g.canonical_configuration({}));value.update(change)
 with pytest.raises(ValueError):g.validate_recipe(value)
@pytest.mark.parametrize('change',[{'id':'has space'},{'id':'é'},{'id':''},{'row_start':2**63-1},{'row_start':-1},{'row_count':33},{'row_count':True},{'targets':[]},{'upstream':{}},{'partition':'test'},{'unavailable':'zero'},{'extra':0}])
def test_strict_requests(change):
 value=request();value.update(change)
 with pytest.raises(ValueError):g.canonical_configuration({'requests':[value]})
def test_int64_checked_range_and_duplicate_targets_ids():
 assert g.validate_request(request(row_start=2**63-2,row_count=1))['row_start']==2**63-2
 for values in ([request(),request()],[request(targets=[dict(kind='input',node='input',port='out')]*2)]):
  with pytest.raises(ValueError):g.canonical_configuration({'requests':values})
@pytest.mark.parametrize('upstream',[None,dict(dtype='float64',shape=[2,1],values=[1,2]),dict(dtype='float32',shape=[1,2],values=[1,2]),dict(dtype='float32',shape=[2,1],values=[True,2]),dict(dtype='float32',shape=[2,1],values=[1e-46,2]),dict(dtype='float32',shape=[2,1],values=[float('nan'),2]),dict(dtype='float32',shape=[2,1],values=[1,2,3])])
def test_upstream_admission(upstream):
 with pytest.raises(ValueError):g.validate_request(request(reduction='upstream',upstream=upstream))
def test_upstream_integral_numbers_canonical_real_and_portable_budget():
 value=g.validate_request(request(reduction='upstream',upstream=dict(dtype='float32',shape=[2,1],values=[1,2])))
 assert type(value['upstream']['values'][0]) is float
 with pytest.raises(ValueError):g.canonical_configuration({'requests':[request(output=dict(node='x'*33000,port='out'))]})
@pytest.mark.parametrize('value',[{'schemaVersion':True,'values':True},{'schemaVersion':1,'values':1},{'schemaVersion':1,'values':False,'extra':1}])
def test_permission_strict(value):
 with pytest.raises(ValueError):g.validate_capture(value)
def test_effective_reduction_guard():
 for reduction,scalarization,expected in [('mean','sum','mean'),('batchmean','sum','mean'),('sum','mean','sum'),('none','mean','mean'),('none','sum','sum')]:
  r=wire(g.canonical_configuration({'gradient_scale':expected}));assert g.validate_compatibility(r,{'config':{'reduction':reduction}},scalarization)
  r['config']['gradient_scale']='sum' if expected=='mean' else 'mean'
  with pytest.raises(ValueError):g.validate_compatibility(r,{'config':{'reduction':reduction}},scalarization)

from runner.tests.test_loss_partition import case,IDS,recipe as loss_recipe
from runner.loss_objectives import create_objective,evaluate_objective
from runner.loss_partition import objective_partition_contribution
@pytest.fixture(autouse=True)
def restore_policy():
 old=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled()
 yield
 torch.use_deterministic_algorithms(old,warn_only=warn)
@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('op',IDS)
@pytest.mark.parametrize('mode',['mean','sum','noneMean','noneSum'])
def test_all_eight_window_scaling_matches_independent_reference_and_native_gradient(device,op,mode):
 if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
 torch.use_deterministic_algorithms(device=='cpu')
 rows,targets,c,x,roles=case(op,device);c['reduction']='none' if mode.startswith('none') else mode
 scalar='sum' if mode in ('sum','noneSum') else 'mean';r=loss_recipe(op,c);objective=create_objective(r,device)
 denominator=g.window_denominator(r,roles,list(x.shape),scalar)
 parts=[objective_partition_contribution(objective,x[i:i+2],{k:v[i:i+2] for k,v in roles.items()},scalar) for i in range(0,5,2)]
 value=sum(g.scaled_contribution(p,denominator,scalar) for p in parts)
 native=evaluate_objective(objective,x,roles,scalar)
 assert value.item()==pytest.approx(native['scalar'].item(),abs=3e-6,rel=3e-5)
 assert torch.allclose(torch.autograd.grad(value,x,retain_graph=True)[0],torch.autograd.grad(native['scalar'],x)[0],atol=3e-6,rtol=3e-5)
 # Independent existing plain-Python score/path enumeration oracle, not adapter.
 from runner.tests.test_loss_objectives import scalar_rows,triplet_oracle,ctc_oracle,flat
 if op=='loss_triplet_margin':expected=triplet_oracle(rows,r['config'])
 elif op=='loss_ctc':
  raw=[ctc_oracle(row[:t],target[:s],0) for row,target,t,s in zip(rows,targets,[4,3,2,3,4],[2,1,0,2,2])]
  expected=raw if c['reduction']=='none' else sum(raw) if mode=='sum' else sum(v/max(s,1) for v,s in zip(raw,[2,1,0,2,2]))/5
 else:
  cfg=dict(r['config'])
  if op=='loss_bce_with_logits':cfg.update(_weight=[.5,0.,2.],_pos_weight=[1.7]*3)
  expected=scalar_rows(op,rows,targets,cfg)
 expected_scalar=(sum(flat(expected))/len(flat(expected)) if scalar=='mean' else sum(flat(expected))) if c['reduction']=='none' else expected
 assert value.item()==pytest.approx(expected_scalar,abs=3e-6,rel=3e-5)

def test_ignored_local_chunk_valid_window_global_undefined_refuses_before_mutation():
 r=loss_recipe('loss_cross_entropy',dict(class_weights=[.1,.3],label_smoothing=.2));target=torch.tensor([-100,-100,1])
 d=g.window_denominator(r,{'target':target},[3,2]);assert d==float(torch.tensor(.3,dtype=torch.float32))
 state={'forwards':0}
 with pytest.raises(ValueError,match='denominator'):
  g.window_denominator(r,{'target':torch.tensor([-100,-100])},[2,2]);state['forwards']+=1
 assert state['forwards']==0

def test_ctc_none_mean_differs_native_mean_and_kl_batchmean():
 _,_,c,x,roles=case('loss_ctc','cpu');r=loss_recipe('loss_ctc',dict(c,reduction='none'))
 assert g.window_denominator(r,roles,list(x.shape))==5
 r=loss_recipe('loss_kl_div',dict(reduction='batchmean'));_,_,_,x,roles=case('loss_kl_div','cpu')
 assert g.window_denominator(r,roles,list(x.shape))==5

def test_sample_broadcast_probability_ce_and_zero_weight_bce_denominators():
 from runner.loss_configuration import validate_recipe as validate_loss
 def lr(op,c):return validate_loss(dict(schemaVersion=1,configurationVersion=1,componentID=op,config=c))
 for op in ('loss_mse','loss_l1'):
  r=lr(op,dict(target_broadcast=True));assert g.window_denominator(r,{'target':torch.zeros(3,4)},[3,2,4])==24
 r=lr('loss_cross_entropy',dict(target_mode='probabilities',class_weights=[.3,2.]));assert g.window_denominator(r,{'target':torch.tensor([[.2,.8],[.4,.6],[.7,.3]])},[3,2])==3
 r=lr('loss_bce_with_logits',dict(weight_mode='configured',weight_shape=[],weight_values=[0.]));assert g.window_denominator(r,{'target':torch.zeros(3,2)},[3,2])==6

def test_scaled_backward_accumulates_and_explicit_reset_independent_update():
 from runner.loss_partition import objective_partition_contribution
 r=loss_recipe('loss_mse');o=create_objective(r);p=torch.nn.Parameter(torch.tensor([2.]));x=torch.tensor([[1.],[2.],[3.]]);target=torch.zeros_like(x)
 denominator=g.window_denominator(r,{'target':target},[3,1])
 for start,end in ((0,2),(2,3)):
  part=objective_partition_contribution(o,p*x[start:end],{'target':target[start:end]})
  g.scaled_contribution(part,denominator,'mean').backward()
 assert p.grad.item()==pytest.approx(2*2*(1+4+9)/3,abs=2e-6)
 before=p.item();torch.optim.SGD([p],lr=.1).step();assert p.item()==pytest.approx(before-.1*2*2*14/3,abs=2e-6)
 p.grad=None;assert p.grad is None
 # New short window resets: derivative uses only its own actual numerator.
 part=objective_partition_contribution(o,p*x[:1],{'target':target[:1]});g.scaled_contribution(part,1.,'mean').backward();assert p.grad.item()==pytest.approx(2*p.item(),abs=2e-6)
