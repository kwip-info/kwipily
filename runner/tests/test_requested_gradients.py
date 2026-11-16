import pytest
import torch
from runner import requested_gradients as r
from runner.tests.test_trainability import Model

def request(reduction='mean',targets=None,**changes):
 value=dict(id='effect',output=dict(node='second',port='out'),partition='validation',row_start=0,row_count=2,reduction=reduction,upstream=None,targets=targets or [dict(kind='input',node='input',port='out')],unavailable='error');value.update(changes);return value
@pytest.mark.parametrize('device',['cpu','mps'])
def test_actual_mean_chain_rule(device):
 if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
 result=r.analyze(Model(device),torch.tensor([[2.],[3.]],device=device),request(),device=device)
 assert result['targets'][0]['values']==[6.,6.]

import copy
from runner.trainability import apply_trainability
from runner.tests.test_trainability import recipe as train_recipe,owner
from runner.gradient_policy import canonical_configuration

def target(node,parameter):return dict(kind='parameter',node=node,parameter=parameter)
@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('reduction,expected',[('sum',[12.,12.]),('mean',[6.,6.]),('upstream',[24.,-12.])])
def test_scalar_parameter_and_signed_upstream_with_finite_difference(device,reduction,expected):
 if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
 m=Model(device);x=torch.tensor([[2.],[3.]],device=device);q=request(reduction,targets=[dict(kind='input',node='input',port='out'),target('first','weight'),target('second','weight')])
 if reduction=='upstream':q['upstream']=dict(dtype='float32',shape=[2,1],values=[2.,-1.])
 values=r.analyze(m,x,q,device=device)['targets'];assert values[0]['values']==expected
 assert values[1]['values']==[4. if reduction=='upstream' else 20. if reduction=='sum' else 10.]
 assert values[2]['values']==[4. if reduction=='upstream' else 17. if reduction=='sum' else 8.5]
 # Centered finite difference independent of autograd: second-weight sensitivity.
 eps=.001
 def scalar(w):
  scores=[w*(3*z+1)-2 for z in (2.,3.)]
  return sum(scores) if reduction=='sum' else sum(scores)/2 if reduction=='mean' else 2*scores[0]-scores[1]
 assert values[2]['values'][0]==pytest.approx((scalar(4+eps)-scalar(4-eps))/(2*eps),abs=1e-8)
@pytest.mark.parametrize('device',['cpu','mps'])
def test_exact_state_modes_rng_and_existing_grad_object_preserved_on_success_and_error(device):
 if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
 m=Model(device);m.train();m.layers['safe_second'].eval()
 params=list(m.parameters())
 for p in params:p.grad=torch.full_like(p,7.)
 grads=[p.grad for p in params];state={k:v.clone() for k,v in m.state_dict().items()};cpu=torch.get_rng_state().clone();mps=torch.mps.get_rng_state().clone() if torch.backends.mps.is_available() else None
 modes=[v.training for v in m.modules()];x=torch.tensor([[2.],[3.]],device=device)
 r.analyze(m,x,request(),device=device)
 with pytest.raises(ValueError):r.analyze(m,x,request(output=dict(node='missing',port='out')),device=device)
 assert [v.training for v in m.modules()]==modes and torch.equal(cpu,torch.get_rng_state())
 if mps is not None:assert torch.equal(mps,torch.mps.get_rng_state())
 for p,grad in zip(params,grads):assert p.grad is grad and torch.equal(grad,torch.full_like(grad,7.))
 assert all(torch.equal(v,m.state_dict()[k]) for k,v in state.items())
def test_measured_zero_frozen_disconnected_and_recording_disabled_distinct():
 m=Model();apply_trainability(m,train_recipe(owner()));m.layers['unused']=torch.nn.Linear(1,1);m.layer_keys['unused']='unused';x=torch.tensor([[2.],[3.]])
 q=request('upstream',targets=[dict(kind='input',node='input',port='out'),target('second','weight'),target('unused','weight')],upstream=dict(dtype='float32',shape=[2,1],values=[0.,0.]),unavailable='reportUnavailable')
 result=r.analyze(m,x,q,device='cpu')['targets'];assert result[0]['status']=='observed' and result[0]['values']==[0.,0.]
 assert [v.get('cause') for v in result]==[None,'frozenParameter','disconnected']
 assert all('values' not in v for v in result[1:])
 q=request(unavailable='reportUnavailable');result=r.analyze(m,x,q,device='cpu',recording=False)['targets'][0]
 assert result['cause']=='recordingDisabled' and 'values' not in result
 with pytest.raises(ValueError,match='unavailable'):r.analyze(m,x,request(targets=[target('second','weight')]),device='cpu')

class EmbeddingModel(torch.nn.Module):
 def __init__(self):
  super().__init__();self.layer_keys={'embed':'embed'};self.layers=torch.nn.ModuleDict({'embed':torch.nn.Embedding(5,2,sparse=True)})
 def forward(self,x,inspect=False):
  y=self.layers['embed'](x)
  return (y,{'input':x,'embed':y}) if inspect else y

def test_actual_integer_input_and_sparse_parameter_derivative_not_densified():
 m=EmbeddingModel();q=request('sum',output=dict(node='embed',port='out'),targets=[dict(kind='input',node='input',port='out'),target('embed','weight')],unavailable='reportUnavailable')
 result=r.analyze(m,torch.tensor([[1],[2]]),q,device='cpu')['targets']
 assert [v['cause'] for v in result]==['integerInput','sparseGradient'] and all('values' not in v for v in result)

def context(qs):
 policy=dict(schemaVersion=1,configurationVersion=1,componentID='gradient_policy',config=canonical_configuration({'requests':qs}))
 return dict(policy=policy,prepared_inputs={'validation':torch.tensor([[2.],[3.],[4.]])},partition_row_ids={'validation':['a','b','c']},source_identity=dict(runID='actual-run',graphSHA256='a'*64,checkpointSHA256='b'*64,preparationManifestSHA256='c'*64,preparationGraphDigest='d'*64),device='cpu')
def test_qualified_artifact_privacy_rows_pins_and_permission():
 c=context([request()]);m=Model()
 with pytest.raises(ValueError,match='Enable'):r.analyze_requested_gradients(m,**c)
 result=r.analyze_requested_gradients(m,**c,derivative_capture={'schemaVersion':1,'values':True})
 assert result['phase']=='finalSnapshotEvaluation' and result['source']['checkpointSHA256']=='b'*64
 assert result['requests'][0]['targets'][0]['values']==[6.,6.]
 assert set(result['requests'][0])=={'request','sample','output','targets'}
 assert set(result['requests'][0]['sample'])=={'partition','rowStart','rowCount','rowIDsSHA256'}
 assert 'rowIDs' not in result['requests'][0]['sample'] and 'values' not in result['requests'][0]['output']
 for change in (dict(row_start=2),dict(row_start=2**63-3)):
  c=context([request(**change)])
  with pytest.raises(ValueError,match='range'):r.analyze_requested_gradients(m,**c,derivative_capture={'schemaVersion':1,'values':True})
 assert r.analyze_requested_gradients(m,**context([])) is None

def test_total_target_budget_refuses_before_any_forward():
 m=Model();calls=[];m.register_forward_pre_hook(lambda *args:calls.append(1))
 c=context([request(id='a',row_count=32),request(id='b',row_count=32)])
 c['prepared_inputs']['validation']=torch.zeros(32,2048);c['partition_row_ids']['validation']=list(range(32))
 with pytest.raises(ValueError,match='65536'):r.analyze_requested_gradients(m,**c,derivative_capture={'schemaVersion':1,'values':True})
 assert calls==[]
def test_exact_upstream_shape_and_actual_input_endpoint():
 m=Model();x=torch.tensor([[2.],[3.]])
 for q in [request('upstream',upstream=dict(dtype='float32',shape=[2,2],values=[1]*4)),request(targets=[dict(kind='input',node='first',port='out')]),request(output=dict(node='second',port='wrong'))]:
  with pytest.raises(ValueError):r.analyze(m,x,q,device='cpu')
def test_native_unsupported_backward_is_hard_error_even_reporting():
 class Unsupported(torch.autograd.Function):
  @staticmethod
  def forward(ctx,x):return x.clone()
  @staticmethod
  def backward(ctx,gradient):raise NotImplementedError('backend deliberately unavailable')
 class Bad(Model):
  def forward(self,x,inspect=False):
   y=Unsupported.apply(x)
   return y,{'input':x,'second':y}
 with pytest.raises(ValueError,match='no fallback'):r.analyze(Bad(),torch.tensor([[2.],[3.]]),request(unavailable='reportUnavailable'),device='cpu')
def test_cancel_restores_exact_state_and_rng():
 m=Model();m.train();rng=torch.get_rng_state().clone();counter=[0]
 def cancel():
  counter[0]+=1
  if counter[0]==2:raise RuntimeError('cancelled')
 with pytest.raises(RuntimeError,match='cancelled'):r.analyze(m,torch.tensor([[2.],[3.]]),request(),device='cpu',check_cancelled=cancel)
 assert m.training and torch.equal(rng,torch.get_rng_state())

@pytest.mark.parametrize('device',['cpu','mps'])
def test_real_graphmodel_nonlinear_parameter_and_input_influence_scalar_formula(device):
 if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
 from runner.engine import GraphModel
 from runner.tests.test_runner import graph
 import math
 m=GraphModel(graph()).to(device)
 with torch.no_grad():
  for name,module in m.layers.items():
   if isinstance(module,torch.nn.Linear):module.weight.fill_(.25);module.bias.fill_(.1)
 x=torch.tensor([[.2,.3],[.4,.1]],device=device)
 q=request('sum',output=dict(node='output',port='out'),targets=[dict(kind='input',node='input',port='out'),target('right','weight')])
 measured=r.analyze(m,x,q,device=device)['targets']
 # Each branch has16units; head has2outputs. ReLU is positive here.
 z=.25*.5+.1;dt=1-math.tanh(z)**2
 assert measured[0]['values']==pytest.approx([2*(16*.25*.25+16*.25*.25*dt)]*4,abs=3e-6)
 assert measured[1]['values']==pytest.approx([.5*dt*(.2+.4),.5*dt*(.3+.1)]*16,abs=3e-6)

def test_preexisting_sparse_grad_reference_and_contents_preserved():
 m=EmbeddingModel();p=m.layers['embed'].weight;original=torch.sparse_coo_tensor(torch.tensor([[1,2]]),torch.tensor([[1.,2.],[3.,4.]]),(5,2));p.grad=original
 q=request('sum',output=dict(node='embed',port='out'),targets=[target('embed','weight')],unavailable='reportUnavailable')
 r.analyze(m,torch.tensor([[1],[2]]),q,device='cpu')
 assert p.grad is original and torch.equal(original.coalesce().values(),torch.tensor([[1.,2.],[3.,4.]]))
