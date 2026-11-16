import pytest
import torch
from runner.engine import GraphModel,run
from runner.tests.test_learning_evidence import affine_submission
from runner.product_execution import ProductExecutionAdapter

def model():return GraphModel(affine_submission()['graph'])

@pytest.mark.parametrize('kind',['rank0','rank1','rank3','empty','width','double','integer','sparse','nan','inf'])
def test_product_boundary_rejects_invalid_inputs(kind):
    cases={'rank0':torch.tensor(1.),'rank1':torch.ones(2),'rank3':torch.ones(2,1,2),'empty':torch.ones(0,2),'width':torch.ones(2,3),'double':torch.ones(2,2,dtype=torch.float64),'integer':torch.ones(2,2,dtype=torch.int64),'sparse':torch.eye(2).to_sparse(),'nan':torch.full((2,2),float('nan')),'inf':torch.full((2,2),float('inf'))}
    with pytest.raises(ValueError,match='Model input'):ProductExecutionAdapter(model()).forward(cases[kind])

def test_noncontiguous_dense_gradients_and_raw_double_remain_supported():
    m=model();x=torch.randn(3,4)[:,::2].requires_grad_();adapter=ProductExecutionAdapter(m)
    torch.testing.assert_close(adapter.forward(x),m(x));assert not x.is_contiguous()
    a=torch.autograd.grad(adapter.forward(x).sum(),x)[0];b=torch.autograd.grad(m(x).sum(),x)[0];torch.testing.assert_close(a,b)
    assert m.double()(x.double()).dtype==torch.float64
    with pytest.raises(ValueError,match='parameter'):ProductExecutionAdapter(m)

def test_invalid_parameters_and_public_output_overflow():
    m=model()
    with torch.no_grad():m.layers['first'].weight[0,0]=float('nan')
    with pytest.raises(ValueError,match='parameter'):ProductExecutionAdapter(m)
    m=model()
    with torch.no_grad():m.layers['first'].weight.fill_(torch.finfo(torch.float32).max)
    a=ProductExecutionAdapter(m)
    with pytest.raises(ValueError,match='output'):a.forward(torch.full((2,2),2.0))

def test_wrong_device_is_actionable():
    if not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
    with pytest.raises(ValueError,match='device'):ProductExecutionAdapter(model()).forward(torch.ones(2,2,device='mps'))

def test_evaluation_overflow_cannot_emit_epoch_or_completion(tmp_path,monkeypatch):
    original=GraphModel.forward
    def overflow(self,x,inspect=False):
        if not self.training:return torch.full((len(x),2),float('inf'),device=x.device)
        return original(self,x,inspect=inspect)
    monkeypatch.setattr(GraphModel,'forward',overflow);events=[]
    with pytest.raises(ValueError,match='Validation'):
        run(affine_submission()['graph'],tmp_path/'run',epochs=1,device='cpu',emit=events.append)
    assert not any(e['event'] in ('epoch','completed') for e in events)
    assert not (tmp_path/'run/manifest.json').exists()

def test_training_full_input_scans_once_per_partition(tmp_path,monkeypatch):
    observed=[];original=ProductExecutionAdapter.validate_tensor
    def capture(value,**kwargs):
        if kwargs.get('check_finite',True):observed.append(kwargs.get('context'))
        return original(value,**kwargs)
    monkeypatch.setattr(ProductExecutionAdapter,'validate_tensor',staticmethod(capture))
    run(affine_submission()['graph'],tmp_path/'run',epochs=3,batch_size=32,device='cpu')
    assert observed.count('Training source')==1
    assert observed.count('Validation source')==1
    assert observed.count('Model input')==1 # bounded final capture only

def test_captured_intermediate_nonfinite_rejected_even_with_finite_output():
    from runner.tests.test_runner import graph
    g=graph();next(n for n in g['nodes'] if n['id']=='relu')['type']='tanh' # left branch activation bounds overflowing intermediate
    m=GraphModel(g)
    with torch.no_grad():m.layers['left'].weight.fill_(torch.finfo(torch.float32).max)
    adapter=ProductExecutionAdapter(m);x=torch.full((2,2),2.0)
    assert torch.isfinite(m(x)).all()
    with pytest.raises(ValueError,match='Captured component left'):adapter.forward(x,inspect=True)

def test_nested_tensor_rejected_before_shape_access():
    x=torch.nested.nested_tensor([torch.ones(2,2),torch.ones(3,2)])
    with pytest.raises(ValueError,match='nested'):ProductExecutionAdapter(model()).forward(x)

def test_finite_logits_nonfinite_validation_loss_rejected(tmp_path,monkeypatch):
    original=torch.nn.CrossEntropyLoss.forward;calls=[]
    def loss(self,logits,target):
        calls.append(1)
        return original(self,logits,target) if len(calls)==1 else torch.tensor(float('inf'))
    monkeypatch.setattr(torch.nn.CrossEntropyLoss,'forward',loss);events=[]
    with pytest.raises(ValueError,match='Validation loss'):
        run(affine_submission()['graph'],tmp_path/'run',epochs=1,device='cpu',emit=events.append)
    assert not any(e['event'] in ('epoch','completed') for e in events)
    assert not (tmp_path/'run/manifest.json').exists()

def test_meta_model_device_rejected_before_value_scan():
    with pytest.raises(ValueError,match='CPU or MPS'):
        ProductExecutionAdapter(model().to('meta'))
