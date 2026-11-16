"""Do not erase finite intermediate guards merely to make graph capture succeed."""
import pytest
import torch
from torch import nn
from runner.engine import _execute_tensor_composition


class Composition(nn.Module):
    def __init__(self,operation):super().__init__();self.operation=operation
    def forward(self,x):
        return _execute_tensor_composition(x[:,:1],x[:,1:],self.operation,{'dim':1} if self.operation=='concat' else {'alpha':0. if self.operation=='zero' else 1.,'sample_broadcast':False},(1,),(1,),(2,) if self.operation=='concat' else (1,))


class Add(nn.Module):
    def __init__(self,alpha):super().__init__();self.alpha=alpha
    def forward(self,x):return _execute_tensor_composition(x[:,:1],x[:,1:],'add',{'alpha':self.alpha,'sample_broadcast':False},(1,),(1,),(1,))


@pytest.mark.parametrize('strict',[False,True])
@pytest.mark.parametrize('device',['cpu','mps'])
def test_composition_export_preserves_eager_source_and_overflow_guards(strict,device):
    if device=='mps' and not torch.backends.mps.is_available():return
    x=torch.ones(2,2,device=device)
    for model in (Composition('concat'),Add(0.),Add(1.)):
        native=torch.export.export(model,(x,),strict=strict).module()
        torch.testing.assert_close(native(x),model(x))
        for bad in (torch.tensor([[float('nan'),1.],[1.,1.]],device=device),torch.tensor([[1.,float('inf')],[1.,1.]],device=device)):
            with pytest.raises(ValueError,match='finite'):model(bad)
            with pytest.raises((RuntimeError,ValueError),match='finite'):
                native(bad)
                if device=='mps':torch.mps.synchronize()
        if isinstance(model,Add) and model.alpha==1.:
            overflow=torch.full((2,2),3e38,device=device)
            with pytest.raises(ValueError,match='nonfinite'):model(overflow)
            with pytest.raises((RuntimeError,ValueError),match='nonfinite'):
                native(overflow)
                if device=='mps':torch.mps.synchronize()
