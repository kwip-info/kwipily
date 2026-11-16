import pytest
import torch
from runner import trainability as t
class Model(torch.nn.Module):
 def __init__(self,device='cpu'):
  super().__init__();self.layer_keys={'first':'safe_first','second':'safe_second'};self.layers=torch.nn.ModuleDict({'safe_first':torch.nn.Linear(1,1),'safe_second':torch.nn.Linear(1,1)}).to(device)
  with torch.no_grad():self.layers['safe_first'].weight.fill_(3);self.layers['safe_first'].bias.fill_(1);self.layers['safe_second'].weight.fill_(4);self.layers['safe_second'].bias.fill_(-2)
 def forward(self,x,inspect=False):
  a=self.layers['safe_first'](x);b=self.layers['safe_second'](a)
  return (b,{'input':x,'first':a,'second':b}) if inspect else b

def recipe(*owners):return dict(schemaVersion=1,configurationVersion=1,componentID='trainability',config={'owners':list(owners)})
def owner(node='second',parameter='weight',mode='frozen'):return dict(node=node,parameter=parameter,mode=mode)
def test_frozen_native_object_not_replaced():
 model=Model();p=model.layers['safe_second'].weight;before=p.clone();t.apply_trainability(model,recipe(owner()))
 assert not p.requires_grad
 assert model.layers['safe_second'].weight is p and torch.equal(p,before)
def test_invalid_owner_rolls_back_all_flags():
 model=Model()
 with pytest.raises(ValueError):t.apply_trainability(model,recipe(owner(),owner('missing')))
 assert all(p.requires_grad for p in model.parameters())

@pytest.mark.parametrize('device',['cpu','mps'])
def test_flags_keep_input_and_earlier_parameter_influence(device):
 if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
 m=Model(device);t.apply_trainability(m,recipe(owner()));x=torch.tensor([[2.]],device=device,requires_grad=True);y=m(x)
 dx,dw=torch.autograd.grad(y.sum(),(x,m.layers['safe_first'].weight))
 assert dx.item()==12 and dw.item()==8
 assert list(p for p in m.parameters() if p.requires_grad)==[m.layers['safe_first'].weight,m.layers['safe_first'].bias,m.layers['safe_second'].bias]
@pytest.mark.parametrize('bad',[True,None,'frozen',{},[{}],[owner(mode='eval')],[owner(),owner()],[owner(node='')],[owner(parameter='a\0b')]])
def test_strict_owner_configuration(bad):
 with pytest.raises(ValueError):t.canonical_configuration({'owners':bad})
def test_inherit_preserves_native_flags_and_canonical_order():
 m=Model();m.layers['safe_first'].weight.requires_grad_(False)
 result=t.apply_trainability(m,recipe(owner('second',mode='inherit'),owner('first',mode='inherit')))
 assert not result[0]['requiresGrad'] and result[2]['requiresGrad']
 assert [x['node'] for x in t.validate_recipe(recipe(owner('second'),owner('first')))['config']['owners']]==['first','second']
@pytest.mark.parametrize('mode',['buffer','lazy','alias','foreign','dtype','nonfinite','budget'])
def test_whole_inventory_and_foreign_rejections_before_flag_changes(mode):
 m=Model()
 if mode=='buffer':m.layers['safe_first'].register_buffer('counter',torch.tensor(1));selected=recipe(owner('first','counter'))
 else:selected=recipe(owner())
 if mode=='lazy':m.layers['later']=torch.nn.LazyLinear(2);m.layer_keys['later']='later'
 if mode=='alias':m.layers['safe_first'].bias=m.layers['safe_second'].bias
 if mode=='foreign':m.register_parameter('foreign',torch.nn.Parameter(torch.ones(1)))
 if mode=='dtype':m.double()
 if mode=='nonfinite':
  with torch.no_grad():m.layers['safe_first'].weight.fill_(float('nan'))
 if mode=='budget':m.layers['big']=torch.nn.Linear(500,500);m.layer_keys['big']='big'
 flags=[p.requires_grad for p in m.parameters()]
 with pytest.raises(ValueError):t.apply_trainability(m,selected)
 assert [p.requires_grad for p in m.parameters()]==flags

def test_frozen_batchnorm_and_dropout_are_not_eval_policy():
 m=Model();m.layers['bn']=torch.nn.BatchNorm1d(1);m.layer_keys['bn']='bn';m.layers['drop']=torch.nn.Dropout(.5);m.layer_keys['drop']='drop'
 t.apply_trainability(m,recipe(owner('bn','weight'),owner('bn','bias')))
 before=m.layers['bn'].running_mean.clone();x=torch.arange(1.,17.).reshape(16,1)
 m.train();a=m.layers['drop'](m.layers['bn'](x));assert not torch.equal(before,m.layers['bn'].running_mean)
 m.eval();before=m.layers['bn'].running_mean.clone();b=m.layers['drop'](m.layers['bn'](x));c=m.layers['drop'](m.layers['bn'](x))
 assert torch.equal(before,m.layers['bn'].running_mean) and torch.equal(b,c) and (a==0).any()
 assert not m.layers['bn'].weight.requires_grad

def test_all_frozen_inference_is_valid_and_explicit_unfreeze_preserves_values():
 m=Model();before={k:v.clone() for k,v in m.state_dict().items()}
 owners=[owner(meta['node'],meta['parameter']) for meta,_ in t.native_owners(m)];t.apply_trainability(m,recipe(*owners))
 assert not any(p.requires_grad for p in m.parameters())
 with torch.no_grad():assert m(torch.tensor([[2.]])).item()==26
 t.apply_trainability(m,recipe(owner('first','weight','trainable')));assert m.layers['safe_first'].weight.requires_grad
 assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())
