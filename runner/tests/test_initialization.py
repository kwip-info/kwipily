"""Native initialization with independent fan, prediction and update expectations."""
import copy
import hashlib
import importlib
import io
import json
import math

import pytest
import torch
from torch import nn


def helper(): return importlib.import_module('runner.initialization')
def rule(node='learn', parameter='weight', algorithm='xavier_uniform', **settings):
    defaults={'xavier_uniform':dict(gain=1.),'kaiming_uniform':dict(a=0.,mode='fan_in',nonlinearity='leaky_relu'),'normal':dict(mean=0.,std=1.),'zeros':{}}
    return dict(owner=dict(node=node, parameter=parameter), algorithm=algorithm, settings={**defaults.get(algorithm,{}),**settings})
def recipe(*rules, seed=42):
    return dict(schemaVersion=1, configurationVersion=1, componentID='initialization', config=dict(seed=seed, rules=list(rules or [rule()])))
class Model(nn.Module):
    def __init__(self, device='cpu', width=3):
        super().__init__(); self.layers=nn.ModuleDict(dict(learn=nn.Linear(2,width), other=nn.Linear(2,2), norm=nn.BatchNorm1d(2))).to(device)
    def forward(self,x): return self.layers['learn'](x)


def test_canonical_recipe_defaults_order_and_deep_copy():
    value=recipe(rule('other',algorithm='zeros'),rule())
    canonical=helper().validate_recipe(value)
    assert canonical['config']['rules']==[rule(gain=1.),rule('other',algorithm='zeros')]
    value['config']['rules'][0]['settings']['foreign']=7
    assert canonical['config']['rules'][1]['settings']=={}
    assert helper().validate_recipe(recipe(rule(algorithm='kaiming_uniform')))['config']['rules'][0]['settings']==dict(a=0.,mode='fan_in',nonlinearity='leaky_relu')
    assert helper().validate_recipe(recipe(rule(algorithm='normal')))['config']['rules'][0]['settings']==dict(mean=0.,std=1.)


@pytest.mark.parametrize('mutate',[
 lambda r:r.update(foreign=1),lambda r:r.update(schemaVersion=True),lambda r:r.update(configurationVersion=2),lambda r:r.update(componentID='foreign'),
 lambda r:r['config'].update(seed=True),lambda r:r['config'].update(seed=-1),lambda r:r['config'].update(seed=2**63),lambda r:r['config'].update(seed=1.5),
 lambda r:r['config'].update(rules=[]),lambda r:r['config'].update(rules=[rule()]*129),lambda r:r['config'].update(rules=[rule(),rule()]),
 lambda r:r['config']['rules'][0].update(algorithm='uniform'),lambda r:r['config']['rules'][0].update(foreign=0),lambda r:r['config']['rules'][0]['owner'].update(foreign=0),
 lambda r:r['config']['rules'][0]['owner'].update(node=''),lambda r:r['config']['rules'][0]['settings'].update(gain=True),lambda r:r['config']['rules'][0]['settings'].update(gain=float('nan')),
 lambda r:r['config']['rules'][0]['settings'].update(gain=101),lambda r:r['config']['rules'][0]['settings'].update(gain=1e-50),lambda r:r['config']['rules'][0]['settings'].update(std=1),
])
def test_strict_recipe_refusal(mutate):
    value=recipe(); mutate(value)
    with pytest.raises(ValueError):helper().validate_recipe(value)


@pytest.mark.parametrize('algorithm,settings',[
 ('normal',dict(mean=-1e6,std=0)),('normal',dict(mean=1e6,std=1e6)),('xavier_uniform',dict(gain=0)),
 ('kaiming_uniform',dict(a=100,mode='fan_out',nonlinearity='relu')),('zeros',{})])
def test_valid_algorithm_boundaries(algorithm,settings):
    assert helper().validate_recipe(recipe(rule(algorithm=algorithm,**settings),seed=2**63-1))['config']['seed']==2**63-1


@pytest.mark.parametrize('algorithm,settings',[
 ('normal',dict(std=-1)),('normal',dict(mean=1e6+1)),('normal',dict(std=1e6+1)),('normal',dict(mean=float('inf'))),
 ('kaiming_uniform',dict(a=-1)),('kaiming_uniform',dict(a=101)),('kaiming_uniform',dict(mode='both')),('kaiming_uniform',dict(nonlinearity='tanh')),('zeros',dict(mean=0))])
def test_algorithm_specific_invalid_controls(algorithm,settings):
    with pytest.raises(ValueError):helper().validate_recipe(recipe(rule(algorithm=algorithm,**settings)))


def test_wire_budget_and_no_input_mutation():
    value=recipe(rule(node='x'*32768)); before=copy.deepcopy(value)
    with pytest.raises(ValueError):helper().validate_recipe(value)
    assert value==before


def test_actual_ownership_collision_mapping_nested_names_and_alias_refusal():
    model=Model(); model.layers['safe']=nn.Sequential(nn.Linear(2,2)); model.layer_keys={'public':'safe','learn':'learn','other':'other','norm':'norm','alias':'alias'}
    selected=recipe(rule('public','0.weight',algorithm='zeros'))
    assert helper().resolve_owners(model,selected)==[dict(node='public',parameter='0.weight',name='layers.safe.0.weight',shape=[2,2],dtype='float32')]
    model.layers['alias']=model.layers['learn']
    with pytest.raises(ValueError,match='alias|same|identity'):helper().resolve_owners(model,recipe(rule(),rule('alias')))


@pytest.mark.parametrize('kind',['foreign','buffer','fan-vector','lazy','dtype','nonfinite','whole-budget'])
def test_invalid_model_and_owner_refuse_before_mutation(kind):
    model=Model(); selected=recipe()
    if kind=='foreign':selected=recipe(rule('missing'))
    elif kind=='buffer':selected=recipe(rule('norm','running_mean',algorithm='zeros'))
    elif kind=='fan-vector':selected=recipe(rule(parameter='bias'))
    elif kind=='lazy':model.layers['later']=nn.LazyLinear(2)
    elif kind=='dtype':model.layers['learn'].double()
    elif kind=='nonfinite':
        with torch.no_grad():model.layers['learn'].weight[0,0]=float('nan')
    elif kind=='whole-budget':model.layers['large']=nn.Linear(500,500)
    before={k:v.detach().clone() for k,v in model.state_dict().items() if not isinstance(v,torch.nn.parameter.UninitializedParameter)} if kind!='lazy' else model.layers['learn'].weight.detach().clone()
    with pytest.raises(ValueError):helper().apply_initialization(model,selected)
    if kind=='lazy':assert torch.equal(before,model.layers['learn'].weight)
    else:
        for k,v in before.items():torch.testing.assert_close(v,model.state_dict()[k],equal_nan=True,rtol=0,atol=0)


def test_lazy_shape_materialized_before_selection_is_supported():
    model=Model();model.layers['lazy']=nn.LazyLinear(3);model.layers['lazy'](torch.ones(1,2))
    helper().apply_initialization(model,recipe(rule('lazy',algorithm='normal',mean=2,std=0)))
    assert torch.equal(model.layers['lazy'].weight,torch.full((3,2),2.))


@pytest.mark.parametrize('algorithm,settings',[
 ('xavier_uniform',dict(gain=1.7)),('kaiming_uniform',dict(a=.4,mode='fan_in',nonlinearity='leaky_relu')),
 ('kaiming_uniform',dict(a=.8,mode='fan_out',nonlinearity='leaky_relu')),('kaiming_uniform',dict(a=8,mode='fan_in',nonlinearity='relu')),
 ('normal',dict(mean=-.7,std=1.3)),('zeros',{})])
def test_independent_fan_bounds_and_seeded_distribution(algorithm,settings):
    model=Model(); model.layers['spatial']=nn.Conv2d(8,16,(3,5),bias=False)
    r=recipe(rule('spatial',algorithm=algorithm,**settings)); values=[]
    for seed in [7,41,173]:
        r['config']['seed']=seed;helper().apply_initialization(model,r); values.extend(model.layers['spatial'].weight.flatten().tolist())
    mean=sum(values)/len(values);variance=sum((v-mean)**2 for v in values)/len(values)
    if algorithm=='zeros':assert set(values)=={0.};return
    if algorithm=='normal':assert abs(mean-settings['mean'])<.06;assert abs(variance-settings['std']**2)<.15;return
    fan_in=8*3*5;fan_out=16*3*5
    bound=settings['gain']*math.sqrt(6/(fan_in+fan_out)) if algorithm=='xavier_uniform' else math.sqrt(3)*math.sqrt(2/(1+settings['a']**2) if settings['nonlinearity']=='leaky_relu' else 2)/math.sqrt(fan_in if settings['mode']=='fan_in' else fan_out)
    assert all(-bound<=v<=bound for v in values)
    assert abs(mean)<bound*.04
    assert abs(variance-bound**2/3)<bound**2*.025


def test_seed_derivation_repeat_order_unrelated_owner_and_global_rng_identity():
    model=Model();r=recipe(rule());rng=torch.get_rng_state().clone();ident=id(model.layers['learn'].weight)
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    helper().apply_initialization(model,r);first=model.layers['learn'].weight.detach().clone()
    message=json.dumps(['kwip.initialization.owner-seed.v1',42,'learn','weight'],sort_keys=True,separators=(',',':')).encode()
    seed=int.from_bytes(hashlib.sha256(message).digest()[:8],'big')%(2**63)
    native=torch.empty_like(first);nn.init.xavier_uniform_(native,generator=torch.Generator().manual_seed(seed))
    assert torch.equal(first,native) and torch.equal(rng,torch.get_rng_state()) and id(model.layers['learn'].weight)==ident
    for key,value in before.items():
        if key!='layers.learn.weight':assert torch.equal(value,model.state_dict()[key])
    helper().apply_initialization(model,recipe(rule('other',algorithm='normal'),rule()))
    assert torch.equal(first,model.layers['learn'].weight)
    helper().apply_initialization(model,recipe(rule(),rule('other',algorithm='normal')))
    assert torch.equal(first,model.layers['learn'].weight)
    helper().apply_initialization(model,recipe(rule(),seed=43))
    assert not torch.equal(first,model.layers['learn'].weight)


def test_candidate_failure_and_cancellation_are_transactional(monkeypatch):
    model=Model();before={k:v.clone() for k,v in model.state_dict().items()};calls=[]
    original=nn.init.normal_
    def broken(tensor,*args,**kwargs):calls.append(1);tensor.fill_(float('inf'));return tensor
    monkeypatch.setattr(nn.init,'normal_',broken)
    with pytest.raises(ValueError):helper().apply_initialization(model,recipe(rule(),rule('other',algorithm='normal')))
    assert calls
    for k,v in before.items():assert torch.equal(v,model.state_dict()[k])
    monkeypatch.setattr(nn.init,'normal_',original)
    ticks=[]
    def cancelled():
        ticks.append(1)
        if len(ticks)==2:raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError,match='cancelled'):helper().apply_initialization(model,recipe(rule(),rule('other')),check_cancelled=cancelled)
    for k,v in before.items():assert torch.equal(v,model.state_dict()[k])


@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('algorithm',['xavier_uniform','kaiming_uniform','normal','zeros'])
def test_actual_device_scalar_affine_and_first_sgd_update_and_standalone(device,algorithm):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
    model=Model(device,width=2);r=recipe(rule(algorithm=algorithm),rule(parameter='bias',algorithm='normal',mean=.3,std=0))
    owners=helper().apply_initialization(model,r)
    assert owners==helper().resolve_owners(model,r)
    weight=model.layers['learn'].weight.detach().cpu().tolist();bias=model.layers['learn'].bias.detach().cpu().tolist();rows=[[.4,-.7],[1.1,.2]]
    expected=[[sum(weight[o][c]*row[c] for c in range(2))+bias[o] for o in range(2)] for row in rows]
    x=torch.tensor(rows,device=device,requires_grad=True);y=model(x)
    torch.testing.assert_close(y.cpu(),torch.tensor(expected),atol=2e-6,rtol=2e-6)
    upstream=[[.2,-.3],[.7,.4]]; y.backward(torch.tensor(upstream,device=device));optimizer=torch.optim.SGD(model.parameters(),lr=.03);optimizer.step()
    wanted_input=[[sum(upstream[n][o]*weight[o][c] for o in range(2)) for c in range(2)] for n in range(2)]
    torch.testing.assert_close(x.grad.cpu(),torch.tensor(wanted_input),atol=2e-6,rtol=2e-6)
    wanted_bias=[bias[o]-.03*sum(upstream[n][o] for n in range(2)) for o in range(2)]
    torch.testing.assert_close(model.layers['learn'].bias.cpu(),torch.tensor(wanted_bias),atol=2e-6,rtol=2e-6)
    wanted=[[weight[o][c]-.03*sum(upstream[n][o]*rows[n][c] for n in range(2)) for c in range(2)] for o in range(2)]
    torch.testing.assert_close(model.layers['learn'].weight.cpu(),torch.tensor(wanted),atol=2e-6,rtol=2e-6)
    code=helper().initialization_recipe_code(r);assert 'runner.' not in code
    namespace={};exec(code,namespace);other=Model(device,width=2);namespace['apply_initialization'](other,namespace['INITIALIZATION_RECIPE'])
    torch.testing.assert_close(other.layers['learn'].weight.cpu(),torch.tensor(weight),rtol=0,atol=0)
    torch.testing.assert_close(other.layers['learn'].bias.cpu(),torch.tensor(bias),rtol=0,atol=0)
    saved=io.BytesIO();torch.save(other.state_dict(),saved);saved.seek(0)
    restored=Model(device,width=2);restored.load_state_dict(torch.load(saved,weights_only=True,map_location='cpu'),strict=True)
    torch.testing.assert_close(restored(torch.tensor(rows,device=device)).cpu(),torch.tensor(expected),atol=2e-6,rtol=2e-6)


@pytest.mark.parametrize('seed',[42.0,10**1000])
def test_seed_requires_json_integer_not_double_or_unbounded_integer(seed):
    with pytest.raises(ValueError):helper().validate_recipe(recipe(seed=seed))

def test_unicode_owner_seed_uses_literal_utf8_and_actual_graphmodel_collision():
    model=Model();model.layers['学习']=nn.Linear(2,3)
    r=recipe(rule('学习'));helper().apply_initialization(model,r)
    encoded=json.dumps(['kwip.initialization.owner-seed.v1',42,'学习','weight'],sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')
    seed=int.from_bytes(hashlib.sha256(encoded).digest()[:8],'big')%(2**63)
    expected=torch.empty(3,2);nn.init.xavier_uniform_(expected,generator=torch.Generator().manual_seed(seed))
    assert torch.equal(expected,model.layers['学习'].weight)
    from runner.engine import GraphModel
    graph=dict(schemaVersion=2,nodes=[dict(id='input',type='input',config={'features':2}),dict(id='items',type='linear',config=dict(in_features=2,out_features=2,bias=True)),dict(id='output',type='output',config={})],edges=[dict(id='a',source='input',sourcePort='out',target='items',targetPort='in'),dict(id='b',source='items',sourcePort='out',target='output',targetPort='in')])
    actual=GraphModel(graph);helper().apply_initialization(actual,recipe(rule('items',algorithm='normal',mean=2,std=0)))
    assert actual.layer_keys['items']!='items'
    assert torch.equal(actual.layers[actual.layer_keys['items']].weight,torch.full((2,2),2.))

@pytest.mark.parametrize('algorithm,key', [('xavier_uniform','gain'),('normal','mean'),('normal','std'),('kaiming_uniform','a'),('kaiming_uniform','mode'),('kaiming_uniform','nonlinearity')])
def test_wire_requires_every_applicable_algorithm_setting(algorithm,key):
    defaults={'xavier_uniform':{'gain':1.},'normal':{'mean':0.,'std':1.},'kaiming_uniform':{'a':0.,'mode':'fan_in','nonlinearity':'leaky_relu'}}
    value=recipe(rule(algorithm=algorithm,**defaults[algorithm]));del value['config']['rules'][0]['settings'][key]
    with pytest.raises(ValueError):helper().validate_recipe(value)

@pytest.mark.parametrize('kind',['sparse','integer'])
def test_foreign_model_parameter_dtype_layout_refuses_actionably_before_native_finite(kind):
    model=Model()
    value=torch.sparse_coo_tensor([[0],[0]],[1.],(2,2)) if kind=='sparse' else torch.ones(2,2,dtype=torch.int64)
    model.layers['other'].weight=nn.Parameter(value,requires_grad=kind=='sparse')
    before=model.layers['learn'].weight.detach().clone()
    with pytest.raises(ValueError):helper().apply_initialization(model,recipe())
    assert torch.equal(before,model.layers['learn'].weight)
