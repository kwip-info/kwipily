"""Strict composition axes, broadcast layouts and legacy configuration identity."""
import pytest
from runner.tensor_layout import BatchedLayout
from runner.tensor_composition_configuration import normalize_tensor_composition as normalize,resolve_tensor_composition as resolve,capability

CONFIGS={'concat':{'dim':-1},'add':{'alpha':1.0,'sample_broadcast':False},'stack':{'dim':-1},'permute':{'dims':[0,2,1]}}

@pytest.mark.parametrize('op',['concat','add'])
def test_historical_empty_is_exact_defaults(op):
    assert normalize(op,{})==CONFIGS[op]
    assert normalize(op,CONFIGS[op])==CONFIGS[op]

@pytest.mark.parametrize('op',CONFIGS)
def test_exact_control_keys_and_no_untrusted_behavior(op):
    for c in [None,[],dict(CONFIGS[op],out='other'),{'code':'x'},{}] if op not in ('concat','add') else [None,[],dict(CONFIGS[op],out='other'),{'code':'x'}]:
        with pytest.raises(ValueError):normalize(op,c)
    with pytest.raises(ValueError):normalize(op,CONFIGS[op],dtype='float64')

@pytest.mark.parametrize('op',['concat','stack'])
@pytest.mark.parametrize('dim',[True,None,0,1.5,'1',-9,8,float('nan'),float('inf')])
def test_invalid_axis_controls(op,dim):
    with pytest.raises(ValueError):normalize(op,{'dim':dim})

@pytest.mark.parametrize('alpha',[True,None,'1',float('nan'),float('inf'),-float('inf'),3.5e38,-3.5e38])
def test_nonfinite_or_untyped_alpha_refuses(alpha):
    with pytest.raises(ValueError):normalize('add',{'alpha':alpha,'sample_broadcast':True})

@pytest.mark.parametrize('value',[None,0,1,'true'])
def test_broadcast_is_strict_boolean(value):
    with pytest.raises(ValueError):normalize('add',{'alpha':1,'sample_broadcast':value})


def test_alpha_zero_negative_fraction_extreme_and_integer_legacy_boundary():
    for alpha in [0,-.5,2.25,3.4028234663852886e38,-3.4028234663852886e38]:assert normalize('add',{'alpha':alpha,'sample_broadcast':True})['alpha']==alpha
    assert normalize('add',{},dtype='int64')==CONFIGS['add']
    for c in [{'alpha':-.5,'sample_broadcast':False},{'alpha':1,'sample_broadcast':True}]:
        with pytest.raises(ValueError):normalize('add',c,dtype='int64')
    with pytest.raises(ValueError):normalize('add',{'alpha':1})

@pytest.mark.parametrize('dims',[[1,0],[0,1,1],[0,2,3],[0,1,8],[0,True],[0,1.5],None,[],[0]*9])
def test_invalid_complete_order(dims):
    with pytest.raises(ValueError):normalize('permute',{'dims':dims})


def test_normalized_orders_and_shape_rank_batch_constraints():
    assert normalize('permute',{'dims':[-4,-2,-1,-3]})=={'dims':[-4,-2,-1,-3]}
    assert resolve('permute',{'dims':[0,2,3,1]},BatchedLayout((2,3,4))).sample==(3,4,2)
    assert resolve('permute',{'dims':[-4,-2,-1,-3]},BatchedLayout((2,3,4))).sample==(3,4,2)
    assert resolve('permute',{'dims':[0,1]},BatchedLayout((3,))).sample==(3,)
    with pytest.raises(ValueError):resolve('permute',{'dims':[0,2,1]},BatchedLayout((2,3,4)))
    for op,c in [('concat',{'dim':-3}),('stack',{'dim':-4})]:
        with pytest.raises(ValueError):resolve(op,c,BatchedLayout((2,3)),BatchedLayout((2,3)))


def test_binary_order_and_new_direction_geometry():
    a=BatchedLayout((2,3));b=BatchedLayout((1,3))
    assert resolve('concat',{'dim':1},a,b).sample==(3,3)
    for axis,wanted in [(1,(2,2,3)),(2,(2,2,3)),(3,(2,3,2)),(-1,(2,3,2))]:assert resolve('stack',{'dim':axis},a,a).sample==wanted
    with pytest.raises(ValueError):resolve('stack',{'dim':1},a,b)
    with pytest.raises(ValueError):resolve('concat',{'dim':2},a,b)
    with pytest.raises(ValueError):resolve('concat',{'dim':4},a,b)
    with pytest.raises(ValueError):resolve('stack',{'dim':1},BatchedLayout((1,)*7),BatchedLayout((1,)*7))
    assert resolve('stack',{'dim':7},BatchedLayout((1,)*6),BatchedLayout((1,)*6)).sample==(1,1,1,1,1,1,2)


def test_sample_only_broadcast_and_allocation_budget():
    a=BatchedLayout((2,3));b=BatchedLayout((3,))
    c={'alpha':-.5,'sample_broadcast':True}
    assert resolve('add',c,a,b).sample==(2,3)
    assert resolve('add',c,BatchedLayout((2,1,3)),BatchedLayout((4,1))).sample==(2,4,3)
    with pytest.raises(ValueError):resolve('add',{},a,b)
    with pytest.raises(ValueError):resolve('add',c,BatchedLayout((2,3)),BatchedLayout((4,)))
    with pytest.raises(ValueError):resolve('add',c,BatchedLayout((100_000_000,1)),BatchedLayout((1,2)))
    with pytest.raises(ValueError):resolve('concat',{'dim':1},BatchedLayout((100_000_000,)),BatchedLayout((1,)))


def test_capability_preserves_historical_layout_as_separate_contract():
    for op in CONFIGS:
        c=capability(op);assert c['adapterID']=='kwip.runner.graph-v2.'+op
        assert c['preservesBatch'] is True and c['maximumRank']==8 and c['graphDTypes']==['float32','int64']
    assert capability('stack')['maximumInputRank']==7
