"""Strict triple transposed geometry and preallocation budgets."""
import pytest
import torch
from runner import conv_transpose3d_configuration as contract
from runner.tensor_layout import BatchedLayout


def config(**changes):
    return dict(in_channels=2,out_channels=4,kernel_size=[2,3,2],stride=[1,1,1],
                padding=[0,0,0],output_padding=[0,0,0],dilation=[1,1,1],groups=2,bias=True) | changes


@pytest.mark.parametrize('changes,expected,uncropped',[
    ({},[4,6,6],[4,6,6]),
    ({'stride':[2,1,3],'padding':[1,1,2],'output_padding':[1,0,2]},[5,4,12],[7,6,16]),
    ({'stride':[1,2,1],'dilation':[2,1,3],'output_padding':[1,1,2]},[6,10,10],[6,10,10]),
    ({'kernel_size':[1,1,1],'stride':[3,2,2],'padding':[2,2,3]},[3,3,3],[7,7,9]),
    ({'out_channels':6,'bias':False},[4,6,6],[4,6,6]),
])
def test_noncubic_formula_and_actual_cpu_shape(changes,expected,uncropped):
    c=config(**changes);r=contract.resolve_conv_transpose3d(c,BatchedLayout((2,3,4,5)))
    assert r['layout']==BatchedLayout((c['out_channels'],*expected))
    assert r['uncroppedOutputShape']==uncropped
    weights=c['in_channels']*(c['out_channels']//c['groups'])*c['kernel_size'][0]*c['kernel_size'][1]*c['kernel_size'][2]
    assert r['parameterCount']==weights+int(c['bias'])*c['out_channels']
    assert r['multiplyAdds']==60*weights
    layer=torch.nn.ConvTranspose3d(c['in_channels'],c['out_channels'],c['kernel_size'],
        stride=c['stride'],padding=c['padding'],output_padding=c['output_padding'],
        dilation=c['dilation'],groups=c['groups'],bias=c['bias'])
    assert tuple(layer(torch.randn(2,2,3,4,5)).shape)==(2,c['out_channels'],*expected)


@pytest.mark.parametrize('key',['kernel_size','stride','padding','output_padding','dilation'])
@pytest.mark.parametrize('value',[1,[1],[1,1],[1,1,1,1],(1,1,1),[True,1,1],[1,False,1],
                                [1,1,True],[1.5,1,1],[1,float('inf'),1],[1,1,float('nan')],[4097,1,1]])
def test_spatial_triples_strict(key,value):
    with pytest.raises(ValueError):contract.normalize_conv_transpose3d(config(**{key:value}))


@pytest.mark.parametrize('axis',[0,1,2])
def test_portable_output_padding_or_rule_on_every_axis(axis):
    c=config();c['output_padding'][axis]=1
    with pytest.raises(ValueError):contract.normalize_conv_transpose3d(c)
    c['dilation'][axis]=2
    assert contract.normalize_conv_transpose3d(c)['output_padding'][axis]==1
    c['output_padding'][axis]=2
    with pytest.raises(ValueError):contract.normalize_conv_transpose3d(c)


@pytest.mark.parametrize('changes',[{'in_channels':True},{'out_channels':513},{'groups':0},
    {'groups':3},{'out_channels':3},{'bias':1},{'bias':None}, {'stride':[1,0,1]},
    {'kernel_size':[1,1,0]},{'dilation':[0,1,1]},{'padding':[0,-1,0]}, {'output_padding':[0,0,-1]}])
def test_scalar_group_boolean_and_spatial_bounds(changes):
    with pytest.raises(ValueError):contract.normalize_conv_transpose3d(config(**changes))


def test_exact_keys_and_disjoint_canonical_triples():
    c=config(kernel_size=[2.0,3.0,2.0]);result=contract.normalize_conv_transpose3d(c)
    assert all(type(v) is int for v in result['kernel_size'])
    result['kernel_size'][0]=99
    assert c['kernel_size']==[2.0,3.0,2.0]
    for key in c:
        missing=dict(c);del missing[key]
        with pytest.raises(ValueError):contract.normalize_conv_transpose3d(missing)
    for key,value in [('padding_mode','zeros'),('padding_policy','valid'),('output_size',[3,4,5])]:
        with pytest.raises(ValueError):contract.normalize_conv_transpose3d(config(**{key:value}))


@pytest.mark.parametrize('shape',[(2,3,4),(2,3,4,5,6),(1,3,4,5),(2,4097,4,5),(2,3,4097,5),(2,3,4,4097)])
def test_input_rank_channels_and_axis_lengths(shape):
    with pytest.raises(ValueError):contract.resolve_conv_transpose3d(config(),BatchedLayout(shape))


@pytest.mark.parametrize('axis',[0,1,2])
def test_output_axis_negative_and_oversized(axis):
    c=config();c['padding'][axis]=10
    with pytest.raises(ValueError):contract.resolve_conv_transpose3d(c,BatchedLayout((2,3,4,5)))
    c=config();c['stride'][axis]=4096
    with pytest.raises(ValueError):contract.resolve_conv_transpose3d(c,BatchedLayout((2,3,4,5)))


def test_parameter_boundary_bias_and_input_based_work_boundary():
    c=config(in_channels=500,out_channels=500,groups=1,kernel_size=[2,2,2],bias=False)
    assert contract.normalize_conv_transpose3d(c)['bias'] is False
    with pytest.raises(ValueError,match='(?i)parameter'):contract.normalize_conv_transpose3d(c|{'bias':True})
    c=config(in_channels=500,out_channels=512,groups=1,kernel_size=[1,1,1],stride=[2,2,2],bias=False)
    r=contract.resolve_conv_transpose3d(c,BatchedLayout((500,5,10,10)))
    assert r['multiplyAdds']==128_000_000
    with pytest.raises(ValueError,match='(?i)(multiply|work|128)'):
        contract.resolve_conv_transpose3d(c,BatchedLayout((500,5,10,11)))


def test_uncropped_volume_exact_boundary_and_independent_overflow():
    c=config(in_channels=1,out_channels=100,groups=1,kernel_size=[1,1,1],
             stride=[100,100,100],output_padding=[99,99,99],padding=[49,49,49],bias=False)
    r=contract.resolve_conv_transpose3d(c,BatchedLayout((1,1,1,1)))
    assert r['uncroppedOutputShape']==[100,100,100]
    assert r['layout']==BatchedLayout((100,2,2,2))
    with pytest.raises(ValueError,match='(?i)uncropped'):
        contract.resolve_conv_transpose3d(c|{'stride':[101,100,100],'output_padding':[100,99,99]},BatchedLayout((1,1,1,1)))


def test_sample_volume_limits_before_allocation():
    with pytest.raises(ValueError):
        contract.resolve_conv_transpose3d(config(),BatchedLayout((2,512,512,512)))
    c=config(in_channels=1,out_channels=1,groups=1,kernel_size=[1,1,1],
             stride=[1000,1000,1000],output_padding=[999,999,999],bias=False)
    with pytest.raises(ValueError,match='(?i)output'):
        contract.resolve_conv_transpose3d(c,BatchedLayout((1,1,1,1)))


def test_input_sample_exact_100m_boundary_and_shared_layout_overflow():
    c=config(in_channels=100,out_channels=1,groups=1,kernel_size=[1,1,1],bias=False)
    accepted=contract.resolve_conv_transpose3d(c,BatchedLayout((100,100,100,100)))
    assert accepted['layout']==BatchedLayout((1,100,100,100))
    assert accepted['multiplyAdds']==100_000_000
    # The shared BatchedLayout boundary rejects 101M before the resolver is entered.
    with pytest.raises(ValueError):
        contract.resolve_conv_transpose3d(c,BatchedLayout((100,101,100,100)))


def test_output_sample_exact_100m_boundary_with_overlapping_uncropped_limit():
    c=config(in_channels=1,out_channels=100,groups=1,kernel_size=[1,1,1],
             stride=[100,100,100],output_padding=[99,99,99],bias=False)
    accepted=contract.resolve_conv_transpose3d(c,BatchedLayout((1,1,1,1)))
    assert accepted['layout']==BatchedLayout((100,100,100,100))
    assert accepted['multiplyAdds']==100
    # This also breaches uncropped volume; output is intentionally diagnosed first.
    with pytest.raises(ValueError,match='output sample volume'):
        contract.resolve_conv_transpose3d(c|{'stride':[101,100,100],'output_padding':[100,99,99]},BatchedLayout((1,1,1,1)))


def test_extreme_kernel_product_rejected_without_allocation():
    with pytest.raises(ValueError,match='(?i)parameter'):
        contract.normalize_conv_transpose3d(config(in_channels=512,out_channels=512,
            groups=1,kernel_size=[4096,4096,4096],bias=False))
