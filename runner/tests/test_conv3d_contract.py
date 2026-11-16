"""Explicit D/H/W Conv3d geometry; no scalar spatial coercion."""
import pytest
import torch
from runner import conv3d_configuration as contract
from runner.tensor_layout import BatchedLayout


def config(**changes):
    return dict(in_channels=4,out_channels=6,kernel_size=[2,3,4],stride=[1,1,1],dilation=[1,1,1],
                groups=2,bias=True,padding_policy='explicit',padding=[0,0,0],padding_mode='zeros') | changes


@pytest.mark.parametrize('settings,expected,pads',[
    ({},(6,8,9,10),(0,0,0,0,0,0)),
    ({'padding':[1,2,3],'stride':[2,3,2]},(6,5,5,8),(1,1,2,2,3,3)),
    ({'dilation':[2,1,2],'padding_policy':'valid'},(6,7,9,7),(0,0,0,0,0,0)),
    ({'padding_policy':'same'},(6,9,11,13),(0,1,1,1,1,2)),
    ({'dilation':[3,2,1],'padding_policy':'same'},(6,9,11,13),(1,2,2,2,1,2)),
    ({'groups':4,'out_channels':8,'bias':False},(8,8,9,10),(0,0,0,0,0,0)),
])
def test_nonsymmetric_geometry_and_parameter_work_counts(settings,expected,pads):
    c=config(**settings);result=contract.resolve_conv3d(c,BatchedLayout((4,9,11,13)))
    assert result['layout']==BatchedLayout(expected)
    assert tuple(result[k] for k in ('paddingFront','paddingBack','paddingTop','paddingBottom','paddingLeft','paddingRight'))==pads
    fan_in=(c['in_channels']//c['groups'])*c['kernel_size'][0]*c['kernel_size'][1]*c['kernel_size'][2]
    assert result['parameterCount']==c['out_channels']*(fan_in+int(c['bias']))
    assert result['multiplyAdds']==expected[1]*expected[2]*expected[3]*c['out_channels']*fan_in


@pytest.mark.parametrize('policy',['explicit','valid','same'])
@pytest.mark.parametrize('mode',['zeros','reflect','replicate','circular'])
def test_all_policy_modes_match_actual_cpu_shape(policy,mode):
    c=config(padding_policy=policy,padding_mode=mode,padding=[1,2,1] if policy=='explicit' else [0,0,0],
             stride=[1,1,1] if policy=='same' else [2,1,3],dilation=[2,1,1])
    result=contract.resolve_conv3d(c,BatchedLayout((4,9,11,13)))
    layer=torch.nn.Conv3d(c['in_channels'],c['out_channels'],c['kernel_size'],stride=c['stride'],dilation=c['dilation'],
        groups=c['groups'],bias=c['bias'],padding_mode=mode,padding=c['padding'] if policy=='explicit' else policy)
    assert tuple(layer(torch.randn(2,4,9,11,13)).shape)==(2,*result['layout'].sample)


@pytest.mark.parametrize('field',['kernel_size','stride','dilation','padding'])
@pytest.mark.parametrize('value',[1,[1],[1,2],[1,2,3,4],[1,True,1],[1,1.5,1],[1,1,None],None,'1,2,3'])
def test_spatial_configuration_requires_strict_triples(field,value):
    with pytest.raises(ValueError):contract.normalize_conv3d(config(**{field:value}))


def test_canonical_integral_float_copy_and_exact_fields():
    authored=config(kernel_size=[2.0,3.0,4.0],padding=[0.0,1.0,2.0])
    normalized=contract.normalize_conv3d(authored)
    assert normalized['kernel_size']==[2,3,4] and all(type(x) is int for x in normalized['kernel_size'])
    normalized['padding'][0]=1
    assert authored['padding']==[0.0,1.0,2.0]
    for key in authored:
        missing=dict(authored);del missing[key]
        with pytest.raises(ValueError):contract.normalize_conv3d(missing)
    with pytest.raises(ValueError):contract.normalize_conv3d(config(extra=True))


@pytest.mark.parametrize('changes',[
    {'in_channels':True},{'out_channels':0},{'groups':513},{'groups':3},{'out_channels':5},
    {'kernel_size':[0,1,1]},{'stride':[1,1,4097]},{'dilation':[1,float('inf'),1]},
    {'padding':[0,-1,0]},{'padding':[0,0,4097]},{'bias':1},{'bias':None},
    {'padding_policy':'SAME'},{'padding_mode':'constant'},
    {'padding_policy':'valid','padding':[0,1,0]},{'padding_policy':'same','stride':[1,1,2]},
    {'in_channels':512,'out_channels':512,'groups':1,'kernel_size':[2,2,2]},
])
def test_strict_configuration_groups_policies_and_parameter_cap(changes):
    with pytest.raises(ValueError):contract.normalize_conv3d(config(**changes))


@pytest.mark.parametrize('shape',[(4,9,11),(4,9,11,13,1),(3,9,11,13),(4,4097,1,1),(4,1,4097,1),(4,1,1,4097)])
def test_sample_contract_and_each_axis_limit(shape):
    with pytest.raises(ValueError):contract.resolve_conv3d(config(),BatchedLayout(shape))


@pytest.mark.parametrize('axis',range(3))
@pytest.mark.parametrize('mode',['reflect','circular'])
def test_padding_each_axis_boundary(axis,mode):
    sizes=[9,11,13];padding=[0,0,0];padding[axis]=sizes[axis]+int(mode=='circular')
    with pytest.raises(ValueError):contract.resolve_conv3d(config(padding_mode=mode,padding=padding),BatchedLayout((4,*sizes)))
    padding[axis]-=1
    assert contract.resolve_conv3d(config(padding_mode=mode,padding=padding),BatchedLayout((4,*sizes)))['layout']


@pytest.mark.parametrize('mode',['zeros','reflect','replicate','circular'])
def test_padded_volume_bound_even_with_tiny_strided_output(mode):
    c=config(in_channels=1,out_channels=1,groups=1,kernel_size=[1,1,1],stride=[4096]*3,padding=[299]*3,padding_mode=mode)
    with pytest.raises(ValueError,match='(?i)(padded|volume|100)'):
        contract.resolve_conv3d(c,BatchedLayout((1,300,300,300)))


def test_output_volume_and_multiply_add_limits_are_independent():
    with pytest.raises(ValueError):contract.resolve_conv3d(config(in_channels=1,out_channels=512,groups=1,kernel_size=[1]*3),BatchedLayout((1,10,50,400)))
    with pytest.raises(ValueError,match='(?i)(work|multiply|128)'):
        contract.resolve_conv3d(config(in_channels=512,out_channels=512,groups=1,kernel_size=[1]*3),BatchedLayout((512,8,8,8)))


def test_depthwise_and_exact_parameter_boundary():
    c=config(in_channels=500,out_channels=500,groups=1,kernel_size=[2,2,2],bias=False)
    assert contract.normalize_conv3d(c)['bias'] is False
    with pytest.raises(ValueError):contract.normalize_conv3d(c | {'bias':True})
    result=contract.resolve_conv3d(config(in_channels=512,out_channels=512,groups=512,kernel_size=[1]*3,bias=False),BatchedLayout((512,4,5,6)))
    assert result['parameterCount']==512 and result['multiplyAdds']==512*4*5*6


@pytest.mark.parametrize('axis',range(3))
def test_invalid_effective_kernel_and_output_length_per_axis(axis):
    kernel=[1,1,1];kernel[axis]=4096
    dilation=[1,1,1];dilation[axis]=4096
    with pytest.raises(ValueError):contract.resolve_conv3d(config(kernel_size=kernel,dilation=dilation,padding_policy='same'),BatchedLayout((4,9,11,13)))
    with pytest.raises(ValueError):contract.resolve_conv3d(config(kernel_size=kernel),BatchedLayout((4,9,11,13)))
