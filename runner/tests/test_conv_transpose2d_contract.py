"""Paired transposed geometry: explicit output ambiguity, cropping and allocation limits."""
import pytest
import torch
from runner import conv_transpose2d_configuration as contract
from runner.tensor_layout import BatchedLayout


def config(**changes):
    return dict(in_channels=4,out_channels=6,kernel_size=[2,3],stride=[1,1],
                padding=[0,0],output_padding=[0,0],dilation=[1,1],groups=2,bias=True) | changes


@pytest.mark.parametrize('changes,expected,uncropped',[
    ({},[5,8],[5,8]),
    ({'stride':[2,3],'padding':[1,2],'output_padding':[1,2]},[7,16],[9,20]),
    ({'stride':[1,2],'dilation':[2,1],'output_padding':[1,1]},[7,14],[7,14]),
    ({'kernel_size':[1,1],'stride':[3,2],'padding':[3,2]},[4,7],[10,11]),
    ({'groups':4,'out_channels':8,'bias':False},[5,8],[5,8]),
])
def test_rectangular_analytical_geometry_and_actual_cpu_shape(changes,expected,uncropped):
    c=config(**changes)
    result=contract.resolve_conv_transpose2d(c,BatchedLayout((4,4,6)))
    assert result['layout']==BatchedLayout((c['out_channels'],*expected))
    assert result['uncroppedOutputShape']==uncropped
    weight_count=4*(c['out_channels']//c['groups'])*c['kernel_size'][0]*c['kernel_size'][1]
    assert result['parameterCount']==weight_count+int(c['bias'])*c['out_channels']
    assert result['multiplyAdds']==4*6*weight_count
    layer=torch.nn.ConvTranspose2d(c['in_channels'],c['out_channels'],c['kernel_size'],
        stride=c['stride'],padding=c['padding'],output_padding=c['output_padding'],
        dilation=c['dilation'],groups=c['groups'],bias=c['bias'])
    assert tuple(layer(torch.randn(2,4,4,6)).shape)==(2,c['out_channels'],*expected)


@pytest.mark.parametrize('key',['kernel_size','stride','padding','output_padding','dilation'])
@pytest.mark.parametrize('value',[1,[1],[1,1,1],(1,1),[True,1],[1,False],[1.5,1],[1,float('inf')],[float('nan'),1],[4097,1]])
def test_spatial_pairs_are_strict_and_bounded(key,value):
    with pytest.raises(ValueError):contract.normalize_conv_transpose2d(config(**{key:value}))


@pytest.mark.parametrize('changes',[{'in_channels':True},{'out_channels':513},{'groups':0},
    {'bias':1},{'bias':None},{'stride':[0,1]},{'kernel_size':[1,0]},
    {'padding':[-1,0]},{'output_padding':[0,-1]}, {'dilation':[0,1]},
    {'groups':3},{'out_channels':5},
    {'stride':[2,1],'dilation':[3,2],'output_padding':[3,0]},
    {'stride':[2,1],'dilation':[3,2],'output_padding':[0,2]}])
def test_scalar_bounds_groups_and_each_axis_or_rule(changes):
    with pytest.raises(ValueError):contract.normalize_conv_transpose2d(config(**changes))


def test_integral_float_pairs_are_canonical_disjoint_copies_and_keys_exact():
    authored=config(kernel_size=[2.0,3.0],padding=[0.0,0.0])
    result=contract.normalize_conv_transpose2d(authored)
    assert all(type(v) is int for key in ('kernel_size','padding') for v in result[key])
    result['kernel_size'][0]=99
    assert authored['kernel_size']==[2.0,3.0]
    for key in authored:
        bad=dict(authored);del bad[key]
        with pytest.raises(ValueError):contract.normalize_conv_transpose2d(bad)
    for key,value in [('padding_policy','same'),('padding_mode','zeros'),('output_size',[4,6])]:
        with pytest.raises(ValueError):contract.normalize_conv_transpose2d(config(**{key:value}))


@pytest.mark.parametrize('shape',[(4,6),(4,4,6,2),(3,4,6),(4,4097,6),(4,4,4097)])
def test_exact_channel_height_width_input(shape):
    with pytest.raises(ValueError):contract.resolve_conv_transpose2d(config(),BatchedLayout(shape))


@pytest.mark.parametrize('changes',[{'padding':[10,0]},{'padding':[0,10]},
    {'stride':[4096,1]},{'stride':[1,4096]}])
def test_each_output_axis_must_be_positive_and_bounded(changes):
    with pytest.raises(ValueError):contract.resolve_conv_transpose2d(config(**changes),BatchedLayout((4,4,6)))


def test_parameter_boundary_bias_and_work_before_cropping():
    c=config(in_channels=500,out_channels=500,groups=1,kernel_size=[2,4],bias=False)
    assert contract.normalize_conv_transpose2d(c)['bias'] is False
    with pytest.raises(ValueError,match='(?i)(parameter|million)'):
        contract.normalize_conv_transpose2d(c | {'bias':True})
    c=config(in_channels=512,out_channels=512,groups=1,kernel_size=[1,1],padding=[11,11])
    with pytest.raises(ValueError,match='(?i)(work|multiply|128)'):
        contract.resolve_conv_transpose2d(c,BatchedLayout((512,23,23)))


def test_uncropped_volume_is_checked_even_with_tiny_final_output():
    c=config(in_channels=1,out_channels=6,groups=1,kernel_size=[2,2],
             dilation=[4096,4096],padding=[2048,2048])
    # Final shape [6,1,1], 30 parameters, 24 multiply-adds; uncropped volume alone exceeds 100M.
    with pytest.raises(ValueError,match='(?i)uncropped'):
        contract.resolve_conv_transpose2d(c,BatchedLayout((1,1,1)))


def test_input_and_output_sample_volume_are_bounded_before_allocation():
    with pytest.raises(ValueError):
        contract.resolve_conv_transpose2d(config(in_channels=512),BatchedLayout((512,512,512)))
    c=config(in_channels=1,out_channels=8,groups=1,kernel_size=[1,1],stride=[4095,4095])
    with pytest.raises(ValueError):contract.resolve_conv_transpose2d(c,BatchedLayout((1,2,2)))


def test_exact_work_boundary_uses_input_area_not_expanded_output_area():
    c=config(in_channels=500,out_channels=512,groups=1,kernel_size=[1,1],stride=[2,2],bias=False)
    accepted=contract.resolve_conv_transpose2d(c,BatchedLayout((500,20,25)))
    assert accepted['multiplyAdds']==128_000_000
    assert accepted['layout']==BatchedLayout((512,39,49))
    with pytest.raises(ValueError,match='(?i)(work|multiply|128)'):
        contract.resolve_conv_transpose2d(c,BatchedLayout((500,20,26)))


def test_exact_uncropped_volume_boundary_accepts_then_rejects():
    c=config(in_channels=1,out_channels=100,groups=1,kernel_size=[1,1],
             stride=[1000,1000],output_padding=[999,999],padding=[499,499],bias=False)
    accepted=contract.resolve_conv_transpose2d(c,BatchedLayout((1,1,1)))
    assert accepted['uncroppedOutputShape']==[1000,1000]
    assert accepted['layout']==BatchedLayout((100,2,2))
    with pytest.raises(ValueError,match='(?i)uncropped'):
        contract.resolve_conv_transpose2d(c | {'stride':[1001,1000],'output_padding':[1000,999]},BatchedLayout((1,1,1)))
