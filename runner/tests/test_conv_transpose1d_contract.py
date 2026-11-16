"""Transposed cross-correlation geometry, independent of ordinary convolution padding."""
import pytest
import torch
from runner import conv_transpose1d_configuration as contract
from runner.tensor_layout import BatchedLayout


def config(**changes):
    return dict(in_channels=4,out_channels=6,kernel_size=3,stride=1,padding=0,
                output_padding=0,dilation=1,groups=2,bias=True) | changes


@pytest.mark.parametrize('changes,length,expected,uncropped',[
    ({},7,9,9),
    ({'stride':2,'padding':1,'output_padding':1},7,14,16),
    ({'stride':1,'dilation':2,'output_padding':1},7,12,12),
    ({'stride':3,'dilation':1,'output_padding':2},7,23,23),
    ({'kernel_size':1,'stride':3,'padding':4},7,11,19),
    ({'groups':4,'out_channels':8,'bias':False},7,9,9),
])
def test_analytical_geometry_parameter_orientation_and_work(changes,length,expected,uncropped):
    c=config(**changes);r=contract.resolve_conv_transpose1d(c,BatchedLayout((4,length)))
    assert r['layout']==BatchedLayout((c['out_channels'],expected))
    assert r['uncroppedOutputLength']==uncropped
    assert r['parameterCount']==c['in_channels']*(c['out_channels']//c['groups'])*c['kernel_size']+int(c['bias'])*c['out_channels']
    assert r['multiplyAdds']==length*c['in_channels']*(c['out_channels']//c['groups'])*c['kernel_size']
    layer=torch.nn.ConvTranspose1d(c['in_channels'],c['out_channels'],c['kernel_size'],stride=c['stride'],padding=c['padding'],output_padding=c['output_padding'],dilation=c['dilation'],groups=c['groups'],bias=c['bias'])
    assert tuple(layer(torch.randn(2,4,length)).shape)==(2,c['out_channels'],expected)


@pytest.mark.parametrize('key,value',[
    ('in_channels',True),('out_channels',0),('groups',513),('kernel_size',1.5),
    ('stride',0),('dilation',4097),('padding',-1),('output_padding',4097),
    ('bias',1),('bias',None),('kernel_size',[3]),('stride',float('inf')),('padding',float('nan')),
])
def test_strict_authored_types_and_ranges(key,value):
    with pytest.raises(ValueError):contract.normalize_conv_transpose1d(config(**{key:value}))


def test_exact_keys_and_integral_float_canonical_copy():
    authored=config(kernel_size=3.0,padding=1.0,output_padding=0.0)
    normalized=contract.normalize_conv_transpose1d(authored)
    for key in ('kernel_size','padding','output_padding'):assert type(normalized[key]) is int
    normalized['padding']=0
    assert type(authored['padding']) is float and authored['padding']==1
    for key in authored:
        missing=dict(authored);del missing[key]
        with pytest.raises(ValueError):contract.normalize_conv_transpose1d(missing)
    for key,value in [('padding_policy','same'),('padding_mode','zeros'),('output_size',[2,6,9])]:
        with pytest.raises(ValueError):contract.normalize_conv_transpose1d(config(**{key:value}))


@pytest.mark.parametrize('changes',[{'groups':3},{'out_channels':5},{'stride':1,'dilation':1,'output_padding':1},{'stride':2,'dilation':3,'output_padding':3}])
def test_groups_and_output_padding_or_rule(changes):
    with pytest.raises(ValueError):contract.normalize_conv_transpose1d(config(**changes))


@pytest.mark.parametrize('shape',[(4,), (4,5,6), (3,7), (4,4097)])
def test_input_sample_contract(shape):
    with pytest.raises(ValueError):contract.resolve_conv_transpose1d(config(),BatchedLayout(shape))


def test_negative_oversize_output_and_uncropped_allocation_guard():
    with pytest.raises(ValueError):contract.resolve_conv_transpose1d(config(padding=10),BatchedLayout((4,7)))
    with pytest.raises(ValueError):contract.resolve_conv_transpose1d(config(stride=4096),BatchedLayout((4,7)))
    # This extreme uncropped extent must reject before allocation. With valid final
    # length and padding<=4096, uncropped volume cannot independently reach 100m.
    with pytest.raises(ValueError):contract.resolve_conv_transpose1d(config(kernel_size=4096,dilation=4096),BatchedLayout((4,7)))


def test_parameter_bias_boundary_and_work_limit():
    c=config(in_channels=500,out_channels=500,groups=1,kernel_size=8,bias=False)
    assert contract.normalize_conv_transpose1d(c)['bias'] is False
    with pytest.raises(ValueError):contract.normalize_conv_transpose1d(c | {'bias':True})
    with pytest.raises(ValueError,match='(?i)(work|multiply|128)'):
        contract.resolve_conv_transpose1d(config(in_channels=512,out_channels=512,groups=1,kernel_size=1),BatchedLayout((512,4096)))
    depthwise=contract.resolve_conv_transpose1d(config(in_channels=512,out_channels=512,groups=512,kernel_size=1,bias=False),BatchedLayout((512,4096)))
    assert depthwise['parameterCount']==512 and depthwise['multiplyAdds']==512*4096
