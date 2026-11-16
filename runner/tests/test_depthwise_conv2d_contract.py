"""Depthwise is an explicit authored specialization of the established grouped Conv2d contract."""
import pytest
import torch
from runner import depthwise_conv2d_configuration as contract
from runner.conv2d_configuration import resolve_conv2d
from runner.tensor_layout import BatchedLayout


def config(**changes):
    return dict(in_channels=2,channel_multiplier=3,kernel_size=[2,3],stride=[1,1],
                dilation=[1,1],bias=True,padding_policy='explicit',padding=[0,0],padding_mode='zeros') | changes


@pytest.mark.parametrize('policy',['explicit','valid','same'])
@pytest.mark.parametrize('mode',['zeros','reflect','replicate','circular'])
def test_all_padding_policies_modes_match_derived_geometry_and_actual_cpu(policy,mode):
    c=config(padding_policy=policy,padding_mode=mode,padding=[1,1] if policy=='explicit' else [0,0])
    derived=contract.conv2d_config(c)
    assert derived=={key:value for key,value in c.items() if key!='channel_multiplier'} | {'out_channels':6,'groups':2}
    result=contract.resolve_depthwise_conv2d(c,BatchedLayout((2,5,7)))
    assert result==resolve_conv2d(derived,BatchedLayout((2,5,7)))
    layer=torch.nn.Conv2d(2,6,c['kernel_size'],groups=2,padding=c['padding'] if policy=='explicit' else policy,padding_mode=mode)
    assert tuple(layer.weight.shape)==(6,1,2,3)
    assert tuple(layer(torch.randn(2,2,5,7)).shape)==(2,*result['layout'].sample)


def test_channels_are_isolated_and_multiplier_outputs_belong_to_input_channel():
    c=config(kernel_size=[1,1],bias=False)
    d=contract.conv2d_config(c)
    layer=torch.nn.Conv2d(d['in_channels'],d['out_channels'],d['kernel_size'],groups=d['groups'],bias=False).double()
    with torch.no_grad():layer.weight.copy_(torch.arange(1,7,dtype=torch.float64).reshape(6,1,1,1))
    x=torch.zeros(1,2,2,3,dtype=torch.float64,requires_grad=True)
    with torch.no_grad():x[:,0].fill_(2);x[:,1].fill_(7)
    y=layer(x)
    assert y[:,0:3].flatten().tolist()==[2.0]*6+[4.0]*6+[6.0]*6
    assert y[:,3:6].flatten().tolist()==[28.0]*6+[35.0]*6+[42.0]*6
    gradient,weight_gradient=torch.autograd.grad(y[:,:3].sum(),(x,layer.weight))
    assert torch.count_nonzero(weight_gradient[3:]).item()==0
    assert torch.all(weight_gradient[:3]==12)
    assert torch.count_nonzero(gradient[:,1]).item()==0
    assert torch.all(gradient[:,0]==6)


def test_mixed_stride_dilation_analytical_geometry_and_parameter_count():
    c=config(in_channels=3,channel_multiplier=2,kernel_size=[2,3],stride=[2,1],dilation=[2,2],padding=[1,2])
    r=contract.resolve_depthwise_conv2d(c,BatchedLayout((3,7,9)))
    assert r['layout']==BatchedLayout((6,4,9))
    assert r['parameterCount']==42
    assert r['multiplyAdds']==4*9*6*6


@pytest.mark.parametrize('changes',[{'in_channels':True},{'channel_multiplier':False},
    {'in_channels':0},{'channel_multiplier':0},{'in_channels':513},{'channel_multiplier':513},
    {'channel_multiplier':1.5},{'in_channels':float('inf')},{'channel_multiplier':float('nan')},
    {'in_channels':257,'channel_multiplier':2},{'bias':1},{'kernel_size':3},
    {'padding_policy':'same','stride':[2,1]},{'padding_policy':'valid','padding':[1,0]},
    {'padding_mode':'unknown'},{'kernel_size':[1,True]}])
def test_strict_authored_settings_and_derived_channel_limit(changes):
    with pytest.raises(ValueError):contract.normalize_depthwise_conv2d(config(**changes))


def test_exact_keys_canonical_values_and_no_derived_config_leak():
    c=config(in_channels=2.0,channel_multiplier=3.0,kernel_size=[2.0,3.0])
    normalized=contract.normalize_depthwise_conv2d(c)
    assert set(normalized)==set(c)
    assert type(normalized['in_channels']) is int and type(normalized['channel_multiplier']) is int
    assert all(type(v) is int for v in normalized['kernel_size'])
    derived=contract.conv2d_config(c);derived['kernel_size'][0]=99;derived['out_channels']=1
    normalized['padding'][0]=5
    assert c['kernel_size']==[2.0,3.0] and c['padding']==[0,0]
    assert contract.conv2d_config(c)['kernel_size']==[2,3]
    for key in c:
        missing=dict(c);del missing[key]
        with pytest.raises(ValueError):contract.normalize_depthwise_conv2d(missing)
    for key in ('groups','out_channels'):
        with pytest.raises(ValueError):contract.normalize_depthwise_conv2d(config(**{key:2}))


def test_maximum_channels_and_multiplier_endpoints():
    assert contract.conv2d_config(config(in_channels=1,channel_multiplier=512))['out_channels']==512
    d=contract.conv2d_config(config(in_channels=512,channel_multiplier=1))
    assert d['out_channels']==512 and d['groups']==512


def test_delegated_parameter_work_and_padding_refusals():
    with pytest.raises(ValueError,match='(?i)parameter'):
        contract.normalize_depthwise_conv2d(config(in_channels=512,channel_multiplier=1,kernel_size=[64,64]))
    with pytest.raises(ValueError,match='(?i)(multiply|work|128)'):
        contract.resolve_depthwise_conv2d(config(in_channels=512,channel_multiplier=1,kernel_size=[3,3]),BatchedLayout((512,200,200)))
    with pytest.raises(ValueError):
        contract.resolve_depthwise_conv2d(config(padding=[2,0],padding_mode='reflect'),BatchedLayout((2,2,5)))
    with pytest.raises(ValueError):contract.resolve_depthwise_conv2d(config(),BatchedLayout((3,5,7)))
    with pytest.raises(ValueError):contract.resolve_depthwise_conv2d(config(),BatchedLayout((2,5)))


def test_exact_parameter_budget_and_bias_crossing():
    c=config(in_channels=500,channel_multiplier=1,kernel_size=[20,200],bias=False)
    assert contract.normalize_depthwise_conv2d(c)['bias'] is False
    with pytest.raises(ValueError,match='(?i)parameter'):
        contract.normalize_depthwise_conv2d(c|{'bias':True})


def test_exact_work_budget_and_next_width():
    c=config(in_channels=512,channel_multiplier=1,kernel_size=[1,250],bias=False)
    assert contract.resolve_depthwise_conv2d(c,BatchedLayout((512,10,349)))['multiplyAdds']==128_000_000
    with pytest.raises(ValueError,match='(?i)(multiply|work|128)'):
        contract.resolve_depthwise_conv2d(c,BatchedLayout((512,10,350)))


def test_exact_output_volume_and_next_height():
    c=config(in_channels=1,channel_multiplier=100,kernel_size=[1,1],bias=False)
    assert contract.resolve_depthwise_conv2d(c,BatchedLayout((1,1000,1000)))['layout']==BatchedLayout((100,1000,1000))
    with pytest.raises(ValueError):contract.resolve_depthwise_conv2d(c,BatchedLayout((1,1001,1000)))


def test_exact_padded_volume_with_tiny_output():
    c=config(in_channels=100,channel_multiplier=1,kernel_size=[1,1],padding=[499,499],stride=[1000,1000],bias=False)
    assert contract.resolve_depthwise_conv2d(c,BatchedLayout((100,2,2)))['layout']==BatchedLayout((100,1,1))
    with pytest.raises(ValueError,match='(?i)padded'):
        contract.resolve_depthwise_conv2d(c|{'padding':[500,499]},BatchedLayout((100,2,2)))
