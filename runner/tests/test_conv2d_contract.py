"""Paired Conv2d allocation-free contract and independent backend shape checks."""
import pytest
import torch
from runner import conv2d_configuration as contract
from runner.tensor_layout import BatchedLayout


def config(**changes):
    return dict(in_channels=4,out_channels=6,kernel_size=[3,2],stride=[1,1],dilation=[1,1],
                groups=2,bias=True,padding_policy='explicit',padding=[0,0],padding_mode='zeros') | changes


@pytest.mark.parametrize('settings,shape,expected,pads',[
    ({},(4,11,13),(6,9,12),(0,0,0,0)),
    ({'padding':[2,1],'stride':[2,3]},(4,11,13),(6,7,5),(1,1,2,2)),
    ({'dilation':[2,3],'padding_policy':'valid'},(4,11,13),(6,7,10),(0,0,0,0)),
    ({'kernel_size':[4,2],'padding_policy':'same'},(4,11,13),(6,11,13),(0,1,1,2)),
    ({'kernel_size':[4,2],'dilation':[3,2],'padding_policy':'same'},(4,11,13),(6,11,13),(1,1,4,5)),
    ({'out_channels':8,'groups':4,'bias':False},(4,11,13),(8,9,12),(0,0,0,0)),
])
def test_analytical_layout_counts_and_axis_padding(settings,shape,expected,pads):
    cfg=config(**settings);result=contract.resolve_conv2d(cfg,BatchedLayout(shape))
    assert result['layout']==BatchedLayout(expected)
    assert tuple(result[key] for key in ('paddingLeft','paddingRight','paddingTop','paddingBottom'))==pads
    fan_in=(cfg['in_channels']//cfg['groups'])*cfg['kernel_size'][0]*cfg['kernel_size'][1]
    assert result['parameterCount']==cfg['out_channels']*(fan_in+int(cfg['bias']))
    assert result['multiplyAdds']==expected[1]*expected[2]*cfg['out_channels']*fan_in


@pytest.mark.parametrize('policy',['explicit','valid','same'])
@pytest.mark.parametrize('mode',['zeros','reflect','replicate','circular'])
def test_all_policy_mode_shapes_match_actual_torch(policy,mode):
    cfg=config(kernel_size=[4,2],padding_policy=policy,padding_mode=mode,
        padding=[2,1] if policy=='explicit' else [0,0],
        stride=[1,1] if policy=='same' else [2,3],dilation=[2,1])
    resolved=contract.resolve_conv2d(cfg,BatchedLayout((4,17,19)))
    layer=torch.nn.Conv2d(cfg['in_channels'],cfg['out_channels'],cfg['kernel_size'],stride=cfg['stride'],
        dilation=cfg['dilation'],groups=cfg['groups'],bias=cfg['bias'],padding_mode=mode,
        padding=cfg['padding'] if policy=='explicit' else policy)
    assert tuple(layer(torch.randn(2,4,17,19)).shape)==(2,*resolved['layout'].sample)


@pytest.mark.parametrize('field',['kernel_size','stride','dilation','padding'])
@pytest.mark.parametrize('value',[1,[1],[1,2,3],[True,1],[1,1.5],[1,None],'1,2',None])
def test_pairs_are_exact_and_never_scalar_coerced(field,value):
    with pytest.raises(ValueError):contract.normalize_conv2d(config(**{field:value}))


def test_pair_normalization_and_no_authored_mutation():
    authored=config(kernel_size=[3.0,2.0],padding=[1.0,0.0])
    canonical=contract.normalize_conv2d(authored)
    assert canonical['kernel_size']==[3,2] and all(type(v) is int for v in canonical['kernel_size'])
    assert canonical['padding']==[1,0] and all(type(v) is int for v in canonical['padding'])
    canonical['kernel_size'][0]=1
    assert authored['kernel_size']==[3.0,2.0] and type(authored['kernel_size'][0]) is float
    for key in authored:
        missing=dict(authored);del missing[key]
        with pytest.raises(ValueError):contract.normalize_conv2d(missing)
    with pytest.raises(ValueError):contract.normalize_conv2d(config(extra=1))


@pytest.mark.parametrize('changes',[
    {'in_channels':True},{'out_channels':0},{'groups':513},{'groups':3},{'out_channels':5},
    {'kernel_size':[0,1]},{'stride':[1,4097]},{'dilation':[1,float('inf')]},
    {'padding':[0,-1]},{'padding':[4097,0]},{'bias':1},{'bias':None},
    {'padding_policy':'SAME'},{'padding_mode':'constant'},
    {'padding_policy':'valid','padding':[0,1]}, {'padding_policy':'same','stride':[1,2]},
    {'in_channels':512,'out_channels':512,'groups':1,'kernel_size':[3,3]},
])
def test_strict_values_policy_groups_and_parameter_budget(changes):
    with pytest.raises(ValueError):contract.normalize_conv2d(config(**changes))


@pytest.mark.parametrize('shape',[(4,11),(4,11,13,1),(3,11,13),(4,4097,1),(4,1,4097)])
def test_input_rank_channels_and_axis_limits(shape):
    with pytest.raises(ValueError):contract.resolve_conv2d(config(),BatchedLayout(shape))


@pytest.mark.parametrize('changes,shape',[
    ({'kernel_size':[13,1]},(4,11,13)),
    ({'padding_mode':'reflect','padding':[11,1]},(4,11,13)),
    ({'padding_mode':'reflect','padding':[1,13]},(4,11,13)),
    ({'padding_mode':'circular','padding':[12,1]},(4,11,13)),
    ({'padding_mode':'circular','padding':[1,14]},(4,11,13)),
    ({'kernel_size':[4096,1],'dilation':[4096,1],'padding_policy':'same'},(4,11,13)),
    ({'kernel_size':[1,1],'padding':[4096,0]},(4,11,13)),
    ({'in_channels':512,'out_channels':512,'groups':1,'kernel_size':[1,1]},(512,32,32)),
])
def test_per_axis_padding_length_and_work_guards(changes,shape):
    with pytest.raises(ValueError):contract.resolve_conv2d(config(**changes),BatchedLayout(shape))


@pytest.mark.parametrize('mode',['zeros','reflect','replicate','circular'])
def test_padded_volume_rejected_before_allocation_for_all_modes(mode):
    # Input/output can be bounded and cheap after striding, yet prepadding is not.
    cfg=config(in_channels=8,out_channels=8,groups=8,kernel_size=[1,1],
               stride=[4096,4096],padding=[2047,2047],padding_mode=mode)
    with pytest.raises(ValueError,match='(?i)(padded|volume|100)'):
        contract.resolve_conv2d(cfg,BatchedLayout((8,2048,2048)))


def test_depthwise_boundary_and_parameter_bias_count():
    cfg=config(in_channels=500,out_channels=500,groups=1,kernel_size=[2,4],bias=False)
    assert contract.normalize_conv2d(cfg)['bias'] is False # exactly 2m parameters
    with pytest.raises(ValueError):contract.normalize_conv2d(cfg | {'bias':True})
    bounded=config(in_channels=512,out_channels=512,groups=512,kernel_size=[1,1],bias=False)
    result=contract.resolve_conv2d(bounded,BatchedLayout((512,8,8)))
    assert result['parameterCount']==512 and result['multiplyAdds']==512*8*8


def test_output_volume_retains_shared_hundred_million_bound():
    cfg=config(in_channels=1,out_channels=512,groups=1,kernel_size=[1,1],bias=False)
    # 102.4m output elements is below the work cap but above tensor layout capacity.
    with pytest.raises(ValueError):contract.resolve_conv2d(cfg,BatchedLayout((1,500,400)))
