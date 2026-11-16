"""Conv1d allocation-free contract, independent formulas and pinned backend shapes."""
import pytest
import torch
from runner import conv1d_configuration as contract
from runner.tensor_layout import BatchedLayout


def config(**changes):
    return dict(in_channels=4,out_channels=6,kernel_size=3,stride=1,dilation=1,
                groups=2,bias=True,padding_policy='explicit',padding=0,padding_mode='zeros',**{}) | changes


@pytest.mark.parametrize('settings,length,expected,pads',[
    ({},11,(6,9),(0,0)),
    ({'padding':2,'stride':2},11,(6,7),(2,2)),
    ({'dilation':2,'padding_policy':'valid'},11,(6,7),(0,0)),
    ({'kernel_size':4,'padding_policy':'same'},11,(6,11),(1,2)),
    ({'kernel_size':4,'dilation':3,'padding_policy':'same'},11,(6,11),(4,5)),
    ({'groups':4,'out_channels':8,'bias':False},11,(8,9),(0,0)),
])
def test_analytical_layout_parameter_and_work_counts(settings,length,expected,pads):
    cfg=config(**settings)
    result=contract.resolve_conv1d(cfg,BatchedLayout((4,length)))
    assert result['layout']==BatchedLayout(expected)
    assert (result['paddingLeft'],result['paddingRight'])==pads
    assert result['parameterCount']==cfg['out_channels']*((cfg['in_channels']//cfg['groups'])*cfg['kernel_size']+int(cfg['bias']))
    assert result['multiplyAdds']==expected[1]*cfg['out_channels']*(cfg['in_channels']//cfg['groups'])*cfg['kernel_size']


@pytest.mark.parametrize('mode',['zeros','reflect','replicate','circular'])
@pytest.mark.parametrize('policy',['explicit','valid','same'])
def test_representative_output_shapes_match_pinned_torch(mode,policy):
    cfg=config(padding_mode=mode,padding_policy=policy,padding=2 if policy=='explicit' else 0,
               kernel_size=4 if policy=='same' else 3)
    result=contract.resolve_conv1d(cfg,BatchedLayout((4,13)))
    layer=torch.nn.Conv1d(cfg['in_channels'],cfg['out_channels'],cfg['kernel_size'],
        stride=cfg['stride'],dilation=cfg['dilation'],groups=cfg['groups'],bias=cfg['bias'],
        padding=cfg['padding'] if policy=='explicit' else policy,padding_mode=mode)
    actual=layer(torch.randn(2,4,13))
    assert tuple(actual.shape)==(2,*result['layout'].sample)


@pytest.mark.parametrize('key,value',[
    ('in_channels',True),('out_channels',0),('groups',513),('kernel_size',1.5),
    ('stride',0),('dilation',4097),('padding',-1),('bias',1),('bias',None),
    ('padding_policy','SAME'),('padding_mode','constant'),('padding_policy',[]),
    ('padding_mode',None),('kernel_size',float('inf')),('padding',float('nan')),
])
def test_strict_authored_types_and_ranges(key,value):
    with pytest.raises(ValueError):contract.normalize_conv1d(config(**{key:value}))


def test_unknown_missing_keys_canonical_integral_floats_and_no_input_mutation():
    authored=config(kernel_size=3.0,padding=1.0)
    normalized=contract.normalize_conv1d(authored)
    assert type(normalized['kernel_size']) is int and type(normalized['padding']) is int
    assert type(authored['kernel_size']) is float
    for key in authored:
        incomplete=dict(authored);del incomplete[key]
        with pytest.raises(ValueError):contract.normalize_conv1d(incomplete)
    with pytest.raises(ValueError):contract.normalize_conv1d(config(unknown=1))


@pytest.mark.parametrize('changes',[
    {'groups':3},{'out_channels':5},{'padding_policy':'valid','padding':1},
    {'padding_policy':'same','padding':1},{'padding_policy':'same','stride':2},
    {'in_channels':512,'out_channels':512,'groups':1,'kernel_size':8},
])
def test_invalid_group_policy_and_parameter_combinations(changes):
    with pytest.raises(ValueError):contract.normalize_conv1d(config(**changes))


@pytest.mark.parametrize('shape',[(4,), (4,5,6), (3,11), (4,4097)])
def test_input_sample_contract(shape):
    with pytest.raises(ValueError):contract.resolve_conv1d(config(),BatchedLayout(shape))


@pytest.mark.parametrize('changes,length',[
    ({'kernel_size':13},11),
    ({'padding_mode':'reflect','padding':11},11),
    ({'padding_mode':'circular','padding':12},11),
    ({'kernel_size':4096,'dilation':4096,'padding_policy':'same'},11),
    ({'kernel_size':1,'padding':4096},11),
    ({'in_channels':512,'out_channels':512,'groups':1,'kernel_size':1},4096),
])
def test_shape_padding_and_work_guards(changes,length):
    cfg=config(**changes)
    with pytest.raises(ValueError):contract.resolve_conv1d(cfg,BatchedLayout((cfg['in_channels'],length)))


def test_padding_mode_boundary_and_depthwise_work_are_not_overrestricted():
    assert contract.resolve_conv1d(config(padding_mode='reflect',padding=10),BatchedLayout((4,11)))['paddingLeft']==10
    assert contract.resolve_conv1d(config(padding_mode='circular',padding=11),BatchedLayout((4,11)))['paddingRight']==11
    assert contract.resolve_conv1d(config(padding_mode='replicate',padding=20),BatchedLayout((4,11)))['layout'].sample==(6,49)
    cfg=config(in_channels=512,out_channels=512,groups=512,kernel_size=1,bias=False)
    result=contract.resolve_conv1d(cfg,BatchedLayout((512,4096)))
    assert result['parameterCount']==512 and result['multiplyAdds']==512*4096
