"""Strict live Conv2d discovery; paired spatial geometry is part of the contract."""
from runner.tests.cli_discovery import capabilities_cli
from runner import depthwise_conv2d_configuration
from runner.engine import capabilities
from runner.tests.test_depthwise_conv2d_graph import registry


def expected():
    return {
        'version': {'major':1,'minor':0,'patch':0},
        'adapterID':'kwip.runner.graph-v2.depthwise_conv2d','graphSchemaVersions':[2],
        'sampleRank':3,'preservesBatch':True,'graphDTypes':['float32'],
        'spatialParameterLength':2,
        'channelMinimum':1,'channelMaximum':512,
        'lengthMinimum':1,'lengthMaximum':4096,
        'kernelMinimum':1,'kernelMaximum':4096,
        'strideMinimum':1,'strideMaximum':4096,
        'dilationMinimum':1,'dilationMaximum':4096,
        'channelMultiplierMinimum':1,'channelMultiplierMaximum':512,
        'outputChannelMaximum':512,'groupsRule':'inputChannels',
        'outputChannelsRule':'inputChannelsTimesMultiplier',
        'configurationKeys':['in_channels','channel_multiplier','kernel_size','stride','dilation',
                             'bias','padding_policy','padding','padding_mode'],
        'paddingMinimum':0,'paddingMaximum':4096,
        'maximumParameters':2_000_000,'maximumMultiplyAdds':128_000_000,
        'maximumSampleElements':100_000_000,'maximumPaddedSampleElements':100_000_000,
        'paddingPolicies':['explicit','valid','same'],
        'paddingModes':['zeros','reflect','replicate','circular'],
        'biasModes':[False,True],
    }


def test_live_probe_emits_exact_paired_conv2d_contract():
    assert capabilities()['componentCapabilities']['depthwise_conv2d']==expected()


def test_descriptor_nested_values_do_not_leak_between_probes():
    actual=depthwise_conv2d_configuration.capability()
    assert actual==expected()
    actual['version']['major']=99
    actual['graphSchemaVersions'].append(1)
    actual['graphDTypes'].append('float64')
    actual['paddingPolicies'].clear()
    actual['biasModes'].pop()
    actual['configurationKeys'].clear()
    assert depthwise_conv2d_configuration.capability()==expected()
    assert capabilities()['componentCapabilities']['depthwise_conv2d']==expected()


def test_actual_cli_advertises_specialization_and_export_executes_derived_groups(registry):
    import json
    import subprocess
    import sys
    from pathlib import Path
    import torch
    from runner.engine import GraphModel, export_source
    from runner.tests.test_depthwise_conv2d_graph import graph
    result=capabilities_cli()
    assert result.returncode==0,result.stderr
    events=[json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    payload=next(e for e in events if e.get('event')=='capabilities')
    assert payload['componentCapabilities']['depthwise_conv2d']==expected()
    model=GraphModel(graph({'channel_multiplier':3,'bias':False,'padding_policy':'same'}),registry)
    scope={};exec(export_source(model),scope);exported=scope['Model']()
    exported.load_state_dict(model.state_dict())
    layer=exported.layers['convolution']
    assert isinstance(layer,torch.nn.Conv2d) and layer.groups==2 and layer.out_channels==6
    assert layer.bias is None and tuple(layer.weight.shape)==(6,1,2,3)
    x=torch.tensor([[0.25,-0.5],[1.0,2.0]])
    torch.testing.assert_close(exported(x),model(x),rtol=0,atol=0)
