"""Exact live transposed-convolution discovery and executable source boundary."""
from runner.tests.cli_discovery import capabilities_cli
import json
from pathlib import Path
import subprocess
import sys

import torch

from runner import conv_transpose1d_configuration
from runner.engine import GraphModel, capabilities, export_source
from runner.tests.test_conv_transpose1d_graph import graph, registry


def expected():
    return {
        'version': {'major':1,'minor':0,'patch':0},
        'adapterID':'kwip.runner.graph-v2.conv_transpose1d','graphSchemaVersions':[2],
        'sampleRank':2,'preservesBatch':True,'graphDTypes':['float32'],
        'spatialParameterKind':'scalar',
        'channelMinimum':1,'channelMaximum':512,
        'lengthMinimum':1,'lengthMaximum':4096,
        'kernelMinimum':1,'kernelMaximum':4096,
        'strideMinimum':1,'strideMaximum':4096,
        'dilationMinimum':1,'dilationMaximum':4096,
        'groupMinimum':1,'groupMaximum':512,
        'paddingMinimum':0,'paddingMaximum':4096,
        'outputPaddingMinimum':0,'outputPaddingMaximum':4096,
        'maximumParameters':2_000_000,'maximumMultiplyAdds':128_000_000,
        'maximumSampleElements':100_000_000,'maximumUncroppedSampleElements':100_000_000,
        'configurationKeys':['in_channels','out_channels','kernel_size','stride','padding',
                             'output_padding','dilation','groups','bias'],
        'paddingMode':'zeros','outputPaddingRule':'lessThanStrideOrDilation',
        'supportsDynamicOutputSize':False,'biasModes':[False,True],
    }


def test_live_probe_and_actual_cli_emit_exact_scalar_contract():
    assert capabilities()['componentCapabilities']['conv_transpose1d']==expected()
    result=capabilities_cli()
    assert result.returncode==0,result.stderr
    events=[json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    payload=next(event for event in events if event.get('event')=='capabilities')
    assert payload['componentCapabilities']['conv_transpose1d']==expected()


def test_descriptor_nested_values_are_independent():
    value=conv_transpose1d_configuration.capability()
    assert value==expected()
    value['version']['major']=99
    value['graphSchemaVersions'].append(1)
    value['graphDTypes'].append('float64')
    value['configurationKeys'].clear()
    value['biasModes'].pop()
    assert conv_transpose1d_configuration.capability()==expected()
    assert capabilities()['componentCapabilities']['conv_transpose1d']==expected()


def test_discovered_or_rule_has_executable_export_with_owned_output_padding(registry):
    descriptor=capabilities()['componentCapabilities']['conv_transpose1d']
    assert descriptor['outputPaddingRule']=='lessThanStrideOrDilation'
    authored=graph({'stride':1,'dilation':2,'output_padding':1,'bias':False})
    model=GraphModel(authored,registry)
    scope={}
    exec(export_source(model),scope)
    exported=scope['Model']()
    exported.load_state_dict(model.state_dict())
    layer=exported.layers[model.layer_keys['convolution']]
    assert isinstance(layer,torch.nn.ConvTranspose1d)
    assert layer.output_padding==(1,) and layer.stride==(1,) and layer.dilation==(2,)
    assert layer.bias is None
    inputs=torch.tensor([[0.25,-0.75],[1.5,0.125]],dtype=torch.float32)
    torch.testing.assert_close(exported(inputs),model(inputs),rtol=0,atol=0)
