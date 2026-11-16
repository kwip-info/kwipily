"""Strict live Conv2d discovery; paired spatial geometry is part of the contract."""
from runner import conv2d_configuration
from runner.engine import capabilities


def expected():
    return {
        'version': {'major':1,'minor':0,'patch':0},
        'adapterID':'kwip.runner.graph-v2.conv2d','graphSchemaVersions':[2],
        'sampleRank':3,'preservesBatch':True,'graphDTypes':['float32'],
        'spatialParameterLength':2,
        'channelMinimum':1,'channelMaximum':512,
        'lengthMinimum':1,'lengthMaximum':4096,
        'kernelMinimum':1,'kernelMaximum':4096,
        'strideMinimum':1,'strideMaximum':4096,
        'dilationMinimum':1,'dilationMaximum':4096,
        'groupMinimum':1,'groupMaximum':512,
        'paddingMinimum':0,'paddingMaximum':4096,
        'maximumParameters':2_000_000,'maximumMultiplyAdds':128_000_000,
        'maximumSampleElements':100_000_000,'maximumPaddedSampleElements':100_000_000,
        'paddingPolicies':['explicit','valid','same'],
        'paddingModes':['zeros','reflect','replicate','circular'],
        'biasModes':[False,True],
    }


def test_live_probe_emits_exact_paired_conv2d_contract():
    assert capabilities()['componentCapabilities']['conv2d']==expected()


def test_descriptor_nested_values_do_not_leak_between_probes():
    actual=conv2d_configuration.capability()
    assert actual==expected()
    actual['version']['major']=99
    actual['graphSchemaVersions'].append(1)
    actual['graphDTypes'].append('float64')
    actual['paddingPolicies'].clear()
    actual['biasModes'].pop()
    assert conv2d_configuration.capability()==expected()
    assert capabilities()['componentCapabilities']['conv2d']==expected()
