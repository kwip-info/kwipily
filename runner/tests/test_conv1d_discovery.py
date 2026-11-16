"""Live Conv1d discovery must match its reviewed bounded contract exactly."""
from runner import conv1d_configuration
from runner.engine import capabilities


def expected():
    return {
        'version': {'major':1,'minor':0,'patch':0},
        'adapterID':'kwip.runner.graph-v2.conv1d','graphSchemaVersions':[2],
        'sampleRank':2,'preservesBatch':True,'graphDTypes':['float32'],
        'channelMinimum':1,'channelMaximum':512,
        'lengthMinimum':1,'lengthMaximum':4096,
        'kernelMinimum':1,'kernelMaximum':4096,
        'strideMinimum':1,'strideMaximum':4096,
        'dilationMinimum':1,'dilationMaximum':4096,
        'groupMinimum':1,'groupMaximum':512,
        'paddingMinimum':0,'paddingMaximum':4096,
        'maximumParameters':2_000_000,'maximumMultiplyAdds':128_000_000,
        'paddingPolicies':['explicit','valid','same'],
        'paddingModes':['zeros','reflect','replicate','circular'],
        'biasModes':[False,True],
    }


def test_live_probe_contains_exact_conv1d_descriptor():
    assert capabilities()['componentCapabilities']['conv1d']==expected()


def test_descriptor_returns_independent_nested_state():
    first=conv1d_configuration.capability()
    assert first==expected()
    first['version']['minor']=99
    first['paddingPolicies'].clear()
    first['paddingModes'].append('unsupported')
    first['graphSchemaVersions'].append(1)
    assert conv1d_configuration.capability()==expected()
    assert capabilities()['componentCapabilities']['conv1d']==expected()
