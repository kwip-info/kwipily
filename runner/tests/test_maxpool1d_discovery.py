from runner import maxpool1d_configuration
from runner.engine import capabilities


def expected():
    return dict(version=dict(major=1,minor=0,patch=0),adapterID='kwip.runner.graph-v2.max_pool1d',
        graphSchemaVersions=[2],sampleRank=2,preservesBatch=True,graphDTypes=['float32'],
        channelMinimum=1,channelMaximum=512,lengthMinimum=1,lengthMaximum=4096,
        kernelMinimum=1,kernelMaximum=4096,strideMinimum=1,strideMaximum=4096,
        dilationMinimum=1,dilationMaximum=4096,paddingMinimum=0,paddingMaximum=2048,
        paddingKernelDivisor=2,nullStrideFollowsKernel=True,ceilModes=[False,True],
        returnIndicesModes=[False,True],outputDTypes={'out':'float32','indices':'int64'})


def test_probe_and_independent_descriptor():
    assert capabilities()['componentCapabilities']['max_pool1d']==expected()
    first=maxpool1d_configuration.capability();first['outputDTypes']['indices']='float32'
    first['ceilModes'].clear()
    assert maxpool1d_configuration.capability()==expected()
