from runner import maxpool2d_configuration
from runner.engine import capabilities


def expected():
    return dict(version=dict(major=1,minor=0,patch=0),adapterID='kwip.runner.graph-v2.max_pool2d',
        graphSchemaVersions=[2],sampleRank=3,preservesBatch=True,graphDTypes=['float32'],
        channelMinimum=1,channelMaximum=512,spatialMinimum=1,spatialMaximum=4096,spatialAxes=["height","width"],indicesMeaning="flattenedSpatialPosition",
        kernelMinimum=1,kernelMaximum=4096,strideMinimum=1,strideMaximum=4096,
        dilationMinimum=1,dilationMaximum=4096,paddingMinimum=0,paddingMaximum=2048,
        paddingKernelDivisor=2,nullStrideFollowsKernel=True,ceilModes=[False,True],
        returnIndicesModes=[False,True],outputDTypes={'out':'float32','indices':'int64'})


def test_probe_and_independent_descriptor():
    assert capabilities()['componentCapabilities']['max_pool2d']==expected()
    first=maxpool2d_configuration.capability();first['outputDTypes']['indices']='float32'
    first['ceilModes'].clear()
    assert maxpool2d_configuration.capability()==expected()
