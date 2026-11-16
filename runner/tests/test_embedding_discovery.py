import pytest
from runner.engine import capabilities


def test_discovery_reports_product_lookup_domain_and_mps_fallback():
    descriptor=capabilities()['componentCapabilities']['embedding']
    assert descriptor == {
        'version':{'major':1,'minor':0,'patch':0},
        'adapterID':'kwip.runner.graph-v2.embedding',
        'graphSchemaVersions':[2],
        'inputDType':'int64','weightDType':'float32',
        'maximumInputSampleRank':6,'maximumOutputRank':8,
        'maximumPreparedColumns':128,'maximumParameters':2000000,
        'preservesBatch':True,'outOfRangePolicy':'reject',
        'paddingGradient':'suppressed','frequencyScaling':True,
        'frozenWeightsMayRenormalize':True,
        'trainingOptimizer':'Adam','trainableSparseGradients':False,
        'previewCacheSchemaForMaxNorm':2,
        'mpsMaxNormExecution':'cpuFallback',
    }
    descriptor['graphSchemaVersions'].append(99)
    assert capabilities()['componentCapabilities']['embedding']['graphSchemaVersions']==[2]
