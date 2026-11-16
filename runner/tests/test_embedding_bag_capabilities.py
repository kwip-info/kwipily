"""Independent literal descriptor and actual discovery regression checks."""
import copy
import pytest
from runner import bag_source, embedding_bag, engine

EXPECTED = {'bag_input': {'version': {'major': 1, 'minor': 0, 'patch': 0}, 'adapterID': 'kwip.runner.graph-v2.bag_input', 'graphSchemaVersions': [2], 'packetSchemaVersion': 1, 'layouts': ['fixed', 'packed'], 'coordinateDTypes': {'indices': 'int64', 'offsets': 'int64', 'per_sample_weights': 'float32'}, 'maximumItemsPerBag': 128, 'maximumBags': 4096, 'maximumItems': 1000000, 'maximumOutputElements': 2000000, 'integerDerivatives': 'unavailable', 'sourceWeights': 'realPairedInput', 'batching': 'wholeBags', 'emptyPackedBags': 'preserve'}, 'embedding_bag': {'version': {'major': 1, 'minor': 0, 'patch': 0}, 'adapterID': 'kwip.runner.graph-v2.embedding_bag', 'graphSchemaVersions': [2], 'packetSchemaVersion': 1, 'layouts': ['fixed', 'packed'], 'coordinateDTypes': {'indices': 'int64', 'offsets': 'int64', 'per_sample_weights': 'float32'}, 'maximumItemsPerBag': 128, 'maximumBags': 4096, 'maximumItems': 1000000, 'maximumOutputElements': 2000000, 'integerDerivatives': 'unavailable', 'sourceWeights': 'realPairedInput', 'batching': 'wholeBags', 'emptyPackedBags': 'preserve', 'modes': ['mean', 'sum', 'max'], 'weightedModes': ['sum'], 'outputDType': 'float32', 'maximumParameters': 2000000, 'maximumStaticTableElements': 100000000, 'paddingGradient': 'suppressed', 'frozenWeightsMayRenormalize': True, 'trainableSparseAdam': False, 'mpsMaxNormExecution': 'reject', 'mpsTrainableFrequencyScaling': 'reject', 'sourceWeightsDerivatives': 'sumOnly'}}

@pytest.mark.parametrize("component,module",[("bag_input",bag_source),("embedding_bag",embedding_bag)])
def test_real_descriptor_exact_reviewed_coordinate_and_device_policy(component,module):
    assert module.capability()==EXPECTED[component]

@pytest.mark.parametrize("component,module",[("bag_input",bag_source),("embedding_bag",embedding_bag)])
def test_descriptor_returns_fresh_nested_values(component,module):
    first=module.capability()
    first['layouts'].append('dense')
    first['coordinateDTypes']['indices']='float32'
    assert module.capability()==EXPECTED[component]

def test_actual_engine_discovery_registers_independent_pair_without_old_embedding_changes():
    result=engine.capabilities()['componentCapabilities']
    assert result['bag_input']==EXPECTED['bag_input']
    assert result['embedding_bag']==EXPECTED['embedding_bag']
    assert result['embedding']==engine.embedding_capability()
    assert result['embedding']['mpsMaxNormExecution']=='cpuFallback'
    assert result['embedding_bag']['mpsMaxNormExecution']=='reject'
