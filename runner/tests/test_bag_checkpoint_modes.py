"""Actual trained bag weights-only reuse and honest frozen-workflow boundary."""
import hashlib
import json
import pytest
import torch
from runner import engine, frozen_checkpoint
from runner.optimizer_configuration import canonical_configuration
from runner.tests.test_bag_source_graph import graph
from runner.tests.test_prepared_bag_loader import fixture, load


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def inventory(path):return {str(p.relative_to(path)):sha(p) for p in path.rglob('*') if p.is_file()}

@pytest.fixture(autouse=True)
def preserve_determinism():
    enabled=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled()
    yield
    torch.use_deterministic_algorithms(enabled,warn_only=warn)


def trained(tmp_path):
    prepared=load(fixture(tmp_path/'source'))
    value=graph();value['nodes'][0]['config']['maximum_items_per_bag']=2
    recipe=dict(schemaVersion=1,componentID='optimizer_sgd',configurationVersion=1,
                config=canonical_configuration('optimizer_sgd',dict(lr=.03,momentum=.5)))
    options=dict(registry=engine.REGISTRY,prepared=prepared,device='cpu',batch_size=3,seed=42,optimizer_recipe=recipe)
    engine.run(value,tmp_path/'parent',epochs=2,**options)
    return value,options,tmp_path/'parent'


def test_actual_bag_weights_only_finetune_transfers_table_and_resets_memory_without_parent_changes(tmp_path,monkeypatch):
    value,options,parent=trained(tmp_path)
    before=inventory(parent);saved=torch.load(parent/'checkpoint.pt',weights_only=True,map_location='cpu')
    transferred=[];original=engine.GraphModel.load_state_dict
    def observe(self,state,*args,**kwargs):
        result=original(self,state,*args,**kwargs)
        transferred.append({k:v.detach().cpu().clone() for k,v in self.state_dict().items()})
        return result
    monkeypatch.setattr(engine.GraphModel,'load_state_dict',observe)
    child_options={**options,'seed':77,'optimizer_recipe':{**options['optimizer_recipe'],
                    'config':{**options['optimizer_recipe']['config'],'lr':0.}}}
    engine.run(value,tmp_path/'child',epochs=1,finetune_checkpoint=parent/'checkpoint.pt',**child_options)
    child=torch.load(tmp_path/'child/checkpoint.pt',weights_only=True,map_location='cpu')
    assert len(transferred)==1
    assert set(transferred[0])==set(saved['modelState'])
    for key,parameter in saved['modelState'].items():
        assert torch.equal(transferred[0][key],parameter)
        assert torch.equal(child['modelState'][key],parameter), 'Zero-rate reuse preserves the exact trained bag table'
    assert child['epoch']==1 and child['seed']==77
    steps=(len(options['prepared'][1]['trainingRowIDs'])+2)//3
    assert child['optimizerUpdates']==[steps]
    assert saved['optimizerUpdates']==[steps*2]
    assert any(not torch.equal(child['optimizerState']['state'][key]['momentum_buffer'],saved['optimizerState']['state'][key]['momentum_buffer']) for key in child['optimizerState']['state'])
    manifest=json.loads((tmp_path/'child/manifest.json').read_text())
    assert manifest['lineage']['guarantee']=='weightsOnly'
    # This native bag graph owns its table Parameter and no buffers.
    assert set(saved['modelState'])=={'layers.bag.weight'}
    assert set(manifest['lineage']['restoredState'])=={'parameters'}
    assert {'optimizer','randomGenerator','epoch','metrics'}.issubset(manifest['lineage']['resetState'])
    assert manifest['provenance']['bagSources']==options['prepared'][1]['bagSources']
    assert inventory(parent)==before


def test_frozen_dense_evaluation_refuses_real_bag_checkpoint_actionably_before_tensor_or_model_allocation(tmp_path,monkeypatch):
    _,_,parent=trained(tmp_path)
    before=inventory(parent)
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('Unsupported frozen bag source reached tensor deserialization'))
    monkeypatch.setattr(frozen_checkpoint,'GraphModel',lambda *a,**k:pytest.fail('Unsupported frozen bag source reached model allocation'))
    with pytest.raises(ValueError,match='Bag Input packets.*packet inference export'):
        frozen_checkpoint.load_frozen(parent/'checkpoint.pt',expected_manifest_sha256=sha(parent/'manifest.json'),
            expected_checkpoint_sha256=sha(parent/'checkpoint.pt'),registry_path=engine.REGISTRY,
            expected_registry_sha256=sha(engine.REGISTRY))
    assert inventory(parent)==before
