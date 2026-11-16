import json
import pytest
import torch
from runner.preview import preview
from runner.tests.test_embedding_graph import graph,registry
from runner.tests.test_embedding_lifecycle import prepared

@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('maximum',[None,1.0])
def test_index_cache_reuse_and_forward_mutation_state(tmp_path,registry,maximum,device):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
    value=graph();value['nodes'][1]['config']['max_norm']=maximum
    options=dict(registry=registry,device=device,prepared=prepared())
    full=preview(value,tmp_path/'full',**options)
    staged=preview(value,tmp_path/'stage',stop_at='lookup',**options)
    resumed=preview(value,tmp_path/'resumed',resume_from='flat',cache=staged['cachePath'],**options)
    assert full['previews']==resumed['previews']
    assert full['previews']['source']['dtype']=='int64'
    assert full['previews']['source']['inferenceOnly'] is True
    assert resumed['reusedNodeIDs']==['source','lookup']
    assert resumed['executedNodeIDs']==['flat','head','result']
    assert full['finalWeightsHash']==resumed['finalWeightsHash']
    assert (full['initialWeightsHash']!=full['finalWeightsHash']) == (maximum is not None)
    meta=json.loads((tmp_path/'stage/activation-cache.json').read_text())
    assert meta['schemaVersion']==(2 if maximum is not None else 1)


@pytest.mark.parametrize('resume_from',['source','lookup','flat','result'])
def test_resume_restores_only_already_executed_mutations(tmp_path,registry,resume_from):
    value=graph();value['nodes'][1]['config']['max_norm']=.25
    options=dict(registry=registry,device='cpu',prepared=prepared())
    first=preview(value,tmp_path/'first',**options)
    resumed=preview(value,tmp_path/'resumed',resume_from=resume_from,cache=first['cachePath'],**options)
    assert resumed['previews']==first['previews']
    assert resumed['finalWeightsHash']==first['finalWeightsHash']


@pytest.mark.parametrize('damage',['missing','shape','dtype','nan'])
def test_forward_mutation_cache_rejects_invalid_state(tmp_path,registry,damage):
    import hashlib
    import torch
    from pathlib import Path
    value=graph();value['nodes'][1]['config']['max_norm']=1.0
    options=dict(registry=registry,device='cpu',prepared=prepared())
    first=preview(value,tmp_path/'first',stop_at='lookup',**options)
    path=Path(first['cachePath']);payload=torch.load(path,weights_only=True)
    if damage=='missing':payload['mutatedWeights'].clear()
    elif damage=='shape':payload['mutatedWeights']['lookup']=torch.zeros(1)
    elif damage=='dtype':payload['mutatedWeights']['lookup']=torch.zeros(5,3,dtype=torch.int64)
    else:payload['mutatedWeights']['lookup'].fill_(float('nan'))
    torch.save(payload,path)
    meta_path=path.with_suffix('.json');meta=json.loads(meta_path.read_text())
    meta['fileHash']=hashlib.sha256(path.read_bytes()).hexdigest();meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='forward-mutated'):
        preview(value,tmp_path/'bad',resume_from='flat',cache=path,**options)
    assert not (tmp_path/'bad').exists()
