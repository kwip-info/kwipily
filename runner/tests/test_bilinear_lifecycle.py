"""Real Bilinear training/state acceptance before palette exposure."""
import json
import pytest
import torch
from runner.engine import GraphModel, run
from runner.preview import preview
from runner.tests.test_bilinear_graph import graph, registry
from runner.tests.test_checkpoint_lifecycle import state, assert_nested_equal


@pytest.mark.parametrize('bias',[True,False])
def test_training_exact_resume_export_and_cached_inference(tmp_path,registry,bias):
    value=graph(bias); full=tmp_path/'full'; first=tmp_path/'first'; resumed=tmp_path/'resumed'
    options=dict(registry=registry,device='cpu',batch_size=32,sample_limit=128)
    run(value,full,epochs=4,gradient_capture={'schemaVersion':1,'everyEpochs':1,'maxBatchesPerEpoch':1,'maxCaptures':4},**options)
    run(value,first,epochs=2,**options)
    original=(first/'checkpoint.pt').read_bytes()
    run(value,resumed,epochs=4,resume_checkpoint=first/'checkpoint.pt',**options)
    for key in ('modelState','optimizerState','rngState'):
        assert_nested_equal(state(full)[key],state(resumed)[key])
    assert json.loads((full/'metrics.json').read_text())==json.loads((resumed/'metrics.json').read_text())
    assert (first/'checkpoint.pt').read_bytes()==original
    inventory=json.loads((full/'gradient-summary.json').read_text())['inventory']['parameters']
    weight=next(p for p in inventory if p['nodeID']=='interaction' and p['shape']==[2,3,5])
    assert weight['elementCount']==30
    model=GraphModel(value,registry); model.load_state_dict(state(full)['modelState'])
    namespace={};exec((full/'model.py').read_text(),namespace)
    exported=namespace['Model']();exported.load_state_dict(state(full)['modelState'])
    x=torch.tensor([[.2,.8],[-.7,1.1]])
    torch.testing.assert_close(exported(x),model(x),atol=0,rtol=0)
    fresh=preview(value,tmp_path/'preview',registry=registry,device='cpu',checkpoint=full/'checkpoint.pt')
    reused=preview(value,tmp_path/'reused',registry=registry,device='cpu',checkpoint=full/'checkpoint.pt',resume_from='interaction',cache=fresh['cachePath'])
    assert reused['previews']==fresh['previews']
    changed=graph(not bias)
    with pytest.raises(ValueError,match='semantic architecture'):
        run(changed,tmp_path/'incompatible',epochs=5,resume_checkpoint=first/'checkpoint.pt',**options)
    run(value,tmp_path/'fine',epochs=1,finetune_checkpoint=first/'checkpoint.pt',seed=19,**options)
    assert state(tmp_path/'fine')['epoch']==1
    assert json.loads((tmp_path/'fine'/'manifest.json').read_text())['lineage']['mode']=='weightsOnlyFineTune'
    assert (first/'checkpoint.pt').read_bytes()==original
