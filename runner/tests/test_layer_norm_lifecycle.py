"""Actual training, captured state, resume and cache tests before native exposure."""
import json
import pytest
import torch
from runner.engine import GraphModel, run
from runner.preview import preview
from runner.tests.test_checkpoint_lifecycle import state, assert_nested_equal
from runner.tests.test_layer_norm_execution import graph, registry

@pytest.mark.parametrize('affine,bias',[(False,False),(True,False),(True,True)])
def test_exact_resume_capture_export_and_configuration_invalidation(tmp_path,registry,affine,bias):
    value=graph([2,3],affine,bias)
    full,first,resumed=(tmp_path/name for name in ('full','first','resumed'))
    options={'registry':registry,'device':'cpu','batch_size':64,'sample_limit':128}
    run(value,full,epochs=4,gradient_capture={'schemaVersion':1,'everyEpochs':1,'maxBatchesPerEpoch':1,'maxCaptures':4},**options)
    run(value,first,epochs=2,**options)
    parent=(first/'checkpoint.pt').read_bytes()
    run(value,resumed,epochs=4,resume_checkpoint=first/'checkpoint.pt',**options)
    for key in ('modelState','optimizerState','rngState'):
        assert_nested_equal(state(full)[key],state(resumed)[key])
    assert json.loads((full/'metrics.json').read_text())==json.loads((resumed/'metrics.json').read_text())
    assert (first/'checkpoint.pt').read_bytes()==parent
    captured=json.loads((full/'gradient-summary.json').read_text())['inventory']['parameters']
    norm=[p for p in captured if p['nodeID']=='probabilities']
    assert len(norm)==int(affine)*(1+int(bias))
    assert all(p['shape']==[2,3] and p['elementCount']==6 for p in norm)
    saved=state(full)['modelState']
    if affine:assert not torch.equal(saved['layers.probabilities.weight'],torch.ones(2,3))
    model=GraphModel(value,registry);model.load_state_dict(saved,strict=True)
    scope={};exec((full/'model.py').read_text(),scope)
    exported=scope['Model']();exported.load_state_dict(saved,strict=True)
    x=torch.tensor([[.2,.3],[-1.,2.]])
    torch.testing.assert_close(exported(x),model(x),atol=0,rtol=0)
    fresh=preview(value,tmp_path/'preview',registry=registry,device='cpu',checkpoint=full/'checkpoint.pt')
    reused=preview(value,tmp_path/'reused',registry=registry,device='cpu',checkpoint=full/'checkpoint.pt',resume_from='probabilities',cache=fresh['cachePath'])
    assert reused['previews']==fresh['previews']
    for index,change in enumerate([{'eps':1e-3},{'normalized_shape':[3]},{'elementwise_affine':not affine},{'bias':not bias}]):
        changed=graph([2,3],affine,bias);changed['nodes'][3]['config'].update(change)
        with pytest.raises(ValueError,match='semantic architecture'):
            run(changed,tmp_path/f'bad-resume-{index}',epochs=5,resume_checkpoint=first/'checkpoint.pt',**options)
        with pytest.raises(ValueError):
            preview(changed,tmp_path/f'bad-cache-{index}',registry=registry,device='cpu',checkpoint=full/'checkpoint.pt',resume_from='probabilities',cache=fresh['cachePath'])
        assert not (tmp_path/f'bad-resume-{index}'/'manifest.json').exists()
    assert (first/'checkpoint.pt').read_bytes()==parent

@pytest.mark.parametrize('device',['cpu','mps'])
def test_finetune_restores_affine_state_with_new_optimizer(tmp_path,registry,device):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
    value=graph([3],True,True);parent=tmp_path/'parent';child=tmp_path/'child'
    run(value,parent,registry=registry,device=device,epochs=2,batch_size=64,sample_limit=128)
    original=(parent/'checkpoint.pt').read_bytes();events=[]
    run(value,child,registry=registry,device=device,epochs=1,batch_size=128,sample_limit=128,seed=19,split_seed=29,learning_rate=.01,finetune_checkpoint=parent/'checkpoint.pt',emit=events.append)
    result=state(child)
    assert result['epoch']==1 and result['seed']==19 and events[0]['initialEpoch']==0
    assert all(item['step'].item()==1 for item in result['optimizerState']['state'].values())
    assert json.loads((child/'manifest.json').read_text())['lineage']['mode']=='weightsOnlyFineTune'
    assert (parent/'checkpoint.pt').read_bytes()==original
    model=GraphModel(value,registry).to(device);model.load_state_dict(result['modelState'],strict=True)
    assert torch.isfinite(model(torch.tensor([[.1,.2]],device=device))).all()
