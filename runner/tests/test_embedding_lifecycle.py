import json
import pytest
import torch
from runner.engine import GraphModel,run
from runner.product_execution import ProductExecutionAdapter
from runner.tests.test_embedding_graph import graph,registry
from runner.tests.test_checkpoint_lifecycle import state,assert_nested_equal


def prepared():
    x=torch.tensor([[1,2],[2,4],[0,1],[4,3],[2,1],[3,4],[1,0],[0,4]])
    y=torch.tensor([0,1,0,1,0,1,0,1])
    return ((x,y,x[:4],y[:4]),{'sourceHash':'embedding-fixture','splitHash':'embedding-split'}, {})


def test_adapter_preserves_index_input_and_captured_dtype(registry):
    model=GraphModel(graph(),registry);adapter=ProductExecutionAdapter(model)
    x=prepared()[0][0]
    output,stages=adapter.forward(x,inspect=True)
    assert output.shape==(8,2) and stages['source'].dtype==torch.int64
    with pytest.raises(ValueError,match='int64'):adapter.forward(x.float())


@pytest.mark.parametrize('changes',[{}, {'max_norm':1.0}, {'freeze':True}])
def test_dense_exact_resume_and_portable_trained_weights(tmp_path,registry,changes):
    value=graph();value['nodes'][1]['config'].update(changes);options=dict(registry=registry,device='cpu',batch_size=3,prepared=prepared())
    full,first,resumed=[tmp_path/name for name in ('full','first','resumed')]
    run(value,full,epochs=4,**options)
    run(value,first,epochs=2,**options)
    parent=(first/'checkpoint.pt').read_bytes()
    run(value,resumed,epochs=4,resume_checkpoint=first/'checkpoint.pt',**options)
    for key in ('modelState','optimizerState','rngState'):assert_nested_equal(state(full)[key],state(resumed)[key])
    assert json.loads((full/'metrics.json').read_text())==json.loads((resumed/'metrics.json').read_text())
    assert (first/'checkpoint.pt').read_bytes()==parent
    model=GraphModel(value,registry);model.load_state_dict(state(full)['modelState'],strict=True)
    scope={};exec((full/'model.py').read_text(),scope)
    exported=scope['Model']();exported.load_state_dict(state(full)['modelState'],strict=True)
    torch.testing.assert_close(exported(prepared()[0][0]),model(prepared()[0][0]),atol=0,rtol=0)


def test_sparse_adam_rejected_before_creating_artifacts(tmp_path,registry):
    value=graph();value['nodes'][1]['config']['sparse']=True
    with pytest.raises(ValueError,match='Adam.*sparse|sparse.*Adam'):
        run(value,tmp_path/'invalid',epochs=1,registry=registry,device='cpu',prepared=prepared())
    assert not (tmp_path/'invalid').exists()


@pytest.mark.parametrize('device',['cpu','mps'])
def test_embedding_finetune_uses_fresh_optimizer(tmp_path,registry,device):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
    value=graph();parent=tmp_path/'parent';child=tmp_path/'child'
    options=dict(registry=registry,device=device,prepared=prepared())
    run(value,parent,epochs=2,batch_size=3,**options)
    original=(parent/'checkpoint.pt').read_bytes()
    run(value,child,epochs=1,batch_size=8,seed=19,learning_rate=.01,finetune_checkpoint=parent/'checkpoint.pt',**options)
    assert state(child)['epoch']==1
    assert all(item['step'].item()==1 for item in state(child)['optimizerState']['state'].values())
    assert json.loads((child/'manifest.json').read_text())['lineage']['mode']=='weightsOnlyFineTune'
    assert (parent/'checkpoint.pt').read_bytes()==original


@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('changes',[{}, {'max_norm':.5}, {'freeze':True,'max_norm':.5}, {'scale_grad_by_freq':True,'padding_idx':0}])
def test_gradient_capture_does_not_change_lookup_training(tmp_path,registry,device,changes):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
    value=graph();value['nodes'][1]['config'].update(changes)
    options=dict(registry=registry,device=device,batch_size=3,prepared=prepared(),epochs=2)
    run(value,tmp_path/'off',**options)
    events=[]
    run(value,tmp_path/'on',gradient_capture={'schemaVersion':1,'everyEpochs':1,'maxBatchesPerEpoch':2,'maxCaptures':8},emit=events.append,**options)
    for key in ('modelState','optimizerState','rngState'):
        assert_nested_equal(state(tmp_path/'off')[key],state(tmp_path/'on')[key])
    assert (tmp_path/'off/metrics.json').read_bytes()==(tmp_path/'on/metrics.json').read_bytes()
    report=json.loads((tmp_path/'on/gradient-summary.json').read_text())
    names=[p['name'] for p in report['inventory']['parameters']]
    assert any('lookup' in name for name in names) == (not changes.get('freeze',False))
    assert len(report['summaries'])==2
    for summary in report['summaries']:
        assert summary['observedBatchCount']==2
        for batch in summary['batches']:
            assert batch['missingParameterCount']==0
            assert batch['nonfiniteElementCount']==0
            assert batch['presentElementCount']==report['inventory']['elementCount']
            assert batch['l2Norm'] is not None
    assert len([e for e in events if e['event']=='gradientSummary'])==2

@pytest.mark.parametrize('frequency',[False,True])
def test_embedding_capture_matches_independent_scatter_summary(frequency):
    import math
    from runner.embedding import CheckedEmbedding
    from runner.gradient_capture import GradientCapture
    from runner.tests.test_embedding_contract import config
    layer=CheckedEmbedding(config(padding_idx=0,scale_grad_by_freq=frequency))
    indices=torch.tensor([[0,2,2],[1,2,4]])
    upstream=torch.arange(18,dtype=torch.float32).reshape(2,3,3)-4
    (layer(indices)*upstream).sum().backward()
    expected=torch.zeros(5,3,dtype=torch.float64)
    counts={i:indices.flatten().tolist().count(i) for i in range(5)}
    for i,g in zip(indices.flatten().tolist(),upstream.reshape(-1,3).double()):
        if i!=0:expected[i]+=g/(counts[i] if frequency else 1)
    capture=GradientCapture(layer,{'schemaVersion':1,'everyEpochs':1,'maxBatchesPerEpoch':1,'maxCaptures':1},run_id='analytical',initial_epoch=0,final_epoch=1,device='cpu')
    capture.begin_epoch(1,1,2);capture.observe(0,2)
    batch=capture.end_epoch()['batches'][0]
    assert batch['presentElementCount']==15
    assert batch['zeroElementCount']==int((expected==0).sum())
    assert math.isclose(batch['l2Norm'],float(torch.linalg.vector_norm(expected)),rel_tol=1e-6)
    assert math.isclose(batch['maxAbs'],float(expected.abs().max()),rel_tol=1e-6)
    assert batch['missingElementCount']==0 and batch['nonfiniteElementCount']==0


def test_multidimensional_predictions_keep_integer_coordinates_without_scatter_projection(tmp_path,registry):
    value=graph();value['nodes'][0]['config']['shape']=[1,2]
    data,source,extra=prepared()
    data=(data[0].reshape(-1,1,2),data[1],data[2].reshape(-1,1,2),data[3])
    events=[]
    run(value,tmp_path/'run',epochs=1,registry=registry,device='cpu',prepared=(data,source,extra),emit=events.append)
    completed=events[-1]
    assert completed['event']=='completed'
    assert completed['predictions']==[]
    assert completed['predictionInput']=={'dtype':'int64','sampleShape':[1,2]}
    assert [r['input'] for r in completed['predictionRows']]==data[2].tolist()
    assert [r['label'] for r in completed['predictionRows']]==data[3].tolist()
    assert all(type(v) is int for row in completed['predictionRows'] for v in row['input'][0])


def test_prediction_artifact_retains_typed_inputs_and_run_identity(tmp_path, registry):
    import hashlib
    result = run(graph(), tmp_path/'run', epochs=1, registry=registry, device='cpu', prepared=prepared())
    manifest = json.loads((tmp_path/'run/manifest.json').read_text())
    metadata = manifest['predictionPreview']
    raw = (tmp_path/'run/predictions.json').read_bytes()
    assert metadata == dict(schemaVersion=1, file='predictions.json', bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    artifact = json.loads(raw)
    assert artifact['schemaVersion'] == 1 and artifact['runID'] == result['runID']
    for field in ('predictionInput','predictionRows','evaluationCount'):
        assert artifact[field] == result[field]
    assert artifact['selection'] == 'firstRowsOfValidationPartition'
    assert all(type(v) is int for row in artifact['predictionRows'] for v in row['input'])
    assert 'outputDir' not in artifact
