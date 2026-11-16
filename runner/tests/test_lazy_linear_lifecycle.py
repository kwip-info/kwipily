"""Real LazyLinear lifecycle acceptance using an unexposed private registry."""
import copy
import json
import pytest
import torch
from runner.engine import GraphModel, run
from runner.preview import preview
from runner.checkpoint_lifecycle import contract
from runner.frozen_checkpoint import load_frozen
from runner.tests.test_lazy_linear_graph import graph, registry, make_graph
from runner.tests.test_checkpoint_lifecycle import state, assert_nested_equal
from runner.tests.test_frozen_checkpoint import sha, rewrite


@pytest.mark.parametrize('bias', [False, True])
def test_training_resume_capture_export_finetune_and_frozen_load(tmp_path, registry, bias):
    authored = graph(bias); original = copy.deepcopy(authored)
    full,first,resumed = (tmp_path/name for name in ('full','first','resumed'))
    options = dict(registry=registry,device='cpu',batch_size=32,sample_limit=128)
    run(authored,full,epochs=4,gradient_capture={'schemaVersion':1,'everyEpochs':1,'maxBatchesPerEpoch':1,'maxCaptures':4},**options)
    run(authored,first,epochs=2,**options)
    parent = {p.name:p.read_bytes() for p in first.iterdir() if p.is_file()}
    run(authored,resumed,epochs=4,resume_checkpoint=first/'checkpoint.pt',**options)
    for key in ('modelState','optimizerState','rngState'):
        assert_nested_equal(state(full)[key],state(resumed)[key])
    assert json.loads((full/'metrics.json').read_text()) == json.loads((resumed/'metrics.json').read_text())
    inventory = json.loads((full/'gradient-summary.json').read_text())['inventory']['parameters']
    owned = [p for p in inventory if p['nodeID']=='lazy']
    assert sorted(p['shape'] for p in owned) == ([[2],[2,2]] if bias else [[2,2]])
    namespace = {}; source = (full/'model.py').read_text(); exec(source,namespace)
    assert 'nn.LazyLinear(' in source
    exported = namespace['Model'](); exported.load_state_dict(state(full)['modelState'],strict=True)
    model = GraphModel(authored,registry); model.materialize_lazy_parameters()
    model.load_state_dict(state(full)['modelState'],strict=True)
    x = torch.tensor([[.2,.8],[-.7,1.1]])
    torch.testing.assert_close(exported(x),model(x),atol=0,rtol=0)
    rng = torch.get_rng_state().clone()
    frozen = load_frozen(full/'checkpoint.pt',expected_manifest_sha256=sha(full/'manifest.json'),
        expected_checkpoint_sha256=sha(full/'checkpoint.pt'),registry_path=registry,expected_registry_sha256=sha(registry))
    assert torch.equal(torch.get_rng_state(),rng)
    assert not frozen.model.training and all(not p.requires_grad for p in frozen.model.parameters())
    torch.testing.assert_close(frozen.model(x),model(x),atol=0,rtol=0)
    fine = tmp_path/'fine'
    run(authored,fine,epochs=1,finetune_checkpoint=first/'checkpoint.pt',seed=19,batch_size=1024,
        registry=registry,device='cpu',sample_limit=128)
    assert state(fine)['epoch']==1 and state(fine)['seed']==19
    assert all(value['step'].item()==1 for value in state(fine)['optimizerState']['state'].values())
    assert json.loads((fine/'manifest.json').read_text())['lineage']['mode']=='weightsOnlyFineTune'
    assert parent == {p.name:p.read_bytes() for p in first.iterdir() if p.is_file()}
    assert authored == original


@pytest.mark.parametrize('bias', [False, True])
def test_initialized_checkpoint_and_reused_previews(tmp_path, registry, bias):
    authored = graph(bias)
    initialized = preview(authored,tmp_path/'initialized',registry=registry,device='cpu')
    assert initialized['weights']=='initialized'
    again = preview(authored,tmp_path/'again',registry=registry,device='cpu')
    assert initialized['previews']==again['previews']
    initialized_reuse = preview(authored,tmp_path/'initial-reuse',registry=registry,device='cpu',
        resume_from='lazy',cache=initialized['cachePath'])
    assert initialized_reuse['previews']==initialized['previews']
    assert initialized_reuse['reusedNodeIDs']==['input']
    run(authored,tmp_path/'trained',epochs=2,registry=registry,device='cpu',sample_limit=128)
    checkpoint = tmp_path/'trained/checkpoint.pt'
    fresh = preview(authored,tmp_path/'fresh',registry=registry,device='cpu',checkpoint=checkpoint)
    reused = preview(authored,tmp_path/'reused',registry=registry,device='cpu',checkpoint=checkpoint,
        resume_from='lazy',cache=fresh['cachePath'])
    assert fresh['weights']=='checkpoint' and reused['previews']==fresh['previews']
    assert reused['reusedNodeIDs']==['input']
    with pytest.raises(ValueError,match='stale'):
        preview(authored,tmp_path/'wrong-weights',registry=registry,device='cpu',checkpoint=checkpoint,
            resume_from='lazy',cache=initialized['cachePath'])


def test_changed_upstream_inferred_width_rejects_continuation(tmp_path, registry):
    authored = make_graph([('input','input',{'features':2}),
        ('expand','linear',{'in_features':2,'out_features':3,'bias':True}),
        ('lazy','lazy_linear',{'out_features':2,'bias':True}),('output','output',{})])
    run(authored,tmp_path/'parent',epochs=2,registry=registry,device='cpu',sample_limit=128)
    changed = copy.deepcopy(authored); changed['nodes'][1]['config']['out_features']=4
    with pytest.raises(ValueError,match='architecture'):
        run(changed,tmp_path/'bad',epochs=4,registry=registry,device='cpu',sample_limit=128,
            resume_checkpoint=tmp_path/'parent/checkpoint.pt')
    assert not (tmp_path/'bad/manifest.json').exists()


def test_initialization_and_parameter_order_bound_only_for_lazy_graphs():
    authored = graph()
    authored['nodes'].insert(2,dict(id='other',type='lazy_linear',config={'out_features':2,'bias':True}))
    authored['nodes'].insert(3,dict(id='merge',type='add',config={}))
    authored['edges']=[dict(id=str(i),source=a,sourcePort='out',target=b,targetPort=p)
        for i,(a,b,p) in enumerate([('input','lazy','in'),('input','other','in'),('lazy','merge','a'),('other','merge','b'),('merge','output','in')])]
    def build(value, order):
        return contract(value,[torch.ones(2,2)],{'sourceHash':'a'*64,'splitHash':'b'*64},42,32,.03,'cpu',1,order)
    first = build(authored,['input','lazy','other','merge','output'])
    second = build(authored,['input','other','lazy','merge','output'])
    assert first['semanticGraphDigest']==second['semanticGraphDigest']
    assert first['schedule']['lazyInitializationOrder']==['lazy','other']
    assert second['schedule']['lazyInitializationOrder']==['other','lazy']
    assert first['schedule']['parameterNodeOrder']==['lazy','other']
    assert second['schedule']['parameterNodeOrder']==['other','lazy']
    ordinary = make_graph([('input','input',{'features':2}),('linear','linear',{'in_features':2,'out_features':2,'bias':True}),('output','output',{})])
    legacy = build(ordinary,['input','linear','output'])
    assert set(legacy['schedule'])=={'batchSize','shuffle','dropLast','numThreads','linearNodeOrder'}
    assert legacy['schedule']['linearNodeOrder']==['linear']


def test_preview_rejects_incompatible_lazy_shape_before_loading_state(tmp_path, registry, monkeypatch):
    authored = graph(); saved = tmp_path/'parent'
    run(authored,saved,epochs=1,registry=registry,device='cpu',sample_limit=128)
    def mutate(manifest,checkpoint):
        key = next(k for k in checkpoint['modelState'] if k.endswith('.weight'))
        checkpoint['modelState'][key] = torch.ones(2,3)
    rewrite(saved,mutate)
    def forbidden(*args,**kwargs): raise AssertionError('Malformed weight reached lazy load_state_dict')
    monkeypatch.setattr(torch.nn.Module,'load_state_dict',forbidden)
    with pytest.raises(ValueError,match='(?i)(shape|incompatible|parameter)'):
        preview(authored,tmp_path/'bad-preview',registry=registry,device='cpu',checkpoint=saved/'checkpoint.pt')

@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='Apple GPU unavailable')
def test_mps_training_and_initialized_checkpoint_previews(tmp_path, registry):
    authored = graph(False)
    initialized = preview(authored,tmp_path/'initialized-mps',registry=registry,device='mps')
    run(authored,tmp_path/'trained-mps',epochs=2,registry=registry,device='mps',sample_limit=128,batch_size=32)
    trained = preview(authored,tmp_path/'trained-preview-mps',registry=registry,device='mps',checkpoint=tmp_path/'trained-mps/checkpoint.pt')
    assert initialized['device']==trained['device']=='mps'
    assert initialized['weights']=='initialized' and trained['weights']=='checkpoint'
    assert torch.isfinite(torch.tensor(trained['previews']['output']['values'])).all()
    assert json.loads((tmp_path/'trained-mps/manifest.json').read_text())['checkpoint']['contract']['guarantee']=='weightsOnly'
