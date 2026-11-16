"""Real Conv2d training and retained state, before public catalog exposure."""
import hashlib
import json
import pytest
import torch
from runner.engine import GraphModel,REGISTRY,run
from runner.preview import preview
from runner.checkpoint_lifecycle import contract
from runner.frozen_checkpoint import load_frozen
from runner.tests.test_conv2d_graph import graph,registry
from runner.tests.test_checkpoint_lifecycle import state,assert_nested_equal


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def convolution(value):return next(node for node in value['nodes'] if node['type']=='conv2d')


@pytest.mark.parametrize('bias',[False,True])
def test_conv2d_exact_cpu_resume_frozen_export_and_captured_parameters(tmp_path,registry,bias):
    value=graph({'bias':bias});full=tmp_path/'full';first=tmp_path/'first';resumed=tmp_path/'resumed'
    options=dict(registry=registry,device='cpu',batch_size=32,sample_limit=128)
    def consume_rng(event):
        if event['event']=='epoch':torch.rand(7)
    run(value,full,epochs=4,emit=consume_rng,gradient_capture={'schemaVersion':1,'everyEpochs':1,'maxBatchesPerEpoch':1,'maxCaptures':4},**options)
    run(value,first,epochs=2,emit=consume_rng,**options)
    parent={file.name:file.read_bytes() for file in first.iterdir() if file.is_file()}
    run(value,resumed,epochs=4,resume_checkpoint=first/'checkpoint.pt',emit=consume_rng,**options)
    for key in ('modelState','optimizerState','rngState'):assert_nested_equal(state(full)[key],state(resumed)[key])
    assert json.loads((full/'metrics.json').read_bytes())==json.loads((resumed/'metrics.json').read_bytes())
    expected_order=[node['id'] for node in value['nodes'] if node['type'] in ('linear','conv2d')]
    assert state(full)['contract']['schedule']['parameterNodeOrder']==expected_order
    assert 'lazyInitializationOrder' not in state(full)['contract']['schedule']
    node=convolution(value);config=node['config']
    captured=json.loads((full/'gradient-summary.json').read_bytes())['inventory']['parameters']
    shapes=[item['shape'] for item in captured if item['nodeID']==node['id']]
    expected=[[config['out_channels'],config['in_channels']//config['groups'],*config['kernel_size']]]
    if bias:expected.append([config['out_channels']])
    assert sorted(shapes)==sorted(expected)
    frozen=load_frozen(full/'checkpoint.pt',expected_manifest_sha256=sha(full/'manifest.json'),
        expected_checkpoint_sha256=sha(full/'checkpoint.pt'),registry_path=registry,expected_registry_sha256=sha(registry))
    model=GraphModel(value,registry);model.load_state_dict(state(full)['modelState'],strict=True)
    namespace={};exec((full/'model.py').read_text(),namespace)
    exported=namespace['Model']();exported.load_state_dict(state(full)['modelState'],strict=True)
    inputs=torch.tensor([[.2,.3],[-1.,2.]])
    torch.testing.assert_close(exported(inputs),model(inputs),atol=0,rtol=0)
    torch.testing.assert_close(frozen.model(inputs),model(inputs),atol=0,rtol=0)
    assert not frozen.model.training and all(not parameter.requires_grad for parameter in frozen.model.parameters())
    assert parent=={file.name:file.read_bytes() for file in first.iterdir() if file.is_file()}


def test_conv2d_finetune_resets_optimizer_and_epoch(tmp_path,registry):
    value=graph();parent=tmp_path/'parent';child=tmp_path/'child'
    run(value,parent,registry=registry,device='cpu',epochs=2,batch_size=32,sample_limit=128)
    before=(parent/'checkpoint.pt').read_bytes();events=[]
    run(value,child,registry=registry,device='cpu',epochs=1,batch_size=128,sample_limit=128,
        seed=19,split_seed=29,learning_rate=.01,finetune_checkpoint=parent/'checkpoint.pt',emit=events.append)
    checkpoint=state(child)
    assert checkpoint['epoch']==1 and checkpoint['seed']==19 and events[0]['initialEpoch']==0
    assert all(item['step'].item()==1 for item in checkpoint['optimizerState']['state'].values())
    assert json.loads((child/'manifest.json').read_bytes())['lineage']['mode']=='weightsOnlyFineTune'
    assert (parent/'checkpoint.pt').read_bytes()==before


def test_conv2d_initialized_and_trained_cache_reuse_and_configuration_invalidation(tmp_path,registry):
    value=graph();node=convolution(value);parent=tmp_path/'parent'
    initialized=preview(value,tmp_path/'initialized',registry=registry,device='cpu')
    initial_reuse=preview(value,tmp_path/'initial-reuse',registry=registry,device='cpu',resume_from=node['id'],cache=initialized['cachePath'])
    assert initial_reuse['previews']==initialized['previews']
    run(value,parent,registry=registry,device='cpu',epochs=2,batch_size=32,sample_limit=128)
    first=preview(value,tmp_path/'trained',registry=registry,device='cpu',checkpoint=parent/'checkpoint.pt')
    reused=preview(value,tmp_path/'reused',registry=registry,device='cpu',checkpoint=parent/'checkpoint.pt',resume_from=node['id'],cache=first['cachePath'])
    assert reused['previews']==first['previews']
    with pytest.raises(ValueError,match='stale'):
        preview(value,tmp_path/'wrong-weights',registry=registry,device='cpu',checkpoint=parent/'checkpoint.pt',resume_from=node['id'],cache=initialized['cachePath'])
    changes=[{'bias':False},{'groups':1},{'dilation':[2,1]},{'padding_policy':'same'},{'padding_mode':'replicate'}]
    for index,change in enumerate(changes):
        changed=graph(change)
        with pytest.raises(ValueError,match='semantic architecture'):
            run(changed,tmp_path/f'bad-resume-{index}',registry=registry,device='cpu',epochs=4,batch_size=32,sample_limit=128,resume_checkpoint=parent/'checkpoint.pt')
        with pytest.raises(ValueError):
            preview(changed,tmp_path/f'bad-cache-{index}',registry=registry,device='cpu',checkpoint=parent/'checkpoint.pt',resume_from=node['id'],cache=first['cachePath'])
        assert not (tmp_path/f'bad-resume-{index}/manifest.json').exists()


@pytest.mark.parametrize('mixed',[False,True])
def test_conv2d_schedule_binds_optimizer_order_including_existing_parameter_families(mixed):
    value=graph();catalog=json.loads(REGISTRY.read_bytes())['components']
    extras=('conv1d','lazy_linear','bilinear','layer_norm','embedding') if mixed else ()
    for kind in extras:
        definition=next(item for item in catalog if item['id']==kind)
        value['nodes'].append(dict(id='extra-'+kind,type=kind,config={field['key']:field['defaultValue'] for field in definition['fields']}))
    order=[node['id'] for node in value['nodes']]
    build=lambda ordering:contract(value,[torch.ones(2,2)],{'sourceHash':'a'*64,'splitHash':'b'*64},42,32,.03,'cpu',2,ordering)
    first=build(order);changed=order.copy();ident=convolution(value)['id'];linear=next(node['id'] for node in value['nodes'] if node['type']=='linear')
    a,b=changed.index(ident),changed.index(linear);changed[a],changed[b]=changed[b],changed[a]
    second=build(changed)
    assert first['semanticGraphDigest']==second['semanticGraphDigest']
    parameter_ids={node['id'] for node in value['nodes'] if node['type'] in ('linear','conv2d',*extras)}
    assert first['schedule']['parameterNodeOrder']==[key for key in order if key in parameter_ids]
    assert second['schedule']['parameterNodeOrder']==[key for key in changed if key in parameter_ids]
    if mixed:assert first['schedule']['lazyInitializationOrder']==['extra-lazy_linear']
    else:assert 'lazyInitializationOrder' not in first['schedule']


@pytest.mark.parametrize('prior',['ordinary','conv1d'])
def test_previous_nonconv_and_conv1d_checkpoint_schedules_are_unchanged(prior):
    from runner.tests.test_runner import graph as ordinary
    from runner.tests.test_conv1d_graph import graph as one_dimensional
    value=(ordinary if prior=='ordinary' else one_dimensional)();order=[node['id'] for node in value['nodes']]
    result=contract(value,[torch.ones(2,2)],{'sourceHash':'a'*64,'splitHash':'b'*64},42,32,.03,'cpu',2,order)
    expected={'batchSize':32,'shuffle':False,'dropLast':False,'numThreads':torch.get_num_threads(),
        'linearNodeOrder':[node['id'] for node in value['nodes'] if node['type']=='linear']}
    if prior=='conv1d':expected['parameterNodeOrder']=['expand','convolution','head']
    assert result['schedule']==expected


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='MPS unavailable')
def test_conv2d_mps_training_and_initialized_trained_preview(tmp_path,registry):
    value=graph({'bias':False})
    initialized=preview(value,tmp_path/'initialized-mps',registry=registry,device='mps')
    run(value,tmp_path/'trained-mps',epochs=2,registry=registry,device='mps',batch_size=32,sample_limit=128)
    trained=preview(value,tmp_path/'trained-preview-mps',registry=registry,device='mps',checkpoint=tmp_path/'trained-mps/checkpoint.pt')
    assert initialized['device']==trained['device']=='mps'
    assert initialized['weights']=='initialized' and trained['weights']=='checkpoint'
    assert torch.isfinite(torch.tensor(trained['previews']['output']['values'])).all()
    assert state(tmp_path/'trained-mps')['contract']['guarantee']=='weightsOnly'
