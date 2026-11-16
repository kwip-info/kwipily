"""Private Conv1d lifecycle: real training, continuation, frozen and cached outputs."""
import hashlib
import json
import pytest
import torch
from runner.engine import GraphModel,REGISTRY,run
from runner.preview import preview
from runner.checkpoint_lifecycle import contract
from runner.frozen_checkpoint import load_frozen
from runner.tests.test_conv1d_graph import graph,registry
from runner.tests.test_checkpoint_lifecycle import state,assert_nested_equal


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('bias',[False,True])
def test_conv_exact_resume_frozen_export_and_gradient_inventory(tmp_path,registry,bias):
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
    assert state(full)['contract']['schedule']['parameterNodeOrder']==['expand','convolution','head']
    assert 'lazyInitializationOrder' not in state(full)['contract']['schedule']
    captured=json.loads((full/'gradient-summary.json').read_bytes())['inventory']['parameters']
    conv=[item for item in captured if item['nodeID']=='convolution']
    assert sorted(item['shape'] for item in conv)==([[4],[4,1,3]] if bias else [[4,1,3]])
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


def test_conv_finetune_resets_optimizer_rng_and_epoch(tmp_path,registry):
    value=graph();parent=tmp_path/'parent';child=tmp_path/'child'
    run(value,parent,registry=registry,device='cpu',epochs=2,batch_size=32,sample_limit=128)
    before=(parent/'checkpoint.pt').read_bytes();events=[]
    run(value,child,registry=registry,device='cpu',epochs=1,batch_size=128,sample_limit=128,
        seed=19,split_seed=29,learning_rate=.01,finetune_checkpoint=parent/'checkpoint.pt',emit=events.append)
    checkpoint=state(child)
    assert checkpoint['epoch']==1 and checkpoint['seed']==19 and events[0]['initialEpoch']==0
    assert all(item['step'].item()==1 for item in checkpoint['optimizerState']['state'].values())
    manifest=json.loads((child/'manifest.json').read_bytes())
    assert manifest['lineage']['mode']=='weightsOnlyFineTune'
    assert (parent/'checkpoint.pt').read_bytes()==before


def test_conv_preview_cache_and_configuration_invalidation(tmp_path,registry):
    value=graph();parent=tmp_path/'parent'
    run(value,parent,registry=registry,device='cpu',epochs=2,batch_size=32,sample_limit=128)
    first=preview(value,tmp_path/'preview',registry=registry,device='cpu',checkpoint=parent/'checkpoint.pt')
    reused=preview(value,tmp_path/'reused',registry=registry,device='cpu',checkpoint=parent/'checkpoint.pt',resume_from='convolution',cache=first['cachePath'])
    assert reused['previews']==first['previews'] and reused['reusedNodeIDs']==['input','expand','shaped']
    for index,change in enumerate([{'bias':False},{'groups':1},{'dilation':2},{'padding_policy':'same'},{'padding_mode':'replicate'}]):
        changed=graph(change)
        with pytest.raises(ValueError,match='semantic architecture'):
            run(changed,tmp_path/f'bad-resume-{index}',registry=registry,device='cpu',epochs=4,batch_size=32,sample_limit=128,resume_checkpoint=parent/'checkpoint.pt')
        with pytest.raises(ValueError):
            preview(changed,tmp_path/f'bad-cache-{index}',registry=registry,device='cpu',checkpoint=parent/'checkpoint.pt',resume_from='convolution',cache=first['cachePath'])
        assert not (tmp_path/f'bad-resume-{index}/manifest.json').exists()


@pytest.mark.parametrize('lazy',[False,True])
def test_conv_schedule_binds_all_parameter_node_order(lazy):
    value=graph();catalog=json.loads(REGISTRY.read_bytes())['components']
    for kind in ('bilinear','layer_norm','embedding')+ (('lazy_linear',) if lazy else ()):
        component=next(item for item in catalog if item['id']==kind)
        defaults={field['key']:field['defaultValue'] for field in component['fields']}
        value['nodes'].append(dict(id=kind,type=kind,config=defaults))
    order=[node['id'] for node in value['nodes']]
    def build(order):return contract(value,[torch.ones(2,2)],{'sourceHash':'a'*64,'splitHash':'b'*64},42,32,.03,'cpu',2,order)
    first=build(order);swapped=list(order);a=swapped.index('expand');b=swapped.index('convolution');swapped[a],swapped[b]=swapped[b],swapped[a]
    second=build(swapped)
    assert first['semanticGraphDigest']==second['semanticGraphDigest']
    extra=['lazy_linear'] if lazy else []
    assert first['schedule']['parameterNodeOrder']==['expand','convolution','head','bilinear','layer_norm','embedding']+extra
    assert second['schedule']['parameterNodeOrder']==['convolution','expand','head','bilinear','layer_norm','embedding']+extra
    if lazy:assert first['schedule']['lazyInitializationOrder']==['lazy_linear']
    else:assert 'lazyInitializationOrder' not in first['schedule']


def test_nonconv_checkpoint_schedule_keeps_historical_fields():
    from runner.tests.test_runner import graph as ordinary_graph
    value=ordinary_graph();order=[node['id'] for node in value['nodes']]
    result=contract(value,[torch.ones(2,2)],{'sourceHash':'a'*64,'splitHash':'b'*64},42,32,.03,'cpu',2,order)
    assert set(result['schedule'])=={'batchSize','shuffle','dropLast','numThreads','linearNodeOrder'}
    assert result['schedule']['linearNodeOrder']==['left','right','head']
