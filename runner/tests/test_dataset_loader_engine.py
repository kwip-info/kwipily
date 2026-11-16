"""Real prepared training with selected native Dataset/DataLoader policies."""
import copy
import json
import pytest
import torch
from runner import engine
from runner.tests.prepared_objective_fixtures import prepared_window_fixture as loader_prepared
from runner.tests.test_gradient_engine import optimizer,policy,saved,oracle
from runner.tests.test_dataset_loader_configuration import recipe
from runner.tests.test_checkpoint_lifecycle import assert_nested_equal

@pytest.fixture(autouse=True)
def preserve_policy():
    old=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled()
    yield
    torch.use_deterministic_algorithms(old,warn_only=warn)



def setup(tmp):
    graph,r,m,data=loader_prepared(tmp/'source','loss_mse')
    return graph,dict(seed=41,device='cpu',prepared=data,objective_recipe=r,target_mapping=m,optimizer_recipe=optimizer())

@pytest.mark.parametrize('device',['cpu','mps'])
def test_selected_original_subset_actual_affine_updates_and_window_trace(tmp_path,device):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
    graph,kwargs=setup(tmp_path);kwargs['device']=device
    torch.manual_seed(41);initial=engine.GraphModel(graph,bundled_dataset=False,objective_context={'recipe':kwargs['objective_recipe'],'targetMapping':kwargs['target_mapping']}).state_dict()
    trace=[];positions=[4,1,0,7,3]
    engine.run(graph,tmp_path/'run',epochs=2,batch_size=2,gradient_policy=policy(),gradient_observer=trace.append,dataset_recipe=recipe(kind='subset',indices=positions),loader_recipe=recipe('data_loader',batch_size=2),**kwargs)
    x=kwargs['prepared'][0][0][positions].tolist();y=kwargs['prepared'][0][1]['target'][positions].tolist()
    expected=oracle(initial,x,y,2,2,2);state=saved(tmp_path/'run')
    torch.testing.assert_close(state['modelState']['layers.learn.weight'],torch.tensor(expected[0]),atol=3e-6,rtol=3e-5)
    torch.testing.assert_close(state['modelState']['layers.learn.bias'],torch.tensor(expected[1]),atol=3e-6,rtol=3e-5)
    assert [v['rowIndices'] for v in trace if v['phase']=='windowStart']==[[4,1,0,7],[3]]*2
    manifest=json.loads((tmp_path/'run/manifest.json').read_text())
    assert manifest['settings']['batchesPerEpoch']==3
    assert manifest['dataLoading']['selectedRows']==5 and manifest['dataLoading']['retainedRows']==5
    assert state['dataLoadingState']==dict(schemaVersion=1,completedEpochs=2,nextEpoch=2,pendingBatches=0)


def test_drop_last_counts_and_actual_optimizer_updates(tmp_path):
    graph,kwargs=setup(tmp_path)
    engine.run(graph,tmp_path/'run',epochs=1,batch_size=5,loader_recipe=recipe('data_loader',batch_size=5,drop_last=True),**kwargs)
    manifest=json.loads((tmp_path/'run/manifest.json').read_text());state=saved(tmp_path/'run')
    assert manifest['settings']['batchesPerEpoch']==2 and state['optimizerUpdates']==[2,2]
    assert manifest['dataLoading']['retainedRows']==10 and manifest['dataLoading']['droppedRows']==2

@pytest.mark.parametrize('bad',['batchMismatch','dropAll','subsetForeign'])
def test_selected_admission_before_model_state_mutation(tmp_path,bad,monkeypatch):
    graph,kwargs=setup(tmp_path);options=dict(loader_recipe=recipe('data_loader',batch_size=2))
    if bad=='dropAll':options['loader_recipe']=recipe('data_loader',batch_size=20,drop_last=True)
    elif bad=='subsetForeign':options['dataset_recipe']=recipe(kind='subset',indices=[99999])
    # A model constructor sentinel establishes refusal happened before allocation.
    monkeypatch.setattr(engine,'GraphModel',lambda *a,**k:pytest.fail('model allocated before selected loader admission'))
    with pytest.raises(ValueError):engine.run(graph,tmp_path/'run',epochs=1,batch_size=3 if bad=='batchMismatch' else options['loader_recipe']['config']['batch_size'],**options,**kwargs)


def test_dataset_only_large_legacy_batch_is_not_clamped_or_forged_selected_loader(tmp_path):
    graph,kwargs=setup(tmp_path)
    engine.run(graph,tmp_path/'run',epochs=1,batch_size=3000,dataset_recipe=recipe(kind='boundedIterable',max_rows=5),**kwargs)
    manifest=json.loads((tmp_path/'run/manifest.json').read_text());state=saved(tmp_path/'run')
    assert manifest['settings']['requestedBatchSize']==3000
    assert manifest['settings']['batchesPerEpoch']==1 and manifest['dataLoading']['selectedRows']==5
    assert state['contract']['dataLoading']['loaderRecipe'] is None
    assert 'loaderRecipe' not in manifest['settings']


def test_actual_cpu_epoch_resume_with_shuffle_workers_scheduler_and_changed_recipe_refusal(tmp_path):
    from runner.scheduler_configuration import canonical_configuration
    graph,kwargs=setup(tmp_path)
    scheduler=dict(schemaVersion=1,configurationVersion=1,componentID='scheduler_step_lr',config=canonical_configuration('scheduler_step_lr',dict(step_size=1,gamma=.5)))
    kwargs.update(batch_size=5,dataset_recipe=recipe(kind='subset',indices=[7,4,1,0,9,2,6]),loader_recipe=recipe('data_loader',batch_size=5,shuffle=True,num_workers=1,prefetch_factor=1,timeout=10),gradient_policy=policy(),scheduler_recipe=scheduler)
    engine.run(graph,tmp_path/'full',epochs=4,**kwargs);engine.run(graph,tmp_path/'parent',epochs=2,**kwargs)
    parent=(tmp_path/'parent/checkpoint.pt').read_bytes()
    engine.run(graph,tmp_path/'resumed',epochs=4,resume_checkpoint=tmp_path/'parent/checkpoint.pt',**kwargs)
    a=saved(tmp_path/'full');b=saved(tmp_path/'resumed')
    assert_nested_equal(a,b)
    for file in ('metrics.json','rate-trace.json'):assert (tmp_path/'full'/file).read_bytes()==(tmp_path/'resumed'/file).read_bytes()
    assert parent==(tmp_path/'parent/checkpoint.pt').read_bytes()
    changed=copy.deepcopy(kwargs);changed['loader_recipe']['config']['seed']+=1
    with pytest.raises(ValueError):engine.run(graph,tmp_path/'wrong',epochs=4,resume_checkpoint=tmp_path/'parent/checkpoint.pt',**changed)


def varied_preparation(tmp,op,config=None):
    """Genuine re-preparation of a sealed CSV with varied typed target rows."""
    import hashlib
    from runner import dataflow
    from runner.dataflow_cli import run_dataflow
    from runner.loss_prepared_data import load_prepared_objective
    from runner.tests.test_loss_prepared_data import CASES
    graph,r,m,_=loader_prepared(tmp,op,config)
    path=tmp/'source.csv';lines=path.read_text().splitlines();columns=lines[0].split(',')
    rows=[row.split(',') for row in lines[1:]]
    for i,row in enumerate(rows):
        if op in ('loss_cross_entropy','loss_nll'):row[columns.index('target0')]=str([0,1,2,-100][i%4])
        if op=='loss_ctc':
            row[columns.index('target_lengths0')]=str(1 if i%2 else 2)
            row[columns.index('target1')]=str(-999 if i%2 else 2)
            row[columns.index('input_lengths0')]=str(2 if i%2 else 3)
    path.write_text(lines[0]+'\n'+'\n'.join(','.join(row) for row in rows)+'\n')
    flow=json.loads((tmp/'dataflow.json').read_text());out=tmp/'varied-preparation'
    run_dataflow(tmp/'dataflow.json',{'s':str(path)},out,lambda _:None)
    data=load_prepared_objective(out/'manifest.json','p','training','p','validation',m,r,expected_manifest_sha256=hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest(),expected_graph_digest=dataflow.digest(flow),sample_shape=[len(m['inputColumns'])],prediction_shape=CASES[op][0])
    return graph,r,m,data


def independent_native_objective(pred,roles,op,c,scalarization):
    """Native functional oracle, independently configured; no runner objective."""
    F=torch.nn.functional;reduction=c['reduction'];target=roles.get('target')
    if op=='loss_cross_entropy':raw=F.cross_entropy(pred,target,weight=torch.tensor(c['class_weights'],device=pred.device) if c['class_weights'] else None,ignore_index=c['ignore_index'],label_smoothing=c['label_smoothing'],reduction=reduction)
    elif op=='loss_nll':raw=F.nll_loss(pred,target,weight=torch.tensor(c['class_weights'],device=pred.device) if c['class_weights'] else None,ignore_index=c['ignore_index'],reduction=reduction)
    elif op=='loss_mse':raw=F.mse_loss(pred,target,reduction=reduction)
    elif op=='loss_l1':raw=F.l1_loss(pred,target,reduction=reduction)
    elif op=='loss_bce_with_logits':raw=F.binary_cross_entropy_with_logits(pred,target,weight=roles['weight'],reduction=reduction)
    elif op=='loss_kl_div':raw=F.kl_div(pred,target,reduction=reduction,log_target=c['log_target'])
    elif op=='loss_ctc':
        packed=torch.cat([t[:int(n)] for t,n in zip(target,roles['target_lengths'])])
        raw=F.ctc_loss(pred.transpose(0,1),packed,roles['input_lengths'],roles['target_lengths'],blank=c['blank'],zero_infinity=c['zero_infinity'],reduction=reduction)
    else:
        a,p,n=pred.unbind(1);raw=F.triplet_margin_loss(a,p,n,margin=c['margin'],p=c['p'],eps=c['eps'],swap=c['swap'],reduction=reduction)
    return raw.mean() if reduction=='none' and scalarization=='mean' else raw.sum() if reduction=='none' else raw

@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('op',['loss_cross_entropy','loss_nll','loss_bce_with_logits','loss_mse','loss_l1','loss_kl_div','loss_ctc','loss_triplet_margin'])
@pytest.mark.parametrize('reduction,scale',[('mean','mean'),('sum','sum'),('none','mean'),('none','sum')])
def test_all_eight_selected_shuffled_windows_match_independent_native_full_window(tmp_path,device,op,reduction,scale):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
    from runner.batch_loading import plan_epoch
    from runner.dataset_sources import make_source
    from runner.tests.test_loss_prepared_data import CASES
    c={'reduction':reduction}
    if op in ('loss_cross_entropy','loss_nll'):c['class_weights']=[.5,1.2,2.]
    graph,r,m,data=varied_preparation(tmp_path/'source',op,c)
    positions=[9,4,1,0,8,2,6];dr=recipe(kind='subset',indices=positions);lr=recipe('data_loader',batch_size=3,shuffle=True)
    torch.manual_seed(41);initial=engine.GraphModel(graph,bundled_dataset=False,objective_context={'recipe':r,'targetMapping':m}).state_dict()
    w=initial['layers.learn.weight'].to(device).clone().requires_grad_();b=initial['layers.learn.bias'].to(device).clone().requires_grad_()
    native_optimizer=torch.optim.SGD([w,b],lr=.02,momentum=.5)
    source=make_source(data[0][0],data[0][1],data[1]['trainingRowIDs'],dr)
    for epoch in range(2):
        plans=plan_epoch(source,lr,epoch)
        for start in range(0,len(plans),2):
            chosen=[i for p in plans[start:start+2] for i in p.row_indices]
            x=data[0][0][chosen].to(device);roles={k:v[chosen].to(device) for k,v in data[0][1].items()}
            if op=='loss_triplet_margin':x=x.reshape(-1,3,2)
            prediction=torch.nn.functional.linear(x,w,b)
            if op!='loss_triplet_margin':prediction=prediction.reshape(len(chosen),*CASES[op][0])
            if op in ('loss_nll','loss_kl_div','loss_ctc'):prediction=prediction.log_softmax(-1)
            loss=independent_native_objective(prediction,roles,op,r['config'],scale)
            native_optimizer.zero_grad();loss.backward();native_optimizer.step()
    engine.run(graph,tmp_path/'run',epochs=2,seed=41,device=device,batch_size=3,prepared=data,objective_recipe=r,target_mapping=m,objective_scalarization=scale,optimizer_recipe=optimizer(),gradient_policy=policy(scale=scale),dataset_recipe=dr,loader_recipe=lr)
    actual=saved(tmp_path/'run')
    torch.testing.assert_close(actual['modelState']['layers.learn.weight'],w.detach().cpu(),atol=8e-6,rtol=5e-5)
    torch.testing.assert_close(actual['modelState']['layers.learn.bias'],b.detach().cpu(),atol=8e-6,rtol=5e-5)
    assert actual['optimizerUpdates']==[4,4]


def test_selected_ignored_batch_refuses_before_model_allocation(tmp_path,monkeypatch):
    graph,r,m,data=varied_preparation(tmp_path/'source','loss_cross_entropy')
    ignored=(data[0][1]['target']==-100).nonzero().flatten().tolist()
    assert ignored
    monkeypatch.setattr(engine,'GraphModel',lambda *a,**k:pytest.fail('model allocated before undefined selected batch admission'))
    with pytest.raises(ValueError,match='denominator|batch'):
        engine.run(graph,tmp_path/'run',epochs=1,batch_size=1,prepared=data,objective_recipe=r,target_mapping=m,dataset_recipe=recipe(kind='subset',indices=ignored),loader_recipe=recipe('data_loader',batch_size=1))

@pytest.mark.parametrize('device',['cpu','mps'])
def test_selected_kl_batchmean_window(device,tmp_path):
    test_all_eight_selected_shuffled_windows_match_independent_native_full_window(tmp_path,device,'loss_kl_div','batchmean','mean')


def test_selected_normalization_final_group_guard_and_dropped_tail(tmp_path):
    graph,kwargs=setup(tmp_path)
    graph=copy.deepcopy(graph)
    graph['nodes'].insert(-1,dict(id='norm',type='batch_norm1d',config=dict(num_features=2,track_running_stats=True)))
    graph['edges'][-1]['target']='norm';graph['edges'].append(dict(id='norm-out',source='norm',sourcePort='out',target='output',targetPort='in'))
    common=dict(epochs=1,batch_size=2,dataset_recipe=recipe(kind='subset',indices=[4,1,0,7,3]),gradient_policy=policy(),**kwargs)
    with pytest.raises(ValueError,match='Normalization|normalization|BatchNorm|batch'):
        engine.run(graph,tmp_path/'badnorm',loader_recipe=recipe('data_loader',batch_size=2),**common)
    engine.run(graph,tmp_path/'norm',loader_recipe=recipe('data_loader',batch_size=2,drop_last=True),**common)
    state=saved(tmp_path/'norm')
    assert state['modelState']['layers.norm.num_batches_tracked'].item()==2
    assert state['optimizerUpdates']==[1,1,1,1]


def test_onecycle_selected_actual_update_horizon_not_original_batches(tmp_path):
    from runner.scheduler_configuration import canonical_configuration
    graph,kwargs=setup(tmp_path)
    r=dict(schemaVersion=1,configurationVersion=1,componentID='scheduler_one_cycle_lr',config=canonical_configuration('scheduler_one_cycle_lr',dict(max_lr=.04,planned_total_batches=4)))
    engine.run(graph,tmp_path/'run',epochs=2,batch_size=2,dataset_recipe=recipe(kind='subset',indices=[4,1,0,7,3]),loader_recipe=recipe('data_loader',batch_size=2),gradient_policy=policy(),scheduler_recipe=r,**kwargs)
    trace=json.loads((tmp_path/'run/rate-trace.json').read_text())
    assert len(trace['records'])==4 and saved(tmp_path/'run')['optimizerUpdates']==[4,4]

@pytest.mark.parametrize('device',['cpu','mps'])
def test_real_prepared_bags_workers_two_resume_or_fresh_finetune(tmp_path,device):
    from runner.tests.test_prepared_bag_loader import fixture,load
    from runner.tests.test_bag_source_graph import graph as bag_graph
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
    data=load(fixture(tmp_path/'source'));graph=bag_graph();graph['nodes'][0]['config']['maximum_items_per_bag']=2
    common=dict(seed=41,batch_size=4,device=device,prepared=data,dataset_recipe=recipe(kind='boundedIterable',max_rows=9),loader_recipe=recipe('data_loader',batch_size=4,shuffle=True,num_workers=2,prefetch_factor=1,timeout=10,collation='whole_bags'))
    engine.run(graph,tmp_path/'parent',epochs=1,**common)
    parent=(tmp_path/'parent/checkpoint.pt').read_bytes()
    if device=='cpu':
        engine.run(graph,tmp_path/'full',epochs=2,**common)
        engine.run(graph,tmp_path/'child',epochs=2,resume_checkpoint=tmp_path/'parent/checkpoint.pt',**common)
        assert_nested_equal(saved(tmp_path/'full'),saved(tmp_path/'child'))
        assert (tmp_path/'full/metrics.json').read_bytes()==(tmp_path/'child/metrics.json').read_bytes()
    else:
        with pytest.raises(ValueError,match='CPU'):
            engine.run(graph,tmp_path/'unsupported',epochs=2,resume_checkpoint=tmp_path/'parent/checkpoint.pt',**common)
        engine.run(graph,tmp_path/'child',epochs=1,finetune_checkpoint=tmp_path/'parent/checkpoint.pt',**common)
        assert saved(tmp_path/'child')['dataLoadingState']['nextEpoch']==1
    assert parent==(tmp_path/'parent/checkpoint.pt').read_bytes()
    manifest=json.loads((tmp_path/'child/manifest.json').read_text())
    assert manifest['dataLoading']['selectedRows']==9
    assert manifest['predictionPreview']['schemaVersion']==3

@pytest.mark.parametrize('selected',[True,False])
def test_weights_only_child_selected_or_absent_loader_resets_parent_policy(tmp_path,selected):
    graph,kwargs=setup(tmp_path)
    engine.run(graph,tmp_path/'parent',epochs=1,batch_size=3,loader_recipe=recipe('data_loader',batch_size=3,shuffle=True),**kwargs)
    raw=(tmp_path/'parent/checkpoint.pt').read_bytes();child=dict(kwargs)
    if selected:child.update(dataset_recipe=recipe(kind='subset',indices=[2,0,1]),loader_recipe=recipe('data_loader',batch_size=2))
    engine.run(graph,tmp_path/'child',epochs=1,batch_size=2,finetune_checkpoint=tmp_path/'parent/checkpoint.pt',**child)
    state=saved(tmp_path/'child');manifest=json.loads((tmp_path/'child/manifest.json').read_text())
    assert ('dataLoading' in state['contract'])==selected
    assert ('dataLoadingState' in state)==selected
    assert manifest['lineage']['resetState'].count('dataLoading')==1
    assert raw==(tmp_path/'parent/checkpoint.pt').read_bytes()

@pytest.mark.parametrize('mutation',['state','presence','population','hash','boolMetadata'])
def test_resealed_parent_loader_state_and_metadata_refuse(tmp_path,mutation):
    import hashlib
    graph,kwargs=setup(tmp_path);kwargs.update(batch_size=3,loader_recipe=recipe('data_loader',batch_size=3))
    engine.run(graph,tmp_path/'parent',epochs=1,**kwargs);p=tmp_path/'parent';state=saved(p);manifest=json.loads((p/'manifest.json').read_text())
    if mutation=='state':state['dataLoadingState']['pendingBatches']=1
    elif mutation=='presence':state.pop('dataLoadingState')
    elif mutation=='population':manifest['dataLoading']['retainedRows']-=1
    elif mutation=='boolMetadata':manifest['dataLoading']['schemaVersion']=True
    else:manifest['provenance']['dataLoading']['selectionSHA256']='a'*64
    torch.save(state,p/'checkpoint.pt');raw=(p/'checkpoint.pt').read_bytes();manifest['checkpoint'].update(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
    (p/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):engine.run(graph,tmp_path/'child',epochs=2,resume_checkpoint=p/'checkpoint.pt',**kwargs)


def test_independent_discovery_sibling_preserves_forward_component_table():
    from runner.dataset_loader_configuration import capability
    value=engine.capabilities()
    assert value.get('dataLoadingCapabilities')=={k:capability(k) for k in ('dataset_source','data_loader')}
    assert not {'dataset_source','data_loader'} & set(value['componentCapabilities'])


def test_absent_source_loading_retains_prior_complete_checkpoint_metrics_and_export_bytes(tmp_path):
    import importlib.util,subprocess,sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]
    source=(root / 'Tests/historical/ded5dac/runner/engine.py').read_text()
    path=root/'.local/data-loading-accepted-engine-reference.py';path.write_text(source)
    spec=importlib.util.spec_from_file_location('runner._data_loading_accepted_engine_reference',path)
    old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
    graph,kwargs=setup(tmp_path)
    old.run(graph,tmp_path/'old',epochs=2,batch_size=5,**kwargs)
    engine.run(graph,tmp_path/'new',epochs=2,batch_size=5,**kwargs)
    assert_nested_equal(saved(tmp_path/'old'),saved(tmp_path/'new'))
    for file in ('model.py','metrics.json','objective_recipe.py','training_recipe.py'):
        assert (tmp_path/'old'/file).read_bytes()==(tmp_path/'new'/file).read_bytes()
    a=json.loads((tmp_path/'old/manifest.json').read_text());b=json.loads((tmp_path/'new/manifest.json').read_text())
    for field in ('settings','runtime','provenance','evaluation','lineage'):assert a[field]==b[field]
    assert 'dataLoading' not in b and not (tmp_path/'new/data_loading_recipe.py').exists()
