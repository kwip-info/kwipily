"""Capture-off has a real file/collector boundary, independently of UI visibility."""
import json
import torch
from runner.engine import run
from runner.tests.test_runner import graph

OFF = dict(schemaVersion=3,activationValues=False,predictionRows=False,performanceMeasurements=False,retainComponentOutputs=False,retainDiagnosticDetails=False)

def test_off_omits_optional_artifacts_but_keeps_checkpoint_resume_evidence(tmp_path):
    events=[]
    result=run(graph(),tmp_path/'off',epochs=2,device='cpu',sample_capture=OFF,emit=events.append)
    files={p.name for p in (tmp_path/'off').iterdir()}
    assert {'checkpoint.pt','manifest.json','metrics.json','graph.json','model.py'}<=files
    assert not files & {'activations.json','predictions.json','prediction-table.json','performance.json','gradient-summary.json','derivative-results.json'}
    manifest=json.loads((tmp_path/'off/manifest.json').read_bytes())
    assert manifest['sampleCapture']==OFF
    assert not set(manifest)&{'activationPreview','predictionPreview','performanceCapture','gradientCapture','derivativeResults'}
    assert 'previews' not in result and 'predictionRows' not in result
    assert len([e for e in events if e['event']=='epoch'])==2
    assert not any(e['event']=='modelCaptureStarted' for e in events)
    child=tmp_path/'resume'
    run(graph(),child,epochs=3,device='cpu',resume_checkpoint=tmp_path/'off/checkpoint.pt',sample_capture=OFF,emit=lambda _:None)
    assert len(json.loads((child/'metrics.json').read_bytes()))==3
    assert torch.load(child/'checkpoint.pt',weights_only=True,map_location='cpu')['epoch']==3


def test_connected_prediction_table_is_kept_without_duplicate_optional_samples(tmp_path):
    from runner.tests.test_prepared_numeric_width import prepared,load,model_graph
    preparation,_=prepared(tmp_path)
    value=load(preparation,sample_shape=[3])
    policy={**OFF,'predictionRows':True}
    run(model_graph(),tmp_path/'required-output',epochs=1,device='cpu',prepared=value,sample_capture=policy,emit=lambda _:None)
    manifest=json.loads((tmp_path/'required-output/manifest.json').read_bytes())
    assert manifest['predictionTable']['rowCount']==4
    assert (tmp_path/'required-output/prediction-table.json').exists()
    assert not (tmp_path/'required-output/predictions.json').exists()
    assert not (tmp_path/'required-output/activations.json').exists()


def test_preparation_off_omits_intermediate_copies_keeps_model_partitions_and_replay(tmp_path):
    import hashlib
    from runner.dataflow_cli import run_dataflow,inspect_dataflow
    from runner.dataflow_storage import completed_inventory
    from runner.tests.test_prepared_numeric_width import prepared,load
    full,_=prepared(tmp_path)
    source=tmp_path/'source.csv';graph=tmp_path/'preparation.json'
    minimal=tmp_path/'minimal';events=[]
    manifest=run_dataflow(graph,{'source':str(source)},minimal,events.append,retain_outputs=False)
    assert manifest['observationRetention']=='requiredOnly'
    assert not (minimal/'previews.json').exists()
    assert all(not port['preview']['rows'] for event in events if event['event']=='dataflowPreview' for port in event['outputs'].values() if port['kind']=='table')
    table_entries=[a for a in manifest['artifacts'] if a['kind']=='table']
    assert {(a['nodeID'],a['port']) for a in table_entries}=={('split',p) for p in ('training','validation','test','exclusions')}
    data,_,_=load(minimal,sample_shape=[3])
    assert [len(data[i]) for i in (0,2)]==[12,4]
    old={a['port']:a['sha256'] for a in json.loads((full/'manifest.json').read_bytes())['artifacts'] if a.get('nodeID')=='split'}
    assert all(a['sha256']==old[a['port']] for a in table_entries)
    assert sum(a['bytes'] for a in manifest['artifacts']) < sum(a['bytes'] for a in json.loads((full/'manifest.json').read_bytes())['artifacts'])
    pin=hashlib.sha256((minimal/'manifest.json').read_bytes()).hexdigest()
    completed_inventory(minimal/'manifest.json',pin,lazy=True)
    inspected=[];inspect_dataflow(minimal/'manifest.json',pin,inspected.append)
    assert {e['nodeID'] for e in inspected if e['event']=='dataflowPreview'}=={'split'}
    # Source code still reproduces the same preparation without resurrecting optional copies.
    import subprocess,sys
    bindings=tmp_path/'bindings.json';bindings.write_text(json.dumps({'source':str(source)}))
    result=subprocess.run([sys.executable,str(minimal/'prep.py'),'--source-bindings',str(bindings),'--output',str(tmp_path/'replay')],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    again=json.loads((tmp_path/'replay/manifest.json').read_bytes())
    assert again['artifacts']==manifest['artifacts']
    # Required partition loss is refused even with a newly pinned envelope.
    manifest['artifacts']=[a for a in manifest['artifacts'] if a.get('port')!='training']
    (minimal/'manifest.json').write_text(json.dumps(manifest))
    import pytest
    with pytest.raises(ValueError,match='Required preparation outputs'):
        completed_inventory(minimal/'manifest.json',lazy=True)


def test_history_policy_rejects_collection_when_retention_disabled():
    import pytest
    from runner.capture_policy import validate_training_sample_capture_policy
    for fields in [{'activationValues':True},{'performanceMeasurements':True},{'retainComponentOutputs':0}]:
        with pytest.raises(ValueError):validate_training_sample_capture_policy({**OFF,**fields})


def test_off_skips_gradient_collectors_but_preserves_training_recipe(tmp_path,monkeypatch):
    from runner import engine
    from runner.tests.test_loss_engine import prepared
    from runner.tests.test_gradient_engine import policy,optimizer
    from runner.tests.test_requested_gradients import request
    def forbidden(*args,**kwargs):raise AssertionError('Optional diagnostic collector invoked while retention is off')
    from runner import gradient_capture,requested_gradients
    monkeypatch.setattr(gradient_capture,'GradientCapture',forbidden)
    monkeypatch.setattr(requested_gradients,'analyze_requested_gradients',forbidden)
    graph,recipe,mapping,data=prepared(tmp_path/'source','loss_mse')
    selected=policy(requests=[request('sum',output=dict(node='output',port='out'))])
    engine.run(graph,tmp_path/'run',epochs=1,batch_size=5,device='cpu',prepared=data,objective_recipe=recipe,target_mapping=mapping,optimizer_recipe=optimizer(),gradient_policy=selected,derivative_capture=dict(schemaVersion=1,values=True),gradient_capture=dict(schemaVersion=1,everyEpochs=1,maxBatchesPerEpoch=1,maxCaptures=2),sample_capture=OFF,emit=lambda _:None)
    files={p.name for p in (tmp_path/'run').iterdir()}
    assert {'checkpoint.pt','gradient_policy.py','training_recipe.py','objective_recipe.py'}<=files
    assert not {'derivative-results.json','gradient-summary.json','activations.json','predictions.json'}&files


def test_off_bag_input_still_publishes_replayable_model(tmp_path):
    from runner.tests.test_bag_training import prepared,graph
    from runner import engine
    result=run(graph(),tmp_path/'bag',registry=engine.REGISTRY,prepared=prepared(),epochs=1,batch_size=3,device='cpu',sample_capture=OFF,emit=lambda _:None)
    assert result['evaluationCount']==4
    assert (tmp_path/'bag/checkpoint.pt').is_file()
    assert not (tmp_path/'bag/activations.json').exists()
    assert not (tmp_path/'bag/predictions.json').exists()


def test_nonterminal_model_input_and_explicit_view_remain_required(tmp_path):
    from runner.tests.test_prepared_numeric_width import prepared
    from runner.dataflow_cli import run_dataflow
    from runner.dataflow_storage import completed_inventory
    prepared(tmp_path)
    path=tmp_path/'preparation.json';graph=json.loads(path.read_bytes())
    graph['nodes'].append(dict(id='view',operator='kwip.data.view',config={}))
    graph['edges'].append(dict(**{'from':dict(node='split',port='training'),'to':dict(node='view',port='table')}))
    path.write_text(json.dumps(graph))
    manifest=run_dataflow(path,{'source':str(tmp_path/'source.csv')},tmp_path/'retained',lambda _:None,retain_outputs=False,required_nodes=['split'])
    assert manifest['requiredOutputNodes']==['split']
    endpoints={(a.get('nodeID'),a.get('port')) for a in manifest['artifacts'] if a['kind']=='table'}
    assert {('split',p) for p in ('training','validation','test','exclusions')}<=endpoints
    assert ('view','table') in endpoints and ('source','table') not in endpoints and ('cast','table') not in endpoints
    assert manifest['tableViews']['view']['rowCount']==12
    completed_inventory(tmp_path/'retained/manifest.json',lazy=True)
