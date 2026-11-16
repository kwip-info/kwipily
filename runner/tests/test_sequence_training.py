import copy
import hashlib
import json
import pytest
import torch
from runner import dataflow as df
from runner.sequence_windows import DEFAULTS
from runner.named_prepared_data import load_named_prepared
from runner.dataflow_cli import run_dataflow
from runner.engine import run
from runner.tests.test_sequence_objective import plan,recipe
from runner.tests.test_named_prepared_integration import fixture as text_fixture
from runner.tests.test_embedding_graph import graph as embedding_graph


def fixture(tmp_path,capture=True,named=True):
    _,dataflow,_=text_fixture(tmp_path)
    dataflow['nodes']=[n for n in dataflow['nodes'] if n['id']!='view']
    dataflow['nodes'].append(dict(id='windows',operator='kwip.data.sequenceWindows',config={**DEFAULTS,'mode':'partitions','contextLength':3,'stride':3,'incomplete':'pad','inputColumn':'x','targetColumn':'y','lengthColumn':'n','maskColumn':'m'}))
    dataflow['edges']=[e for e in dataflow['edges'] if e['to']['node']!='view']
    for role in ('training','validation'):dataflow['edges'].append({'from':dict(node='text',port=role),'to':dict(node='windows',port=role)})
    path=tmp_path/'windows.json';path.write_text(json.dumps(dataflow));out=tmp_path/'windows-prepared'
    run_dataflow(path,{'source':str(tmp_path/'texts.csv')},out,lambda _:None,engine='polars',retain_outputs=capture,required_nodes=['text','windows'])
    p=plan()
    for tensor in p['inputs']+p['roles']:tensor['vocabularySize']=4
    model=embedding_graph();model['nodes']=[n for n in model['nodes'] if n['id']!='flat'];model['nodes'][0]['config']['shape']=[3]
    model['nodes'][-2]['config'].update(in_features=3,out_features=4);model['nodes'][1]['config']['num_embeddings']=4
    model['edges']=[dict(id=str(i),source=a['id'],sourcePort='out',target=b['id'],targetPort='in') for i,(a,b) in enumerate(zip(model['nodes'],model['nodes'][1:]))];model['namedInputs']={'tokens':'source'}
    if not named:model.pop('namedInputs')
    target=dict(schemaVersion=2,componentID='loss_cross_entropy',inputColumns=['x'],roles=[dict(role='target',columns=['y'],sampleShape=[3],dtype='int64')],tripletSourceGroups=[])
    pin=hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest()
    prepared=load_named_prepared(out/'manifest.json','windows','training','windows','validation',p,model,expected_manifest_sha256=pin,expected_graph_digest=df.digest(dataflow),objective_recipe=recipe(),target_mapping=target)
    return prepared,model,target,out,p

@pytest.mark.parametrize('capture',[True,False])
def test_real_window_training_valid_tokens_and_required_only_capture(tmp_path,capture):
    prepared,model,target,_,_=fixture(tmp_path,capture)
    x,y,vx,vy=prepared[0]
    assert y['target'].shape==(48,3) and int((y['target']==-100).sum())==48
    result=run(model,tmp_path/'trained',epochs=2,device='cpu',batch_size=7,prepared=prepared,objective_recipe=recipe(),target_mapping=target,**({} if capture else {'sample_capture':dict(schemaVersion=3,activationValues=False,predictionRows=False,performanceMeasurements=False,retainComponentOutputs=False,retainDiagnosticDetails=False)}))
    manifest=json.loads((tmp_path/'trained/manifest.json').read_bytes())
    metric=manifest['evaluation']['objective'];assert metric['activeCount']==32 and metric['ignoredCount']==16 and metric['aggregationDenominator']==32
    if capture:assert len(json.loads((tmp_path/'trained/predictions.json').read_bytes())['predictionRows'])==8
    else:assert not (tmp_path/'trained/predictions.json').exists()
    assert result['evaluationCount']==16 and torch.isfinite(torch.tensor(metric['effectiveScalar']))
    assert json.loads((tmp_path/'trained/preparation/manifest.json').read_bytes()).get('observationRetention')==('requiredOnly' if not capture else None)


def test_sequence_cpu_continuation_is_exact_and_invalid_mapping_refused(tmp_path):
    prepared,model,target,out,p=fixture(tmp_path)
    kwargs=dict(device='cpu',batch_size=7,prepared=prepared,objective_recipe=recipe(),target_mapping=target)
    run(model,tmp_path/'one',epochs=1,**kwargs);run(model,tmp_path/'full',epochs=2,**kwargs);run(model,tmp_path/'resumed',epochs=2,resume_checkpoint=tmp_path/'one/checkpoint.pt',**kwargs)
    a=torch.load(tmp_path/'full/checkpoint.pt',weights_only=True);b=torch.load(tmp_path/'resumed/checkpoint.pt',weights_only=True)
    for key,value in a['modelState'].items():assert torch.equal(value,b['modelState'][key])
    assert json.loads((tmp_path/'full/metrics.json').read_bytes())==json.loads((tmp_path/'resumed/metrics.json').read_bytes())
    p=copy.deepcopy(p);p['sequenceObjective']['maskColumn']='absent'
    with pytest.raises(ValueError):load_named_prepared(out/'manifest.json','windows','training','windows','validation',p,model,expected_manifest_sha256=hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest(),expected_graph_digest=json.loads((out/'manifest.json').read_bytes())['graphDigest'],objective_recipe=recipe(),target_mapping=target)


def test_sequence_preview_admission_and_drop_exact_length():
    from runner.named_inputs import sequence_prediction_row_limit,convert_rows
    from runner.tests.test_sequence_objective import schema
    p=plan()
    assert sequence_prediction_row_limit(p,[3,5],16)==8
    with pytest.raises(ValueError,match='before training'):sequence_prediction_row_limit(p,[4096,512],16)
    p['sequenceObjective']['maskColumn']=''
    x,y=convert_rows(p,schema()[:3],[[[2,3,4],[3,4,2],3]])
    assert y['target'].tolist()==[[3,4,2]]
    with pytest.raises(ValueError):convert_rows(p,schema()[:3],[[[2,3,4],[3,4,2],2]])


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='MPS unavailable')
def test_real_sequence_mps_training_and_masked_gradient(tmp_path):
    prepared,model,target,_,_=fixture(tmp_path)
    result=run(model,tmp_path/'mps',epochs=1,device='mps',batch_size=7,prepared=prepared,objective_recipe=recipe(),target_mapping=target)
    assert result['device']=='mps' and result['evaluationCount']==16
    logits=torch.randn(2,3,4,device='mps',requires_grad=True)
    y=prepared[0][1]['target'][:2].to('mps')
    loss=torch.nn.functional.cross_entropy(logits.transpose(1,2),y,ignore_index=-100);loss.backward()
    assert torch.equal(logits.grad[y==-100].cpu(),torch.zeros_like(logits.grad[y==-100]).cpu())


def test_required_token_parent_and_window_values_cannot_be_substituted(tmp_path):
    from runner.dataflow_storage import completed_inventory
    from runner.dataflow_cache import table_from_artifact
    from runner.sequence_objective import verify_sequence_preparation
    _,_,target,out,p=fixture(tmp_path)
    manifest,files,_=completed_inventory(out/'manifest.json',hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest(),lazy=True,selected_nodes={'windows'})
    outputs={e['port']:table_from_artifact(json.loads(files[e['path']])) for e in manifest['artifacts'] if e['kind']=='table' and e['nodeID']=='windows'}
    states={}
    for e in manifest['artifacts']:
        if e['kind']=='state':
            state=df.load_state(files[e['path']].decode());states[state['contentDigest']]=state
    verify=lambda m,o:verify_sequence_preparation(m,files,('windows','training'),('windows','validation'),p,o,recipe(),target,states,lambda:None)
    verify(manifest,outputs)
    missing=copy.deepcopy(manifest);missing['artifacts']=[e for e in missing['artifacts'] if not(e.get('nodeID')=='text' and e.get('kind')=='table')]
    with pytest.raises(ValueError,match='required replay'):verify(missing,outputs)
    damaged=dict(outputs);t=outputs['training'];rows=[list(r) for r in t.rows];rows[next(i for i,r in enumerate(rows) if r[-2]<3)][-1]=[1,1,1];damaged['training']=df.derive(t,rows=rows)
    with pytest.raises(ValueError):verify(manifest,damaged)
    p['inputs'][0]['paddingValue']=p['roles'][0]['paddingValue']=0
    with pytest.raises(ValueError):verify(manifest,outputs)


def test_architecture_code_export_does_not_relax_training_admission(tmp_path):
    from runner.engine import GraphModel,export_source
    _,graph,_,_,_=fixture(tmp_path)
    with pytest.raises(ValueError,match='Classification output'):GraphModel(graph,bundled_dataset=False)
    model=GraphModel(graph,bundled_dataset=False,architecture_only=True)
    namespace={};exec(export_source(model),namespace)
    exported=namespace['Model']();exported.load_state_dict(model.state_dict())
    x={'tokens':torch.tensor([[2,3,2]],dtype=torch.int64)}
    with torch.inference_mode():torch.testing.assert_close(exported(x),model(x))


def test_single_unnamed_boundary_uses_the_same_bounded_sequence_capture(tmp_path):
    prepared,model,target,_,_=fixture(tmp_path,named=False)
    run(model,tmp_path/'unnamed',epochs=1,device='cpu',batch_size=7,prepared=prepared,objective_recipe=recipe(),target_mapping=target)
    artifact=json.loads((tmp_path/'unnamed/predictions.json').read_bytes())
    assert len(artifact['predictionRows'])==8
    assert artifact['evaluationCount']==16
    assert (tmp_path/'unnamed/manifest.json').is_file() and (tmp_path/'unnamed/checkpoint.pt').is_file()
