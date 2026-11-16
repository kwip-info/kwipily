import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from runner import prebuilt_models as pb
from runner.sklearn_workflow import run_prepared
from runner.tests.test_sklearn_workflow import request

@pytest.mark.parametrize('algorithm',list(pb.catalog()))
def test_complete_fit_save_reload_frozen_inference_and_export(tmp_path,algorithm):
    r=request(tmp_path);c=pb.defaults(algorithm)
    r['configuration']=dict(task='prebuilt',algorithm=algorithm,operation='fit',settings=c['options'])
    if c['task']=='clustering':r['prepared']['targetColumn']=''
    out=tmp_path/'fit';m=run_prepared(r,out,lambda _:None)
    table=json.loads((out/'predictions.json').read_bytes())
    assert len(table['rows'])==m['evaluation']['count']==(24 if c['task']=='clustering' else 8)
    population=json.loads((out/'population.json').read_bytes())
    assert table['lineage']['observationIDs']==population['inputRowIDs']
    from runner.dataflow_table_reader import read_table_artifact
    for name in ['predictions.json']+(['profiles.json'] if c['task']=='clustering' else ['confusion.json'] if c['task']=='classification' else []):
        artifact=out/name;loaded=read_table_artifact(artifact,hashlib.sha256(artifact.read_bytes()).hexdigest())
        assert len(loaded.rows)>0
        loaded.rows.store.close();loaded.row_ids.store.close()
    for f in m['artifacts']:
        raw=(out/f['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==f['sha256'] and len(raw)==f['bytes']
    if algorithm=='dbscan':
        assert not (out/'model-state.json').exists()
        assert (out/'profiles.json').exists()
        return
    state=(out/'model-state.json').read_bytes();model=pb.load(state)
    assert model['featureColumns']==r['prepared']['featureColumns']
    r['configuration']['operation']='predict';r['model']=dict(path=str(out/'model-state.json'),sha256=hashlib.sha256(state).hexdigest())
    follow=run_prepared(r,tmp_path/'predict',lambda _:None)
    assert (tmp_path/'predict/model-state.json').read_bytes()==state
    assert follow['evaluation']==dict(count=24,**({} if c['task']!='clustering' else {k:v for k,v in follow['evaluation'].items() if k!='count'}))
    x=np.asarray([[0.,1.],[2.,3.]],dtype=np.float64);inputs=tmp_path/'inputs.json';inputs.write_text(json.dumps(x.tolist()))
    result=json.loads(subprocess.check_output([sys.executable,str(out/'portable/inference.py'),str(inputs)]))
    assert result==pb.predict(model,x)
    assert 'accuracy' not in follow['evaluation']


def test_invalid_changed_model_and_cancellation_never_publish(tmp_path):
    r=request(tmp_path);c=pb.defaults('kmeans');r['configuration']=dict(task='prebuilt',algorithm='kmeans',operation='fit',settings=c['options']);r['prepared']['targetColumn']=''
    def cancelled():raise InterruptedError('cancelled')
    with pytest.raises(InterruptedError):run_prepared(r,tmp_path/'cancelled',lambda _:None,check_cancelled=cancelled)
    assert not (tmp_path/'cancelled').exists()
    bad=copy.deepcopy(r);bad['prepared']['manifestSHA256']='0'*64
    with pytest.raises(ValueError):run_prepared(bad,tmp_path/'bad',lambda _:None)
    assert not (tmp_path/'bad').exists()


def test_catalog_copy_is_exact():
    root=Path(__file__).resolve().parents[2]
    assert (root/'runner/prebuilt-models.json').read_bytes()==(root/'Sources/KwipMLCore/Resources/prebuilt-models.json').read_bytes()

def test_code_registry_identity_ignores_only_descriptive_prose():
    from runner.workflow_code import _registry_identity
    left={'operators':[{'id':'example','configSchema':{'type':'object'}}]}
    right=copy.deepcopy(left);right['operators'][0]['description']='Human-facing explanation omitted by the native schema codec'
    assert _registry_identity(left)==_registry_identity(right)
    right['operators'][0]['configSchema']['additionalProperties']=False
    assert _registry_identity(left)!=_registry_identity(right)

def test_large_population_keeps_ids_in_artifact_not_main_thread_receipt(tmp_path):
    from runner.dataflow_cli import run_dataflow
    graph={'schemaVersion':1,'nodes':[{'id':'source','operator':'kwip.data.sourceCSV','config':{'separator':','}}],'edges':[]}
    graph['nodes'].append(dict(id='numeric',operator='kwip.data.columns',config=dict(rows=[dict(source=n,included=True,outputName=n,targetType='keep' if n=='id' else 'float64') for n in ['id','x','y','z']],castFailure='error',trimText=False,newColumns='drop',missingPolicy='error')))
    graph['edges']=[{'from':{'node':'source','port':'table'},'to':{'node':'numeric','port':'table'}}]
    gp=tmp_path/'graph.json';gp.write_text(json.dumps(graph));csv=tmp_path/'rows.csv'
    csv.write_text('id,x,y,z\n'+''.join(f'{i},{i%4+.01*(i%13)},{i%4+.02*(i%7)},{i%4+.03*(i%5)}\n' for i in range(10000)))
    prep=tmp_path/'prep';run_dataflow(gp,{'source':str(csv)},prep,lambda _:None)
    raw=(prep/'manifest.json').read_bytes();c=pb.defaults('kmeans')
    r={'schemaVersion':1,'backendID':'sklearn','configuration':dict(task='prebuilt',algorithm='kmeans',operation='fit',settings=c['options']),
       'prepared':dict(manifest=str(prep/'manifest.json'),manifestSHA256=hashlib.sha256(raw).hexdigest(),graphDigest=json.loads(raw)['graphDigest'],trainingNode='numeric',trainingPort='table',validationNode='numeric',validationPort='table',featureColumns=['x','y','z'],targetColumn='')}
    out=tmp_path/'fit';m=run_prepared(r,out,lambda _:None)
    assert len(pb.canonical(m))<65536
    assert "inputRowIDs" not in m["provenance"]
    population=json.loads((out/'population.json').read_bytes())
    assert len(population['inputRowIDs'])==10000
    assert m['evaluation']['count']==10000
    assert len(json.loads((out/'predictions.json').read_bytes())['rows'])==10000

@pytest.mark.parametrize('operation',['fit','evaluate'])
@pytest.mark.parametrize('target_type',['string','bool'])
def test_regression_refuses_implicit_target_coercion(tmp_path,operation,target_type):
    from runner import dataflow as df
    from runner.dataflow_cli import run_dataflow
    csv=tmp_path/'targets.csv'
    csv.write_text('x,label\n'+''.join(f'{i},{"true" if i%2 else "false"}\n' if target_type=='bool' else f'{i},{i*2}\n' for i in range(30)))
    columns=copy.deepcopy(df.DEFINITIONS['columns']['defaultConfig'])
    columns.update(rows=[dict(source='x',included=True,outputName='x',targetType='float64'),dict(source='label',included=True,outputName='label',targetType='bool' if target_type=='bool' else 'keep')])
    graph=dict(schemaVersion=1,nodes=[dict(id='source',operator='kwip.data.sourceCSV',config={}),dict(id='columns',operator='kwip.data.columns',config=columns),dict(id='split',operator='kwip.data.partition',config=copy.deepcopy(df.DEFINITIONS['partition']['defaultConfig']))],edges=[{'from':dict(node='source',port='table'),'to':dict(node='columns',port='table')},{'from':dict(node='columns',port='table'),'to':dict(node='split',port='table')}])
    gp=tmp_path/'graph.json';gp.write_text(json.dumps(graph));prep=tmp_path/'prep'
    run_dataflow(gp,{'source':str(csv)},prep,lambda _:None)
    raw=(prep/'manifest.json').read_bytes();c=pb.defaults('ridge')
    r=dict(schemaVersion=1,backendID='sklearn',configuration=dict(task='prebuilt',algorithm='ridge',operation=operation,settings=c['options']),prepared=dict(manifest=str(prep/'manifest.json'),manifestSHA256=hashlib.sha256(raw).hexdigest(),graphDigest=json.loads(raw)['graphDigest'],trainingNode='split',trainingPort='training',validationNode='split',validationPort='validation',featureColumns=['x'],targetColumn='label'))
    if operation=='evaluate':
        x=np.arange(10,dtype=np.float64).reshape(-1,1);model,_=pb.fit(c,x,x[:,0]*2);model['featureColumns']=['x']
        state=tmp_path/'model.json';state.write_bytes(pb.canonical(model));r['model']=dict(path=str(state),sha256=hashlib.sha256(state.read_bytes()).hexdigest())
    with pytest.raises(ValueError,match='numeric target'):
        run_prepared(r,tmp_path/'result',lambda _:None)
    assert not (tmp_path/'result').exists()

@pytest.mark.parametrize('oversized',['input','state'])
def test_portable_inference_bounds_files_before_json_parsing(tmp_path,oversized):
    x=np.arange(6,dtype=np.float64).reshape(-1,1);model,_=pb.fit(pb.defaults('ridge'),x,x[:,0])
    package=tmp_path/'portable';pb.export(model,package)
    inputs=tmp_path/'input.json';inputs.write_text('[[1.0]]')
    path=inputs if oversized=='input' else package/'state.json'
    with path.open('wb') as stream:stream.truncate(64_000_001 if oversized=='input' else 32_000_001)
    result=subprocess.run([sys.executable,str(package/'inference.py'),str(inputs)],capture_output=True,text=True,timeout=15)
    assert result.returncode!=0
    assert ('Input exceeds its byte budget' if oversized=='input' else 'Model exceeds its byte budget') in result.stderr
    assert 'JSONDecodeError' not in result.stderr

@pytest.mark.parametrize('operation',['predict','evaluate'])
def test_frozen_operations_preserve_model_and_have_only_input_population(tmp_path,operation):
    r=request(tmp_path);c=pb.defaults('ridge')
    r['configuration']=dict(task='prebuilt',algorithm='ridge',operation='fit',settings=c['options'])
    fit_out=tmp_path/'fit';fit=run_prepared(r,fit_out,lambda _:None)
    assert fit['schemaVersion']==1
    assert fit['provenance']['trainingSamples']==24 and fit['provenance']['validationSamples']==8
    state=(fit_out/'model-state.json').read_bytes();pin=hashlib.sha256(state).hexdigest()
    r['configuration']['operation']=operation;r['model']=dict(path=str(fit_out/'model-state.json'),sha256=pin)
    events=[];out=tmp_path/'frozen';m=run_prepared(r,out,events.append)
    population=json.loads((out/'population.json').read_bytes())
    table=json.loads((out/'predictions.json').read_bytes())
    assert m['schemaVersion']==2 and m['provenance']['modelSHA256']==pin
    assert (out/'model-state.json').read_bytes()==state
    assert set(population)=={'inputRowIDs'} and population['inputRowIDs']==table['lineage']['observationIDs']
    assert m['provenance']['inputSamples']==len(population['inputRowIDs'])==24
    assert not {'trainingSamples','validationSamples','trainingRowIDs','validationRowIDs'} & m['provenance'].keys()
    assert set(m['provenance']['preparation'])=={'manifestSHA256','graphDigest','inputNode','inputPort'}
    assert m['provenance']['preparation']['inputNode']==r['prepared']['trainingNode']
    # This fixture deliberately predicts the prepared Training table: labels must
    # describe operation input without falsifying the table's existing role.
    assert m['predictionPartition']==table['lineage']['role']=='training'
    start=next(e for e in events if e['event']=='backendStarted')
    assert start['inputSamples']==24 and 'trainingSamples' not in start and 'validationSamples' not in start
    assert ('rmse' in m['evaluation'])==(operation=='evaluate')
