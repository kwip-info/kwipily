import hashlib
import json
from pathlib import Path
import pytest
pytest.importorskip('tensorflow')
from runner.dataflow_cli import run_dataflow
from runner.tensorflow_workflow import run_prepared
ROOT=Path(__file__).resolve().parents[2]

def request(tmp_path):
    path=tmp_path/'prepared'
    run_dataflow(ROOT/'examples/dataflow/scaler.json',{'source':str(ROOT/'examples/dataflow/observations.csv')},path,lambda _:None)
    raw=(path/'manifest.json').read_bytes();example=json.loads((ROOT/'runner/backend_examples/tensorflow_classifier.json').read_text())
    return {'schemaVersion':1,'backendID':'tensorflow','configuration':{'graph':example['graph'],'settings':{**example['settings'],'epochs':2}},'prepared':{'manifest':str(path/'manifest.json'),'manifestSHA256':hashlib.sha256(raw).hexdigest(),'graphDigest':json.loads(raw)['graphDigest'],'trainingNode':'training','trainingPort':'table','validationNode':'heldout','validationPort':'table','featureColumns':['feature','second'],'targetColumn':'group'}}

def test_actual_preparation_train_export(tmp_path):
    source=request(tmp_path);events=[];output=tmp_path/'run'
    result=run_prepared(source,output,events.append)
    assert result['backendID']=='tensorflow' and result['evaluation']['samples']==8
    assert result['provenance']['trainingSamples']==32
    assert result['provenance']['preparation']['manifestSHA256']==source['prepared']['manifestSHA256']
    assert set(result['provenance']['trainingRowIDs']).isdisjoint(result['provenance']['validationRowIDs'])
    assert (output/'portable/inference.py').is_file() and (output/'preparation/prep.py').is_file()
    assert source['prepared']['manifest'] not in (output/'manifest.json').read_text()
    assert events[0]['event']=='backendStarted' and events[-1]['event']=='backendCompleted'
    for item in result['artifacts']:
        raw=(output/item['path']).read_bytes();assert item['sha256']==hashlib.sha256(raw).hexdigest()
    with pytest.raises(FileExistsError):run_prepared(source,output,events.append)

def test_wrong_pin_and_failed_completion_no_published_success(tmp_path):
    source=request(tmp_path);source['prepared']['manifestSHA256']='0'*64
    with pytest.raises(ValueError):run_prepared(source,tmp_path/'bad',lambda _:None)
    assert not (tmp_path/'bad').exists()
    source=request(tmp_path/'other')
    def emit(event):
        if event['event']=='backendCompleted':raise RuntimeError('delivery failed')
    with pytest.raises(RuntimeError):run_prepared(source,tmp_path/'failed',emit)
    assert not (tmp_path/'failed').exists()

def test_native_style_cli_dispatch(tmp_path):
    import subprocess
    import sys
    source=request(tmp_path)
    path=tmp_path/'request.json';raw=json.dumps(source).encode();path.write_bytes(raw)
    output=tmp_path/'cli-run'
    result=subprocess.run([sys.executable,str(ROOT/'runner/main.py'),'--fit-backend',str(path),'--backend-request-sha256',hashlib.sha256(raw).hexdigest(),'--output',str(output)],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+'\n'+result.stderr
    events=[json.loads(line) for line in result.stdout.splitlines()]
    assert events[-1]['event']=='backendCompleted'

def test_prediction_capture_requires_explicit_boolean_opt_in(tmp_path):
    source=request(tmp_path);plain=run_prepared(source,tmp_path/'plain',lambda _:None)
    assert 'predictions' not in plain
    source['capturePredictions']=True
    captured=run_prepared(source,tmp_path/'captured',lambda _:None)
    predictions=captured['predictions']
    assert len(predictions['logits'])==8
    assert predictions['rowIDs']==captured['provenance']['validationRowIDs'][:8]
    assert predictions['classes']==[int(row[1]>row[0]) for row in predictions['logits']]
    source['capturePredictions']=False
    assert 'predictions' not in run_prepared(source,tmp_path/'disabled',lambda _:None)
    source['capturePredictions']=1
    with pytest.raises(ValueError):run_prepared(source,tmp_path/'invalid',lambda _:None)
