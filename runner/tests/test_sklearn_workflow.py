import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
pytest.importorskip("sklearn", reason="BACK-01 optional isolated framework environment")
from runner.dataflow_cli import run_dataflow
from runner.sklearn_workflow import run_prepared
from runner import sklearn_backend
ROOT=Path(__file__).resolve().parents[2]


def request(tmp_path):
    path=tmp_path/'prepared'
    run_dataflow(ROOT/'examples/dataflow/scaler.json',{'source':str(ROOT/'examples/dataflow/observations.csv')},path,lambda _:None)
    raw=(path/'manifest.json').read_bytes()
    return {'schemaVersion':1,'backendID':'sklearn','configuration':sklearn_backend.default_configuration(),
            'prepared':{'manifest':str(path/'manifest.json'),'manifestSHA256':hashlib.sha256(raw).hexdigest(),
                        'graphDigest':json.loads(raw)['graphDigest'],'trainingNode':'training','trainingPort':'table',
                        'validationNode':'heldout','validationPort':'table','featureColumns':['feature','second'],'targetColumn':'group'}}


def test_real_preparation_fit_predict_saved_portable_export(tmp_path):
    source=request(tmp_path);events=[];output=tmp_path/'run'
    result=run_prepared(source,output,events.append)
    assert result['backendID']=='sklearn'
    assert result['evaluation']['count']==8
    assert result['provenance']['trainingSamples']==24  # Current 40-row fixture retains 60/20/20 including Test.
    assert result['provenance']['preparation']['manifestSHA256']==source['prepared']['manifestSHA256']
    assert set(result['provenance']['trainingRowIDs']).isdisjoint(result['provenance']['validationRowIDs'])
    assert result['provenance']['preparation']['fittedStateDigests']
    assert [event['event'] for event in events]==['backendStarted','backendCompleted']
    assert (output/'portable/inference.py').is_file()
    assert (output/'preparation/prep.py').is_file()
    assert source['prepared']['manifest'] not in (output/'manifest.json').read_text()
    for item in result['artifacts']:
        raw=(output/item['path']).read_bytes()
        assert item['sha256']==hashlib.sha256(raw).hexdigest()
        assert item['bytes']==len(raw)
    before=(output/'manifest.json').read_bytes()
    with pytest.raises(FileExistsError):run_prepared(source,output,events.append)
    assert (output/'manifest.json').read_bytes()==before


def test_changed_pin_unknown_settings_cancellation_never_publish(tmp_path):
    source=request(tmp_path)
    source['prepared']['manifestSHA256']='0'*64
    with pytest.raises(ValueError):run_prepared(source,tmp_path/'bad',lambda _:None)
    assert not (tmp_path/'bad').exists()
    source=request(tmp_path/'other');source['device']='mps'
    with pytest.raises(ValueError):run_prepared(source,tmp_path/'bad',lambda _:None)
    source.pop('device')
    def cancel():raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError):run_prepared(source,tmp_path/'cancelled',lambda _:None,check_cancelled=cancel)
    assert not (tmp_path/'cancelled').exists()


def test_failure_emitting_completion_does_not_leave_success(tmp_path):
    source=request(tmp_path)
    def emit(event):
        if event['event']=='backendCompleted':raise RuntimeError('delivery failed')
    with pytest.raises(RuntimeError):run_prepared(source,tmp_path/'failed',emit)
    assert not (tmp_path/'failed').exists()


def test_real_cli_request_pin_and_example_resource(tmp_path):
    import subprocess
    import sys
    source=request(tmp_path);path=tmp_path/'request.json';path.write_text(json.dumps(source));pin=hashlib.sha256(path.read_bytes()).hexdigest()
    command=[sys.executable,str(ROOT/'runner/sklearn_workflow.py'),'--request',str(path),'--request-sha256',pin,'--output',str(tmp_path/'cli')]
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert json.loads(result.stdout.splitlines()[-1])['event']=='backendCompleted'
    command[command.index(pin)]='0'*64
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode!=0
    assert 'hash' in json.loads(result.stdout)['message']
    example=json.loads((ROOT/'runner/backend_examples/sklearn_logistic.json').read_text())
    assert example['configuration']==sklearn_backend.default_configuration()
    lesson=json.loads((ROOT/example['lessonResource']).read_text())
    assert lesson['assessment'].startswith('Starter')
    assert lesson['id']=='sklearn.logistic-foundation'


def test_shared_main_backend_dispatch(tmp_path):
    import subprocess,sys
    source=request(tmp_path);path=tmp_path/'request.json';path.write_text(json.dumps(source));pin=hashlib.sha256(path.read_bytes()).hexdigest()
    command=[sys.executable,str(ROOT/'runner/main.py'),'--fit-backend',str(path),'--backend-request-sha256',pin,'--output',str(tmp_path/'shared')]
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert json.loads(result.stdout.splitlines()[-1])['event']=='backendCompleted'
    for extra in (['--epochs','2'],['--registry',str(ROOT/'Sources/KwipMLCore/Resources/components.json')]):
        result=subprocess.run(command+extra,capture_output=True,text=True)
        assert result.returncode!=0
        assert 'requires only' in result.stdout


def test_prediction_capture_is_explicit_opt_in(tmp_path):
    source=request(tmp_path)
    silent=run_prepared(source,tmp_path/'silent',lambda _:None)
    assert 'predictions' not in silent
    source['capturePredictions']=True
    captured=run_prepared(source,tmp_path/'captured',lambda _:None)
    assert captured['predictions']['rowIDs']==captured['provenance']['validationRowIDs'][:8]
    assert len(captured['predictions']['classes'])==8
    source['capturePredictions']=1
    with pytest.raises(ValueError):run_prepared(source,tmp_path/'bad-flag',lambda _:None)


@pytest.mark.parametrize('supervised_arguments', [False, True])
def test_verified_packaged_bootstrap_accepts_only_backend_and_pinned_registry(tmp_path, supervised_arguments):
    """Exercise the real bootstrap and production argument augmentation, not main.py alone."""
    import importlib.util, shutil, subprocess, sys
    from runner.worker_bootstrap import RuntimeContext, _arguments
    payload=tmp_path/'runtime';payload.mkdir()
    shutil.copytree(ROOT/'runner',payload/'runner',ignore=shutil.ignore_patterns('tests','__pycache__','*.pyc','.pytest_cache'))
    shutil.copy2(ROOT/'scripts/runtime_launch.py',payload/'launch.py')
    shutil.copy2(ROOT/'Sources/KwipMLCore/Resources/components.json',payload/'components.json')
    spec=importlib.util.spec_from_file_location('backend_bootstrap_test',payload/'launch.py')
    bootstrap=importlib.util.module_from_spec(spec);spec.loader.exec_module(bootstrap)
    (payload/'runtime-manifest.json').write_text(json.dumps({'schemaVersion':1,'workerIntegrityVersion':1,'workerSourceHashes':bootstrap.worker_source_hashes(payload)}))
    source=request(tmp_path);path=tmp_path/'request.json';path.write_text(json.dumps(source));pin=hashlib.sha256(path.read_bytes()).hexdigest()
    arguments=['--fit-backend',str(path),'--backend-request-sha256',pin,'--output',str(tmp_path/'packaged')]
    if supervised_arguments:
        context=RuntimeContext(payload/'launch.py',payload/'runner/native/lifetime_watchdog.dylib',payload/'components.json',isolated=True)
        arguments=_arguments({'arguments':arguments},context)
        assert arguments[-2:]==['--registry',str(payload/'components.json')]
    command=[sys.executable,'-I','-B',str(payload/'launch.py'),*arguments]
    result=subprocess.run(command,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    assert json.loads(result.stdout.splitlines()[-1])['event']=='backendCompleted'
    assert (tmp_path/'packaged/portable/inference.py').is_file()
    for extra in (['--registry',str(tmp_path/'untrusted.json')],['--epochs','2']):
        refused=subprocess.run(command+extra,capture_output=True,text=True,timeout=30)
        assert refused.returncode!=0
        assert not any(json.loads(line).get('event')=='backendCompleted' for line in refused.stdout.splitlines())
