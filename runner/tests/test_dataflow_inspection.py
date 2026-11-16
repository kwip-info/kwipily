import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import pytest
from runner.dataflow_cli import run_dataflow
from runner.tests.test_dataflow_cli import invoke, ROOT


def fixture(tmp_path,policy='off',**options):
    output=tmp_path/'result';events=[]
    run_dataflow(ROOT/'examples/dataflow/scaler.json',{'source':str(ROOT/'examples/dataflow/observations.csv')},output,events.append,cache_policy=policy,**options)
    path=output/'manifest.json'
    return path,hashlib.sha256(path.read_bytes()).hexdigest(),events

@pytest.mark.parametrize('policy',['off','materialize','reuse'])
def test_restore_matches_outputs_without_source_or_execution(tmp_path,policy):
    options={}
    if policy=='reuse':
        prior,pin,_=fixture(tmp_path/'old','materialize');options=dict(reuse_manifest=prior,reuse_manifest_sha256=pin)
    path,pin,original=fixture(tmp_path,policy,**options)
    from runner.dataflow_cli import inspect_dataflow
    events=[]
    with patch('runner.dataflow.execute',side_effect=AssertionError('must not execute')):
        inspect_dataflow(path,pin,events.append)
    assert [e['outputs'] for e in events if e['event']=='dataflowPreview']==[e['outputs'] for e in original if e['event']=='dataflowPreview']
    assert events[-1]['cache']==original[-1]['cache']
    assert events[-1]['provenance']['sourceRevalidated'] is False
    assert events[-1]['manifestSHA256']==pin


def test_real_cli_pin_and_all_artifacts_verified(tmp_path):
    path,pin,_=fixture(tmp_path)
    process,events=invoke('--inspect-dataflow',path,'--dataflow-manifest-sha256',pin)
    assert process.returncode==0,events
    assert events[-1]['event']=='dataflowInspectionCompleted'
    (path.parent/'prep.py').write_text('raise Exception("must never execute")')
    process,events=invoke('--inspect-dataflow',path,'--dataflow-manifest-sha256',pin)
    assert process.returncode==1
    assert not any(e['event']=='dataflowPreview' for e in events)


def test_inspection_cancellation_preserves_original(tmp_path):
    path,pin,_=fixture(tmp_path)
    from runner.dataflow_cli import inspect_dataflow
    def stop(event):raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):inspect_dataflow(path,pin,stop)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==pin

@pytest.mark.parametrize('extra',[[],['--dataflow-manifest-sha256','0'*64],['--dataflow-manifest-sha256','0'*64,'--epochs','2']])
def test_inspection_rejects_missing_wrong_pin_or_unrelated_flags(tmp_path,extra):
    path,_,_=fixture(tmp_path)
    process,events=invoke('--inspect-dataflow',path,*extra)
    assert process.returncode==1
    assert events[-1]['event']=='error'


def repin(path,change):
    manifest=json.loads(path.read_bytes());change(manifest)
    path.write_text(json.dumps(manifest));return hashlib.sha256(path.read_bytes()).hexdigest()

@pytest.mark.parametrize('mutation',[
    lambda m:m.update(schemaVersion=True),
    lambda m:m['artifacts'][0].update(path='../escape'),
    lambda m:m['artifacts'][0].update(bytes=True),
    lambda m:m['artifacts'].append(m['artifacts'][0]),
    lambda m:m['cache'].update(schemaVersion=2),
    lambda m:m['cache'].update(executedNodeIDs=[]),
    lambda m:m.update(registryDigest='0'*64),
])
def test_malformed_historical_manifest_never_emits_outputs(tmp_path,mutation):
    path,_,_=fixture(tmp_path);pin=repin(path,mutation)
    from runner.dataflow_cli import inspect_dataflow
    events=[]
    with pytest.raises(ValueError):inspect_dataflow(path,pin,events.append)
    assert [e['event'] for e in events]==['dataflowInspectionStarted']


def test_historical_runtime_is_disclosed_not_current_reuse_claim(tmp_path):
    path,_,_=fixture(tmp_path,'materialize')
    pin=repin(path,lambda m:(m.update(numpyVersion='historical-1'),m['cache'].update(implementationDigest='1'*64)))
    from runner.dataflow_cli import inspect_dataflow
    events=[];inspect_dataflow(path,pin,events.append)
    assert events[-1]['provenance']['numpyVersion']=='historical-1'
    assert events[-1]['provenance']['cacheReuseVerified'] is False
    assert events[-1]['cache']['implementationDigest']=='1'*64


def test_symlink_artifact_is_rejected_even_with_matching_bytes(tmp_path):
    path,pin,_=fixture(tmp_path)
    target=path.parent/'prep.py';outside=tmp_path/'outside.py';target.rename(outside);target.symlink_to(outside)
    from runner.dataflow_cli import inspect_dataflow
    with pytest.raises(ValueError,match='regular'):inspect_dataflow(path,pin,lambda _:None)


def test_real_cli_cancel_preserves_entire_saved_result(tmp_path):
    import subprocess
    import sys
    path,pin,_=fixture(tmp_path)
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in path.parent.iterdir()}
    # Stop at started callback deterministically before verified-file reads.
    script="""import sys,signal,time
sys.path.insert(0,sys.argv[1])
from dataflow_cli import inspect_dataflow
signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
def emit(event):
 print(event['event'],flush=True)
 if event['event']=='dataflowInspectionStarted':time.sleep(10)
try:inspect_dataflow(sys.argv[2],sys.argv[3],emit)
except KeyboardInterrupt:sys.exit(130)
"""
    process=subprocess.Popen([sys.executable,'-c',script,str(ROOT/'runner'),str(path),pin],stdout=subprocess.PIPE,text=True)
    assert process.stdout.readline().strip()=='dataflowInspectionStarted'
    process.terminate();stdout,_=process.communicate(timeout=5)
    assert process.returncode==130 and 'dataflowInspectionCompleted' not in stdout
    assert before=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in path.parent.iterdir()}


def rewrite_artifact(path,kind,change):
    manifest=json.loads(path.read_bytes())
    entry=next(e for e in manifest['artifacts'] if e.get('kind')==kind)
    file=path.parent/entry['path'];value=json.loads(file.read_bytes());change(value)
    raw=json.dumps(value).encode();file.write_bytes(raw)
    entry.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    path.write_text(json.dumps(manifest));return hashlib.sha256(path.read_bytes()).hexdigest()

@pytest.mark.parametrize('kind,change',[
    ('table',lambda v:v.update(lineage=[])),
    ('table',lambda v:v['rows'][0].append('extra')),
    ('state',lambda v:v.update(schemaVersion=999)),
    ('state',lambda v:v.update(operator='kwip.data.filter')),
])
def test_repinned_malformed_output_no_partial_delivery(tmp_path,kind,change):
    path,_,_=fixture(tmp_path);pin=rewrite_artifact(path,kind,change)
    from runner.dataflow_cli import inspect_dataflow
    events=[]
    with pytest.raises(ValueError):inspect_dataflow(path,pin,events.append)
    assert [e['event'] for e in events]==['dataflowInspectionStarted']


def test_missing_port_and_aggregate_bounds_have_no_partial_delivery(tmp_path):
    from runner.dataflow_cli import inspect_dataflow
    path,pin,_=fixture(tmp_path);events=[]
    with patch('runner.dataflow_cli.MAX_INSPECTION_PREVIEW_BYTES',1):
        with pytest.raises(ValueError,match='aggregate'):inspect_dataflow(path,pin,events.append)
    assert [e['event'] for e in events]==['dataflowInspectionStarted']
    pin=repin(path,lambda m:m['artifacts'].pop(0));events=[]
    with pytest.raises(ValueError,match='missing'):inspect_dataflow(path,pin,events.append)
    assert [e['event'] for e in events]==['dataflowInspectionStarted']


def test_pinned_archived_program_is_data_only(tmp_path):
    path,_,_=fixture(tmp_path)
    manifest=json.loads(path.read_bytes());entry=next(e for e in manifest['artifacts'] if e['path']=='dataflow.py')
    payload=b'raise RuntimeError("archived program must never be imported")'
    (path.parent/entry['path']).write_bytes(payload);entry.update(bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())
    path.write_text(json.dumps(manifest));pin=hashlib.sha256(path.read_bytes()).hexdigest()
    from runner.dataflow_cli import inspect_dataflow
    events=[];inspect_dataflow(path,pin,events.append)
    assert events[-1]['event']=='dataflowInspectionCompleted'


@pytest.mark.parametrize('when',['before-read','after-artifact','before-terminal'])
def test_cooperative_saved_inspection_cancellation(tmp_path,monkeypatch,when):
    from runner import dataflow_cli,dataflow_storage
    path,pin,_=fixture(tmp_path);events=[];read_artifact=False
    original=dataflow_storage.bounded_regular_read
    def read(file,limit):
        nonlocal read_artifact
        value=original(file,limit)
        if Path(file).name!='manifest.json':read_artifact=True
        return value
    monkeypatch.setattr(dataflow_storage,'bounded_regular_read',read)
    def check():
        if when=='before-read' or (when=='after-artifact' and read_artifact) or (when=='before-terminal' and any(e['event']=='dataflowPreview' for e in events)):
            raise InterruptedError('cancelled saved inspection')
    with pytest.raises(InterruptedError):dataflow_cli.inspect_dataflow(path,pin,events.append,check_cancelled=check)
    assert not any(e['event']=='dataflowInspectionCompleted' for e in events)
    if when!='before-terminal':assert not any(e['event']=='dataflowPreview' for e in events)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==pin
