"""Source inspection remains bounded, pinned and cancellable through its supervisor."""
import json
import os
from pathlib import Path
import sys
import pytest
from runner.tests.test_operation_authorization import operation_authority,publish,claim
from runner.tests.test_evaluation_authorization import encoded,digest
from runner.tests.test_operation_bootstrap import control
from runner.tests.test_worker_bootstrap import entry,run_supervisor
from runner.tests.test_lifetime_watchdog import libraries


@pytest.fixture
def source_operation(operation_authority):
    f=operation_authority;v=f[1]
    for key in ('runID','directoryID','phase'):del v[key]
    folder=f[0]/'input/data';folder.mkdir(parents=True)
    raw=b'feature,label\n1,0\n2,1\n';path=folder/'original.csv';path.write_bytes(raw)
    v.update(schemaVersion=2,toolMode='sourceInspection',arguments=['--inspect-data',str(path),'--csv-separator',','],
        inputFiles=[dict(path='input/data/original.csv',bytes=len(raw),sha256=digest(raw))])
    publish(f);return f


def test_retained_source_inspection_rechecks_input_and_no_extra_ceiling(source_operation):
    f=source_operation;f[1].update(executionBudget=dict(schemaVersion=1,requestedSeconds=86400,maximumRunSeconds=86400,source='application'),remainingSeconds=86400);publish(f)
    with claim(f) as authority:
        path=f[0]/'input/data/original.csv';before=path.stat();raw=path.read_bytes()
        path.write_bytes(b'x'+raw[1:]);os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
        with pytest.raises(ValueError):authority.check()


@pytest.mark.parametrize('change',['extra','duplicate','separator','missing-separator','escape','inventory','hash','bytes','bool-size','symlink','registry'])
def test_inspection_contract_rejects_before_claim(source_operation,change):
    f=source_operation;v=f[1]
    if change=='extra':v['arguments']+=['--output','/tmp/escape']
    elif change=='duplicate':v['arguments']+=['--inspect-data',v['arguments'][1]]
    elif change=='separator':v['arguments'][-1]='::'
    elif change=='missing-separator':del v['arguments'][-2:]
    elif change=='escape':v['arguments'][1]='/tmp/source.csv'
    elif change=='inventory':v['inputFiles'].append(dict(v['inputFiles'][0]))
    elif change=='hash':v['inputFiles'][0]['sha256']='0'*64
    elif change=='bytes':v['inputFiles'][0]['bytes']=10000001
    elif change=='bool-size':v['inputFiles'][0]['bytes']=True
    elif change=='registry':v['arguments']+=['--registry','/tmp/registry']
    else:
        path=Path(v['arguments'][1]);path.rename(path.with_name('old'));path.symlink_to('old')
    publish(f)
    with pytest.raises((ValueError,OSError)):claim(f)
    assert not (f[0]/'claim.json').exists()


@pytest.mark.skipif(sys.platform!='darwin',reason='Native macOS lifetime guard')
@pytest.mark.parametrize('separator',[',','\t',';','|'])
def test_actual_guarded_inspection_emits_exact_source_hash(source_operation,tmp_path,libraries,separator):
    f=source_operation;raw=(separator.join(['feature','label'])+'\n'+separator.join(['1','0'])+'\n').encode()
    Path(f[1]['arguments'][1]).write_bytes(raw);f[1]['arguments'][-1]=separator
    f[1]['inputFiles'][0].update(bytes=len(raw),sha256=digest(raw));publish(f)
    guard,_=libraries;path=entry(tmp_path,guard)
    marker='    print(json.dumps({"event":"engineMarker","arguments":result.arguments}),flush=True)'
    text=path.read_text();assert marker in text
    path.write_text(text.replace(marker,'    from main import main\n    sys.argv=[sys.argv[0],*result.arguments]\n    sys.exit(main(check_cancelled=result.authority.check))'))
    code,events,error=run_supervisor(path,control(f))
    assert code==0,(events,error)
    assert events[0]['toolMode']=='sourceInspection'
    assert events[-1]==dict(event='dataPreview',columns=['feature','label'],rows=[['1','0']],rowCount=1,sourceSHA256=digest(raw))


@pytest.mark.parametrize('boundary',[1,2])
def test_cancelled_inspection_does_not_return_source_evidence(tmp_path,boundary):
    from runner.data import inspect_data
    path=tmp_path/'source.csv';path.write_text('a,b\n1,2\n')
    calls=0
    def check():
        nonlocal calls
        calls+=1
        if calls==boundary:raise InterruptedError('cancelled source inspection')
    with pytest.raises(InterruptedError):inspect_data(path,check_cancelled=check)


def test_inspection_hash_uses_same_read_as_preview(tmp_path,monkeypatch):
    from runner import data
    path=tmp_path/'source.csv';raw=b'a,b\n1,2\n';path.write_bytes(raw)
    original=data.read_csv;calls=[]
    def read(*args,**kwargs):
        calls.append(True);result=original(*args,**kwargs);path.write_text('changed');return result
    monkeypatch.setattr(data,'read_csv',read)
    result=data.inspect_data(path)
    assert calls==[True] and result['sourceSHA256']==digest(raw) and result['rows']==[['1','2']]


def test_main_cancellation_before_emit_suppresses_inspection_event(tmp_path):
    import subprocess
    root=Path(__file__).resolve().parents[2]
    source=tmp_path/'source.csv';source.write_text('a,b\n1,2\n')
    script='''import sys
sys.path.insert(0,sys.argv[1])
import main
calls=0
def check():
 global calls
 calls+=1
 if calls>=4:raise InterruptedError('cancelled before emit')
sys.argv=['main','--inspect-data',sys.argv[2]]
sys.exit(main.main(check_cancelled=check))
'''
    result=subprocess.run([sys.executable,'-I','-c',script,str(root/'runner'),str(source)],capture_output=True,text=True,timeout=10)
    assert result.returncode==1,result.stdout+result.stderr
    events=[json.loads(line) for line in result.stdout.splitlines()]
    assert events and all(event['event']!='dataPreview' for event in events)


def test_options_contract_and_preview(source_operation):
    from runner.source_inspection_operation_contract import parse_contract
    f=source_operation;f[1]['arguments'][-2:]=['--csv-options',json.dumps({'separator':',','emptyField':'text'})]
    assert parse_contract(f[1],f[0])
    f[1]['arguments'][-1]=json.dumps({'separator':',','unknown':True})
    with pytest.raises(ValueError):parse_contract(f[1],f[0])
