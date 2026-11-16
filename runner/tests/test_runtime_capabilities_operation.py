"""Runtime discovery uses the same retained authority as other workflow tools."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import pytest
from runner.tests.test_operation_authorization import operation_authority,publish,claim
from runner.tests.test_operation_bootstrap import control
from runner.tests.test_worker_bootstrap import entry,run_supervisor
from runner.tests.test_lifetime_watchdog import libraries


@pytest.fixture
def discovery(operation_authority):
    f=operation_authority;v=f[1]
    for key in ('runID','directoryID','phase'):del v[key]
    v.update(schemaVersion=2,toolMode='runtimeCapabilities',arguments=['--capabilities'],inputFiles=[])
    publish(f);return f


def test_discovery_policy_has_no_new_ceiling_and_owner_revocation_is_checked(discovery):
    f=discovery
    f[1].update(executionBudget=dict(schemaVersion=1,requestedSeconds=86400,maximumRunSeconds=86400,source='application'),remainingSeconds=86400);publish(f)
    with claim(f) as authority:
        authority.check()
        fcntl.flock(f[3][0],fcntl.LOCK_UN)
        with pytest.raises(ValueError):authority.check()


@pytest.mark.parametrize('change',['extra','duplicate','registry','input','missing','other'])
def test_invalid_runtime_discovery_cannot_claim(discovery,change):
    f=discovery;v=f[1]
    if change=='extra':v['arguments']+=['--device','cpu']
    elif change=='duplicate':v['arguments']+=['--capabilities']
    elif change=='registry':v['arguments']+=['--registry','/tmp/registry']
    elif change=='input':v['inputFiles']=[dict(path='input/extra',bytes=0,sha256='0'*64)]
    elif change=='missing':v['arguments']=[]
    else:v['arguments']=['--dataflow-registry']
    publish(f)
    with pytest.raises(ValueError):claim(f)
    assert not (f[0]/'claim.json').exists()


@pytest.mark.skipif(sys.platform!='darwin',reason='Native macOS lifetime guard')
@pytest.mark.parametrize('timeout',[False,True])
def test_guarded_runtime_discovery_or_captured_timeout(discovery,tmp_path,libraries,timeout):
    f=discovery;guard,_=libraries;path=entry(tmp_path,guard)
    marker='    print(json.dumps({"event":"engineMarker","arguments":result.arguments}),flush=True)'
    text=path.read_text();assert marker in text
    code='    from main import main\n    sys.argv=[sys.argv[0],*result.arguments]\n    sys.exit(main(check_cancelled=result.authority.check))'
    if timeout:
        f[1]['remainingSeconds']=1;publish(f)
        code='    import time\n    time.sleep(30)\n'+code
    path.write_text(text.replace(marker,code))
    status,events,error=run_supervisor(path,control(f))
    assert status==(124 if timeout else 0),(events,error)
    assert events[0]['toolMode']=='runtimeCapabilities'
    if timeout:assert not any(event['event']=='capabilities' for event in events)
    else:
        assert [event['event'] for event in events]==['operationReady','capabilities']
        assert 'cpu' in events[-1]['devices'] and events[-1]['torchVersion']
        assert events[-1]['executionContract'] and events[-1]['componentCapabilities']
    fcntl.flock(f[3][1],fcntl.LOCK_EX|fcntl.LOCK_NB)


@pytest.mark.parametrize('boundary',[1,2])
def test_discovery_checks_cancellation_before_compute_and_publication(tmp_path,boundary):
    root=Path(__file__).resolve().parents[2]
    script='''import sys
sys.path.insert(0,sys.argv[1])
import engine,main
calls=0
computed=False
def capabilities():
 global computed
 computed=True
 return {'event':'capabilities'}
engine.capabilities=capabilities
boundary=sys.argv[2]
sys.argv=['main','--capabilities']
def retained_check():
 global calls
 calls+=1
 if calls==int(boundary):raise InterruptedError('cancelled discovery')
status=main.main(check_cancelled=retained_check)
assert computed==(int(boundary)==2)
sys.exit(status)
'''
    result=subprocess.run([sys.executable,'-I','-c',script,str(root/'runner'),str(boundary)],capture_output=True,text=True,timeout=10)
    assert result.returncode==1,result.stdout+result.stderr
    events=[json.loads(line) for line in result.stdout.splitlines()]
    assert events and all(event['event']!='capabilities' for event in events)
