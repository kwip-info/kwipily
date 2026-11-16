"""Ordinary work enters through retained supervisor before loading the engine."""
import sys
import pytest
from runner.tests.test_operation_authorization import operation_authority, publish
from runner.tests.test_evaluation_authorization import encoded, digest
from runner.tests.test_worker_bootstrap import entry, run_supervisor
from runner.tests.test_lifetime_watchdog import libraries


def control(f,mode='--kwip-operation-supervise'):
    return [mode,str(f[0]),digest(encoded(f[1])),digest(encoded(f[2]))]


@pytest.mark.skipif(sys.platform!='darwin',reason='Native macOS lifetime guard')
def test_operation_reexec_exact_cli_and_ready(operation_authority,tmp_path,libraries):
    f=operation_authority; watchdog,_=libraries
    code,events,errors=run_supervisor(entry(tmp_path,watchdog),control(f))
    assert code==0,(events,errors)
    assert events[0]==dict(event='operationReady',operationID=f[1]['operationID'],phase='preparation',
        requestSHA256=digest(encoded(f[1])),ownershipSHA256=digest(encoded(f[2])))
    assert [event['event'] for event in events]==['operationReady','engineMarker']
    assert events[1]['arguments']==f[1]['arguments']


@pytest.mark.parametrize('budgeted',[True,False])
def test_supervisor_receives_only_pinned_remaining_allowance(operation_authority,tmp_path,monkeypatch,budgeted):
    from runner import worker_bootstrap,worker_supervisor
    f=operation_authority
    if not budgeted:
        del f[1]['executionBudget'];del f[1]['remainingSeconds'];publish(f)
    calls=[]
    monkeypatch.setattr(worker_bootstrap,'_parent_revoked',lambda:False)
    monkeypatch.setattr(worker_supervisor,'supervise',lambda command,**kwargs:calls.append((command,kwargs)) or 124)
    assert worker_bootstrap.prepare(control(f),worker_bootstrap.RuntimeContext(tmp_path/'entry.py',tmp_path/'guard',None))==124
    assert calls[0][1]['max_wall_seconds']==(8.5 if budgeted else None)
    assert '--kwip-operation-worker' in calls[0][0]


@pytest.mark.parametrize('args',[['--kwip-operation-supervise'],['--kwip-operation-worker'],
    ['--capabilities','--kwip-operation-supervise'],['--kwip-operation-supervise=x']])
def test_operation_reserved_controls_reject(args,tmp_path):
    from runner.worker_bootstrap import prepare,RuntimeContext
    with pytest.raises(ValueError):prepare(args,RuntimeContext(tmp_path/'entry.py',tmp_path/'guard',None))


def test_closed_parent_never_claims(operation_authority,tmp_path,monkeypatch):
    from runner import worker_bootstrap
    monkeypatch.setattr(worker_bootstrap,'_parent_revoked',lambda:True)
    assert worker_bootstrap.prepare(control(operation_authority),worker_bootstrap.RuntimeContext(tmp_path/'entry.py',tmp_path/'guard',None))==130
    assert not (operation_authority[0]/'claim.json').exists()


@pytest.mark.skipif(sys.platform!='darwin',reason='Native macOS lifetime guard')
@pytest.mark.parametrize('finish',['deadline','cancel'])
def test_real_operation_supervisor_stops_reaps_and_releases_activity(operation_authority,tmp_path,libraries,finish):
    import fcntl
    import json
    import os
    import subprocess
    from runner.tests.test_worker_supervisor import line
    f=operation_authority;watchdog,_=libraries
    # Use ample startup time; the deadline test waits for the actual pinned limit.
    f[1]['remainingSeconds']=3;publish(f)
    path=entry(tmp_path,watchdog)
    source=path.read_text().replace('if isinstance(result,int):sys.exit(result)', '''if isinstance(result,int):
        import os
        try:os.waitpid(-1,os.WNOHANG)
        except ChildProcessError:print(json.dumps({"event":"reaped"}),flush=True)
        else:raise AssertionError('Supervisor returned with an unreaped child')
        sys.exit(result)''')
    source=source.replace('except Exception as error:', '    import time\n    time.sleep(30)\nexcept Exception as error:')
    path.write_text(source)
    process=subprocess.Popen([sys.executable,'-I','-B',str(path),*control(f)],stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,bufsize=0)
    try:
        assert json.loads(line(process))['event']=='operationReady'
        assert json.loads(line(process))['event']=='engineMarker'
        with pytest.raises(BlockingIOError):fcntl.flock(f[3][1],fcntl.LOCK_EX|fcntl.LOCK_NB)
        if finish=='cancel':process.stdin.close()
        assert process.wait(timeout=10)==(124 if finish=='deadline' else 130)
        assert json.loads(line(process))['event']=='reaped'
        fcntl.flock(f[3][1],fcntl.LOCK_EX|fcntl.LOCK_NB)
        if finish=='deadline':os.fstat(process.stdin.fileno())
    finally:
        if not process.stdin.closed:process.stdin.close()
        if process.poll() is None:process.terminate();process.wait(timeout=8)
        process.stdout.close();process.stderr.close()
