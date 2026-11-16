"""Draft code runs as a bounded retained tool with one pinned graph input."""
import json
import os
import sys
import signal
from contextlib import nullcontext
from runner.tests.test_runner import preserved_cli_signal_handlers
import pytest
from runner.tests.test_operation_authorization import operation_authority,publish,claim
from runner.tests.test_evaluation_authorization import encoded,digest
from runner.tests.test_operation_bootstrap import control
from runner.tests.test_worker_bootstrap import entry,run_supervisor
from runner.tests.test_lifetime_watchdog import libraries


@pytest.fixture
def code_operation(operation_authority):
    from runner.tests.test_runner import graph
    f=operation_authority;value=f[1]
    for key in ('runID','directoryID','phase'):del value[key]
    inputs=f[0]/'input';inputs.mkdir(mode=0o700)
    raw=encoded(graph());(inputs/'graph.json').write_bytes(raw)
    value.update(schemaVersion=2,toolMode='codeExport',inputFiles=[dict(path='input/graph.json',bytes=len(raw),sha256=digest(raw))],
        arguments=['--export-code','--graph',str(inputs/'graph.json')])
    publish(f);return f


@pytest.fixture
def workflow_code_operation(code_operation):
    f=code_operation;value=f[1];inputs=f[0]/'input'
    (inputs/'graph.json').unlink()
    raw=encoded({'schemaVersion':1,'scope':'workflow'})
    (inputs/'workflow-code.json').write_bytes(raw)
    value['inputFiles']=[dict(path='input/workflow-code.json',bytes=len(raw),sha256=digest(raw))]
    value['arguments']=['--export-workflow-code','--workflow-code-plan',str(inputs/'workflow-code.json')]
    publish(f);return f


def test_exact_workflow_code_tool_claim_rechecks_snapshot(workflow_code_operation):
    f=workflow_code_operation
    with claim(f) as authority:
        authority.check()
        (f[0]/'input/workflow-code.json').write_bytes(b'{}')
        with pytest.raises(ValueError):authority.check()


@pytest.mark.parametrize('change',['graph-input','graph-flags','mixed-inputs','wrong-path','extra-flag','wrong-order','oversize','symlink','hardlink'])
def test_workflow_code_crossed_or_unpinned_inputs_never_claim(workflow_code_operation,change):
    f=workflow_code_operation;v=f[1];path=f[0]/'input/workflow-code.json'
    if change=='graph-input':v['inputFiles'][0]['path']='input/graph.json'
    elif change=='graph-flags':v['arguments']=['--export-code','--graph',str(path)]
    elif change=='mixed-inputs':v['inputFiles'].append(dict(v['inputFiles'][0],path='input/graph.json'))
    elif change=='wrong-path':v['arguments'][-1]='/tmp/workflow-code.json'
    elif change=='extra-flag':v['arguments']+=['--source-bindings','/tmp/private.json']
    elif change=='wrong-order':v['arguments']=['--workflow-code-plan',str(path),'--export-workflow-code']
    elif change=='oversize':v['inputFiles'][0]['bytes']=2_000_001
    elif change=='symlink':path.rename(path.with_name('alias'));path.symlink_to('alias')
    elif change=='hardlink':os.link(path,path.with_name('alias'))
    publish(f)
    with pytest.raises((ValueError,OSError)):claim(f)
    assert not (f[0]/'claim.json').exists()


def test_exact_code_tool_claim_rechecks_graph(code_operation):
    f=code_operation
    with claim(f) as authority:
        authority.check()
        (f[0]/'input/graph.json').write_bytes(b'{}')
        with pytest.raises(ValueError):authority.check()


@pytest.mark.parametrize('change',['argv-extra','argv-order','registry','wrong-path','wrong-inventory','extra-input','oversize','hash','symlink','hardlink','over-ceiling'])
def test_code_tool_invalid_contract_never_claims(code_operation,change):
    f=code_operation;v=f[1];path=f[0]/'input/graph.json'
    if change=='argv-extra':v['arguments']+=['--output','/tmp/escape']
    elif change=='argv-order':v['arguments']=['--graph',str(path),'--export-code']
    elif change=='registry':v['arguments']+=['--registry','/tmp/registry']
    elif change=='wrong-path':v['arguments'][-1]='/tmp/graph.json'
    elif change=='wrong-inventory':v['inputFiles'][0]['path']='input/assessment.json'
    elif change=='extra-input':v['inputFiles'].append(dict(v['inputFiles'][0]))
    elif change=='oversize':v['inputFiles'][0]['bytes']=2000001
    elif change=='hash':v['inputFiles'][0]['sha256']='0'*64
    elif change=='symlink':path.rename(path.with_name('alias'));path.symlink_to('alias')
    elif change=='hardlink':os.link(path,path.with_name('alias'))
    else:v.update(executionBudget=dict(schemaVersion=1,requestedSeconds=60,maximumRunSeconds=60,source='project'),remainingSeconds=30.01)
    publish(f)
    with pytest.raises((ValueError,OSError)):claim(f)
    assert not (f[0]/'claim.json').exists()


@pytest.mark.skipif(sys.platform!='darwin',reason='Native macOS lifetime guard')
def test_guarded_actual_code_generation_returns_compilable_source(code_operation,tmp_path,libraries):
    guard,_=libraries;path=entry(tmp_path,guard)
    marker='    print(json.dumps({"event":"engineMarker","arguments":result.arguments}),flush=True)'
    source=path.read_text();assert marker in source
    path.write_text(source.replace(marker,'    from main import main\n    sys.argv=[sys.argv[0],*result.arguments]\n    sys.exit(main(check_cancelled=result.authority.check))'))
    code,events,error=run_supervisor(path,control(code_operation))
    assert code==0,(events,error)
    assert events[0]['event']=='operationReady' and events[0]['toolMode']=='codeExport'
    assert events[-1]['event']=='codeExport'
    compile(events[-1]['code'],'model.py','exec')
    assert 'torch' in events[-1]['code']


@pytest.mark.skipif(sys.platform!='darwin',reason='Native macOS lifetime guard')
@pytest.mark.parametrize('finish',['deadline','cancel'])
@pytest.mark.parametrize('operation_name',['code_operation','workflow_code_operation'])
def test_code_worker_timeout_and_cancellation_release_activity(request,tmp_path,libraries,finish,operation_name):
    import fcntl
    import subprocess
    from runner.tests.test_worker_supervisor import line
    f=request.getfixturevalue(operation_name);f[1]['remainingSeconds']=3;publish(f)
    guard,_=libraries;path=entry(tmp_path,guard)
    source=path.read_text().replace('if isinstance(result,int):sys.exit(result)', '''if isinstance(result,int):
        import os
        try:os.waitpid(-1,os.WNOHANG)
        except ChildProcessError:print(json.dumps({"event":"reaped"}),flush=True)
        else:raise AssertionError('Unreaped owned worker')
        sys.exit(result)''')
    path.write_text(source.replace('except Exception as error:', '    import time\n    time.sleep(30)\nexcept Exception as error:'))
    process=subprocess.Popen([sys.executable,'-I','-B',str(path),*control(f)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,bufsize=0)
    try:
        assert json.loads(line(process))['toolMode']=='codeExport'
        assert json.loads(line(process))['event']=='engineMarker'
        with pytest.raises(BlockingIOError):fcntl.flock(f[3][1],fcntl.LOCK_EX|fcntl.LOCK_NB)
        if finish=='cancel':process.stdin.close()
        assert process.wait(timeout=10)==(124 if finish=='deadline' else 130)
        assert json.loads(line(process))['event']=='reaped'
        fcntl.flock(f[3][1],fcntl.LOCK_EX|fcntl.LOCK_NB)
    finally:
        if not process.stdin.closed:process.stdin.close()
        if process.poll() is None:process.terminate();process.wait(timeout=8)
        process.stdout.close();process.stderr.close()


def test_workflow_main_generates_only_from_retained_snapshot(workflow_code_operation,monkeypatch,capsys):
    import types
    from runner import main,operation_authorization
    f=workflow_code_operation
    monkeypatch.setitem(sys.modules,'operation_authorization',operation_authorization)
    received=[]
    def generate(snapshot,check_cancelled):
        check_cancelled();received.append(snapshot)
        return {'code':'print("workflow")\n','files':{'workflow.py':'print("workflow")\n'},'planDigest':'a'*64,'scope':'workflow'}
    monkeypatch.setitem(sys.modules,'runner.workflow_code',types.SimpleNamespace(generate_package=generate))
    monkeypatch.setattr(sys,'argv',['main.py',*f[1]['arguments']])
    with preserved_cli_signal_handlers() as previous:
        with claim(f) as authority:
            assert main.main(authority.check)==0
        assert all(signal.getsignal(number) is main.cancel for number in previous)
    assert all(signal.getsignal(number)==handler for number,handler in previous.items())
    event=json.loads(capsys.readouterr().out)
    assert event['event']=='codeExport' and event['files']['workflow.py']==event['code']
    assert received==[{'schemaVersion':1,'scope':'workflow'}]
    compile(event['code'],'workflow.py','exec')


@pytest.mark.parametrize('action',['tamper','cancel','systemExit'])
def test_workflow_main_rechecks_authority_and_cancellation_before_success(workflow_code_operation,monkeypatch,capsys,action):
    import types
    from runner import main,operation_authorization
    f=workflow_code_operation
    monkeypatch.setitem(sys.modules,'operation_authorization',operation_authorization)
    def generate(snapshot,check_cancelled):
        if action=='systemExit':
            assert all(signal.getsignal(number) is main.cancel for number in (signal.SIGINT,signal.SIGTERM))
            raise SystemExit(9)
        if action=='cancel':raise main.Cancelled('Run cancelled.')
        (f[0]/'input/workflow-code.json').write_bytes(b'{}')
        return {'code':'pass\n','files':{'workflow.py':'pass\n'}}
    monkeypatch.setitem(sys.modules,'runner.workflow_code',types.SimpleNamespace(generate_package=generate))
    monkeypatch.setattr(sys,'argv',['main.py',*f[1]['arguments']])
    # Catch SystemExit outside the borrowed signal scope: its finally must run
    # during exception propagation, not merely after a locally handled error.
    with (pytest.raises(SystemExit) if action=='systemExit' else nullcontext()) as error:
        with preserved_cli_signal_handlers() as previous:
            with claim(f) as authority:
                if action=='systemExit':main.main(authority.check)
                else:assert main.main(authority.check)==(130 if action=='cancel' else 1)
            assert all(signal.getsignal(number) is main.cancel for number in previous)
    if action=='systemExit':assert error.value.code==9
    assert all(signal.getsignal(number)==handler for number,handler in previous.items())
    output=capsys.readouterr().out
    if action=='systemExit':assert output==''
    else:assert json.loads(output)['event']=='error'


def test_workflow_main_requires_managed_authority(monkeypatch,capsys,tmp_path):
    from runner import main
    source=tmp_path/'private.json';source.write_text('{}')
    monkeypatch.setattr(sys,'argv',['main.py','--export-workflow-code','--workflow-code-plan',str(source)])
    with preserved_cli_signal_handlers() as previous:
        assert main.main()==1
        assert all(signal.getsignal(number) is main.cancel for number in previous)
    assert all(signal.getsignal(number)==handler for number,handler in previous.items())
    assert json.loads(capsys.readouterr().out)['event']=='error'
