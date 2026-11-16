"""Launch routing must establish authority and native lifetime before engine imports."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

import pytest
from runner.tests.test_worker_authorization import launch_fixture, digest, encoded, claim
from runner.tests.test_lifetime_watchdog import libraries

ROOT = Path(__file__).resolve().parents[2]


def control(fixture, mode='--kwip-supervise'):
    folder, intent, _, _ = fixture
    return [mode, str(folder), intent['nonce'], digest(encoded(intent)), intent['ownershipSHA256']]


def test_ordinary_cli_does_not_claim_or_load_native(tmp_path):
    from runner.worker_bootstrap import RuntimeContext, prepare
    assert prepare(['--capabilities'], RuntimeContext(tmp_path/'main.py', tmp_path/'missing.dylib', None)) is None


@pytest.mark.parametrize('arguments', [
    ['--kwip-supervise'], ['--kwip-worker'], ['--kwip-unknown'],
    ['--capabilities', '--kwip-worker'], ['--kwip-supervise=x'],
])
def test_reserved_controls_fail_closed(tmp_path, arguments):
    from runner.worker_bootstrap import RuntimeContext, prepare
    with pytest.raises(ValueError):
        prepare(arguments, RuntimeContext(tmp_path/'main.py', tmp_path/'missing.dylib', None))


def entry(tmp_path, watchdog, registry=None):
    path = tmp_path/'trusted_entry.py'
    path.write_text(f'''
import sys,json
from pathlib import Path
sys.path.insert(0,{str(ROOT/'runner')!r})
from worker_bootstrap import RuntimeContext, PreparedWorker, prepare
try:
    result=prepare(sys.argv[1:],RuntimeContext(Path(__file__),Path({str(watchdog)!r}),{('Path('+repr(str(registry))+')') if registry else 'None'}))
    if isinstance(result,int):sys.exit(result)
    assert isinstance(result,PreparedWorker)
    print(json.dumps({{"event":"engineMarker","arguments":result.arguments}}),flush=True)
except Exception as error:
    print(json.dumps({{"event":"error","message":str(error)}}),flush=True)
    sys.exit(1)
''')
    return path


def run_supervisor(path, args, *, timeout=15):
    process = subprocess.Popen([sys.executable,'-I','-B',str(path),*args], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    captured = {}
    def drain(name, stream):
        captured[name] = stream.read()
    readers = [threading.Thread(target=drain, args=(name, stream))
               for name, stream in [('output', process.stdout), ('errors', process.stderr)]]
    for reader in readers: reader.start()
    try:
        # Drain both pipes concurrently, retaining the parent lifetime writer.
        # A capabilities event can exceed one pipe buffer; waiting first deadlocks it.
        process.wait(timeout=timeout)
        for reader in readers: reader.join(timeout=5)
        assert not any(reader.is_alive() for reader in readers)
        output=captured['output']; errors=captured['errors']
        return process.returncode,[json.loads(line) for line in output.splitlines()],errors
    finally:
        if process.poll() is None:process.kill();process.wait(timeout=5)
        for reader in readers: reader.join(timeout=5)
        process.stdin.close();process.stdout.close();process.stderr.close()


@pytest.mark.skipif(sys.platform!='darwin',reason='Native macOS lifetime library')
def test_authorized_reexec_emits_ready_before_engine_and_uses_captured_args(launch_fixture,tmp_path,libraries):
    watchdog,_=libraries
    registry=tmp_path/'components.json';registry.write_text('{}')
    code,events,errors=run_supervisor(entry(tmp_path,watchdog,registry),control(launch_fixture))
    assert code==0, (events,errors)
    assert [event['event'] for event in events]==['launchReady','engineMarker']
    assert events[0]['nonce']==launch_fixture[1]['nonce']
    assert events[0]['phase']=='preparation'
    assert events[1]['arguments']==launch_fixture[1]['arguments']+['--registry',str(registry)]


@pytest.mark.parametrize('state',['invalid','revoked','unclaimed'])
def test_invalid_authority_never_reaches_engine(launch_fixture,tmp_path,state):
    path=entry(tmp_path,tmp_path/'missing.dylib')
    args=control(launch_fixture)
    if state=='invalid':args[3]='0'*64
    elif state=='revoked':
        folder,intent,_,_=launch_fixture
        (folder/f"claim-{intent['nonce']}.json").write_bytes(encoded(dict(schemaVersion=1,nonce=intent['nonce'],intentSHA256=args[3],state='revoked')))
    else:
        args[0]='--kwip-worker'
        args+=['--kwip-lifetime-fd','50','--kwip-activity-fd','51']
    code,events,errors=run_supervisor(path,args)
    assert code==1,(events,errors)
    assert [event['event'] for event in events]==['error']


def test_missing_guard_fails_before_ready_or_engine(launch_fixture,tmp_path):
    code,events,errors=run_supervisor(entry(tmp_path,tmp_path/'missing.dylib'),control(launch_fixture))
    assert code==1,(events,errors)
    assert [event['event'] for event in events]==['error']
    assert 'lifetime' in events[0]['message'].lower() or 'guard' in events[0]['message'].lower()


def test_actual_main_rejects_internal_mode_before_engine_import(tmp_path):
    for name in ('main.py','worker_bootstrap.py','worker_authorization.py','worker_supervisor.py','execution_budget.py'):
        shutil.copy2(ROOT/'runner'/name,tmp_path/name)
    marker=tmp_path/'engine-imported'
    (tmp_path/'engine.py').write_text(f'from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError("engine import forbidden")\n')
    process=subprocess.run([sys.executable,'-I','-B',str(tmp_path/'main.py'),'--kwip-unknown'],capture_output=True,text=True,timeout=10)
    assert process.returncode==1
    assert not marker.exists()
    assert json.loads(process.stdout)['event']=='error'
    assert 'Traceback' not in process.stderr


@pytest.mark.parametrize('state',['missing','stale'])
def test_development_attestation_rejects_guard_before_loading(launch_fixture,tmp_path,state):
    from runner.worker_bootstrap import RuntimeContext, prepare
    source=tmp_path/'guard.c';source.write_text('canonical source')
    library=tmp_path/'guard.dylib';library.write_bytes(b'not a library')
    manifest=tmp_path/'guard.json'
    if state=='stale':
        manifest.write_bytes(encoded(dict(schemaVersion=1,sourceSHA256='0'*64,librarySHA256=digest(library.read_bytes()))))
    authority=claim(launch_fixture)
    reader,writer=os.pipe()
    try:
        args=control(launch_fixture,'--kwip-worker')+['--kwip-lifetime-fd',str(reader),'--kwip-activity-fd',str(authority.activity_fd)]
        context=RuntimeContext(tmp_path/'main.py',library,None,native_manifest=manifest,native_source=source)
        with pytest.raises((ValueError,OSError),match='(?i)(manifest|attestation|digest|source|no such file)'):
            prepare(args,context)
    finally:
        for fd in (reader,writer):
            try:os.close(fd)
            except OSError:pass
        authority.close()


@pytest.mark.skipif(sys.platform!='darwin' or __import__('platform').machine()!='arm64',
                    reason='Canonical payload packaging requires arm64 macOS')
@pytest.mark.parametrize('mutate_before_reentry',[False,True])
def test_packaged_entry_revalidates_inventory_on_authorized_child_reentry(
        launch_fixture,tmp_path,mutate_before_reentry):
    """Use the real packager/launcher and a sealed marker entry, without importing ML."""
    import importlib.util
    spec=importlib.util.spec_from_file_location('bootstrap_test_package',ROOT/'scripts/runtime_package.py')
    package=importlib.util.module_from_spec(spec);spec.loader.exec_module(package)
    payload=tmp_path/'payload';payload.mkdir()
    package.copy_worker_sources(payload)
    # The marker entry is itself inventoried. Its test-only supervisor hook mutates
    # a covered resource after initial validation but before the child exec.
    (payload/'runner/main.py').write_text(f'''
import json,sys
from pathlib import Path
from worker_bootstrap import RuntimeContext,PreparedWorker,prepare
import worker_supervisor
context=RuntimeContext(**_KWIP_RUNTIME_CONTEXT)
original_supervise=worker_supervisor.supervise
def wrapped_supervise(command,**kwargs):
    if {mutate_before_reentry!r}:
        context.registry.write_text('{{"tampered":true}}')
    return original_supervise(command,**kwargs)
worker_supervisor.supervise=wrapped_supervise
try:
    result=prepare(sys.argv[1:],context)
    if isinstance(result,int):sys.exit(result)
    assert isinstance(result,PreparedWorker)
    print(json.dumps({{"event":"engineMarker","arguments":result.arguments}}),flush=True)
except Exception as error:
    print(json.dumps({{"event":"error","message":str(error)}}),flush=True)
    sys.exit(1)
''')
    (payload/'runtime-manifest.json').write_text(json.dumps({
        'schemaVersion':1,'workerIntegrityVersion':1,
        'workerSourceHashes':package.worker_source_hashes(payload),
    }))
    code,events,errors=run_supervisor(payload/'launch.py',control(launch_fixture))
    if mutate_before_reentry:
        assert code==1,(events,errors)
        assert [event['event'] for event in events]==['error']
        assert 'integrity' in events[0]['message'].lower()
    else:
        assert code==0,(events,errors)
        assert [event['event'] for event in events]==['launchReady','engineMarker']
        assert events[0]['nonce']==launch_fixture[1]['nonce']
        assert events[1]['arguments']==launch_fixture[1]['arguments']+['--registry',str(payload/'components.json')]
