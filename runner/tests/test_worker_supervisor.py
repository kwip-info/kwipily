"""Actual supervisor/worker lifetime and inherited lease behavior on macOS."""
import fcntl
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time

import pytest
from runner.tests.test_lifetime_watchdog import libraries

pytestmark = pytest.mark.skipif(sys.platform != 'darwin', reason='macOS supervised worker contract')
ROOT = Path(__file__).resolve().parents[2]


def line(process, timeout=8):
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ)
        assert selector.select(timeout), 'Missing subprocess handshake'
        value = process.stdout.readline()
        assert value, process.stderr.read().decode() if process.poll() is not None else 'Unexpected pipe EOF'
        return value


def launch(tmp_path, mode, native=None, *, anchored=False, max_wall_seconds=None, max_resident_bytes=None):
    child = tmp_path / 'child.py'
    child.write_text('''
import argparse,ctypes,json,os,signal,sys,time
p=argparse.ArgumentParser();p.add_argument('--kwip-lifetime-fd',type=int);p.add_argument('--kwip-activity-fd',type=int)
a=p.parse_args();mode=os.environ['FIXTURE_MODE']
assert a.kwip_activity_fd>2
if mode in ('stubborn','memory'):signal.signal(signal.SIGTERM,signal.SIG_IGN)
if mode=='native':
 lib=ctypes.PyDLL(os.environ['WATCHDOG']);arm=lib.kwip_lifetime_arm;arm.argtypes=[ctypes.c_int];arm.restype=ctypes.c_int
 assert arm(a.kwip_lifetime_fd)==0
print(json.dumps({'pid':os.getpid(),'ready':True}),flush=True)
if mode=='memory':allocation=bytearray(192*1024*1024)
if mode=='normal':print('normal output',flush=True);sys.exit(7)
if mode=='sleeping':time.sleep(30)
if mode=='native':ctypes.PyDLL(os.environ['BLOCKER']).hold_gil_forever()
else:
 while True:signal.pause()
''')
    script = tmp_path / 'supervisor.py'
    script.write_text('''
import os,sys
sys.path.insert(0,sys.argv[1])
from worker_supervisor import supervise
parent,activity=int(sys.argv[3]),int(sys.argv[4])
deadline=os.environ.get('FIXTURE_DEADLINE')
options={} if deadline is None else {'max_wall_seconds':float(deadline)}
memory=os.environ.get('FIXTURE_MEMORY')
if memory is not None:options['max_resident_bytes']=int(memory)
code=supervise([sys.executable,'-I',sys.argv[2]],parent_fd=parent,activity_fd=activity,grace_seconds=.15,**options)
os.fstat(parent);os.fstat(activity)
print('supervisor returned '+str(code),flush=True)
if deadline is not None or memory is not None:
 try:os.waitpid(-1,os.WNOHANG)
 except ChildProcessError:print('OWNED_CHILD_REAPED',flush=True)
 else:raise AssertionError('Supervisor returned with an unreaped child')
sys.exit(code)
''')
    parent_reader, parent_writer = os.pipe()
    lock = tmp_path / 'activity.lock'; activity = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(activity, fcntl.LOCK_SH)
    environment = dict(os.environ, FIXTURE_MODE=mode)
    environment.pop('FIXTURE_DEADLINE', None)
    environment.pop('FIXTURE_MEMORY', None)
    if max_resident_bytes is not None: environment['FIXTURE_MEMORY'] = str(max_resident_bytes)
    if max_wall_seconds is not None: environment['FIXTURE_DEADLINE'] = str(max_wall_seconds)
    if native: environment.update(WATCHDOG=str(native[0]), BLOCKER=str(native[1]))
    command = [sys.executable, '-I', str(script), str(ROOT / 'runner'), str(child),
               str(parent_reader), str(activity)]
    if anchored:
        # Retained session leader owns the supervisor Popen and anchors all group
        # signalling even after supervisor death; no orphan PID is signalled.
        broker = tmp_path / 'broker.py'
        broker.write_text('''
import os,signal,subprocess,sys
signal.signal(signal.SIGALRM,lambda *_:os.killpg(os.getpgrp(),signal.SIGKILL))
signal.alarm(20)
parent,activity=map(int,sys.argv[1:3])
worker=subprocess.Popen(sys.argv[3:],pass_fds=(parent,activity),stdin=subprocess.DEVNULL)
os.close(parent);os.close(activity)
try:
 for line in sys.stdin:
  if line.strip()=='KILL_SUPERVISOR':
   worker.kill();assert worker.wait(timeout=5)==-signal.SIGKILL
   print('SUPERVISOR_KILLED',flush=True)
  elif line.strip()=='RESUME':os.killpg(os.getpgrp(),signal.SIGCONT)
finally:
 os.killpg(os.getpgrp(),signal.SIGKILL)
''')
        command = [sys.executable, '-I', str(broker), str(parent_reader), str(activity), *command]
    process = subprocess.Popen(command, pass_fds=(parent_reader, activity),
                               stdin=subprocess.PIPE if anchored else None,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
                               env=environment, start_new_session=True)
    os.close(parent_reader); os.close(activity)
    return process, parent_writer, lock


def exclusive_available(path):
    descriptor = os.open(path, os.O_RDWR)
    try:
        try: fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: return False
        return True
    finally: os.close(descriptor)


def cleanup(process, writer):
    if writer is not None: os.close(writer)
    if process.poll() is None:
        # Test-only session contains only this test's supervisor and its one child.
        os.killpg(process.pid, signal.SIGKILL)
    process.communicate(timeout=8)


def test_normal_child_output_and_exit_are_preserved(tmp_path):
    process, writer, lock = launch(tmp_path, 'normal')
    try:
        assert json.loads(line(process))['ready']
        assert line(process) == b'normal output\n'
        assert line(process) == b'supervisor returned 7\n'
        assert process.wait(timeout=8) == 7
        assert exclusive_available(lock)
    finally: cleanup(process, writer)


@pytest.mark.parametrize('deadline', [5, 86400])
def test_deadline_preserves_fast_child_exit_and_reaps_it(tmp_path, deadline):
    process, writer, lock = launch(tmp_path, 'normal', max_wall_seconds=deadline)
    try:
        assert json.loads(line(process))['ready']
        assert line(process) == b'normal output\n'
        assert line(process) == b'supervisor returned 7\n'
        assert line(process) == b'OWNED_CHILD_REAPED\n'
        assert process.wait(timeout=8) == 7
        assert exclusive_available(lock)
    finally: cleanup(process, writer)


@pytest.mark.parametrize('mode', ['sleeping', 'stubborn'])
def test_deadline_stops_and_reaps_real_owned_worker_with_distinct_exit(tmp_path, mode):
    process, writer, lock = launch(tmp_path, mode, max_wall_seconds=1)
    try:
        assert json.loads(line(process))['ready']
        assert not exclusive_available(lock)
        assert process.wait(timeout=8) == 124
        assert line(process) == b'supervisor returned 124\n'
        assert line(process) == b'OWNED_CHILD_REAPED\n'
        assert exclusive_available(lock), 'Deadline returned before actual worker exit released the lease'
        os.fstat(writer)  # Parent lifetime was not revoked to manufacture cancellation.
    finally: cleanup(process, writer)


@pytest.mark.parametrize('trigger', ['eof', 'signal'])
def test_cancellation_before_deadline_retains_cancellation_exit(tmp_path, trigger):
    process, writer, lock = launch(tmp_path, 'stubborn', max_wall_seconds=5)
    try:
        assert json.loads(line(process))['ready']
        if trigger == 'eof': os.close(writer); writer = None
        else: process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=8) == 130
        assert line(process) == b'supervisor returned 130\n'
        assert line(process) == b'OWNED_CHILD_REAPED\n'
        assert exclusive_available(lock)
    finally: cleanup(process, writer)


@pytest.mark.parametrize('deadline', [0, -1, 86400.01, True, False, float('nan'), float('inf'), -float('inf'), '1', [], {}])
def test_invalid_deadline_rejects_before_spawn_and_preserves_caller_state(tmp_path, deadline):
    from runner.worker_supervisor import supervise
    reader, writer = os.pipe()
    lease = os.open(tmp_path / 'lease', os.O_CREAT | os.O_RDWR, 0o600)
    marker = tmp_path / 'must-not-launch'
    before = descriptor_snapshot()
    handlers = {name: signal.getsignal(name) for name in (signal.SIGTERM, signal.SIGINT)}
    try:
        with pytest.raises(ValueError):
            supervise([sys.executable, '-c', f'from pathlib import Path;Path({str(marker)!r}).touch()'],
                      parent_fd=reader, activity_fd=lease, max_wall_seconds=deadline)
        assert not marker.exists()
        assert descriptor_snapshot() == before
        assert {name: signal.getsignal(name) for name in handlers} == handlers
    finally:
        for descriptor in (reader, writer, lease): os.close(descriptor)


@pytest.mark.parametrize('trigger', ['eof', 'bytes', 'signal'])
def test_parent_loss_escalates_stubborn_owned_worker_and_releases_lease(tmp_path, trigger):
    process, writer, lock = launch(tmp_path, 'stubborn')
    try:
        assert json.loads(line(process))['ready']
        assert not exclusive_available(lock)
        if trigger == 'eof': os.close(writer); writer = None
        elif trigger == 'bytes': os.write(writer, b'unexpected')
        else: process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=8) == 130
        assert exclusive_available(lock), 'Supervisor returned before worker released inherited lease'
        assert line(process) == b'supervisor returned 130\n'
    finally: cleanup(process, writer)


def test_supervisor_death_preserves_child_lock_until_native_guard_exits(libraries, tmp_path):
    # Stop from inside the GIL-held call itself; the test never signals a cached
    # worker PID. SIGCONT goes to a group anchored by the retained broker Popen.
    source, blocker = tmp_path / 'stopped.c', tmp_path / 'stopped.dylib'
    source.write_text('#include <unistd.h>\n#include <signal.h>\n'
                      'void hold_gil_forever(void) { (void)write(1,"GIL_HELD\\n",9); '
                      'raise(SIGSTOP); for (;;) pause(); }\n')
    compiled = subprocess.run(['xcrun', 'clang', '-dynamiclib', '-Wall', '-Wextra', '-Werror',
                               '-mmacosx-version-min=14.0', str(source), '-o', str(blocker)],
                              capture_output=True, text=True, timeout=30)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    process, writer, lock = launch(tmp_path, 'native', (libraries[0], blocker), anchored=True)
    try:
        child = json.loads(line(process))['pid']
        assert line(process) == b'GIL_HELD\n'
        # PID is used only to observe the child stopping, never for signalling.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            observed = subprocess.run(['ps', '-o', 'stat=', '-p', str(child)], capture_output=True, text=True, timeout=2)
            if 'T' in observed.stdout: break
            time.sleep(.01)
        else: pytest.fail('Owned child did not stop')
        process.stdin.write(b'KILL_SUPERVISOR\n'); process.stdin.flush()
        assert line(process) == b'SUPERVISOR_KILLED\n'
        assert not exclusive_available(lock), 'Supervisor death prematurely released worker authority'
        process.stdin.write(b'RESUME\n'); process.stdin.flush()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if exclusive_available(lock): break
            time.sleep(.01)
        else: pytest.fail('Native EOF guard did not release child lease while GIL was held')
    finally:
        cleanup(process, writer)


def test_invalid_descriptors_and_grace_reject_before_launch(tmp_path):
    # Import belongs inside the test: missing module is honest tests-first red.
    from runner.worker_supervisor import supervise
    reader, writer = os.pipe()
    activity = os.open(tmp_path / 'regular', os.O_CREAT | os.O_RDWR, 0o600)
    marker = tmp_path / 'executed'
    command = [sys.executable, '-c', f'from pathlib import Path;Path({str(marker)!r}).touch()']
    try:
        for parent, lease, grace in [(-1, activity, 1), (writer, activity, 1),
                                     (activity, activity, 1), (reader, writer, 1),
                                     (reader, 2, 1), (reader, activity, -1),
                                     (reader, activity, 11), (reader, activity, float('nan')),
                                     (reader, activity, float('inf'))]:
            with pytest.raises((ValueError, OSError)):
                supervise(command, parent_fd=parent, activity_fd=lease, grace_seconds=grace)
        os.set_blocking(reader, False)
        with pytest.raises((ValueError, OSError)):
            supervise(command, parent_fd=reader, activity_fd=activity)
        assert not marker.exists()
        for fd in [reader, writer, activity]: os.fstat(fd)
    finally:
        for fd in [reader, writer, activity]: os.close(fd)


def test_graceful_guarded_worker_acknowledges_before_lifetime_writer_closes(libraries, tmp_path):
    watchdog, _ = libraries
    child = tmp_path / 'graceful.py'
    child.write_text('''
import argparse,ctypes,signal,time
p=argparse.ArgumentParser();p.add_argument('--kwip-lifetime-fd',type=int);p.add_argument('--kwip-activity-fd',type=int);a=p.parse_args()
lib=ctypes.PyDLL(%r);arm=lib.kwip_lifetime_arm;arm.argtypes=[ctypes.c_int];arm.restype=ctypes.c_int
assert arm(a.kwip_lifetime_fd)==0
def finish(signum,frame):
 time.sleep(.05)
 print('GRACEFUL_ACK',flush=True)
 raise SystemExit(0)
signal.signal(signal.SIGTERM,finish)
print('READY',flush=True)
while True:signal.pause()
''' % str(watchdog))
    script = '''
import sys
sys.path.insert(0,sys.argv[1]);from worker_supervisor import supervise
sys.exit(supervise([sys.executable,'-I',sys.argv[2]],parent_fd=int(sys.argv[3]),activity_fd=int(sys.argv[4]),grace_seconds=1))
'''
    reader, writer = os.pipe()
    lease = os.open(tmp_path / 'lease', os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(lease, fcntl.LOCK_SH)
    process = subprocess.Popen([sys.executable, '-I', '-c', script, str(ROOT / 'runner'), str(child), str(reader), str(lease)],
                               pass_fds=(reader, lease), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               bufsize=0, start_new_session=True)
    os.close(reader); os.close(lease)
    try:
        assert line(process) == b'READY\n'
        os.close(writer); writer = None
        assert line(process) == b'GRACEFUL_ACK\n'
        assert process.wait(timeout=8) == 130
    finally: cleanup(process, writer)


def descriptor_snapshot():
    # Ignore the transient descriptor used to enumerate /dev/fd itself.
    result = {}
    for name in os.listdir('/dev/fd'):
        try:
            descriptor = int(name)
            info = os.fstat(descriptor)
            result[descriptor] = (info.st_dev, info.st_ino, info.st_mode)
        except OSError:
            pass
    return result


def test_preexisting_eof_prevents_launch_and_restores_handlers(tmp_path):
    from runner.worker_supervisor import supervise
    reader, writer = os.pipe(); os.close(writer)
    lease = os.open(tmp_path / 'lease', os.O_CREAT | os.O_RDWR, 0o600)
    marker = tmp_path / 'must-not-launch'
    before = descriptor_snapshot()
    handlers = {name: signal.getsignal(name) for name in (signal.SIGTERM, signal.SIGINT)}
    try:
        result = supervise([sys.executable, '-c', f'from pathlib import Path;Path({str(marker)!r}).touch()'],
                           parent_fd=reader, activity_fd=lease)
        assert result == 130
        assert not marker.exists()
        assert descriptor_snapshot() == before
        assert {name: signal.getsignal(name) for name in handlers} == handlers
    finally: os.close(reader); os.close(lease)


def test_spawn_failure_closes_created_fds_and_preserves_callers_and_handlers(tmp_path):
    from runner.worker_supervisor import supervise
    reader, writer = os.pipe()
    lease = os.open(tmp_path / 'lease', os.O_CREAT | os.O_RDWR, 0o600)
    before = descriptor_snapshot()
    handlers = {name: signal.getsignal(name) for name in (signal.SIGTERM, signal.SIGINT)}
    try:
        with pytest.raises(FileNotFoundError):
            supervise([str(tmp_path / 'missing-executable')], parent_fd=reader, activity_fd=lease)
        for descriptor in (reader, writer, lease): os.fstat(descriptor)
        assert descriptor_snapshot() == before
        assert {name: signal.getsignal(name) for name in handlers} == handlers
    finally:
        for descriptor in (reader, writer, lease): os.close(descriptor)


def test_normal_return_restores_custom_signal_handlers(tmp_path):
    from runner.worker_supervisor import supervise
    reader, writer = os.pipe()
    lease = os.open(tmp_path / 'lease', os.O_CREAT | os.O_RDWR, 0o600)
    previous = {name: signal.getsignal(name) for name in (signal.SIGTERM, signal.SIGINT)}
    def custom(_signum, _frame):
        raise AssertionError('Unexpected test signal')
    before = descriptor_snapshot()
    try:
        for name in previous: signal.signal(name, custom)
        assert supervise([sys.executable, '-c', 'raise SystemExit(9)'], parent_fd=reader, activity_fd=lease) == 9
        assert all(signal.getsignal(name) is custom for name in previous)
        assert descriptor_snapshot() == before
    finally:
        for name, handler in previous.items(): signal.signal(name, handler)
        for descriptor in (reader, writer, lease): os.close(descriptor)


@pytest.mark.parametrize('max_wall_seconds', [None, 5.0])
def test_observed_cancellation_wins_over_simultaneously_observed_zero_exit(tmp_path, monkeypatch, max_wall_seconds):
    from runner import worker_supervisor
    reader, writer = os.pipe()
    lease = os.open(tmp_path / 'lease', os.O_CREAT | os.O_RDWR, 0o600)
    original_popen = subprocess.Popen
    handlers = {name: signal.getsignal(name) for name in (signal.SIGTERM, signal.SIGINT)}
    before = descriptor_snapshot()
    children = []

    def completed_child_with_cancellation(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        children.append(child)
        assert child.wait(timeout=5) == 0
        original_poll = child.poll
        observed = False

        def poll():
            nonlocal observed
            if not observed:
                observed = True
                # Deliver the real handler while supervise observes terminal exit.
                signal.raise_signal(signal.SIGTERM)
            return original_poll()
        child.poll = poll
        return child

    monkeypatch.setattr(worker_supervisor.subprocess, 'Popen', completed_child_with_cancellation)
    try:
        result = worker_supervisor.supervise([sys.executable, '-c', 'raise SystemExit(0)'],
                                             parent_fd=reader, activity_fd=lease, max_wall_seconds=max_wall_seconds)
        assert result == 130, 'Already-observed cancellation must not become successful completion'
        assert len(children) == 1 and children[0].returncode == 0
        assert {name: signal.getsignal(name) for name in handlers} == handlers
        assert descriptor_snapshot() == before
    finally:
        for child in children:
            if child.returncode is None: child.kill()
            child.wait(timeout=5)
        for descriptor in (reader, writer, lease): os.close(descriptor)
