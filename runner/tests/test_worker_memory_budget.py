"""Sampled per-owned-worker resident RAM budget; not a GPU or whole-app cap."""
import json
import os
import signal
import sys
import subprocess
import pytest
from runner.tests.test_worker_supervisor import launch, line, cleanup, exclusive_available

pytestmark = pytest.mark.skipif(sys.platform != 'darwin', reason='macOS libproc resident memory')


def test_resident_reader_observes_live_process_and_refuses_invalid_pid():
    from runner.worker_memory import ResidentMemoryReader
    reader = ResidentMemoryReader()
    assert reader.read(os.getpid()) > 0
    with pytest.raises((ValueError, OSError)):
        reader.read(-1)


def test_memory_watchdog_terminates_reaps_stubborn_owned_worker(tmp_path):
    maximum = 128 * 1024 * 1024
    process, writer, lock = launch(tmp_path, 'memory', max_resident_bytes=maximum)
    try:
        assert json.loads(line(process))['ready']
        event = json.loads(line(process))
        assert event['event'] == 'error'
        assert event['code'] == 'residentMemoryLimitExceeded'
        assert event['maximumResidentBytes'] == maximum
        assert event['observedResidentBytes'] > maximum
        assert 'Increase' in event['message']
        assert process.wait(timeout=8) == 125
        assert line(process) == b'supervisor returned 125\n'
        assert line(process) == b'OWNED_CHILD_REAPED\n'
        assert exclusive_available(lock)
    finally:
        cleanup(process, writer)


def test_memory_budget_preserves_normal_exit_and_parent_cancellation(tmp_path):
    for mode in ['normal', 'stubborn']:
        folder = tmp_path / mode; folder.mkdir()
        process, writer, lock = launch(folder, mode, max_resident_bytes=128*1024*1024)
        try:
            assert json.loads(line(process))['ready']
            if mode == 'normal':
                assert line(process) == b'normal output\n'
                expected = 7
            else:
                os.close(writer); writer = None; expected = 130
            assert process.wait(timeout=8) == expected
            assert line(process) == f'supervisor returned {expected}\n'.encode()
            assert line(process) == b'OWNED_CHILD_REAPED\n'
            assert exclusive_available(lock)
        finally:
            cleanup(process, writer)


@pytest.mark.parametrize('limit', [True,False,0,-1,134217727,1099511627777,134217728.0,'134217728',float('nan')])
def test_invalid_memory_budget_refuses_before_spawn(tmp_path, limit):
    from runner.worker_supervisor import supervise
    reader, writer = os.pipe()
    lease = os.open(tmp_path/'lease', os.O_CREAT|os.O_RDWR, 0o600)
    marker = tmp_path/'never'
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        with pytest.raises(ValueError):
            supervise([sys.executable,'-c',f'open({str(marker)!r},"w").close()'], parent_fd=reader, activity_fd=lease, max_resident_bytes=limit)
        assert not marker.exists()
        assert handlers == {sig: signal.getsignal(sig) for sig in handlers}
    finally:
        for fd in (reader, writer, lease): os.close(fd)


def test_unavailable_observation_stops_and_reaps_instead_of_disabling_budget(tmp_path, monkeypatch, capsys):
    from runner.worker_supervisor import supervise
    from runner.worker_memory import ResidentMemoryReader
    def unavailable(self, pid):
        raise OSError('injected observation failure')
    monkeypatch.setattr(ResidentMemoryReader, 'read', unavailable)
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'])
    original_popen = subprocess.Popen
    owned = []
    def record_owned(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        owned.append(process)
        return process
    monkeypatch.setattr(subprocess, 'Popen', record_owned)
    reader, writer = os.pipe()
    lease = os.open(tmp_path/'lease', os.O_CREAT|os.O_RDWR, 0o600)
    try:
        result = supervise([sys.executable,'-c','import time;time.sleep(30)'],
                           parent_fd=reader, activity_fd=lease, grace_seconds=.05,
                           max_resident_bytes=128*1024*1024)
        assert result == 125
        output = capsys.readouterr()
        event = json.loads(output.out)
        assert event['code'] == 'residentMemoryMonitoringFailed'
        assert event['observedResidentBytes'] is None
        assert 'could not be observed' in output.err
        assert len(owned) == 1 and owned[0].returncode is not None
        with pytest.raises(ChildProcessError): os.waitpid(owned[0].pid, os.WNOHANG)
        assert unrelated.poll() is None  # Reaping one worker must not target other children.
    finally:
        unrelated.terminate(); unrelated.wait(timeout=5)
        for fd in (reader, writer, lease): os.close(fd)
