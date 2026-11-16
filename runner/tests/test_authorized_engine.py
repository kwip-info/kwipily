"""Actual data preparation through immutable launch authority and the native guard."""
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import pytest
from runner.tests.test_worker_authorization import launch_fixture
from runner.tests.test_worker_bootstrap import control

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(platform.system() != 'Darwin' or platform.machine() != 'arm64',
                    reason='Native arm64 macOS watchdog')
def test_verified_packaged_entry_prepares_real_data(launch_fixture, tmp_path):
    spec = importlib.util.spec_from_file_location('authorized_engine_package', ROOT/'scripts/runtime_package.py')
    package = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(package)
    payload = tmp_path/'runtime'
    payload.mkdir()
    package.copy_worker_sources(payload)
    hashes = package.worker_source_hashes(payload)
    (payload/'runtime-manifest.json').write_text(json.dumps({
        'schemaVersion': 1, 'workerIntegrityVersion': 1, 'workerSourceHashes': hashes,
    }))
    _, intent, _, _ = launch_fixture
    request = Path(intent['arguments'][1])
    request.write_bytes((ROOT/'examples/dataflow/scaler.json').read_bytes())
    request.with_name('bindings.json').write_text(json.dumps({
        'source': str(ROOT/'examples/dataflow/observations.csv'),
    }))
    reader, writer = os.pipe()
    process = None
    try:
        process = subprocess.Popen([sys.executable, '-I', '-B', str(payload/'launch.py'),
                                    *control(launch_fixture)], stdin=reader,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        os.close(reader)
        reader = None
        # communicate drains both streams while our separate lifetime writer stays open.
        output, errors = process.communicate(timeout=45)
        events = [json.loads(line) for line in output.splitlines()]
        assert process.returncode == 0, (events, errors)
        assert events[0] == {'event': 'launchReady', 'nonce': intent['nonce'], 'phase': 'preparation'}
        assert events[1]['event'] == 'dataflowStarted'
        assert events[-1]['event'] == 'dataflowCompleted'
        manifest = json.loads((request.parent/'artifacts/manifest.json').read_text())
        assert manifest['status'] == 'completed'
        assert len([event for event in events if event['event'] == 'dataflowPreview']) == 8
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
        if reader is not None:
            os.close(reader)
        os.close(writer)
