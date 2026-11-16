"""Diagnostic preparation gate must hold real work and require a matching resume."""
import importlib.util
from pathlib import Path
import sys
import types
import uuid

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'recovery_uat_snapshot.py'


def load_overlay():
    spec = importlib.util.spec_from_file_location('recovery_uat_snapshot', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gate(tmp_path, monkeypatch, *, arm=None, resume=None):
    sys.path.insert(0, str(SCRIPT.parents[1] / 'runner'))
    from dataflow_storage import bounded_regular_read
    namespace = {'Path': Path, 'bounded_regular_read': bounded_regular_read}
    exec(load_overlay().PREPARATION_GATE, namespace)
    clock = types.SimpleNamespace(now=0)
    clock.monotonic = lambda: clock.now
    def sleep(seconds):
        clock.now += 100
    clock.sleep = sleep
    monkeypatch.setitem(sys.modules, 'time', clock)
    if arm is not None:
        (tmp_path / 'recovery-uat-arm').write_bytes(arm)
    if resume is not None:
        (tmp_path / 'recovery-uat-resume').write_bytes(resume)
    events = []
    return namespace['_recovery_uat_pause'], events, clock


def test_unarmed_does_not_pause(tmp_path, monkeypatch):
    pause, events, clock = gate(tmp_path, monkeypatch)
    pause(tmp_path / 'artifacts', 'node', events.append)
    assert events == [] and clock.now == 0


def test_matching_nonce_resumes_and_emits_identity(tmp_path, monkeypatch):
    nonce = str(uuid.uuid4()).upper().encode()
    pause, events, clock = gate(tmp_path, monkeypatch, arm=nonce, resume=nonce)
    pause(tmp_path / 'artifacts', 'source', events.append)
    assert events == [{'event': 'recoveryUATPaused', 'stage': 'duringPreparation', 'nodeID': 'source', 'nonce': nonce.decode()}]
    assert clock.now == 0


def test_wrong_nonce_times_out_without_continuing(tmp_path, monkeypatch):
    pause, events, clock = gate(tmp_path, monkeypatch, arm=str(uuid.uuid4()).encode(), resume=str(uuid.uuid4()).encode())
    with pytest.raises(RuntimeError, match='timed out'):
        pause(tmp_path / 'artifacts', 'source', events.append)
    assert len(events) == 1 and clock.now == 300


def test_symlink_arm_rejected(tmp_path, monkeypatch):
    pause, events, clock = gate(tmp_path, monkeypatch)
    other = tmp_path / 'other'
    other.write_bytes(str(uuid.uuid4()).encode())
    (tmp_path / 'recovery-uat-arm').symlink_to(other)
    with pytest.raises(ValueError):
        pause(tmp_path / 'artifacts', 'source', events.append)
    assert not events


def test_overlay_is_fail_closed_and_after_materialization():
    module = load_overlay()
    source = (SCRIPT.parents[1] / 'runner/dataflow_cli.py').read_text()
    result = module.overlay_worker(source)
    assert 'emit(event)\n        if not recovery_uat_consumed:' in result
    assert result.count('            _recovery_uat_pause(output, node_id, emit)') == 1
    with pytest.raises(ValueError):
        module.overlay_worker(result)
    with pytest.raises(ValueError):
        module.overlay_worker(source.replace('        emit(event)', '        emit(dict(event))'))


def test_real_preparation_stages_bytes_before_pause_and_only_pauses_once(tmp_path):
    root = SCRIPT.parents[1]
    sys.path.insert(0, str(root))
    worker = root / 'runner/dataflow_cli.py'
    namespace = {'__file__': str(worker), '__package__': 'runner'}
    exec(load_overlay().overlay_worker(worker.read_text()), namespace)
    nonce = str(uuid.uuid4()).upper().encode()
    (tmp_path / 'recovery-uat-arm').write_bytes(nonce)
    output = tmp_path / 'artifacts'
    events = []
    def receive(event):
        events.append(event)
        if event['event'] == 'recoveryUATPaused':
            assert events[-2]['event'] == 'dataflowPreview'
            staged = list(tmp_path.glob('.dataflow-*/node-*.json'))
            assert staged and staged[0].stat().st_size > 0
            assert not (output / 'manifest.json').exists()
            (tmp_path / 'recovery-uat-resume').write_bytes(nonce)
    namespace['run_dataflow'](root / 'examples/dataflow/scaler.json',
        {'source': str(root / 'examples/dataflow/observations.csv')}, output, receive)
    assert sum(e['event'] == 'recoveryUATPaused' for e in events) == 1
    assert events[-1]['event'] == 'dataflowCompleted'
    assert (output / 'manifest.json').exists()


def test_snapshot_identifies_diagnostics_without_changing_executable(tmp_path, monkeypatch):
    import io
    import tarfile
    import plistlib
    root = SCRIPT.parents[1]
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode='w') as archive:
        for relative in ('runner/dataflow_cli.py', 'scripts/build-app.sh'):
            archive.add(root / relative, arcname=relative)
    module = load_overlay()
    def git(args, **kwargs):
        if args[1] == 'status': return b''
        if args[1] == 'rev-parse': return 'abc123\n'
        if args[1] == 'archive': return payload.getvalue()
        raise AssertionError(args)
    monkeypatch.setattr(module.subprocess, 'check_output', git)
    destination = tmp_path / 'snapshot'
    record = module.create_snapshot(destination)
    build = (destination / 'scripts/build-app.sh').read_text()
    plist = plistlib.loads(build.split("<<'PLIST'\n", 1)[1].split('\nPLIST', 1)[0].encode())
    assert plist['CFBundleExecutable'] == 'KwipML'
    import json
    name = json.loads((root / 'Sources/KwipMLCore/Resources/ProductBrand.json').read_text())['name']
    assert plist['CFBundleDisplayName'] == name + ' Recovery UAT'
    assert 'Set :CFBundleDisplayName $PRODUCT_DISPLAY_NAME Recovery UAT' in build
    assert 'Set :CFBundleName $PRODUCT_DISPLAY_NAME Recovery UAT' in build
    assert 'Set :CFBundleDisplayName $PRODUCT_DISPLAY_NAME Development' not in build
    assert build.count('swift build -Xswiftc -DKWIPML_RECOVERY_UAT') == 2
    assert record['baselineRevision'] == 'abc123'


def test_build_overlay_refuses_missing_or_ambiguous_display_identity():
    module = load_overlay()
    source = (SCRIPT.parents[1] / 'scripts/build-app.sh').read_text()
    import json
    name = json.loads((SCRIPT.parents[1] / 'Sources/KwipMLCore/Resources/ProductBrand.json').read_text())['name']
    anchor = f'<key>CFBundleDisplayName</key><string>{name} Development</string>'
    for broken in (source.replace(anchor, ''), source + '\n' + anchor):
        with pytest.raises(ValueError, match='display identity'):
            module.overlay_build(broken)


@pytest.mark.parametrize('key', ['CFBundleName', 'CFBundleDisplayName'])
def test_build_overlay_refuses_missing_or_ambiguous_brand_setter(key):
    module = load_overlay()
    source = (SCRIPT.parents[1] / 'scripts/build-app.sh').read_text()
    setter = f'Set :{key} $PRODUCT_DISPLAY_NAME Development'
    assert source.count(setter) == 1
    for broken in (source.replace(setter, ''), source + '\n' + setter):
        with pytest.raises(ValueError, match='display identity'):
            module.overlay_build(broken)
    with pytest.raises(ValueError, match='build command|display identity'):
        module.overlay_build(module.overlay_build(source))
