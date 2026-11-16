"""Synthetic gate tests cannot qualify a real signed application."""
import copy
import json
from pathlib import Path
import pytest
from scripts import packaged_boundary as boundary


@pytest.fixture
def candidate(tmp_path):
    app = tmp_path / 'Kwip ML.app'
    app.mkdir()
    (app / 'binary').write_bytes(b'synthetic executable')
    archive = tmp_path / 'app-notary.zip'
    archive.write_bytes(b'synthetic archive')
    release = dict(coordinates={'build': 99, 'sourceCommit': 'test', 'runtimeManifestSHA256': 'runtime'},
                   appZipSHA256=boundary.digest(archive), packagedBoundaryRequired=True, assembledAppTreeSHA256=boundary.tree_digest(app))
    folder = tmp_path / 'packaged-boundary'
    folder.mkdir()
    evidence = folder / 'observation.txt'
    evidence.write_text('Synthetic observation, not actual UAT')
    value = dict(schemaVersion=1, coordinates=release['coordinates'], appZipSHA256=release['appZipSHA256'],
                 appTreeSHA256=boundary.tree_digest(app), cases={name: dict(result='passed', observation='Synthetic',
                 evidence=evidence.name, sha256=boundary.digest(evidence)) for name in boundary.CASES})
    (folder / 'observations.json').write_text(json.dumps(value))
    return tmp_path, app, release, folder, value


def test_gate_binds_evidence_to_signed_payload(candidate):
    out, app, release, folder, value = candidate
    assert boundary.require_gate(out, release) == boundary.digest(folder / 'observations.json')
    (app / 'binary').write_bytes(b'changed')
    with pytest.raises(ValueError, match='payload changed'):
        boundary.require_gate(out, release)


@pytest.mark.parametrize('mutation', ['missing', 'failed', 'empty', 'tampered', 'outside', 'symlink', 'coordinates'])
def test_incomplete_or_invalid_native_evidence_refuses(candidate, mutation):
    out, app, release, folder, value = candidate
    first = boundary.CASES[0]
    if mutation == 'missing': value['cases'].pop(first)
    if mutation == 'failed': value['cases'][first]['result'] = 'failed'
    if mutation == 'empty': value['cases'][first]['observation'] = ''
    if mutation == 'tampered': (folder / 'observation.txt').write_text('Changed')
    if mutation == 'outside': value['cases'][first]['evidence'] = '../app-notary.zip'
    if mutation == 'symlink':
        (folder / 'link').symlink_to(folder / 'observation.txt')
        value['cases'][first]['evidence'] = 'link'
    if mutation == 'coordinates': value['coordinates'] = {'build': 98}
    with pytest.raises(ValueError): boundary.validate(value, release, app, folder)


def test_modes_and_links_are_part_of_identity(candidate):
    out, app, release, folder, value = candidate
    (app / 'binary').chmod(0o700)
    assert boundary.tree_digest(app) != value['appTreeSHA256']
    before = boundary.tree_digest(app)
    (app / 'alias').symlink_to('binary')
    assert boundary.tree_digest(app) != before


def test_archive_tampering_and_absent_receipt_fail_closed(candidate):
    out, app, release, folder, value = candidate
    (out / 'app-notary.zip').write_bytes(b'changed')
    with pytest.raises(ValueError, match='archive changed'): boundary.require_gate(out, release)
    (folder / 'observations.json').unlink()
    with pytest.raises(ValueError): boundary.require_gate(out, release)


def test_legacy_candidate_keeps_original_contract(tmp_path):
    assert boundary.require_gate(tmp_path, {'phase': 'appSubmitted'}) is None


def test_notary_cannot_submit_new_candidate_before_gate(candidate, monkeypatch):
    from scripts import direct_release
    out, app, release, folder, value = candidate
    release['phase'] = 'assembled'
    (folder / 'observations.json').unlink()
    calls = []
    monkeypatch.setattr(direct_release.subprocess, 'run', lambda *a, **k: calls.append(a))
    with pytest.raises(ValueError): direct_release.notary(None, out, release)
    assert calls == []


def test_observation_cannot_rebind_modified_app_to_original_archive(candidate):
    out, app, release, folder, value = candidate
    (app / 'binary').write_bytes(b'altered before starting observations')
    value['appTreeSHA256'] = boundary.tree_digest(app)
    with pytest.raises(ValueError): boundary.validate(value, release, app, folder)


def test_phase_events_do_not_grow_for_unchanged_apple_poll(tmp_path):
    from scripts.direct_release import save
    path = tmp_path / 'release.json'
    receipt = {'phase': 'appSubmitted'}
    save(path, receipt)
    first = copy.deepcopy(receipt['phaseEvents'])
    save(path, receipt)
    assert receipt['phaseEvents'] == first
    receipt['phase'] = 'appAccepted'
    save(path, receipt)
    assert [event['phase'] for event in receipt['phaseEvents']] == ['appSubmitted', 'appAccepted']


def test_real_cli_retains_observations_and_rejects_tampering(candidate):
    import subprocess
    import sys
    out, app, release, folder, value = candidate
    release['phase'] = 'assembled'
    (out / 'release.json').write_text(json.dumps(release))
    (folder / 'observations.json').unlink()
    command = [sys.executable, str(Path(boundary.__file__)), '--output', str(out)]
    subprocess.run([*command, 'start'], check=True, capture_output=True)
    incomplete = subprocess.run([*command, 'verify'], capture_output=True)
    assert incomplete.returncode != 0
    for name in boundary.CASES:
        subprocess.run([*command, 'record', '--case', name, '--evidence', 'observation.txt',
                        '--result', 'passed', '--observation', 'Synthetic CLI test; not real UAT'], check=True, capture_output=True)
    subprocess.run([*command, 'verify'], check=True, capture_output=True)
    duplicate = subprocess.run([*command, 'record', '--case', boundary.CASES[0], '--evidence', 'observation.txt',
                                '--result', 'passed', '--observation', 'overwrite'], capture_output=True)
    assert duplicate.returncode != 0
    (folder / 'observation.txt').write_text('Modified after record')
    assert subprocess.run([*command, 'verify'], capture_output=True).returncode != 0


def test_operation_events_retain_failure_without_private_exception(tmp_path):
    from scripts.release_operation_events import measured
    path = tmp_path / 'operations.jsonl'
    ticks = iter([1.0, 3.0, 5.0, 8.0])
    assert measured(path, 'compute', 'test', lambda: 42, clock=lambda: next(ticks)) == 42
    def fail(): raise ValueError('secret-looking private subprocess detail')
    with pytest.raises(ValueError): measured(path, 'provider', 'test-failure', fail, clock=lambda: next(ticks))
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert [e['event'] for e in events] == ['started','completed','started','failed']
    assert [e.get('elapsedSeconds') for e in events] == [None,2,None,3]
    assert events[0]['id'] == events[1]['id'] != events[2]['id'] == events[3]['id']
    assert 'secret-looking' not in path.read_text()


def test_toolchain_context_is_scoped_and_cannot_drift(monkeypatch):
    from types import SimpleNamespace
    from scripts.direct_release import notary_environment
    monkeypatch.setenv('DEVELOPER_DIR', '/sealed/watchdog/compiler')
    args = SimpleNamespace(notary_developer_dir='/existing/notary/context')
    receipt = {'toolchainContexts': {'notaryDeveloperDir': '/existing/notary/context'}}
    env = notary_environment(args, receipt)
    assert env['DEVELOPER_DIR'] == '/existing/notary/context'
    import os
    assert os.environ['DEVELOPER_DIR'] == '/sealed/watchdog/compiler'
    args.notary_developer_dir = '/other/context'
    with pytest.raises(ValueError): notary_environment(args, receipt)


def test_signed_diagnostic_can_never_submit_or_become_candidate(tmp_path, monkeypatch):
    from scripts import direct_release, release_candidate
    receipt = {'diagnosticOnly': True, 'phase': 'assembled'}
    calls = []
    monkeypatch.setattr(direct_release.subprocess, 'run', lambda *a, **k: calls.append(a))
    with pytest.raises(ValueError, match='Diagnostic-only'): direct_release.notary(None, tmp_path, receipt)
    with pytest.raises(ValueError, match='Diagnostic-only'): release_candidate.binding({}, receipt, '')
    assert calls == []


def test_signed_diagnostic_marker_survives_receipt_flag_removal(tmp_path, monkeypatch):
    import plistlib
    from scripts import direct_release
    info = tmp_path / 'Kwip ML.app/Contents/Info.plist'
    info.parent.mkdir(parents=True)
    info.write_bytes(plistlib.dumps({'KwipMLDiagnosticOnly': True}))
    calls = []
    monkeypatch.setattr(direct_release.subprocess, 'run', lambda *a, **k: calls.append(a))
    with pytest.raises(ValueError, match='Diagnostic-only'):
        direct_release.notary(None, tmp_path, {'phase': 'assembled'})
    assert calls == []
