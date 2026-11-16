"""Local release orchestration; fakes never establish native release acceptance."""
import copy
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]


def module():
    spec = importlib.util.spec_from_file_location('release_candidate', ROOT / 'scripts/release_candidate.py')
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def manifest():
    return {'schemaVersion': 1, 'version': '1.1.0', 'build': 12, 'sourceCommit': 'a' * 40,
            'runtimeManifestSHA256': 'b' * 64, 'minimumSystemVersion': '14.0',
            'architecture': 'arm64', 'notesReviewed': True,
            'changes': ['Explain the actual shipped change.'],
            'compatibility': ['All paid v1 licenses remain valid.'], 'knownLimits': []}


def test_manifest_requires_reviewed_versioned_notes_and_same_paid_major():
    mod = module()
    assert mod.validate_manifest(manifest()) == manifest()
    for key, value in [('version', '2.0.0'), ('build', True), ('notesReviewed', False),
                       ('changes', []), ('architecture', 'x86_64'),
                       ('sourceCommit', 'main'), ('minimumSystemVersion', '13.0')]:
        with pytest.raises(ValueError):
            mod.validate_manifest({**manifest(), key: value})
    with pytest.raises(ValueError):
        mod.validate_manifest({**manifest(), 'unknown': 'value'})


def test_baseline_must_be_real_scope_exact_commit_with_logs(tmp_path):
    mod = module()
    path = tmp_path / 'PASSED.json'
    receipt = {'commit': 'a' * 40, 'completedAt': '2026-10-11T16:00:00+00:00',
               'scope': mod.BASELINE_SCOPE, 'hostedResult': False,
               'nativeUAT': False, 'releaseAcceptance': False}
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        mod.validate_baseline(path, 'a' * 40)
    (tmp_path / 'output.log').write_text('local baseline log\n')
    assert mod.validate_baseline(path, 'a' * 40)['commit'] == 'a' * 40
    for bad in [{**receipt, 'commit': 'c' * 40}, {**receipt, 'scope': 'focused subset'},
                {**receipt, 'hostedResult': True}, {**receipt, 'completedAt': 'bad'}]:
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            mod.validate_baseline(path, 'a' * 40)


def test_updates_require_monotonic_build_and_no_version_regression():
    mod = module()
    old = {'phase': 'verified', 'coordinates': {'version': '1.0.0', 'build': 11}}
    mod.validate_previous(manifest(), old)
    for coordinates in [{'version': '1.0.0', 'build': 12}, {'version': '1.2.0', 'build': 10}]:
        with pytest.raises(ValueError):
            mod.validate_previous(manifest(), {**old, 'coordinates': coordinates})
    with pytest.raises(ValueError):
        mod.validate_previous(manifest(), {**old, 'phase': 'assembled'})


def test_resume_polls_existing_submission_and_stops_without_resubmitting(tmp_path):
    mod = module()
    state = {'phase': 'appSubmitted', 'appSubmissionID': 'existing-id'}
    calls = []
    def step(name):
        calls.append((name, copy.deepcopy(state)))
    assert mod.drive(lambda: copy.deepcopy(state), step, wait_seconds=0)['phase'] == 'appSubmitted'
    assert calls == [('notary', state)]


def test_driver_advances_all_native_phases_without_publication():
    mod = module()
    state = {'phase': 'assembled'}
    phases = iter(['appSubmitted', 'appAccepted', 'dmgBuilt', 'dmgSubmitted', 'verified'])
    calls = []
    def step(name):
        calls.append(name)
        state['phase'] = next(phases)
    assert mod.drive(lambda: copy.deepcopy(state), step, wait_seconds=0)['phase'] == 'verified'
    assert calls == ['notary'] * 5


def test_driver_refuses_uncertain_upload_and_nonadvancing_nonwaiting_phase():
    mod = module()
    calls = []
    with pytest.raises(ValueError):
        mod.drive(lambda: {'phase': 'assembled', 'pendingUpload': 'app'}, calls.append, wait_seconds=0)
    assert calls == []
    with pytest.raises(ValueError):
        mod.drive(lambda: {'phase': 'appAccepted'}, calls.append, wait_seconds=0)
    assert calls == ['notary']


def test_bounded_wait_polls_same_submission_and_sleeps_at_most_sixty_seconds():
    mod = module()
    clock = [0]
    state = {'phase': 'dmgSubmitted', 'dmgSubmissionID': 'same-id'}
    calls, sleeps = [], []
    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds
    result = mod.drive(lambda: state.copy(), calls.append, wait_seconds=12, poll_seconds=5,
                       now=lambda: clock[0], sleep=sleep)
    assert result == state and sleeps == [5, 5, 2]
    assert calls == ['notary'] * 4


def test_candidate_binding_refuses_notes_or_artifact_substitution():
    mod = module()
    value = manifest()
    coordinates = {key: value[key] for key in ['version', 'build', 'sourceCommit', 'runtimeManifestSHA256']}
    native = {'coordinates': coordinates, 'phase': 'verified', 'publicLaunch': False,
              'dmgSHA256': 'd' * 64, 'dmgBytes': 10}
    binding = mod.binding(value, native, 'e' * 64)
    assert binding['notes'] == {key: value[key] for key in ['changes', 'compatibility', 'knownLimits']}
    assert binding['published'] is False and binding['nativeUAT'] is False
    with pytest.raises(ValueError):
        mod.binding(value, {**native, 'coordinates': {**coordinates, 'build': 13}}, 'e' * 64)
    for key, changed in [('artifactSHA256', 'c' * 64), ('artifactBytes', 11), ('phase', 'assembled')]:
        with pytest.raises(ValueError):
            mod.validate_binding(binding, {**binding, key: changed})
    for key, changed in [('dmgSHA256', None), ('dmgSHA256', 'not-a-hash'), ('dmgBytes', True), ('dmgBytes', 0)]:
        with pytest.raises(ValueError):
            mod.binding(value, {**native, key: changed}, 'e' * 64)


def test_lock_refuses_concurrent_release_operator(tmp_path):
    mod = module()
    with mod.release_lock(tmp_path / 'candidate.lock'):
        with pytest.raises(ValueError):
            with mod.release_lock(tmp_path / 'candidate.lock'):
                pass


def test_notes_export_is_deterministic_and_names_the_exact_version_build():
    mod = module()
    notes = mod.render_notes(manifest())
    assert notes.startswith('Kwip ML 1.1.0 (build 12)\n')
    assert 'Changes\n- Explain the actual shipped change.' in notes
    assert 'Compatibility\n- All paid v1 licenses remain valid.' in notes
    assert notes == mod.render_notes(copy.deepcopy(manifest()))


def test_output_refuses_escape_symlink_and_parent_traversal(tmp_path, monkeypatch):
    mod = module()
    monkeypatch.setattr(mod, 'ROOT', tmp_path)
    allowed = tmp_path / '.local/direct-releases'
    allowed.mkdir(parents=True)
    assert mod.output_path(allowed / 'v1-12') == allowed / 'v1-12'
    (allowed / 'link').symlink_to(tmp_path, target_is_directory=True)
    for path in [tmp_path / 'outside', allowed, allowed / 'link/build', allowed / 'other/../v1-12']:
        with pytest.raises(ValueError):
            mod.output_path(path)


def test_cli_plan_checks_receipts_but_never_assembles_or_writes_output(tmp_path, monkeypatch):
    mod = module()
    monkeypatch.setattr(mod, 'ROOT', tmp_path)
    monkeypatch.setattr(mod, 'require_source', lambda commit: None)
    data = manifest()
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(data))
    baseline = tmp_path / 'PASSED.json'
    baseline.write_text(json.dumps({'commit': data['sourceCommit'], 'completedAt': '2026-10-11T16:00:00+00:00',
        'scope': mod.BASELINE_SCOPE, 'hostedResult': False, 'nativeUAT': False, 'releaseAcceptance': False}))
    (tmp_path / 'output.log').write_text('actual local log')
    old = tmp_path / 'previous.json'
    old.write_text(json.dumps({'phase': 'verified', 'coordinates': {'version': '1.0.0', 'build': 11}}))
    out = tmp_path / '.local/direct-releases/v1-build12'
    calls = []
    monkeypatch.setattr(mod.subprocess, 'run', lambda command, **kwargs: calls.append(command))
    monkeypatch.setattr(mod.sys, 'argv', ['release_candidate.py', '--manifest', str(manifest_path),
        '--baseline', str(baseline), '--previous-release', str(old), '--output', str(out),
        '--runtime', str(tmp_path / 'runtime'), '--profile', str(tmp_path / 'profile'), '--identity', 'D' * 40])
    assert mod.main() == 0
    assert len(calls) == 1 and calls[0][-2:] == ['--step', 'plan']
    assert calls[0][calls[0].index('--version') + 1] == '1.1.0'
    assert not out.exists()


def test_cli_prepares_once_then_resumes_same_artifact_and_refuses_changed_notes(tmp_path, monkeypatch):
    mod = module()
    monkeypatch.setattr(mod, 'ROOT', tmp_path)
    monkeypatch.setattr(mod, 'require_source', lambda commit: None)
    data = manifest()
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(data))
    baseline = tmp_path / 'PASSED.json'
    baseline.write_text(json.dumps({'commit': data['sourceCommit'], 'completedAt': '2026-10-11T16:00:00+00:00',
        'scope': mod.BASELINE_SCOPE, 'hostedResult': False, 'nativeUAT': False, 'releaseAcceptance': False}))
    (tmp_path / 'output.log').write_text('actual local log')
    old = tmp_path / 'previous.json'
    old.write_text(json.dumps({'phase': 'verified', 'coordinates': {'version': '1.0.0', 'build': 11}}))
    out = tmp_path / '.local/direct-releases/v1-build12'
    profile = tmp_path / 'profile'
    profile.write_bytes(b'synthetic approved profile')
    calls = []
    phases = iter(['appSubmitted', 'appAccepted', 'dmgBuilt', 'dmgSubmitted', 'verified'])
    def process(command, **kwargs):
        step = command[-1]
        calls.append(step)
        if step == 'assemble':
            out.mkdir()
            receipt = {'phase': 'assembled', 'coordinates': {key: data[key] for key in mod.COORDINATES},
                       'publicLaunch': False, 'identity': 'D' * 40,
                       'profileSHA256': mod.sha256(profile)}
        else:
            receipt = json.loads((out / 'release.json').read_text())
            if step == 'notary':
                receipt['phase'] = next(phases)
                if receipt['phase'] == 'verified':
                    receipt.update(dmgSHA256='d' * 64, dmgBytes=100)
        (out / 'release.json').write_text(json.dumps(receipt))
    monkeypatch.setattr(mod.subprocess, 'run', process)
    monkeypatch.setattr(mod.sys, 'argv', ['release_candidate.py', '--manifest', str(manifest_path),
        '--baseline', str(baseline), '--previous-release', str(old), '--output', str(out),
        '--runtime', str(tmp_path / 'runtime'), '--profile', str(tmp_path / 'profile'), '--identity', 'D' * 40,
        '--step', 'advance'])
    assert mod.main() == 0 and calls == ['assemble'] + ['notary'] * 5
    saved = json.loads((out / 'candidate.json').read_text())
    assert saved['artifactSHA256'] == 'd' * 64 and saved['phase'] == 'verified'
    assert (out / 'release-notes.txt').read_text() == mod.render_notes(data)
    assert mod.main() == 0 and calls[-1] == 'verify' and calls.count('assemble') == 1
    manifest_path.write_text(json.dumps({**data, 'changes': ['Substituted notes.']}))
    count = len(calls)
    assert mod.main() == 1 and len(calls) == count


def test_source_requires_exact_clean_head_and_refuses_staging_configuration(monkeypatch):
    mod = module()
    class Result:
        def __init__(self, value):
            self.stdout = value
    state = {'head': 'a' * 40, 'dirty': ''}
    monkeypatch.setattr(mod.subprocess, 'run', lambda command, **kwargs:
                        Result(state['dirty'] if command[-1] == '--porcelain' else state['head']))
    monkeypatch.delenv('KWIPML_COMMERCE_STAGING', raising=False)
    mod.require_source('a' * 40)
    with pytest.raises(ValueError):
        mod.require_source('c' * 40)
    state['dirty'] = ' M changed.swift'
    with pytest.raises(ValueError):
        mod.require_source('a' * 40)
    state['dirty'] = ''
    monkeypatch.setenv('KWIPML_COMMERCE_STAGING', '1')
    with pytest.raises(ValueError):
        mod.require_source('a' * 40)


def resumed_cli(tmp_path, monkeypatch):
    """An already submitted candidate; no native/provider commands are real."""
    import hashlib
    mod = module()
    monkeypatch.setattr(mod, 'ROOT', tmp_path)
    monkeypatch.setattr(mod, 'require_source', lambda commit: None)
    data = manifest()
    manifest_path = tmp_path / 'manifest.json'; manifest_path.write_text(json.dumps(data))
    baseline = tmp_path / 'PASSED.json'
    baseline.write_text(json.dumps({'commit': data['sourceCommit'], 'completedAt': '2026-10-11T16:00:00+00:00',
        'scope': mod.BASELINE_SCOPE, 'hostedResult': False, 'nativeUAT': False, 'releaseAcceptance': False}))
    (tmp_path / 'output.log').write_text('actual local log')
    previous = tmp_path / 'previous.json'
    previous.write_text(json.dumps({'phase': 'verified', 'coordinates': {'version': '1.0.0', 'build': 11}}))
    profile = tmp_path / 'profile'; profile.write_bytes(b'synthetic approved profile')
    out = tmp_path / '.local/direct-releases/v1-build12'; out.mkdir(parents=True)
    native = {'phase': 'appSubmitted', 'appSubmissionID': 'existing-apple-id',
              'coordinates': {key: data[key] for key in mod.COORDINATES}, 'publicLaunch': False,
              'identity': 'D' * 40, 'profileSHA256': hashlib.sha256(profile.read_bytes()).hexdigest()}
    (out / 'release.json').write_text(json.dumps(native))
    saved = mod.binding(data, native, mod.sha256(baseline))
    (out / 'candidate.json').write_text(json.dumps(saved))
    argv = ['release_candidate.py', '--manifest', str(manifest_path), '--baseline', str(baseline),
            '--previous-release', str(previous), '--output', str(out), '--runtime', str(tmp_path / 'runtime'),
            '--profile', str(profile), '--identity', 'D' * 40, '--step', 'advance']
    monkeypatch.setattr(mod.sys, 'argv', argv)
    calls = []
    monkeypatch.setattr(mod.subprocess, 'run', lambda command, **kwargs: calls.append(command[-1]))
    return mod, out, profile, argv, calls


@pytest.mark.parametrize('changed', ['identity', 'profile'])
def test_resumed_authority_change_refused_before_any_advancement_or_receipt_mutation(tmp_path, monkeypatch, changed):
    mod, out, profile, argv, calls = resumed_cli(tmp_path, monkeypatch)
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    if changed == 'identity': argv[argv.index('--identity') + 1] = 'E' * 40
    else: profile.write_bytes(b'different authority profile')
    assert mod.main() == 1
    assert calls == []
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before


@pytest.mark.parametrize('interruption', ['fsync', 'replace'])
def test_metadata_interruption_resumes_same_submission_without_reassembly_or_resubmission(tmp_path, monkeypatch, interruption):
    mod, out, profile, argv, calls = resumed_cli(tmp_path, monkeypatch)
    before = (out / 'candidate.json').read_bytes()
    operation = getattr(mod.os, interruption)
    def interrupted(*args, **kwargs): raise OSError('synthetic metadata interruption')
    monkeypatch.setattr(mod.os, interruption, interrupted)
    assert mod.main() == 1
    assert calls == [] and (out / 'candidate.json').read_bytes() == before
    assert list(out.glob('.candidate-*.tmp')) == []
    monkeypatch.setattr(mod.os, interruption, operation)
    # A process kill may leave old scratch metadata. It must neither become the
    # authoritative receipt nor prevent polling the original Apple submission.
    (out / 'candidate.json.tmp').write_text('abandoned partial metadata')
    assert mod.main() == 0
    assert calls == ['notary']
    assert json.loads((out / 'release.json').read_text())['appSubmissionID'] == 'existing-apple-id'
    assert json.loads((out / 'candidate.json').read_text())['phase'] == 'appSubmitted'
