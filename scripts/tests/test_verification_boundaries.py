from pathlib import Path
import pytest
from scripts.verify_feature import audit_junit, validate_manifest
import json


def test_repository_selections_exist():
    root = Path(__file__).resolve().parents[2]
    validate_manifest(json.loads((root / 'scripts/verification_boundaries.json').read_text()))


@pytest.mark.parametrize('body', ['', '<testcase name="a"><skipped/></testcase>', '<testcase name="a"><failure/></testcase>', '<testcase name="a"/><testcase name="a"/>'])
def test_empty_skipped_failed_duplicate_selection_cannot_pass(tmp_path, body):
    path = tmp_path / 'tests.xml'
    path.write_text('<testsuite>' + body + '</testsuite>')
    with pytest.raises(ValueError): audit_junit(path)


def test_duration_report_uses_actual_cases(tmp_path):
    path = tmp_path / 'tests.xml'
    path.write_text('<testsuite><testcase classname="Suite" name="a" time="0.25"/><testcase classname="Suite" name="b" time="2"/></testsuite>')
    result = audit_junit(path)
    assert result['executed'] == 2 and result['seconds'] == 2.25
    assert result['slowest'][0]['id'] == 'Suite.b'


def test_serial_swift_requires_exact_discovered_start_and_terminal_events(tmp_path):
    from scripts.verify_feature import swift_selection, swift_junit
    expected = swift_selection('Suite.Cases/testOne\nOther.No/testOther\n', ['Cases'])
    log = "Test Case '-[Suite.Cases testOne]' started.\nTest Case '-[Suite.Cases testOne]' passed (0.123 seconds).\n"
    output = tmp_path / 'swift.xml'
    swift_junit(log, expected, output)
    assert audit_junit(output)['executed'] == 1
    for bad in ['', log + log, log.replace('testOne', 'testWrong'), log.splitlines()[0]]:
        with pytest.raises(ValueError): swift_junit(bad, expected, output)
    with pytest.raises(ValueError): swift_selection('Suite.Other/testOne\n', ['Cases'])
    swift_junit(log.replace('passed', 'skipped'), expected, output)
    with pytest.raises(ValueError): audit_junit(output)


def test_fixture_worker_changes_invalidate_identity(tmp_path, monkeypatch):
    from scripts.verify_feature import fixture_identity
    (tmp_path / 'runner').mkdir()
    worker = tmp_path / 'runner/worker.py'
    worker.write_text('original')
    (tmp_path / 'requirements-runner.txt').write_text('pinned')
    monkeypatch.setenv('KWIPML_TEST_DEVELOPMENT_ROOT', str(tmp_path))
    selection = {'requiredEnvironment': ['KWIPML_TEST_DEVELOPMENT_ROOT']}
    before = fixture_identity(selection)
    worker.write_text('changed')
    assert fixture_identity(selection) != before


def test_contract_integration_partitions_have_exact_union():
    from scripts.verify_feature import audit_partitions
    data = {'features': {'all': {'python': ['a.py'], 'swift': ['Cases']},
                         'contract': {'python': ['a.py'], 'swift': ['Cases/testContract']},
                         'worker': {'python': [], 'swift': ['Cases/testWorker']}},
            'partitions': {'all': ['contract', 'worker']}}
    discovery = 'Suite.Cases/testContract\nSuite.Cases/testWorker\n'
    assert audit_partitions(data, discovery)
    data['features']['worker']['swift'] = ['Cases/testContract']
    with pytest.raises(ValueError): audit_partitions(data, discovery)


def test_shared_compute_excludes_competing_feature_and_release_work(tmp_path):
    from scripts.verification_lock import compute_lock
    path = tmp_path / 'compute.lock'
    with compute_lock(path, 'assembly'):
        with pytest.raises(RuntimeError, match='owns compute'):
            with compute_lock(path, 'feature'): pass
    with compute_lock(path, 'full-baseline'): pass
    assert path.read_text() == ''


def test_dashboard_does_not_upgrade_focused_evidence():
    from scripts.verification_report import render
    value = {'feature':'contract','source':{'commit':'a'*40,'untracked':{},'diffSHA256':'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'},
             'reports':{'python':{'executed':2}},'wallSeconds':1.5,'selection':{'resources':['cpu']},
             'fullBaseline':False,'nativeUAT':False,'releaseAcceptance':False}
    assert 'not full-baseline' in render([value])
    value['nativeUAT'] = True
    with pytest.raises(ValueError): render([value])


def test_compute_lock_excludes_another_process_and_releases_after_exit(tmp_path):
    import subprocess, sys
    from scripts.verification_lock import compute_lock
    path = tmp_path / 'compute.lock'
    command = [sys.executable, '-c', "from scripts.verification_lock import compute_lock; from pathlib import Path; import sys; c=compute_lock(Path(sys.argv[1]), 'feature'); c.__enter__(); c.__exit__(None,None,None)", str(path)]
    with compute_lock(path, 'full-baseline'):
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode != 0 and 'owns compute' in result.stderr
    assert subprocess.run(command, capture_output=True).returncode == 0
