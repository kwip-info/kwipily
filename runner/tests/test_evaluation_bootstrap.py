"""Evaluation bootstrap gates inference on its own authority and native guard."""
import json
from pathlib import Path
import sys

import pytest
from runner.tests.test_evaluation_authorization import evaluation_authority, encoded, digest
from runner.tests.test_worker_bootstrap import entry, run_supervisor
from runner.tests.test_lifetime_watchdog import libraries


def control(f, mode='--kwip-evaluate-supervise'):
    return [mode, str(f[0]), digest(encoded(f[1])), digest(encoded(f[2]))]


@pytest.mark.skipif(sys.platform != 'darwin', reason='Native macOS lifetime guard')
def test_evaluation_reexec_ready_and_exact_derived_cli_before_engine(evaluation_authority, tmp_path, libraries):
    f = evaluation_authority; watchdog, _ = libraries
    registry = tmp_path / 'runtime-registry.json'; registry.write_text('{}')
    code, events, errors = run_supervisor(entry(tmp_path, watchdog, registry), control(f))
    assert code == 0, (events, errors)
    assert events[0] == dict(event='evaluationReady', evaluationID=f[1]['evaluationID'],
                            requestSHA256=digest(encoded(f[1])), ownershipSHA256=digest(encoded(f[2])))
    assert [event['event'] for event in events] == ['evaluationReady', 'engineMarker']
    assert events[1]['arguments'] == ['--evaluate-frozen', str(f[0] / 'request.json'),
        '--evaluation-request-sha256', digest(encoded(f[1])), '--registry', str(registry)]


@pytest.mark.parametrize('state', ['wrong-pin','unclaimed-worker','missing-guard'])
def test_evaluation_invalid_authority_or_guard_never_reaches_engine(evaluation_authority, tmp_path, state):
    args = control(evaluation_authority)
    if state == 'wrong-pin': args[2] = '0' * 64
    elif state == 'unclaimed-worker': args = control(evaluation_authority, '--kwip-evaluate-worker') + ['--kwip-lifetime-fd','50','--kwip-activity-fd','51']
    code, events, errors = run_supervisor(entry(tmp_path, tmp_path / 'missing.dylib'), args)
    assert code == 1, (events, errors)
    assert [event['event'] for event in events] == ['error']


@pytest.mark.parametrize('arguments', [['--kwip-evaluate-supervise'], ['--kwip-evaluate-worker'],
    ['--kwip-evaluate-supervise=x'], ['--capabilities','--kwip-evaluate-worker']])
def test_evaluation_reserved_controls_reject_bad_arity_and_placement(tmp_path, arguments):
    from runner.worker_bootstrap import RuntimeContext, prepare
    with pytest.raises(ValueError): prepare(arguments, RuntimeContext(tmp_path / 'entry.py', tmp_path / 'guard.dylib', None))


def test_already_closed_parent_returns_cancelled_without_claim(evaluation_authority, tmp_path):
    import subprocess
    f = evaluation_authority
    path = entry(tmp_path, tmp_path / 'missing.dylib')
    process = subprocess.run([sys.executable, '-I', '-B', str(path), *control(f)],
                             input='', capture_output=True, text=True, timeout=10)
    assert process.returncode == 130, process.stdout + process.stderr
    assert process.stdout == ''
    assert not (f[0] / 'claim.json').exists()
