"""Pinned frozen evaluation artifacts and fail-closed CLI routing."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import pytest

from runner.frozen_evaluation_cli import execute_request
from runner.tests.test_prepared_test_data import fixture as preparation_fixture
from runner.tests.test_runner import graph
from runner.engine import run
from runner.prepared_data import load_prepared

ROOT = Path(__file__).resolve().parents[2]


def sha(raw): return hashlib.sha256(raw).hexdigest()
def write(path, value):
    raw = json.dumps(value, allow_nan=False).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


@pytest.fixture
def evaluation_fixture(tmp_path, *, segmented=False):
    manifest, provenance, _, source = preparation_fixture(tmp_path / 'build',segmented=segmented)
    prepared = load_prepared(manifest, 'training', 'table', 'validation', 'table', ['b', 'a'], 'label')
    trained = tmp_path / 'trained'
    run(graph(), trained, epochs=1, device='cpu', prepared=prepared)
    folder = tmp_path / 'evaluation'; inputs = folder / 'input'; inputs.mkdir(parents=True)
    shutil.copytree(manifest.parent, inputs / 'preparation/artifacts')
    (inputs / 'training/artifacts').mkdir(parents=True)
    for name in ['manifest.json', 'checkpoint.pt']:
        shutil.copyfile(trained / name, inputs / 'training/artifacts' / name)
    registry = ROOT / 'Sources/KwipMLCore/Resources/components.json'
    shutil.copyfile(registry, inputs / 'model-registry.json')
    shutil.copyfile(ROOT / 'runner/dataflow-operators.json', inputs / 'data-registry.json')
    project = str(uuid.uuid4()).upper(); run_id = str(uuid.uuid4()).upper(); identity = 'a' * 64
    parent = dict(projectID=project, directoryID=str(uuid.uuid4()).upper(), runID=run_id,
                  requestIdentity=identity, modelManifestSHA256=sha((trained / 'manifest.json').read_bytes()),
                  preparationManifestSHA256=sha(manifest.read_bytes()), checkpointSHA256=sha((trained / 'checkpoint.pt').read_bytes()))
    write(inputs / 'capture.json', dict(schemaVersion=5, compilerContractVersion=4, projectID=project,
        directoryID=parent['directoryID'], runID=run_id, requestIdentity=identity))
    write(inputs / 'workflow.json', dict(schemaVersion=5, projectID=project))
    write(inputs / 'runtime-evidence.json', dict(schemaVersion=1))
    parent['outcomeSHA256'] = sha(write(inputs / 'outcome.json', dict(schemaVersion=2, status='completed',
        projectID=project, directoryID=parent['directoryID'], runID=run_id, requestIdentity=identity,
        modelManifestSHA256=parent['modelManifestSHA256'], preparationManifestSHA256=parent['preparationManifestSHA256'])))
    files = [dict(path=str(path.relative_to(inputs)), bytes=path.stat().st_size, sha256=sha(path.read_bytes()))
             for path in sorted(inputs.rglob('*')) if path.is_file()]
    receipt = write(inputs / 'snapshot.json', dict(schemaVersion=1, reference=parent,
                   testEndpoint=dict(node='split', port='test'), files=files))
    path = folder / 'request.json'
    raw = write(path, dict(schemaVersion=1, kind='frozenEvaluation', evaluationID=str(uuid.uuid4()).upper(),
        projectID=project, requestIdentity='b' * 64, snapshotReceiptSHA256=sha(receipt),
        settings=dict(device='cpu', batchSize=5, previewLimit=2)))
    source.unlink()
    return path, sha(raw)


def test_real_full_population_artifacts_pins_progress_and_exclusive_output(evaluation_fixture):
    path, pin = evaluation_fixture; events = []
    manifest = execute_request(path, pin, events.append)
    output = path.parent / 'artifacts'
    assert manifest == json.loads((output / 'manifest.json').read_bytes())
    assert manifest['kind'] == 'frozenEvaluation' and manifest['status'] == 'completed'
    assert manifest['requestSHA256'] == pin
    summary = json.loads((output / 'summary.json').read_bytes())
    raw = (output / 'predictions.jsonl').read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    assert len(rows) == summary['evaluatedCount'] == 14 and raw.endswith(b'\n')
    assert summary['preview'] == rows[:2] and 'predictions' not in summary
    assert 'testRowIDs' not in manifest['testProvenance']
    assert summary['lossMean'] * 14 == pytest.approx(summary['lossSum'])
    assert sum(map(sum, summary['confusionMatrix'])) == 14
    assert [e['completedCount'] for e in events if e['event'] == 'evaluationProgress'] == [0, 5, 10, 14]
    assert events[0]['event'] == 'evaluationStarted' and events[-1]['event'] == 'evaluationCompleted'
    assert events[-1]['manifestSHA256'] == sha((output / 'manifest.json').read_bytes())
    for file in manifest['artifacts']:
        data = (output / file['path']).read_bytes()
        assert len(data) == file['bytes'] and sha(data) == file['sha256']
    before = (output / 'manifest.json').read_bytes()
    with pytest.raises((ValueError, FileExistsError)):
        execute_request(path, pin, lambda _: None)
    assert (output / 'manifest.json').read_bytes() == before


@pytest.mark.parametrize('mutation', ['request-pin', 'snapshot-pin', 'checkpoint', 'unknown-request', 'wrong-project', 'unknown-file', 'symlink', 'ancestor-symlink'])
def test_tampered_or_unsafe_snapshot_never_publishes_manifest(evaluation_fixture, mutation):
    path, pin = evaluation_fixture
    if mutation == 'request-pin': pin = '0' * 64
    elif mutation == 'checkpoint': (path.parent / 'input/training/artifacts/checkpoint.pt').write_bytes(b'bad')
    elif mutation == 'symlink':
        target = path.parent / 'input/model-registry.json'; raw = target.read_bytes(); target.unlink()
        external = path.parent / 'external.json'; external.write_bytes(raw); target.symlink_to(external)
    elif mutation == 'ancestor-symlink':
        target = path.parent / 'input/preparation'; external = path.parent / 'elsewhere-preparation'
        target.rename(external); target.symlink_to(external, target_is_directory=True)
    elif mutation == 'unknown-file':
        receipt_path = path.parent / 'input/snapshot.json'; receipt = json.loads(receipt_path.read_bytes())
        receipt['files'].append(dict(path='../escape', bytes=0, sha256=sha(b'')))
        value = json.loads(path.read_bytes()); value['snapshotReceiptSHA256'] = sha(write(receipt_path, receipt)); pin = sha(write(path, value))
    else:
        value = json.loads(path.read_bytes())
        if mutation == 'snapshot-pin': value['snapshotReceiptSHA256'] = '0' * 64
        elif mutation == 'unknown-request': value['output'] = '/tmp/elsewhere'
        else: value['projectID'] = str(uuid.uuid4()).upper()
        pin = sha(write(path, value))
    with pytest.raises((ValueError, OSError)):
        execute_request(path, pin, lambda _: None)
    assert not (path.parent / 'artifacts/manifest.json').exists()


def test_batch_cancellation_preserves_no_completed_manifest(evaluation_fixture):
    path, pin = evaluation_fixture; events = []
    class Cancelled(Exception): pass
    def emit(event):
        events.append(event)
        if event.get('completedCount') == 5: raise Cancelled()
    with pytest.raises(Cancelled): execute_request(path, pin, emit)
    assert not (path.parent / 'artifacts/manifest.json').exists()
    assert not any(e['event'] == 'evaluationCompleted' for e in events)


def test_cli_route_rejects_training_options_and_runtime_registry_cannot_override_snapshot(evaluation_fixture):
    path, pin = evaluation_fixture
    command = [sys.executable, str(ROOT / 'runner/main.py'), '--evaluate-frozen', str(path), '--evaluation-request-sha256', pin]
    rejected = subprocess.run(command + ['--epochs', '1'], capture_output=True, text=True, timeout=30)
    assert rejected.returncode != 0 and not (path.parent / 'artifacts/manifest.json').exists()
    result = subprocess.run(command + ['--registry', '/not-used/current-runtime-registry.json'], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout.splitlines()[-1])['event'] == 'evaluationCompleted'


def test_serialization_cancellation_cannot_publish_manifest(evaluation_fixture):
    path, pin = evaluation_fixture
    class Cancelled(Exception): pass
    def check():
        if (path.parent / 'artifacts/predictions.jsonl').exists(): raise Cancelled()
    with pytest.raises(Cancelled):
        execute_request(path, pin, lambda _: None, check_cancelled=check)
    assert not (path.parent / 'artifacts/manifest.json').exists()


def test_evaluator_zero_callback_cancels_before_any_forward():
    import torch
    from runner.final_evaluation import evaluate_classifier
    class NoForward(torch.nn.Module):
        def forward(self, x): pytest.fail('Zero-progress cancellation must precede forward')
    class Cancelled(Exception): pass
    calls = []
    def callback(completed, total):
        calls.append((completed, total)); raise Cancelled()
    with pytest.raises(Cancelled):
        evaluate_classifier(NoForward(), torch.zeros((1, 2)), torch.tensor([0]), ['one'],
                            batch_size=1, device='cpu', on_batch=callback)
    assert calls == [(0, 1)]


def test_capture_off_keeps_full_evaluation_output_without_optional_summary_preview(evaluation_fixture):
    path,_=evaluation_fixture
    value=json.loads(path.read_bytes());value['retainComponentOutputs']=False;path.write_text(json.dumps(value))
    manifest=execute_request(path,sha(path.read_bytes()),lambda _:None)
    output=path.parent/'artifacts';summary=json.loads((output/'summary.json').read_bytes())
    assert summary['preview']==[] and summary['evaluatedCount']==14
    assert len((output/'predictions.jsonl').read_bytes().splitlines())==14
    assert manifest['settings']['previewLimit']==2
