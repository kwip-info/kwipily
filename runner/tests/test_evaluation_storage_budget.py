"""Evaluation/export scratch and receipts share bounded pre-write accounting."""
import json
from pathlib import Path
import pytest
from runner.tests.test_frozen_evaluation_cli import evaluation_fixture
from runner.tests.test_portable_evaluation import completed, completed_snapshot


def test_evaluation_scratch_is_charged_before_any_completed_artifact(evaluation_fixture, monkeypatch):
    from runner import frozen_evaluation_cli as worker
    request, pin = evaluation_fixture
    monkeypatch.setattr(worker, 'EVALUATION_STORAGE_BYTES', 1, raising=False)
    events = []
    with pytest.raises(ValueError, match='storage|Storage|budget|Budget'):
        worker.execute_request(request, pin, events.append)
    assert not (request.parent/'artifacts/manifest.json').exists()
    assert not any(event['event']=='evaluationCompleted' for event in events)


def test_export_scratch_and_package_share_the_same_budget(completed, tmp_path, monkeypatch):
    from runner import portable_evaluation_export as worker
    root, pin = completed
    receipt = json.loads((root/'input/snapshot.json').read_bytes())
    scratch_size = sum(value['bytes'] for value in receipt['files'])
    monkeypatch.setattr(worker, 'EXPORT_STORAGE_BYTES', scratch_size + 1, raising=False)
    destination = tmp_path/'export'
    with pytest.raises(ValueError, match='storage|Storage|budget|Budget'):
        worker.build_package(root, destination, pin)
    assert not (destination/'manifest.json').exists()


def test_export_manifest_has_a_prewrite_bound(completed, tmp_path, monkeypatch):
    from runner import portable_evaluation_export as worker
    root, pin = completed
    monkeypatch.setattr(worker, 'EXPORT_MANIFEST_BYTES', 1, raising=False)
    destination = tmp_path/'export'
    with pytest.raises(ValueError, match='storage|Storage|budget|Budget|bound'):
        worker.build_package(root, destination, pin)
    assert not (destination/'manifest.json').exists()


def test_evaluation_failure_does_not_remove_an_unowned_receipt(evaluation_fixture):
    from runner.frozen_evaluation_cli import execute_request
    request, pin = evaluation_fixture
    output = request.parent/'artifacts'
    sentinel = b'foreign receipt'
    def replaced():
        if output.exists() and not (output/'manifest.json').exists():
            (output/'manifest.json').write_bytes(sentinel)
    with pytest.raises((ValueError, FileExistsError)):
        execute_request(request, pin, lambda _: None, check_cancelled=replaced)
    assert (output/'manifest.json').read_bytes() == sentinel


def test_export_failure_does_not_remove_an_unowned_receipt(completed, tmp_path):
    from runner.portable_evaluation_export import build_package
    root, pin = completed
    destination = tmp_path/'export'
    sentinel = b'foreign receipt'
    def replaced():
        if destination.exists() and any(destination.iterdir()) and not (destination/'manifest.json').exists():
            (destination/'manifest.json').write_bytes(sentinel)
    with pytest.raises((ValueError, FileExistsError, OSError)):
        build_package(root, destination, pin, check_cancelled=replaced)
    assert (destination/'manifest.json').read_bytes() == sentinel


def test_evaluation_verified_scratch_lives_under_operation_and_is_removed(evaluation_fixture, monkeypatch):
    from runner import frozen_evaluation_cli as worker
    request, pin = evaluation_fixture
    original = worker.load_frozen
    seen = []
    def observe(checkpoint, **kwargs):
        scratch = Path(checkpoint).parents[2]
        seen.append(scratch)
        assert scratch.parent == request.parent
        assert scratch.is_dir()
        return original(checkpoint, **kwargs)
    monkeypatch.setattr(worker, 'load_frozen', observe)
    worker.execute_request(request, pin, lambda _: None)
    assert len(seen) == 1 and not seen[0].exists()
    assert (request.parent/'artifacts/manifest.json').is_file()


def test_export_verified_scratch_lives_under_export_owner_not_saved_evaluation(completed, tmp_path, monkeypatch):
    from runner import frozen_evaluation_cli as worker
    from runner.portable_evaluation_export import build_package
    root, pin = completed
    owner = tmp_path/'export-operation'
    owner.mkdir()
    original = worker.load_frozen
    seen = []
    before = set(root.iterdir())
    def observe(checkpoint, **kwargs):
        scratch = Path(checkpoint).parents[2]
        seen.append(scratch)
        assert scratch.parent == owner
        assert set(root.iterdir()) == before
        return original(checkpoint, **kwargs)
    monkeypatch.setattr(worker, 'load_frozen', observe)
    build_package(root, owner/'package', pin)
    assert len(seen) == 1 and not seen[0].exists()
    assert set(owner.iterdir()) == {owner/'package'}


@pytest.mark.parametrize('operation', ['evaluation', 'export'])
def test_verified_scratch_cleanup_preserves_replacement_directory(evaluation_fixture, tmp_path, monkeypatch, operation):
    from runner import frozen_evaluation_cli as worker
    from runner.portable_evaluation_export import build_package
    request, pin = evaluation_fixture
    owner = request.parent
    if operation == 'export':
        worker.execute_request(request, pin, lambda _: None)
        pin = worker.sha((request.parent/'artifacts/manifest.json').read_bytes())
        owner = tmp_path/'export-operation'
        owner.mkdir()
    seen = []
    def replace(checkpoint, **kwargs):
        scratch = Path(checkpoint).parents[2]
        assert scratch.parent == owner
        moved = scratch.with_name(scratch.name + '-moved')
        scratch.rename(moved)
        scratch.mkdir()
        (scratch/'foreign.txt').write_bytes(b'preserve me')
        seen.append((scratch, moved))
        raise RuntimeError('stop after scratch replacement')
    monkeypatch.setattr(worker, 'load_frozen', replace)
    with pytest.raises(RuntimeError, match='scratch replacement'):
        if operation == 'evaluation': worker.execute_request(request, pin, lambda _: None)
        else: build_package(request.parent, owner/'package', pin)
    scratch, moved = seen[0]
    assert (scratch/'foreign.txt').read_bytes() == b'preserve me'
    assert not (moved/'training/artifacts/checkpoint.pt').exists(), 'Owned captured weights must be cleaned through the held directory.'
    if operation == 'evaluation': assert not (owner/'artifacts/manifest.json').exists()
    else: assert not (owner/'package/manifest.json').exists()
