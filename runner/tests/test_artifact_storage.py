"""Byte limits precede physical writes and publication never replaces others."""
from pathlib import Path
import pytest


def test_high_water_seek_and_overwrite_charge_one_allocation(tmp_path):
    from runner.artifact_storage import ArtifactStorage
    storage=ArtifactStorage(tmp_path,10)
    def write(stream):
        stream.write(b'1234');assert stream.tell()==4
        stream.seek(1);stream.write(b'x');stream.seek(8);stream.write(b'y');stream.flush()
    receipt=storage.atomic_write('one',9,write)
    assert receipt.st_size==9 and (tmp_path/'one').read_bytes()==b'1x34\0\0\0\0y'
    storage.atomic_bytes('two',b'z',1)
    with pytest.raises(ValueError):storage.atomic_bytes('three',b'x',1)
    assert not (tmp_path/'three').exists() and not (tmp_path/'three.tmp').exists()


@pytest.mark.parametrize('operation',['write','seek'])
def test_rejected_extension_never_reaches_file(tmp_path,operation):
    from runner.artifact_storage import ArtifactStorage
    storage=ArtifactStorage(tmp_path,5)
    def write(stream):
        stream.write(b'ok')
        with pytest.raises(ValueError):
            if operation=='write':stream.write(b'123456')
            else:stream.seek(6)
        assert (tmp_path/'output.tmp').stat().st_size<=2
        raise InterruptedError('abort')
    with pytest.raises(InterruptedError):storage.atomic_write('output',5,write)
    assert list(tmp_path.iterdir())==[]
    storage.atomic_bytes('retry',b'12345',5)


@pytest.mark.parametrize('replacement',['target','temporary'])
def test_atomic_failure_preserves_unowned_replacement(tmp_path,replacement):
    from runner.artifact_storage import ArtifactStorage
    storage=ArtifactStorage(tmp_path,10)
    def write(stream):
        stream.write(b'owned')
        path=tmp_path/('value.tmp' if replacement=='temporary' else 'value')
        if path.exists():path.unlink()
        path.write_bytes(b'other')
    with pytest.raises((ValueError,FileExistsError)):storage.atomic_write('value',10,write)
    assert (tmp_path/('value.tmp' if replacement=='temporary' else 'value')).read_bytes()==b'other'
    if replacement=='temporary':assert not (tmp_path/'value').exists()


def test_existing_target_and_symlink_parent_are_preserved(tmp_path):
    from runner.artifact_storage import ArtifactStorage
    storage=ArtifactStorage(tmp_path,10)
    (tmp_path/'value').write_bytes(b'old')
    with pytest.raises((ValueError,FileExistsError)):storage.atomic_bytes('value',b'new',3)
    outside=tmp_path/'outside';outside.mkdir();(tmp_path/'link').symlink_to(outside,target_is_directory=True)
    with pytest.raises((ValueError,OSError)):storage.atomic_bytes('link/value',b'new',3)
    assert (tmp_path/'value').read_bytes()==b'old' and list(outside.iterdir())==[]


def test_cancel_before_commit_removes_owned_temporary(tmp_path):
    from runner.artifact_storage import ArtifactStorage
    ready=False
    def check():
        if ready:raise InterruptedError('cancelled')
    def write(stream):
        nonlocal ready
        stream.write(b'value');ready=True
    storage=ArtifactStorage(tmp_path,10,check_cancelled=check)
    with pytest.raises(InterruptedError):storage.atomic_write('value',10,write)
    assert list(tmp_path.iterdir())==[]


def test_torch_serialization_uses_bounded_seekable_sink(tmp_path):
    import torch
    from runner.artifact_storage import ArtifactStorage
    storage=ArtifactStorage(tmp_path,10000)
    storage.atomic_write('tensor.pt',10000,lambda stream:torch.save({'x':torch.arange(5)},stream))
    assert torch.equal(torch.load(tmp_path/'tensor.pt',weights_only=True)['x'],torch.arange(5))


def test_training_checkpoint_rejected_before_oversized_write(tmp_path,monkeypatch):
    from runner import engine
    from runner.tests.test_runner import graph
    monkeypatch.setitem(engine.TRAINING_ARTIFACT_LIMITS,'checkpoint.pt',8)
    def save(value,stream):
        stream.write(b'123456789')
    monkeypatch.setattr(engine.torch,'save',save)
    output=tmp_path/'training'
    with pytest.raises(ValueError):engine.run(graph(),output,epochs=1,device='cpu')
    assert not (output/'checkpoint.pt').exists() and not (output/'checkpoint.pt.tmp').exists()
    assert not (output/'manifest.json').exists()


def test_training_aggregate_failure_cannot_complete(tmp_path,monkeypatch):
    from runner import engine
    from runner.tests.test_runner import graph
    monkeypatch.setattr(engine,'TRAINING_STORAGE_BYTES',1)
    events=[]
    with pytest.raises(ValueError):engine.run(graph(),tmp_path/'training',epochs=1,device='cpu',emit=events.append)
    assert not (tmp_path/'training/manifest.json').exists()
    assert not any(event['event']=='completed' for event in events)


def test_preview_cache_bound_precedes_cache_publication(tmp_path,monkeypatch):
    from runner import preview
    from runner.tests.test_runner import graph
    monkeypatch.setitem(preview.PREVIEW_ARTIFACT_LIMITS,'activation-cache.pt',8)
    monkeypatch.setattr(preview.torch,'save',lambda value,stream:stream.write(b'123456789'))
    output=tmp_path/'preview';events=[]
    with pytest.raises(ValueError):preview.preview(graph(),output,device='cpu',stop_at='output',emit=events.append)
    assert not (output/'activation-cache.pt').exists() and not (output/'preview.json').exists()
    assert not any(event['event']=='previewCompleted' for event in events)


def test_shared_budget_covers_descriptor_root_and_scratch(tmp_path):
    import os
    from runner.artifact_storage import ArtifactStorage,ArtifactStorageBudget
    scratch=tmp_path/'scratch';scratch.mkdir();output=tmp_path/'output';output.mkdir()
    budget=ArtifactStorageBudget(10)
    first=ArtifactStorage(scratch,budget=budget)
    fd=os.open(output,os.O_RDONLY|os.O_DIRECTORY)
    second=ArtifactStorage(fd,budget=budget);os.close(fd)
    first.atomic_bytes('source',b'123456',6)
    second.atomic_bytes('payload',b'1234',4)
    with pytest.raises(ValueError):second.atomic_bytes('manifest.json',b'{}',2)
    assert not (output/'manifest.json').exists()
    first.close();second.close()
