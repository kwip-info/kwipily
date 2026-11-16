"""Pinned bounded ranges, dedup accounting, tamper and owned rollback."""
import hashlib,json,os
import pytest
from runner.artifact_storage import ArtifactStorage
from runner.dataflow_disk import disk_scope
from runner.table_artifact_pack import TableArtifactPack,TableArtifactPackReader,MAX_OBJECT_BYTES


def test_objects_deduplicate_inside_one_sealed_pack_and_read_by_pin(tmp_path):
    with disk_scope(directory=tmp_path) as scratch:
        with ArtifactStorage(tmp_path,maximum_bytes=1_000_000) as storage:
            with TableArtifactPack(storage,maximum_bytes=1_000_000) as pack:
                first=pack.add_json([['private sample',1],['other',None]])
                again=pack.add_json([['private sample',1],['other',None]])
                second=pack.add_json(['row:1','row:2'])
                assert first==again and first!=second
            descriptor=pack.descriptor
            assert storage.budget.used_bytes==descriptor['bytes']<1000
        assert scratch.used==0
    assert {p.name for p in tmp_path.iterdir()}=={'tables.kwipdata'}
    with TableArtifactPackReader(tmp_path,descriptor,maximum_bytes=1_000_000) as reader:
        reader.audit()
        assert reader.read_json(first)==[['private sample',1],['other',None]]
        assert reader.read_json(second)==['row:1','row:2']
    assert hashlib.sha256((tmp_path/'tables.kwipdata').read_bytes()).hexdigest()==descriptor['sha256']


def test_cancel_and_failed_insert_roll_back_pack_and_scratch(tmp_path):
    with disk_scope(directory=tmp_path) as scratch:
        with ArtifactStorage(tmp_path,maximum_bytes=1000) as storage:
            with pytest.raises(ValueError):
                with TableArtifactPack(storage,maximum_bytes=1000) as pack:
                    pack.add_json(['small'])
                    pack.add_json(['too large'*500])
            assert storage.budget.used_bytes==0 and scratch.used==0
    assert list(tmp_path.iterdir())==[]


def test_range_tamper_is_rejected_even_when_size_and_mtime_are_restored(tmp_path):
    with ArtifactStorage(tmp_path,maximum_bytes=1000) as storage:
        with TableArtifactPack(storage,maximum_bytes=1000) as pack:reference=pack.add_json(['before'])
    path=tmp_path/'tables.kwipdata'
    with TableArtifactPackReader(tmp_path,pack.descriptor,maximum_bytes=1000) as reader:
        reader.audit();before=path.stat()
        with path.open('r+b') as stream:stream.seek(reference['offset']);stream.write(b'X')
        os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
        with pytest.raises(ValueError):reader.read_json(reference)
        with pytest.raises(ValueError):reader.audit()


def test_symlink_and_foreign_or_oversized_ranges_are_refused(tmp_path):
    with ArtifactStorage(tmp_path,maximum_bytes=1000) as storage:
        with TableArtifactPack(storage,maximum_bytes=1000) as pack:reference=pack.add_json(['safe'])
    with TableArtifactPackReader(tmp_path,pack.descriptor,maximum_bytes=1000) as reader:
        for changed in ({'offset':True},{'offset':-1},{'bytes':0},{'bytes':MAX_OBJECT_BYTES+1},{'path':'/private'},{'offset':1000},{'sha256':'0'*64}):
            with pytest.raises(ValueError):reader.read_json({**reference,**changed})
    target=tmp_path/'moved';(tmp_path/'tables.kwipdata').rename(target)
    (tmp_path/'tables.kwipdata').symlink_to(target)
    with pytest.raises(ValueError):TableArtifactPackReader(tmp_path,pack.descriptor,maximum_bytes=1000)


def test_object_encoder_bounds_allocation_before_encoding_large_text(tmp_path):
    with ArtifactStorage(tmp_path,maximum_bytes=MAX_OBJECT_BYTES*2) as storage:
        with pytest.raises(ValueError):
            with TableArtifactPack(storage,maximum_bytes=MAX_OBJECT_BYTES*2) as pack:
                pack.add_json(['x'*(MAX_OBJECT_BYTES+1)])
        assert storage.budget.used_bytes==0
    assert list(tmp_path.iterdir())==[]


def test_open_failure_and_swallowed_object_failure_cannot_publish(tmp_path):
    with disk_scope(directory=tmp_path) as scratch:
        with ArtifactStorage(tmp_path,maximum_bytes=0) as storage:
            with pytest.raises(ValueError):
                with TableArtifactPack(storage,maximum_bytes=1000):pass
            assert storage.budget.used_bytes==0 and scratch.used==0
        with ArtifactStorage(tmp_path,maximum_bytes=1000) as storage:
            with pytest.raises(ValueError,match='cannot be published'):
                with TableArtifactPack(storage,maximum_bytes=1000) as pack:
                    with pytest.raises(ValueError):pack.add_json([float('nan')])
            assert storage.budget.used_bytes==0 and scratch.used==0
    assert list(tmp_path.iterdir())==[]


def test_a_valid_range_is_distinct_from_whole_pack_acceptance(tmp_path):
    with ArtifactStorage(tmp_path,maximum_bytes=1000) as storage:
        with TableArtifactPack(storage,maximum_bytes=1000) as pack:
            broken=pack.add_json(['before']);valid=pack.add_json(['valid'])
    with (tmp_path/'tables.kwipdata').open('r+b') as stream:stream.seek(broken['offset']);stream.write(b'X')
    with TableArtifactPackReader(tmp_path,pack.descriptor,maximum_bytes=1000) as reader:
        assert reader.read_json(valid)==['valid']
        with pytest.raises(ValueError,match='pin differs'):reader.audit()


def test_expanding_unicode_is_refused_without_allocating_the_complete_encoding(tmp_path):
    import tracemalloc
    value='😀'*1_000_000 # Allocated before tracing; JSON ASCII escaping is 12 MB.
    with ArtifactStorage(tmp_path,maximum_bytes=MAX_OBJECT_BYTES*2) as storage:
        tracemalloc.start()
        try:
            with pytest.raises(ValueError):
                with TableArtifactPack(storage,maximum_bytes=MAX_OBJECT_BYTES*2) as pack:pack.add_json([value])
            _,peak=tracemalloc.get_traced_memory()
        finally:tracemalloc.stop()
        assert peak<8_000_000,peak
        assert storage.budget.used_bytes==0
    assert list(tmp_path.iterdir())==[]


def test_object_encoding_checks_bounded_work_not_every_json_token(tmp_path):
    checks=0
    def check():
        nonlocal checks
        checks+=1
    with ArtifactStorage(tmp_path,maximum_bytes=1000000,check_cancelled=check) as storage:
        with TableArtifactPack(storage,maximum_bytes=1000000) as pack:ref=pack.add_json(list(range(10000)))
    assert checks<1000
    with TableArtifactPackReader(tmp_path,pack.descriptor,maximum_bytes=1000000) as reader:
        assert reader.read_json(ref)==list(range(10000))


def test_nested_row_encoding_shares_one_bounded_work_counter():
    from runner.table_artifact_pack import encode_object
    value=[{'row':[number,'sample',None], 'identity':str(number)} for number in range(256)]
    checks=0
    def check():
        nonlocal checks
        checks+=1
    encoded=encode_object(value,check)
    assert encoded==json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')
    assert 3<=checks<40, 'Nested containers must not restart the authority-work interval'


def test_nested_row_encoding_still_cancels_inside_structural_walk():
    from runner.table_artifact_pack import encode_object
    checks=0
    class Cancelled(Exception):pass
    def check():
        nonlocal checks
        checks+=1
        if checks==3:raise Cancelled()
    with pytest.raises(Cancelled):
        encode_object([{'row':[number,'sample',None]} for number in range(10000)],check)
    assert checks==3


def test_flat_scalar_column_uses_one_bounded_shape_admission(monkeypatch):
    import runner.table_artifact_pack as module
    calls=[];shape=module._shape
    def counted(*a,**kw):
        calls.append(1);return shape(*a,**kw)
    monkeypatch.setattr(module,'_shape',counted)
    values=[None,True,2**63-1,-0.0,'é😀\\"\n']*50
    actual=module.encode_object(values)
    import json
    assert actual==json.dumps(values,ensure_ascii=True,sort_keys=True,separators=(',',':'),allow_nan=False).encode('ascii')
    assert len(calls)==1


def test_small_frames_batch_physical_writes_without_changing_offsets_or_pins(tmp_path,monkeypatch):
    from runner import artifact_storage
    writes=[];original=artifact_storage._BoundedSink.write
    def measured(sink,data):writes.append(len(data));return original(sink,data)
    monkeypatch.setattr(artifact_storage._BoundedSink,'write',measured)
    with ArtifactStorage(tmp_path,maximum_bytes=1_000_000) as storage:
        with TableArtifactPack(storage,1_000_000,buffer_bytes=262144) as pack:
            refs=[pack.add_json([i,'x'*100]) for i in range(100)]
            assert pack.add_json([0,'x'*100])==refs[0]
        assert len(writes)<5
    with TableArtifactPackReader(tmp_path,pack.descriptor,maximum_bytes=1_000_000) as reader:
        reader.audit()
        assert [reader.read_json(ref) for ref in refs]==[[i,'x'*100] for i in range(100)]


def test_buffered_pack_flush_keeps_live_cancellation_and_owned_rollback(tmp_path):
    revoked=False
    def check():
        if revoked:raise InterruptedError('revoked')
    with ArtifactStorage(tmp_path,maximum_bytes=1_000_000,check_cancelled=check) as storage:
        with pytest.raises(InterruptedError,match='revoked'):
            with TableArtifactPack(storage,1_000_000,buffer_bytes=262144) as pack:
                pack.add_json(['buffered'])
                revoked=True
        assert storage.budget.used_bytes==0
    assert list(tmp_path.iterdir())==[]
