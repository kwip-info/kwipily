"""Immutable live segments and relative references across paged metadata."""
import pytest
from runner.artifact_storage import ArtifactStorage
from runner.table_artifact_pack import TableArtifactPack
from runner.segmented_table_store import SegmentResolver,StoredReference,SegmentWriter
from runner.packed_ordered_index import OrderedIndex


def test_metadata_children_keep_their_owner_context_across_segments(tmp_path):
    with ArtifactStorage(tmp_path,maximum_bytes=100000) as storage:
        with TableArtifactPack(storage,maximum_bytes=100000,name='table-pack-'+'1'*32+'.kwipdata') as first:
            leaf=first.add_json({'schemaVersion':1,'kind':'orderedIndex','height':0,'count':1,'entries':[{'count':1,'value':'first'}]})
            middle=first.add_json({'schemaVersion':1,'kind':'orderedIndex','height':1,'count':1,'entries':[{'count':1,'reference':{'pack':None,'object':leaf}}]})
        with TableArtifactPack(storage,maximum_bytes=100000,name='table-pack-'+'2'*32+'.kwipdata') as second:
            root=second.add_json({'schemaVersion':1,'kind':'orderedIndex','height':2,'count':1,'entries':[{'count':1,'reference':{'pack':first.descriptor,'object':middle}}]})
    with SegmentResolver(tmp_path,maximum_bytes=100000) as resolver:
        index=OrderedIndex(resolver,{'reference':{'pack':second.descriptor,'object':root},'count':1,'height':2},maximum_count=10)
        assert index.locate(0)==('first',0)
        assert list(index.records())==[(1,'first')]
        assert len(resolver.descriptors)==2
        resolver.audit()


def test_writer_deduplicates_sealed_objects_and_publishes_before_read(tmp_path):
    with ArtifactStorage(tmp_path,maximum_bytes=100000) as storage:
        with SegmentWriter(storage,maximum_bytes=100000) as writer:
            first=writer.add_json(['same'])
            with pytest.raises(ValueError):first.reference()
            writer.seal()
            first_ref=first.reference()
            with SegmentResolver(tmp_path,maximum_bytes=100000) as resolver:assert resolver.read_json(first_ref)==['same']
            again=writer.add_json(['same'])
            assert again.reference()==first_ref
            second=writer.add_json({'earlier':writer.relative(first)})
            writer.seal()
            with SegmentResolver(tmp_path,maximum_bytes=100000) as resolver:
                value,children=resolver.read_node(second.reference())
                assert children.read_json(value['earlier'])==['same']
            assert len(writer.descriptors)==2


def test_unsealed_error_discards_only_the_active_segment(tmp_path):
    with ArtifactStorage(tmp_path,maximum_bytes=100000) as storage:
        with pytest.raises(ValueError):
            with SegmentWriter(storage,maximum_bytes=100000) as writer:
                first=writer.add_json(['completed']);writer.seal()
                writer.add_json(['unpublished']);raise ValueError('stop')
        assert len(list(tmp_path.glob('table-pack-*.kwipdata')))==1
        with SegmentResolver(tmp_path,maximum_bytes=100000) as resolver:assert resolver.read_json(first.reference())==['completed']


def test_duplicate_never_consumes_remaining_retention_allowance(tmp_path):
    with ArtifactStorage(tmp_path,maximum_bytes=1000) as storage:
        with SegmentWriter(storage,maximum_bytes=1000) as writer:
            first=writer.add_json(['a'*900]);writer.seal()
            used=storage.budget.used_bytes
            assert writer.add_json(['a'*900]).reference()==first.reference()
            assert storage.budget.used_bytes==used
            assert writer.active is None


def test_relative_external_conflicts_and_aggregate_allowance(tmp_path):
    with ArtifactStorage(tmp_path,maximum_bytes=100000) as storage:
        with SegmentWriter(storage,maximum_bytes=100000) as writer:
            first=writer.add_json(['a']);writer.seal()
            second=writer.add_json(['b']);writer.seal()
    with SegmentResolver(tmp_path,maximum_bytes=100000) as resolver:
        with pytest.raises(ValueError):resolver.read_json({'pack':None,'object':first.object})
        assert resolver.read_json(first.reference())==['a']
        forged=first.reference();forged['pack']['sha256']='0'*64
        with pytest.raises(ValueError):resolver.read_json(forged)
    with SegmentResolver(tmp_path,maximum_bytes=first.owner.descriptor['bytes']) as resolver:
        resolver.read_json(first.reference())
        with pytest.raises(ValueError):resolver.read_json(second.reference())


def test_caught_failure_cannot_publish_unsealed_partial_segment(tmp_path):
    with ArtifactStorage(tmp_path,maximum_bytes=1000) as storage:
        with pytest.raises(ValueError,match='failed segment transaction'):
            with SegmentWriter(storage,maximum_bytes=1000) as writer:
                first=writer.add_json(['completed']);writer.seal()
                writer.add_json(['pending'])
                with pytest.raises(ValueError):writer.add_json(['x'*2000])
        assert len(list(tmp_path.glob('*.kwipdata')))==1
        assert storage.budget.used_bytes==first.owner.descriptor['bytes']


def test_repeated_small_objects_share_bounded_encoding_work_checks(tmp_path):
    checks=0
    def check():
        nonlocal checks
        checks+=1
    with ArtifactStorage(tmp_path,maximum_bytes=100000,check_cancelled=check) as storage:
        with SegmentWriter(storage,maximum_bytes=100000) as writer:
            refs=[writer.add_json(['same',2**60+1,None]) for _ in range(4096)]
        assert all(ref.reference()==refs[0].reference() for ref in refs)
    assert checks<500
    with SegmentResolver(tmp_path,maximum_bytes=100000) as resolver:
        assert resolver.read_json(refs[0].reference())==['same',2**60+1,None]
        resolver.audit()


def test_encoding_batch_revocation_discards_unsealed_objects(tmp_path):
    revoked=False;seen=0
    def check():
        if revoked:raise RuntimeError('owner revoked')
    with ArtifactStorage(tmp_path,maximum_bytes=100000,check_cancelled=check) as storage:
        with pytest.raises(RuntimeError,match='owner revoked'):
            with SegmentWriter(storage,maximum_bytes=100000) as writer:
                for i in range(4096):
                    if i==129:revoked=True
                    writer.add_json(['same']);seen+=1
    assert seen<258
    assert not list(tmp_path.glob('*.kwipdata'))


def test_revocation_at_seal_cleans_active_transaction(tmp_path):
    revoked=False
    def check():
        if revoked:raise RuntimeError('owner revoked')
    with ArtifactStorage(tmp_path,maximum_bytes=100000,check_cancelled=check) as storage:
        with pytest.raises(RuntimeError,match='owner revoked'):
            with SegmentWriter(storage,maximum_bytes=100000) as writer:
                writer.add_json(['not published'])
                revoked=True
        assert storage.budget.used_bytes==0
    assert not list(tmp_path.iterdir())
