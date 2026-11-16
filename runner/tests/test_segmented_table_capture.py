"""Logical capture privacy, bounded chunks, immutable live read and reuse."""
from runner import dataflow as df
from runner.artifact_storage import ArtifactStorage
from runner.segmented_table_store import SegmentWriter,SegmentResolver
from runner.segmented_table_capture import capture_table,read_capture


def table(rows):
    columns=({'name':'visible','dtype':'string','nullable':False},)
    ids=tuple('row:'+str(i) for i in range(len(rows)))
    return df.Table(columns,tuple((v,) for v in rows),ids,'a'*64,data_version=df.digest({'columns':columns,'rows':tuple((v,) for v in rows),'ids':ids}))


def test_capture_live_reads_and_reuses_identical_column_chunks(tmp_path):
    original=table(['value'+str(i) for i in range(600)])
    with ArtifactStorage(tmp_path,maximum_bytes=1000000) as storage:
        with SegmentWriter(storage,maximum_bytes=1000000) as writer:
            first=capture_table(writer,original)
            # First completed table is readable before the next is captured.
            with SegmentResolver(tmp_path,maximum_bytes=1000000) as resolver:
                result=read_capture(resolver,first)
                assert list(result.rows)==list(original.rows)
                assert list(result.row_ids)==list(original.row_ids)
                assert result.data_version==original.data_version
                result.rows.store.close()
            used=storage.budget.used_bytes
            second=capture_table(writer,original)
            # The second manifest uses absolute links to the sealed first
            # segment, so its metadata differs. Its population chunks are reused.
            assert storage.budget.used_bytes-used<4096
            with SegmentResolver(tmp_path,maximum_bytes=1000000) as resolver:
                duplicate=read_capture(resolver,second)
                assert list(duplicate.rows)==list(original.rows)
                duplicate.rows.store.close()


def test_sequential_retained_rows_use_bounded_live_checks(tmp_path):
    checks=0;revoked=False
    def check():
        nonlocal checks
        checks+=1
        if revoked:raise RuntimeError('revoked')
    original=table(['value'+str(i) for i in range(2048)])
    with ArtifactStorage(tmp_path,maximum_bytes=1000000) as storage:
        with SegmentWriter(storage,maximum_bytes=1000000) as writer:ref=capture_table(writer,original)
    with SegmentResolver(tmp_path,maximum_bytes=1000000,check_cancelled=check) as resolver:
        result=read_capture(resolver,ref,check_cancelled=check)
        checks=0
        assert list(result.rows)==list(original.rows)
        assert checks<250
        iterator=iter(result.rows);next(iterator);revoked=True
        seen=1
        import pytest
        with pytest.raises(RuntimeError,match='revoked'):
            for _ in iterator:seen+=1
        assert seen<=128
        # Random access is independently guarded, even for a warm cached row.
        with pytest.raises(RuntimeError,match='revoked'):result.rows[0]
        result.rows.store.close()


def test_required_only_capture_does_not_retain_unrequested_parent_cells(tmp_path):
    parent=table(['KEEP','DROPPED-PRIVATE-MARKER'])
    selected=table(['KEEP'])
    with ArtifactStorage(tmp_path,maximum_bytes=1000000) as storage:
        with SegmentWriter(storage,maximum_bytes=1000000) as writer:
            ref=capture_table(writer,selected)
    assert all(b'DROPPED-PRIVATE-MARKER' not in p.read_bytes() for p in tmp_path.glob('*.kwipdata'))
    with SegmentResolver(tmp_path,maximum_bytes=1000000) as resolver:
        result=read_capture(resolver,ref)
        assert list(result.rows)==[('KEEP',)]
        resolver.audit()
        result.rows.store.close()


def test_corrupt_chunk_or_false_table_identity_cannot_be_accepted(tmp_path):
    import pytest
    with ArtifactStorage(tmp_path,maximum_bytes=1000000) as storage:
        with SegmentWriter(storage,maximum_bytes=1000000) as writer:
            ref=capture_table(writer,table(['exact']))
            with SegmentResolver(tmp_path,maximum_bytes=1000000) as resolver:
                metadata,_=resolver.read_node(ref)
            metadata['lineage']['dataVersion']='0'*64
            bad=writer.add_json(metadata);writer.seal()
            # Replacing metadata alone leaves its self-relative children without
            # an owner in this new segment, which must also be refused.
            with SegmentResolver(tmp_path,maximum_bytes=1000000) as resolver:
                with pytest.raises(ValueError):read_capture(resolver,bad.reference())
    path=tmp_path/ref['pack']['path'];raw=path.read_bytes();path.write_bytes(raw.replace(b'exact',b'wrong'))
    with SegmentResolver(tmp_path,maximum_bytes=1000000) as resolver:
        with pytest.raises(ValueError):read_capture(resolver,ref)


def test_empty_nullable_and_exact_numeric_populations_round_trip(tmp_path):
    import pytest
    for ordinal,rows in enumerate(([],[(None,2**60),(1.5,1)],[(2.0,2**60+1)])):
        root=tmp_path/str(ordinal);root.mkdir()
        columns=({'name':'float','dtype':'float64','nullable':True},{'name':'integer','dtype':'int64','nullable':False})
        ids=tuple('n:'+str(i) for i in range(len(rows)))
        original=df.Table(columns,tuple(rows),ids,'b'*64,data_version=df.digest({'columns':columns,'rows':rows,'ids':ids}))
        with ArtifactStorage(root,maximum_bytes=1000000) as storage:
            with SegmentWriter(storage,maximum_bytes=1000000) as writer:ref=capture_table(writer,original)
        with SegmentResolver(root,maximum_bytes=1000000) as resolver:
            restored=read_capture(resolver,ref)
            assert list(restored.rows)==rows
            assert restored.data_version==original.data_version
            restored.rows.store.close();resolver.audit()


def test_logical_projection_never_captures_hidden_column(tmp_path):
    from runner.dataflow_columnar import ColumnarStore
    from runner.dataflow_shared import projection_table
    from runner.dataflow_disk import disk_scope
    columns=({'name':'visible','dtype':'string','nullable':False},{'name':'secret','dtype':'string','nullable':False})
    with disk_scope(tmp_path):
        owner=ColumnarStore(columns,lambda:None)
        owner.append(('ok','HIDDEN-COLUMN-MARKER'),'r:1')
        rows,ids=owner.finish()
        original=df.Table(columns,rows,ids,'a'*64,data_version=df.digest({'columns':columns,'rows':rows,'ids':ids}))
        selected=projection_table(original,(columns[0],),(0,),lambda:None)
        with ArtifactStorage(tmp_path,maximum_bytes=1000000) as storage:
            with SegmentWriter(storage,maximum_bytes=1000000) as writer:ref=capture_table(writer,selected)
        assert all(b'HIDDEN-COLUMN-MARKER' not in p.read_bytes() for p in tmp_path.glob('*.kwipdata'))
        with SegmentResolver(tmp_path,maximum_bytes=1000000) as resolver:
            restored=read_capture(resolver,ref)
            assert list(restored.rows)==[('ok',)];restored.rows.store.close()
        selected.rows.store.close();owner.close()


def test_shared_table_reader_opens_versioned_envelope_and_preserves_legacy(tmp_path):
    import hashlib,json
    from runner.segmented_table_capture import envelope
    from runner.dataflow_table_reader import read_table_artifact
    original=table(['one','two'])
    with ArtifactStorage(tmp_path,maximum_bytes=1000000) as storage:
        with SegmentWriter(storage,maximum_bytes=1000000) as writer:ref=capture_table(writer,original)
        raw=json.dumps(envelope(original,ref)).encode();storage.atomic_bytes('table.json',raw,100000)
    result=read_table_artifact(tmp_path/'table.json',hashlib.sha256(raw).hexdigest())
    assert list(result.rows)==[('one',),('two',)]
    assert result.data_version==original.data_version
    resolver=result.rows.store.owned_resolver
    result.rows.store.close();assert resolver.closed


def test_small_scalar_capture_batches_admission_without_losing_cancellation(tmp_path):
    import pytest
    checks=0
    def check():
        nonlocal checks
        checks+=1
    original=table(['record'+str(i) for i in range(10000)])
    with ArtifactStorage(tmp_path,maximum_bytes=2000000,check_cancelled=check) as storage:
        with SegmentWriter(storage,maximum_bytes=2000000) as writer:ref=capture_table(writer,original)
    assert checks<3000
    with SegmentResolver(tmp_path,maximum_bytes=2000000) as resolver:
        result=read_capture(resolver,ref);assert len(result.rows)==10000;result.rows.store.close()
    cancelled=tmp_path/'cancelled';cancelled.mkdir();checks=0
    def stop():
        nonlocal checks
        checks+=1
        if checks==20:raise InterruptedError('cancelled capture')
    with ArtifactStorage(cancelled,maximum_bytes=2000000,check_cancelled=stop) as storage:
        with pytest.raises(InterruptedError):
            with SegmentWriter(storage,maximum_bytes=2000000) as writer:capture_table(writer,original)
    assert not list(cancelled.glob('*.kwipdata'))


def test_explicitly_captured_parent_columns_are_referenced_without_recapturing_cells(tmp_path,monkeypatch):
    from runner.dataflow_columnar import ColumnarStore
    from runner.dataflow_shared import projection_table
    from runner.dataflow_disk import disk_scope
    columns=({'name':'visible','dtype':'string','nullable':False},{'name':'other','dtype':'int64','nullable':False})
    with disk_scope(tmp_path):
        owner=ColumnarStore(columns,lambda:None)
        for ordinal in range(600):owner.append(('visible'+str(ordinal),ordinal),'r:'+str(ordinal))
        rows,ids=owner.finish()
        original=df.Table(columns,rows,ids,'a'*64,data_version=df.digest({'columns':columns,'rows':rows,'ids':ids}))
        selected=projection_table(original,(columns[1],),(1,),lambda:None)
        with ArtifactStorage(tmp_path,maximum_bytes=2000000) as storage:
            with SegmentWriter(storage,maximum_bytes=2000000) as writer:
                capture_table(writer,original);used=storage.budget.used_bytes
                def refuse(*_):raise AssertionError('Captured parent cells should not be read again for a projection.')
                monkeypatch.setattr(owner,'_read_chunk',refuse)
                ref=capture_table(writer,selected)
                assert storage.budget.used_bytes-used<4096
        selected.rows.store.close();owner.close()
    with SegmentResolver(tmp_path,maximum_bytes=2000000) as resolver:
        restored=read_capture(resolver,ref)
        assert list(restored.rows)==[(i,) for i in range(600)]
        assert restored.data_version==selected.data_version
        restored.rows.store.close();resolver.audit()


def test_small_scalar_capture_sizes_buffers_without_encoding_each_row(tmp_path,monkeypatch):
    import runner.segmented_table_capture as module
    # The conservative scalar bound includes worst-case escaping. Encode only
    # the published column chunks, rather than each row again just to size it.
    def unexpected_row_encoding(*args,**kwargs):
        raise AssertionError('Small scalar capture repeated row JSON encoding')
    encoder=module.encode_object
    monkeypatch.setattr(module,'encode_object',unexpected_row_encoding)
    original=table(['plain','quote"\\\n','\U0001f642'*200]*200)
    with ArtifactStorage(tmp_path,maximum_bytes=3_000_000) as storage:
        with SegmentWriter(storage,maximum_bytes=3_000_000) as writer:ref=capture_table(writer,original)
    monkeypatch.setattr(module,'encode_object',encoder)
    with SegmentResolver(tmp_path,maximum_bytes=3_000_000) as resolver:
        restored=read_capture(resolver,ref)
        assert list(restored.rows)==list(original.rows)
        assert list(restored.row_ids)==list(original.row_ids)
        assert restored.data_version==original.data_version
        restored.rows.store.close()


def test_computed_capture_reuses_requested_parent_without_reading_unchanged_cells(tmp_path,monkeypatch):
    from runner.dataflow_columnar import ColumnarStore
    from runner.dataflow_formula import apply
    from runner.dataflow_disk import disk_scope
    from runner.polars_preparation import preparation_engine
    columns=({'name':'text','dtype':'string','nullable':False},{'name':'n','dtype':'int64','nullable':False})
    with disk_scope(tmp_path), preparation_engine('polars'):
        owner=ColumnarStore(columns,lambda:None)
        expected=[('x'*10000+str(i),i,i*2) for i in range(600)]
        for text,n,_ in expected:owner.append((text,n),'r:'+str(n))
        rows,ids=owner.finish();original=df.derive(df.Table(columns,rows,ids,'a'*64))
        changed=apply(original,{'operations':[{'id':'double','expression':'[n] * 2','outputs':[{'name':'twice','dtype':'int64','write':'new'}],'nullPolicy':'preserve'}]},lambda:None,share=True)
        try:
            with ArtifactStorage(tmp_path,maximum_bytes=12_000_000) as storage:
                with SegmentWriter(storage,maximum_bytes=12_000_000) as writer:
                    capture_table(writer,original);used=storage.budget.used_bytes
                    def refuse(*_):raise AssertionError('Computed capture read unchanged parent cells again')
                    monkeypatch.setattr(owner,'_read_chunk',refuse)
                    ref=capture_table(writer,changed)
                    assert storage.budget.used_bytes-used<100_000
            with SegmentResolver(tmp_path,maximum_bytes=12_000_000) as resolver:
                restored=read_capture(resolver,ref)
                assert list(restored.rows)==expected
                assert restored.data_version==changed.data_version
                assert list(restored.row_ids)==['r:'+str(i) for i in range(600)]
                restored.rows.store.close();resolver.audit()
        finally:changed.rows.store.close();owner.close()


def test_repeated_immutable_snapshot_reuses_the_sealed_population_without_row_reads(tmp_path,monkeypatch):
    from runner.dataflow_columnar import ColumnarStore
    from runner.dataflow_disk import disk_scope
    import pytest
    revoked=False
    def guard():
        if revoked:raise RuntimeError('revoked snapshot')
    columns=({'name':'value','dtype':'string','nullable':False},)
    with disk_scope(tmp_path):
        owner=ColumnarStore(columns,guard)
        for i in range(600):owner.append(('value'+str(i),),'r:'+str(i))
        rows,ids=owner.finish();original=df.derive(df.Table(columns,rows,ids,'a'*64))
        with ArtifactStorage(tmp_path,maximum_bytes=2_000_000,check_cancelled=guard) as storage:
            with SegmentWriter(storage,maximum_bytes=2_000_000) as writer:
                first=capture_table(writer,original);used=storage.budget.used_bytes
                def refuse(*_):raise AssertionError('Repeated snapshot re-read immutable cells')
                monkeypatch.setattr(owner,'_read_chunk',refuse)
                second=capture_table(writer,original)
                assert second==first and storage.budget.used_bytes==used
                revoked=True
                with pytest.raises(RuntimeError,match='revoked snapshot'):capture_table(writer,original)
                revoked=False
        with pytest.raises(ValueError,match='active unpoisoned'):
            capture_table(writer,original)
        owner.close()
    with SegmentResolver(tmp_path,maximum_bytes=2_000_000) as resolver:
        restored=read_capture(resolver,second)
        assert list(restored.rows)==[('value'+str(i),) for i in range(600)]
        restored.rows.store.close();resolver.audit()


def test_selection_snapshot_references_only_an_explicitly_captured_parent(tmp_path,monkeypatch):
    from runner.dataflow_columnar import ColumnarStore
    from runner.dataflow_disk import disk_scope
    from runner.dataflow_shared import filter_table,projection_table
    from runner.tests.test_filter_rows import config
    columns=({'name':'keep','dtype':'bool','nullable':False},{'name':'text','dtype':'string','nullable':False})
    with disk_scope(tmp_path):
        owner=ColumnarStore(columns,lambda:None)
        for i in range(600):owner.append((i%2==0,'large'+str(i)+'x'*3000),'r:'+str(i))
        rows,ids=owner.finish();original=df.derive(df.Table(columns,rows,ids,'a'*64))
        selected=filter_table(original,config(mode='expression',expression='[keep]'),lambda:None)
        projected=projection_table(selected['matching'],(columns[1],),[1],lambda:None)
        try:
            with ArtifactStorage(tmp_path,maximum_bytes=8_000_000) as storage:
                with SegmentWriter(storage,maximum_bytes=8_000_000) as writer:
                    capture_table(writer,original);used=storage.budget.used_bytes
                    def refuse(*_):raise AssertionError('Selection capture copied parent cells')
                    monkeypatch.setattr(owner,'_read_chunk',refuse)
                    selected_ref=capture_table(writer,selected['matching'])
                    projected_ref=capture_table(writer,projected)
                    assert storage.budget.used_bytes-used<30_000
            with SegmentResolver(tmp_path,maximum_bytes=8_000_000) as resolver:
                metadata,_=resolver.read_node(selected_ref)
                assert metadata['schemaVersion']==2 and metadata['selectionDepth']==1
                restored=read_capture(resolver,projected_ref)
                assert list(restored.rows)==[('large'+str(i)+'x'*3000,) for i in range(0,600,2)]
                assert list(restored.row_ids)==['r:'+str(i) for i in range(0,600,2)]
                assert restored.data_version==projected.data_version
                restored.rows.store.close();resolver.audit()
        finally:
            projected.rows.store.close()
            for item in selected.values():item.rows.store.close()
            owner.close()


def test_nested_selection_retains_ordered_identity_and_complete_validation(tmp_path):
    from runner.dataflow_columnar import ColumnarStore
    from runner.dataflow_disk import disk_scope
    from runner.dataflow_shared import filter_table
    from runner.tests.test_filter_rows import config
    columns=({'name':'n','dtype':'int64','nullable':False},)
    with disk_scope(tmp_path):
        owner=ColumnarStore(columns,lambda:None)
        for i in range(700):owner.append((i,),'r:'+str(i))
        rows,ids=owner.finish();original=df.derive(df.Table(columns,rows,ids,'a'*64))
        first=filter_table(original,config(mode='expression',expression='[n] >= 100'),lambda:None)
        second=filter_table(first['matching'],config(mode='expression',expression='[n] < 600'),lambda:None)
        try:
            with ArtifactStorage(tmp_path,maximum_bytes=2_000_000) as storage:
                with SegmentWriter(storage,maximum_bytes=2_000_000) as writer:
                    capture_table(writer,original);capture_table(writer,first['matching'])
                    ref=capture_table(writer,second['matching'])
            with SegmentResolver(tmp_path,maximum_bytes=2_000_000) as resolver:
                metadata,_=resolver.read_node(ref);assert metadata['selectionDepth']==2
                restored=read_capture(resolver,ref)
                assert list(restored.rows)==[(i,) for i in range(100,600)]
                assert list(restored.row_ids)==['r:'+str(i) for i in range(100,600)]
                assert restored.data_version==second['matching'].data_version
                restored.rows.store.close();resolver.audit()
        finally:
            for results in (second,first):
                for item in results.values():item.rows.store.close()
            owner.close()


def test_selection_chunk_refuses_duplicate_out_of_range_and_unbounded_depth():
    import pytest
    from runner.segmented_table_capture import _CaptureStore
    class Index:
        def locate_span(self,_):return 'selected',0,Reader(),1
    class Reader:
        def read_node(self,reference):
            return (value if reference=='selected' else ['one','two']),self
    store=object.__new__(_CaptureStore);store.selection_depth=1
    good={'schemaVersion':1,'kind':'selectedColumn','source':'source','sourceCount':2,'positions':[0]}
    for bad in ({**good,'positions':[0,0]},{**good,'positions':[-1]},{**good,'positions':[True]},
                {**good,'positions':[2]},{**good,'sourceCount':1},{**good,'source':'selected'}):
        value=bad
        with pytest.raises(ValueError):store._values(Index(),0)


def test_snapshot_index_metadata_is_range_verified_once_then_bounded_in_memory(tmp_path,monkeypatch):
    from runner.segmented_table_capture import _SnapshotIndexReader
    from runner.packed_ordered_index import OrderedIndex
    import pytest
    original=table(['v'+str(i) for i in range(3000)])
    with ArtifactStorage(tmp_path,maximum_bytes=2_000_000) as storage:
        with SegmentWriter(storage,maximum_bytes=2_000_000) as writer:ref=capture_table(writer,original)
    with SegmentResolver(tmp_path,maximum_bytes=2_000_000) as resolver:
        metadata,context=resolver.read_node(ref)
        calls=0;read=resolver._read
        def count(*a,**kw):
            nonlocal calls
            calls+=1;return read(*a,**kw)
        monkeypatch.setattr(resolver,'_read',count)
        reader=_SnapshotIndexReader(resolver,context.owner)
        index=OrderedIndex(reader,metadata['columnIndexes'][0],maximum_count=3000)
        for position in range(3000):index.locate_span(position)
        assert calls<20
        assert len(reader.state[0])<=128 and reader.state[1]<=2_097_152
        resolver.close()
        with pytest.raises(ValueError,match='closed'):index.locate_span(0)
