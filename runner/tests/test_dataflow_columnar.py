import json
from pathlib import Path
import pytest
from runner.dataflow_columnar import ColumnarStore, ColumnarRows, ColumnarIDs
from runner.dataflow_disk import ScratchBudget, disk_scope, iter_json

COLUMNS=({'name':'__kwip_id','dtype':'string','nullable':True},{'name':'number','dtype':'int64','nullable':False})

def test_concurrent_readers_have_independent_readonly_handles_and_scoped_cleanup(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from runner.dataflow_columnar import columnar_reader_scope
    import threading,sqlite3
    store=ColumnarStore(COLUMNS,directory=tmp_path)
    for i in range(9000):store.append((str(i),i),str(i))
    rows,ids=store.finish();writer=store.connection
    barrier=threading.Barrier(4,timeout=5)
    def read():
        with columnar_reader_scope():
            connection=store.connection
            assert connection is not writer
            with pytest.raises(sqlite3.OperationalError):connection.execute('DELETE FROM ids')
            barrier.wait()
            assert list(ids)==list(map(str,range(9000)))
            assert sum(row[1] for row in rows)==sum(range(9000))
            return id(connection)
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:handles=list(pool.map(lambda _:read(),range(4)))
        assert len(set(handles))==4 and store.active_reader_count==0
        assert writer.execute('SELECT COUNT(*) FROM ids').fetchone()==(9000,)
    finally:store.close()


def test_reader_scope_closes_an_early_cursor_without_a_late_generator_error(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from runner.dataflow_columnar import columnar_reader_scope
    store=ColumnarStore(COLUMNS,directory=tmp_path);store.append(('one',1),'one');store.finish()
    def early():
        with columnar_reader_scope():
            cursor=store.connection.execute('SELECT identity FROM ids');assert cursor.fetchone()==('one',)
            return cursor
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:cursor=pool.submit(early).result()
        cursor.close()
        with pytest.raises(ValueError,match='closed'):cursor.fetchone()
        assert store.active_reader_count==0
    finally:store.close()


def test_closing_owner_store_closes_its_live_cursor_before_late_generator_cleanup(tmp_path):
    store=ColumnarStore(COLUMNS,directory=tmp_path)
    for i in range(3):store.append((str(i),i),str(i))
    rows,ids=store.finish();iterator=rows.iter_pairs();next(iterator)
    store.close()
    iterator.close() # Must be idempotent after ownership teardown, not unraisable.


def test_finalized_index_is_readonly_and_new_reader_refuses_replaced_path(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import sqlite3
    store=ColumnarStore(COLUMNS,directory=tmp_path);store.append(('one',1),'one');store.finish()
    try:
        with pytest.raises(sqlite3.OperationalError):store.connection.execute('DELETE FROM ids')
        original=store.directory/'ids.sqlite3';original.rename(store.directory/'preserved.sqlite3')
        foreign=tmp_path/'foreign';foreign.write_bytes(b'not a database')
        original.symlink_to(foreign)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(lambda:store.connection)
            with pytest.raises(ValueError,match='index changed'):future.result()
        assert store.active_reader_count==0 and foreign.read_bytes()==b'not a database'
    finally:store.close()

def test_paired_read_checks_cached_chunk_and_rejects_closed_store(tmp_path):
    cancelled=[False]
    def check():
        if cancelled[0]:raise InterruptedError('cancelled')
    store=ColumnarStore(COLUMNS,directory=tmp_path,check_cancelled=check,identity_prefix='s:')
    store.append(('value',7),'s:one');rows,ids=store.finish()
    assert rows.pair_at(0)==(('value',7),'s:one')
    cancelled[0]=True
    with pytest.raises(InterruptedError):rows.pair_at(0)
    cancelled[0]=False
    store.close()
    with pytest.raises(ValueError,match='closed'):rows.pair_at(0)
    for bad in (True,-1,1,slice(None)):
        with pytest.raises(IndexError):rows.pair_at(bad)


def test_selected_pair_admission_checks_byte_work_before_256_large_rows(tmp_path):
    cancelled=[False]
    def check():
        if cancelled[0]:raise InterruptedError('cancelled')
    store=ColumnarStore(({'name':'text','dtype':'string','nullable':False},),directory=tmp_path,check_cancelled=check)
    try:
        for i in range(10):store.append(('\x01'*100000,),str(i))
        rows,ids=store.finish();selected=rows.iter_selected_pairs(range(10));next(selected)
        cancelled[0]=True
        with pytest.raises(InterruptedError):
            for _ in range(4):next(selected)
    finally:store.close()

def test_columnar_roundtrip_order_schema_collision_and_random_access(tmp_path):
    store=ColumnarStore(COLUMNS,directory=tmp_path,identity_prefix='source:')
    values=[('c',3), (None,2), ('a\n\"😀',1)]
    for i,row in enumerate(values):store.append(row,f'source:{i}')
    rows,ids=store.finish()
    assert isinstance(rows,ColumnarRows) and isinstance(ids,ColumnarIDs)
    assert list(rows)==values and list(ids)==['source:0','source:1','source:2']
    assert rows[-1]==values[-1] and rows[::2]==tuple(values[::2])
    assert ids[1]=='source:1' and 'source:1' in ids and 'other:1' not in ids
    assert rows==tuple(values) and ids==tuple(list(ids))
    assert json.loads(b''.join(iter_json(rows)))==[list(r) for r in values]
    assert list(ids.sorted_iter())==sorted(ids)
    store.close(); assert list(tmp_path.iterdir())==[]

def test_columnar_multiple_bounded_chunks_and_prefix_lexical_ids(tmp_path):
    store=ColumnarStore(({'name':'x','dtype':'string','nullable':False},),directory=tmp_path,identity_prefix='s:')
    for i in range(20000):store.append((str(i),),f's:{i}')
    rows,ids=store.finish()
    assert len(list(store.directory.glob('*.arrow')))>=3
    assert all(p.stat().st_size<=2*1024*1024 for p in store.directory.glob('*.arrow'))
    assert len(rows)==20000 and rows[8192]==('8192',) and rows[-1]==('19999',)
    assert list(ids.sorted_iter())==sorted(f's:{i}' for i in range(20000))
    store.close()

def test_columnar_shared_budget_refusal_removes_owned_files(tmp_path):
    budget=ScratchBudget(50000)
    with disk_scope(directory=tmp_path,budget=budget):
        store=ColumnarStore(({'name':'x','dtype':'string','nullable':False},))
        with pytest.raises(ValueError,match='scratch'):
            for i in range(100):store.append(('x'*1000,),str(i))
            store.finish()
    assert budget.used==0 and list(tmp_path.iterdir())==[]

@pytest.mark.parametrize('row',[(True,), (2**63,), (None,), ('oops',)])
def test_columnar_rejects_wrong_typed_values_and_cleans_up(tmp_path,row):
    store=ColumnarStore(({'name':'x','dtype':'int64','nullable':False},),directory=tmp_path)
    with pytest.raises(ValueError):store.append(row,'a');store.finish()
    assert list(tmp_path.iterdir())==[]

def test_columnar_duplicate_ids_refused_across_chunks(tmp_path):
    store=ColumnarStore(({'name':'x','dtype':'int64','nullable':False},),directory=tmp_path)
    with pytest.raises(ValueError,match='unique'):
        for i in range(9000):store.append((i,),str(i if i<8999 else 0))
        store.finish()
    assert list(tmp_path.iterdir())==[]

def test_columnar_cancellation_and_scope_inheritance(tmp_path):
    calls=0
    def check():
        nonlocal calls
        calls+=1
        if calls>3:raise RuntimeError('cancelled')
    with disk_scope(directory=tmp_path,check_cancelled=check):
        store=ColumnarStore(({'name':'x','dtype':'int64','nullable':False},))
        with pytest.raises(RuntimeError,match='cancelled'):
            for i in range(1000):store.append((i,),str(i))
            store.finish()
    assert list(tmp_path.iterdir())==[]

def test_columnar_exact_scratch_accounting_and_two_stores(tmp_path):
    budget=ScratchBudget()
    with disk_scope(directory=tmp_path,budget=budget):
        a=ColumnarStore(COLUMNS);b=ColumnarStore(COLUMNS)
        a.append(('a',1),'a');a.finish();b.append(('b',2),'b');b.finish()
        measured=sum(p.stat().st_size for p in tmp_path.rglob('*') if p.is_file())
        assert budget.used==measured
        a.close();assert budget.used==sum(p.stat().st_size for p in tmp_path.rglob('*') if p.is_file())
        b.close();assert budget.used==0

def test_columnar_empty_table_and_large_record_chunking(tmp_path):
    columns=({'name':'x','dtype':'string','nullable':True},)
    empty=ColumnarStore(columns,directory=tmp_path);rows,ids=empty.finish()
    assert len(rows)==0 and list(rows)==[] and list(ids)==[]
    empty.close()
    store=ColumnarStore(columns,directory=tmp_path)
    for i in range(8):store.append(('a'*700000,),str(i))
    rows,ids=store.finish()
    assert len(rows)==8 and rows[5]==('a'*700000,)
    assert all(p.stat().st_size<=2*1024*1024 for p in store.directory.glob('*.arrow'))
    store.close()

def test_columnar_pairs_use_same_order_across_chunks(tmp_path):
    store=ColumnarStore(COLUMNS,directory=tmp_path,identity_prefix='s:')
    for i in range(5000):store.append((str(i),i),f's:{i}')
    rows,ids=store.finish()
    assert list(rows.iter_pairs())==list(zip(rows,ids))
    store.close()

def test_columnar_record_budget_and_duplicate_during_append_cleanup(tmp_path):
    columns=({'name':'x','dtype':'string','nullable':False},)
    huge=ColumnarStore(columns,directory=tmp_path)
    with pytest.raises(ValueError,match='1 MiB'):huge.append(('a'*1048576,),'a')
    assert list(tmp_path.iterdir())==[]
    store=ColumnarStore(columns,directory=tmp_path)
    with pytest.raises(ValueError,match='unique'):
        for i in range(9000):store.append(('a',),str(i if i<4096 else i-4096))
    assert list(tmp_path.iterdir())==[]

def test_columnar_ipc_overhead_splits_before_write(tmp_path,monkeypatch):
    from runner import dataflow_columnar as m
    monkeypatch.setattr(m,'MAX_BATCH_BYTES',10000)
    columns=tuple({'name':f'c{i}','dtype':'bool','nullable':False} for i in range(20))
    store=ColumnarStore(columns,directory=tmp_path)
    for i in range(100):store.append(tuple(bool(i%2) for _ in columns),str(i))
    rows,ids=store.finish()
    assert len(rows)==100 and all(p.stat().st_size<=10000 for p in store.directory.glob('*.arrow'))
    store.close()

def test_columnar_writer_wrapped_overflow_splits_without_changing_rows(tmp_path,monkeypatch):
    from runner import dataflow_columnar as m
    original=m._IPCBuffer.write
    calls=0
    def first_overflow(self,content):
        nonlocal calls
        calls+=1
        if calls==1:
            self.overflow=True
            raise m._ChunkOverflow('test bounded write')
        return original(self,content)
    monkeypatch.setattr(m._IPCBuffer,'write',first_overflow)
    store=ColumnarStore(({'name':'x','dtype':'int64','nullable':False},),directory=tmp_path)
    for i in range(10):store.append((i,),str(i))
    rows,ids=store.finish()
    assert len(store.chunks)==2 and list(rows)==[(i,) for i in range(10)] and list(ids)==list(map(str,range(10)))
    store.close()


def test_native_scalar_batch_writes_without_scalar_row_appends(tmp_path,monkeypatch):
    import polars as pl
    from runner.dataflow_columnar import ColumnarStore
    columns=({'name':'n','dtype':'int64','nullable':False},{'name':'text','dtype':'string','nullable':True},{'name':'x','dtype':'float64','nullable':True})
    store=ColumnarStore(columns,directory=tmp_path,identity_prefix='batch:')
    frame=pl.DataFrame({'c0':pl.Series([2**60+1,-1,0],dtype=pl.Int64),'c1':['é😀',None,''], 'c2':[-0.0,1.5,None]})
    def refuse(*a,**kw):raise AssertionError('Native payload batch became Python row appends')
    monkeypatch.setattr(store,'append',refuse)
    try:
        assert store.append_frame(frame,pl.Series(['batch:a','batch:b','batch:c'])) is True
        rows,ids=store.finish()
        assert list(rows)==[(2**60+1,'é😀',-0.0),(-1,None,1.5),(0,'',None)]
        assert list(ids)==['batch:a','batch:b','batch:c']
    finally:store.close()


def test_native_batch_rejects_nonfinite_and_duplicate_identity_transactionally(tmp_path):
    import polars as pl
    from runner.dataflow_columnar import ColumnarStore
    import pytest
    for i,(values,ids) in enumerate([([float('inf')],['r']),([1.0,2.0],['r','r'])]):
        parent=tmp_path/str(i);parent.mkdir();store=ColumnarStore(({'name':'x','dtype':'float64','nullable':False},),directory=parent)
        with pytest.raises(ValueError):store.append_frame(pl.DataFrame({'c0':values}),pl.Series(ids))
        assert store.closed and not list(parent.iterdir())
