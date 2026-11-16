"""Owned native buffers stay resident until the captured arena needs a spill."""
import pytest
import polars as pl
from runner.dataflow_columnar import ColumnarStore
from runner.dataflow_disk import disk_scope

COLUMNS=({'name':'value','dtype':'int64','nullable':False},)

def test_native_buffers_avoid_ipc_files_until_pressure_and_release_all_custody(tmp_path):
    from runner.native_frame_arena import native_buffer_scope
    with disk_scope(directory=tmp_path) as scratch,native_buffer_scope(maximum_bytes=100_000) as arena:
        stores=[]
        for group in range(5):
            store=ColumnarStore(COLUMNS,directory=tmp_path)
            assert store.append_frame(pl.int_range(0,2000,eager=True,dtype=pl.Int64).rename('c0').to_frame(),pl.Series('id',[f'{group}:{i}' for i in range(2000)]))
            rows,ids=store.finish();stores.append(store)
            assert rows[1999]==(1999,)
            if group==0:assert not list(store.directory.glob('*.arrow'))
        assert arena.report()['spillBytes']>0
        assert arena.report()['residentBytes']<=arena.limit
        for store in stores:
            assert store._read_chunk(0).get_column('c0').to_list()==list(range(2000))
        for store in stores:store.close()
        assert arena.report()['residentBytes']==0
        assert scratch.used==0


def test_resident_reads_keep_live_cancellation(tmp_path):
    from runner.native_frame_arena import native_buffer_scope
    cancelled=False
    def check():
        if cancelled:raise InterruptedError('cancelled')
    with disk_scope(directory=tmp_path),native_buffer_scope(maximum_bytes=100_000):
        store=ColumnarStore(COLUMNS,check_cancelled=check,directory=tmp_path)
        store.append((1,),'r:1');rows,_=store.finish()
        assert rows[0]==(1,)
        cancelled=True
        with pytest.raises(InterruptedError):rows[0]
        store.close()
