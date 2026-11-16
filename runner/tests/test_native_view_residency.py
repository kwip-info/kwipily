"""Repeated inspection of one virtual batch must reuse its admitted native frame."""
from runner.dataflow_disk import disk_scope
from runner.native_frame_arena import native_buffer_scope,current_arena
from runner.tests.test_native_relational_batches import native
from runner.dataflow_sort import sort_table


def test_virtual_sort_batch_is_not_rebuilt_for_each_row(tmp_path,monkeypatch):
    with disk_scope(directory=tmp_path),native_buffer_scope(maximum_bytes=1_000_000):
        table=native([(100-i,str(i)) for i in range(100)],tmp_path,'s:')
        output=sort_table(table,{'keys':[{'column':'key','direction':'ascending','missing':'last','comparison':'exact'}]})
        # Drop any frame populated by canonical output hashing first.
        current_arena().release(output.rows.store)
        calls=[];read=table.rows.store._read_chunk
        def observed(index):calls.append(index);return read(index)
        monkeypatch.setattr(table.rows.store,'_read_chunk',observed)
        assert [output.rows[i][0] for i in range(100)]==list(range(1,101))
        assert calls==[0]
        table.rows.store.close()
        try:output.rows[0]
        except ValueError:pass
        else:raise AssertionError('Cached virtual frame bypassed parent lifetime')
        output.rows.store.close()
