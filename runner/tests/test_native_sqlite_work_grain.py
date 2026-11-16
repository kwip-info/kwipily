"""Native VM work is bounded by instructions, not a full grant walk per1000ops."""
import time
import pytest
from runner.dataflow_disk import disk_scope
from runner.native_frame_arena import native_buffer_scope
from runner.dataflow_ordering import scratch


def test_captured_native_vm_grain_interrupts_a_real_expensive_query(tmp_path):
    checks=[]
    def check():
        checks.append(1)
        if len(checks)>1:raise InterruptedError('revoked')
    with disk_scope(directory=tmp_path),native_buffer_scope(sqlite_instructions=32768):
        before=time.monotonic()
        with pytest.raises(InterruptedError,match='revoked'):
            with scratch(check) as (db,account):
                db.execute('WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n WHERE x<10000) SELECT sum(a.x*b.x) FROM n a,n b').fetchone()
        assert time.monotonic()-before<0.5
    assert len(checks)==2


def test_vm_grain_is_a_qualified_physical_choice_not_an_unbounded_number():
    with pytest.raises(ValueError):
        with native_buffer_scope(sqlite_instructions=10**9):pass
