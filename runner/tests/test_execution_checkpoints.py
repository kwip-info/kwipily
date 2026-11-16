"""Cancellation remains bounded by work; rows do not each walk launch authority."""
import pytest
from runner.execution_checkpoints import RowCheckpoints
from runner.dataflow_union import pairs, plan
from runner.dataflow_ordering import checked_pairs
from runner.tests.test_dataflow_union import table, settings


def test_union_and_ordering_preserve_every_row_with_bounded_checks():
    calls=[]
    a=table(['value'],[(i,) for i in range(2048)])
    args={'table1':a,'table2':a};config=settings()
    columns,mappings,*_=plan(args,config)
    result=list(pairs(args,config,columns,mappings,lambda:calls.append(1)))
    assert [row for row,_ in result]==list(a.rows)*2
    assert len({rid for _,rid in result})==4096
    assert 1<len(calls)<40
    calls.clear()
    result=list(checked_pairs(a,lambda:calls.append(1)))
    assert [(row,rid) for _,row,rid in result]==list(zip(a.rows,a.row_ids))
    assert 1<len(calls)<20


def test_revocation_is_observed_before_another_work_batch():
    revoked=False
    def check():
        if revoked:raise RuntimeError('revoked')
    a=table(['value'],[(i,) for i in range(2048)])
    iterator=checked_pairs(a,check)
    assert next(iterator)[0]==0
    revoked=True
    seen=1
    with pytest.raises(RuntimeError,match='revoked'):
        for _ in iterator:seen+=1
    assert seen<=128


def test_wide_text_and_structured_rows_check_more_often():
    for row in [('x'*9000,),tuple(range(128)),({'untrusted':'structure'},)]:
        calls=[];guard=RowCheckpoints(lambda:calls.append(1))
        for _ in range(128):guard.admit(row)
        guard.finish()
        assert len(calls)>2


def test_finish_and_empty_iteration_observe_revocation():
    guard=RowCheckpoints(lambda:(_ for _ in ()).throw(RuntimeError('revoked')))
    with pytest.raises(RuntimeError,match='revoked'):guard.finish()
