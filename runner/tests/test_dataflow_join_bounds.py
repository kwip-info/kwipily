"""Join resource preflight and reusable exact-output verification."""
from dataclasses import replace
import pytest
from runner import dataflow as df
from runner import dataflow_join as join
from runner.tests.test_dataflow_join import table, settings


def test_payload_expansion_rejected_before_output_derivation(monkeypatch):
    left=table([(1,'é\\"'*20000)],dtypes=['int64','string'])
    right=table([(1,i) for i in range(400)],source='b')
    def forbidden(*args,**kwargs):raise AssertionError('output derived before byte preflight')
    monkeypatch.setattr(df,'derive',forbidden)
    with pytest.raises(ValueError,match='byte'):join.join_tables(left,right,settings())


def test_shared_diagnostics_do_not_multiply_and_distinct_history_is_preserved():
    a=table([(1,2)]);b=table([(1,3)],source='b')
    history=tuple({'kind':'observation','n':i} for i in range(100))
    result=join.join_tables(replace(a,diagnostics=history),replace(b,diagnostics=history),settings())
    assert result.diagnostics[:-1]==history
    assert len(result.diagnostics)==101


def test_distinct_diagnostic_overflow_rejected_before_derivation(monkeypatch):
    a=replace(table([(1,2)]),diagnostics=tuple({'n':i} for i in range(16384)))
    b=table([(1,3)],source='b')
    monkeypatch.setattr(df,'derive',lambda *a,**k:pytest.fail('derived before history bound'))
    with pytest.raises(ValueError,match='diagnostic'):join.join_tables(a,b,settings())


@pytest.mark.parametrize('mode',['inner','left','right','full'])
def test_plan_size_equals_serialized_artifact_and_iterator(mode):
    left=table([(1,'é\\"'),(2,'x')],dtypes=['int64','string'])
    right=table([(1,4),(3,8)],source='b')
    plan=join.plan_join(left,right,settings(mode=mode))
    result=join.join_tables(left,right,settings(mode=mode))
    assert list(plan.iter_rows())==list(zip(result.rows,result.row_ids))
    lineage={**result.evidence(),'dataVersion':result.data_version,'partitionDigest':result.partition_digest,
             'fittedStateDigests':list(result.fitted_states),'diagnostics':list(result.diagnostics)}
    artifact={'schemaVersion':1,'kind':'table','columns':result.columns,'rows':result.rows,'lineage':lineage}
    assert plan.artifact_bytes==len(df.json.dumps(artifact,allow_nan=False,sort_keys=True,separators=(',',':')).encode())
    join.verify_join_output(left,right,settings(mode=mode),result)
    bad=df.derive(result,rows=[('bad',)+row[1:] for row in result.rows])
    with pytest.raises(ValueError,match='Cached join'):join.verify_join_output(left,right,settings(mode=mode),bad)


def test_reconverging_chain_remains_bounded_without_losing_unique_events():
    current=table([(1,2)])
    for _ in range(24):
        combined=join.join_tables(current,current,settings())
        current=df.derive(combined,columns=current.columns,rows=[r[:2] for r in combined.rows])
    assert len(current.diagnostics)==24
    assert len(current.rows)==1


def test_metadata_bytes_count_toward_preflight(monkeypatch):
    a=replace(table([(1,2)]),diagnostics=({'note':'é'*1000},))
    b=table([(1,3)],source='b')
    monkeypatch.setattr(join,'MAX_ARTIFACT_BYTES',5000)
    monkeypatch.setattr(df,'derive',lambda *a,**k:pytest.fail('derived before metadata budget'))
    with pytest.raises(ValueError,match='byte'):join.join_tables(a,b,settings())
