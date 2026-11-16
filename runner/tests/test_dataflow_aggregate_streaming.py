"""Differential semantics plus a real columnar read-complexity regression."""
from dataclasses import replace
from decimal import Decimal
import pytest
from runner import dataflow as df
from runner.dataflow_aggregate import aggregate_table as reference
from runner.dataflow_aggregate_streaming import aggregate_table as streamed
from runner.tests.test_dataflow_aggregate import config
from runner.tests.test_dataflow_join import table
from runner.dataflow_disk import disk_scope
from runner.workload_policy import WorkloadPolicy,workload_scope


def equivalent(actual,expected):
    for key in ('columns','source_digest','role','partition_digest','fitted_states','diagnostics','data_version'):
        assert getattr(actual,key)==getattr(expected,key),key
    assert tuple(actual.rows)==tuple(expected.rows)
    assert tuple(actual.row_ids)==tuple(expected.row_ids)
    actual.rows.store.close()


@pytest.mark.parametrize('rows,dtypes',[
    ([(2,10),(1,3),(2,None),(None,8),(1,9),(None,4)],['int64','int64']),
    ([(1,1e16),(1,1.0),(1,-1e16),(2,8.0)],['int64','float64']),
    ([(1,None),(1,None)],['int64','float64']),
    ([(1,2**53+1),(1,1.0)],['int64','float64'])])
def test_group_order_rational_values_membership_and_lineage_match_reference(rows,dtypes):
    source=table(rows,dtypes=dtypes)
    for selected in (config(),config(groupBy=[])):
        equivalent(streamed(source,selected),reference(source,selected))


@pytest.mark.parametrize('nulls',['skip','propagate','error'])
@pytest.mark.parametrize('values',[['0.10','0.20','-0.30'],['2.0',None,'-1'],['invalid',None,'2'],['1e16000','-1e16000','0.10']])
def test_exact_decimal_order_and_null_error_precedence(values,nulls):
    source=table([(1,v) for v in values],dtypes=['int64','string'])
    selected=config(aggregations=[{'output':'total','op':'sumDecimal','column':'value','nullPolicy':nulls}])
    try:expected=reference(source,selected)
    except ValueError as error:
        with pytest.raises(ValueError) as actual:streamed(source,selected)
        assert str(actual.value)==str(error)
    else:equivalent(streamed(source,selected),expected)


def test_null_key_drop_empty_groups_and_overflow_match_reference():
    cases=[(table([(1,4),(None,8)]),config(nullKeys='drop')),
           (table([(None,4)]),config(nullKeys='drop')),
           (table([(1,2**63-1),(1,1)]),config())]
    for source,selected in cases:
        try:expected=reference(source,selected)
        except ValueError as error:
            with pytest.raises(ValueError) as actual:streamed(source,selected)
            assert str(actual.value)==str(error)
        else:equivalent(streamed(source,selected),expected)


def test_duplicate_id_precedes_bad_value_and_does_not_leak_scratch():
    source=table([(1,3),(1,4)])
    source=replace(source,row_ids=('duplicate','duplicate'),rows=((1,'bad'),(1,4)))
    with disk_scope() as budget:
        with pytest.raises(ValueError,match='unique bounded observation'):streamed(source,config())
        assert budget.used==0


def test_columnar_groups_scan_each_input_chunk_once_instead_of_per_reducer(monkeypatch):
    from runner.polars_preparation import preparation_engine
    from runner.dataflow_columnar import ColumnarStore
    checks=[0]
    def check():checks[0]+=1
    count=9000;groups=73
    with workload_scope(WorkloadPolicy('balanced')),disk_scope(check_cancelled=check),preparation_engine('polars'):
        columns=({'name':'key','dtype':'int64','nullable':False},{'name':'money','dtype':'string','nullable':False})
        store=ColumnarStore(columns,check)
        expected={}
        for i in range(count):
            value=f'{i%101}.{i%100:02d}';key=i%groups
            store.append((key,value),f'row:{i:08d}')
            expected[key]=expected.get(key,Decimal(0))+Decimal(value)
        rows,ids=store.finish();source=df.derive(df.Table(columns,rows,ids,'a'*64))
        reads=[0];original=store._read_chunk
        def counted(index):reads[0]+=1;return original(index)
        monkeypatch.setattr(store,'_read_chunk',counted);checks[0]=0
        selected={'groupBy':['key'],'nullKeys':'error','maxRows':100000,'aggregations':[
            {'op':'countRows','output':'count'},
            {'op':'sumDecimal','column':'money','output':'total','nullPolicy':'error'}]}
        result=streamed(source,selected,check)
        assert reads[0]<=len(store.chunks)+1
        assert checks[0]<count,checks[0]
        assert [r[0] for r in result.rows]==list(expected)
        for key,n,total in result.rows:
            assert n==sum(i%groups==key for i in range(count))
            assert Decimal(total)==expected[key]
        result.rows.store.close();store.close()


def test_cancellation_cleans_owned_staging_and_preserves_shared_budget():
    source=table([(i%7,i) for i in range(1000)])
    calls=[0]
    def cancel():
        calls[0]+=1
        if calls[0]>15:raise KeyboardInterrupt()
    with disk_scope() as budget:
        before=budget.used
        with pytest.raises(KeyboardInterrupt):streamed(source,config(),cancel)
        assert budget.used==before


def test_kernel_classification_covers_the_actual_registry_and_refuses_forged_choices():
    from runner.preparation_kernels import KNOWN,build_kernel_plan,validate_kernel_plan
    assert KNOWN=={item['id'].removeprefix(df.PREFIX) for item in df.REGISTRY['operators']}
    nodes={'summary':('aggregate',config())}
    plan=build_kernel_plan(nodes)
    assert plan['nodes']['summary']['kernelID']=='aggregate.exactStreaming.v1'
    assert not plan['nodes']['summary']['parallel']
    plan['nodes']['summary']['parallel']=0
    with pytest.raises(ValueError):validate_kernel_plan(plan,nodes)
    text=config(aggregations=[{'op':'joinText','column':'value','output':'joined','nullPolicy':'skip','separator':',','distinct':False,'skipEmpty':False}])
    assert build_kernel_plan({'text':('aggregate',text)})['nodes']['text']['reason']=='orderedTextRequiresReference'


@pytest.mark.parametrize('grouping',[['key'],[]])
@pytest.mark.parametrize('reducer',['countRows','countNonNull','sum','mean','min','max'])
@pytest.mark.parametrize('null_policy',['skip','propagate','error'])
def test_native_projection_reducer_parity_with_nulls(tmp_path,grouping,reducer,null_policy):
    from runner.dataflow_columnar import ColumnarStore
    from runner.polars_preparation import preparation_engine
    from runner.dataflow_native_relational import qualified
    columns=({'name':'key','dtype':'int64','nullable':True},{'name':'value','dtype':'float64','nullable':True},{'name':'unused','dtype':'string','nullable':False})
    raw=((2,1e16,'x'),(1,None,'unused'),(2,1.0,'xx'),(2,-1e16,'xxx'),(None,8.0,'xxxx'))
    aggregation={'output':'out','op':reducer}
    if reducer!='countRows':aggregation.update(column='value',nullPolicy=null_policy)
    selected={'groupBy':grouping,'nullKeys':'group','aggregations':[aggregation],'maxRows':100}
    with disk_scope(directory=tmp_path),preparation_engine('polars'):
        store=ColumnarStore(columns,directory=tmp_path)
        for i,row in enumerate(raw):store.append(row,f'r:{i}')
        rows,ids=store.finish();source=df.derive(df.Table(columns,rows,ids,'a'*64));assert qualified(source)
        plain=replace(source,rows=raw,row_ids=tuple(ids))
        try:expected=reference(plain,selected)
        except ValueError as error:
            with pytest.raises(ValueError) as actual:streamed(source,selected)
            assert str(actual.value)==str(error)
        else:equivalent(streamed(source,selected),expected)
        store.close()
