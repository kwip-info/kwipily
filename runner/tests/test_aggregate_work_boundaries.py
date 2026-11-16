"""Small groups share a bounded work interval, not two full checks per group."""
from runner import dataflow as df
from runner.dataflow_aggregate_streaming import aggregate_table
from runner.dataflow_disk import disk_scope


def test_many_single_row_groups_share_cooperative_work_boundaries(tmp_path):
    calls=[]
    def check():calls.append(1)
    columns=({'name':'group','dtype':'int64','nullable':False},)
    table=df.Table(columns,tuple((i,) for i in range(1000)),tuple(f'r:{i}' for i in range(1000)),'a'*64)
    config={'groupBy':['group'],'nullKeys':'group','aggregations':[{'output':'count','op':'countRows'}],'maxRows':2000}
    with disk_scope(directory=tmp_path):
        actual=aggregate_table(table,config,check)
        assert tuple(actual.rows)==tuple((i,1) for i in range(1000))
        assert len(calls)<600
        actual.rows.store.close()


def test_owned_native_aggregate_reads_only_reducer_and_key_columns(tmp_path,monkeypatch):
    from runner.dataflow_columnar import ColumnarStore,ColumnarRows
    from runner.polars_preparation import preparation_engine
    columns=({'name':'group','dtype':'int64','nullable':False},{'name':'amount','dtype':'string','nullable':False},{'name':'unused','dtype':'string','nullable':False})
    config={'groupBy':['group'],'nullKeys':'group','aggregations':[{'output':'total','op':'sumDecimal','column':'amount','nullPolicy':'error'}],'maxRows':20}
    with disk_scope(directory=tmp_path),preparation_engine('polars'):
        store=ColumnarStore(columns,directory=tmp_path)
        for i in range(10):store.append((i%2,'1.95','large unrelated data'*100),f'r:{i}')
        rows,ids=store.finish();table=df.derive(df.Table(columns,rows,ids,'a'*64))
        def forbidden(*a,**k):raise AssertionError('Full payload rows reconstructed for Aggregate')
        monkeypatch.setattr(ColumnarRows,'__iter__',forbidden)
        actual=aggregate_table(table,config)
        assert tuple(actual.rows)==((0,'9.75'),(1,'9.75'))
        actual.rows.store.close();store.close()
