"""Relational execution must not reconstruct already admitted payload rows."""
from dataclasses import replace
import pytest
from runner import dataflow as df
from runner.dataflow_columnar import ColumnarStore
from runner.dataflow_disk import disk_scope
from runner.dataflow_union import union_tables
from runner.dataflow_sort import sort_table
from runner.polars_preparation import preparation_engine

COLUMNS=({'name':'key','dtype':'int64','nullable':True},{'name':'text','dtype':'string','nullable':False})

def native(rows,directory,prefix):
    store=ColumnarStore(COLUMNS,directory=directory,identity_prefix=prefix)
    for i,row in enumerate(rows):store.append(row,prefix+str(i))
    values,ids=store.finish()
    return df.derive(df.Table(COLUMNS,values,ids,'a'*64))

def test_union_uses_parent_buffers_and_keeps_identity_and_order(tmp_path,monkeypatch):
    from runner import dataflow_union as module
    with disk_scope(directory=tmp_path),preparation_engine('polars'):
        a=native([(2,'é'),(None,'😀')],tmp_path,'a:');b=native([(1,'b')],tmp_path,'b:')
        config={'inputs':['table1','table2'],'match':'name','coverage':'exact','typePolicy':'strict','sourceColumn':'origin'}
        expected=union_tables({'table1':replace(a,rows=tuple(a.rows),row_ids=tuple(a.row_ids)),'table2':replace(b,rows=tuple(b.rows),row_ids=tuple(b.row_ids))},config)
        def forbidden(*args,**kwargs):raise AssertionError('Reference Python payload iterator used')
        monkeypatch.setattr(module,'pairs',forbidden)
        actual=union_tables({'table1':a,'table2':b},config)
        assert tuple(actual.rows)==tuple(expected.rows)
        assert tuple(actual.row_ids)==tuple(expected.row_ids)
        assert actual.data_version==expected.data_version
        assert actual.diagnostics==expected.diagnostics
        assert actual.rows.store.parents==(a.rows.store,b.rows.store)
        actual.rows.store.close();assert not a.rows.store.closed
        a.rows.store.close();b.rows.store.close();expected.rows.store.close()

@pytest.mark.parametrize('direction',['ascending','descending'])
@pytest.mark.parametrize('missing',['first','last'])
def test_native_sort_is_stable_and_keeps_exact_values_and_identity(tmp_path,monkeypatch,direction,missing):
    from runner import dataflow_sort as module
    with disk_scope(directory=tmp_path),preparation_engine('polars'):
        table=native([(2,'a'),(None,'n'),(1,'é'),(2,'b'),(None,'m')],tmp_path,'s:')
        config={'keys':[{'column':'key','direction':direction,'missing':missing,'comparison':'exact'}]}
        expected=sort_table(replace(table,rows=tuple(table.rows),row_ids=tuple(table.row_ids)),config)
        def forbidden(*args,**kwargs):raise AssertionError('Reference Python payload iterator used')
        monkeypatch.setattr(module,'checked_pairs',forbidden)
        actual=sort_table(table,config)
        assert tuple(actual.rows)==tuple(expected.rows)
        assert tuple(actual.row_ids)==tuple(expected.row_ids)
        assert actual.data_version==expected.data_version
        assert actual.diagnostics==expected.diagnostics
        table.rows.store.close()
        with pytest.raises(ValueError,match='closed'):actual.rows[0]
        actual.rows.store.close();expected.rows.store.close()

@pytest.mark.parametrize('mode',['inner','left','right','full'])
@pytest.mark.parametrize('null_policy',['neverMatch','match'])
def test_native_join_keeps_parent_payloads_and_exact_pair_identities(tmp_path,monkeypatch,mode,null_policy):
    from runner import dataflow_join as module
    with disk_scope(directory=tmp_path),preparation_engine('polars'):
        a=native([(2,'a'),(None,'n'),(1,'é'),(2,'b')],tmp_path,'a:')
        b=native([(2,'x'),(3,'y'),(None,'z')],tmp_path,'b:')
        config={'leftKeys':['key'],'rightKeys':['key'],'mode':mode,'nullPolicy':null_policy,'cardinality':'manyToMany','leftPrefix':'Left','rightPrefix':'Right','maxRows':100}
        expected=module.join_tables(replace(a,rows=tuple(a.rows),row_ids=tuple(a.row_ids)),replace(b,rows=tuple(b.rows),row_ids=tuple(b.row_ids)),config)
        def forbidden(*args,**kwargs):raise AssertionError('Join reconstructed payload rows in Python')
        monkeypatch.setattr(module.JoinPlan,'iter_rows',forbidden)
        actual=module.join_tables(a,b,config)
        assert tuple(actual.rows)==tuple(expected.rows)
        assert tuple(actual.row_ids)==tuple(expected.row_ids)
        assert actual.data_version==expected.data_version
        assert actual.diagnostics==expected.diagnostics
        actual.rows.store.close();a.rows.store.close();b.rows.store.close()


def test_native_join_keeps_work_checks_independent_of_physical_batch_size(tmp_path):
    from runner.dataflow_join import join_tables
    with disk_scope(directory=tmp_path),preparation_engine('polars'):
        a=native([(i,'a') for i in range(1000)],tmp_path,'a:');b=native([(1,'b')],tmp_path,'b:')
        c={'leftKeys':['key'],'rightKeys':['key'],'mode':'left','nullPolicy':'neverMatch','cardinality':'manyToOne','leftPrefix':'L','rightPrefix':'R','maxRows':2000}
        actual=join_tables(a,b,c)
        assert len(actual.rows)==1000
        assert len(actual.rows.store.chunks)==1
        actual.rows.store.close();a.rows.store.close();b.rows.store.close()
