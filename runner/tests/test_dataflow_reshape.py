from dataclasses import replace
import pytest
from runner.dataflow import Table, derive
from runner.dataflow_reshape import reshape_table, verify_output, defaults


def table(names, rows, types):
    return derive(Table(tuple(dict(name=n,dtype=t,nullable=any(r[i] is None for r in rows)) for i,(n,t) in enumerate(zip(names,types))),tuple(map(tuple,rows)),tuple(f's:{i}' for i in range(len(rows))),'a'*64))

def config(**changes):
    c=defaults();c.update(changes);return c

def test_long_order_missing_types_identity_and_cache():
    t=table(['id','jan','feb'],[(1,10,None),(2,20,25)],['int64']*3)
    c=config(idColumns=['id'],valueColumns=['feb','jan'])
    out=reshape_table(t,c)
    assert list(out.rows)==[(1,'feb',None),(1,'jan',10),(2,'feb',25),(2,'jan',20)]
    assert [x['name'] for x in out.columns]==['id','field','value']
    verify_output(t,c,out)
    with pytest.raises(ValueError,match='Cached'):verify_output(t,c,replace(out,source_digest='b'*64))
    drop=reshape_table(t,{**c,'missingValues':'drop'})
    assert list(drop.rows)==[(1,'jan',10),(2,'feb',25),(2,'jan',20)]
    assert set(drop.row_ids)<set(out.row_ids)

def test_wide_multiple_values_order_fill_and_duplicates():
    t=table(['id','month','sales','units'],[(1,'feb',12,2),(1,'jan',10,1),(2,'jan',30,3)],['int64','string','int64','int64'])
    c=config(mode='wide',rowKeys=['id'],columnKey='month',values=[dict(column='sales',aggregation='sum',prefix='sales_',nullPolicy='skip'),dict(column='units',aggregation='max',prefix='units_',nullPolicy='skip')])
    out=reshape_table(t,c)
    assert [x['name'] for x in out.columns]==['id','sales_feb','sales_jan','units_feb','units_jan']
    assert list(out.rows)==[(1,12,10,2,1),(2,None,30,None,3)]
    verify_output(t,c,out)
    assert list(reshape_table(t,{**c,'columnOrder':'sorted','missingCombinations':'fill','fillValue':0}).rows)==[(1,12,10,2,1),(2,0,30,0,3)]
    dup=table(['id','month','sales','units'],[(1,'jan',10,1),(1,'jan',20,2)],['int64','string','int64','int64'])
    with pytest.raises(ValueError,match='Repeated'):reshape_table(dup,c)
    assert list(reshape_table(dup,{**c,'duplicates':'aggregate'}).rows)==[(1,30,2)]

def test_refusals_widening_null_names_collisions_and_cancellation(monkeypatch):
    t=table(['id','a','b'],[(1,2**53+1,1.5)],['int64','int64','float64'])
    c=config(idColumns=['id'],valueColumns=['a','b'])
    with pytest.raises(ValueError,match='types'):reshape_table(t,c)
    with pytest.raises(ValueError,match='exact'):reshape_table(t,{**c,'typePolicy':'losslessNumeric'})
    for t2 in [replace(t,role='training'),replace(t,partition_digest='p'),replace(t,fitted_states=('f',))]:
        with pytest.raises(ValueError,match='before Partition'):reshape_table(t2,c)
    with pytest.raises(ValueError,match='disjoint'):reshape_table(t,{**c,'valueColumns':['id']})
    with pytest.raises(ValueError,match='names'):reshape_table(t,{**c,'fieldName':'id','valueColumns':['a']})
    t=table(['id','name','x'],[(1,None,2),(1,'a',3)],['int64','string','int64'])
    c=config(mode='wide',rowKeys=['id'],columnKey='name',values=[dict(column='x',aggregation='first',prefix='',nullPolicy='skip')])
    with pytest.raises(ValueError,match='Missing column'):reshape_table(t,c)
    assert list(reshape_table(t,{**c,'nullColumnNames':'drop'}).rows)==[(1,3)]
    assert list(reshape_table(t,{**c,'nullColumnNames':'label','nullLabel':'unknown'}).rows)==[(1,2,3)]
    with pytest.raises(ValueError,match='column limit'):reshape_table(t,{**c,'nullColumnNames':'label','maxColumns':2})
    def cancel():raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError,match='cancelled'):reshape_table(t,c,cancel)
    import runner.dataflow as df
    monkeypatch.setattr(df,'MAX_TABLE_CELLS',2)
    with pytest.raises(ValueError,match='cell'):reshape_table(t,{**c,'nullColumnNames':'label'})

@pytest.mark.parametrize('engine',['disk','polars'])
def test_long_streams_over_100k_and_replays(engine):
    from runner.polars_preparation import preparation_engine
    t=table(['a','b'],[(i,i+1) for i in range(50001)],['int64','int64'])
    c=config(valueColumns=['a','b'])
    with preparation_engine(engine):out=reshape_table(t,c)
    assert len(out.rows)==100002 and out.rows[-1]==('b',50001)
    verify_output(t,c,out)
