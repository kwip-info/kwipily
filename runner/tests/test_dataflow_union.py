from dataclasses import replace
import pytest
from runner.dataflow import Table, derive
from runner.dataflow_union import union_tables

def table(names, rows, types=None):
    types=types or ['int64']*len(names)
    return derive(Table(tuple(dict(name=n,dtype=t,nullable=any(r[i] is None for r in rows)) for i,(n,t) in enumerate(zip(names,types))),tuple(map(tuple,rows)),tuple(f'same:{i}' for i in range(len(rows))),'a'*64))

def config(**changes):
    return dict(inputs=['table1','table2'],match='name',coverage='all',typePolicy='strict',sourceColumn='',**changes)

def settings(**changes):
    value=config();value.update(changes);return value

def test_name_alignment_null_fill_duplicates_and_stable_port_identity():
    a=table(['id','value'],[(1,2),(1,2)])
    b=table(['value','other'],[(3,4)])
    args={'table1':a,'table2':b}
    result=union_tables(args,settings(sourceColumn='origin'))
    assert list(result.rows)==[(1,2,None,'table1'),(1,2,None,'table1'),(None,3,4,'table2')]
    assert [c['name'] for c in result.columns]==['id','value','other','origin']
    assert len(set(result.row_ids))==3
    reverse=union_tables(args,settings(inputs=['table2','table1']))
    assert list(reverse.rows)==[(3,4,None),(2,None,1),(2,None,1)]
    assert set(reverse.row_ids)==set(result.row_ids)

def test_position_shared_exact_and_numeric_widening():
    a=table(['id','value'],[(1,2)])
    b=table(['different'],[(3.5,)],['float64'])
    with pytest.raises(ValueError,match='types'):union_tables({'table1':a,'table2':b},settings(match='position'))
    result=union_tables({'table1':a,'table2':b},settings(match='position',coverage='shared',typePolicy='losslessNumeric'))
    assert list(result.rows)==[(1.0,),(3.5,)]
    assert result.columns[0]['name']=='id' and result.columns[0]['dtype']=='float64'
    for match in ['name','position']:
        with pytest.raises(ValueError,match='identical'):union_tables({'table1':a,'table2':b},settings(match=match,coverage='exact'))
    b=table(['value','id'],[(4,3)])
    assert list(union_tables({'table1':a,'table2':b},settings(coverage='exact')).rows)==[(1,2),(3,4)]

def test_lossy_numbers_lineage_and_source_collision_refused():
    a=table(['x'],[(2**53+1,)])
    b=table(['x'],[(1.5,)],['float64'])
    with pytest.raises(ValueError,match='exactly'):union_tables({'table1':a,'table2':b},settings(typePolicy='losslessNumeric'))
    for changed in [replace(a,role='training'),replace(a,partition_digest='p'),replace(a,fitted_states=('f',))]:
        with pytest.raises(ValueError,match='before partitioning'):union_tables({'table1':changed,'table2':a},settings())
    with pytest.raises(ValueError,match='identifier'):union_tables({'table1':a,'table2':a},settings(sourceColumn='x'))

@pytest.mark.parametrize('changes',[{'inputs':['table1','table1']},{'inputs':['table1']},{'inputs':['table1','bad']},{'match':'guess'},{'coverage':'guess'},{'typePolicy':'guess'},{'sourceColumn':' '},{'extra':1}])
def test_invalid_config(changes):
    a=table(['x'],[(1,)])
    with pytest.raises(ValueError):union_tables({'table1':a,'table2':a},settings(**changes))

@pytest.mark.parametrize("engine",["disk","polars"])
def test_more_than_100k_rows_stream_and_cancel(engine):
    a=table(['x'],[(i,) for i in range(60000)])
    from runner.polars_preparation import preparation_engine
    with preparation_engine(engine):
        from runner.dataflow import _execute_node
        result=_execute_node('union',settings(),{'table1':a,'table2':a},'',None,lambda:None)['table']
    assert len(result.rows)==120000 and result.rows[-1]==(59999,)
    assert result.row_ids[0]!=result.row_ids[60000]
    def cancel():raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError,match='cancelled'):union_tables({'table1':a,'table2':a},settings(),cancel)


def test_same_name_type_checks_and_invalid_rows():
    a=table(['x'],[(1,)])
    for dtype,value in [('string','1'),('bool',True)]:
        b=table(['x'],[(value,)],[dtype])
        with pytest.raises(ValueError,match='types'):union_tables({'table1':a,'table2':b},settings(typePolicy='losslessNumeric'))
    with pytest.raises(ValueError,match='declared'):union_tables({'table1':replace(a,rows=(('bad',),)),'table2':a},settings())
    with pytest.raises(ValueError,match='unique'):union_tables({'table1':replace(a,rows=((1,),(2,)),row_ids=('a','a')),'table2':a},settings())
    with pytest.raises(ValueError,match='exactly one'):union_tables({'table1':a},settings())
    empty=table(['x'],[])
    assert list(union_tables({'table1':empty,'table2':a},settings()).rows)==[(1,)]

@pytest.mark.parametrize('engine',['disk','polars'])
def test_image_references_pass_through_with_missing_fill(engine):
    from runner.tests.test_image_tables import table as image_table
    from runner.polars_preparation import preparation_engine
    image,_,ref=image_table();plain=table(['key'],[(9,)])
    with preparation_engine(engine):result=union_tables({'table1':image,'table2':plain},settings())
    assert list(result.rows)==[(1,ref),(2,None),(9,None)]
    assert result.columns[1]==dict(name='image',dtype='image',nullable=True)
