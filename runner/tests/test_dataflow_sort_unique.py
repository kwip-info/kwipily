from dataclasses import replace
import pytest
from runner.dataflow import Table, derive
from runner.dataflow_sort import sort_table
from runner.dataflow_unique import unique_table

def table(rows):
    return derive(Table((dict(name='name',dtype='string',nullable=True),dict(name='score',dtype='int64',nullable=True)),tuple(rows),tuple('row:'+str(i) for i in range(len(rows))),'a'*64,role='training',partition_digest='p',fitted_states=('f',)))
def sort_config(**changes):
    return dict(keys=[dict(column='name',direction='ascending',missing='last',comparison='casefold'),dict(column='score',direction='descending',missing='first',comparison='exact')],**changes)
def unique_config(**changes):
    c=dict(keys=['name'],keep='first',missing='equal',comparison='casefold');c.update(changes);return c

@pytest.mark.parametrize('engine',['disk','polars'])
def test_stable_mixed_sort_and_lineage(engine):
    from runner.polars_preparation import preparation_engine
    t=table([('B',1),('a',2),('A',2),(None,9),('a',None),('b',3)])
    with preparation_engine(engine):out=sort_table(t,sort_config())
    assert list(out.row_ids)==['row:4','row:1','row:2','row:5','row:0','row:3']
    assert out.columns==t.columns and set(out.row_ids)==set(t.row_ids)
    assert (out.role,out.partition_digest,out.fitted_states)==('training','p',('f',))
    c=sort_config();c['keys'][0].update(direction='descending',missing='first')
    assert list(sort_table(t,c).row_ids)==['row:3','row:5','row:0','row:4','row:1','row:2']

@pytest.mark.parametrize('engine',['disk','polars'])
@pytest.mark.parametrize('keep,expected',[('first',[0,3,5]),('last',[2,4,5])])
def test_unique_disjoint_complete_source_order(engine,keep,expected):
    from runner.polars_preparation import preparation_engine
    t=table([('Straße',1),('STRASSE',2),('straße',3),(None,1),(None,2),('z',4)])
    with preparation_engine(engine):out=unique_table(t,unique_config(keep=keep))
    assert list(out['unique'].row_ids)==['row:'+str(i) for i in expected]
    assert list(out['duplicates'].row_ids)==['row:'+str(i) for i in range(6) if i not in expected]
    for result in out.values():
        assert (result.role,result.partition_digest,result.fitted_states)==('training','p',('f',))
    assert set(out['unique'].row_ids).isdisjoint(out['duplicates'].row_ids)
    assert set(out['unique'].row_ids)|set(out['duplicates'].row_ids)==set(t.row_ids)

def test_null_exact_compound_empty_and_replay_verification():
    from runner.dataflow_sort import verify_output as verify_sort
    from runner.dataflow_unique import verify_output as verify_unique
    t=table([('a',1),('A',1),(None,1),(None,1)])
    assert len(unique_table(t,unique_config(missing='distinct'))['unique'].rows)==3
    assert len(unique_table(t,unique_config(comparison='exact'))['unique'].rows)==3
    assert len(unique_table(t,unique_config(keys=['name','score']))['unique'].rows)==2
    with pytest.raises(ValueError,match='missing'):unique_table(t,unique_config(missing='stop'))
    out=sort_table(t,sort_config());verify_sort(t,sort_config(),out)
    with pytest.raises(ValueError):verify_sort(t,sort_config(),replace(out,row_ids=tuple(reversed(out.row_ids))))
    results=unique_table(t,unique_config());verify_unique(t,unique_config(),results)
    with pytest.raises(ValueError):verify_unique(t,unique_config(),dict(unique=results['duplicates'],duplicates=results['unique']))
    empty=table([])
    assert len(sort_table(empty,sort_config()).rows)==0
    assert all(len(x.rows)==0 for x in unique_table(empty,unique_config()).values())

@pytest.mark.parametrize('which',['sort','unique'])
def test_large_population_and_cancellation(which):
    t=table([('a' if i%2 else 'b',i) for i in range(100001)])
    if which=='sort':
        out=sort_table(t,sort_config());assert len(out.rows)==100001 and out.rows[0]==('a',99999)
    else:
        out=unique_table(t,unique_config());assert len(out['unique'].rows)==2 and len(out['duplicates'].rows)==99999
    def cancel():raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError,match='cancelled'):
        (sort_table(t,sort_config(),cancel) if which=='sort' else unique_table(t,unique_config(),cancel))

@pytest.mark.parametrize('which',['sort','unique'])
def test_invalid_keys_and_unknown_settings(which):
    t=table([('a',1)])
    run=sort_table if which=='sort' else unique_table;c=sort_config() if which=='sort' else unique_config()
    for keys in [[],['missing']] if which=='unique' else [[],[dict(column='missing',direction='ascending',missing='last',comparison='exact')]]:
        with pytest.raises(ValueError):run(t,{**c,'keys':keys})
    with pytest.raises(ValueError):run(t,{**c,'unexpected':True})

@pytest.mark.parametrize('engine',['disk','polars'])
def test_images_nonkey_temporals_and_numeric_equivalence(engine):
    from runner.tests.test_image_tables import table as image_table
    from runner.polars_preparation import preparation_engine
    image,_,ref=image_table()
    with preparation_engine(engine):
        config=dict(keys=[dict(column='key',direction='descending',missing='last',comparison='exact')])
        out=sort_table(image,config)
        assert list(out.rows)==[(2,None),(1,ref)]
        result=unique_table(image,dict(keys=['key'],keep='last',missing='equal',comparison='exact'))
        assert list(result['unique'].rows)==list(image.rows) and len(result['duplicates'].rows)==0
        with pytest.raises(ValueError,match='Images'):unique_table(image,dict(keys=['image'],keep='first',missing='equal',comparison='exact'))
        for dtype,unit in [('date','day'),('time','microsecond'),('datetime','microsecond')]:
            metadata=dict(version=1,unit=unit)
            if dtype=='datetime':metadata['timezone']='UTC'
            t=derive(Table((dict(name='t',dtype=dtype,nullable=False,temporal=metadata),),((2,),(1,),(1,)),('a','b','c'),'a'*64))
            assert list(sort_table(t,dict(keys=[dict(column='t',direction='ascending',missing='last',comparison='exact')])).row_ids)==['b','c','a']
            u=unique_table(t,dict(keys=['t'],keep='last',missing='equal',comparison='exact'))
            assert list(u['unique'].row_ids)==['a','c'] and u['unique'].columns==t.columns
        t=derive(Table((dict(name='n',dtype='float64',nullable=False),),((0.0,),(-0.0,),(1.0,),(1.0,)),('a','b','c','d'),'a'*64))
        assert list(unique_table(t,dict(keys=['n'],keep='first',missing='equal',comparison='exact'))['unique'].row_ids)==['a','c']

@pytest.mark.parametrize('which',['sort','unique'])
def test_shared_scratch_budget_cancellation_cleanup(tmp_path,which):
    from runner.dataflow_disk import disk_scope,ScratchBudget
    t=table([('a',i) for i in range(300)])
    revoked=False
    class RevokingRows(tuple):
        def __iter__(self):
            nonlocal revoked
            for position,row in enumerate(super().__iter__()):
                if position==100:revoked=True
                yield row
    t=replace(t,rows=RevokingRows(t.rows))
    def cancel():
        if revoked:raise RuntimeError('cancelled')
    budget=ScratchBudget()
    with disk_scope(directory=tmp_path,budget=budget):
        with pytest.raises(RuntimeError,match='cancelled'):
            (sort_table(t,sort_config(),cancel) if which=='sort' else unique_table(t,unique_config(),cancel))
    assert budget.used==0 and list(tmp_path.iterdir())==[]
    with disk_scope(directory=tmp_path,budget=1):
        with pytest.raises(ValueError,match='budget'):
            (sort_table(t,sort_config()) if which=='sort' else unique_table(t,unique_config()))
    assert list(tmp_path.iterdir())==[]

@pytest.mark.parametrize('engine',['disk','polars'])
def test_real_practice_resources(engine):
    from pathlib import Path
    from runner.polars_csv_configuration import DEFAULTS
    from runner.dataflow import source_csv
    from runner.polars_preparation import preparation_engine
    resources=Path(__file__).resolve().parents[2]/'Sources/KwipMLCore/Resources'
    with preparation_engine(engine):
        options={**DEFAULTS,'emptyField':'null','schemaOverrides':{'priority':'Int64'}}
        t=source_csv(resources/'sort-practice.csv',options=options)
        c=dict(keys=[dict(column='customer',direction='ascending',missing='last',comparison='casefold'),dict(column='priority',direction='descending',missing='first',comparison='exact')])
        assert [r[0] for r in sort_table(t,c).rows]==['S5','S2','S3','S6','S1','S4']
        t=source_csv(resources/'unique-practice.csv',options=options)
        u=unique_table(t,dict(keys=['customer','region'],keep='first',missing='equal',comparison='casefold'))
        assert [r[0] for r in u['unique'].rows]==['U1','U4','U6','U7']
        assert [r[0] for r in u['duplicates'].rows]==['U2','U3','U5']
