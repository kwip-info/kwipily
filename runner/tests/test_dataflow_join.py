"""Join semantics against SQL and hand-checked lineage/budget fixtures."""
import copy
import sqlite3
from dataclasses import replace
import pytest
from runner.dataflow import Table, derive
from runner.dataflow_join import join_tables


def table(rows, names=('key', 'value'), source='a', dtypes=None):
    dtypes = dtypes or ['int64'] * len(names)
    columns = tuple(dict(name=n, dtype=t, nullable=any(r[i] is None for r in rows))
                    for i, (n, t) in enumerate(zip(names, dtypes)))
    return derive(Table(columns, tuple(map(tuple, rows)), tuple(f'{source}:{i}' for i in range(len(rows))), source * 64))


def config(**changes):
    return dict(leftKeys=['key'], rightKeys=['key'], mode='inner', nullPolicy='neverMatch',
                cardinality='manyToMany', leftPrefix='left_', rightPrefix='right_', maxRows=100000, **changes)


def settings(**changes):
    value=config();value.update(changes);return value


@pytest.mark.parametrize('mode',['inner','left','right','full'])
def test_columnar_join_scans_left_once_and_keeps_exact_pair_order(tmp_path,monkeypatch,mode):
    from runner.dataflow_columnar import ColumnarStore,ColumnarRows
    from runner.dataflow_join import plan_join
    left=table([(i%7,i) for i in range(9000)])
    right=table([(1,10),(1,11),(3,30),(9,90)],source='b')
    expected=list(plan_join(left,right,settings(mode=mode)).iter_rows())
    stores=[];checks=[0]
    def check():checks[0]+=1
    def disk_table(source):
        store=ColumnarStore(source.columns,directory=tmp_path,check_cancelled=check)
        stores.append(store)
        for row,rid in zip(source.rows,source.row_ids):store.append(row,rid)
        rows,ids=store.finish()
        return replace(source,rows=rows,row_ids=ids)
    try:
        plan=plan_join(disk_table(left),disk_table(right),settings(mode=mode))
        def forbidden_index(self,index):raise AssertionError('Join should not perform separate row indexing')
        monkeypatch.setattr(ColumnarRows,'__getitem__',forbidden_index)
        before=checks[0];actual=list(plan.iter_rows())
        assert actual==expected
        # Admissions are bounded batches, rather than one authority walk per
        # parent lookup or every output match.
        assert checks[0]-before < (len(left.rows)+len(actual))//128+100
    finally:
        for store in stores:store.close()
    assert list(tmp_path.iterdir())==[]


def test_columnar_join_expansion_keeps_live_cancellation(tmp_path):
    from runner.dataflow_columnar import ColumnarStore
    from runner.dataflow_join import plan_join
    cancelled=[False];stores=[]
    def check():
        if cancelled[0]:raise InterruptedError('cancelled')
    def disk_table(source):
        store=ColumnarStore(source.columns,directory=tmp_path,check_cancelled=check)
        stores.append(store)
        for row,rid in zip(source.rows,source.row_ids):store.append(row,rid)
        rows,ids=store.finish()
        return replace(source,rows=rows,row_ids=ids)
    try:
        plan=plan_join(disk_table(table([(1,1)])),disk_table(table([(1,i) for i in range(1000)],source='b')),settings())
        result=plan.iter_rows();next(result)
        cancelled[0]=True
        # Gather admission is bounded by256 rows (and separately by byte work),
        # matching existing sequential IPC iteration; it is no longer per pair.
        with pytest.raises(InterruptedError):
            for _ in range(256):next(result)
    finally:
        for store in stores:store.close()
    assert list(tmp_path.iterdir())==[]


@pytest.mark.parametrize('mode', ['inner', 'left', 'right', 'full'])
def test_values_match_sql_and_preserve_deterministic_order(mode):
    left=table([(1,10),(2,20),(2,21),(None,30),(4,40)])
    right=table([(2,200),(2,201),(3,300),(None,400)],source='b')
    before=copy.deepcopy((left,right))
    actual=join_tables(left,right,settings(mode=mode))
    db=sqlite3.connect(':memory:')
    try:
        db.execute('create table l(k integer,v integer)');db.execute('create table r(k integer,v integer)')
        db.executemany('insert into l values (?,?)',left.rows);db.executemany('insert into r values (?,?)',right.rows)
        expected=db.execute(f'select l.k,l.v,r.k,r.v from l {mode} join r on l.k=r.k order by l.rowid is null,l.rowid,r.rowid').fetchall()
    finally:db.close()
    assert list(actual.rows)==expected
    assert [c['name'] for c in actual.columns]==['left_key','left_value','right_key','right_value']
    assert len(set(actual.row_ids))==len(actual.rows)
    assert actual==join_tables(left,right,settings(mode=mode))
    assert (left,right)==before
    assert actual.role=='unassigned' and not actual.fitted_states and not actual.partition_digest
    assert actual.source_digest not in (left.source_digest,right.source_digest)


def test_paired_keys_null_policy_and_column_schema():
    left=table([(1,'a',5),(1,None,6)],names=('id','group','x'),dtypes=['int64','string','int64'])
    right=table([(1,'a',9),(1,None,8)],names=('code','group','y'),source='b',dtypes=['int64','string','int64'])
    c=settings(leftKeys=['id','group'],rightKeys=['code','group'],nullPolicy='match')
    result=join_tables(left,right,c)
    assert result.rows==((1,'a',5,1,'a',9),(1,None,6,1,None,8))
    assert [col['name'] for col in result.columns]==['id','left_group','x','code','right_group','y']
    assert len(join_tables(left,right,{**c,'nullPolicy':'neverMatch'}).rows)==1
    with pytest.raises(ValueError,match='missing join key'):join_tables(left,right,{**c,'nullPolicy':'error'})


@pytest.mark.parametrize('cardinality,left_rows,right_rows',[
    ('oneToOne',[(1,1),(1,2)],[(1,3)]),('oneToMany',[(1,1),(1,2)],[(1,3)]),
    ('manyToOne',[(1,1)],[(1,3),(1,4)])])
def test_cardinality_is_checked_before_expansion(cardinality,left_rows,right_rows):
    with pytest.raises(ValueError,match='cardinality'):join_tables(table(left_rows),table(right_rows,source='b'),settings(cardinality=cardinality))


def test_budget_checks_and_no_matches_are_actionable():
    left=table([(1,i) for i in range(400)])
    right=table([(1,i) for i in range(400)],source='b')
    with pytest.raises(ValueError,match='row limit'):join_tables(left,right,settings())
    with pytest.raises(ValueError,match='no rows'):join_tables(table([(1,2)]),table([(3,4)],source='b'),settings())
    with pytest.raises(ValueError,match='row limit'):join_tables(table([(1,2),(2,3)]),table([(8,9)],source='b'),settings(mode='full',maxRows=2))


def test_partition_and_fitted_inputs_cannot_lose_safety_evidence():
    source=table([(1,2)])
    for changed in [replace(source,role='training'),replace(source,partition_digest='p'),replace(source,fitted_states=('f',))]:
        with pytest.raises(ValueError,match='before splitting'):join_tables(changed,source,settings())
        with pytest.raises(ValueError,match='before splitting'):join_tables(source,changed,settings())


@pytest.mark.parametrize('changes',[{'leftKeys':[]},{'leftKeys':['key','key']},{'rightKeys':['key','value']},
    {'mode':'cross'},{'nullPolicy':'guess'},{'cardinality':'guess'},{'maxRows':True},{'maxRows':0},
    {'maxRows':100001},{'leftPrefix':''},{'rightPrefix':'left_'},{'unknown':1}])
def test_bad_configuration_is_rejected(changes):
    with pytest.raises(ValueError):join_tables(table([(1,2)]),table([(1,3)],source='b'),settings(**changes))


def test_dtype_collisions_and_invalid_identity_rejected():
    left=table([(1,2)])
    with pytest.raises(ValueError,match='dtype'):join_tables(left,table([('1',3)],source='b',dtypes=['string','int64']),settings())
    right=table([(1,3,5)],names=('key','value','left_key'),source='b')
    with pytest.raises(ValueError,match='column'):join_tables(left,right,settings())
    with pytest.raises(ValueError,match='identit'):join_tables(replace(left,row_ids=()),left,settings())


def test_row_identity_depends_on_parents_not_order_or_unrelated_values():
    left=table([(1,10),(2,20)]);right=table([(1,30),(2,40)],source='b')
    original=join_tables(left,right,settings())
    reordered=derive(left,rows=left.rows[::-1],ids=left.row_ids[::-1])
    result=join_tables(reordered,right,settings())
    assert result.row_ids==original.row_ids[::-1]
    changed=derive(right,rows=[(1,300),(2,40)])
    result=join_tables(left,changed,settings())
    assert result.row_ids==original.row_ids and result.data_version!=original.data_version
    info=result.diagnostics[-1]
    assert info['leftRows']==2 and info['rightRows']==2 and info['outputRows']==2
    assert info['leftUnmatched']==0 and info['rightUnmatched']==0


def test_output_cell_budget_and_nullable_outer_columns():
    names=tuple(f'c{i}' for i in range(64))
    left=table([(1,)*64]*200,names=names)
    right=table([(1,)*64]*200,names=names,source='b')
    with pytest.raises(ValueError,match='five million'):join_tables(left,right,settings(leftKeys=['c0'],rightKeys=['c0']))
    result=join_tables(table([(1,2)]),table([(9,3)],source='b'),settings(mode='full'))
    assert result.rows==((1,2,None,None),(None,None,9,3))
    assert all(c['nullable'] for c in result.columns)


def test_null_never_match_does_not_violate_key_uniqueness():
    left=table([(None,2),(None,3),(1,4)])
    right=table([(1,5)],source='b')
    result=join_tables(left,right,settings(cardinality='oneToOne',mode='left'))
    assert len(result.rows)==3
    with pytest.raises(ValueError,match='cardinality'):
        join_tables(left,right,settings(cardinality='oneToOne',nullPolicy='match'))


def test_output_projection_order_names_and_legacy_identity():
    left=table([(1,10),(2,20)])
    right=table([(1,30)],source='b')
    legacy=join_tables(left,right,settings(mode='left'))
    c=settings(mode='left',outputColumns=[
        {'side':'right','source':'value','name':'context','included':True},
        {'side':'left','source':'key','name':'id','included':True},
        {'side':'left','source':'value','name':'ignored','included':False}],newColumns='drop')
    result=join_tables(left,right,c)
    assert [x['name'] for x in result.columns]==['context','id']
    assert result.rows==((30,1),(None,2))
    assert result.columns[0]['nullable'] is True
    assert result.row_ids==legacy.row_ids
    from runner.dataflow_join import verify_join_output
    verify_join_output(left,right,c,result)
    with pytest.raises(ValueError):verify_join_output(left,right,c,legacy)


@pytest.mark.parametrize('policy,expected',[('keep',['id','left_value','right_key','right_value']),('drop',['id'])])
def test_output_new_column_policy(policy,expected):
    result=join_tables(table([(1,2)]),table([(1,3)],source='b'),settings(
        outputColumns=[{'side':'left','source':'key','name':'id','included':True}],newColumns=policy))
    assert [x['name'] for x in result.columns]==expected


@pytest.mark.parametrize('columns,policy',[
    ([], 'drop'),
    ([{'side':'left','source':'missing','name':'x','included':True}],'keep'),
    ([{'side':'left','source':'key','name':'x','included':True}],'error'),
    ([{'side':'left','source':'key','name':'x','included':True},{'side':'right','source':'key','name':'x','included':True}],'drop'),
    ([{'side':'left','source':'key','name':'x','included':True}]*2,'drop'),
    ([{'side':'left','source':'key','name':'x','included':1}],'drop'),
])
def test_output_projection_refuses_invalid_columns(columns,policy):
    with pytest.raises(ValueError):join_tables(table([(1,2)]),table([(1,3)],source='b'),settings(outputColumns=columns,newColumns=policy))


@pytest.mark.parametrize('mode',['inner','left','right','full'])
@pytest.mark.parametrize('side',['left','right','both'])
def test_projected_budget_is_exact_serialized_writer_size(mode,side):
    from runner.dataflow_join import plan_join, _encoded
    left=table([(1,'λ\n"'),(2,'x')],dtypes=['int64','string'])
    right=table([(1,'abc'),(3,'d')],source='b',dtypes=['int64','string'])
    rows=[dict(side=s,source='value',name=s,included=True) for s in ('left','right') if side in (s,'both')]
    c=settings(mode=mode,outputColumns=rows,newColumns='drop')
    result=join_tables(left,right,c);plan=plan_join(left,right,c)
    envelope={'schemaVersion':1,'kind':'table','columns':result.columns,'rows':result.rows,
        'lineage':{'role':'unassigned','sourceDigest':result.source_digest,'membershipDigest':'0'*64,
          'observationIDs':result.row_ids,'dataVersion':'0'*64,'partitionDigest':'','fittedStateDigests':[],'diagnostics':result.diagnostics}}
    assert len(_encoded(envelope))==plan.artifact_bytes


@pytest.mark.parametrize('engine',['disk','polars'])
def test_projected_join_engine_parity_and_selected_image(engine):
    from runner import dataflow as df
    from runner.polars_preparation import preparation_engine
    from runner.tests.test_image_tables import table as image_table
    source,_,ref=image_table()
    right=table([(1,9)],source='b')
    c=settings(outputColumns=[dict(side='left',source='image',name='pixels',included=True)],newColumns='drop')
    with preparation_engine(engine):
        output=df._execute_node('join',c,{'left':source,'right':right},'',None,lambda:None)['table']
        assert output.columns[0]['dtype']=='image' and output.rows[0][0]==ref
