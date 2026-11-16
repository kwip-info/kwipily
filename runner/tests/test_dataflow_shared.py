"""Real shared IPC ownership, reference values and last-reference cleanup."""
from dataclasses import replace
import gc
import pytest
from runner import dataflow as df
from runner.dataflow_columnar import ColumnarStore
from runner.dataflow_columns import apply
from runner.dataflow_disk import disk_scope
from runner.dataflow_shared import projection_table,filter_table


def source_table(directory,check=lambda:None):
    columns=({'name':'id','dtype':'int64','nullable':False},
             {'name':'text','dtype':'string','nullable':True})
    store=ColumnarStore(columns,directory=directory,check_cancelled=check,identity_prefix='s:')
    for i in range(9000):store.append((i,None if i%13==0 else f'value-{i}'),f's:{i:05d}')
    rows,ids=store.finish()
    return df.derive(df.Table(columns,rows,ids,'a'*64))


def config():
    return {'rows':[{'source':'text','outputName':'renamed','included':True,'targetType':'keep'}],
            'newColumns':'drop','castFailure':'error','missingPolicy':'preserve','trimText':False}


def test_projection_uses_parent_chunks_without_allocating_another_row_store(tmp_path):
    with disk_scope(directory=tmp_path) as budget:
        original=source_table(tmp_path)
        expected=apply(original,config(),lambda:None)
        before=budget.used;files=set(tmp_path.rglob('*'))
        columns=({'name':'renamed','dtype':'string','nullable':True},)
        actual=projection_table(original,columns,[1],lambda:None)
        assert budget.used==before and set(tmp_path.rglob('*'))==files
        assert actual==expected
        assert actual.rows.store.parent is original.rows.store
        assert actual.rows.pair_at(31)==((original.rows[31][1],),original.row_ids[31])
        actual.rows.store.close()
        assert not original.rows.store.closed
        with pytest.raises(ValueError,match='closed'):next(iter(actual.rows))
        expected.rows.store.close();original.rows.store.close()
        assert budget.used==0


def test_projection_holds_parent_until_last_reference_and_flattens_nested_views(tmp_path):
    with disk_scope(directory=tmp_path) as budget:
        original=source_table(tmp_path);root=original.rows.store
        projected=projection_table(original,(original.columns[1],),[1],lambda:None)
        nested=projection_table(projected,({'name':'again',**{k:v for k,v in original.columns[1].items() if k!='name'}},),[0],lambda:None)
        assert nested.rows.store.parent is root
        projected.rows.store.close()
        assert nested.rows[1]==('value-1',)
        del original,projected,root;gc.collect()
        assert budget.used>0
        nested.rows.store.close();del nested;gc.collect()
        assert budget.used==0 and list(tmp_path.iterdir())==[]


def test_projection_keeps_live_cancellation_and_rejects_bad_geometry(tmp_path):
    cancelled=[False]
    def check():
        if cancelled[0]:raise InterruptedError('cancelled')
    original=source_table(tmp_path,check)
    actual=projection_table(original,(original.columns[0],),[0],check)
    cancelled[0]=True
    with pytest.raises(InterruptedError):actual.rows[0]
    cancelled[0]=False
    for selected in ([True],[-1],[2],[0,0],[]):
        with pytest.raises(ValueError):projection_table(original,(original.columns[0],),selected,check)
    original.rows.store.close()
    with pytest.raises(ValueError,match='closed'):actual.rows[0]
    actual.rows.store.close()


def test_non_columnar_input_explicitly_falls_back():
    original=df.Table(({'name':'x','dtype':'int64','nullable':False},),((1,),),('r',),'a'*64)
    assert projection_table(original,original.columns,[0],lambda:None) is None


def test_one_checkpoint_does_not_repeat_the_same_owner_guard(tmp_path):
    calls=[]
    def check():calls.append('owner')
    original=source_table(tmp_path,check)
    actual=projection_table(original,(original.columns[0],),[0],check)
    calls.clear();actual.rows.store.check_cancelled()
    assert calls==['owner']
    # Every new checkpoint must still consult the live guard.
    actual.rows.store.check_cancelled();assert calls==['owner','owner']
    actual.rows.store.close();original.rows.store.close()


def test_distinct_projection_and_parent_guards_are_both_live(tmp_path):
    calls=[];revoked=False
    def root_guard():
        calls.append('root')
        if revoked:raise RuntimeError('root revoked')
    def view_guard():calls.append('view')
    original=source_table(tmp_path,root_guard)
    actual=projection_table(original,(original.columns[0],),[0],view_guard)
    calls.clear();actual.rows.store.check_cancelled();assert calls==['view','root']
    revoked=True
    with pytest.raises(RuntimeError,match='root revoked'):actual.rows.store.check_cancelled()
    actual.rows.store.close();original.rows.store.close()


def test_shared_admission_refuses_metadata_that_mislabels_the_physical_values(tmp_path):
    from runner.tests.test_filter_rows import config as filter_config
    root=ColumnarStore(({'name':'flag','dtype':'int64','nullable':True},),directory=tmp_path)
    root.append((0,),'zero');root.append((None,),'missing');rows,ids=root.finish()
    try:
        for column in ({'name':'flag','dtype':'bool','nullable':True},{'name':'flag','dtype':'int64','nullable':False}):
            source=df.Table((column,),rows,ids,'a'*64)
            assert projection_table(source,(column,),[0],lambda:None) is None
            assert filter_table(source,filter_config(mode='expression',expression='NOT [flag]'),lambda:None) is None
    finally:root.close()


def test_projecting_a_selection_preserves_mapping_and_releases_only_its_lease(tmp_path):
    from runner.tests.test_filter_rows import config as filter_config
    columns=({'name':'keep','dtype':'bool','nullable':False},{'name':'value','dtype':'int64','nullable':False})
    root=ColumnarStore(columns,directory=tmp_path)
    for i in range(9000):root.append((i%2==0,i),str(i))
    rows,ids=root.finish();original=df.derive(df.Table(columns,rows,ids,'a'*64))
    selected=filter_table(original,filter_config(mode='expression',expression='[keep]'),lambda:None)
    projected=projection_table(selected['matching'],(columns[1],),[1],lambda:None)
    try:
        assert list(projected.rows)==[(i,) for i in range(0,9000,2)]
        assert list(projected.row_ids)==list(map(str,range(0,9000,2)))
        projected.rows.store.close()
        assert selected['matching'].rows[1]==(True,2)
    finally:
        for item in selected.values():item.rows.store.close()
        root.close()


def test_selection_cancel_releases_maps_without_closing_parent(tmp_path):
    from runner.tests.test_filter_rows import config as filter_config
    active=[False];calls=[0]
    def check():
        if active[0]:
            calls[0]+=1
            if calls[0]>40:raise InterruptedError('cancelled')
    columns=({'name':'keep','dtype':'bool','nullable':False},)
    with disk_scope(directory=tmp_path) as budget:
        root=ColumnarStore(columns,check)
        for i in range(9000):root.append((i%2==0,),str(i))
        rows,ids=root.finish();original=df.derive(df.Table(columns,rows,ids,'a'*64))
        before=budget.used;files=set(tmp_path.rglob('*'));active[0]=True
        with pytest.raises(InterruptedError):filter_table(original,filter_config(mode='expression',expression='[keep]'),check)
        active[0]=False
        assert budget.used==before and set(tmp_path.rglob('*'))==files
        assert original.rows[1]==(False,)
        root.close()


@pytest.mark.parametrize('expression',['[a]','NOT [a]','[a] AND [b]','[a] OR [b]','NOT ([a] OR [b])','[a] IS NULL','[a] IS NOT NULL','true','false','null'])
@pytest.mark.parametrize('null_policy',['error','nonmatching'])
def test_boolean_filter_native_selection_matches_three_valued_reference(tmp_path,expression,null_policy):
    from runner.dataflow_filter import apply as reference
    from runner.tests.test_filter_rows import config as filter_config
    columns=tuple({'name':n,'dtype':'bool','nullable':True} for n in ('a','b'))
    values=[(a,b) for a in (True,False,None) for b in (True,False,None)]*500
    store=ColumnarStore(columns,directory=tmp_path,identity_prefix='s:')
    for i,row in enumerate(values):store.append(row,f's:{i:06d}')
    rows,ids=store.finish();original=df.derive(df.Table(columns,rows,ids,'b'*64))
    selected=filter_config(mode='expression',expression=expression,nullPolicy=null_policy)
    actual=expected=None
    try:
        try:expected=reference(original,selected,lambda:None)
        except ValueError as error:
            with pytest.raises(ValueError) as failure:filter_table(original,selected,lambda:None)
            assert str(failure.value)==str(error)
        else:
            actual=filter_table(original,selected,lambda:None)
            assert actual==expected
            for port in actual:
                assert actual[port].rows.store.parent is store
                assert actual[port].rows.store.mapping.columns[0]['name']=='position'
    finally:
        for output in (actual,expected):
            if output:
                for result in output.values():result.rows.store.close()
        store.close()
    assert list(tmp_path.iterdir())==[]


def test_boolean_filter_keeps_payload_and_selection_batches_native(tmp_path,monkeypatch):
    from runner.tests.test_filter_rows import config as filter_config
    columns=({'name':'n','dtype':'int64','nullable':False},)
    root=ColumnarStore(columns,directory=tmp_path)
    for i in range(9000):root.append((i,),str(i))
    rows,ids=root.finish();original=df.derive(df.Table(columns,rows,ids,'a'*64))
    def refuse(*a,**kw):raise AssertionError('Native filter became scalar row-store appends')
    monkeypatch.setattr(ColumnarStore,'append',refuse)
    result=None
    try:
        result=filter_table(original,filter_config(mode='expression',expression='[n] >= 5000'),lambda:None)
        assert list(result['matching'].rows)==[(i,) for i in range(5000,9000)]
        assert list(result['matching'].row_ids)==list(map(str,range(5000,9000)))
        assert len(result['nonmatching'].rows)==5000
    finally:
        if result:
            for output in result.values():output.rows.store.close()
        root.close()
