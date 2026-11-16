"""Fused pure regions keep every logical table and never own external effects."""
from dataclasses import replace
import pytest
from runner import dataflow as df
from runner.dataflow_columnar import ColumnarStore
from runner.dataflow_disk import disk_scope
from runner.native_frame_arena import native_buffer_scope
from runner.table_identity import deferred_identity_scope


def steps():
    return [('label','formula',{'operations':[{'id':'a','expression':'trim([x])','outputs':[{'name':'clean','dtype':'string','write':'new'}],'nullPolicy':'preserve'}]}),
            ('rename','columns',{'rows':[{'source':'clean','included':True,'outputName':'result','targetType':'keep'}],'newColumns':'drop','castFailure':'error','trimText':False,'missingPolicy':'preserve'})]


def test_two_logical_tables_share_one_fused_native_query_and_parent_ids(tmp_path,monkeypatch):
    from runner.native_query_fusion import try_fuse
    from runner.dataflow_formula import apply as formula
    from runner.dataflow_columns import apply as columns
    schema=({'name':'x','dtype':'string','nullable':True},)
    with disk_scope(directory=tmp_path),native_buffer_scope(),deferred_identity_scope():
        root=ColumnarStore(schema,directory=tmp_path)
        for i,value in enumerate((' a ',None,'😀')):root.append((value,),f'r:{i}')
        rows,ids=root.finish();source=df.derive(df.Table(schema,rows,ids,'a'*64))
        a=formula(replace(source,rows=tuple(rows),row_ids=tuple(ids)),steps()[0][2],lambda:None)
        b=columns(a,steps()[1][2],lambda:None)
        def forbidden(*a,**k):raise AssertionError('Fused region allocated an intermediate SQLite/IPC table')
        monkeypatch.setattr(ColumnarStore,'__init__',forbidden)
        result=try_fuse(source,steps(),lambda:None)
        assert result is not None
        for key,expected in (('label',a),('rename',b)):
            actual=result[key]['table']
            assert tuple(actual.rows)==tuple(expected.rows)
            assert tuple(actual.row_ids)==tuple(expected.row_ids)
            assert actual.data_version==expected.data_version
        first=result['label']['table'].rows.store.parent
        second=result['rename']['table'].rows.store.parent
        assert first is second and first.native_collects==len(root.chunks)
        root.close()
        with pytest.raises(ValueError,match='closed'):result['rename']['table'].rows[0]


def test_unsafe_or_failing_regions_fall_back_before_publication(tmp_path):
    from runner.native_query_fusion import try_fuse
    schema=({'name':'x','dtype':'string','nullable':True},)
    with disk_scope(directory=tmp_path),native_buffer_scope(),deferred_identity_scope():
        root=ColumnarStore(schema,directory=tmp_path);root.append((None,),'r:0');rows,ids=root.finish()
        source=df.derive(df.Table(schema,rows,ids,'a'*64))
        nodes=steps();nodes[0][2]['operations'][0]['nullPolicy']='error'
        assert try_fuse(source,nodes,lambda:None) is None
        assert try_fuse(source,[('write','outputDelimited',{}),*steps()],lambda:None) is None
        assert try_fuse(source,[('cast','columns',{'rows':[]}),*steps()],lambda:None) is None
        root.close()


def test_captured_graph_really_executes_fused_region_with_all_logical_callbacks(tmp_path,monkeypatch):
    from runner import native_query_fusion as fusion
    from runner.preparation_kernels import build_kernel_plan
    from runner.polars_preparation import preparation_engine
    source=tmp_path/'source.csv';source.write_text('x\n a \nb\n')
    graph={'schemaVersion':1,'nodes':[{'id':'source','operator':'kwip.data.sourceCSV','config':{'separator':',','parserProfile':'polars'}},*[
        {'id':key,'operator':'kwip.data.'+op,'config':config} for key,op,config in steps()]],'edges':[
        {'from':{'node':'source','port':'table'},'to':{'node':'label','port':'table'}},
        {'from':{'node':'label','port':'table'},'to':{'node':'rename','port':'table'}}]}
    calls=[];original=fusion.try_fuse
    def measured(*args,**kwargs):
        result=original(*args,**kwargs)
        if result is not None:calls.append(tuple(result))
        return result
    monkeypatch.setattr(fusion,'try_fuse',measured)
    from runner import preparation_scheduler
    monkeypatch.setattr(preparation_scheduler,'maximum_native_tasks',lambda:1)
    nodes,_,_=df.validate(graph);callbacks=[]
    result=df.execute(graph,{'source':str(source)},lambda key,ports:callbacks.append(key),engine='polars',kernel_plan=build_kernel_plan(nodes))
    assert calls==[('label','rename')]
    assert callbacks==['source','label','rename']
    assert tuple(result['rename']['table'].rows)==(('a',),('b',))
    for output in result.values():output['table'].rows.store.close()


@pytest.mark.parametrize('expression',[ 'coalesce([x], "fallback")','length(trim(coalesce([x], "")))','[x] IS NULL','trim([x])'])
def test_fusion_preserves_replacements_nulls_unicode_and_logical_values(tmp_path,expression):
    from runner.native_query_fusion import try_fuse
    from runner.dataflow_formula import apply
    from runner.dataflow_columns import apply as columns
    kind='int64' if expression.startswith('length') else 'bool' if expression.endswith('NULL') else 'string'
    nodes=steps();nodes[0][2]['operations']=[{'id':'replace','expression':expression,'outputs':[{'name':'x','dtype':kind,'write':'replace'}],'nullPolicy':'preserve'}]
    nodes[1][2]['rows'][0]['source']='x'
    schema=({'name':'x','dtype':'string','nullable':True},)
    with disk_scope(directory=tmp_path),native_buffer_scope(),deferred_identity_scope():
        store=ColumnarStore(schema,directory=tmp_path)
        for i,value in enumerate((None,'','  a\u2003','😀')):store.append((value,),f'r:{i}')
        rows,ids=store.finish();source=df.derive(df.Table(schema,rows,ids,'a'*64))
        expected_a=apply(replace(source,rows=tuple(rows),row_ids=tuple(ids)),nodes[0][2],lambda:None)
        expected_b=columns(expected_a,nodes[1][2],lambda:None)
        actual=try_fuse(source,nodes,lambda:None);assert actual is not None
        for key,expected in (('label',expected_a),('rename',expected_b)):
            result=actual[key]['table'];assert tuple(result.rows)==tuple(expected.rows)
            assert result.data_version==expected.data_version
        store.close()


def test_fused_cache_hit_still_observes_revocation(tmp_path):
    from runner.native_query_fusion import try_fuse
    revoked=False
    def check():
        if revoked:raise InterruptedError('revoked')
    with disk_scope(directory=tmp_path),native_buffer_scope(),deferred_identity_scope():
        schema=({'name':'x','dtype':'string','nullable':False},);store=ColumnarStore(schema,check,directory=tmp_path)
        store.append((' a ',),'r:1');rows,ids=store.finish();source=df.derive(df.Table(schema,rows,ids,'a'*64))
        actual=try_fuse(source,steps(),check);assert actual['rename']['table'].rows[0]==('a',)
        revoked=True
        with pytest.raises(InterruptedError,match='revoked'):actual['rename']['table'].rows[0]
        store.close()


def test_fusion_accounts_for_new_scalar_arrays_even_when_source_strings_are_empty(tmp_path):
    from runner.native_query_fusion import try_fuse
    nodes=steps();nodes[0][2]['operations'][0]['expression']='length([x])';nodes[0][2]['operations'][0]['outputs'][0]['dtype']='int64'
    with disk_scope(directory=tmp_path) as budget,native_buffer_scope(maximum_bytes=10000),deferred_identity_scope():
        schema=({'name':'x','dtype':'string','nullable':False},);store=ColumnarStore(schema,directory=tmp_path)
        for i in range(2000):store.append(('',),f'r:{i}')
        rows,ids=store.finish();source=df.derive(df.Table(schema,rows,ids,'a'*64))
        assert try_fuse(source,nodes,lambda:None) is None
        store.close();assert budget.used==0


@pytest.mark.parametrize('expression,kind',[('9223372036854775808','int64'),('1e309','float64')])
def test_invalid_numeric_literals_keep_reference_errors_instead_of_publishing_native_values(tmp_path,expression,kind):
    from runner.native_query_fusion import try_fuse
    from runner.dataflow_formula import apply
    nodes=steps();nodes[0][2]['operations'][0]['expression']=expression;nodes[0][2]['operations'][0]['outputs'][0]['dtype']=kind
    with disk_scope(directory=tmp_path),native_buffer_scope(),deferred_identity_scope():
        schema=({'name':'x','dtype':'string','nullable':False},);store=ColumnarStore(schema,directory=tmp_path)
        store.append(('a',),'r:0');rows,ids=store.finish();source=df.derive(df.Table(schema,rows,ids,'a'*64))
        assert try_fuse(source,nodes,lambda:None) is None
        with pytest.raises(ValueError):apply(source,nodes[0][2],lambda:None,share=True)
        store.close()
