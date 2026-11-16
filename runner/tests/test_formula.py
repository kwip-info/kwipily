import json
import pytest
from runner import dataflow as df
from runner import dataflow_formula as formula


def source():
    columns=({'name':'price','dtype':'float64','nullable':False},{'name':'quantity','dtype':'int64','nullable':False},{'name':'name','dtype':'string','nullable':True},{'name':'label','dtype':'string','nullable':False})
    return df.derive(df.Table(columns,((12.5,2,'Ada   Lovelace','ABC-104'),(8.,3,None,'XYZ-207'),(4.,5,'Grace Hopper','bad')),('a','b','c'),'a'*64))


def operation(expression,*outputs,policy='preserve',write='new',id='op'):
    return {'id':id,'expression':expression,'outputs':[{'name':name,'dtype':dtype,'write':write} for name,dtype in outputs],'nullPolicy':policy}


def test_ordered_formulas_reference_prior_outputs_preserve_identity_and_roles():
    table=df.replace(source(),role='training',partition_digest='b'*64,fitted_states=('c'*64,))
    out=formula.apply(table,{'operations':[operation('[price] * [quantity]',('total','float64')),operation('[total] + 1',('adjusted','float64'),id='next')]},lambda:None)
    assert [row[-2:] for row in out.rows]==[(25.,26.),(24.,25.),(20.,21.)]
    assert list(out.row_ids)==['a','b','c'] and out.role=='training' and out.partition_digest=='b'*64 and out.fitted_states==table.fitted_states


def test_regex_replace_extract_match_and_expression_functions():
    c={'operations':[operation('regex_replace([name], "\\\\s+", " ", "all", "")',('clean','string')),operation('regex_extract([label], "^([A-Z]+)-([0-9]+)$", "full", "null", "", "1", "2")',('code','string'),('batch','string'),id='extract'),operation('regex_match([label], "^ABC", "search", "")',('valid','bool'),id='match'),operation('lower(coalesce([clean], "Unknown"))',('lowered','string'),id='lower')]}
    out=formula.apply(source(),c,lambda:None)
    assert [row[4:] for row in out.rows]==[('Ada Lovelace','ABC','104',True,'ada lovelace'),(None,'XYZ','207',False,'unknown'),('Grace Hopper',None,None,False,'grace hopper')]


def test_atomic_names_types_unknown_columns_and_unsafe_expressions_refuse():
    cases=[operation('[missing]',('x','float64')),operation('[price]',('price','float64')),operation('[price]',('x','int64')),operation('__import__("os")',('x','string')),operation('[price] / 0',('x','float64')),operation('regex_extract([label], "(A)", "search", "null", "", "2")',('x','string'))]
    for op in cases:
        with pytest.raises(ValueError):formula.apply(source(),{'operations':[op]},lambda:None)
    with pytest.raises(ValueError):formula.apply(source(),{'operations':[operation('[quantity]',('x','int64')),operation('[quantity]',('x','int64'),id='again')]},lambda:None)
    with pytest.raises(ValueError):formula.apply(source(),{'operations':[operation('[quantity]',('missing','int64'),write='replace')]},lambda:None)


def test_null_policy_and_explicit_replacement():
    out=formula.apply(source(),{'operations':[operation('upper([name])',('name','string'),write='replace')]},lambda:None)
    assert out.rows[0][2]=='ADA   LOVELACE' and out.rows[1][2] is None
    with pytest.raises(ValueError,match='missing'):formula.apply(source(),{'operations':[operation('upper([name])',('x','string'),policy='error')]},lambda:None)


def test_cancellation_and_bounded_unsafe_regex():
    def cancelled():raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError,match='cancelled'):formula.apply(source(),{'operations':[operation('[quantity]',('x','int64'))]},cancelled)
    columns=({'name':'text','dtype':'string','nullable':False},)
    table=df.derive(df.Table(columns,(('a'*5000+'!',),),('one',),'a'*64))
    with pytest.raises(ValueError,match='budget'):formula.apply(table,{'operations':[operation('regex_match([text], "(a+)+$", "search", "")',('x','bool'))]},lambda:None,timeout=.1)

@pytest.mark.parametrize('flag,pattern,value,expected',[
    ('ignoreCase','^abc$','ABC',True),('','^abc$','ABC',False),
    ('multiline','^bar','foo\nbar',True),('','^bar','foo\nbar',False),
    ('dotAll','a.b','a\nb',True),('','a.b','a\nb',False)])
def test_regex_flags_are_real_not_decorative(flag,pattern,value,expected):
    columns=({'name':'text','dtype':'string','nullable':False},)
    table=df.derive(df.Table(columns,((value,),),('one',),'a'*64))
    expression='regex_match([text], '+json.dumps(pattern)+', "search", '+json.dumps(flag)+')'
    out=formula.apply(table,{'operations':[operation(expression,('matched','bool'))]},lambda:None)
    assert out.rows[0][-1] is expected


def test_named_optional_captures_first_replacement_backreferences_and_empty_syntax_validation():
    columns=({'name':'text','dtype':'string','nullable':False},)
    table=df.derive(df.Table(columns,(('ABC-104 ABC-207',),('ABC',)),('one','two'),'a'*64))
    cfg={'operations':[operation('regex_extract([text], "(?P<code>ABC)(?:-([0-9]+))?", "search", "null", "", "code", "2")',('code','string'),('batch','string')),operation('regex_replace([text], "ABC-([0-9]+)", "ID-\\\\1", "first", "")',('changed','string'),id='replace')]}
    out=formula.apply(table,cfg,lambda:None)
    assert out.rows[0][1:]==('ABC','104','ID-104 ABC-207') and out.rows[1][1:]==('ABC',None,'ABC')
    empty=df.derive(table,rows=(),ids=())
    with pytest.raises(ValueError,match='Regex'):formula.apply(empty,{'operations':[operation('regex_replace([text], "(", "x", "all", "")',('x','string'))]},lambda:None)


@pytest.mark.parametrize('engine',['disk','polars'])
def test_numeric_formula_streams_above_legacy_cap(engine):
    from runner.polars_preparation import preparation_engine
    from runner.dataflow_disk import disk_scope
    columns=({'name':'x','dtype':'int64','nullable':False},)
    with disk_scope(),preparation_engine(engine):
        writer=df.row_store(columns,lambda:None,identity_prefix='source:')
        for i in range(100001):writer.append((i,),'source:'+str(i))
        rows,ids=writer.finish();table=df.derive(df.Table(columns,rows,ids,'a'*64))
        out=formula.apply(table,{'operations':[operation('[x] * 2',('twice','int64'))]},lambda:None)
        try:
            assert len(out.rows)==100001 and out.rows[-1]==(100000,200000)
            assert sum(row[-1] for row in out.rows)==100001*100000
            assert isinstance(out.rows,df.StoredRows)
        finally:out.rows.store.close();writer.close()


@pytest.mark.parametrize('engine',['disk','polars'])
def test_full_pipeline_portable_replay_cache_and_attributed_errors(engine,tmp_path):
    from pathlib import Path
    from runner.dataflow_cli import run_dataflow
    from runner.tests.test_dataflow_portable_dependencies import replay
    from runner.polars_csv_configuration import DEFAULTS
    source=Path(__file__).resolve().parents[2]/'Sources/KwipMLCore/Resources/formula-observations.csv'
    cfg={'operations':[operation('[price] * [quantity]',('total','float64')),operation('regex_extract([product_label], "^([A-Z]+)-([0-9]+)$", "full", "null", "", "1", "2")',('product_code','string'),('batch','string'),id='extract')]}
    graph={'schemaVersion':1,'nodes':[{'id':'s','operator':'kwip.data.sourceCSV','config':{**DEFAULTS,'emptyField':'null','schemaOverrides':{'price':'Float64','quantity':'Int64'}}},{'id':'f','operator':'kwip.data.formula','config':cfg}],'edges':[{'from':{'node':'s','port':'table'},'to':{'node':'f','port':'table'}}]}
    path=tmp_path/'graph.json';path.write_text(json.dumps(graph));binding=tmp_path/'bindings.json';binding.write_text(json.dumps({'s':str(source)}))
    exported=tmp_path/'exported';manifest=run_dataflow(path,{'s':str(source)},exported,lambda _:None,engine=engine,cache_policy='materialize')
    entry=next(e for e in manifest['artifacts'] if e.get('nodeID')=='f' and e.get('port')=='table')
    output=json.loads((exported/entry['path']).read_bytes())
    assert [row[-3] for row in output['rows']]==[25.,24.,20.,10.,10.,18.,15.,18.]
    assert output['rows'][2][-2:]==['LMN','003'] and output['rows'][3][-2:]==[None,None]
    elsewhere=tmp_path/'elsewhere';elsewhere.mkdir();reproduced=replay(exported,binding,tmp_path/'replayed',elsewhere)
    assert manifest['artifacts']==reproduced['artifacts']
    mp=exported/'manifest.json'
    cached=run_dataflow(path,{'s':str(source)},tmp_path/'cached',lambda _:None,engine=engine,cache_policy='reuse',reuse_manifest=mp,reuse_manifest_sha256=__import__('hashlib').sha256(mp.read_bytes()).hexdigest())
    assert 'f' in cached['cache']['reusedNodeIDs']

@pytest.mark.parametrize('engine',['disk','polars'])
def test_image_metadata_passes_through_unchanged_and_image_scalar_access_refuses(engine):
    from runner.tests.test_image_tables import table as image_table
    from runner.polars_preparation import preparation_engine
    source,_,reference=image_table()
    with preparation_engine(engine):out=formula.apply(source,{'operations':[operation('[key] + 1',('next','int64'))]},lambda:None)
    assert list(out.rows)==[(1,reference,2),(2,None,3)]
    assert list(out.row_ids)==list(source.row_ids)
    with pytest.raises(ValueError,match='[Ii]mage'):formula.apply(source,{'operations':[operation('[image]',('x','string'))]},lambda:None)


def test_later_regex_batches_report_global_row_numbers():
    columns=({'name':'text','dtype':'string','nullable':True},)
    rows=tuple(('a',) for _ in range(8192))+((None,),)
    source=df.derive(df.Table(columns,rows,tuple(str(i) for i in range(len(rows))),'a'*64))
    with pytest.raises(ValueError,match='row 8193'):
        formula.apply(source,{'operations':[operation('regex_match([text], "a", "full", "")',('x','bool'),policy='error')]},lambda:None)


def test_full_run_capture_preserves_long_text_with_explicit_output_bounds():
    columns=({'name':'text','dtype':'string','nullable':False},)
    table=df.derive(df.Table(columns,(('a'*5000,),),('one',),'a'*64))
    out=formula.apply(table,{'operations':[operation('regex_extract([text], "(a+)", "full", "null", "", "1")',('capture','string'))]},lambda:None)
    assert out.rows[0][-1]=='a'*5000
    too_long=df.derive(table,rows=(('a'*16385,),))
    with pytest.raises(ValueError,match='16384'):formula.apply(too_long,{'operations':[operation('regex_match([text], "a+", "full", "")',('x','bool'))]},lambda:None)


def test_numeric_expression_failure_keeps_typed_code_through_formula_wrapper():
    with pytest.raises(ValueError) as failure:
        formula.apply(source(), {'operations':[operation('[name] + "private suffix"',('composed','string'))]}, lambda:None)
    assert getattr(failure.value, 'code', None) == 'expression_numeric_required'
    event=df.DataFlowExecutionError('formula','formula',{'table':source()},failure.value)
    assert event.detail['code'] == 'expression_numeric_required'
    assert event.detail['nodeID'] == 'formula'
