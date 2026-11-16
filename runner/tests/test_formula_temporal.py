import json
import pytest
from runner import dataflow as df, dataflow_formula as f, dataflow_temporal as t

def op(expression,name,dtype,id='op'):
    return {'id':id,'expression':expression,'outputs':[{'name':name,'dtype':dtype,'write':'new'}],'nullPolicy':'preserve'}

def parse_call(column='text',dtype='datetime',timezone='UTC'):
    return 'temporal_parse(['+column+'], '+', '.join(json.dumps(x) for x in [dtype,'iso','UTC',timezone,'stop','stop','stop','v1'])+')'

def test_temporal_ordered_formula_exact_metadata_and_identity():
    columns=({'name':'text','dtype':'string','nullable':True},)
    table=df.derive(df.Table(columns,(('1970-01-01T00:00:00.000001Z',),(None,)),('one','two'),'a'*64))
    config={'operations':[op(parse_call(),'instant','datetime'),op('temporal_timezone([instant], "America/New_York")','local','datetime','zone'),op('temporal_format([local], "iso", "America/New_York")','formatted','string','format'),op('temporal_part([local], "hour", "America/New_York")','hour','int64','part'),op('temporal_difference([instant], [local], "microseconds", "integer")','elapsed','int64','difference')]}
    out=f.apply(table,config,lambda:None)
    assert list(out.rows)==[('1970-01-01T00:00:00.000001Z',1,1,'1969-12-31T19:00:00.000001-05:00',19,0),(None,None,None,None,None,None)]
    assert out.columns[1]['temporal']==t.metadata('datetime','UTC')
    assert out.columns[2]['temporal']==t.metadata('datetime','America/New_York')
    assert list(out.row_ids)==['one','two']

def test_unsupported_arithmetic_and_cross_type_difference_refused_before_rows():
    columns=({'name':'instant','dtype':'datetime','nullable':False,'temporal':t.metadata('datetime')},{'name':'date','dtype':'date','nullable':False,'temporal':t.metadata('date')})
    for expression in ('[instant] + 1','temporal_difference([instant], [date], "days", "integer")','temporal_part([date], "hour", "UTC")'):
        with pytest.raises(ValueError):f.compile_plan(columns,{'operations':[op(expression,'result','int64')]})

def test_temporal_passthrough_and_coalesce_preserve_metadata():
    columns=({'name':'a','dtype':'datetime','nullable':True,'temporal':t.metadata('datetime')},{'name':'b','dtype':'datetime','nullable':False,'temporal':t.metadata('datetime')})
    result,_=f.compile_plan(columns,{'operations':[op('coalesce([a], [b])','out','datetime')]})
    assert result[-1]['temporal']==t.metadata('datetime')
    mismatch=[dict(column) for column in columns];mismatch[1]['temporal']=t.metadata('datetime','Europe/London')
    with pytest.raises(ValueError,match='metadata'):f.compile_plan(mismatch,{'operations':[op('coalesce([a], [b])','out','datetime')]})

def test_temporal_and_regex_share_supervised_worker():
    columns=({'name':'text','dtype':'string','nullable':False},)
    table=df.derive(df.Table(columns,(('1970-01-01',),),('one',),'a'*64))
    out=f.apply(table,{'operations':[op(parse_call(dtype='date'),'day','date'),op('regex_match([text], "^1970", "search", "")','match','bool','regex')]},lambda:None)
    assert list(out.rows)==[('1970-01-01',0,True)]

@pytest.mark.parametrize('engine',['disk','polars'])
def test_temporal_storage_engine_preserves_exact_microseconds(engine):
    from runner.polars_preparation import preparation_engine
    columns=({'name':'text','dtype':'string','nullable':False},)
    table=df.derive(df.Table(columns,(('1969-12-31T23:59:59.999999Z',),('2024-01-01T00:00:00.123456Z',)),('one','two'),'a'*64))
    with preparation_engine(engine):
        out=f.apply(table,{'operations':[op(parse_call(),'instant','datetime')]},lambda:None)
        try:
            assert [row[-1] for row in out.rows]==[-1,1704067200123456]
            assert out.columns[-1]['temporal']==t.metadata('datetime')
        finally:out.rows.store.close()

def test_temporal_query_search_and_stable_sort(tmp_path):
    from runner.table_view import build_index,query_index
    columns=({'name':'instant','dtype':'datetime','nullable':True,'temporal':t.metadata('datetime')},)
    table=df.derive(df.Table(columns,((1,),(-1,),(None,),(1,)),('a','b','c','d'),'a'*64))
    path=tmp_path/'view.sqlite3';receipt=build_index(table,path,source_sha256='a'*64)
    assert query_index(path,receipt['index']['sha256'],search='1969-12-31')['positions']==[1]
    assert query_index(path,receipt['index']['sha256'],sort_column=0)['positions']==[1,0,3,2]

def test_temporal_metadata_version_and_physical_values_are_strict():
    from runner.dataflow_fitted import schema_valid,typed
    for metadata in ({'version':True,'unit':'day'},{'version':1,'unit':'microsecond'},{'version':1,'unit':'day','timezone':'UTC'}):
        with pytest.raises(ValueError):schema_valid([{'name':'day','dtype':'date','nullable':False,'temporal':metadata}])
    assert not typed(True,'datetime') and not typed(1.0,'date')

@pytest.mark.parametrize('engine',['disk','polars'])
def test_temporal_artifacts_cache_and_standalone_replay(engine,tmp_path):
    import hashlib
    from runner.dataflow_cli import run_dataflow
    from runner.dataflow_table_reader import read_table_artifact
    from runner.polars_csv_configuration import DEFAULTS
    from runner.tests.test_dataflow_portable_dependencies import replay
    source=tmp_path/'source.csv';source.write_text('text\n1969-12-31T23:59:59.999999Z\n2024-01-01T00:00:00.123456Z\n')
    graph={'schemaVersion':1,'nodes':[{'id':'s','operator':'kwip.data.sourceCSV','config':DEFAULTS},{'id':'f','operator':'kwip.data.formula','config':{'operations':[op(parse_call(),'instant','datetime')]}}], 'edges':[{'from':{'node':'s','port':'table'},'to':{'node':'f','port':'table'}}]}
    path=tmp_path/'graph.json';path.write_text(json.dumps(graph));bindings=tmp_path/'bindings.json';bindings.write_text(json.dumps({'s':str(source)}))
    first=tmp_path/'first';manifest=run_dataflow(path,{'s':str(source)},first,lambda _:None,engine=engine,cache_policy='materialize')
    entry=next(e for e in manifest['artifacts'] if e.get('nodeID')=='f' and e.get('port')=='table')
    raw=json.loads((first/entry['path']).read_bytes())
    assert raw['schemaVersion']==3
    assert raw['columns'][-1]['temporal']==t.metadata('datetime')
    assert [row[-1] for row in raw['rows']]==[-1,1704067200123456]
    table=read_table_artifact(first/entry['path'],entry['sha256'])
    try:assert [row[-1] for row in table.rows]==[-1,1704067200123456]
    finally:table.rows.store.close();table.row_ids.store.close()
    second=tmp_path/'second';cached=run_dataflow(path,{'s':str(source)},second,lambda _:None,engine=engine,cache_policy='reuse',reuse_manifest=first/'manifest.json',reuse_manifest_sha256=hashlib.sha256((first/'manifest.json').read_bytes()).hexdigest())
    assert 'f' in cached['cache']['reusedNodeIDs']
    elsewhere=tmp_path/'elsewhere';elsewhere.mkdir()
    reproduced=replay(first,bindings,tmp_path/'replayed',elsewhere)
    again=next(e for e in reproduced['artifacts'] if e.get('nodeID')=='f' and e.get('port')=='table')
    assert again['sha256']==entry['sha256']

def test_numeric_fitted_parameters_preserve_temporal_pass_through():
    from runner import dataflow_fitted as fitted,portable_fitted as portable
    columns=({'name':'x','dtype':'float64','nullable':False},{'name':'instant','dtype':'datetime','nullable':False,'temporal':t.metadata('datetime','America/New_York')})
    table=df.derive(df.Table(columns,((1.0,-1),(3.0,1704067200123456)),('a','b'),'a'*64,'training',partition_digest='b'*64))
    from runner.tests.legacy_fitted_fixtures import frozen_state
    state=frozen_state(table,fitted.SCALER,{'columns':['x']},'c'*64)
    params=fitted.portable_parameters(state)
    rows,schema,_=portable.apply(table.rows,list(columns),params)
    assert rows==[[-1.0,-1],[1.0,1704067200123456]]
    assert schema[1]==columns[1]
    with pytest.raises(ValueError,match='temporal'):portable.apply(((1.0,True),),list(columns),params)
    from runner.dataflow_preprocess import apply as preprocess,rule_defaults
    with pytest.raises(ValueError,match='temporal'):preprocess({'training':table},{'mode':'partitions','columns':[dict(rule_defaults('instant'),missing='mostFrequent',tiePolicy='smallest')]},'c'*64,lambda:None)

def test_aggregate_temporal_reducers_preserve_metadata_and_columns_cast_refuses():
    from runner import dataflow_aggregate as aggregate,dataflow_columns as columns_tool
    schema=({'name':'group','dtype':'string','nullable':False},{'name':'instant','dtype':'datetime','nullable':False,'temporal':t.metadata('datetime','America/New_York')})
    table=df.derive(df.Table(schema,(('x',-1),('x',1704067200123456)),('a','b'),'a'*64))
    config={'groupBy':['group'],'nullKeys':'group','aggregations':[{'op':'min','column':'instant','output':'first','nullPolicy':'skip'},{'op':'max','column':'instant','output':'last','nullPolicy':'skip'}],'maxRows':100}
    plan=aggregate.plan_aggregate(table,config)
    assert [column['temporal'] for column in plan.columns[1:]]==[t.metadata('datetime','America/New_York')]*2
    assert [row for row,_ in plan.iter_rows()]==[('x',-1,1704067200123456)]
    cast={'rows':[{'source':'instant','included':True,'outputName':'instant','targetType':'string'}],'newColumns':'keep','castFailure':'error','missingPolicy':'preserve','trimText':False}
    with pytest.raises(ValueError,match='Temporal'):columns_tool.apply(table,cast,lambda:None)
    cast['rows'][0]['targetType']='keep';cast['rows'][0]['outputName']='renamed'
    out=columns_tool.apply(table,cast,lambda:None)
    try:assert out.columns[0]['temporal']==t.metadata('datetime','America/New_York')
    finally:out.rows.store.close()

@pytest.mark.parametrize('engine',['disk','polars'])
@pytest.mark.parametrize('dtype,text',[('date','1970-01-01'),('datetime','1970-01-01T00:00:00Z'),('time','00:00:00')])
@pytest.mark.parametrize('target',['string','int64','float64','bool'])
def test_actual_graph_current_columns_refuses_typed_temporal_source(engine,dtype,text,target,tmp_path):
    from runner.polars_preparation import preparation_engine
    from runner.polars_csv_configuration import DEFAULTS
    source=tmp_path/'source.csv';source.write_text('text\n'+text+'\n')
    graph={'schemaVersion':1,'nodes':[
        {'id':'source','operator':'kwip.data.sourceCSV','config':DEFAULTS},
        {'id':'parse','operator':'kwip.data.formula','config':{'operations':[op(parse_call(dtype=dtype),'temporal_value',dtype)]}},
        {'id':'columns','operator':'kwip.data.columns','config':dict(rows=[dict(source='temporal_value',included=True,outputName='temporal_value',targetType=target)],castFailure='error',trimText=False,newColumns='keep',missingPolicy='preserve')}],
        'edges':[{'from':{'node':'source','port':'table'},'to':{'node':'parse','port':'table'}},{'from':{'node':'parse','port':'table'},'to':{'node':'columns','port':'table'}}]}
    with preparation_engine(engine),pytest.raises(ValueError,match='Formula Parse, Format or Extract') as failure:
        df.execute(graph,{'source':str(source)})
    assert failure.value.detail['nodeID']=='columns'
