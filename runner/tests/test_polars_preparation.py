"""Engine parity is values, identities, provenance and refusals, not row counts alone."""
import pytest
from runner import dataflow as df
from runner.table_batches import table_batches
from runner.tests.test_large_table_execution import graph
from runner.tests.test_columns import config as columns_config,item
from runner.tests.test_filter_rows import config as filter_config,condition
from runner.dataflow_preprocess import rule_defaults


def signature(table):
    return (table.columns, tuple(table.rows), tuple(table.row_ids), table.source_digest,
            table.data_version, table.role, table.partition_digest, table.fitted_states, table.diagnostics)


@pytest.mark.parametrize('options,content', [
    ({'separator':';','nullMarkers':['NA']},'x;y\n1;NA\n2;a\n3;b\n'),
    ({'separator':'§','quoteChar':'~','escapeMode':'custom','escapeChar':'!'},'x§y\n~a!~b~§c\nz§d\n'),
])
def test_csv_options_and_local_operations_match_disk(tmp_path, options, content):
    path=tmp_path/'source.csv';path.write_text(content)
    flow=graph(('preprocess',{'mode':'table','columns':[dict(rule_defaults('y'),missing='constant',value='replacement')]}),
               ('columns',columns_config([item('y','value'),item('x')],newColumns='drop')),
               ('filterRows',filter_config(conditions=[condition('x','ne','never')])) )
    flow['nodes'][0]['config']=options
    expected=df.execute(flow,{'source':str(path)},engine='disk')
    actual=df.execute(flow,{'source':str(path)},engine='polars')
    assert {key:signature(value.get('table',value.get('matching'))) for key,value in actual.items()}=={key:signature(value.get('table',value.get('matching'))) for key,value in expected.items()}


@pytest.mark.parametrize('op,config',[
    ('filterRows',filter_config(conditions=[condition('x','eq','1')])),
    ('preprocess',{'mode':'table','columns':[dict(rule_defaults('x'),missing='error')]}),
])
def test_null_errors_do_not_become_silent_polars_drops(tmp_path,op,config):
    path=tmp_path/'source.csv';path.write_text('x\n1\nNA\n')
    flow=graph((op,config));flow['nodes'][0]['config']={'nullMarkers':['NA']}
    errors=[]
    for engine in ('disk','polars'):
        with pytest.raises(df.DataFlowExecutionError) as error:df.execute(flow,{'source':str(path)},engine=engine)
        errors.append(error.value.detail)
    assert errors[0]==errors[1]


def test_batches_are_row_and_byte_bounded_and_keep_identity():
    table=df.Table(({'name':'x','dtype':'string','nullable':False},),
                   (('long text',),('more text',),('last',)),('a','b','c'),'source')
    batches=list(table_batches(table,max_rows=2,max_bytes=55))
    assert len(batches)>=2
    assert tuple(row for batch in batches for row in batch.rows)==table.rows
    assert tuple(rid for batch in batches for rid in batch.observation_ids)==table.row_ids
    assert all(len(batch.rows)<=2 and batch.encoded_bytes<=55 for batch in batches)
    with pytest.raises(ValueError,match='batch byte budget'):list(table_batches(table,max_bytes=5))


def test_polars_numeric_filter_preserves_large_integer_comparison(tmp_path):
    path=tmp_path/'source.csv';path.write_text('x\n9007199254740993\n9007199254740992\n')
    flow=graph(('columns',columns_config([item('x',dtype='int64')])),('filterRows',filter_config(conditions=[condition('x','gt',9007199254740992.0)])))
    expected=df.execute(flow,{'source':str(path)},engine='disk')['1']['matching']
    actual=df.execute(flow,{'source':str(path)},engine='polars')['1']['matching']
    assert signature(actual)==signature(expected)


def test_unknown_engine_refused_before_source_access(tmp_path):
    with pytest.raises(ValueError,match='engine'):df.execute(graph(),{'source':str(tmp_path/'absent')},engine='unknown')


@pytest.mark.parametrize('dtype,values,literal',[
    ('string',('é','a','z'),'m'),('bool',(False,True,False),True),
    ('int64',(1,2,3),2),('float64',(-0.0,0.5,2.25),0.5),
    ('int64',(-2**63,2**63-1,0),2**70),
])
@pytest.mark.parametrize('comparison',['eq','ne','lt','le','gt','ge'])
def test_typed_filter_values_and_failures_are_identical(dtype,values,literal,comparison):
    from runner.polars_preparation import preparation_engine
    table=df.derive(df.Table(({'name':'__position','dtype':dtype,'nullable':False},),
                            tuple((value,) for value in values),('a','b','c'),'source'))
    config=filter_config(conditions=[condition('__position',comparison,literal)])
    results=[]
    for engine in ('disk','polars'):
        with preparation_engine(engine):
            try:results.append(signature(df._execute_node('filterRows',config,{'table':table},'',None,lambda:None)['matching']))
            except ValueError as error:results.append((type(error),str(error),getattr(error,'code',None),getattr(error,'suggestion',None)))
    assert results[0]==results[1]


def test_invalid_typed_fill_refused_even_when_no_values_are_missing():
    from runner.polars_preparation import preparation_engine
    table=df.derive(df.Table(({'name':'x','dtype':'int64','nullable':True},),((1,),),('a',),'s'))
    config={'mode':'table','columns':[dict(rule_defaults('x'),missing='constant',value='not a number')]}
    for engine in ('disk','polars'):
        with preparation_engine(engine),pytest.raises(ValueError,match='constant|typed|literal'):
            df._execute_node('preprocess',config,{'table':table},'',None,lambda:None)


def test_engine_context_does_not_leak_and_cache_separates_runtime():
    from runner.polars_preparation import preparation_engine,current_engine
    from runner.dataflow_cache import implementation_digest
    baseline=implementation_digest()
    with preparation_engine('polars'):
        assert current_engine()=='polars'
        assert implementation_digest()!=baseline
    assert current_engine()=='disk' and implementation_digest()==baseline


def test_cancelled_polars_batch_cleans_owned_scratch(tmp_path):
    from runner.polars_preparation import preparation_engine
    from runner.dataflow_columns import apply
    from runner.dataflow_disk import disk_scope
    table=df.Table(({'name':'x','dtype':'string','nullable':False},),
                   tuple(('value',) for _ in range(5000)),tuple(str(i) for i in range(5000)),'s')
    calls=0
    def check():
        nonlocal calls
        calls+=1
        if calls==8:raise RuntimeError('cancelled')
    with disk_scope(directory=tmp_path),preparation_engine('polars'):
        with pytest.raises(RuntimeError,match='cancelled'):apply(table,columns_config([item('x')]),check)
    assert not list(tmp_path.iterdir())


def test_native_profile_selects_polars_and_preserves_training_bridge_and_history(tmp_path):
    import hashlib,json
    from runner import dataflow_cli as cli
    from runner.prepared_data import load_prepared
    source=tmp_path/'source.csv'
    with source.open('w') as stream:
        stream.write('x,y,label\n')
        for i in range(100005):stream.write(f'{i},{i%17},{i%2}\n')
    flow=graph(('filterRows',filter_config(conditions=[condition('label','ne','never')])),
               ('sampleRows',{'count':200,'seed':42}),
               ('columns',columns_config([item('x',dtype='float64'),item('y',dtype='float64'),item('label',dtype='int64')])),
               ('partition',{'strategy':'random','validationPercent':20,'testPercent':20,'seed':42}))
    flow['nodes'][0]['config']={'parserProfile':'polars'}
    path=tmp_path/'flow.json';path.write_text(json.dumps(flow));output=tmp_path/'prepared'
    manifest=cli.run_dataflow(path,{'source':str(source)},output,lambda _:None,cache_policy='materialize')
    assert manifest['preparationEngine']['name']=='polars'
    assert (output/'preparation.py').exists() and (output/'preparation-plan.json').exists()
    assert 'polars==2.0.0' in (output/'requirements-preparation.txt').read_text()
    data,provenance,_=load_prepared(output/'manifest.json','3','training','3','validation',['x','y'],'label')
    assert len(data[0])==120 and len(data[2])==40
    assert set(provenance['trainingRowIDs']).isdisjoint(provenance['validationRowIDs'])
    source.unlink()
    pin=hashlib.sha256((output/'manifest.json').read_bytes()).hexdigest()
    events=[];cli.inspect_dataflow(output/'manifest.json',pin,events.append)
    assert events[-1]['event']=='dataflowInspectionCompleted'
    assert events[1]['outputs']['table']['preview']['rowCount']==100005
    assert not list(output.glob('.row-scratch-*'))


def test_native_options_admission_does_not_import_native_libraries():
    import subprocess,sys
    result=subprocess.run([sys.executable,'-S','-c',
        'import sys;from runner.csv_parser import parse_options_json;'
        'assert parse_options_json(\'{"parserProfile":"polars"}\')["parserProfile"]=="polars";'
        'assert "polars" not in sys.modules;assert "torch" not in sys.modules'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_cancelled_preparation_does_not_open_graph_or_resolve_engine(tmp_path):
    from runner import dataflow_cli as cli
    def cancelled():raise RuntimeError('cancelled before capture')
    with pytest.raises(RuntimeError,match='cancelled before capture'):
        cli.run_dataflow(tmp_path/'absent.json',{},tmp_path/'output',lambda _:None,check_cancelled=cancelled)
