"""Out-of-core table preparation preserves the established v1 values/identities."""
import hashlib
import json
import tracemalloc
import pytest
from runner import dataflow as df, dataflow_cli as cli
from runner.csv_parser import CSVStream, parse_csv
from runner.tests.test_columns import config as columns_config,item
from runner.tests.test_filter_rows import config as filter_config,condition
from runner.tests.test_formula import operation

def graph(*operators):
    nodes=[{'id':'source','operator':'kwip.data.sourceCSV','config':{}}]
    edges=[]
    for i,(op,config) in enumerate(operators):
        node={'id':str(i),'operator':'kwip.data.'+op,'config':config}
        edges.append({'from':{'node':nodes[-1]['id'],'port':'matching' if nodes[-1]['operator']=='kwip.data.filterRows' else 'table'},'to':{'node':node['id'],'port':'table'}})
        nodes.append(node)
    return {'schemaVersion':1,'nodes':nodes,'edges':edges}

def test_stream_parser_matches_shared_csv_dialects(tmp_path):
    for encoding, content, options in [('utf-8','a;b\r\n"x\ny";"say !""hi!"""\r\n',{'separator':';'}),
                                      ('utf-16','a,b\n1,2\n',{'encoding':'utf-16'}),
                                      ('utf-8','1;2\n3;NA\n',{'separator':';','headerMode':'generated','nullMarkers':['NA']})]:
        path=tmp_path/'input.csv';path.write_text(content,encoding=encoding,newline='')
        expected=parse_csv(path,options)
        with CSVStream(path,options) as source:
            assert source.header == expected[0]
            assert list(source) == expected[1]
            assert source.source_sha256 == expected[2]

def test_500k_rows_execute_streamed_local_operators_with_bounded_python_memory(tmp_path):
    source=tmp_path/'large.csv';source.write_text('x,y\n'+'1,2\n'*500_001)
    flow=graph(('columns',columns_config([item('x',dtype='int64'),item('y')])),('filterRows',filter_config(conditions=[condition('x','eq',1)])),
               ('columns',columns_config([item('x','value')],newColumns='drop')))
    seen=[]
    result=df.execute(flow,{'source':str(source)},retain_results=False,on_node=lambda n,p:seen.append((n,len(p.get('table',p.get('matching')).rows),type(p.get('table',p.get('matching')).rows).__name__)))
    assert result == {}
    assert len(seen)==4 and all(count==500_001 and kind=='DiskRows' for _,count,kind in seen)

def test_retired_split_requires_repair_before_source_access(tmp_path,monkeypatch):
    source=tmp_path/'large.csv';source.write_text('x\n'+'1\n'*100_001)
    with pytest.raises(ValueError,match='repair'):
        df.execute(graph(('split',{'validationPercent':20,'seed':1})),{'source':str(source)},retain_results=False)

def test_large_source_sample_retains_original_ids_and_is_deterministic(tmp_path):
    source=tmp_path/'large.csv';source.write_text('x\n'+'1\n'*100_001)
    flow=graph(('sampleRows',{'count':20,'seed':7}))
    result=df.execute(flow,{'source':str(source)})
    sampled=result['0']['table']
    assert len(sampled.rows)==20
    assert list(sampled.row_ids)==list(df.execute(flow,{'source':str(source)})['0']['table'].row_ids)
    assert all(identity.startswith(hashlib.sha256(source.read_bytes()).hexdigest()+':') for identity in sampled.row_ids)

def test_large_cli_publishes_exact_artifacts_reuses_and_inspects_without_full_table_reads(tmp_path,monkeypatch):
    from runner import dataflow_storage
    from runner.dataflow_table_reader import read_table_artifact
    source=tmp_path/'source.csv';source.write_text('x,y\n'+'1,2\n'*100_005)
    flow=graph(('columns',columns_config([item('x')],newColumns='drop')),('sampleRows',{'count':20,'seed':7}))
    path=tmp_path/'flow.json';path.write_text(json.dumps(flow))
    output=tmp_path/'first';events=[]
    manifest=cli.run_dataflow(path,{'source':str(source)},output,events.append,cache_policy='materialize')
    assert (output/'previews.json').exists()
    assert not list(output.glob('.row-scratch-*'))
    for entry in manifest['artifacts']:
        assert hashlib.sha256((output/entry['path']).read_bytes()).hexdigest()==entry['sha256']
    pin=hashlib.sha256((output/'manifest.json').read_bytes()).hexdigest()
    original=dataflow_storage.bounded_regular_read
    def no_full_table(path,limit):
        if str(path).endswith(('node-000.json','node-001.json')):raise AssertionError('large table read into bytes')
        return original(path,limit)
    monkeypatch.setattr(dataflow_storage,'bounded_regular_read',no_full_table)
    restored=[];cli.inspect_dataflow(output/'manifest.json',pin,restored.append)
    assert restored[-1]['event']=='dataflowInspectionCompleted'
    assert restored[1]['outputs']['table']['preview']['rowCount']==100_005
    reused=cli.run_dataflow(path,{'source':str(source)},tmp_path/'second',lambda _:None,cache_policy='reuse',reuse_manifest=output/'manifest.json',reuse_manifest_sha256=pin)
    assert '0' in reused['cache']['reusedNodeIDs']
    table=next(entry for entry in manifest['artifacts'] if entry.get('nodeID')=='0')
    value=read_table_artifact(output/table['path'],table['sha256'])
    assert len(value.rows)==100_005 and list(value.rows[:2])==[('1',),('1',)]

def test_large_failure_and_cancellation_leave_no_success_or_spill_files(tmp_path):
    source=tmp_path/'bad.csv';source.write_text('x\n'+'1\n'*20_000+'2,3\n')
    path=tmp_path/'flow.json';path.write_text(json.dumps(graph()))
    events=[]
    with pytest.raises(ValueError,match='row width'):
        cli.run_dataflow(path,{'source':str(source)},tmp_path/'failed',events.append,retain_partial=True)
    assert not any(event['event']=='dataflowCompleted' for event in events)
    assert not list(tmp_path.rglob('rows.sqlite3'))
    assert not list(tmp_path.rglob('manifest.json'))

def test_streamed_calculation_and_expression_filter_match_eager_values_and_ids(tmp_path):
    source=tmp_path/'expressions.csv';source.write_text('x\n'+'1\n'*10_005)
    flow=graph(('columns',columns_config([item('x',dtype='int64')])),('formula',{'operations':[operation('[x] + 2',('z','int64'),policy='error')]}))
    result=df.execute(flow,{'source':str(source)})
    table=result['1']['table']
    assert len(table.rows)==10_005 and table.rows[0]==(1,3)
    from runner.dataflow_filter import apply
    small=df.Table(table.columns,tuple(table.rows[:8]),tuple(table.row_ids[:8]),table.source_digest)
    config=filter_config(mode='expression',expression='[z] == 3')
    eager=apply(small,config,lambda:None)
    actual=apply(table,config,lambda:None)
    assert len(actual['matching'].rows)==10_005 and len(actual['nonmatching'].rows)==0
    assert tuple(actual['matching'].rows[:8])==tuple(eager['matching'].rows)
    assert tuple(actual['matching'].row_ids[:8])==tuple(eager['matching'].row_ids)


def test_large_upstream_can_sample_into_training_without_decoding_upstream_tables(tmp_path,monkeypatch):
    from runner.prepared_data import load_prepared
    from runner import dataflow_storage
    source=tmp_path/'observations.csv'
    with source.open('w') as stream:
        stream.write('x,y,label\n')
        for index in range(100_005):stream.write(f'{index},{index%17},{index%2}\n')
    flow=graph(('sampleRows',{'count':200,'seed':42}),('columns',columns_config([item('x',dtype='float64'),item('y',dtype='float64'),item('label',dtype='int64')])),('partition',{'strategy':'random','validationPercent':20,'testPercent':20,'seed':42}))
    path=tmp_path/'flow.json';path.write_text(json.dumps(flow));output=tmp_path/'prepared'
    manifest=cli.run_dataflow(path,{'source':str(source)},output,lambda _:None)
    first=next(entry['path'] for entry in manifest['artifacts'] if entry.get('nodeID')=='source')
    original=dataflow_storage.bounded_regular_read
    def forbid_upstream(path,limit):
        if path.name==first:raise AssertionError('Training decoded complete upstream source')
        return original(path,limit)
    monkeypatch.setattr(dataflow_storage,'bounded_regular_read',forbid_upstream)
    data,provenance,files=load_prepared(output/'manifest.json','2','training','2','validation',['x','y'],'label')
    assert len(data[0])==120 and len(data[2])==40
    assert set(provenance['trainingRowIDs']).isdisjoint(provenance['validationRowIDs'])
    assert first in files
    from runner.dataflow_storage import admit_model_ports
    with pytest.raises(ValueError,match='Row Sample'):
        admit_model_ports(files,manifest,{('source','table')})
