"""Callback materialization releases tables without changing graph semantics."""
import gc
import hashlib
import json
import weakref
from pathlib import Path
import pytest
from runner import dataflow as df, dataflow_cli as cli
from runner.dataflow_cache import Materialization
from runner.tests.test_columns import config,item

BASE=Path(__file__).resolve().parents[1]/'dataflow_examples'

def chain(count=4, columns=2):
    names=[f'x{i}' for i in range(columns)]
    nodes=[{'id':'source','operator':'kwip.data.sourceCSV','config':{}}]
    edges=[]
    for i in range(count):
        node=f'step{i}'
        nodes.append({'id':node,'operator':'kwip.data.columns','config':config([item(name) for name in names],newColumns='drop')})
        edges.append({'from':{'node':nodes[-2]['id'],'port':'table'},'to':{'node':node,'port':'table'}})
    return {'schemaVersion':1,'nodes':nodes,'edges':edges}

def test_streaming_releases_consumed_tables_after_callback_and_preserves_default(tmp_path):
    source=tmp_path/'source.csv';source.write_text('x0,x1\n1,2\n3,4\n')
    graph=chain();expected=df.execute(graph,{'source':str(source)})
    refs={};seen=[]
    def observe(node,ports):
        gc.collect()
        if node=='step2':
            assert refs['source']() is None
            assert refs['step0']() is None
        assert ports==expected[node]
        refs[node]=weakref.ref(ports['table']);seen.append(node)
    # Compare values without retaining expected Table instances in the streaming run.
    result=df.execute(graph,{'source':str(source)},on_node=observe,retain_results=False)
    assert result=={} and seen==[node['id'] for node in graph['nodes']]
    gc.collect();assert all(ref() is None for ref in refs.values())
    assert len(expected)==5

def test_large_chain_uses_live_cells_not_cumulative_materialized_cells(tmp_path):
    source=tmp_path/'large.csv';source.write_text(','.join(f'x{i}' for i in range(6))+'\n'+('1,2,3,4,5,6\n'*100_000))
    graph=chain(count=9,columns=6);seen=[]
    retained=df.execute(graph,{'source':str(source)})
    assert all(isinstance(ports['table'].rows,df.DiskRows) for ports in retained.values())
    del retained;gc.collect()
    assert df.execute(graph,{'source':str(source)},on_node=lambda n,p:seen.append((n,len(p['table'].rows))),retain_results=False)=={}
    assert len(seen)==10 and all(rows==100_000 for _,rows in seen)

def test_branched_fitted_states_and_cli_reuse_preserve_every_artifact(tmp_path,monkeypatch):
    graph=json.loads((BASE/'preparation.json').read_text());bindings={'source':str(BASE/'observations.csv')}
    expected=df.execute(graph,bindings);observed={}
    assert df.execute(graph,bindings,on_node=lambda n,p:observed.update({n:p}),retain_results=False)=={}
    assert observed==expected
    first=tmp_path/'first';a=cli.run_dataflow(BASE/'preparation.json',bindings,first,lambda _:None,cache_policy='materialize')
    prior=Materialization('reuse',first/'manifest.json',hashlib.sha256((first/'manifest.json').read_bytes()).hexdigest())
    df.execute(graph,bindings,materialization=prior,retain_results=False)
    assert set(prior.verified_files)=={'dataflow.json','preparation-plan.json'}, 'Only bounded graph and demand-proof documents remain retained; decoded table payloads do not'
    def forbidden(*a,**k):raise AssertionError('cache hit re-fitted state')
    monkeypatch.setattr('runner.dataflow_preprocess._fit',forbidden)
    b=cli.run_dataflow(BASE/'preparation.json',bindings,tmp_path/'second',lambda _:None,cache_policy='reuse',
        reuse_manifest=first/'manifest.json',reuse_manifest_sha256=hashlib.sha256((first/'manifest.json').read_bytes()).hexdigest())
    assert a['artifacts']==b['artifacts']

class Cancelled(Exception):pass

def test_source_cancelled_inside_parse_before_consuming_all_rows(tmp_path,monkeypatch):
    source=tmp_path/'source.csv';source.write_text('x\n'+('1\n'*20_000))
    original=df.csv.reader;count=0
    def reader(*args,**kwargs):
        underlying=original(*args,**kwargs)
        class Spy:
            @property
            def line_num(self):return underlying.line_num
            def __iter__(self):return self
            def __next__(self):
                nonlocal count
                result=next(underlying);count+=1;return result
        return Spy()
    monkeypatch.setattr(df.csv,'reader',reader)
    def check():
        if count>512:raise Cancelled()
    with pytest.raises(Cancelled):df.source_csv(source,check_cancelled=check)
    assert 512<count<2_000

def test_cast_cancelled_inside_large_transform(tmp_path,monkeypatch):
    source=tmp_path/'source.csv';source.write_text('x0\n'+('1\n'*20_000))
    graph=chain(count=0,columns=1)
    graph['nodes'].append({'id':'cast','operator':'kwip.data.columns','config':config([item('x0',dtype='float64')])})
    graph['edges'].append({'from':{'node':'source','port':'table'},'to':{'node':'cast','port':'table'}})
    original=df.cast_value;count=0
    def cast(*args):
        nonlocal count
        count+=1;return original(*args)
    monkeypatch.setattr(df,'cast_value',cast)
    def check():
        if count>512:raise Cancelled()
    with pytest.raises(Cancelled):df.execute(graph,{'source':str(source)},check_cancelled=check)
    assert 512<count<2_000

def test_streaming_still_bounds_simultaneously_live_input_and_output(tmp_path):
    source=tmp_path/'wide.csv'
    source.write_text(','.join(f'x{i}' for i in range(40))+'\n'+(','.join(['1']*40)+'\n')*100_000)
    assert source.stat().st_size<df.MAX_BYTES
    # Disk-backed input/output no longer consume the eager retained-cell budget.
    seen=[]
    df.execute(chain(count=1,columns=40),{'source':str(source)},retain_results=False,
        on_node=lambda n,p:seen.append((len(p['table'].rows),isinstance(p['table'].rows,df.DiskRows))))
    assert seen==[(100_000,True),(100_000,True)]

def test_large_cli_persists_all_streamed_outputs_with_exact_inventory(tmp_path):
    source=tmp_path/'large.csv';source.write_text(','.join(f'x{i}' for i in range(6))+'\n'+('1,2,3,4,5,6\n'*100_000))
    graph=chain(count=9,columns=6);path=tmp_path/'graph.json';path.write_text(json.dumps(graph))
    events=[]
    manifest=cli.run_dataflow(path,{'source':str(source)},tmp_path/'output',events.append,cache_policy='materialize')
    tables=[entry for entry in manifest['artifacts'] if entry['kind']=='table']
    assert len(tables)==10 and len({entry['sha256'] for entry in tables})==1
    assert manifest['cache']['materializedBytes']==sum(entry['bytes'] for entry in tables)
    assert [event['nodeID'] for event in events if event['event']=='dataflowPreview']==[node['id'] for node in graph['nodes']]
    assert all(event['outputs']['table']['preview']['rowCount']==100_000 for event in events if event['event']=='dataflowPreview')
    for entry in manifest['artifacts']:
        content=(tmp_path/'output'/entry['path']).read_bytes()
        assert len(content)==entry['bytes'] and hashlib.sha256(content).hexdigest()==entry['sha256']
    assert events[-1]['event']=='dataflowCompleted'
