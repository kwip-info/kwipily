"""Grouped preparation must remain reproducible through cache and standalone export."""
import hashlib
import json
from pathlib import Path
import pytest
from runner.dataflow import execute, digest, membership_digest
from runner.dataflow_cli import run_dataflow
from runner.tests.test_dataflow_portable_dependencies import replay
from runner.tests.test_columns import config as columns_config,item


def fixture(tmp_path):
    source=tmp_path/'measurements.csv'
    source.write_text('subject,value\na,1\nb,8\na,3\nb,12\n')
    config={'groupBy':['subject'],'nullKeys':'error','aggregations':[
        {'op':'mean','column':'value','nullPolicy':'error','output':'average'},
        {'op':'countRows','output':'observations'}],'maxRows':1000}
    graph={'schemaVersion':1,'nodes':[
        {'id':'source','operator':'kwip.data.sourceCSV','config':{}},
        {'id':'cast','operator':'kwip.data.columns','config':columns_config([item('subject'),item('value',dtype='float64')])},
        {'id':'summary','operator':'kwip.data.aggregate','config':config}],
        'edges':[{'from':{'node':a,'port':'table'},'to':{'node':b,'port':'table'}} for a,b in [('source','cast'),('cast','summary')]]}
    path=tmp_path/'graph.json';path.write_text(json.dumps(graph))
    return graph,path,{'source':str(source)}


def test_aggregate_execution_cache_export_and_source_change(tmp_path):
    graph,path,bindings=fixture(tmp_path)
    first=tmp_path/'first';manifest=run_dataflow(path,bindings,first,lambda _:None,cache_policy='materialize')
    entry=next(e for e in manifest['artifacts'] if e.get('nodeID')=='summary')
    actual=json.loads((first/entry['path']).read_text())
    assert actual['rows']==[['a',2.0,2],['b',10.0,2]]
    assert [c['name'] for c in actual['columns']]==['subject','average','observations']
    pin=hashlib.sha256((first/'manifest.json').read_bytes()).hexdigest()
    cached=run_dataflow(path,bindings,tmp_path/'cached',lambda _:None,cache_policy='reuse',reuse_manifest=first/'manifest.json',reuse_manifest_sha256=pin)
    assert cached['cache']['reusedNodeIDs']==['cast','summary']
    assert cached['artifacts']==manifest['artifacts']
    private=tmp_path/'bindings.json';private.write_text(json.dumps(bindings));cwd=tmp_path/'unrelated';cwd.mkdir()
    assert replay(first,private,tmp_path/'replay',cwd)['artifacts']==manifest['artifacts']
    Path(bindings['source']).write_text('subject,value\na,5\nb,8\na,3\nb,12\n')
    changed=run_dataflow(path,bindings,tmp_path/'changed',lambda _:None,cache_policy='reuse',reuse_manifest=first/'manifest.json',reuse_manifest_sha256=pin)
    assert changed['cache']['reusedNodeIDs']==[]


def test_exact_decimal_formula_and_sum_replay_outside_repository_and_reuse_cache(tmp_path):
    source=tmp_path/'decimal-source.csv'
    source.write_text('quantity,price\n1e50,0.005\n-1e50,0.005\n3,0.005\n')
    operation={'id':'multiply','expression':'decimal_multiply([quantity], [price])',
        'outputs':[{'name':'amount','dtype':'string','write':'new'}],'nullPolicy':'error'}
    graph={'schemaVersion':1,'nodes':[
        {'id':'source','operator':'kwip.data.sourceCSV','config':{}},
        {'id':'formula','operator':'kwip.data.formula','config':{'operations':[operation]}},
        {'id':'summary','operator':'kwip.data.aggregate','config':{'groupBy':[],'nullKeys':'error',
            'aggregations':[{'op':'sumDecimal','column':'amount','output':'total','nullPolicy':'error'}],'maxRows':1000}}],
        'edges':[{'from':{'node':a,'port':'table'},'to':{'node':b,'port':'table'}} for a,b in [('source','formula'),('formula','summary')]]}
    path=tmp_path/'graph.json';path.write_text(json.dumps(graph));bindings={'source':str(source)}
    first=tmp_path/'first';manifest=run_dataflow(path,bindings,first,lambda _:None,cache_policy='materialize')
    entry=next(e for e in manifest['artifacts'] if e.get('nodeID')=='summary')
    actual=json.loads((first/entry['path']).read_text())
    assert actual['rows']==[['0.015']]
    assert actual['columns'][0]['dtype']=='string'
    pin=hashlib.sha256((first/'manifest.json').read_bytes()).hexdigest()
    cached=run_dataflow(path,bindings,tmp_path/'cached',lambda _:None,cache_policy='reuse',reuse_manifest=first/'manifest.json',reuse_manifest_sha256=pin)
    assert cached['cache']['reusedNodeIDs']==['formula','summary']
    private=tmp_path/'bindings.json';private.write_text(json.dumps(bindings));cwd=tmp_path/'unrelated';cwd.mkdir()
    assert replay(first,private,tmp_path/'replay',cwd)['artifacts']==manifest['artifacts']


@pytest.mark.parametrize('bad',[{'op':'countRows','output':'n','column':'value'}, {'op':'sum','column':'value','output':'n'}, {'op':'median','column':'value','output':'n'}])
def test_invalid_reducer_precedes_missing_source_read(tmp_path,bad):
    graph,_,_=fixture(tmp_path);graph['nodes'][-1]['config']['aggregations']=[bad]
    with pytest.raises(ValueError) as error:execute(graph,{'source':'/missing.csv'})
    assert 'missing.csv' not in str(error.value)
    assert 'Unknown operator' not in str(error.value)


@pytest.mark.parametrize('mutation',['ids','dropped','values','role'])
def test_resealed_aggregate_cache_cannot_forge_group_evidence(tmp_path,mutation):
    _,path,bindings=fixture(tmp_path)
    first=tmp_path/'first';manifest=run_dataflow(path,bindings,first,lambda _:None,cache_policy='materialize')
    entry=next(e for e in manifest['artifacts'] if e.get('nodeID')=='summary')
    payload=first/entry['path'];value=json.loads(payload.read_text());lineage=value['lineage']
    if mutation=='ids':lineage['observationIDs'][0]='invented'
    elif mutation=='dropped':value['rows'].pop();lineage['observationIDs'].pop()
    elif mutation=='role':lineage['role']='training'
    else:value['rows'][0][1]=99.0
    lineage['dataVersion']=digest({'columns':value['columns'],'rows':value['rows'],'ids':lineage['observationIDs']})
    lineage['membershipDigest']=membership_digest(lineage['observationIDs'])
    raw=json.dumps(value).encode();payload.write_bytes(raw);entry.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    manifest['cache']['materializedBytes']=sum(e['bytes'] for e in manifest['artifacts'] if e['kind'] in ('table','state'))
    manifest_path=first/'manifest.json';manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='[Aa]ggregat'):
        run_dataflow(path,bindings,tmp_path/'bad',lambda _:None,cache_policy='reuse',reuse_manifest=manifest_path,reuse_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    assert not (tmp_path/'bad').exists()


@pytest.mark.parametrize('engine',['disk','polars'])
def test_text_join_cache_and_standalone_export(tmp_path,engine):
    graph,path,bindings=fixture(tmp_path)
    graph['nodes'][-1]['config']['aggregations'].append({'op':'joinText','column':'subject','output':'subjects','nullPolicy':'skip','separator':'\n','distinct':False,'skipEmpty':False})
    path.write_text(json.dumps(graph))
    first=tmp_path/'first';manifest=run_dataflow(path,bindings,first,lambda _:None,cache_policy='materialize',engine=engine)
    pin=hashlib.sha256((first/'manifest.json').read_bytes()).hexdigest()
    cached=run_dataflow(path,bindings,tmp_path/'cached',lambda _:None,cache_policy='reuse',engine=engine,reuse_manifest=first/'manifest.json',reuse_manifest_sha256=pin)
    assert cached['artifacts']==manifest['artifacts']
    assert 'summary' in cached['cache']['reusedNodeIDs']
    entry=next(e for e in manifest['artifacts'] if e.get('nodeID')=='summary')
    assert json.loads((first/entry['path']).read_text())['rows']==[['a',2.0,2,'a\na'],['b',10.0,2,'b\nb']]
    private=tmp_path/'bindings.json';private.write_text(json.dumps(bindings));cwd=tmp_path/'unrelated';cwd.mkdir()
    assert replay(first,private,tmp_path/'replay',cwd)['artifacts']==manifest['artifacts']
