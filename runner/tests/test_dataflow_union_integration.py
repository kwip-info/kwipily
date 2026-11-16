import hashlib
import json
from pathlib import Path
import pytest
from runner.dataflow_cli import run_dataflow
from runner.dataflow import digest, membership_digest
from runner.tests.test_dataflow_portable_dependencies import replay


def fixture(tmp_path):
    a=tmp_path/'a.csv';a.write_text('id,value\n1,a\n1,a\n')
    b=tmp_path/'b.csv';b.write_text('value,extra\nb,4\n')
    config=dict(inputs=['table1','table2'],match='name',coverage='all',typePolicy='strict',sourceColumn='input')
    graph=dict(schemaVersion=1,nodes=[dict(id='a',operator='kwip.data.sourceCSV',config={}),dict(id='b',operator='kwip.data.sourceCSV',config={}),dict(id='union',operator='kwip.data.union',config=config)],edges=[dict(**{'from':dict(node=n,port='table'),'to':dict(node='union',port=p)}) for n,p in [('a','table1'),('b','table2')]])
    path=tmp_path/'graph.json';path.write_text(json.dumps(graph));return path,dict(a=str(a),b=str(b))

@pytest.mark.parametrize('engine',['disk','polars'])
def test_cache_standalone_replay_and_changed_sources(tmp_path,engine):
    path,bindings=fixture(tmp_path)
    first=tmp_path/'first';manifest=run_dataflow(path,bindings,first,lambda _:None,engine=engine,cache_policy='materialize')
    entry=next(x for x in manifest['artifacts'] if x.get('nodeID')=='union')
    assert json.loads((first/entry['path']).read_text())['rows']==[['1','a',None,'table1'],['1','a',None,'table1'],[None,'b','4','table2']]
    pin=hashlib.sha256((first/'manifest.json').read_bytes()).hexdigest()
    cached=run_dataflow(path,bindings,tmp_path/'cached',lambda _:None,engine=engine,cache_policy='reuse',reuse_manifest=first/'manifest.json',reuse_manifest_sha256=pin)
    assert cached['cache']['reusedNodeIDs']==['union']
    assert cached['artifacts']==manifest['artifacts']
    private=tmp_path/'bindings.json';private.write_text(json.dumps(bindings));cwd=tmp_path/'cwd';cwd.mkdir()
    assert replay(first,private,tmp_path/'replay',cwd)['artifacts']==manifest['artifacts']
    Path(bindings['b']).write_text('value,extra\nc,5\n')
    changed=run_dataflow(path,bindings,tmp_path/'changed',lambda _:None,engine=engine,cache_policy='reuse',reuse_manifest=first/'manifest.json',reuse_manifest_sha256=pin)
    assert changed['cache']['reusedNodeIDs']==[]

@pytest.mark.parametrize('mutation',['value','order','identity','role'])
def test_resealed_cache_cannot_forge_union_output(tmp_path,mutation):
    path,bindings=fixture(tmp_path);first=tmp_path/'first'
    manifest=run_dataflow(path,bindings,first,lambda _:None,cache_policy='materialize')
    entry=next(x for x in manifest['artifacts'] if x.get('nodeID')=='union');payload=first/entry['path'];value=json.loads(payload.read_text());lineage=value['lineage']
    if mutation=='value':value['rows'][0][1]='forged'
    if mutation=='order':value['rows'].reverse();lineage['observationIDs'].reverse()
    if mutation=='identity':lineage['observationIDs'][0]='forged'
    if mutation=='role':lineage['role']='training'
    lineage['dataVersion']=digest(dict(columns=value['columns'],rows=value['rows'],ids=lineage['observationIDs']));lineage['membershipDigest']=membership_digest(lineage['observationIDs'])
    raw=json.dumps(value).encode();payload.write_bytes(raw);entry.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    manifest['cache']['materializedBytes']=sum(x['bytes'] for x in manifest['artifacts'] if x['kind'] in ('table','state'))
    pinfile=first/'manifest.json';pinfile.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='Union'):
        run_dataflow(path,bindings,tmp_path/'bad',lambda _:None,cache_policy='reuse',reuse_manifest=pinfile,reuse_manifest_sha256=hashlib.sha256(pinfile.read_bytes()).hexdigest())
    assert not (tmp_path/'bad').exists()


def test_working_sample_and_view_complete_demand(tmp_path):
    path,bindings=fixture(tmp_path)
    sampled=run_dataflow(path,bindings,tmp_path/'sample',lambda _:None,sampling={'mode':'rowCount','rowCount':1})
    assert sampled['executionCoverage']['nodes']['union']['rowCounts']=={'table':2}
    graph=json.loads(path.read_text());graph['nodes'].append(dict(id='view',operator='kwip.data.view',config={}))
    graph['edges'].append({'from':dict(node='union',port='table'),'to':dict(node='view',port='table')});path.write_text(json.dumps(graph))
    complete=run_dataflow(path,bindings,tmp_path/'complete',lambda _:None,sampling={'mode':'rowCount','rowCount':1})
    assert complete['executionCoverage']['status']=='complete'
    assert complete['executionCoverage']['nodes']['union']['rowCounts']=={'table':3}

def test_dynamic_ports_reject_unknown_missing_and_duplicate_before_source_reads(tmp_path):
    from runner.dataflow import validate
    path,_=fixture(tmp_path);graph=json.loads(path.read_text())
    graph['edges'][0]['to']['port']='table3'
    with pytest.raises(ValueError):validate(graph)
    graph['edges'].pop(0)
    with pytest.raises(ValueError):validate(graph)
