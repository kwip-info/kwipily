import hashlib
import json
import pytest
from runner.dataflow import execute, digest, membership_digest
from runner.dataflow_cli import run_dataflow
from runner.tests.test_dataflow_join import settings
from runner.tests.test_dataflow_portable_dependencies import replay


def fixture(tmp_path,mode='inner'):
    a=tmp_path/'a.csv';a.write_text('key,x\n1,10\n2,20\n2,21\n')
    b=tmp_path/'b.csv';b.write_text('key,y\n2,200\n3,300\n')
    graph={'schemaVersion':1,'nodes':[
        {'id':key,'operator':'kwip.data.sourceCSV','config':{}} for key in ('a','b')]+[
        {'id':'joined','operator':'kwip.data.join','config':settings(mode=mode)}],
        'edges':[{'from':{'node':key,'port':'table'},'to':{'node':'joined','port':port}} for key,port in [('a','left'),('b','right')]]}
    path=tmp_path/'graph.json';path.write_text(json.dumps(graph))
    return graph,path,{'a':str(a),'b':str(b)}


@pytest.mark.parametrize('mode',['inner','left','right','full'])
@pytest.mark.parametrize('projected',[False,True])
def test_join_execution_cache_and_standalone_export(tmp_path,mode,projected):
    graph,path,bindings=fixture(tmp_path,mode)
    if projected:
        graph['nodes'][-1]['config'].update(outputColumns=[dict(side='right',source='y',name='context',included=True),dict(side='left',source='key',name='id',included=True)],newColumns='drop')
        path.write_text(json.dumps(graph))
    first=tmp_path/'first';manifest=run_dataflow(path,bindings,first,lambda _:None,cache_policy='materialize')
    pin=hashlib.sha256((first/'manifest.json').read_bytes()).hexdigest()
    cached=run_dataflow(path,bindings,tmp_path/'cached',lambda _:None,cache_policy='reuse',reuse_manifest=first/'manifest.json',reuse_manifest_sha256=pin)
    assert cached['cache']['reusedNodeIDs']==['joined']
    assert cached['artifacts']==manifest['artifacts']
    private=tmp_path/'bindings.json';private.write_text(json.dumps(bindings));cwd=tmp_path/'unrelated';cwd.mkdir()
    exported=replay(first,private,tmp_path/'replay',cwd)
    assert exported['artifacts']==manifest['artifacts']
    from pathlib import Path
    Path(bindings['b']).write_text('key,y\n2,999\n3,300\n')
    changed=run_dataflow(path,bindings,tmp_path/'changed',lambda _:None,cache_policy='reuse',reuse_manifest=first/'manifest.json',reuse_manifest_sha256=pin)
    assert changed['cache']['reusedNodeIDs']==[]


def test_invalid_join_config_precedes_missing_source_read(tmp_path):
    graph,path,bindings=fixture(tmp_path)
    graph['nodes'][-1]['config']['rightKeys']=['key','y']
    with pytest.raises(ValueError,match='equal length'):execute(graph,{key:'/missing.csv' for key in bindings})


@pytest.mark.parametrize('mutation',['ids','dropped','source','role','values','fitted'])
def test_resealed_invalid_join_cache_rejected(tmp_path,mutation):
    graph,path,bindings=fixture(tmp_path)
    first=tmp_path/'first';manifest=run_dataflow(path,bindings,first,lambda _:None,cache_policy='materialize')
    entry=next(e for e in manifest['artifacts'] if e.get('nodeID')=='joined')
    payload=first/entry['path'];value=json.loads(payload.read_text());lineage=value['lineage']
    if mutation=='ids':lineage['observationIDs'][0]='invented'
    elif mutation=='dropped':value['rows'].pop();lineage['observationIDs'].pop()
    elif mutation=='source':lineage['sourceDigest']='f'*64
    elif mutation=='role':lineage['role']='training'
    elif mutation=='values':value['rows'][0][-1]='fake'
    else:lineage['fittedStateDigests']=['a'*64]
    lineage['dataVersion']=digest({'columns':value['columns'],'rows':value['rows'],'ids':lineage['observationIDs']})
    lineage['membershipDigest']=membership_digest(lineage['observationIDs'])
    raw=json.dumps(value).encode();payload.write_bytes(raw);entry.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    manifest['cache']['materializedBytes']=sum(e['bytes'] for e in manifest['artifacts'] if e['kind'] in ('table','state'))
    manifest_path=first/'manifest.json';manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='[Jj]oin'):
        run_dataflow(path,bindings,tmp_path/'bad',lambda _:None,cache_policy='reuse',reuse_manifest=manifest_path,reuse_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    assert not (tmp_path/'bad').exists()
