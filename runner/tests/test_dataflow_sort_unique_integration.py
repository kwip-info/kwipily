import hashlib
import json
from pathlib import Path
import pytest
from runner.dataflow_cli import run_dataflow
from runner.polars_csv_configuration import DEFAULTS
from runner.tests.test_dataflow_portable_dependencies import replay


def fixture(tmp_path,tool):
    source=tmp_path/'records.csv';source.write_text('request,customer,priority\nS1,Beta,1\nS2,alpha,2\nS3,ALPHA,2\nS4,,9\nS5,alpha,\nS6,beta,3\n')
    config=(dict(keys=[dict(column='customer',direction='ascending',missing='last',comparison='casefold'),dict(column='priority',direction='descending',missing='first',comparison='exact')]) if tool=='sort' else dict(keys=['customer'],keep='last',missing='equal',comparison='casefold'))
    graph=dict(schemaVersion=1,nodes=[dict(id='source',operator='kwip.data.sourceCSV',config={**DEFAULTS,'emptyField':'null','schemaOverrides':{'priority':'Int64'}}),dict(id=tool,operator='kwip.data.'+tool,config=config)],edges=[{'from':dict(node='source',port='table'),'to':dict(node=tool,port='table')}])
    path=tmp_path/'graph.json';path.write_text(json.dumps(graph));return path,dict(source=str(source))

def output_rows(folder,manifest,tool):
    return {entry['port']:json.loads((folder/entry['path']).read_text())['rows'] for entry in manifest['artifacts'] if entry.get('nodeID')==tool and entry['kind']=='table'}

@pytest.mark.parametrize('engine',['disk','polars'])
@pytest.mark.parametrize('tool',['sort','unique'])
def test_exact_outputs_cache_and_standalone(tmp_path,engine,tool):
    path,bindings=fixture(tmp_path,tool);first=tmp_path/'first'
    manifest=run_dataflow(path,bindings,first,lambda _:None,engine=engine,cache_policy='materialize')
    rows=output_rows(first,manifest,tool)
    if tool=='sort':assert [r[0] for r in rows['table']]==['S5','S2','S3','S6','S1','S4']
    else:
        assert [r[0] for r in rows['unique']]==['S4','S5','S6']
        assert [r[0] for r in rows['duplicates']]==['S1','S2','S3']
    pin=hashlib.sha256((first/'manifest.json').read_bytes()).hexdigest()
    cached=run_dataflow(path,bindings,tmp_path/'cached',lambda _:None,engine=engine,cache_policy='reuse',reuse_manifest=first/'manifest.json',reuse_manifest_sha256=pin)
    assert cached['cache']['reusedNodeIDs']==[tool]
    assert cached['artifacts']==manifest['artifacts']
    if tool=='unique':
        expected_paths={'unique':'node-001.json','duplicates':'node-002.json'}
        for receipt in (manifest,cached):
            assert {entry['port']:entry['path'] for entry in receipt['artifacts'] if entry.get('nodeID')==tool}==expected_paths
    private=tmp_path/'bindings.json';private.write_text(json.dumps(bindings));cwd=tmp_path/'cwd';cwd.mkdir()
    assert replay(first,private,tmp_path/'replay',cwd)['artifacts']==manifest['artifacts']
    Path(bindings['source']).write_text('request,customer,priority\nnew,z,1\n')
    changed=run_dataflow(path,bindings,tmp_path/'changed',lambda _:None,engine=engine,cache_policy='reuse',reuse_manifest=first/'manifest.json',reuse_manifest_sha256=pin)
    assert changed['cache']['reusedNodeIDs']==[]

@pytest.mark.parametrize('tool',['sort','unique'])
def test_working_population_and_complete_view_demand(tmp_path,tool):
    path,bindings=fixture(tmp_path,tool)
    sampled=run_dataflow(path,bindings,tmp_path/'sample',lambda _:None,sampling={'mode':'rowCount','rowCount':1})
    expected={'table':1} if tool=='sort' else {'unique':1,'duplicates':0}
    assert sampled['executionCoverage']['nodes'][tool]['rowCounts']==expected
    graph=json.loads(path.read_text());graph['nodes'].append(dict(id='view',operator='kwip.data.view',config={}))
    graph['edges'].append({'from':dict(node=tool,port='table' if tool=='sort' else 'unique'),'to':dict(node='view',port='table')});path.write_text(json.dumps(graph))
    complete=run_dataflow(path,bindings,tmp_path/'complete',lambda _:None,sampling={'mode':'rowCount','rowCount':1})
    expected={'table':6} if tool=='sort' else {'unique':3,'duplicates':3}
    assert complete['executionCoverage']['status']=='complete'
    assert complete['executionCoverage']['nodes'][tool]['rowCounts']==expected

@pytest.mark.parametrize('tool',['sort','unique'])
@pytest.mark.parametrize('mutation',['value','order','identity','role'])
def test_resealed_cache_refuses_forged_outputs(tmp_path,tool,mutation):
    from runner.dataflow import digest,membership_digest
    path,bindings=fixture(tmp_path,tool);first=tmp_path/'first'
    manifest=run_dataflow(path,bindings,first,lambda _:None,cache_policy='materialize')
    entry=next(x for x in manifest['artifacts'] if x.get('nodeID')==tool and x['kind']=='table')
    payload=first/entry['path'];value=json.loads(payload.read_text());lineage=value['lineage']
    if mutation=='value':value['rows'][0][0]='forged'
    if mutation=='order':value['rows'].reverse();lineage['observationIDs'].reverse()
    if mutation=='identity':lineage['observationIDs'][0]='forged'
    if mutation=='role':lineage['role']='training'
    lineage['dataVersion']=digest(dict(columns=value['columns'],rows=value['rows'],ids=lineage['observationIDs']));lineage['membershipDigest']=membership_digest(lineage['observationIDs'])
    raw=json.dumps(value).encode();payload.write_bytes(raw);entry.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    manifest['cache']['materializedBytes']=sum(x['bytes'] for x in manifest['artifacts'] if x['kind'] in ('table','state'))
    pinfile=first/'manifest.json';pinfile.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):run_dataflow(path,bindings,tmp_path/'bad',lambda _:None,cache_policy='reuse',reuse_manifest=pinfile,reuse_manifest_sha256=hashlib.sha256(pinfile.read_bytes()).hexdigest())
    assert not (tmp_path/'bad').exists()
