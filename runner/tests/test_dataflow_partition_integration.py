"""Partition contracts survive materialization, export and test-role isolation."""
import hashlib
import json
import math
from pathlib import Path
import pytest
from runner.dataflow import execute, digest, membership_digest,fit_observation_ids
from runner.dataflow_cli import run_dataflow
from runner.tests.test_dataflow_portable_dependencies import replay
from runner.tests.test_columns import config as columns_config,item
from runner.dataflow_preprocess import rule_defaults


def fixture(tmp_path,operator):
    source=tmp_path/'observations.csv'
    source.write_text('subject,time,x,label\n'+''.join(f'g{g},{g*1000},{g+offset/10},{g%2}\n' for g in range(60) for offset in (1,2)))
    config=({'strategy':'entity','groupBy':['subject'],'seed':42,'validationPercent':20,'testPercent':20} if operator=='entity' else
            {'strategy':'time','timeColumn':'time','timeEncoding':'epochMilliseconds','trainingEnd':35000,'validationEnd':47000,'embargoMilliseconds':0,'entityColumns':['subject'],'entityPolicy':'requireDisjoint'})
    graph={'schemaVersion':1,'nodes':[
        {'id':'source','operator':'kwip.data.sourceCSV','config':{}},
        {'id':'cast','operator':'kwip.data.columns','config':columns_config([item('subject'),item('time',dtype='int64'),item('x',dtype='float64'),item('label',dtype='int64')])},
        {'id':'partition','operator':'kwip.data.partition','config':config},
        {'id':'fit','operator':'kwip.data.preprocess','config':{'mode':'partitions','columns':[dict(rule_defaults('x'),missing='error',scaling='standard')]}},
        {'id':'test-scaled','operator':'kwip.data.view','config':{}}],
        'edges':[{'from':{'node':a,'port':ap},'to':{'node':b,'port':bp}} for a,ap,b,bp in [('source','table','cast','table'),('cast','table','partition','table'),('partition','training','fit','training'),('partition','validation','fit','validation'),('partition','test','fit','test'),('fit','test','test-scaled','table')]]}
    path=tmp_path/'graph.json';path.write_text(json.dumps(graph))
    return graph,path,{'source':str(source)}


@pytest.mark.parametrize('operator',['entity','time'])
def test_three_partitions_transform_cache_and_independent_export(tmp_path,operator):
    graph,path,bindings=fixture(tmp_path,operator)
    output=tmp_path/'first';manifest=run_dataflow(path,bindings,output,lambda _:None,cache_policy='materialize')
    def artifact(node,port):
        entry=next(e for e in manifest['artifacts'] if e.get('nodeID')==node and e.get('port')==port)
        return json.loads((output/entry['path']).read_text())
    partitions={role:artifact('partition',role) for role in ('training','validation','test')}
    memberships=[set(t['lineage']['observationIDs']) for t in partitions.values()]
    assert all(not a&b for i,a in enumerate(memberships) for b in memberships[i+1:])
    assert len(set.union(*memberships))==120
    assert [len(partitions[role]['rows']) for role in ('training','validation','test')]==[72,24,24]
    assert artifact('test-scaled','table')['lineage']['role']=='test'
    fit=artifact('fit','state')
    assert set(fit_observation_ids(fit))==memberships[0]
    assert artifact('partition','exclusions')['rows']==[]
    # Numerical oracle uses only the retained raw Training population.
    values=[row[2] for row in partitions['training']['rows']]
    mean=math.fsum(values)/len(values);scale=math.sqrt(math.fsum((value-mean)**2 for value in values)/len(values))
    assert math.isclose(fit['rules'][0]['mean'],mean,rel_tol=1e-12,abs_tol=1e-12)
    assert math.isclose(fit['rules'][0]['scale'],scale,rel_tol=1e-12,abs_tol=1e-12)
    scaled=artifact('test-scaled','table')
    assert scaled['lineage']['observationIDs']==partitions['test']['lineage']['observationIDs']
    assert [row[2] for row in scaled['rows']]==pytest.approx([(row[2]-mean)/scale for row in partitions['test']['rows']],rel=1e-12,abs=1e-12)
    assert [column['dtype'] for column in scaled['columns']]==['string','int64','float64','int64']
    entities=[{row[0] for row in partitions[role]['rows']} for role in ('training','validation','test')]
    assert all(not left&right for i,left in enumerate(entities) for right in entities[i+1:])
    for role in ('training','validation','test'):
        assert [row[1] for row in partitions[role]['rows']]==sorted(row[1] for row in partitions[role]['rows'])
    pin=hashlib.sha256((output/'manifest.json').read_bytes()).hexdigest()
    cached=run_dataflow(path,bindings,tmp_path/'cached',lambda _:None,cache_policy='reuse',reuse_manifest=output/'manifest.json',reuse_manifest_sha256=pin)
    assert cached['artifacts']==manifest['artifacts']
    # Cache hits preserve each port's artifact name as well as its table values.
    expected_paths={'training':'node-002.json','validation':'node-003.json','test':'node-004.json','exclusions':'node-005.json'}
    for receipt in (manifest,cached):
        assert {entry['port']:entry['path'] for entry in receipt['artifacts'] if entry.get('nodeID')=='partition'}==expected_paths
    assert set(cached['cache']['reusedNodeIDs'])=={'cast','partition','fit','test-scaled'}
    private=tmp_path/'bindings.json';private.write_text(json.dumps(bindings));cwd=tmp_path/'unrelated';cwd.mkdir()
    assert replay(output,private,tmp_path/'replayed',cwd)['artifacts']==manifest['artifacts']


@pytest.mark.parametrize('operator',['entity','time'])
def test_test_rows_cannot_fit_statistics(tmp_path,operator):
    graph,_,bindings=fixture(tmp_path,operator)
    next(e for e in graph['edges'] if e['to']['node']=='fit')['from']['port']='test'
    with pytest.raises(ValueError,match='[Tt]raining'):execute(graph,bindings)


@pytest.mark.parametrize('operator',['entity','time'])
def test_resealed_test_membership_forgery_is_rejected(tmp_path,operator):
    _,path,bindings=fixture(tmp_path,operator)
    first=tmp_path/'first';manifest=run_dataflow(path,bindings,first,lambda _:None,cache_policy='materialize')
    entry=next(e for e in manifest['artifacts'] if e.get('nodeID')=='partition' and e.get('port')=='test')
    payload=first/entry['path'];value=json.loads(payload.read_text());lineage=value['lineage']
    lineage['observationIDs'][0]='invented'
    lineage['dataVersion']=digest({'columns':value['columns'],'rows':value['rows'],'ids':lineage['observationIDs']})
    lineage['membershipDigest']=membership_digest(lineage['observationIDs'])
    raw=json.dumps(value).encode();payload.write_bytes(raw);entry.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    manifest['cache']['materializedBytes']=sum(e['bytes'] for e in manifest['artifacts'] if e['kind'] in ('table','state'))
    mp=first/'manifest.json';mp.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='[Pp]artition'):
        run_dataflow(path,bindings,tmp_path/'bad',lambda _:None,cache_policy='reuse',reuse_manifest=mp,reuse_manifest_sha256=hashlib.sha256(mp.read_bytes()).hexdigest())
    assert not (tmp_path/'bad').exists()


def test_three_populations_and_exclusions_artifact_budget_precede_source_read(tmp_path,monkeypatch):
    from runner import dataflow
    config={'strategy':'entity','groupBy':['subject'],'seed':42,'validationPercent':20,'testPercent':20}
    nodes=[{'id':'source','operator':'kwip.data.sourceCSV','config':{}}]
    nodes += [{'id':f'partition{i}','operator':'kwip.data.partition','config':config} for i in range(100)]
    edges=[{'from':{'node':'source','port':'table'},'to':{'node':node['id'],'port':'table'}} for node in nodes[1:]]
    graph=tmp_path/'too-many.json';graph.write_text(json.dumps({'schemaVersion':1,'nodes':nodes,'edges':edges}))
    monkeypatch.setattr(dataflow,'source_csv',lambda *args:pytest.fail('Source read before artifact count preflight'))
    with pytest.raises(ValueError,match='artifact.*count|inventory.*limit'):
        run_dataflow(graph,{'source':'/missing.csv'},tmp_path/'output',lambda _:None)
    assert not (tmp_path/'output').exists()


def test_time_embargo_exclusions_are_exact_exhaustive_and_cache_verified(tmp_path):
    graph,path,bindings=fixture(tmp_path,'time')
    next(node for node in graph['nodes'] if node['id']=='partition')['config']['embargoMilliseconds']=1000
    path.write_text(json.dumps(graph));output=tmp_path/'first'
    manifest=run_dataflow(path,bindings,output,lambda _:None,cache_policy='materialize')
    def artifact(port):
        entry=next(entry for entry in manifest['artifacts'] if entry.get('nodeID')=='partition' and entry.get('port')==port)
        return json.loads((output/entry['path']).read_text())
    outputs={role:artifact(role) for role in ('training','validation','test','exclusions')}
    assert [len(outputs[role]['rows']) for role in outputs]==[72,22,22,4]
    assert [row[1] for row in outputs['exclusions']['rows']]==[36000,36000,48000,48000]
    assert outputs['exclusions']['lineage']['role']=='unassigned'
    ids=[set(table['lineage']['observationIDs']) for table in outputs.values()]
    assert len(set.union(*ids))==120 and all(not left&right for i,left in enumerate(ids) for right in ids[i+1:])
    pin=hashlib.sha256((output/'manifest.json').read_bytes()).hexdigest()
    cached=run_dataflow(path,bindings,tmp_path/'cached',lambda _:None,cache_policy='reuse',reuse_manifest=output/'manifest.json',reuse_manifest_sha256=pin)
    assert cached['artifacts']==manifest['artifacts'] and 'partition' in cached['cache']['reusedNodeIDs']
