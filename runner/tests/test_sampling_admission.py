"""Samples remain explorable evidence, never implicit complete training/cache input."""
import copy
import hashlib
import json
from pathlib import Path
import pytest
from runner import dataflow
from runner.dataflow_cache import Materialization
from runner.dataflow_cli import run_dataflow
from runner.dataflow_storage import require_complete_coverage
from runner.preparation_demand import build_demand_plan, coverage_report

BASE = Path(__file__).resolve().parents[1] / 'dataflow_examples'


def coverage(graph, sampling=None, complete_nodes=None):
    plan = build_demand_plan(graph, sampling, complete_nodes)
    return coverage_report(plan, {node['id']: {port['id']:1 for port in dataflow.DEFINITIONS[node['operator'].removeprefix(dataflow.PREFIX)]['outputs'] if port['kind'] in ('table','dataset')} for node in graph['nodes']})


def test_global_complete_admission_refuses_sample_even_when_manifest_completed():
    # Dedicated sampleable graph: the evolving practice includes View sinks,
    # which correctly promote their entire upstream branch to complete.
    graph = {'schemaVersion': 1, 'nodes': [
        {'id': 'source', 'operator': 'kwip.data.sourceCSV', 'config': {}}], 'edges': []}
    manifest = {'status': 'completed', 'graphDigest': dataflow.digest(graph),
                'executionCoverage': coverage(graph, {'mode': 'rowCount', 'rowCount': 4})}
    with pytest.raises(ValueError, match='sample|complete'):
        require_complete_coverage(manifest)
    require_complete_coverage({'status': 'completed'})  # Existing full artifacts predate coverage.


def test_selected_complete_branch_can_be_admitted_but_sampled_branch_cannot():
    graph = {'schemaVersion': 1, 'nodes': [
        {'id': 'source', 'operator': 'kwip.data.sourceCSV', 'config': {}},
        {'id': 'full', 'operator': 'kwip.data.selectRename', 'config': {'columns': ['x'], 'rename': {}}},
        {'id': 'sample', 'operator': 'kwip.data.selectRename', 'config': {'columns': ['x'], 'rename': {}}}],
        'edges': [{'from': {'node': 'source', 'port': 'table'}, 'to': {'node': node, 'port': 'table'}} for node in ('full', 'sample')]}
    manifest = {'status': 'completed', 'graphDigest': dataflow.digest(graph),
                'executionCoverage': coverage(graph, {'mode': 'rowCount', 'rowCount': 3}, ['full'])}
    plan={'schemaVersion':2,'graph':graph,'graphDigest':dataflow.digest(graph),'demand':build_demand_plan(graph,{'mode':'rowCount','rowCount':3},['full']).to_document()}
    plan_raw=json.dumps(plan,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
    manifest['preparationPlanDigest']=hashlib.sha256(plan_raw).hexdigest()
    files = {'dataflow.json': json.dumps(graph).encode(),'preparation-plan.json':plan_raw}
    require_complete_coverage(manifest, files, nodes=['full'])
    with pytest.raises(ValueError, match='sample|complete'):
        require_complete_coverage(manifest, files, nodes=['sample'])
    malformed = copy.deepcopy(manifest)
    malformed['executionCoverage']['nodes']['source']['completeness'] = 'sampled'
    with pytest.raises(ValueError):
        require_complete_coverage(malformed, files, nodes=['full'])


def test_cache_key_pins_effective_sample_policy_even_when_table_values_identical():
    from runner.preparation_demand import build_demand_plan
    graph = {'schemaVersion': 1, 'nodes': [{'id': 's', 'operator': 'kwip.data.sourceCSV', 'config': {}}], 'edges': []}
    cache = Materialization('materialize')
    complete = cache.key('s', 'sourceCSV', {}, {}, {}, dataflow.digest(graph))
    cache.set_demand(build_demand_plan(graph, {'mode': 'rowCount', 'rowCount': 10}))
    sampled = cache.key('s', 'sourceCSV', {}, {}, {}, dataflow.digest(graph))
    assert sampled != complete
    cache.set_demand(build_demand_plan(graph, {'mode': 'rowCount', 'rowCount': 20}))
    assert cache.key('s', 'sourceCSV', {}, {}, {}, dataflow.digest(graph)) != sampled


def sampled_fixture(root):
    from runner.preparation_plan import build_plan
    graph={'schemaVersion':1,'nodes':[
        {'id':'source','operator':'kwip.data.sourceCSV','config':{}},
        {'id':'training','operator':'kwip.data.selectRename','config':{'columns':['x','z','y'],'rename':{}}},
        {'id':'validation','operator':'kwip.data.selectRename','config':{'columns':['x','z','y'],'rename':{}}}],
        'edges':[{'from':{'node':'source','port':'table'},'to':{'node':node,'port':'table'}} for node in ('training','validation')]}
    plan=build_plan(graph,engine='disk',engine_version='1',sampling={'mode':'rowCount','rowCount':3})
    root.mkdir()
    artifacts=[]
    for name,value,kind in [('dataflow.json',graph,'graph'),('preparation-plan.json',plan.to_document(),'preparationPlan'),('rows.json',{},'table')]:
        raw=json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
        (root/name).write_bytes(raw)
        artifacts.append({'path':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'kind':kind})
    manifest={'schemaVersion':1,'kind':'dataflowExecution','status':'completed','graphDigest':dataflow.digest(graph),
              'artifacts':artifacts,'preparationPlanDigest':plan.plan_digest,
              'executionCoverage':coverage(graph,{'mode':'rowCount','rowCount':3})}
    return graph,manifest


@pytest.mark.parametrize('tamper',['none','status','promote','remove'])
def test_actual_training_load_refuses_sample_before_tensor_conversion(tmp_path,monkeypatch,tamper):
    from runner.prepared_data import load_prepared
    root=tmp_path/'prepared'
    graph,manifest=sampled_fixture(root)
    if tamper=='status':
        manifest['executionCoverage']['status']='complete'
    elif tamper=='promote':
        manifest['executionCoverage']=coverage(graph)
    elif tamper=='remove':
        del manifest['executionCoverage']
    path=root/'manifest.json'
    raw=json.dumps(manifest).encode();path.write_bytes(raw)
    monkeypatch.setattr('torch.tensor',lambda *a,**k: (_ for _ in ()).throw(AssertionError('Tensor allocation before sample refusal')))
    with pytest.raises(ValueError,match='sample|complete|coverage|demand'):
        load_prepared(path,'training','table','validation','table',['x','z'],'y',
                      expected_manifest_sha256=hashlib.sha256(raw).hexdigest(),expected_graph_digest=dataflow.digest(graph))


def test_prior_sample_coverage_cannot_hit_full_cache_even_with_relabelled_cache_key():
    graph={'schemaVersion':1,'nodes':[
        {'id':'source','operator':'kwip.data.sourceCSV','config':{}},
        {'id':'output','operator':'kwip.data.selectRename','config':{'columns':['x'],'rename':{}}}],
        'edges':[{'from':{'node':'source','port':'table'},'to':{'node':'output','port':'table'}}]}
    cache=Materialization('materialize')
    cache.policy='reuse'
    full_key=cache.key('output','selectRename',{'columns':['x'],'rename':{}},{},{},dataflow.digest(graph))
    cache.prior={'output':{'cacheKey':full_key}}
    cache.prior_coverage=coverage(graph,{'mode':'rowCount','rowCount':3})
    assert cache.lookup('output',full_key,'selectRename',{},dataflow.digest(graph)) is None


def test_real_cache_does_not_reclassify_identical_sample_rows_as_full(tmp_path):
    graph={'schemaVersion':1,'nodes':[
        {'id':'source','operator':'kwip.data.sourceCSV','config':{}},
        {'id':'selected','operator':'kwip.data.selectRename','config':{'columns':['x'],'rename':{}}}],
        'edges':[{'from':{'node':'source','port':'table'},'to':{'node':'selected','port':'table'}}]}
    graph_path=tmp_path/'graph.json';graph_path.write_text(json.dumps(graph))
    source=tmp_path/'source.csv';source.write_text('x\n1\n2\n3\n4\n')
    policy={'schemaVersion':1,'mode':'rowCount','rowCount':100}
    def run(name, sampling=None, prior=None):
        options={'sampling':sampling,'cache_policy':'reuse' if prior else 'materialize'}
        if prior:
            path=tmp_path/prior/'manifest.json'
            options.update(reuse_manifest=path,reuse_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        return run_dataflow(graph_path,{'source':str(source)},tmp_path/name,lambda event:None,**options)
    sampled=run('sample',policy)
    full=run('full',prior='sample')
    assert sampled['executionCoverage']['status']=='sampled'
    assert full['executionCoverage']['status']=='complete'
    assert full['cache']['reusedNodeIDs']==[]
    changed=run('sample-from-full',policy,prior='full')
    assert changed['cache']['reusedNodeIDs']==[]
    same=run('same-sample',policy,prior='sample')
    assert same['cache']['reusedNodeIDs']==['selected']


def test_optional_reader_runtime_version_changes_invalidate_retained_cache(monkeypatch):
    import importlib.metadata
    from runner.dataflow_cache import implementation_digest
    original=importlib.metadata.version
    initial=implementation_digest()
    def changed(name):
        return '99.0.0-test' if name=='openpyxl' else original(name)
    monkeypatch.setattr(importlib.metadata,'version',changed)
    assert implementation_digest()!=initial
    def absent(name):
        if name=='openpyxl':raise importlib.metadata.PackageNotFoundError(name)
        return original(name)
    monkeypatch.setattr(importlib.metadata,'version',absent)
    assert implementation_digest()!=initial
