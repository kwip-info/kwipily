"""Demand is explicit: complete consumers cannot inherit sampled ancestors."""
import math
import pytest
from runner.preparation_demand import normalize_sampling,build_demand_plan,sample_table,coverage_report
from runner import dataflow
from runner.tests.test_columns import config,item


def graph():
    def node(i,op,c):return {'id':i,'operator':'kwip.data.'+op,'config':c}
    def edge(a,b):return {'from':{'node':a,'port':'table'},'to':{'node':b,'port':'table'}}
    return {'schemaVersion':1,'nodes':[node('s','sourceCSV',{'separator':','}),
        node('full','columns',config([item('x')],newColumns='drop')),
        node('sample','columns',config([item('x')],newColumns='drop'))],
        'edges':[edge('s','full'),edge('s','sample')]}


def test_count_and_percentage_have_unambiguous_bounded_configuration():
    assert normalize_sampling(None)=={'schemaVersion':1,'mode':'all'}
    assert normalize_sampling({'schemaVersion':1,'mode':'rowCount','rowCount':200})['rowCount']==200
    assert normalize_sampling({'schemaVersion':1,'mode':'percentage','percentage':12.5})['percentage']==12.5
    for bad in ({'mode':'rowCount','rowCount':0},{'mode':'rowCount','rowCount':True},
                {'mode':'rowCount','rowCount':5_000_001},{'mode':'percentage','percentage':0},
                {'mode':'percentage','percentage':float('nan')},{'mode':'percentage','percentage':101},
                {'mode':'all','rowCount':1},{'mode':'random'},{'schemaVersion':True,'mode':'all'}):
        with pytest.raises(ValueError):normalize_sampling(bad)


def test_ancestor_full_demand_keeps_unrelated_branch_sampled():
    plan=build_demand_plan(graph(),{'mode':'rowCount','rowCount':10},complete_nodes=['full'])
    assert plan.complete_node_ids==('s','full')
    assert plan.sampled_node_ids==('sample',)
    assert plan.policy_for('s')=={'schemaVersion':1,'mode':'all'}
    assert plan.policy_for('sample')['rowCount']==10
    assert plan.to_document()['selection']=='sourceOrderPrefix'
    assert plan.to_document()['percentageRequiresCardinalityScan'] is False
    with pytest.raises(ValueError):plan.policy_for('unknown')
    with pytest.raises(ValueError):build_demand_plan(graph(),complete_nodes=['unknown'])


def test_default_all_and_100_percent_are_complete_without_count_scan():
    for policy in (None,{'mode':'percentage','percentage':100}):
        plan=build_demand_plan(graph(),policy)
        assert plan.sampled_node_ids==()
        assert plan.complete_node_ids==('s','full','sample')
        assert plan.to_document()['percentageRequiresCardinalityScan'] is False


def test_percentage_demand_discloses_required_count_scan():
    plan=build_demand_plan(graph(),{'mode':'percentage','percentage':20})
    assert plan.to_document()['percentageRequiresCardinalityScan'] is True
    assert plan.sampled_node_ids==('s','full','sample')


def test_table_prefix_retains_original_observation_ids_and_lineage():
    table=dataflow.derive(dataflow.Table(({'name':'x','dtype':'int64','nullable':False},),
          tuple((i,) for i in range(10)),tuple('source:'+str(i) for i in range(10)),
          'source',role='training',partition_digest='split',fitted_states=('state',)))
    sampled=sample_table(table,{'mode':'rowCount','rowCount':3})
    assert list(sampled.rows)==[(0,),(1,),(2,)]
    assert list(sampled.row_ids)==['source:0','source:1','source:2']
    assert sampled.role=='training' and sampled.partition_digest=='split' and sampled.fitted_states==('state',)
    assert sampled.source_digest==table.source_digest and sampled.data_version!=table.data_version
    percentage=sample_table(table,{'mode':'percentage','percentage':25})
    assert len(percentage.rows)==math.ceil(10*.25)
    assert sample_table(table,{'mode':'all'}) is table


def test_coverage_success_is_not_full_or_failed_partial():
    plan=build_demand_plan(graph(),{'mode':'rowCount','rowCount':10},complete_nodes=['full'])
    report=coverage_report(plan,{'s':{'table':100},'full':{'table':100},'sample':{'table':10}})
    assert report['schemaVersion']==1
    assert report['status']=='sampled'
    assert report['nodes']['s']['completeness']=='complete'
    assert report['nodes']['sample']['completeness']=='sampled'
    assert report['nodes']['sample']['rowCounts']=={'table':10}
    assert report['completeNodeIDs']==['s','full']
    assert report['sampledNodeIDs']==['sample']
    with pytest.raises(ValueError):coverage_report(plan,{'s':{'table':100}})


def test_run_captures_sample_policy_coverage_and_exact_ids(tmp_path):
    import json,hashlib
    from runner.dataflow_cli import run_dataflow
    source=tmp_path/'source.csv';payload=b'x\n'+b'v\n'*20;source.write_bytes(payload)
    document=graph();document['nodes'][0]['config']['parserProfile']='polars'
    path=tmp_path/'graph.json';path.write_text(json.dumps(document))
    manifest=run_dataflow(path,{'s':str(source)},tmp_path/'sample',lambda event:None,
                          sampling={'mode':'rowCount','rowCount':5})
    assert manifest['executionCoverage']['status']=='sampled'
    assert all(record['rowCounts']=={'table':5} for record in manifest['executionCoverage']['nodes'].values())
    artifact=next(entry for entry in manifest['artifacts'] if entry.get('nodeID')=='s')
    captured=json.loads((tmp_path/'sample'/artifact['path']).read_text())
    assert captured['lineage']['observationIDs']==[hashlib.sha256(payload).hexdigest()+':'+str(i) for i in range(5)]
    plan=json.loads((tmp_path/'sample'/'preparation-plan.json').read_text())
    assert plan['schemaVersion']==2 and plan['demand']['sampling']['rowCount']==5


def test_complete_shared_source_and_sampled_branch_do_not_repeat_percentage(tmp_path):
    import json
    from runner.dataflow_cli import run_dataflow
    source=tmp_path/'source.csv';source.write_text('x\n'+'v\n'*20)
    document=graph();document['nodes'][0]['config']['parserProfile']='polars'
    document['nodes'].append({'id':'later','operator':'kwip.data.columns','config':config([item('x')],newColumns='drop')})
    document['edges'].append({'from':{'node':'sample','port':'table'},'to':{'node':'later','port':'table'}})
    path=tmp_path/'graph.json';path.write_text(json.dumps(document))
    manifest=run_dataflow(path,{'s':str(source)},tmp_path/'mixed',lambda event:None,
                          sampling={'mode':'percentage','percentage':25},complete_nodes=['full'])
    rows={key:value['rowCounts']['table'] for key,value in manifest['executionCoverage']['nodes'].items()}
    assert rows=={'s':20,'full':20,'sample':5,'later':5}


def test_invalid_sampling_refuses_before_artifact_creation(tmp_path):
    import json
    from runner.dataflow_cli import run_dataflow
    path=tmp_path/'graph.json';path.write_text(json.dumps(graph()))
    output=tmp_path/'refused'
    with pytest.raises(ValueError,match='row count'):
        run_dataflow(path,{'s':'/unopened.csv'},output,lambda event:None,
                     sampling={'mode':'rowCount','rowCount':0},cache_policy='materialize')
    assert not output.exists()


def test_coverage_cannot_relabel_sample_without_matching_pinned_demand():
    import copy
    from runner.preparation_demand import validate_coverage
    plan=build_demand_plan(graph(),{'mode':'rowCount','rowCount':5})
    report=coverage_report(plan,{key:{'table':5} for key in ('s','full','sample')})
    assert validate_coverage(report,graph(),demand_plan=plan)==report
    forged=copy.deepcopy(report);forged['nodes']['s']['completeness']='complete'
    with pytest.raises(ValueError):validate_coverage(forged,graph(),demand_plan=plan)
    forged=coverage_report(build_demand_plan(graph()),{key:{'table':5} for key in ('s','full','sample')})
    with pytest.raises(ValueError):validate_coverage(forged,graph(),demand_plan=plan)


def test_view_forces_full_ancestors_and_builds_pinned_disk_index(tmp_path):
    import json
    from runner.dataflow_cli import run_dataflow
    from runner.table_view import query_index
    document=graph();document['nodes'][0]['config']['parserProfile']='polars'
    document['nodes'].append({'id':'view','operator':'kwip.data.view','config':{}})
    document['edges'].append({'from':{'node':'full','port':'table'},'to':{'node':'view','port':'table'}})
    source=tmp_path/'source.csv';source.write_text('x\n'+'v\n'*100_001)
    path=tmp_path/'graph.json';path.write_text(json.dumps(document));output=tmp_path/'view-run'
    manifest=run_dataflow(path,{'s':str(source)},output,lambda _:None,sampling={'mode':'rowCount','rowCount':7})
    rows={key:value['rowCounts']['table'] for key,value in manifest['executionCoverage']['nodes'].items()}
    assert rows=={'s':100_001,'full':100_001,'sample':7,'view':100_001}
    receipt=manifest['tableViews']['view'];index=receipt['index']
    assert index['kind']=='tableViewIndex' and index['bytes']<=50_000_000
    table=next(entry for entry in manifest['artifacts'] if entry.get('nodeID')=='view' and entry['kind']=='table')
    assert index['sourceSHA256']==table['sha256']
    page=query_index(output/index['path'],index['sha256'],offset=99_999,limit=2,source_sha256=table['sha256'])
    assert page['rows']==[['v'],['v']] and page['positions']==[99_999,100_000]
    assert page['totalRows']==100_001 and page['nextOffset'] is None
    assert sum(entry['bytes'] for entry in manifest['artifacts'])<=200_000_000


def test_sampling_cli_configuration_imports_no_data_engines():
    import subprocess,sys
    result=subprocess.run([sys.executable,'-c',
        "import sys; from runner.preparation_demand_configuration import parse_sampling_json,parse_complete_nodes_json; "
        "assert parse_sampling_json('{\"mode\":\"rowCount\",\"rowCount\":5}')['rowCount']==5; "
        "assert parse_complete_nodes_json('[\"source\"]')==['source']; "
        "assert not {'polars','numpy','torch'}&set(sys.modules)"],capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stderr


# Reuse the real retained-operation fixture; demand cannot loosen its file leases.
from runner.tests.test_operation_authorization import operation_authority,publish,claim
from runner.tests.test_preparation_operation_authorization import preparation_operation


def test_owned_preparation_admits_bounded_demand_options(preparation_operation):
    import json
    fixture=preparation_operation
    fixture[1]['arguments'] += ['--dataflow-sampling',json.dumps({'mode':'rowCount','rowCount':8}),
                                '--dataflow-complete-nodes','["training"]']
    publish(fixture)
    with claim(fixture) as authority:authority.check()


@pytest.mark.parametrize('flag,value',[
    ('--dataflow-sampling','{"mode":"rowCount","rowCount":0}'),
    ('--dataflow-sampling','{"mode":"percentage","percentage":NaN}'),
    ('--dataflow-sampling','{"mode":"all","mode":"rowCount","rowCount":5}'),
    ('--dataflow-complete-nodes','["unknown"]'),
    ('--dataflow-complete-nodes','["training","training"]'),
    ('--dataflow-complete-nodes','[true]'),
])
def test_owned_preparation_refuses_invalid_or_unknown_demand(preparation_operation,flag,value):
    fixture=preparation_operation;fixture[1]['arguments'] += [flag,value];publish(fixture)
    with pytest.raises(ValueError):claim(fixture)


def test_successful_sample_history_is_inspectable_with_its_incomplete_coverage(tmp_path):
    import json,hashlib
    from runner.dataflow_cli import run_dataflow,inspect_dataflow
    document=graph();document['nodes'][0]['config']['parserProfile']='polars'
    path=tmp_path/'graph.json';path.write_text(json.dumps(document))
    source=tmp_path/'source.csv';source.write_text('x\n'+'v\n'*20);output=tmp_path/'sample'
    run_dataflow(path,{'s':str(source)},output,lambda _:None,sampling={'mode':'rowCount','rowCount':3})
    source.unlink();manifest=output/'manifest.json';events=[]
    inspect_dataflow(manifest,hashlib.sha256(manifest.read_bytes()).hexdigest(),events.append)
    assert events[-1]['executionCoverage']['status']=='sampled'
    assert all(event['coverage']['completeness']=='sampled' for event in events if event['event']=='dataflowPreview')
    assert events[-1]['provenance']['sourceRevalidated'] is False
