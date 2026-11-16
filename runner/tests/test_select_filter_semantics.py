import copy
import csv
import io
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from runner import dataflow
from runner.tests.test_columns import config as columns_config,item
from runner.tests.test_filter_rows import config as filter_config,condition
from runner.dataflow_cli import run_dataflow

BASE=Path(__file__).resolve().parents[1]/'dataflow_examples'

def flow(tmp_path,operator,config,rows,headers=('value','other'),cast=None):
    source=tmp_path/'source.csv';stream=io.StringIO();writer=csv.writer(stream);writer.writerow(headers);writer.writerows(rows);source.write_text(stream.getvalue())
    nodes=[{'id':'source','operator':'kwip.data.sourceCSV','config':{}}];edges=[];previous='source'
    if cast:
        nodes.append({'id':'cast','operator':'kwip.data.columns','config':columns_config([item(name,dtype=cast.get(name,'keep')) for name in headers])});edges.append({'from':{'node':previous,'port':'table'},'to':{'node':'cast','port':'table'}});previous='cast'
    if operator=='filter':
        operator='filterRows';config=filter_config(conditions=[condition(config['column'],config['comparison'],config['value'])])
    nodes.append({'id':'operation','operator':'kwip.data.'+operator,'config':config});edges.append({'from':{'node':previous,'port':'table'},'to':{'node':'operation','port':'table'}})
    return {'schemaVersion':1,'nodes':nodes,'edges':edges},{'source':str(source)},previous

def test_select_reorders_drops_and_simultaneously_swaps_exact_names(tmp_path):
    graph,bindings,_=flow(tmp_path,'selectRename',{'columns':['other',' value '],'rename':{'other':' value ',' value ':'other'}},[['2','a','unused'],['3','b','unused']],headers=(' value ','other','drop'))
    result=dataflow.execute(graph,bindings);before=result['source']['table'];out=result['operation']['table']
    assert [c['name'] for c in out.columns]==[' value ','other']
    assert out.rows==( ('a','2'),('b','3') )
    assert out.row_ids==before.row_ids and out.evidence()==before.evidence()
    assert out.data_version!=before.data_version

@pytest.mark.parametrize('config',[
 {'columns':['value','other'],'rename':{'value':'other'}},
 {'columns':['value'],'rename':{'other':'x'}},
 {'columns':['value'],'rename':{'value':'   '}},
 {'columns':['value','value'],'rename':{}},
])
def test_invalid_selection_rename_rejected_before_loading(tmp_path,config):
    graph,_,_=flow(tmp_path,'selectRename',config,[[1,2]])
    with pytest.raises(ValueError):dataflow.execute(graph,{'source':str(tmp_path/'absent.csv')})

def test_unknown_selection_column_is_explicit_error(tmp_path):
    graph,bindings,_=flow(tmp_path,'selectRename',{'columns':['missing'],'rename':{}},[[1,2]])
    with pytest.raises(ValueError,match='Unknown column'):dataflow.execute(graph,bindings)

@pytest.mark.parametrize('dtype,values,literal,expected',[
 ('int64',[1,2,3],2,{'eq':[1],'ne':[0,2],'lt':[0],'le':[0,1],'gt':[2],'ge':[1,2]}),
 ('float64',[1,2,3],2.0,{'eq':[1],'ne':[0,2],'lt':[0],'le':[0,1],'gt':[2],'ge':[1,2]}),
 ('string',['2','10','3'],'2',{'eq':[0],'ne':[1,2],'lt':[1],'le':[0,1],'gt':[2],'ge':[0,2]}),
 ('bool',['false','true','true'],True,{'eq':[1,2],'ne':[0],'lt':[0],'le':[0,1,2],'gt':[],'ge':[1,2]}),
])
@pytest.mark.parametrize('comparison',['eq','ne','lt','le','gt','ge'])
def test_all_supported_comparisons_preserve_order_and_membership(tmp_path,dtype,values,literal,expected,comparison):
    graph,bindings,previous=flow(tmp_path,'filter',{'column':'value','comparison':comparison,'value':literal},[[v,str(i)] for i,v in enumerate(values)],cast={'value':dtype})
    if dtype=='bool' and comparison not in ('eq','ne'):
        with pytest.raises(ValueError,match='Boolean columns'):dataflow.execute(graph,bindings)
        return
    result=dataflow.execute(graph,bindings);before=result[previous]['table'];out=result['operation']['matching'];selected=expected[comparison]
    assert list(out.rows)==[before.rows[i] for i in selected]
    assert list(out.row_ids)==[before.row_ids[i] for i in selected]
    assert out.source_digest==before.source_digest and out.columns==before.columns

@pytest.mark.parametrize('dtype,literal',[('string',2),('bool',1),('int64',True),('float64','2')])
def test_filter_does_not_coerce_literal_types(tmp_path,dtype,literal):
    value='true' if dtype=='bool' else '2'
    graph,bindings,_=flow(tmp_path,'filter',{'column':'value','comparison':'eq','value':literal},[[value,'x']],cast={'value':dtype})
    with pytest.raises(ValueError,match='literal|value'):dataflow.execute(graph,bindings)

def test_filter_requires_missing_policy_before_comparison(tmp_path):
    graph,bindings,_=flow(tmp_path,'filter',{'column':'value','comparison':'ne','value':'x'},[['','first'],['ok','second']])
    with pytest.raises(ValueError,match='missing|Missing'):dataflow.execute(graph,bindings)

def sensor_graph():return json.loads((BASE/'sensor-preparation.json').read_text())

def test_sensor_example_expected_rows_training_only_fit_and_reference_values():
    result=dataflow.execute(sensor_graph(),{'source':str(BASE/'sensor-readings.csv')})
    assert len(result['source']['table'].rows)==48
    accepted=result['quality-filter']['matching'];usable=result['range-filter']['matching']
    assert len(accepted.rows)==40 and len(usable.rows)==36
    assert len(result['partition']['training'].rows)==22 and len(result['partition']['validation'].rows)==8
    assert list(usable.row_ids)==[rid for row,rid in zip(accepted.rows,accepted.row_ids) if row[0]>=0]
    selected=result['model-columns']['table'];assert [c['name'] for c in selected.columns]==['amount','temperature','label']
    train=result['partition']['training'];held=result['partition']['validation'];state=result['prepare']['state']
    assert set(train.row_ids).isdisjoint(held.row_ids) and state['fitEvidence']['observationIDs']==list(train.row_ids)
    numeric=np.array([row[:2] for row in train.rows]);validation=np.array([row[:2] for row in held.rows])
    np.testing.assert_allclose([r['mean'] for r in state['rules']],numeric.mean(axis=0));np.testing.assert_allclose([r['scale'] for r in state['rules']],numeric.std(axis=0))
    np.testing.assert_allclose(np.array([row[:2] for row in result['validation']['table'].rows]),(validation-numeric.mean(axis=0))/numeric.std(axis=0))

def test_sensor_export_replay_and_select_filter_cache_invalidation(tmp_path):
    graph=sensor_graph();bindings={'source':str(BASE/'sensor-readings.csv')};path=tmp_path/'graph.json';path.write_text(json.dumps(graph));first=tmp_path/'first'
    original=run_dataflow(path,bindings,first,lambda _:None,cache_policy='materialize')
    private=tmp_path/'bindings.json';private.write_text(json.dumps(bindings));replay=tmp_path/'replay'
    p=subprocess.run([sys.executable,str(first/'prep.py'),'--source-bindings',str(private),'--output',str(replay)],capture_output=True,text=True)
    assert p.returncode==0,p.stdout+p.stderr
    assert original['artifacts']==json.loads((replay/'manifest.json').read_text())['artifacts']
    pin=hashlib.sha256((first/'manifest.json').read_bytes()).hexdigest()
    from runner.prepared_data import load_prepared
    tensors,provenance,_=load_prepared(first/'manifest.json','training','table','validation','table',['amount','temperature'],'label',expected_manifest_sha256=pin,expected_graph_digest=original['graphDigest'])
    assert tensors[0].shape==(22,2) and tensors[2].shape==(8,2) and set(tensors[1].tolist())==set(tensors[3].tolist())=={0,1}
    for index,(node,change) in enumerate([('columns',lambda c:c['rows'].reverse()),('columns',lambda c:c['rows'][4].update(outputName='observation')),('range-filter',lambda c:c['conditions'][0].update(value=10))]):
        changed=copy.deepcopy(graph);change(next(n for n in changed['nodes'] if n['id']==node)['config']);path.write_text(json.dumps(changed))
        result=run_dataflow(path,bindings,tmp_path/f'changed-{index}',lambda _:None,cache_policy='reuse',reuse_manifest=first/'manifest.json',reuse_manifest_sha256=pin)
        assert node in result['cache']['executedNodeIDs'] and 'validation' in result['cache']['executedNodeIDs']
        if node=='range-filter':assert 'quality-filter' in result['cache']['reusedNodeIDs']

def test_filter_and_select_preserve_existing_partition_and_fitted_lineage():
    graph=sensor_graph()
    graph['nodes'] += [{'id':'post-filter','operator':'kwip.data.filterRows','config':filter_config(conditions=[condition('amount','ge',0)])}, {'id':'post-select','operator':'kwip.data.selectRename','config':{'columns':['temperature','amount'],'rename':{'amount':'normalized'}}}]
    graph['edges'] += [{'from':{'node':'training','port':'table'},'to':{'node':'post-filter','port':'table'}},{'from':{'node':'post-filter','port':'matching'},'to':{'node':'post-select','port':'table'}}]
    result=dataflow.execute(graph,{'source':str(BASE/'sensor-readings.csv')});before=result['training']['table'];filtered=result['post-filter']['matching'];selected=result['post-select']['table']
    assert list(filtered.row_ids)==[rid for row,rid in zip(before.rows,before.row_ids) if row[0]>=0]
    assert selected.row_ids==filtered.row_ids
    for value in (filtered,selected):
        assert value.role=='training' and value.partition_digest==before.partition_digest and value.fitted_states==before.fitted_states
    assert [c['dtype'] for c in selected.columns]==['float64','float64']
    assert selected.evidence()['membershipDigest']==dataflow.membership_digest(selected.row_ids)

def test_numeric_literal_cross_type_is_explicitly_supported(tmp_path):
    graph,bindings,_=flow(tmp_path,'filter',{'column':'value','comparison':'gt','value':1.5},[[1,'a'],[2,'b']],cast={'value':'int64'})
    assert list(dataflow.execute(graph,bindings)['operation']['matching'].rows)==[(2,'b')]

@pytest.mark.parametrize('value',[float('nan'),float('inf')])
def test_nonfinite_filter_literal_fails_before_source(tmp_path,value):
    graph,_,_=flow(tmp_path,'filter',{'column':'value','comparison':'eq','value':value},[[1,'a']])
    with pytest.raises(ValueError):dataflow.execute(graph,{'source':str(tmp_path/'missing')})
