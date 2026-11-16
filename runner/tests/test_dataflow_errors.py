import json
from pathlib import Path
import subprocess
import sys
import pytest
from runner.dataflow_cli import run_dataflow

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'runner/dataflow_examples'

def case(tmp_path,mode):
    graph=json.loads((BASE/'sensor-preparation.json').read_text());node=next(n for n in graph['nodes'] if n['id']=='quality-filter')
    source=BASE/'sensor-readings.csv'
    if mode=='empty':node['config']['conditions'][0]['value']='no-matching-quality'
    elif mode=='unknown':node['config']['conditions'][0]['column']='unknown-quality'
    elif mode=='type':node['config']['conditions'][0]['value']=3
    elif mode=='missing':
        source=tmp_path/'missing.csv';source.write_text((BASE/'sensor-readings.csv').read_text().replace(',accepted,',',,',1))
        next(n for n in graph['nodes'] if n['id']=='source')['config']['emptyField']='null'
        next(n for n in graph['nodes'] if n['id']=='columns')['config']['missingPolicy']='preserve'
    graphpath=tmp_path/'graph.json';graphpath.write_text(json.dumps(graph))
    return graphpath,{'source':str(source)}

@pytest.mark.parametrize('mode,cause',[('unknown','Unknown column'),('type','literal type'),('missing','null')])
def test_actual_cli_failure_identifies_user_block_and_nonautomatic_repair(tmp_path,mode,cause):
    graph,bindings=case(tmp_path,mode);private=tmp_path/'bindings.json';private.write_text(json.dumps(bindings));labels=tmp_path/'labels.json';labels.write_text(json.dumps({'quality-filter':'Keep accepted measurements'}));output=tmp_path/'out'
    p=subprocess.run([sys.executable,str(ROOT/'runner/main.py'),'--execute-dataflow',str(graph),'--source-bindings',str(private),'--dataflow-node-labels',str(labels),'--output',str(output)],capture_output=True,text=True)
    assert p.returncode==1
    events=[json.loads(line) for line in p.stdout.splitlines()];error=events[-1]
    detail=error['dataflowError']
    assert detail['schemaVersion']==1 and detail['code']=='nodeExecution'
    assert detail['nodeID']=='quality-filter' and detail['nodeLabel']=='Keep accepted measurements'
    assert detail['operatorID']=='kwip.data.filterRows' and detail['componentTitle']=='Filter'
    assert detail['inputRowCount']==48 and detail['cause']
    assert cause.lower() in detail['cause'].lower()
    assert 'configuration' in detail['suggestion'].lower() and 'No settings were changed automatically' in detail['suggestion']
    assert 'Keep accepted measurements' in error['message'] and '48' in error['message']
    assert not any(e['event']=='dataflowCompleted' for e in events)
    assert not (output/'manifest.json').exists() and not list(tmp_path.glob('.dataflow-*'))

def test_healthy_filter_is_not_rewritten_or_diagnosed(tmp_path):
    graph,bindings=case(tmp_path,'healthy');events=[]
    manifest=run_dataflow(graph,bindings,tmp_path/'out',events.append,node_labels={'quality-filter':'My quality rule'})
    assert manifest['status']=='completed' and events[-1]['event']=='dataflowCompleted'
    assert not any('dataflowError' in event for event in events)
    assert next(event for event in events if event.get('nodeID')=='quality-filter' and event.get('event')=='dataflowPreview')['outputs']['matching']['preview']['rowCount']==40

def test_cancellation_at_node_boundary_is_not_operator_failure(tmp_path):
    graph,bindings=case(tmp_path,'healthy');events=[]
    def cancel(event):
        events.append(event)
        if event['event']=='dataflowPreview':raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):run_dataflow(graph,bindings,tmp_path/'out',cancel,node_labels={})
    assert not (tmp_path/'out').exists() and not any('dataflowError' in e for e in events)

@pytest.mark.parametrize('labels',[{'unknown':'label'},{'source':''},{'source':'x'*201},{'source':3},['source']])
def test_label_map_is_bounded_and_rejects_unknown_nodes_before_artifacts(tmp_path,labels):
    graph,bindings=case(tmp_path,'healthy')
    with pytest.raises(ValueError,match='node labels'):run_dataflow(graph,bindings,tmp_path/'out',lambda _:None,node_labels=labels)
    assert not (tmp_path/'out').exists()

def test_label_sidecar_rejects_symlink_oversize_and_invalid_json_without_paths(tmp_path):
    from runner.dataflow_cli import read_node_labels
    target=tmp_path/'private-labels.json';target.write_text('{}');link=tmp_path/'link';link.symlink_to(target)
    for path in (link,tmp_path/'absent'):
        with pytest.raises(ValueError) as error:read_node_labels(path)
        assert str(tmp_path) not in str(error.value)
    target.write_bytes(b' '*65537)
    with pytest.raises(ValueError,match='64 KB'):read_node_labels(target)
    target.write_text('{broken')
    with pytest.raises(ValueError,match='valid JSON'):read_node_labels(target)

def test_labels_preserve_scientific_cache_identity_and_change_only_derived_presentation(tmp_path):
    import hashlib
    graph,bindings=case(tmp_path,'healthy');first=tmp_path/'first'
    a=run_dataflow(graph,bindings,first,lambda _:None,cache_policy='materialize',node_labels={'quality-filter':'Private Label A'})
    pin=hashlib.sha256((first/'manifest.json').read_bytes()).hexdigest()
    b=run_dataflow(graph,bindings,tmp_path/'next',lambda _:None,cache_policy='reuse',reuse_manifest=first/'manifest.json',reuse_manifest_sha256=pin,node_labels={'quality-filter':'Private Label B'})
    assert b['cache']['executedNodeIDs']==['source'] and len(b['cache']['reusedNodeIDs'])==9
    scientific=lambda manifest:[entry for entry in manifest['artifacts'] if entry['kind'] in ('table','state','graph','registry','previews')]
    assert scientific(a)==scientific(b)
    assert a['graphDigest']==b['graphDigest']
    assert a['preparationPlanDigest']!=b['preparationPlanDigest']
    assert json.loads((first/'preparation-plan.json').read_text())['nodeLabels']=={'quality-filter':'Private Label A'}
    assert 'Private Label A' in (first/'preparation.py').read_text()
    assert str(next(iter(bindings.values()))) not in (first/'preparation.py').read_text()+(first/'preparation-plan.json').read_text()
    assert 'Private Label' not in (first/'dataflow.json').read_text()+(first/'manifest.json').read_text()

def test_unlabelled_generic_node_failure_has_component_and_identity(tmp_path):
    graph,bindings=case(tmp_path,'healthy');value=json.loads(graph.read_text());next(n for n in value['nodes'] if n['id']=='columns')['config']['rows'][0]['source']='unknown';graph.write_text(json.dumps(value))
    with pytest.raises(ValueError) as error:run_dataflow(graph,bindings,tmp_path/'out',lambda _:None)
    detail=error.value.detail
    assert detail['code']=='nodeExecution' and detail['nodeID']=='columns' and detail['nodeLabel'] is None
    assert detail['inputRowCount']==48 and detail['suggestion']


def test_empty_current_filter_branch_is_valid_and_keeps_the_nonmatching_population(tmp_path):
    graph,bindings=case(tmp_path,'empty');value=json.loads(graph.read_text())
    value['nodes']=value['nodes'][:3]
    value['edges']=[edge for edge in value['edges'] if edge['to']['node'] in {'source','columns','quality-filter'}]
    graph.write_text(json.dumps(value));events=[]
    manifest=run_dataflow(graph,bindings,tmp_path/'out',events.append)
    preview=next(event for event in events if event.get('nodeID')=='quality-filter' and event['event']=='dataflowPreview')
    assert preview['outputs']['matching']['preview']['rowCount']==0
    assert preview['outputs']['nonmatching']['preview']['rowCount']==48
    assert manifest['status']=='completed' and events[-1]['event']=='dataflowCompleted'
    assert not any('dataflowError' in event for event in events)
