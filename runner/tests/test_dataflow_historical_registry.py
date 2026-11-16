"""Read-only historical tables survive additive registry growth, never execute code."""
import copy
import hashlib
import json
from unittest.mock import patch
import pytest
from runner import dataflow
from runner.dataflow_cli import run_dataflow,inspect_dataflow


def test_running_operation_archives_the_loaded_registry_snapshot(tmp_path,monkeypatch):
    from runner import dataflow_cli
    source=tmp_path/'input.csv';source.write_text('x\n1\n')
    graph={'schemaVersion':1,'nodes':[{'id':'source','operator':'kwip.data.sourceCSV','config':{}}],'edges':[]}
    path=tmp_path/'graph.json';path.write_text(json.dumps(graph))
    changed=copy.deepcopy(dataflow.REGISTRY);changed['operators'][0]['title']='Edited during execution'
    original=dataflow_cli.bounded_regular_read
    def edited_catalog(path,limit):
        if path.name=='dataflow-operators.json':return json.dumps(changed).encode()
        return original(path,limit)
    monkeypatch.setattr(dataflow_cli,'bounded_regular_read',edited_catalog)
    output=tmp_path/'prepared';manifest=run_dataflow(path,{'source':str(source)},output,lambda _:None)
    assert json.loads((output/'dataflow-operators.json').read_bytes())==dataflow.REGISTRY
    assert manifest['registryDigest']==dataflow.digest(dataflow.REGISTRY)
    events=[];pin=hashlib.sha256((output/'manifest.json').read_bytes()).hexdigest()
    inspect_dataflow(output/'manifest.json',pin,events.append)
    assert events[-1]['event']=='dataflowInspectionCompleted'


def historical(tmp_path,mutation=lambda registry:None):
    source=tmp_path/'old.csv';source.write_text('x\n001\n002\n')
    graph={'schemaVersion':1,'nodes':[{'id':'source','operator':'kwip.data.sourceCSV','config':{'separator':','}}],'edges':[]}
    path=tmp_path/'graph.json';path.write_text(json.dumps(graph));output=tmp_path/'historical'
    run_dataflow(path,{'source':str(source)},output,lambda _:None)
    definition=copy.deepcopy(dataflow.DEFINITIONS['sourceCSV'])
    definition['version']={'major':1,'minor':0,'patch':0};definition['title']='CSV input'
    # Before Table Input formats/native profile, sourceCSV accepted this schema.
    definition['configSchema']={'type':'object','properties':{'separator':{'type':'string','enum':[',','\t',';','|']}},'required':['separator'],'additionalProperties':False}
    registry={'schemaVersion':1,'packageID':dataflow.REGISTRY['packageID'],'operators':[definition]}
    mutation(registry)
    registry_bytes=json.dumps(registry,sort_keys=True).encode();(output/'dataflow-operators.json').write_bytes(registry_bytes)
    plan=json.loads((output/'preparation-plan.json').read_text());plan['schemaVersion']=1;plan.pop('demand');plan['registryDigest']=dataflow.digest(registry)
    plan_bytes=json.dumps(plan,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode();(output/'preparation-plan.json').write_bytes(plan_bytes)
    manifest=json.loads((output/'manifest.json').read_text());manifest.pop('executionCoverage');manifest['registryDigest']=dataflow.digest(registry);manifest['preparationPlanDigest']=hashlib.sha256(plan_bytes).hexdigest()
    for entry in manifest['artifacts']:
        if entry['path'] in ('dataflow-operators.json','preparation-plan.json'):
            payload=(output/entry['path']).read_bytes();entry.update(bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())
    manifest_path=output/'manifest.json';manifest_path.write_text(json.dumps(manifest))
    source.unlink()
    return manifest_path,hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def test_old_csv_registry_subset_and_schema_remain_readable_without_source_or_code(tmp_path):
    path,pin=historical(tmp_path);events=[]
    with patch('runner.dataflow.execute',side_effect=AssertionError('historical code must not execute')):
        inspect_dataflow(path,pin,events.append)
    preview=next(event for event in events if event['event']=='dataflowPreview')
    assert preview['outputs']['table']['preview']['rows']==[['001'],['002']]
    assert preview['outputs']['table']['preview']['rowCount']==2
    assert events[-1]['provenance']['sourceRevalidated'] is False


@pytest.mark.parametrize('mutation',[
    lambda registry:registry['operators'][0]['version'].update(major=2),
    lambda registry:registry['operators'][0]['outputs'][0].update(kind='state'),
    lambda registry:registry['operators'].append(copy.deepcopy(registry['operators'][0])),
    lambda registry:registry['operators'].clear(),
])
def test_incompatible_or_ambiguous_used_operator_contract_never_emits_outputs(tmp_path,mutation):
    path,pin=historical(tmp_path,mutation);events=[]
    with pytest.raises(ValueError):inspect_dataflow(path,pin,events.append)
    assert [event['event'] for event in events]==['dataflowInspectionStarted']
