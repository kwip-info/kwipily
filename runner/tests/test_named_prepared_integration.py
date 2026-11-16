import hashlib
import json
from pathlib import Path
import pytest
import torch
from runner import dataflow as df
from runner.dataflow_cli import run_dataflow
from runner.text_encoding import DEFAULTS
from runner.named_prepared_data import load_named_prepared
from runner.engine import run
from runner.tests.test_embedding_graph import graph as embedding_graph
from runner.tests.test_named_inputs import named_graph


def fixture(tmp_path,segmented=False):
    source=tmp_path/'texts.csv'
    source.write_text('text,label\n'+''.join(('ababa' if i%2==0 else 'babab')+','+str(i%2)+'\n' for i in range(40)))
    def default(op):return next(x['defaultConfig'] for x in df.REGISTRY['operators'] if x['id']=='kwip.data.'+op)
    columns={**default('columns'),'rows':[{'source':'text','included':True,'outputName':'text','targetType':'keep'},{'source':'label','included':True,'outputName':'label','targetType':'int64'}]}
    graph={'schemaVersion':1,'nodes':[{'id':'source','operator':'kwip.data.sourceCSV','config':default('sourceCSV')},{'id':'columns','operator':'kwip.data.columns','config':columns},{'id':'partition','operator':'kwip.data.partition','config':default('partition')},{'id':'text','operator':'kwip.data.textEncode','config':DEFAULTS},{'id':'view','operator':'kwip.data.view','config':{}}],'edges':[]}
    for a,ap,b,bp in [('source','table','columns','table'),('columns','table','partition','table'),('partition','training','text','training'),('partition','validation','text','validation'),('text','validation','view','table')]:graph['edges'].append({'from':{'node':a,'port':ap},'to':{'node':b,'port':bp}})
    path=tmp_path/'dataflow.json';path.write_text(json.dumps(graph));out=tmp_path/'prepared'
    from runner.workload_policy import LEGACY
    events=[];run_dataflow(path,{'source':str(source)},out,events.append,engine='polars',workload_policy=LEGACY.to_document() if segmented else None)
    return out,graph,events


@pytest.mark.parametrize("segmented",[False,True])
def test_real_polars_text_preparation_named_integer_training_and_portable_replay(tmp_path,segmented):
    out,dataflow,events=fixture(tmp_path,segmented)
    manifest=out/'manifest.json';pin=hashlib.sha256(manifest.read_bytes()).hexdigest()
    model=embedding_graph();model['nodes'][0]['config']['shape']=[5];model['nodes'][3]['config']['in_features']=15
    model['namedInputs']={'tokens':'source'}
    plan={'schemaVersion':1,'inputs':[{'name':'tokens','nodeID':'source','representation':'tokens','dtype':'int64','sampleShape':[5],'columns':['tokens'],'lengthPolicy':'error','paddingValue':1,'vocabularySize':4}],'roles':[{'name':'target','representation':'numeric','dtype':'int64','sampleShape':[],'columns':['label']}],'maximumBatchBytes':1048576}
    prepared=load_named_prepared(manifest,'text','training','text','validation',plan,model,expected_manifest_sha256=pin,expected_graph_digest=df.digest(dataflow))
    assert prepared[0][0]['tokens'].dtype==torch.int64
    assert len(prepared[0][0]['tokens'])+len(prepared[0][2]['tokens'])==32
    result=run(model,tmp_path/'training',epochs=2,device='cpu',batch_size=4,prepared=prepared)
    assert result['evaluationCount']==8
    assert (tmp_path/'training'/'checkpoint.pt').is_file()
    assert (tmp_path/'training'/'preparation'/'manifest.json').is_file()
    assert (out/'text_encoding.py').is_file()
    from runner.table_view import query_index
    index=next(x for x in json.loads(manifest.read_bytes())['artifacts'] if x['kind']=='tableViewIndex')
    page=query_index(out/index['path'],index['sha256'],offset=0,limit=10,search='',source_sha256=index['sourceSHA256'])
    assert page['columns'][-2]['dtype']=='int64Sequence'
    assert len(page['rows'])==8


def test_named_frozen_validation_test_and_exact_cpu_continuation(tmp_path):
    from runner.frozen_checkpoint import load_frozen
    from runner.engine import REGISTRY
    from runner.prepared_data import load_evaluation_partition
    from runner.final_evaluation import evaluate_classifier
    out,dataflow,_=fixture(tmp_path)
    manifest=out/'manifest.json';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();pin=sha(manifest)
    model=embedding_graph();model['nodes'][0]['config']['shape']=[5];model['nodes'][3]['config']['in_features']=15;model['namedInputs']={'tokens':'source'}
    plan={'schemaVersion':1,'inputs':[{'name':'tokens','nodeID':'source','representation':'tokens','dtype':'int64','sampleShape':[5],'columns':['tokens'],'lengthPolicy':'error','paddingValue':1,'vocabularySize':4}],'roles':[{'name':'target','representation':'numeric','dtype':'int64','sampleShape':[],'columns':['label']}],'maximumBatchBytes':1048576}
    prepared=load_named_prepared(manifest,'text','training','text','validation',plan,model,expected_manifest_sha256=pin,expected_graph_digest=df.digest(dataflow))
    first=tmp_path/'first';full=tmp_path/'full';resumed=tmp_path/'resumed'
    run(model,first,epochs=1,device='cpu',batch_size=4,prepared=prepared)
    run(model,full,epochs=2,device='cpu',batch_size=4,prepared=prepared)
    run(model,resumed,epochs=2,device='cpu',batch_size=4,prepared=prepared,resume_checkpoint=first/'checkpoint.pt')
    a=torch.load(full/'checkpoint.pt',weights_only=True);b=torch.load(resumed/'checkpoint.pt',weights_only=True)
    assert a['contract']==b['contract']
    for name,value in a['modelState'].items():assert torch.equal(value,b['modelState'][name])
    assert json.loads((full/'metrics.json').read_bytes())==json.loads((resumed/'metrics.json').read_bytes())
    frozen=load_frozen(full/'checkpoint.pt',expected_manifest_sha256=sha(full/'manifest.json'),expected_checkpoint_sha256=sha(full/'checkpoint.pt'),registry_path=REGISTRY,expected_registry_sha256=sha(REGISTRY))
    assert frozen.input_contract=={'namedInputPlan':plan}
    for role,node,port in [('validation','text','validation'),('test','partition','test')]:
        (x,y),provenance,_=load_evaluation_partition(manifest,node,port,frozen.training_provenance,partition_role=role,expected_manifest_sha256=pin,named_plan=plan,model_graph=model)
        result=evaluate_classifier(frozen.model,x,y,provenance[role+'RowIDs'],batch_size=3,device='cpu')
        assert result['evaluatedCount']==8
        assert x['tokens'].dtype==torch.int64
    with pytest.raises(ValueError):load_evaluation_partition(manifest,'text','training',frozen.training_provenance,partition_role='validation',expected_manifest_sha256=pin,named_plan=plan,model_graph=model)
    import copy
    changed=copy.deepcopy(prepared);changed[1]['namedInputPlan']['maximumBatchBytes']=2048576
    with pytest.raises(ValueError):run(model,tmp_path/'changed',epochs=2,device='cpu',batch_size=4,prepared=changed,resume_checkpoint=first/'checkpoint.pt')
