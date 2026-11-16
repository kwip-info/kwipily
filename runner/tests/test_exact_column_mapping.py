"""Native column transport must preserve whitespace and delimiter characters exactly."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import pytest
from runner.dataflow_cli import run_dataflow
from runner.tests.test_columns import config,item
from runner.tests.test_runner import graph as model_graph

ROOT=Path(__file__).resolve().parents[2]
FEATURES=[' x','y,unit']

def fixture(tmp_path):
    source=tmp_path/'source.csv'
    with source.open('w') as stream:
        writer=csv.writer(stream);writer.writerow([' x','x','y,unit','label'])
        writer.writerows([[i+1,1000+i,i*i+1,i%2] for i in range(40)])
    model=tmp_path/'model.json';model.write_text(json.dumps(model_graph()))
    return source,model

def invoke(model,out,*flags):
    return subprocess.run([sys.executable,str(ROOT/'runner/main.py'),'--graph',str(model),'--output',str(out),'--epochs','1','--device','cpu',*flags],capture_output=True,text=True)

@pytest.mark.parametrize('mode',['csv','prepared'])
def test_json_columns_train_exact_selected_headers(tmp_path,mode):
    source,model=fixture(tmp_path)
    flags=['--dataset-csv',str(source)]
    if mode=='prepared':
        graph={'schemaVersion':1,'nodes':[{'id':'s','operator':'kwip.data.sourceCSV','config':{}},{'id':'c','operator':'kwip.data.columns','config':config([item(name,dtype='int64' if name=='label' else 'float64') for name in [' x','x','y,unit','label']])},{'id':'split','operator':'kwip.data.partition','config':{'strategy':'random','seed':42,'validationPercent':20,'testPercent':20,'stratifyColumn':'label'}}],'edges':[{'from':{'node':'s','port':'table'},'to':{'node':'c','port':'table'}},{'from':{'node':'c','port':'table'},'to':{'node':'split','port':'table'}}]}
        request=tmp_path/'dataflow.json';request.write_text(json.dumps(graph))
        run_dataflow(request,{'s':str(source)},tmp_path/'prep',lambda _:None)
        manifest_bytes=(tmp_path/'prep/manifest.json').read_bytes()
        flags=['--prepared-manifest-sha256',hashlib.sha256(manifest_bytes).hexdigest(),'--prepared-graph-digest',json.loads(manifest_bytes)['graphDigest'],'--prepared-dataflow',str(tmp_path/'prep/manifest.json'),'--prepared-training-node','split','--prepared-training-port','training','--prepared-validation-node','split','--prepared-validation-port','validation']
    out=tmp_path/'run'
    result=invoke(model,out,*flags,'--feature-columns-json',json.dumps(FEATURES),'--target-column','label')
    assert result.returncode==0,result.stdout+result.stderr
    assert json.loads((out/'manifest.json').read_text())['provenance']['featureColumns']==FEATURES

def test_json_columns_preview_and_mutual_exclusion(tmp_path):
    source,model=fixture(tmp_path)
    flags=['--dataset-csv',str(source),'--feature-columns-json',json.dumps(FEATURES),'--target-column','label']
    result=invoke(model,tmp_path/'preview',*flags,'--preview','--stop-at','output')
    assert result.returncode==0,result.stdout+result.stderr
    result=invoke(model,tmp_path/'conflict',*flags,'--feature-columns','x,y')
    assert result.returncode==1 and 'not allowed' in result.stdout

@pytest.mark.parametrize('value',['{}','[true,"x"]','["x"]','["x","x"]','["", "y"]'])
def test_malformed_json_mapping_fails_before_output(tmp_path,value):
    source,model=fixture(tmp_path)
    out=tmp_path/'bad'
    result=invoke(model,out,'--dataset-csv',str(source),'--feature-columns-json',value,'--target-column','label')
    assert result.returncode==1
    assert not (out/'manifest.json').exists()
