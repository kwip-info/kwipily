"""A real learned token model exported without the app or mutable source data."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import pytest
import torch
from runner import dataflow as df
from runner.engine import run,GraphModel
from runner.named_prepared_data import load_named_prepared
from runner.tests.test_named_prepared_integration import fixture
from runner.tests.test_embedding_graph import graph as embedding_graph

ROOT=Path(__file__).resolve().parents[2]


def test_saved_tokenizer_named_mapping_predictions_and_cli_survive_source_removal(tmp_path):
    out,dataflow,_=fixture(tmp_path)
    manifest=out/'manifest.json';pin=hashlib.sha256(manifest.read_bytes()).hexdigest()
    graph=embedding_graph();graph['nodes'][0]['config']['shape']=[5];graph['nodes'][3]['config']['in_features']=15;graph['namedInputs']={'tokens':'source'}
    plan={'schemaVersion':1,'inputs':[{'name':'tokens','nodeID':'source','representation':'tokens','dtype':'int64','sampleShape':[5],'columns':['tokens'],'lengthPolicy':'error','paddingValue':1,'vocabularySize':4}],'roles':[{'name':'target','representation':'numeric','dtype':'int64','sampleShape':[],'columns':['label']}],'maximumBatchBytes':1048576}
    prepared=load_named_prepared(manifest,'text','training','text','validation',plan,graph,expected_manifest_sha256=pin,expected_graph_digest=df.digest(dataflow))
    origin=tmp_path/'learned';run(graph,origin,epochs=1,device='cpu',prepared=prepared)
    target=tmp_path/'portable';target.mkdir()
    for name in ('model.py','checkpoint.pt'):shutil.copyfile(origin/name,target/name)
    for name in ('named_inputs.py','portable_text_encoding.py'):shutil.copyfile(out/name,target/name)
    shutil.copyfile(ROOT/'Sources/KwipMLApp/Resources/inference.py',target/'inference.py')
    entry=next(e for e in json.loads(manifest.read_bytes())['artifacts'] if e['kind']=='state')
    state=json.loads((out/entry['path']).read_bytes())
    tokenizer={'schemaVersion':1,'configuration':state['configuration'],'vocabulary':state['vocabulary'],'unicodeVersion':state['unicodeVersion'],'sourceStateSHA256':state['contentDigest']}
    (target/'tokenizers.json').write_text(json.dumps({'tokens':tokenizer}))
    contract={'schemaVersion':4,'input':{'kind':'namedInputs','plan':plan,'namedValues':True,'preprocessing':'capturedPreparation'},'output':{'kind':'logits','dtype':'float32','shape':['N',2]}}
    (target/'input-contract.json').write_text(json.dumps(contract))
    entries=[{'path':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in target.iterdir()]
    (target/'export-manifest.json').write_text(json.dumps({'schemaVersion':1,'kind':'kwipmlTrainedModel','files':entries}))
    reference=GraphModel(graph,bundled_dataset=False);reference.load_state_dict(torch.load(origin/'checkpoint.pt',weights_only=True)['modelState']);reference.eval()
    shutil.rmtree(out);shutil.rmtree(origin);(tmp_path/'texts.csv').unlink()
    spec=importlib.util.spec_from_file_location('named_export',target/'inference.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    predictor=helper.load_model()
    assert predictor.encode_text(['ababa','babab'],'tokens')==[[2,3,2,3,2],[3,2,3,2,3]]
    records=[{'tokens':ids} for ids in predictor.encode_text(['ababa','babab'],'tokens')]
    tensors=predictor.prepare_records(records)
    torch.testing.assert_close(predictor.predict(tensors,batch_size=1),reference(tensors),rtol=1e-6,atol=1e-7)
    with pytest.raises(ValueError):predictor.predict({'tokens':torch.tensor([[99,2,2,2,2]])})
    with pytest.raises(ValueError):predictor.prepare_records([{'tokens':[2]}])
    with pytest.raises(ValueError):predictor.predict({'wrong':tensors['tokens']})
    result=subprocess.run([sys.executable,str(target/'inference.py'),'--example'],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    (target/'tokenizers.json').write_text('{}')
    with pytest.raises(ValueError):helper.load_model()
