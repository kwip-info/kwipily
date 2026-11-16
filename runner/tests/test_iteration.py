import json
import subprocess
import sys
from pathlib import Path
import torch
import pytest
from runner.engine import run, GraphModel, load_data
from runner.preview import preview
from runner.tests.test_runner import graph,csv_fixture

ROOT=Path(__file__).resolve().parents[2]

def test_separate_model_and_split_seed(tmp_path):
    for name,seed,split in [('a',1,42),('b',2,42),('c',1,43)]:
        run(graph(),tmp_path/name,epochs=1,device='cpu',seed=seed,split_seed=split,sample_limit=40,validation_percent=25,batch_size=7,learning_rate=.01)
    manifests=[json.loads((tmp_path/n/'manifest.json').read_text()) for n in ['a','b','c']]
    a,b,c=manifests
    assert a['provenance']['splitHash']==b['provenance']['splitHash']!=c['provenance']['splitHash']
    assert a['provenance']['sourceHash']==c['provenance']['sourceHash']
    assert set(a['provenance']['trainingRowIDs']).isdisjoint(a['provenance']['validationRowIDs'])
    assert a['provenance']['samples']==40 and a['evaluation']['count']==10
    assert a['settings']['batchSize']==7 and a['settings']['lastBatchSize']==2 and a['settings']['learningRate']==.01
    sa=torch.load(tmp_path/'a/checkpoint.pt',weights_only=True)['modelState']; sb=torch.load(tmp_path/'b/checkpoint.pt',weights_only=True)['modelState']
    assert any(not torch.equal(sa[k],sb[k]) for k in sa)

def test_weighted_epoch_loss_and_batches(tmp_path,monkeypatch):
    class NoStep(torch.optim.Adam):
        def step(self,closure=None): return None
    monkeypatch.setattr(torch.optim,'Adam',NoStep)
    data,_=load_data(split_seed=42,sample_limit=40,validation_percent=25)
    torch.manual_seed(42); expected=GraphModel(graph())
    expected_loss=torch.nn.functional.cross_entropy(expected(data[0]),data[1]).item()
    seen=[]; original=GraphModel.forward
    def forward(self,x,inspect=False):
        if self.training: seen.append(len(x))
        return original(self,x,inspect)
    monkeypatch.setattr(GraphModel,'forward',forward)
    events=[]; run(graph(),tmp_path,epochs=1,device='cpu',sample_limit=40,validation_percent=25,batch_size=7,emit=events.append)
    assert seen==[7,7,7,7,2]
    assert next(e for e in events if e['event']=='epoch')['trainLoss']==pytest.approx(expected_loss,abs=1e-6)
    assert any(e['event']=='trainingProgress' and e['kind']=='training' for e in events)

@pytest.mark.parametrize('kwargs',[{'batch_size':0},{'batch_size':1.5},{'sample_limit':19},{'sample_limit':1001},{'split_seed':-1},{'validation_percent':0},{'validation_percent':50.5},{'learning_rate':0},{'learning_rate':float('nan')},{'learning_rate':True}])
def test_iteration_ranges_reject(tmp_path,kwargs):
    with pytest.raises(ValueError): run(graph(),tmp_path,epochs=1,device='cpu',**kwargs)
    assert not list(tmp_path.iterdir())

def test_csv_subsample_preserves_classes(tmp_path):
    p=csv_fixture(tmp_path/'data.csv')
    data,provenance=load_data(dataset_csv=p,feature_columns=['first','second'],target_column='label',split_seed=3,sample_limit=20,validation_percent=30)
    assert len(data[0])==14 and len(data[2])==6
    assert set(data[1].tolist())==set(data[3].tolist())=={0,1}
    assert len(provenance['selectedRowIDs'])==20

def test_preview_split_config_invalidates(tmp_path):
    a=preview(graph(),tmp_path/'a',device='cpu',sample_limit=40,split_seed=3,validation_percent=25)
    with pytest.raises(ValueError,match='stale'):
        preview(graph(),tmp_path/'b',device='cpu',sample_limit=40,split_seed=4,validation_percent=25,resume_from='merge',cache=a['cachePath'])

def test_export_code_without_source_or_output(tmp_path):
    g=tmp_path/'graph.json';g.write_text(json.dumps(graph()))
    result=subprocess.run([sys.executable,str(ROOT/'runner/main.py'),'--graph',str(g),'--export-code'],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    event=json.loads(result.stdout);assert event['event']=='codeExport'
    assert 'class Model(nn.Module)' in event['code']
    assert set(tmp_path.iterdir())=={g}
    code=compile(event['code'],'model.py','exec'); namespace={};exec(code,namespace)
    assert namespace['Model']()(torch.randn(3,2)).shape==(3,2)

def test_recipe_version_marks_training_and_preview(tmp_path):
    run(graph(),tmp_path/'train',epochs=1,device='cpu')
    assert json.loads((tmp_path/'train/manifest.json').read_text())['recipeVersion']==2
    result=preview(graph(),tmp_path/'preview',device='cpu')
    assert result['recipeVersion']==2
    metadata=Path(result['cachePath']).with_suffix('.json')
    value=json.loads(metadata.read_text()); assert value['recipeVersion']==2
    del value['recipeVersion']; metadata.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='stale'):
        preview(graph(),tmp_path/'resume',device='cpu',resume_from='merge',cache=result['cachePath'])
