import json,subprocess,sys
from pathlib import Path
import pytest
from runner.worker_authorization import validate_phase_arguments
from runner.tests.test_preview_operation_authorization import preview_operation
from runner.tests.test_model_training_operation import training
from runner.tests.test_operation_authorization import publish,claim,operation_authority
from runner.tests.test_dataset_loader_configuration import recipe
from runner.tests.test_runner import graph
ROOT=Path(__file__).resolve().parents[2]

@pytest.mark.parametrize('flag,value',[('--dataset-recipe-json',recipe(kind='subset',indices=[4,1,0])),('--loader-recipe-json',recipe('data_loader',batch_size=32))])
def test_both_authorities_deliver_exact_recipe(training,flag,value):
    text=json.dumps(value)
    assert validate_phase_arguments('training',[flag,text])[flag]==text
    with pytest.raises(ValueError):validate_phase_arguments('preparation',[flag,text])
    f=training;f[1]['arguments'] += [flag,text];publish(f)
    with claim(f) as authority:authority.check()

@pytest.mark.parametrize('flag',['--dataset-recipe-json','--loader-recipe-json'])
@pytest.mark.parametrize('text',['{}','null',' '*32769,'{"schemaVersion":1,"schemaVersion":1,"componentID":"dataset_source","configurationVersion":1,"config":{"kind":"indexed","indices":[],"max_rows":100000}}'],ids=['empty','null','oversize','duplicate'])
def test_malformed_raw_recipe_both_gates(training,flag,text):
    with pytest.raises(ValueError):validate_phase_arguments('training',[flag,text])
    f=training;f[1]['arguments'] += [flag,text];publish(f)
    with pytest.raises(ValueError):claim(f)
    assert not (f[0]/'claim.json').exists()


def test_large_valid_subset_wire_scoped_bound(training):
    text=json.dumps(recipe(kind='subset',indices=list(range(2000))))
    assert 8192<len(text.encode())<32768
    assert validate_phase_arguments('training',['--dataset-recipe-json',text])['--dataset-recipe-json']==text
    f=training;f[1]['arguments'] += ['--dataset-recipe-json',text];publish(f)
    with claim(f) as authority:authority.check()


def test_raw_loader_batch_mismatch_refuses_before_claim(training):
    f=training;f[1]['arguments'] += ['--loader-recipe-json',json.dumps(recipe('data_loader',batch_size=1))];publish(f)
    with pytest.raises(ValueError):claim(f)

@pytest.mark.parametrize('mode',['--preview','--export-code','--capabilities'])
@pytest.mark.parametrize('flag,value',[('--dataset-recipe-json',recipe()),('--loader-recipe-json',recipe('data_loader'))])
def test_cli_selected_recipe_never_silently_ignored(mode,flag,value):
    result=subprocess.run([sys.executable,str(ROOT/'runner/main.py'),mode,flag,json.dumps(value)],capture_output=True,text=True,timeout=30)
    assert result.returncode==1 and 'training only' in result.stdout.lower(),result.stdout+result.stderr

@pytest.mark.parametrize('device',['cpu','mps'])
def test_actual_loose_cli_selected_native_loading(tmp_path,device):
    import torch
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
    path=tmp_path/'graph.json';path.write_text(json.dumps(graph()))
    selected=recipe(kind='boundedIterable',max_rows=9);loader=recipe('data_loader',batch_size=4,shuffle=True,num_workers=1,prefetch_factor=1,timeout=10)
    result=subprocess.run([sys.executable,str(ROOT/'runner/main.py'),'--graph',str(path),'--output',str(tmp_path/'run'),'--epochs','2','--sample-limit','32','--batch-size','4','--device',device,'--dataset-recipe-json',json.dumps(selected),'--loader-recipe-json',json.dumps(loader)],capture_output=True,text=True,timeout=40)
    assert result.returncode==0,result.stdout+result.stderr
    manifest=json.loads((tmp_path/'run/manifest.json').read_text())
    assert manifest['settings']['datasetRecipe']==selected and manifest['settings']['loaderRecipe']==loader
    assert manifest['dataLoading']['retainedRows']==9
    assert (tmp_path/'run/data_loading_recipe.py').is_file()
