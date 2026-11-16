"""Strict inline initialization authority and actual standalone training delivery."""
import json
from pathlib import Path
import subprocess
import sys
import pytest
from runner.tests.test_initialization import recipe,rule
from runner.tests.test_model_training_operation import training
from runner.tests.test_preview_operation_authorization import preview_operation
from runner.tests.test_operation_authorization import operation_authority,publish,claim
from runner.worker_authorization import validate_phase_arguments
from runner.tests.test_runner import graph
ROOT=Path(__file__).resolve().parents[2]

@pytest.mark.parametrize('algorithm',['xavier_uniform','kaiming_uniform','normal','zeros'])
def test_standalone_authority_admits_then_rechecks_pinned_recipe(training,algorithm):
    f=training;selected=recipe(rule('head',algorithm=algorithm))
    f[1]['arguments']+=['--initialization-recipe-json',json.dumps(selected)];publish(f)
    with claim(f) as authority:
        authority.check();f[1]['arguments'][-1]=json.dumps(recipe(rule('head',algorithm='zeros'),seed=43));publish(f)
        with pytest.raises(ValueError):authority.check()

@pytest.mark.parametrize('bad',[None,{},dict(schemaVersion=1,configurationVersion=1,componentID='initialization',config={'seed':42.0,'rules':[]}),recipe(rule('head',gain=True))])
def test_both_authorization_gates_refuse_malformed_recipe_before_claim(training,bad):
    text=json.dumps(bad);f=training;f[1]['arguments']+=['--initialization-recipe-json',text];publish(f)
    with pytest.raises(ValueError):claim(f)
    assert not (f[0]/'claim.json').exists()
    with pytest.raises(ValueError):validate_phase_arguments('training',['--initialization-recipe-json',text])


def test_phase_authority_exact_recipe_and_large_valid_wire_limit():
    # Long actual-owner strings are syntactically valid here; ownership belongs
    # to the engine. This crosses old8192 bounds without allocating model arrays.
    selected=recipe(rule('a'*9000,algorithm='zeros'));text=json.dumps(selected)
    assert 8192<len(text.encode())<32768
    assert validate_phase_arguments('training',['--initialization-recipe-json',text])['--initialization-recipe-json']==text
    with pytest.raises(ValueError):validate_phase_arguments('preparation',['--initialization-recipe-json',text])
    with pytest.raises(ValueError):validate_phase_arguments('training',['--initialization-recipe-json',' '*32769])


def test_standalone_large_recipe_allowance_is_only_for_initializer(training):
    f=training;f[1]['arguments']+=['--initialization-recipe-json',json.dumps(recipe(rule('a'*9000,algorithm='zeros')))];publish(f)
    with claim(f) as authority:authority.check()


@pytest.mark.parametrize('algorithm',['xavier_uniform','kaiming_uniform','normal','zeros'])
@pytest.mark.parametrize('device',['cpu','mps'])
def test_actual_cli_delivers_exact_selected_policy_and_new_helper(tmp_path,algorithm,device):
    import torch
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
    model=tmp_path/'graph.json';model.write_text(json.dumps(graph()));selected=recipe(rule('head',algorithm=algorithm))
    result=subprocess.run([sys.executable,str(ROOT/'runner/main.py'),'--graph',str(model),'--output',str(tmp_path/'run'),'--epochs','1','--sample-limit','32','--batch-size','16','--device',device,'--initialization-recipe-json',json.dumps(selected)],text=True,capture_output=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    manifest=json.loads((tmp_path/'run/manifest.json').read_text())
    assert manifest['settings']['initializationRecipe']==selected and manifest['provenance']['initialization']['application']=='applied'
    assert manifest['runtime']['device']==device
    assert (tmp_path/'run/initialization_recipe.py').is_file()
    assert json.loads(result.stdout.splitlines()[-1])['event']=='completed'


@pytest.mark.parametrize('mode',['--preview','--export-code','--capabilities'])
def test_cli_initializer_is_training_only_and_never_silently_ignored(tmp_path,mode):
    result=subprocess.run([sys.executable,str(ROOT/'runner/main.py'),mode,'--initialization-recipe-json',json.dumps(recipe(rule('head')))],text=True,capture_output=True,timeout=30)
    assert result.returncode==1
    assert 'training only' in result.stdout.lower()


def test_cli_duplicate_initializer_refuses(tmp_path):
    model=tmp_path/'graph.json';model.write_text(json.dumps(graph()));text=json.dumps(recipe(rule('head')))
    result=subprocess.run([sys.executable,str(ROOT/'runner/main.py'),'--graph',str(model),'--output',str(tmp_path/'run'),'--epochs','1','--initialization-recipe-json',text,'--initialization-recipe-json',text],text=True,capture_output=True,timeout=30)
    assert result.returncode==1 and 'repeated' in result.stdout.lower()
