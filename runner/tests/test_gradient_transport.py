import json,subprocess,sys
from pathlib import Path
import pytest
from runner.tests.test_model_training_operation import training
from runner.tests.test_operation_authorization import publish,claim,operation_authority
from runner.tests.test_preview_operation_authorization import preview_operation
from runner.worker_authorization import validate_phase_arguments
from runner.tests.test_runner import graph
from runner.tests.test_gradient_engine import policy
from runner.tests.test_trainability import recipe as flags,owner
ROOT=Path(__file__).resolve().parents[2]
@pytest.mark.parametrize('flag,value',[('--gradient-policy-json',policy()),('--trainability-json',flags(owner('head','weight'))),('--derivative-capture-json',dict(schemaVersion=1,values=False))])
def test_exact_phase_and_standalone_authority(training,flag,value):
 text=json.dumps(value);assert validate_phase_arguments('training',[flag,text])[flag]==text
 with pytest.raises(ValueError):validate_phase_arguments('preparation',[flag,text])
 f=training;f[1]['arguments']+=[flag,text];publish(f)
 with claim(f) as authority:authority.check()
@pytest.mark.parametrize('flag,value',[('--gradient-policy-json',{}),('--trainability-json',{}),('--derivative-capture-json',dict(schemaVersion=1,values=1))])
def test_malformed_before_both_claims(training,flag,value):
 f=training;f[1]['arguments']+=[flag,json.dumps(value)];publish(f)
 with pytest.raises(ValueError):claim(f)
 with pytest.raises(ValueError):validate_phase_arguments('training',[flag,json.dumps(value)])
@pytest.mark.parametrize('device',['cpu','mps'])
def test_real_cli_new_policies_and_exact_artifact_delivery(tmp_path,device):
 import torch
 if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
 path=tmp_path/'graph.json';path.write_text(json.dumps(graph()))
 result=subprocess.run([sys.executable,str(ROOT/'runner/main.py'),'--graph',str(path),'--output',str(tmp_path/'run'),'--epochs','1','--sample-limit','32','--batch-size','7','--device',device,'--gradient-policy-json',json.dumps(policy()),'--trainability-json',json.dumps(flags(owner('head','weight')))],capture_output=True,text=True,timeout=30)
 assert result.returncode==0,result.stdout+result.stderr
 manifest=json.loads((tmp_path/'run/manifest.json').read_text());assert manifest['settings']['gradientPolicyRecipe']==policy()
 assert manifest['settings']['optimizerUpdatesPerEpoch']==2
 assert (tmp_path/'run/gradient_policy.py').is_file() and (tmp_path/'run/trainability_recipe.py').is_file()
 assert not (tmp_path/'run/derivative-results.json').exists()
@pytest.mark.parametrize('mode',['--preview','--capabilities','--export-code'])
@pytest.mark.parametrize('flag,value',[('--gradient-policy-json',policy()),('--trainability-json',flags())])
def test_cli_training_only(mode,flag,value):
 result=subprocess.run([sys.executable,str(ROOT/'runner/main.py'),mode,flag,json.dumps(value)],capture_output=True,text=True,timeout=30)
 assert result.returncode==1 and 'training only' in result.stdout.lower(),result.stdout+result.stderr
