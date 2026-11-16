"""Retained objective launch authority stays typed, pinned and Torch-free."""
import json,subprocess,sys
from pathlib import Path
import pytest
from runner.tests.test_model_training_operation import training
from runner.tests.test_preview_operation_authorization import source,inventory,preview_operation
from runner.tests.test_operation_authorization import publish,claim,operation_authority
from runner.loss_configuration import validate_recipe
ROOT=Path(__file__).resolve().parents[2]

def select_objective(f,tmp_path,component='loss_cross_entropy'):
 source(f,'prepared',tmp_path);inventory(f);args=f[1]['arguments']
 for key in ('--feature-columns-json','--target-column'):
  i=args.index(key);del args[i:i+2]
 recipe=validate_recipe({'schemaVersion':1,'componentID':component,'configurationVersion':1,'config':{}})
 mapping={'schemaVersion':1,'componentID':component,'inputColumns':['feature','second'],'roles':[{'role':'target','columns':['group'],'sampleShape':[],'dtype':'int64'}],'tripletSourceGroups':[]}
 args+=['--objective-recipe-json',json.dumps(recipe),'--objective-target-mapping-json',json.dumps(mapping),'--objective-scalarization','mean']
 return recipe,mapping

@pytest.mark.parametrize('component',['loss_cross_entropy','loss_nll'])
def test_objective_selection_admitted_and_exact_authority_rechecks(training,tmp_path,component):
 f=training;_,m=select_objective(f,tmp_path,component);publish(f)
 with claim(f)as authority:
  authority.check();m['roles'][0]['columns']=['foreign']
  args=f[1]['arguments'];args[args.index('--objective-target-mapping-json')+1]=json.dumps(m);publish(f)
  with pytest.raises(ValueError):authority.check()

@pytest.mark.parametrize('change',['orphan','wrong-component','overlap','duplicate','extra-field','scalarization','legacy-target','split','unprepared'])
def test_objective_malformed_selection_never_claims(training,tmp_path,change):
 f=training;r,m=select_objective(f,tmp_path);args=f[1]['arguments']
 if change=='orphan':i=args.index('--objective-recipe-json');del args[i:i+2]
 elif change=='wrong-component':m['componentID']='loss_mse'
 elif change=='overlap':m['roles'][0]['columns']=['feature']
 elif change=='duplicate':m['inputColumns'].append('feature')
 elif change=='extra-field':m['code']='foreign'
 elif change=='scalarization':args[args.index('--objective-scalarization')+1]='median'
 elif change=='legacy-target':args+=['--target-column','group']
 elif change=='split':args+=['--split-seed','7']
 else:
  i=args.index('--prepared-dataflow');del args[i:i+2]
 args[args.index('--objective-target-mapping-json')+1]=json.dumps(m);publish(f)
 with pytest.raises(ValueError):claim(f)
 assert not(f[0]/'claim.json').exists()

def test_objective_validation_is_standard_library_only_in_isolated_reader():
 script="import sys;sys.path.insert(0,sys.argv[1]);import loss_target_mapping,model_training_operation_contract;assert 'torch'not in sys.modules"
 result=subprocess.run([sys.executable,'-I','-c',script,str(ROOT/'runner')],capture_output=True,text=True,timeout=20)
 assert result.returncode==0,result.stderr

from runner.tests.test_lifetime_watchdog import libraries
from runner.tests.test_worker_bootstrap import entry,run_supervisor
from runner.tests.test_operation_bootstrap import control

@pytest.mark.skipif(sys.platform!='darwin',reason='Native macOS lifetime guard')
def test_actual_guarded_objective_training_keeps_authority_and_exports(tmp_path,training,libraries):
 f=training;r,m=select_objective(f,tmp_path);publish(f)
 guard,_=libraries;path=entry(tmp_path,guard)
 marker='    print(json.dumps({"event":"engineMarker","arguments":result.arguments}),flush=True)'
 text=path.read_text();assert marker in text
 path.write_text(text.replace(marker,'    from main import main\n    sys.argv=[sys.argv[0],*result.arguments]\n    sys.exit(main(check_cancelled=result.authority.check))'))
 code,events,error=run_supervisor(path,control(f));assert code==0,(events,error)
 assert events[0]['toolMode']=='modelTraining'and events[-1]['event']=='completed'
 output=f[0]/'artifacts';manifest=json.loads((output/'manifest.json').read_text())
 assert manifest['settings']['objectiveRecipe']==r and manifest['settings']['targetMapping']==m
 assert manifest['provenance']['objectiveMapping']==m and 'targetColumn'not in manifest['provenance']
 assert(output/'objective_recipe.py').is_file()and(output/'checkpoint.pt').stat().st_size>0
