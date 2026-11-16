import copy,json,hashlib
import pytest,torch
from runner import engine
from runner.tests.prepared_objective_fixtures import prepared_window_fixture as prepared
from runner.tests.test_gradient_engine import policy,optimizer,saved
from runner.tests.test_trainability import recipe as flags,owner
from runner.tests.test_checkpoint_lifecycle import assert_nested_equal,rewrite_checkpoint
from runner.scheduler_configuration import canonical_configuration

def options(tmp,device='cpu'):
 graph,r,m,data=prepared(tmp/'source','loss_mse')
 return graph,dict(epochs=2,seed=41,batch_size=5,device=device,prepared=data,objective_recipe=r,target_mapping=m,optimizer_recipe=optimizer(),gradient_policy=policy())
def inventory(root):return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
@pytest.mark.parametrize('implicit',[False,True])
def test_cpu_exact_resume_all_state_and_updated_ownership(tmp_path,implicit):
 graph,kw=options(tmp_path)
 if implicit:kw.pop('optimizer_recipe')
 kw['trainability']=flags(owner('learn','weight'))
 engine.run(graph,tmp_path/'full',**{**kw,'epochs':4});engine.run(graph,tmp_path/'parent',**kw);pins=inventory(tmp_path/'parent')
 engine.run(graph,tmp_path/'resume',resume_checkpoint=tmp_path/'parent/checkpoint.pt',**{**kw,'epochs':4})
 full,resume=saved(tmp_path/'full'),saved(tmp_path/'resume')
 for key in ('modelState','optimizerState','rngState','updateSchedule','objectiveState'):assert_nested_equal(full[key],resume[key])
 assert json.loads((tmp_path/'full/metrics.json').read_text())==json.loads((tmp_path/'resume/metrics.json').read_text())
 assert inventory(tmp_path/'parent')==pins
 for change in (dict(gradient_policy=None),dict(gradient_policy=policy(1)),dict(trainability=None),dict(trainability=flags(owner('learn','bias')))):
  with pytest.raises(ValueError,match='matching'):engine.run(graph,tmp_path/('bad'+str(len(str(change)))),resume_checkpoint=tmp_path/'parent/checkpoint.pt',**{**kw,**change,'epochs':4})
@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('child_flags',[None,'unfreeze','other'])
def test_finetune_original_parent_ownership_then_child_fresh_subset(tmp_path,device,child_flags):
 if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
 graph,kw=options(tmp_path,device);kw['trainability']=flags(owner('learn','weight'))
 engine.run(graph,tmp_path/'parent',**kw);parent=saved(tmp_path/'parent');pins=inventory(tmp_path/'parent');observed=[]
 child=dict(kw,epochs=1,gradient_observer=observed.append)
 child['trainability']=None if child_flags is None else flags(owner('learn','weight','trainable')) if child_flags=='unfreeze' else flags(owner('learn','bias'))
 engine.run(graph,tmp_path/'child',finetune_checkpoint=tmp_path/'parent/checkpoint.pt',**child)
 start=observed[0]
 for native in start['owners']:
  assert native['parameterValues']==parent['modelState'][native['name']].flatten().tolist() and native['optimizerStateKeys']==[]
 assert saved(tmp_path/'child')['optimizerUpdates']==([2] if child_flags=='other' else [2,2])
 assert inventory(tmp_path/'parent')==pins
 if device=='mps':
  with pytest.raises(ValueError,match='CPU'):engine.run(graph,tmp_path/'refused',resume_checkpoint=tmp_path/'parent/checkpoint.pt',**{**kw,'epochs':3})
def test_onecycle_ticks_actual_updates_and_cpu_resume_trace(tmp_path):
 graph,kw=options(tmp_path);kw['scheduler_recipe']=dict(schemaVersion=1,configurationVersion=1,componentID='scheduler_one_cycle_lr',config=canonical_configuration('scheduler_one_cycle_lr',dict(planned_total_batches=8,max_lr=.03)))
 engine.run(graph,tmp_path/'full',**{**kw,'epochs':4});engine.run(graph,tmp_path/'parent',**kw)
 engine.run(graph,tmp_path/'resume',resume_checkpoint=tmp_path/'parent/checkpoint.pt',**{**kw,'epochs':4})
 a,b=saved(tmp_path/'full'),saved(tmp_path/'resume');assert_nested_equal(a,b)
 trace=json.loads((tmp_path/'full/rate-trace.json').read_text());assert [r['batch'] for r in trace['records']]==[1,2]*4
 assert trace['records'][-1]['nextRateUnused'] and len(trace['records'])==8
@pytest.mark.parametrize('bad',['pending','counts','foreign'])
def test_checkpoint_pending_schedule_tamper_refused(tmp_path,bad):
 graph,kw=options(tmp_path);engine.run(graph,tmp_path/'parent',**kw)
 def mutate(value,manifest):
  if bad=='pending':value['updateSchedule']['pendingMicrobatches']=1
  elif bad=='counts':value['updateSchedule']['completedOptimizerUpdates']+=1
  else:value['updateSchedule']['extra']=0
 rewrite_checkpoint(tmp_path/'parent',mutate)
 with pytest.raises(ValueError,match='epoch|schedule'):engine.run(graph,tmp_path/'child',resume_checkpoint=tmp_path/'parent/checkpoint.pt',**{**kw,'epochs':4})

@pytest.mark.parametrize('field',['gradientPolicy','trainability'])
def test_missing_or_forged_optional_manifest_identity_refuses(tmp_path,field):
 graph,kw=options(tmp_path);kw['trainability']=flags(owner('learn','weight'));engine.run(graph,tmp_path/'parent',**kw)
 path=tmp_path/'parent/manifest.json';manifest=json.loads(path.read_text());manifest['provenance'][field]['recipe']['config']['foreign']=1;path.write_text(json.dumps(manifest))
 with pytest.raises(ValueError):engine.run(graph,tmp_path/'child',finetune_checkpoint=tmp_path/'parent/checkpoint.pt',**{**kw,'epochs':1})
