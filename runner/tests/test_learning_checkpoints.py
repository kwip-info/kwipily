"""Real engine lifecycle assays, independent affine update math and strict transport."""
import copy,json,math,time
import pytest
from runner import learning_checkpoints as lesson

def tensor(values,shape):return {'shape':shape,'values':values}
def fixture(seed=41):
 x=[((i*5+seed)%17-8)/8 for i in range(24)]
 y=[.5*x[i]-.25*x[i+1]+.125 if j==0 else -.25*x[i]+.75*x[i+1]-.125 for i in range(0,24,2) for j in range(2)]
 return {'id':'case-'+str(seed),'inputs':{'trainingInputs':tensor(x[:16],[8,2]),'trainingTargets':tensor(y[:16],[8,2]),'heldoutInputs':tensor(x[16:],[4,2]),'heldoutTargets':tensor(y[16:],[4,2]),'seed':tensor([seed],[1])}}
def submission(task,**changes):
 return {'schemaVersion':1,'kind':'checkpointLearning','taskID':task,'configuration':{'trainingMode':'fineTune' if task==lesson.TASKS[0]else'exactResume','parent':'savedLessonRun','epochs':1 if task==lesson.TASKS[0]else 4,'batchSize':4,**changes}}
def read(v):return v['shape'],v['values']
def execute(task,sub=None,fixtures=None):
 e={'fixtures':[],'maximumTensorElements':0};lesson.execute(sub or submission(task),fixtures or [fixture()],e,lambda:False,read,task=task);return e
@pytest.mark.parametrize('task',lesson.TASKS)
def test_actual_saved_reopened_lifecycle_has_measured_state_and_nontrivial_outputs(task):
 e=execute(task);t=e['fixtures'][0]['tensors'];assert t['parentByteDifference']['values']==[0]
 if task==lesson.TASKS[0]:
  for k in ('restoredParameterDifference','restoredPredictionDifference','childUpdateDifference','freshMemoryDifference'):assert max(abs(v)for v in t[k]['values'])<2e-6,k
  assert any(abs(v)>.001 for v in t['savedParameters']['values'])
  assert t['completedEpochs']['values']==[2,1]
 else:
  for k in ('continuationStateDifference','continuationHistoryDifference','continuationRandomDifference'):assert t[k]['values']==[0],k
  assert t['completedEpochs']['values']==[2,4,4]
  assert len(t['resumedParameters']['values'])==6
  assert t['continuationEpochs']['values']==[2,3,4]
@pytest.mark.parametrize('task',lesson.TASKS)
def test_wrong_fresh_mode_is_measured_failure_not_an_answer_flag(task):
 t=execute(task,submission(task,trainingMode='fresh',parent='none'))['fixtures'][0]['tensors']
 if task==lesson.TASKS[0]:assert max(abs(v)for v in t['childUpdateDifference']['values'])>1e-5
 else:assert t['continuationEpochs']['values']!=[2,3,4]
@pytest.mark.parametrize('change',[lambda s:s.update(answer=True),lambda s:s['configuration'].update(path='/tmp/checkpoint.pt'),lambda s:s['configuration'].update(epochs=True),lambda s:s['configuration'].update(batchSize=0),lambda s:s['configuration'].update(parent='../parent'),lambda s:s.update(taskID='other')])
def test_strict_submission_rejects_untrusted_controls_before_execution(change):
 s=submission(lesson.TASKS[0]);change(s)
 with pytest.raises(ValueError):lesson.configuration(s,lesson.TASKS[0])
def test_actual_incompatible_resume_batch_and_nonadvancing_epochs_refuse():
 for changes in ({'batchSize':2},{'epochs':2}):
  with pytest.raises(ValueError):execute(lesson.TASKS[1],submission(lesson.TASKS[1],**changes))
def test_independent_scalar_affine_mse_sgd_reference_known_two_batches():
 # Prediction [1,2], target [0,0], mean loss. Derivatives [1,2] then second batch [0.8,1.6].
 p,m=lesson.affine_update([1,0,0,2,0,0],[[1,0]]*8,[[0,0]]*8)
 assert p==pytest.approx([.77,0,-.23,1.54,0,-.46],abs=1e-12)
 assert m==pytest.approx([1.3,0,1.3,2.6,0,2.6],abs=1e-12)
@pytest.mark.parametrize('task',lesson.TASKS)
def test_four_varied_fixtures_remain_within_existing_assessment_budget(task):
 start=time.monotonic();e=execute(task,fixtures=[fixture(s)for s in (7,41,173,911)])
 assert len(e['fixtures'])==4 and time.monotonic()-start<60 and e['maximumTensorElements']<=1_000_000
 assert len({tuple(f['tensors']['savedParameters']['values'])for f in e['fixtures']})==4
@pytest.mark.parametrize('task',lesson.TASKS)
def test_real_worker_dispatch_four_fixtures_and_identity(tmp_path,task):
 import hashlib,subprocess,sys
 from pathlib import Path
 from runner.learning_evidence import runtime_versions
 request={'assessmentID':'assessment.'+task,'assessmentVersion':{'major':1,'minor':0,'patch':0},'fixtureDigest':'a'*64,'runtime':{'backend':'pytorch','version':runtime_versions()['pytorch'],'capabilities':{'checkpoint_lifecycle':{'major':1,'minor':0,'patch':0}}},'budget':{'milliseconds':60000,'maximumTensorElements':1000000},'fixtures':[fixture(s)for s in (7,41,173,911)]}
 raw=json.dumps(submission(task),sort_keys=True).encode();sp=tmp_path/'submission.json';sp.write_bytes(raw);rp=tmp_path/'request.json';rp.write_text(json.dumps(request))
 done=subprocess.run([sys.executable,str(Path(__file__).parents[1]/'main.py'),'--learning-evidence-request',str(rp),'--learning-submission',str(sp),'--learning-submission-sha256',hashlib.sha256(raw).hexdigest()],capture_output=True,text=True)
 assert done.returncode==0,done.stdout+done.stderr
 evidence=json.loads(done.stdout.strip().splitlines()[-1])['evidence'];assert not evidence.get('backendError'),evidence
 assert len(evidence['fixtures'])==4 and evidence['submissionDigest']==hashlib.sha256(raw).hexdigest()
 assert evidence['elapsedMilliseconds']<60000 and not evidence['cancelled']
 for f in evidence['fixtures']:
  assert f['tensors']['parentByteDifference']['values']==[0]
  key='childUpdateDifference'if task==lesson.TASKS[0]else'continuationStateDifference'
  assert max(abs(v)for v in f['tensors'][key]['values'])<2e-6
@pytest.mark.parametrize('mutation',[lambda f:f['inputs'].update(state=tensor([1],[1])),lambda f:f['inputs']['seed'].update(values=[True]),lambda f:f['inputs']['trainingInputs'].update(shape=[2,8]),lambda f:f['inputs']['heldoutInputs']['values'].__setitem__(0,float('nan'))])
def test_invalid_fixture_refuses_before_preparation_or_training(monkeypatch,mutation):
 f=fixture();mutation(f);monkeypatch.setattr(lesson,'prepared',lambda *a:pytest.fail('Invalid fixture entered preparation'))
 with pytest.raises(ValueError):execute(lesson.TASKS[0],fixtures=[f])
def test_cancel_before_preparation_and_state_member_comparison_refuses_missing_or_changed_type(monkeypatch):
 import torch
 monkeypatch.setattr(lesson,'prepared',lambda *a:pytest.fail('Cancelled fixture entered preparation'))
 with pytest.raises(ValueError,match='cancelled'):lesson.execute(submission(lesson.TASKS[0]),[fixture()],{},lambda:True,read,task=lesson.TASKS[0])
 with pytest.raises(ValueError):lesson.difference({'a':torch.zeros(2)},{})
 with pytest.raises(ValueError):lesson.difference(torch.zeros(2),torch.zeros(2,dtype=torch.float64))
 assert lesson.difference({'a':torch.tensor([1.,2.])},{'a':torch.tensor([1.,3.])})==1
@pytest.mark.parametrize('seed',[7,41,173,911])
def test_prepared_roles_match_supplied_training_and_heldout_rows_exactly(tmp_path,seed):
 from runner.loss_configuration import validate_recipe
 f=fixture(seed);v={k:read(t)for k,t in f['inputs'].items()};recipe=validate_recipe({'schemaVersion':1,'componentID':'loss_mse','configurationVersion':1,'config':{}})
 _,prepared=lesson.prepared(tmp_path,v,seed,recipe);data=prepared[0]
 assert data[0].flatten().tolist()==v['trainingInputs'][1]
 assert data[1]['target'].flatten().tolist()==v['trainingTargets'][1]
 assert data[2].flatten().tolist()==v['heldoutInputs'][1]
 assert data[3]['target'].flatten().tolist()==v['heldoutTargets'][1]
 assert set(prepared[1]['trainingRowIDs']).isdisjoint(prepared[1]['validationRowIDs'])
 graph=json.loads((tmp_path/'flow.json').read_text())
 assert [n['operator']for n in graph['nodes']]==['kwip.data.sourceCSV','kwip.data.columns','kwip.data.partition']
 manifest=json.loads((tmp_path/'prepared/manifest.json').read_text())
 partitions={}
 for role in ('training','validation','test'):
  entry=next(a for a in manifest['artifacts']if a.get('nodeID')=='p'and a.get('port')==role)
  table=json.loads((tmp_path/'prepared'/entry['path']).read_text());partitions[role]=table
  assert table['lineage']['role']==role
  assert len(table['rows'])=={'training':8,'validation':4,'test':4}[role]
 assert [row[0]for row in partitions['training']['rows']]==list(range(8))
 assert [row[0]for row in partitions['validation']['rows']]==list(range(8,12))
 assert [row[0]for row in partitions['test']['rows']]==list(range(12,16))
 ids=[set(t['lineage']['observationIDs'])for t in partitions.values()]
 assert all(ids[i].isdisjoint(ids[j])for i in range(3)for j in range(i))
 assert prepared[1]['preparation']['fittedStateDigests']==[]
@pytest.mark.parametrize('member',['modelState','optimizerState','rngState','scheduler','metrics'])
def test_actual_resumed_artifact_perturbations_fail_the_relevant_comparison(monkeypatch,member):
 import torch
 original=lesson.engine.run
 def altered(graph,output,*args,**kwargs):
  result=original(graph,output,*args,**kwargs)
  if output.name=='child':
   if member=='metrics':
    path=output/'metrics.json';v=json.loads(path.read_text());v[-1]['trainLoss']+=.125;path.write_text(json.dumps(v))
   else:
    path=output/'checkpoint.pt';v=torch.load(path,map_location='cpu',weights_only=True)
    if member=='modelState':next(iter(v['modelState'].values())).flatten()[0]+=.125
    elif member=='optimizerState':next(iter(v['optimizerState']['state'].values()))['momentum_buffer'].flatten()[0]+=.125
    elif member=='rngState':v['rngState'][0]^=1
    else:v['scheduler']['ticks']+=1
    torch.save(v,path)
  return result
 monkeypatch.setattr(lesson.engine,'run',altered)
 t=execute(lesson.TASKS[1])['fixtures'][0]['tensors'];key='continuationHistoryDifference'if member=='metrics'else'continuationRandomDifference'if member=='rngState'else'continuationStateDifference'
 assert t[key]['values'][0]>0
def test_duplicate_training_heldout_feature_row_still_refuses_at_actual_float32_boundary():
 f=fixture(41);f['inputs']['heldoutInputs']['values'][:2]=f['inputs']['trainingInputs']['values'][:2]
 with pytest.raises(ValueError,match='overlap across prepared partitions at float32 precision'):
  execute(lesson.TASKS[0],fixtures=[f])
@pytest.mark.parametrize('task',lesson.TASKS)
def test_packaged_isolated_worker_uses_verified_registry_without_checkout_layout(tmp_path,task):
 import hashlib,importlib.util,os,shutil,subprocess,sys
 from pathlib import Path
 from runner.learning_evidence import runtime_versions
 root=Path(__file__).resolve().parents[2];bundle=tmp_path/'KwipMLRuntime';bundle.mkdir()
 shutil.copy2(root/'scripts/runtime_launch.py',bundle/'launch.py')
 shutil.copytree(root/'runner',bundle/'runner',ignore=shutil.ignore_patterns('__pycache__','tests'))
 registry=bundle/'components.json';shutil.copy2(root/'Sources/KwipMLCore/Resources/components.json',registry)
 spec=importlib.util.spec_from_file_location('checkpoint_packager',root/'scripts/runtime_package.py');packager=importlib.util.module_from_spec(spec);spec.loader.exec_module(packager)
 (bundle/'runtime-manifest.json').write_text('{"schemaVersion":1}');packager.refresh_worker_integrity(bundle)
 assert not (bundle/'Sources').exists()
 request={'assessmentID':'assessment.'+task,'assessmentVersion':{'major':1,'minor':0,'patch':0},'fixtureDigest':'b'*64,'runtime':{'backend':'pytorch','version':runtime_versions()['pytorch'],'capabilities':{'checkpoint_lifecycle':{'major':1,'minor':0,'patch':0}}},'budget':{'milliseconds':60000,'maximumTensorElements':1000000},'fixtures':[fixture(s)for s in (7,41,173,911)]}
 rp=tmp_path/'request.json';rp.write_text(json.dumps(request));raw=json.dumps(submission(task),sort_keys=True).encode();sp=tmp_path/'submission.json';sp.write_bytes(raw)
 args=[sys.executable,'-I','-B',str(bundle/'launch.py'),'--learning-evidence-request',str(rp),'--learning-submission',str(sp),'--learning-submission-sha256',hashlib.sha256(raw).hexdigest()]
 env={k:v for k,v in os.environ.items()if k not in ('PYTHONPATH','PYTHONHOME')and not k.startswith('KWIP')}
 done=subprocess.run(args,cwd=tmp_path,env=env,capture_output=True,text=True)
 assert done.returncode==0,done.stdout+done.stderr
 evidence=json.loads(done.stdout.strip().splitlines()[-1])['evidence'];assert not evidence.get('backendError'),evidence
 assert len(evidence['fixtures'])==4 and evidence['submissionDigest']==hashlib.sha256(raw).hexdigest()
 for f in evidence['fixtures']:
  assert f['tensors']['parentByteDifference']['values']==[0]
  key='childUpdateDifference'if task==lesson.TASKS[0]else'continuationStateDifference'
  assert max(abs(v)for v in f['tensors'][key]['values'])<2e-6
 # Changing the registry cannot bypass the verified packaged inventory.
 changed=json.loads(registry.read_text());linear=next(c for c in changed['components']if c['id']=='linear');linear['fields'][0]['max']=1;registry.write_text(json.dumps(changed))
 tampered=subprocess.run(args,cwd=tmp_path,env=env,capture_output=True,text=True)
 assert tampered.returncode!=0
 # Even when legitimately resealed, construction must honor that selected registry.
 packager.refresh_worker_integrity(bundle)
 restricted=subprocess.run(args,cwd=tmp_path,env=env,capture_output=True,text=True)
 assert restricted.returncode==0,restricted.stdout+restricted.stderr
 rejected=json.loads(restricted.stdout.strip().splitlines()[-1])['evidence']
 assert rejected.get('backendError')and not rejected['fixtures'],rejected
