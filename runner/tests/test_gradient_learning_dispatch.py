"""Actual assessment transport, runtime identity and packaged registry admission."""
import hashlib,json,subprocess,sys
from pathlib import Path
import pytest
from runner import learning_evidence as dispatch
from runner.tests.test_learning_gradients import TASKS,fixture,submission
from runner.learning_gradients import WIDTHS

CAPS={kind:{'major':1,'minor':0,'patch':0}for kind in ('gradient_policy','trainability','optimizer_sgd')}
def request(task):
 return {'assessmentID':'assessment.'+task,'assessmentVersion':{'major':1,'minor':0,'patch':0},'fixtureDigest':'a'*64,'runtime':{'backend':'pytorch','version':dispatch.runtime_versions()['pytorch'],'capabilities':CAPS},'budget':{'milliseconds':60000,'maximumTensorElements':1000000},'fixtures':[fixture(w,n)for n,w in enumerate(WIDTHS)]}

@pytest.mark.parametrize('task',TASKS)
def test_actual_cli_four_fixtures_exact_identity_and_no_answer_transport(tmp_path,task):
 raw=json.dumps(submission(task),sort_keys=True).encode();sp=tmp_path/'submission.json';sp.write_bytes(raw);rp=tmp_path/'request.json';rp.write_text(json.dumps(request(task)))
 done=subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'main.py'),'--learning-evidence-request',str(rp),'--learning-submission',str(sp),'--learning-submission-sha256',hashlib.sha256(raw).hexdigest()],capture_output=True,text=True)
 assert done.returncode==0,done.stdout+done.stderr
 evidence=json.loads(done.stdout.strip().splitlines()[-1])['evidence']
 assert not evidence.get('backendError'),evidence
 assert evidence['assessmentID']=='assessment.'+task and evidence['runtime']==request(task)['runtime']
 assert evidence['submissionDigest']==hashlib.sha256(raw).hexdigest() and evidence['fixtureDigest']=='a'*64
 assert len(evidence['fixtures'])==4 and evidence['elapsedMilliseconds']<60000 and not evidence['cancelled']
 for f in evidence['fixtures']:
  assert set(f)=={'fixtureID','tensors','partitions'} and f['partitions']=={} and len(f['tensors'])==5
  for key,v in f['tensors'].items():
   if key=='windowSchedule':assert v['values']==[1,2,4,1,1,1,1,2,2,2,4,3,2,1,1,4]
   elif key=='recordingAvailability':assert v['values']==[1,0,-1]
   else:assert max(map(abs,v['values']))<2e-6

@pytest.mark.parametrize('mutation',[lambda r:r['runtime']['capabilities'].pop('trainability'),lambda r:r['runtime']['capabilities']['gradient_policy'].update(minor=1),lambda r:r['assessmentVersion'].update(major=2)])
def test_missing_drifted_or_foreign_assessment_identity_refuses(mutation,monkeypatch):
 r=request(TASKS[0]);r['runtime']['capabilities']=json.loads(json.dumps(CAPS));mutation(r)
 monkeypatch.setattr(dispatch.learning_gradients,'execute',lambda *a,**k:pytest.fail('Mismatched identity reached execution.'))
 if r['assessmentVersion']['major']!=1:
  with pytest.raises(ValueError):dispatch.validate_request(r)
 else:
  evidence=dispatch.generate_evidence(r,submission(TASKS[0]),submission_digest='b'*64)
  assert evidence['fixtures']==[] and evidence['runtime']['capabilities']==CAPS

@pytest.mark.parametrize('task',TASKS)
def test_isolated_packaged_worker_uses_verified_registry_without_checkout(tmp_path,task):
 import importlib.util,os,shutil
 root=Path(__file__).resolve().parents[2];bundle=tmp_path/'KwipMLRuntime';bundle.mkdir()
 shutil.copy2(root/'scripts/runtime_launch.py',bundle/'launch.py')
 shutil.copytree(root/'runner',bundle/'runner',ignore=shutil.ignore_patterns('__pycache__','tests'))
 shutil.copy2(root/'Sources/KwipMLCore/Resources/components.json',bundle/'components.json')
 spec=importlib.util.spec_from_file_location('gradient_packager',root/'scripts/runtime_package.py');packager=importlib.util.module_from_spec(spec);spec.loader.exec_module(packager)
 (bundle/'runtime-manifest.json').write_text('{"schemaVersion":1}');packager.refresh_worker_integrity(bundle)
 assert not (bundle/'Sources').exists()
 raw=json.dumps(submission(task),sort_keys=True).encode();sp=tmp_path/'submission.json';sp.write_bytes(raw);rp=tmp_path/'request.json';rp.write_text(json.dumps(request(task)))
 env={k:v for k,v in os.environ.items()if k not in ('PYTHONPATH','PYTHONHOME')and not k.startswith('KWIP')}
 done=subprocess.run([sys.executable,'-I','-B',str(bundle/'launch.py'),'--learning-evidence-request',str(rp),'--learning-submission',str(sp),'--learning-submission-sha256',hashlib.sha256(raw).hexdigest()],cwd=tmp_path,env=env,capture_output=True,text=True)
 assert done.returncode==0,done.stdout+done.stderr
 evidence=json.loads(done.stdout.strip().splitlines()[-1])['evidence'];assert not evidence.get('backendError'),evidence
 assert len(evidence['fixtures'])==4 and evidence['submissionDigest']==hashlib.sha256(raw).hexdigest()
 assert evidence['runtime']['capabilities']==CAPS
