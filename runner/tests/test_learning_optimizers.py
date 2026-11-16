import copy
import math
import pytest
from runner import learning_optimizers

TASKS={'sgd-momentum-steps':'optimizer_sgd','adam-running-averages':'optimizer_adam','adamw-separate-decay':'optimizer_adamw','rmsprop-squared-memory':'optimizer_rmsprop','adagrad-accumulated-steps':'optimizer_adagrad'}
CONFIGS={
'optimizer_sgd':dict(lr=.1,momentum=.5,dampening=.25,weight_decay=.1),
'optimizer_adam':dict(lr=.1,beta1=.5,beta2=.75,eps=.01,weight_decay=.1,amsgrad=True),
'optimizer_adamw':dict(lr=.1,beta1=.5,beta2=.75,eps=.01,weight_decay=.1,amsgrad=True),
'optimizer_rmsprop':dict(lr=.1,alpha=.5,eps=.01,weight_decay=.1,momentum=.25,centered=True),
'optimizer_adagrad':dict(lr=.1,lr_decay=.2,initial_accumulator_value=.5,eps=.01,weight_decay=.1)}
def tensor(v,shape):return {'shape':shape,'values':v}
def fixture(offset=0):return {'id':'case','inputs':{'initial':tensor([1+offset,-2,.5],[3]),'gradients':tensor([.5,-.25,1,-1,.75,-.5,.25,1.5,.75,1,-.5,-1],[4,3]),'active':tensor([1,1,1,0,1,1,0,1],[4,2])}}
def submission(component,config=None):return {'schemaVersion':1,'kind':'optimizerLearningGraph','graph':{'schemaVersion':6,'name':'Update settings','nodes':[{'id':'recipe','type':component,'name':'Chosen optimizer','x':0,'y':0,'config':copy.deepcopy(config if config is not None else CONFIGS[component])},{'id':'train','type':'train','name':'Supplied training context','x':200,'y':0,'config':{}}],'edges':[{'id':'edge','source':'recipe','sourcePort':'out','target':'train','targetPort':'optimizer'}]}}
def read_tensor(v):return v['shape'],v['values']
def execute(task,sub=None,f=None):
 e={'fixtures':[],'maximumTensorElements':0};learning_optimizers.execute(sub or submission(TASKS[task]),[f or fixture()],e,lambda:False,read_tensor,task=task);return e['fixtures'][0]['tensors']
# Independent scalar loops, never native optimizer output or production adapter.
def reference(component,c,f):
 w=f['inputs']['initial']['values'][:];g=f['inputs']['gradients']['values'];active=f['inputs']['active']['values'];count=[0,0];m=[0.]*3;v=[0.]*3;mx=[0.]*3;ga=[0.]*3;acc=[c.get('initial_accumulator_value',0)]*3;present=[component.endswith('adagrad')]*2;out={}
 def record(key,value):out.setdefault(key,[]).extend(value)
 for t in range(4):
  for p,indices in enumerate(((0,1),(2,))):
   if not active[t*2+p]:continue
   count[p]+=1;first=not present[p];present[p]=True
   for i in indices:
    z=g[t*3+i];wd=c.get('weight_decay',0);lr=c['lr']
    if component.endswith('adamw'):w[i]*=1-lr*wd
    else:z+=wd*w[i]
    if component.endswith('sgd'):
     m[i]=z if first else c['momentum']*m[i]+(1-c['dampening'])*z;w[i]-=lr*m[i]
    elif component.endswith('adam') or component.endswith('adamw'):
     b1,b2=c['beta1'],c['beta2'];m[i]=b1*m[i]+(1-b1)*z;v[i]=b2*v[i]+(1-b2)*z*z;mx[i]=max(mx[i],v[i]);w[i]-=lr*m[i]/(1-b1**count[p])/(math.sqrt(mx[i]/(1-b2**count[p]))+c['eps'])
    elif component.endswith('rmsprop'):
     a=c['alpha'];v[i]=a*v[i]+(1-a)*z*z;ga[i]=a*ga[i]+(1-a)*z;m[i]=c['momentum']*m[i]+z/(math.sqrt(v[i]-ga[i]**2)+c['eps']);w[i]-=lr*m[i]
    else:acc[i]+=z*z;w[i]-=lr/(1+(count[p]-1)*c['lr_decay'])*z/(math.sqrt(acc[i])+c['eps'])
  record('updatedParameters',w);record('statePresence',[int(x) for x in present]);record('updateCounts',count)
  if component.endswith('sgd'):record('momentumBuffer',m)
  elif component.endswith('adam') or component.endswith('adamw'):
   record('firstAverage',m);record('squaredAverage',v);record('largestSquaredAverage',mx);record('stepCounts',count)
  elif component.endswith('rmsprop'):
   record('squaredAverage',v);record('gradientAverage',ga);record('momentumBuffer',m);record('stepCounts',count)
  else:record('accumulatedSquares',acc);record('stepCounts',count)
 return out
@pytest.mark.parametrize('task',TASKS)
@pytest.mark.parametrize('offset',[0,.75,-1.25])
def test_real_multistep_update_state_and_missing_gradients(task,offset):
 comp=TASKS[task];f=fixture(offset);actual=execute(task,f=f);expected=reference(comp,CONFIGS[comp],f)
 assert set(actual)==set(expected)
 for key,v in expected.items():assert actual[key]['values']==pytest.approx(v,abs=1e-10,rel=1e-10)
 assert actual['updateCounts']['values']==[1,1,2,1,3,2,3,3]
@pytest.mark.parametrize('task',TASKS)
def test_zero_rate_keeps_weights_but_measures_real_state(task):
 c=copy.deepcopy(CONFIGS[TASKS[task]]);c['lr']=0
 a=execute(task,submission(TASKS[task],c));assert a['updatedParameters']['values']==[1,-2,.5]*4
 assert a['updateCounts']['values']==[1,1,2,1,3,2,3,3]
@pytest.mark.parametrize('mutation',[lambda s:s.update(state={}),lambda s:s['graph']['nodes'][1]['config'].update(seed=41),lambda s:s['graph']['edges'][0].update(targetPort='in'),lambda s:s['graph'].update(schemaVersion=2),lambda s:s['graph']['nodes'][0]['config'].update(unknown=0),lambda s:s['graph']['edges'].clear()])
def test_rejects_untrusted_answers_context_and_wrong_wiring(mutation):
 s=submission('optimizer_sgd');mutation(s)
 with pytest.raises(ValueError):execute('sgd-momentum-steps',s)
def test_cancelled_before_construction():
 with pytest.raises(ValueError,match='cancel'):learning_optimizers.execute(submission('optimizer_sgd'),[fixture()],{'fixtures':[]},lambda:True,read_tensor,task='sgd-momentum-steps')
@pytest.mark.parametrize('task',TASKS)
def test_shared_dispatch_binds_optimizer_recipe_and_actual_update_evidence(task):
 from runner.learning_evidence import generate_evidence
 from runner.tests.test_learning_tensor_transforms import request
 r=request();r['assessmentID']='assessment.'+task;r['runtime']['capabilities']={TASKS[task]:{'major':1,'minor':0,'patch':0}};r['fixtures']=[fixture()]
 e=generate_evidence(r,submission(TASKS[task]),submission_digest='a'*64)
 assert not e.get('backendError'),e
 assert e['submissionDigest']=='a'*64 and e['fixtures'][0]['fixtureID']=='case'
 for key,v in reference(TASKS[task],CONFIGS[TASKS[task]],fixture()).items():assert e['fixtures'][0]['tensors'][key]['values']==pytest.approx(v,abs=1e-10)
@pytest.mark.parametrize('task',TASKS)
def test_wrong_same_shape_rate_runs_but_fails_independent_values(task):
 c=copy.deepcopy(CONFIGS[TASKS[task]]);c['lr']=.2
 actual=execute(task,submission(TASKS[task],c))
 assert actual['updatedParameters']['values']!=pytest.approx(reference(TASKS[task],CONFIGS[TASKS[task]],fixture())['updatedParameters'])
@pytest.mark.parametrize('task',TASKS)
@pytest.mark.parametrize('mode',['guided','practice','assessment'])
def test_actual_cli_four_changed_fixtures_identity_and_updates(tmp_path,task,mode):
 import subprocess,sys,json,hashlib
 from pathlib import Path
 from runner.tests.test_learning_tensor_transforms import request
 r=request();r['assessmentID']='assessment.'+task;r['runtime']['capabilities']={TASKS[task]:{'major':1,'minor':0,'patch':0}};r['fixtures']=[]
 for i in range(4):
  f=fixture(i*.25);f['id']=mode+'-'+str(i);f['inputs']['gradients']['values']=[g+(len(mode)+i)*.125 for g in f['inputs']['gradients']['values']];r['fixtures'].append(f)
 req=tmp_path/'request.json';sub=tmp_path/'submission.json';req.write_text(json.dumps(r));raw=json.dumps(submission(TASKS[task]),sort_keys=True).encode();sub.write_bytes(raw)
 done=subprocess.run([sys.executable,str(Path(__file__).parents[1]/'main.py'),'--learning-evidence-request',str(req),'--learning-submission',str(sub),'--learning-submission-sha256',hashlib.sha256(raw).hexdigest()],capture_output=True,text=True)
 assert done.returncode==0,done.stdout+done.stderr
 e=json.loads(done.stdout.strip().splitlines()[-1])['evidence'];assert not e.get('backendError'),e
 assert e['submissionDigest']==hashlib.sha256(raw).hexdigest() and len(e['fixtures'])==4
 for f,a in zip(r['fixtures'],e['fixtures']):
  assert a['fixtureID']==f['id'] and a['partitions']=={}
  for key,v in reference(TASKS[task],CONFIGS[TASKS[task]],f).items():assert a['tensors'][key]['values']==pytest.approx(v,abs=1e-10)
@pytest.mark.parametrize('mutation',[lambda f:f['inputs']['active']['values'].__setitem__(0,2),lambda f:f['inputs']['initial'].update(shape=[1,3]),lambda f:f['inputs'].update(state=tensor([0],[1])),lambda f:f['inputs']['gradients']['values'].__setitem__(0,float('nan'))])
def test_refuses_untrusted_or_invalid_fixture_before_native_allocation(monkeypatch,mutation):
 f=fixture();mutation(f)
 monkeypatch.setattr(learning_optimizers.torch,'tensor',lambda *a,**k:pytest.fail('Refused fixture allocated native parameter'))
 with pytest.raises(ValueError):execute('sgd-momentum-steps',f=f)
