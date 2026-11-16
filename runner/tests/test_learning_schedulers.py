import copy
import math
import pytest
from runner import learning_schedulers
TASKS={'step-lr-intervals':'scheduler_step_lr','cosine-lr-curve':'scheduler_cosine_annealing_lr','linear-lr-warmup':'scheduler_linear_lr','plateau-lr-improvement':'scheduler_reduce_on_plateau','one-cycle-lr-batches':'scheduler_one_cycle_lr'}
SGD=dict(lr=.1,momentum=.5,dampening=0.,weight_decay=0.,nesterov=False,maximize=False)
CONFIGS={'scheduler_step_lr':dict(step_size=2,gamma=.5),'scheduler_cosine_annealing_lr':dict(T_max=4,eta_min=.02),'scheduler_linear_lr':dict(start_factor=.25,end_factor=1.,total_iters=3),'scheduler_reduce_on_plateau':dict(mode='min',factor=.5,patience=0,threshold=.05,threshold_mode='abs',cooldown=1,min_lr=.01,eps=1e-8,metric='validation_loss'),'scheduler_one_cycle_lr':dict(max_lr=.1,planned_total_batches=8,pct_start=.5,anneal_strategy='cos',cycle_momentum=False,base_momentum=.85,max_momentum=.95,div_factor=10.,final_div_factor=10.,three_phase=False)}
def tensor(v,s):return {'shape':s,'values':v}
def fixture(offset=0):return {'id':'case','inputs':{'initial':tensor([1+offset,-2,.5],[3]),'gradients':tensor([(.5 if i%2 else -.25)+offset+i*.05 for i in range(24)],[8,3]),'active':tensor([1,1,1,0,1,1,0,1,1,1,1,0,1,1,1,1],[8,2]),'epochOfBatch':tensor([1,1,2,2,3,3,4,4],[8]),'metrics':tensor([1+offset,1.1+offset,1.2+offset,.8+offset],[4])}}
def submission(component,config=None):
 return {'schemaVersion':1,'kind':'schedulerLearningGraph','graph':{'schemaVersion':7,'name':'Schedule adjustments','nodes':[{'id':'schedule','type':component,'name':'Rate schedule','x':0,'y':0,'config':copy.deepcopy(CONFIGS[component] if config is None else config)},{'id':'optimizer','type':'optimizer_sgd','name':'Supplied SGD','x':0,'y':100,'config':copy.deepcopy(SGD)},{'id':'train','type':'train','name':'Supplied training context','x':200,'y':0,'config':{}}],'edges':[{'id':'schedule-wire','source':'schedule','sourcePort':'out','target':'train','targetPort':'scheduler'},{'id':'optimizer-wire','source':'optimizer','sourcePort':'out','target':'train','targetPort':'optimizer'}]}}
def read_tensor(v):return v['shape'],v['values']
def execute(task,sub=None,f=None):
 e={'fixtures':[]};learning_schedulers.execute(sub or submission(TASKS[task]),[f or fixture()],e,lambda:False,read_tensor,task=task);return e['fixtures'][0]['tensors']
# Independent scalar reference. Never uses an upstream scheduler or production helper for expected rates.
def reference(component,c,f):
 w=f['inputs']['initial']['values'][:];m=[0.]*3;present=[False,False];out={};ticks=0;rate=.1;best=None;bad=0;cool=0
 if component.endswith('linear_lr'):rate=.1*c['start_factor']
 def cycle(t):
  end=c['pct_start']*c['planned_total_batches']-1;final=c['planned_total_batches']-1
  low=c['max_lr']/c['div_factor'];bottom=low/c['final_div_factor']
  if t<=end:start,finish,x=low,c['max_lr'],t/end
  else:start,finish,x=c['max_lr'],bottom,(t-end)/(final-end)
  return finish+(start-finish)*(1+math.cos(math.pi*x))/2 if c['anneal_strategy']=='cos' else start+(finish-start)*x
 if component.endswith('one_cycle_lr'):rate=cycle(0)
 def record(k,v):out.setdefault(k,[]).extend(v)
 for batch in range(8):
  used=rate
  for pi,indices in enumerate(((0,1),(2,))):
   if f['inputs']['active']['values'][batch*2+pi]==0:continue
   first=not present[pi];present[pi]=True
   for i in indices:
    g=f['inputs']['gradients']['values'][batch*3+i];m[i]=g if first else .5*m[i]+g;w[i]-=used*m[i]
  if component.endswith('one_cycle_lr'):ticks+=1;rate=cycle(ticks)
  elif batch%2==1:
   ticks+=1
   if component.endswith('step_lr'):rate=.1*c['gamma']**(ticks//c['step_size'])
   elif component.endswith('cosine_annealing_lr'):rate=c['eta_min']+(.1-c['eta_min'])*(1+math.cos(math.pi*ticks/c['T_max']))/2
   elif component.endswith('linear_lr'):rate=.1*(c['start_factor']+(c['end_factor']-c['start_factor'])*min(ticks,c['total_iters'])/c['total_iters'])
   else:
    metric=f['inputs']['metrics']['values'][ticks-1]
    if best is None or metric<best-c['threshold']:best=metric;bad=0
    else:bad+=1
    if cool>0:cool-=1;bad=0
    if bad>c['patience']:
     new=max(rate*c['factor'],c['min_lr'])
     if rate-new>c['eps']:rate=new
     cool=c['cooldown'];bad=0
  for key,value in [('usedRates',[used]),('nextRates',[rate]),('scheduleTicks',[ticks]),('updatedParameters',w),('momentumBuffer',m)]:record(key,value)
 for original,resumed in [('usedRates','resumedUsedRates'),('nextRates','resumedNextRates'),('scheduleTicks','resumedTicks'),('updatedParameters','resumedParameters'),('momentumBuffer','resumedMomentum')]:out[resumed]=out[original][len(out[original])//2:]
 return out
@pytest.mark.parametrize('task',TASKS)
@pytest.mark.parametrize('offset',[0,.25,-.25])
def test_native_rates_updates_timing_and_reconstructed_resume_match_scalar_oracle(task,offset):
 f=fixture(offset);a=execute(task,f=f);e=reference(TASKS[task],CONFIGS[TASKS[task]],f)
 assert set(a)==set(e)
 for key,values in e.items():assert a[key]['values']==pytest.approx(values,abs=1e-10,rel=1e-10),key
 assert a['scheduleTicks']['values']==([1,2,3,4,5,6,7,8] if task=='one-cycle-lr-batches' else [0,1,1,2,2,3,3,4])
@pytest.mark.parametrize('task',TASKS)
def test_wrong_config_is_measured_not_static_graded(task):
 c=copy.deepcopy(CONFIGS[TASKS[task]])
 key={'step-lr-intervals':'gamma','cosine-lr-curve':'eta_min','linear-lr-warmup':'start_factor','plateau-lr-improvement':'factor','one-cycle-lr-batches':'max_lr'}[task];c[key]*=.5
 assert execute(task,submission(TASKS[task],c))['updatedParameters']['values']!=pytest.approx(reference(TASKS[task],CONFIGS[TASKS[task]],fixture())['updatedParameters'])
@pytest.mark.parametrize('edit',[lambda s:s.update(answers={}),lambda s:s['graph']['nodes'][1]['config'].update(lr=.2),lambda s:s['graph']['nodes'][2]['config'].update(seed=41),lambda s:s['graph']['edges'][0].update(targetPort='optimizer'),lambda s:s['graph']['edges'].pop(),lambda s:s['graph']['nodes'][0]['config'].update(last_epoch=2),lambda s:s['graph'].update(schemaVersion=6)])
def test_refuses_authored_context_answers_wrong_wiring_or_schema(edit):
 s=submission('scheduler_step_lr');edit(s)
 with pytest.raises(ValueError):execute('step-lr-intervals',s)
@pytest.mark.parametrize('edit',[lambda f:f['inputs'].update(state=tensor([0],[1])),lambda f:f['inputs']['epochOfBatch']['values'].__setitem__(0,2),lambda f:f['inputs']['metrics']['values'].__setitem__(0,float('nan')),lambda f:f['inputs']['active']['values'].__setitem__(0,2),lambda f:f['inputs']['gradients'].update(shape=[4,6])])
def test_invalid_fixture_refuses_before_native_allocation(monkeypatch,edit):
 f=fixture();edit(f);monkeypatch.setattr(learning_schedulers.torch,'tensor',lambda *a,**k:pytest.fail('Invalid fixture allocated native tensor'))
 with pytest.raises(ValueError):execute('step-lr-intervals',f=f)
def test_cancelled_before_native_construction():
 with pytest.raises(ValueError,match='cancel'):learning_schedulers.execute(submission('scheduler_step_lr'),[fixture()],{'fixtures':[]},lambda:True,read_tensor,task='step-lr-intervals')
@pytest.mark.parametrize('task',TASKS)
def test_shared_policy_and_dispatch_preserve_exact_fixture_identity(task):
 from runner.learning_evidence import generate_evidence
 from runner.tests.test_learning_tensor_transforms import request
 r=request();r['assessmentID']='assessment.'+task;r['runtime']['capabilities']={TASKS[task]:{'major':1,'minor':0,'patch':0},'optimizer_sgd':{'major':1,'minor':0,'patch':0}};r['fixtures']=[fixture()]
 e=generate_evidence(r,submission(TASKS[task]),submission_digest='a'*64)
 assert not e.get('backendError'),e
 assert e['submissionDigest']=='a'*64 and e['fixtures'][0]['fixtureID']=='case'
 for key,v in reference(TASKS[task],CONFIGS[TASKS[task]],fixture()).items():assert e['fixtures'][0]['tensors'][key]['values']==pytest.approx(v,abs=1e-10)
@pytest.mark.parametrize('task',TASKS)
@pytest.mark.parametrize('mode',['guided','practice','assessment'])
def test_actual_cli_changed_sequences_and_resume_tables(tmp_path,task,mode):
 import json,hashlib,subprocess,sys
 from pathlib import Path
 from runner.tests.test_learning_tensor_transforms import request
 r=request();r['assessmentID']='assessment.'+task;r['runtime']['capabilities']={TASKS[task]:{'major':1,'minor':0,'patch':0},'optimizer_sgd':{'major':1,'minor':0,'patch':0}};r['fixtures']=[]
 for i in range(4):
  f=fixture(i*.125);f['id']=mode+'-'+str(i);f['inputs']['gradients']['values']=[g+(len(mode)+i)*.03125 for g in f['inputs']['gradients']['values']];r['fixtures'].append(f)
 req=tmp_path/'request.json';sub=tmp_path/'submission.json';req.write_text(json.dumps(r));raw=json.dumps(submission(TASKS[task]),sort_keys=True).encode();sub.write_bytes(raw)
 done=subprocess.run([sys.executable,str(Path(__file__).parents[1]/'main.py'),'--learning-evidence-request',str(req),'--learning-submission',str(sub),'--learning-submission-sha256',hashlib.sha256(raw).hexdigest()],capture_output=True,text=True)
 assert done.returncode==0,done.stdout+done.stderr
 e=json.loads(done.stdout.strip().splitlines()[-1])['evidence'];assert not e.get('backendError'),e
 assert e['submissionDigest']==hashlib.sha256(raw).hexdigest() and len(e['fixtures'])==4
 for f,a in zip(r['fixtures'],e['fixtures']):
  assert a['fixtureID']==f['id'] and a['partitions']=={}
  for key,v in reference(TASKS[task],CONFIGS[TASKS[task]],f).items():assert a['tensors'][key]['values']==pytest.approx(v,abs=1e-10)
@pytest.mark.parametrize('task',TASKS)
def test_private_policy_refuses_foreign_capability_and_authored_answer_fields(task):
 from runner.learning_evidence import generate_evidence
 from runner.tests.test_learning_tensor_transforms import request
 r=request();r['assessmentID']='assessment.'+task;r['runtime']['capabilities']={TASKS[task]:{'major':1,'minor':0,'patch':0}};r['fixtures']=[fixture()]
 e=generate_evidence(r,submission(TASKS[task]),submission_digest='a'*64)
 assert e['fixtures']==[] and e['runtime']['capabilities']!=r['runtime']['capabilities']
 r['runtime']['capabilities']['optimizer_sgd']={'major':1,'minor':0,'patch':0};s=submission(TASKS[task]);s['expected']={'usedRates':[.1]}
 e=generate_evidence(r,s,submission_digest='a'*64);assert e.get('backendError') and e['fixtures']==[]
def test_linear_onecycle_terminal_next_rate_is_measured_but_never_used():
 c=copy.deepcopy(CONFIGS['scheduler_one_cycle_lr']);c['anneal_strategy']='linear'
 actual=execute('one-cycle-lr-batches',submission('scheduler_one_cycle_lr',c))
 expected=reference('scheduler_one_cycle_lr',c,fixture())
 assert actual['usedRates']['values']==pytest.approx(expected['usedRates'])
 assert actual['nextRates']['values'][-1]<0
 assert all(x>=0 for x in actual['usedRates']['values'])
 assert actual['resumedNextRates']['values'][-1]==pytest.approx(actual['nextRates']['values'][-1])
