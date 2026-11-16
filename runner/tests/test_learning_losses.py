"""Independent scalar objectives plus finite-difference raw-score influence."""
import copy,importlib,itertools,math
import pytest
TASKS={'cross-entropy-class-choice':'loss_cross_entropy','bce-independent-labels':'loss_bce_with_logits','mse-error-size':'loss_mse','l1-absolute-error':'loss_l1','nll-log-probability-choice':'loss_nll','kl-distribution-comparison':'loss_kl_div','ctc-sequence-alignment':'loss_ctc','triplet-relative-distance':'loss_triplet_margin'}
CONFIG=[dict(reduction='mean',class_weights=[1,2,.5],label_smoothing=.2),dict(reduction='none',weight_mode='mapped',pos_weight_mode='configured',pos_weight_shape=[2],pos_weight_values=[2,3]),dict(reduction='sum',target_broadcast=True),dict(reduction='none'),dict(class_weights=[1,2,.5]),dict(reduction='batchmean',log_target=True),dict(zero_infinity=True,target_layout='packed'),dict(margin=.8,p=1.5,eps=0,swap=True)]
def cfg(task):return copy.deepcopy(CONFIG[list(TASKS).index(task)])
def roles(task):
 op=TASKS[task];return ['prediction'] if op=='loss_triplet_margin' else ['prediction','target','input_lengths','target_lengths'] if op=='loss_ctc' else ['prediction','target','weight'] if op=='loss_bce_with_logits' else ['prediction','target']
def sample(task):return [3,3] if TASKS[task]=='loss_ctc' else [3,2] if TASKS[task]=='loss_triplet_margin' else [3] if TASKS[task] in ('loss_cross_entropy','loss_nll','loss_kl_div') else [2]
def submission(task):
 op=TASKS[task];rr=roles(task);normal=op in ('loss_nll','loss_kl_div','loss_ctc')
 nodes=[dict(id=r,type='input',config={'fixture':r}) for r in rr]+[dict(id='recipe',type=op,config=cfg(task)),dict(id='check',type='objective_check',config={'scalarization':'mean'})]+[dict(id='out-'+r,type='output',config={'role':r}) for r in ('raw','objective','inputInfluence')]
 if normal:nodes.append(dict(id='log',type='log_softmax',config={'dim':-1}))
 edges=[dict(source='recipe',sourcePort='out',target='check',targetPort='objective')]+[dict(source='check',sourcePort=r,target='out-'+r,targetPort='in') for r in ('raw','objective','inputInfluence')]
 for r in rr:
  if r=='prediction' and normal:edges+=[dict(source=r,sourcePort='out',target='log',targetPort='in'),dict(source='log',sourcePort='out',target='check',targetPort=r)]
  else:edges.append(dict(source=r,sourcePort='out',target='check',targetPort=r))
 for i,n in enumerate(nodes):n.update(name=n['id'],x=i*80.,y=0.)
 for i,e in enumerate(edges):e['id']='wire'+str(i)
 return dict(schemaVersion=1,kind='lossObjectiveGraph',graph=dict(schemaVersion=2,name='Objective',nodes=nodes,edges=edges),objectiveSource=dict(domain='control',localID='recipe'),inputSources={r:dict(domain='model',localID=r) for r in rr})
def fixture(task,n=2,offset=0):
 op=TASKS[task];shape=[n]+sample(task);values=[((i*3+offset)%11-5)*.13+math.sin(i+offset)*.023 for i in range(math.prod(shape))]
 out={'prediction':dict(shape=shape,values=values)}
 def put(r,s,v):out[r]=dict(shape=s,values=v)
 if op in ('loss_cross_entropy','loss_nll'):put('target',[n],[0 if i%2==0 else (-100 if offset==3 else 2) for i in range(n)])
 elif op=='loss_bce_with_logits':put('target',[n,2],[v for _ in range(n) for v in (0.25,.75)]);put('weight',[n,2],[v for i in range(n) for v in (.5+i,.0 if i==1 else 2.)])
 elif op=='loss_mse':put('target',[n],[.2-.3*i for i in range(n)])
 elif op=='loss_l1':put('target',[n,2],[.2-.15*i for i in range(n*2)])
 elif op=='loss_kl_div':put('target',[n,3],[math.log(v) for _ in range(n) for v in (.2,.5,.3)])
 elif op=='loss_ctc':put('target',[n,2],[v for i in range(n) for v in ((1,2) if i%2==0 else (1,1))]);put('input_lengths',[n],[3 if i%2==0 else 2 for i in range(n)]);put('target_lengths',[n],[2]*n)
 return out

def expected(task,f):
 from runner.tests.test_loss_objectives import logs,ctc_oracle,triplet_oracle
 op=TASKS[task];x=f['prediction']['values'];shape=f['prediction']['shape'];n=shape[0];d=shape[-1];c=cfg(task);raw=[];denom=0
 for row in range(n):
  xx=x[row*math.prod(shape[1:]):(row+1)*math.prod(shape[1:])]
  y=f.get('target',{}).get('values',[])
  if op in ('loss_cross_entropy','loss_nll'):
   label=int(y[row]);w=c['class_weights'];lp=logs(xx)
   if label==-100:raw.append(0.);continue
   s=c.get('label_smoothing',0);raw.append(-(1-s)*w[label]*lp[label]-s*sum(a*b for a,b in zip(w,lp))/d);denom+=w[label]
  elif op=='loss_bce_with_logits':
   for j,a in enumerate(xx):
    yy=y[row*d+j];ww=f['weight']['values'][row*d+j];pw=c['pos_weight_values'][j]
    raw.append(ww*((1-yy)*max(a,0)+pw*yy*max(-a,0)+((1-yy)+pw*yy)*math.log1p(math.exp(-abs(a)))))
  elif op in ('loss_mse','loss_l1'):
   for j,a in enumerate(xx):
    yy=y[row] if op=='loss_mse' else y[row*d+j];raw.append((a-yy)**2 if op=='loss_mse' else abs(a-yy))
  elif op=='loss_kl_div':raw.extend(math.exp(yy)*(yy-a) for a,yy in zip(logs(xx),y[row*d:(row+1)*d]))
  elif op=='loss_ctc':
   time=int(f['input_lengths']['values'][row]);length=int(f['target_lengths']['values'][row]);labels=y[row*2:row*2+length];v=ctc_oracle([logs(xx[t*d:(t+1)*d]) for t in range(time)],labels,0);raw.append(v if math.isfinite(v) else 0.)
  else:raw+=triplet_oracle([[xx[:d],xx[d:2*d],xx[2*d:]]],{**c,'reduction':'none'})
 reduction=c.get('reduction','mean')
 if reduction=='sum':scalar=sum(raw)
 elif reduction=='none':scalar=sum(raw)/len(raw)
 elif reduction=='batchmean':scalar=sum(raw)/n
 elif op=='loss_ctc':scalar=sum(v/max(f['target_lengths']['values'][i],1) for i,v in enumerate(raw))/n
 else:scalar=sum(raw)/(denom if op in ('loss_cross_entropy','loss_nll') else len(raw))
 format=list(shape)+([0,1,2] if op=='loss_triplet_margin' else [])
 for r in roles(task)[1:]:format += [1 if op in ('loss_cross_entropy','loss_nll','loss_ctc') else 0,len(f[r]['shape'])]+f[r]['shape']+f[r]['values']
 return {'trainingObjective':dict(shape=[1],values=[scalar]),'evaluationObjective':dict(shape=[1],values=[scalar]),'targetContract':dict(shape=[len(format)],values=format),'reduction':dict(shape=[n] if op in ('loss_cross_entropy','loss_nll','loss_ctc','loss_triplet_margin') else shape,values=raw)}
def full_expected(task,f):
 a=expected(task,f);gradient=[]
 for i,v in enumerate(f['prediction']['values']):
  plus=copy.deepcopy(f);minus=copy.deepcopy(f);plus['prediction']['values'][i]=v+1e-5;minus['prediction']['values'][i]=v-1e-5
  gradient.append((expected(task,plus)['trainingObjective']['values'][0]-expected(task,minus)['trainingObjective']['values'][0])/2e-5)
 a['inputInfluence']=dict(shape=f['prediction']['shape'],values=gradient);return a
@pytest.mark.parametrize('task',TASKS)
@pytest.mark.parametrize('n,offset',[(1,0),(2,2),(3,3)])
def test_independent_all_scores_formats_and_every_raw_score_influence(task,n,offset):
 worker=importlib.import_module('runner.learning_losses');f=fixture(task,n,offset);e=dict(fixtures=[])
 worker.execute(submission(task),[dict(id='heldout',inputs=f)],e,lambda:False,lambda x:(x['shape'],x['values']),task=task)
 actual=e['fixtures'][0]['tensors'];want=full_expected(task,f);assert set(actual)==set(want)
 for k,v in want.items():assert actual[k]['shape']==v['shape'];assert actual[k]['values']==pytest.approx(v['values'],abs=2e-6,rel=2e-5)

@pytest.mark.parametrize('task',TASKS)
def test_actual_shared_policy_and_private_dispatch(task):
 from runner.learning_evidence import generate_evidence
 from runner.tests.test_learning_tensor_transforms import request
 r=request();r['assessmentID']='assessment.'+task;r['runtime']['capabilities']={TASKS[task]:{'major':1,'minor':0,'patch':0}};r['fixtures']=[dict(id='private-'+str(i),inputs=fixture(task,2,i)) for i in range(4)]
 e=generate_evidence(r,submission(task),submission_digest='b'*64)
 assert not e.get('backendError'),e
 assert e['submissionDigest']=='b'*64 and len(e['fixtures'])==4
 for f,a in zip(r['fixtures'],e['fixtures']):
  assert a['fixtureID']==f['id'];want=full_expected(task,f['inputs'])
  for k,v in want.items():assert a['tensors'][k]['values']==pytest.approx(v['values'],abs=2e-6,rel=2e-5)

@pytest.mark.parametrize('task',TASKS)
def test_wrong_valid_configuration_is_measured_and_fails_expected_scores(task):
 worker=importlib.import_module('runner.learning_losses');s=submission(task);op=TASKS[task];c=next(n for n in s['graph']['nodes'] if n['type']==op)['config']
 if op=='loss_cross_entropy':c['label_smoothing']=0
 elif op=='loss_bce_with_logits':c['pos_weight_values']=[1,1]
 elif op=='loss_mse':c['reduction']='mean'
 elif op=='loss_l1':next(n for n in s['graph']['nodes'] if n['type']=='objective_check')['config']['scalarization']='sum'
 elif op=='loss_nll':c['class_weights']=[]
 elif op=='loss_kl_div':c['reduction']='mean'
 elif op=='loss_ctc':c['reduction']='sum'
 else:c['margin']=.1
 f=fixture(task,2,2);e=dict(fixtures=[]);worker.execute(s,[dict(id='wrong-valid',inputs=f)],e,lambda:False,lambda x:(x['shape'],x['values']),task=task)
 assert e['fixtures'][0]['tensors']['trainingObjective']['values']!=pytest.approx(full_expected(task,f)['trainingObjective']['values'],abs=1e-6,rel=1e-5)

@pytest.mark.parametrize('task',TASKS)
@pytest.mark.parametrize('mutation',['answers','source','domain','edge','foreignNode','integerIdentity','scalarization','execution'])
def test_strict_submission_refuses_injected_answers_wrong_sources_and_topology(task,mutation):
 worker=importlib.import_module('runner.learning_losses');s=submission(task)
 if mutation=='answers':s['expected']={}
 elif mutation=='source':s['objectiveSource']['localID']='foreign'
 elif mutation=='domain':s['inputSources']['prediction']['domain']='control'
 elif mutation=='edge':s['graph']['edges'].pop()
 elif mutation=='foreignNode':s['graph']['nodes'][0]['type']='custom_code'
 elif mutation=='integerIdentity':s['schemaVersion']=True
 elif mutation=='scalarization':next(n for n in s['graph']['nodes'] if n['type']=='objective_check')['config']['scalarization']='automatic'
 else:s['graph']['execution']={}
 with pytest.raises(ValueError):worker.execute(s,[dict(id='f',inputs=fixture(task))],dict(fixtures=[]),lambda:False,lambda x:(x['shape'],x['values']),task=task)

@pytest.mark.parametrize('task',TASKS)
@pytest.mark.parametrize('mutation',['nonfinite','boolean','shape','answers','observations','empty','fractionalTarget'])
def test_fixture_contract_refuses_before_native_allocation(task,mutation,monkeypatch):
 import torch
 worker=importlib.import_module('runner.learning_losses');f=fixture(task)
 if mutation=='nonfinite':f['prediction']['values'][0]=float('nan')
 elif mutation=='boolean':f['prediction']['values'][0]=True
 elif mutation=='shape':f['prediction']['shape']=[2,6,6,2]
 elif mutation=='answers':f['gold']=dict(shape=[1],values=[1])
 elif mutation=='observations':f['prediction']['shape'][0]=4
 elif mutation=='empty':f['prediction']['shape']=[]
 elif TASKS[task] in ('loss_cross_entropy','loss_nll','loss_ctc'):f['target']['values'][0]=1.2
 else:f['prediction']['values'][0]='answer'
 def forbidden(*a,**kw):raise AssertionError('Invalid supplied fixture reached native allocation')
 monkeypatch.setattr(torch,'tensor',forbidden)
 with pytest.raises(ValueError):worker.execute(submission(task),[dict(id='f',inputs=f)],dict(fixtures=[]),lambda:False,lambda x:(x['shape'],x['values']),task=task)

@pytest.mark.parametrize('task',TASKS)
@pytest.mark.parametrize('mode',['guided','practice','assessment','cold'])
def test_actual_cli_changed_fixtures_and_cold_source_digest(tmp_path,task,mode):
 import hashlib,json,subprocess,sys
 from pathlib import Path
 from runner.tests.test_learning_tensor_transforms import request
 r=request();r['assessmentID']='assessment.'+task;r['runtime']['capabilities']={TASKS[task]:{'major':1,'minor':0,'patch':0}};r['fixtures']=[dict(id=mode+'-'+str(i),inputs=fixture(task,i%3+1,i+len(mode))) for i in range(4)]
 req=tmp_path/'request.json';sub=tmp_path/'submission.json';req.write_text(json.dumps(r));raw=json.dumps(submission(task),sort_keys=True).encode();sub.write_bytes(raw)
 done=subprocess.run([sys.executable,str(Path(__file__).parents[1]/'main.py'),'--learning-evidence-request',str(req),'--learning-submission',str(sub),'--learning-submission-sha256',hashlib.sha256(raw).hexdigest()],capture_output=True,text=True)
 assert done.returncode==0,done.stdout+done.stderr
 e=json.loads(done.stdout.strip().splitlines()[-1])['evidence'];assert not e.get('backendError'),e
 assert e['submissionDigest']==hashlib.sha256(raw).hexdigest() and len(e['fixtures'])==4
 for f,a in zip(r['fixtures'],e['fixtures']):
  assert a['fixtureID']==f['id'];wanted=full_expected(task,f['inputs']);assert set(a['tensors'])==set(wanted)
  for k,v in wanted.items():assert a['tensors'][k]['shape']==v['shape'];assert a['tensors'][k]['values']==pytest.approx(v['values'],abs=2e-6,rel=2e-5)

@pytest.mark.parametrize('task',TASKS)
def test_cancelled_before_any_result_and_mismatched_version_remains_unmeasured(task):
 worker=importlib.import_module('runner.learning_losses');e=dict(fixtures=[])
 with pytest.raises(ValueError,match='cancelled'):worker.execute(submission(task),[dict(id='f',inputs=fixture(task))],e,lambda:True,lambda x:(x['shape'],x['values']),task=task)
 assert e['fixtures']==[]
 from runner.learning_evidence import generate_evidence
 from runner.tests.test_learning_tensor_transforms import request
 r=request();r['assessmentID']='assessment.'+task;r['runtime']['capabilities']={TASKS[task]:{'major':1,'minor':1,'patch':0}};r['fixtures']=[dict(id='private',inputs=fixture(task))]
 e=generate_evidence(r,submission(task),submission_digest='c'*64);assert e['fixtures']==[] and e['runtime']['capabilities']!=r['runtime']['capabilities']
