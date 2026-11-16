"""Independent scalar coordinate expectations; never native normalization for expected values."""
import copy,importlib,itertools,math
import pytest
TASKS={
 'batch-norm1d-current-group':('batch_norm1d',(3,4),False,False),
 'batch-norm1d-remembered-summary':('batch_norm1d',(3,4),True,True),
 'batch-norm2d-current-group':('batch_norm2d',(3,2,3),False,False),
 'batch-norm2d-remembered-summary':('batch_norm2d',(3,2,3),True,True),
 'instance-norm1d-individual-patterns':('instance_norm1d',(3,4),False,False),
 'instance-norm1d-remembered-summary':('instance_norm1d',(3,4),True,True),
 'instance-norm2d-individual-grids':('instance_norm2d',(3,2,3),False,False),
 'instance-norm2d-remembered-summary':('instance_norm2d',(3,2,3),True,True),
 'group-norm-channel-teams':('group_norm',(4,3),False,False),
 'group-norm-channel-adjustments':('group_norm',(4,3),False,True)}
def tensor(shape,shift=0):return {'shape':list(shape),'values':[((i*7)%29-14)/4+shift for i in range(math.prod(shape))]}
def case(task,n=2):
 op,sample,remembered,affine=TASKS[task];c=sample[0]
 config={'eps':.125,'affine':affine,'bias':True}
 config.update({'num_channels':c,'num_groups':2} if op=='group_norm' else {'num_features':c,'track_running_stats':remembered,'momentum':.25})
 if op.startswith('batch'):config['averaging']='moving'
 nodes=[{'id':'x','type':'input','config':{}},{'id':'norm','type':op,'config':config},{'id':'y','type':'output','config':{}}]
 graph={'schemaVersion':2,'name':'lesson','nodes':nodes,'edges':[{'id':'a','source':'x','sourcePort':'out','target':'norm','targetPort':'in'},{'id':'b','source':'norm','sourcePort':'out','target':'y','targetPort':'in'}]}
 params={'norm':{'weight':{'shape':[c],'values':[.5,-1,2,.75][:c]},'bias':{'shape':[c],'values':[1,-.5,.25,-1][:c]}}} if affine else {}
 s={'schemaVersion':1,'kind':'statefulNormalizationGraph','graph':graph,'parameters':params,'gradientNodeID':'norm' if affine else None}
 inputs={'x':tensor((n,*sample),2),'upstream':tensor((n,*sample),-.75)}
 if remembered:inputs.update(calibrationA=tensor((2,*sample),-3),calibrationB=tensor((3,*sample),7))
 return s,{'id':task+'-'+str(n),'inputs':inputs}
def parse(t):return t['shape'],t['values']
def groups(shape,op,group_count=2):
 n,c,*spatial=shape;s=math.prod(spatial)
 if op.startswith('batch'):return [[(a*c+ch)*s+j for a in range(n) for j in range(s)] for ch in range(c)]
 if op.startswith('instance'):return [[(a*c+ch)*s+j for j in range(s)] for a in range(n) for ch in range(c)]
 return [[(a*c+ch)*s+j for ch in range(g*c//group_count,(g+1)*c//group_count) for j in range(s)] for a in range(n) for g in range(group_count)]
def oracle(task,submission,fixture):
 op,sample,remembered,affine=TASKS[task];shape=fixture['inputs']['x']['shape'];x=fixture['inputs']['x']['values'];u=fixture['inputs']['upstream']['values'];c=shape[1];sp=math.prod(shape[2:]);gamma=submission['parameters'].get('norm',{}).get('weight',{}).get('values',[1]*c);beta=submission['parameters'].get('norm',{}).get('bias',{}).get('values',[0]*c)
 mean=[0.]*c;var=[1.]*c
 if remembered:
  for key in ('calibrationA','calibrationB'):
   t=fixture['inputs'][key];gs=groups(t['shape'],op);m=[0.]*c;v=[0.]*c
   for k,ids in enumerate(gs):
    avg=sum(t['values'][i] for i in ids)/len(ids);vv=sum((t['values'][i]-avg)**2 for i in ids)/(len(ids)-1);ch=k%c
    divisor=t['shape'][0] if op.startswith('instance') else 1;m[ch]+=avg/divisor;v[ch]+=vv/divisor
   mean=[.75*a+.25*b for a,b in zip(mean,m)];var=[.75*a+.25*b for a,b in zip(var,v)]
 y=[0.]*len(x);dx=y.copy();dg=[0.]*c;db=dg.copy()
 gs=groups(shape,op)
 for ids in gs:
  avg=sum(x[i] for i in ids)/len(ids);vv=sum((x[i]-avg)**2 for i in ids)/len(ids)
  z={};h={};inv={}
  for i in ids:
   ch=(i//sp)%c;mu=mean[ch] if remembered else avg;v=var[ch] if remembered else vv;inv[i]=1/math.sqrt(v+.125);z[i]=(x[i]-mu)*inv[i];h[i]=u[i]*gamma[ch];y[i]=z[i]*gamma[ch]+beta[ch];dg[ch]+=u[i]*z[i];db[ch]+=u[i]
  for i in ids:dx[i]=inv[i]*h[i] if remembered else inv[i]*(h[i]-sum(h.values())/len(ids)-z[i]*sum(h[j]*z[j] for j in ids)/len(ids))
 out={'output':{'shape':shape,'values':y},'inputGradient':{'shape':shape,'values':dx},'evaluationOutput':{'shape':shape,'values':y}}
 if affine:out.update(scaleGradient={'shape':[c],'values':dg},biasGradient={'shape':[c],'values':db})
 if remembered:
  for prefix in ('','postEval'):
   out[prefix+'RunningMean' if prefix else 'runningMean']={'shape':[c],'values':mean}
   out[prefix+'RunningVariance' if prefix else 'runningVariance']={'shape':[c],'values':var}
   out[prefix+'BatchesTracked' if prefix else 'batchesTracked']={'shape':[1],'values':[0 if op.startswith('instance') else 2]}
 return out
@pytest.mark.parametrize('task',TASKS)
@pytest.mark.parametrize('n',[1,2,3])
def test_actual_outputs_influence_and_trusted_state_match_independent_coordinates(task,n):
 worker=importlib.import_module('runner.learning_stateful_normalization');s,f=case(task,n);e={'maximumTensorElements':0,'fixtures':[]};worker.execute(s,[f],e,lambda:False,parse,task=task)
 row=e['fixtures'][0];assert set(row)=={'fixtureID','tensors','partitions'} and row['fixtureID']==f['id'] and row['partitions']=={}
 expected=oracle(task,s,f);assert set(row['tensors'])==set(expected)
 for key,want in expected.items():assert row['tensors'][key]['shape']==want['shape'];assert row['tensors'][key]['values']==pytest.approx(want['values'],rel=1e-10,abs=1e-10),key
@pytest.mark.parametrize('mutation',['answer','buffers','mode','seed','unknown','missing','parameterShape','target','nullMomentum','boolChannels','oneValue','extraFixture','nonfinite'])
def test_strict_state_parameter_and_fixture_admission_precedes_allocation(mutation,monkeypatch):
 worker=importlib.import_module('runner.learning_stateful_normalization');s,f=case('batch-norm1d-remembered-summary');node=s['graph']['nodes'][1]
 if mutation in ('answer','buffers','mode','seed'):s[mutation]={}
 elif mutation=='unknown':node['config']['unknown']=1
 elif mutation=='missing':s['parameters']={}
 elif mutation=='parameterShape':s['parameters']['norm']['weight']['shape']=[1,3]
 elif mutation=='target':s['gradientNodeID']='x'
 elif mutation=='nullMomentum':node['config']['momentum']=None
 elif mutation=='boolChannels':node['config']['num_features']=True
 elif mutation=='oneValue':f['inputs']['calibrationA']=tensor([1,3,1])
 elif mutation=='extraFixture':f['inputs']['expected']=tensor([1])
 else:f['inputs']['x']['values'][0]=float('nan')
 monkeypatch.setattr(worker.torch,'tensor',lambda *a,**k:pytest.fail('invalid admission allocated tensor'))
 with pytest.raises(ValueError):worker.execute(s,[f],{'maximumTensorElements':0,'fixtures':[]},lambda:False,parse,task='batch-norm1d-remembered-summary')
def test_cancelled_execution_has_no_partial_measurements():
 worker=importlib.import_module('runner.learning_stateful_normalization');s,f=case('group-norm-channel-teams');e={'maximumTensorElements':0,'fixtures':[]};worker.execute(s,[f],e,lambda:True,parse,task='group-norm-channel-teams');assert e['fixtures']==[]

@pytest.mark.parametrize('task',['batch-norm1d-remembered-summary','batch-norm2d-remembered-summary','instance-norm1d-remembered-summary','instance-norm2d-remembered-summary'])
def test_same_shape_wrong_update_share_measures_a_wrong_result(task):
 worker=importlib.import_module('runner.learning_stateful_normalization');s,f=case(task);expected=oracle(task,s,f);s['graph']['nodes'][1]['config']['momentum']=.75;e={'maximumTensorElements':0,'fixtures':[]};worker.execute(s,[f],e,lambda:False,parse,task=task)
 assert e['fixtures'][0]['tensors']['output']['values']!=pytest.approx(expected['output']['values'],abs=1e-8)
 assert e['fixtures'][0]['tensors']['runningMean']['values']!=pytest.approx(expected['runningMean']['values'],abs=1e-8)
@pytest.mark.parametrize('task',['group-norm-channel-teams','group-norm-channel-adjustments'])
def test_same_shape_wrong_group_count_measures_wrong_values_and_influence(task):
 worker=importlib.import_module('runner.learning_stateful_normalization');s,f=case(task);expected=oracle(task,s,f);s['graph']['nodes'][1]['config']['num_groups']=1;e={'maximumTensorElements':0,'fixtures':[]};worker.execute(s,[f],e,lambda:False,parse,task=task)
 for key in ('output','inputGradient'):assert e['fixtures'][0]['tensors'][key]['values']!=pytest.approx(expected[key]['values'],abs=1e-8)
@pytest.mark.parametrize('task',['batch-norm2d-current-group','batch-norm2d-remembered-summary','instance-norm2d-individual-grids','instance-norm2d-remembered-summary'])
def test_equivalent_noncontiguous_spatial_reordering_matches_independent_oracle(task):
 worker=importlib.import_module('runner.learning_stateful_normalization');s,f=case(task);expected=oracle(task,s,f)
 nodes=s['graph']['nodes'];nodes.extend([{'id':'before','type':'transpose','config':{'dim0':2,'dim1':3}},{'id':'after','type':'transpose','config':{'dim0':2,'dim1':3}}])
 path=['x','before','norm','after','y'];s['graph']['edges']=[{'id':str(i),'source':a,'sourcePort':'out','target':b,'targetPort':'in'} for i,(a,b) in enumerate(zip(path,path[1:]))]
 e={'maximumTensorElements':0,'fixtures':[]};worker.execute(s,[f],e,lambda:False,parse,task=task)
 for key,value in expected.items():assert e['fixtures'][0]['tensors'][key]['values']==pytest.approx(value['values'],rel=1e-10,abs=1e-10),key

def test_batch_statistics_share_observation_influence_but_instance_statistics_do_not():
 worker=importlib.import_module('runner.learning_stateful_normalization');results={}
 for task in ['batch-norm1d-current-group','instance-norm1d-individual-patterns']:
  s,f=case(task);base={'maximumTensorElements':0,'fixtures':[]};worker.execute(s,[f],base,lambda:False,parse,task=task)
  f['inputs']['x']['values'][12:]=[v+100 for v in f['inputs']['x']['values'][12:]];changed={'maximumTensorElements':0,'fixtures':[]};worker.execute(s,[f],changed,lambda:False,parse,task=task)
  results[task]=(base['fixtures'][0]['tensors']['output']['values'][:12],changed['fixtures'][0]['tensors']['output']['values'][:12])
 assert results['batch-norm1d-current-group'][0]!=pytest.approx(results['batch-norm1d-current-group'][1])
 assert results['instance-norm1d-individual-patterns'][0]==pytest.approx(results['instance-norm1d-individual-patterns'][1])

@pytest.mark.parametrize('task',TASKS)
def test_shared_dispatch_binds_exact_task_parameters_state_and_native_evidence(task):
 from runner.learning_evidence import generate_evidence
 from runner.tests.test_learning_tensor_transforms import request
 s,f=case(task);r=request();r['assessmentID']='assessment.'+task;r['runtime']['capabilities']={TASKS[task][0]:{'major':1,'minor':0,'patch':0}};r['fixtures']=[f]
 e=generate_evidence(r,s,submission_digest='a'*64);assert not e.get('backendError'),e
 assert e['submissionDigest']=='a'*64 and e['fixtures'][0]['fixtureID']==f['id'] and e['fixtures'][0]['partitions']=={}
 for key,value in oracle(task,s,f).items():assert e['fixtures'][0]['tensors'][key]['values']==pytest.approx(value['values'],abs=1e-10)
@pytest.mark.parametrize('task',TASKS)
@pytest.mark.parametrize('mode',['guided','practice','assessment'])
def test_actual_cli_four_changed_fixtures_preserve_exact_identity_state_and_influence(tmp_path,task,mode):
 import subprocess,sys,json,hashlib
 from pathlib import Path
 from runner.tests.test_learning_tensor_transforms import request
 s,_=case(task);r=request();r['assessmentID']='assessment.'+task;r['runtime']['capabilities']={TASKS[task][0]:{'major':1,'minor':0,'patch':0}};r['fixtures']=[]
 for i in range(4):
  _,f=case(task,i%3+1);f['id']=mode+'-'+str(i)
  for key in f['inputs']:f['inputs'][key]['values']=[v+(i+len(mode))/8 for v in f['inputs'][key]['values']]
  r['fixtures'].append(f)
 req=tmp_path/'request.json';sub=tmp_path/'submission.json';req.write_text(json.dumps(r));raw=json.dumps(s,sort_keys=True).encode();sub.write_bytes(raw)
 done=subprocess.run([sys.executable,str(Path(__file__).parents[1]/'main.py'),'--learning-evidence-request',str(req),'--learning-submission',str(sub),'--learning-submission-sha256',hashlib.sha256(raw).hexdigest()],capture_output=True,text=True)
 assert done.returncode==0,done.stdout+done.stderr
 e=json.loads(done.stdout.strip().splitlines()[-1])['evidence'];assert not e.get('backendError'),e
 assert e['submissionDigest']==hashlib.sha256(raw).hexdigest() and len(e['fixtures'])==4
 for f,actual in zip(r['fixtures'],e['fixtures']):
  assert set(actual)=={'fixtureID','tensors','partitions'} and actual['fixtureID']==f['id'] and actual['partitions']=={}
  for key,value in oracle(task,s,f).items():assert actual['tensors'][key]['values']==pytest.approx(value['values'],abs=1e-10)
