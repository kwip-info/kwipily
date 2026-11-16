"""Independent scalar recurrence gold; native workers supply only measurements."""
import copy,importlib,math
import pytest
TASKS={'rnn-sequence-memory':'rnn','lstm-separate-memory':'lstm','gru-reset-memory':'gru','rnn-cell-step':'rnn_cell','lstm-cell-step':'lstm_cell','gru-cell-step':'gru_cell'}
def cfg(op):
 c=dict(input_size=2,hidden_size=2,bias=True,state_mode='supplied')
 if not op.endswith('_cell'):c.update(num_layers=1,bidirectional=False,dropout=0.)
 if op.startswith('rnn'):c['nonlinearity']='relu'
 if op=='lstm':c['proj_size']=0
 return c

def roles(op):return ['input','hidden','cell'] if op.startswith('lstm') else ['input','hidden']
def outroles(op):return ['result','hidden','cell'] if op.startswith('lstm') else ['result','hidden']
def shapes(op,n=2):
 cell=op.endswith('_cell');return {'input':[n,2] if cell else [n,3,2],'hidden':[n,2] if cell else [n,1,2],**({'cell':[n,2] if cell else [n,1,2]} if op.startswith('lstm') else {})}
def params(op):
 g=4 if op.startswith('lstm') else 3 if op.startswith('gru') else 1;suffix='' if op.endswith('_cell') else '_l0'
 return {name+suffix:shape for name,shape in [('weight_ih',[g*2,2]),('weight_hh',[g*2,2]),('bias_ih',[g*2]),('bias_hh',[g*2])]}
def fixture(task,n=2,offset=0):
 op=TASKS[task];f={k:{'shape':v,'values':[(i%7+1+offset)*.07 for i in range(math.prod(v))]} for k,v in shapes(op,n).items()}
 for j,(name,shape) in enumerate(params(op).items()):f['parameter.'+name]={'shape':shape,'values':[(i%5-2+offset)*.09+.03*j for i in range(math.prod(shape))]}
 outs={'result':shapes(op,n)['input'],'hidden':shapes(op,n)['hidden'],**({'cell':shapes(op,n)['cell']} if op.startswith('lstm') else {})}
 for role,shape in outs.items():f['upstream.'+role]={'shape':shape,'values':[(i%5-2)*.13 for i in range(math.prod(shape))]}
 return f

def submission(task):
 op=TASKS[task];nodes=[dict(id=r,type='input',config={'fixture':r}) for r in roles(op)]+[dict(id='op',type=op,config=cfg(op))]+[dict(id='output-'+r,type='output',config={'role':r}) for r in outroles(op)]
 edges=[dict(id='in-'+r,source=r,sourcePort='out',target='op',targetPort='in' if r=='input' else r) for r in roles(op)]+[dict(id='out-'+r,source='op',sourcePort='out' if r=='result' else r,target='output-'+r,targetPort='in') for r in outroles(op)]
 for i,node in enumerate(nodes):node.update(name=node['id'],x=i*80.,y=0.)
 return dict(schemaVersion=1,kind='recurrentGraph',graph=dict(schemaVersion=2,name='Remember patterns',nodes=nodes,edges=edges))

def scalar(op,f):
 cell=op.endswith('_cell');n=f['input']['shape'][0];steps=1 if cell else 3;g=4 if op.startswith('lstm') else 3 if op.startswith('gru') else 1;suffix='' if cell else '_l0'
 p={k:f['parameter.'+k+suffix]['values'] for k in ['weight_ih','weight_hh','bias_ih','bias_hh']};result=[];last=[];cells=[]
 for row in range(n):
  h=f['hidden']['values'][row*2:row*2+2];c=f['cell']['values'][row*2:row*2+2] if op.startswith('lstm') else []
  for t in range(steps):
   x=f['input']['values'][(row*steps+t)*2:(row*steps+t+1)*2]
   a=[sum(p['weight_ih'][j*2+k]*x[k] for k in range(2))+p['bias_ih'][j] for j in range(g*2)]
   b=[sum(p['weight_hh'][j*2+k]*h[k] for k in range(2))+p['bias_hh'][j] for j in range(g*2)]
   sig=lambda v:1/(1+math.exp(-v))
   if op.startswith('rnn'):h=[max(0,a[j]+b[j]) for j in range(2)]
   elif op.startswith('lstm'):
    c=[sig(a[2+j]+b[2+j])*c[j]+sig(a[j]+b[j])*math.tanh(a[4+j]+b[4+j]) for j in range(2)];h=[sig(a[6+j]+b[6+j])*math.tanh(c[j]) for j in range(2)]
   else:
    r=[sig(a[j]+b[j]) for j in range(2)];z=[sig(a[2+j]+b[2+j]) for j in range(2)];h=[(1-z[j])*math.tanh(a[4+j]+r[j]*b[4+j])+z[j]*h[j] for j in range(2)]
   result+=h
  last+=h;cells+=c
 return {'result':result,'hidden':last,**({'cell':cells} if c else {})}
def gold(op,f):
 output=scalar(op,f);answer=copy.deepcopy(output)
 def value():return sum(sum(v*w for v,w in zip(vals,f['upstream.'+r]['values'])) for r,vals in scalar(op,f).items())
 for key in [*roles(op),*['parameter.'+p for p in params(op)]]:
  values=f[key]['values'];gr=[]
  for i,v in enumerate(values):
   values[i]=v+1e-6;plus=value();values[i]=v-1e-6;minus=value();values[i]=v;gr.append((plus-minus)/2e-6)
  answer[key+'Gradient']=gr
 return answer

def run(task,f=None,s=None,stopped=lambda:False):
 worker=importlib.import_module('runner.learning_recurrent');e={'fixtures':[],'maximumTensorElements':0};worker.execute(s or submission(task),[dict(id='one',inputs=f or fixture(task))],e,stopped,lambda t:(t['shape'],t['values']),task=task);return e['fixtures'][0]['tensors']
@pytest.mark.parametrize('task',TASKS)
@pytest.mark.parametrize('n',[1,2,3])
def test_all_roles_and_independent_weighted_input_state_and_parameter_influence(task,n):
 f=fixture(task,n,1);expected=gold(TASKS[task],f);actual=run(task,f)
 assert set(actual)==set(expected)
 for key,wanted in expected.items():assert actual[key]['values']==pytest.approx(wanted,abs=2e-7,rel=2e-6),key
@pytest.mark.parametrize('task',TASKS)
@pytest.mark.parametrize('bad',['answers','parameters','unknown','role','port','missing','extra','stateShape','order','nonfinite','parameterShape','fixtureAnswer'])
def test_strict_graph_role_state_weight_and_cheat_refusals(task,bad):
 s=submission(task);f=fixture(task)
 if bad in ('answers','parameters'):s[bad]={}
 elif bad=='unknown':s['graph']['nodes'][len(roles(TASKS[task]))]['config']['inplace']=True
 elif bad=='role':s['graph']['nodes'][-1]['config']['role']='result'
 elif bad=='port':s['graph']['edges'][-1]['sourcePort']='in'
 elif bad=='missing':s['graph']['edges'].pop()
 elif bad=='extra':s['graph']['nodes'].append(dict(id='unused',type='input',name='Unused',x=0,y=0,config={'fixture':'hidden'}))
 elif bad=='stateShape':f['hidden']['shape']=[2,2,1]
 elif bad=='order':s['graph']['edges'][0]['source']=s['graph']['edges'][1]['source']
 elif bad=='nonfinite':f['input']['values'][0]=math.nan
 elif bad=='parameterShape':f[next(k for k in f if k.startswith('parameter.'))]['shape']=[1]
 else:f['expected']={'shape':[1],'values':[0]}
 with pytest.raises(ValueError):run(task,f,s)
@pytest.mark.parametrize('task',TASKS)
def test_wrong_settings_are_measured_and_do_not_pass_as_correct(task):
 s=submission(task);node=next(n for n in s['graph']['nodes'] if n['id']=='op');node['config']['bias']=False
 actual=run(task,s=s);expected=gold(TASKS[task],fixture(task));assert any(actual[r]['values']!=pytest.approx(expected[r],abs=1e-6) for r in outroles(TASKS[task]))
@pytest.mark.parametrize('task',TASKS)
def test_cancel_before_measurement(task):
 with pytest.raises(ValueError):run(task,stopped=lambda:True)
@pytest.mark.parametrize('task',TASKS)
def test_shared_policy_identity_and_real_native_dispatch(task):
 from runner.learning_evidence import generate_evidence
 from runner.tests.test_learning_tensor_transforms import request
 r=request();r['assessmentID']='assessment.'+task;r['runtime']['capabilities']={TASKS[task]:{'major':1,'minor':0,'patch':0}};r['fixtures']=[dict(id='recurrent-shared',inputs=fixture(task))]
 e=generate_evidence(r,submission(task),submission_digest='a'*64)
 assert not e.get('backendError'),e
 assert e['submissionDigest']=='a'*64 and e['fixtures'][0]['fixtureID']=='recurrent-shared'
 for key,v in gold(TASKS[task],fixture(task)).items():assert e['fixtures'][0]['tensors'][key]['values']==pytest.approx(v,abs=2e-7,rel=2e-6)
@pytest.mark.parametrize('task',TASKS)
@pytest.mark.parametrize('mode',['guided','practice','assessment'])
def test_actual_cli_four_changed_fixture_roles_and_all_owned_gradients(tmp_path,task,mode):
 import json,hashlib,subprocess,sys
 from pathlib import Path
 from runner.tests.test_learning_tensor_transforms import request
 r=request();r['assessmentID']='assessment.'+task;r['runtime']['capabilities']={TASKS[task]:{'major':1,'minor':0,'patch':0}};r['fixtures']=[dict(id=mode+'-'+str(i),inputs=fixture(task,i%3+1,i+len(mode))) for i in range(4)]
 req=tmp_path/'request.json';sub=tmp_path/'submission.json';req.write_text(json.dumps(r));raw=json.dumps(submission(task),sort_keys=True).encode();sub.write_bytes(raw)
 done=subprocess.run([sys.executable,str(Path(__file__).parents[1]/'main.py'),'--learning-evidence-request',str(req),'--learning-submission',str(sub),'--learning-submission-sha256',hashlib.sha256(raw).hexdigest()],capture_output=True,text=True)
 assert done.returncode==0,done.stdout+done.stderr
 e=json.loads(done.stdout.strip().splitlines()[-1])['evidence'];assert not e.get('backendError'),e
 assert e['submissionDigest']==hashlib.sha256(raw).hexdigest() and len(e['fixtures'])==4
 for f,a in zip(r['fixtures'],e['fixtures']):
  assert a['fixtureID']==f['id'] and a['partitions']=={}
  wanted=gold(TASKS[task],f['inputs']);assert set(a['tensors'])==set(wanted)
  for key,v in wanted.items():assert a['tensors'][key]['values']==pytest.approx(v,abs=2e-7,rel=2e-6)
@pytest.mark.parametrize('task',TASKS)
def test_foreign_private_identity_and_authored_answers_are_not_measured(task):
 from runner.learning_evidence import generate_evidence
 from runner.tests.test_learning_tensor_transforms import request
 r=request();r['assessmentID']='assessment.'+task;r['runtime']['capabilities']={'linear':{'major':1,'minor':0,'patch':0}};r['fixtures']=[dict(id='private',inputs=fixture(task))]
 e=generate_evidence(r,submission(task),submission_digest='a'*64);assert e['fixtures']==[] and e['runtime']['capabilities']!=r['runtime']['capabilities']
 r['runtime']['capabilities']={TASKS[task]:{'major':1,'minor':0,'patch':0}};s=submission(task);s['expected']={}
 e=generate_evidence(r,s,submission_digest='a'*64);assert e.get('backendError') and e['fixtures']==[]
@pytest.mark.parametrize('bad',['stateShape','parameterShape','nonfinite'])
def test_invalid_fixture_refused_before_native_tensor_allocation(monkeypatch,bad):
 worker=importlib.import_module('runner.learning_recurrent');f=fixture('lstm-separate-memory')
 if bad=='stateShape':f['hidden']['shape']=[2,2,1]
 elif bad=='parameterShape':f['parameter.weight_ih_l0']['shape']=[1]
 else:f['cell']['values'][0]=math.inf
 monkeypatch.setattr(worker.torch,'tensor',lambda *a,**kw:pytest.fail('Invalid fixture allocated tensor'))
 monkeypatch.setattr(worker.torch.nn,'LSTM',lambda *a,**kw:pytest.fail('Invalid fixture constructed recurrent parameters'))
 with pytest.raises(ValueError):run('lstm-separate-memory',f)

def test_gru_reset_applies_after_recurrent_candidate_bias_not_before_matrix_product():
 f=fixture('gru-cell-step',1,1);expected=scalar('gru_cell',f)['result'];wrong=copy.deepcopy(f)
 # Make the two reset gates different, and candidate recurrent cross-channel weights nonzero.
 f['parameter.bias_ih']['values'][:2]=[-1.1,.9];f['parameter.weight_hh']['values'][8:]=[.5,-.4,.3,.7]
 expected=scalar('gru_cell',f)['result'];actual=run('gru-cell-step',f)['result']['values'];assert actual==pytest.approx(expected,abs=1e-12)
 p={k:f['parameter.'+k]['values'] for k in ['weight_ih','weight_hh','bias_ih','bias_hh']};h=f['hidden']['values'];x=f['input']['values'];sig=lambda v:1/(1+math.exp(-v))
 r=[sig(sum(p['weight_ih'][j*2+k]*x[k]+p['weight_hh'][j*2+k]*h[k] for k in range(2))+p['bias_ih'][j]+p['bias_hh'][j]) for j in range(2)]
 z=[sig(sum(p['weight_ih'][(2+j)*2+k]*x[k]+p['weight_hh'][(2+j)*2+k]*h[k] for k in range(2))+p['bias_ih'][2+j]+p['bias_hh'][2+j]) for j in range(2)]
 conventional=[(1-z[j])*math.tanh(sum(p['weight_ih'][(4+j)*2+k]*x[k]+p['weight_hh'][(4+j)*2+k]*r[k]*h[k] for k in range(2))+p['bias_ih'][4+j]+p['bias_hh'][4+j])+z[j]*h[j] for j in range(2)]
 assert actual!=pytest.approx(conventional,abs=1e-4)
