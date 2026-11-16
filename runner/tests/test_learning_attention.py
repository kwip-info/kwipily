"""Independent scalar attention/residual/normalization/feedforward oracle."""
import copy,math,importlib
import pytest
TASKS={'multihead-attention-connections':'multihead_attention','transformer-source-target':'transformer','transformer-encoder-context':'transformer_encoder','transformer-decoder-memory':'transformer_decoder','scaled-attention-similarity':'scaled_dot_product_attention'}
def cfg(op):
 if op=='multihead_attention':return dict(embed_dim=2,num_heads=1,dropout=0.,bias=True,kdim=2,vdim=2,add_bias_kv=False,add_zero_attn=False,return_weights=True,average_weights=False,attention_mask_mode='supplied',padding_mask_mode='supplied')
 if op=='scaled_dot_product_attention':return dict(dropout=0.,scale=None,mask_mode='supplied')
 c=dict(d_model=2,nhead=1,dim_feedforward=3,dropout=0.,activation='gelu',layer_norm_eps=1e-5,norm_first=True,bias=True)
 if op=='transformer':c.update(num_encoder_layers=1,num_decoder_layers=1,source_mask_mode='none',target_mask_mode='causal',memory_mask_mode='supplied',source_padding_mode='none',target_padding_mode='none',memory_padding_mode='none')
 elif op=='transformer_encoder':c.update(num_layers=1,final_norm=True,mask_mode='supplied',padding_mask_mode='none')
 else:c.update(num_layers=1,final_norm=True,target_mask_mode='causal',memory_mask_mode='supplied',target_padding_mode='none',memory_padding_mode='none')
 return c
MASK=dict(mode='blocked',blocked_queries=[0],blocked_keys=[1],padding_keys=[])
def roles(op):
 return ['query','key','value'] if op in ('multihead_attention','scaled_dot_product_attention') else ['source','target'] if op=='transformer' else ['in'] if op=='transformer_encoder' else ['target','memory']
def samples(op):
 if op=='scaled_dot_product_attention':return dict(query=[1,2,2],key=[1,3,2],value=[1,3,2])
 if op=='multihead_attention':return dict(query=[2,2],key=[3,2],value=[3,2])
 if op=='transformer':return dict(source=[3,2],target=[2,2])
 if op=='transformer_encoder':return {'in':[3,2]}
 return dict(target=[2,2],memory=[3,2])
def outs(op):return dict(result=[2,2],weights=[1,2,3]) if op=='multihead_attention' else dict(result=[1,2,2]) if op=='scaled_dot_product_attention' else dict(result=[3,2] if op=='transformer_encoder' else [2,2])
def pshapes(op):
 def mha(prefix=''):return {prefix+'in_proj_weight':[6,2],prefix+'in_proj_bias':[6],prefix+'out_proj.weight':[2,2],prefix+'out_proj.bias':[2]}
 def layer(prefix,decoder=False):
  d=mha(prefix+'self_attn.')
  if decoder:d.update(mha(prefix+'multihead_attn.'))
  d.update({prefix+'linear1.weight':[3,2],prefix+'linear1.bias':[3],prefix+'linear2.weight':[2,3],prefix+'linear2.bias':[2]})
  for i in range(1,4 if decoder else 3):d[prefix+f'norm{i}.weight']=[2];d[prefix+f'norm{i}.bias']=[2]
  return d
 if op=='multihead_attention':return mha()
 if op=='scaled_dot_product_attention':return {}
 if op=='transformer':return {**layer('encoder.layers.0.'),**layer('decoder.layers.0.',True),'encoder.norm.weight':[2],'encoder.norm.bias':[2],'decoder.norm.weight':[2],'decoder.norm.bias':[2]}
 return {**layer('layers.0.',op=='transformer_decoder'),'norm.weight':[2],'norm.bias':[2]}
def fixture(task,n=1,offset=0):
 op=TASKS[task];f={k:dict(shape=[n]+v,values=[(i%7-3+offset)*.14 for i in range(n*math.prod(v))]) for k,v in samples(op).items()}
 for j,(k,s) in enumerate(sorted(pshapes(op).items())):f['parameter.'+k]=dict(shape=s,values=[(1.+(i%3)*.04) if 'norm' in k and k.endswith('weight') else (i%5-2)*.11+.007*j+.013*offset for i in range(math.prod(s))])
 for r,s in outs(op).items():f['upstream.'+r]=dict(shape=[n]+s,values=[(i%7-3)*.17+.02*offset for i in range(n*math.prod(s))])
 return f

def scalar(op,f):
 n=f[roles(op)[0]]['shape'][0];p={k.removeprefix('parameter.'):v['values'] for k,v in f.items() if k.startswith('parameter.')}
 def lin(x,w,b=None):
  out=len(b) if b is not None else len(w)//len(x[0]);return [[sum(w[j*len(row)+k]*row[k] for k in range(len(row)))+(b[j] if b else 0.) for j in range(out)] for row in x]
 def norm(x,prefix):
  out=[]
  for row in x:
   mean=sum(row)/len(row);var=sum((v-mean)**2 for v in row)/len(row);out.append([(v-mean)/math.sqrt(var+1e-5)*p[prefix+'weight'][i]+p[prefix+'bias'][i] for i,v in enumerate(row)])
  return out
 def add(a,b):return [[x+y for x,y in zip(ar,br)] for ar,br in zip(a,b)]
 def attn(q,k,v,prefix='',blocked=False,causal=False,native=True):
  if native:
   w=p[prefix+'in_proj_weight'];b=p[prefix+'in_proj_bias'];q=lin(q,w[:4],b[:2]);k=lin(k,w[4:8],b[2:4]);v=lin(v,w[8:],b[4:])
  weights=[];z=[]
  for i,row in enumerate(q):
   scores=[sum(a*b for a,b in zip(row,kr))/math.sqrt(2) for kr in k];allow=[not ((blocked and i==0 and j==1) or (causal and j>i)) for j in range(len(k))];high=max(s for s,a in zip(scores,allow) if a);ex=[math.exp(s-high) if a else 0. for s,a in zip(scores,allow)];total=sum(ex);wr=[a/total for a in ex];weights+=wr;z.append([sum(wr[j]*v[j][d] for j in range(len(v))) for d in range(2)])
  if native:z=lin(z,p[prefix+'out_proj.weight'],p[prefix+'out_proj.bias'])
  return z,weights
 def ff(x,pr):
  a=lin(x,p[pr+'linear1.weight'],p[pr+'linear1.bias']);a=[[.5*v*(1+math.erf(v/math.sqrt(2))) for v in row] for row in a];return lin(a,p[pr+'linear2.weight'],p[pr+'linear2.bias'])
 def enc(x,pr,blocked=False):
  a=norm(x,pr+'norm1.');x=add(x,attn(a,a,a,pr+'self_attn.',blocked=blocked)[0]);return add(x,ff(norm(x,pr+'norm2.'),pr))
 def dec(x,m,pr):
  a=norm(x,pr+'norm1.');x=add(x,attn(a,a,a,pr+'self_attn.',causal=True)[0]);x=add(x,attn(norm(x,pr+'norm2.'),m,m,pr+'multihead_attn.',blocked=True)[0]);return add(x,ff(norm(x,pr+'norm3.'),pr))
 answer={r:[] for r in outs(op)}
 for b in range(n):
  def source(key):
   raw=f[key]['values'];length=math.prod(samples(op)[key])//2;return [raw[(b*length+t)*2:(b*length+t+1)*2] for t in range(length)]
  if op in ('multihead_attention','scaled_dot_product_attention'):
   y,w=attn(source('query'),source('key'),source('value'),blocked=True,native=op=='multihead_attention')
   if op=='multihead_attention':answer['weights']+=w
  elif op=='transformer_encoder':y=norm(enc(source('in'),'layers.0.',True),'norm.')
  elif op=='transformer_decoder':y=norm(dec(source('target'),source('memory'),'layers.0.'),'norm.')
  else:m=norm(enc(source('source'),'encoder.layers.0.'),'encoder.norm.');y=norm(dec(source('target'),m,'decoder.layers.0.'),'decoder.norm.')
  answer['result'] += [v for row in y for v in row]
 return {r:dict(shape=[n]+s,values=answer[r]) for r,s in outs(op).items()}
def expected(op,f):
 a=scalar(op,f);scratch=copy.deepcopy(f)
 def weighted():return sum(sum(x*y for x,y in zip(t['values'],f['upstream.'+r]['values'])) for r,t in scalar(op,scratch).items())
 for key in roles(op)+['parameter.'+k for k in sorted(pshapes(op))]:
  values=[]
  for i,v in enumerate(f[key]['values']):scratch[key]['values'][i]=v+1e-6;plus=weighted();scratch[key]['values'][i]=v-1e-6;minus=weighted();scratch[key]['values'][i]=v;values.append((plus-minus)/2e-6)
  a[key+'Gradient']=dict(shape=f[key]['shape'],values=values)
 a['blockedConnections']=dict(shape=[f[roles(op)[0]]['shape'][0],3 if op=='transformer_encoder' else 2,3],values=[int(i==0 and j==1) for _ in range(f[roles(op)[0]]['shape'][0]) for i in range(3 if op=='transformer_encoder' else 2) for j in range(3)])
 a['blockedSourcePositions']=dict(shape=[f[roles(op)[0]]['shape'][0],3],values=[0]*(f[roles(op)[0]]['shape'][0]*3))
 return a

def submission(task):
 op=TASKS[task];rr=roles(op);q,k=('in','in') if op=='transformer_encoder' else ('target','source') if op=='transformer' else ('target','memory') if op=='transformer_decoder' else ('query','key')
 nodes=[dict(id=r,type='input',config={'fixture':r}) for r in rr]+[dict(id='mask',type='attention_mask',config=copy.deepcopy(MASK)),dict(id='op',type=op,config=cfg(op))]+[dict(id='output-'+r,type='output',config={'role':r}) for r in outs(op)]
 edges=[dict(source=r,sourcePort='out',target='op',targetPort=r) for r in rr]+[dict(source=q,sourcePort='out',target='mask',targetPort='query'),dict(source=k,sourcePort='out',target='mask',targetPort='key'),dict(source='mask',sourcePort='mask',target='op',targetPort='memory_mask' if op in ('transformer','transformer_decoder') else 'attn_mask' if op=='multihead_attention' else 'mask')]
 if op=='multihead_attention':edges.append(dict(source='mask',sourcePort='padding',target='op',targetPort='key_padding_mask'))
 edges += [dict(source='op',sourcePort='out' if r=='result' else r,target='output-'+r,targetPort='in') for r in outs(op)]
 for i,n in enumerate(nodes):n.update(name=n['id'],x=i*80.,y=0.)
 for i,e in enumerate(edges):e['id']='wire'+str(i)
 return dict(schemaVersion=1,kind='attentionGraph',graph=dict(schemaVersion=2,name='Compare useful values',nodes=nodes,edges=edges))
@pytest.mark.parametrize('task',TASKS)
@pytest.mark.parametrize('n',[1,2,3])
def test_measured_values_and_every_owned_influence_match_independent_scalar(task,n):
 worker=importlib.import_module('runner.learning_attention');f=fixture(task,n);e=dict(fixtures=[]);worker.execute(submission(task),[dict(id='fixture',inputs=f)],e,lambda:False,lambda x:(x['shape'],x['values']),task=task)
 gold=expected(TASKS[task],f);measured=e['fixtures'][0]['tensors'];assert set(measured)==set(gold)
 for k in gold:assert measured[k]['shape']==gold[k]['shape'];assert measured[k]['values']==pytest.approx(gold[k]['values'],abs=3e-6,rel=3e-5),k

@pytest.mark.parametrize('task',TASKS)
@pytest.mark.parametrize('mutation',range(14))
def test_strict_graph_roles_masks_and_authored_answers_refuse(task,mutation):
 worker=importlib.import_module('runner.learning_attention');s=submission(task)
 if mutation==0:s['answer']={}
 elif mutation==1:s['schemaVersion']=True
 elif mutation==2:s['kind']='recurrentGraph'
 elif mutation==3:s['graph']['execution']={}
 elif mutation==4:s['graph']['nodes'][0]['config']['seed']=7
 elif mutation==5:s['graph']['nodes'][0]['config']['fixture']='private-answer'
 elif mutation==6:s['graph']['nodes'][0]['x']=float('inf')
 elif mutation==7:s['graph']['nodes'][-1]['config']['role']='hidden'
 elif mutation==8:s['graph']['edges'][0]['targetPort']='value' if TASKS[task]=='multihead_attention' else 'unknown'
 elif mutation==9:s['graph']['edges'].pop()
 elif mutation==10:s['graph']['edges'].append(copy.deepcopy(s['graph']['edges'][0]))
 elif mutation==11:next(n for n in s['graph']['nodes'] if n['id']=='op')['config']['weights']={}
 elif mutation==12:next(n for n in s['graph']['nodes'] if n['id']=='mask')['config']['blocked_keys']=[999]
 else:next(n for n in s['graph']['nodes'] if n['id']=='mask')['config']['blocked_keys']=[1,1];next(n for n in s['graph']['nodes'] if n['id']=='mask')['config']['blocked_queries']=[0,0]
 with pytest.raises(ValueError):worker.execute(s,[dict(id='f',inputs=fixture(task))],dict(fixtures=[]),lambda:False,lambda x:(x['shape'],x['values']),task=task)
@pytest.mark.parametrize('task',TASKS)
def test_actual_wrong_blocked_connection_changes_measured_output_or_influence(task):
 worker=importlib.import_module('runner.learning_attention');s=submission(task);next(n for n in s['graph']['nodes'] if n['id']=='mask')['config']['blocked_queries']=[1]
 f=fixture(task,2);e=dict(fixtures=[]);worker.execute(s,[dict(id='f',inputs=f)],e,lambda:False,lambda x:(x['shape'],x['values']),task=task)
 gold=expected(TASKS[task],f);actual=e['fixtures'][0]['tensors'];assert any(any(abs(a-b)>3e-6 for a,b in zip(actual[k]['values'],v['values'])) for k,v in gold.items())
@pytest.mark.parametrize('task',TASKS)
def test_cancellation_does_not_measure(task):
 worker=importlib.import_module('runner.learning_attention');e=dict(fixtures=[])
 with pytest.raises(ValueError,match='cancelled'):worker.execute(submission(task),[dict(id='f',inputs=fixture(task))],e,lambda:True,lambda x:(x['shape'],x['values']),task=task)
 assert e['fixtures']==[]
@pytest.mark.parametrize('mutation',['wrongShape','nonfinite','authoredMask','unknownParameter','wrongUpstream','tooMany'])
def test_refusals_precede_native_allocation(monkeypatch,mutation):
 import torch
 worker=importlib.import_module('runner.learning_attention');task='multihead-attention-connections';f=fixture(task)
 if mutation=='wrongShape':f['query']['shape']=[1,2,3]
 elif mutation=='nonfinite':f['key']['values'][0]=float('nan')
 elif mutation=='authoredMask':f['mask']={'shape':[1,2,3],'values':[0]*6}
 elif mutation=='unknownParameter':f['parameter.other']=dict(shape=[1],values=[1])
 elif mutation=='wrongUpstream':f['upstream.weights']['shape']=[1,1,3,2]
 else:f['query']={'shape':[1000,2,2],'values':[0]*4000}
 def forbidden(*a,**kw):raise AssertionError('invalid fixture reached native allocation')
 monkeypatch.setattr(torch,'tensor',forbidden);monkeypatch.setattr(torch.nn,'MultiheadAttention',forbidden)
 with pytest.raises(ValueError):worker.execute(submission(task),[dict(id='f',inputs=f)],dict(fixtures=[]),lambda:False,lambda x:(x['shape'],x['values']),task=task)

def test_sdpa_owns_no_learned_tables_and_masks_have_no_gradient_roles():
 worker=importlib.import_module('runner.learning_attention');task='scaled-attention-similarity';f=fixture(task);e=dict(fixtures=[])
 worker.execute(submission(task),[dict(id='f',inputs=f)],e,lambda:False,lambda x:(x['shape'],x['values']),task=task)
 assert not any(k.startswith('parameter.') or k.startswith('mask') for k in e['fixtures'][0]['tensors'])

def test_mha_blocked_share_and_masked_value_influence_are_zero():
 worker=importlib.import_module('runner.learning_attention');task='multihead-attention-connections';f=fixture(task);f['upstream.result']['values'][2:]=[0,0];f['upstream.weights']['values']=[0]*6;e=dict(fixtures=[])
 worker.execute(submission(task),[dict(id='f',inputs=f)],e,lambda:False,lambda x:(x['shape'],x['values']),task=task)
 a=e['fixtures'][0]['tensors'];assert a['weights']['values'][1]==0;assert a['valueGradient']['values'][2:4]==[0,0]

def test_single_mask_sdpa_honors_utility_padding_union():
 worker=importlib.import_module('runner.learning_attention');task='scaled-attention-similarity';s=submission(task);next(n for n in s['graph']['nodes'] if n['id']=='mask')['config']['padding_keys']=[2]
 f=fixture(task);f['query']['values']=[0]*4;f['key']['values']=[0]*6;f['value']['values']=[1,2,7,8,5,6];e=dict(fixtures=[])
 worker.execute(s,[dict(id='f',inputs=f)],e,lambda:False,lambda x:(x['shape'],x['values']),task=task)
 assert e['fixtures'][0]['tensors']['result']['values']==pytest.approx([1,2,4,5],abs=1e-12)
@pytest.mark.parametrize('task',TASKS)
def test_fully_blocked_original_source_refuses_before_allocation(task,monkeypatch):
 import torch
 worker=importlib.import_module('runner.learning_attention');s=submission(task);next(n for n in s['graph']['nodes'] if n['id']=='mask')['config']['padding_keys']=[0,1,2]
 def forbidden(*a,**kw):raise AssertionError('fully blocked mask reached allocation')
 monkeypatch.setattr(torch,'tensor',forbidden)
 with pytest.raises(ValueError,match='allowed'):worker.execute(s,[dict(id='f',inputs=fixture(task))],dict(fixtures=[]),lambda:False,lambda x:(x['shape'],x['values']),task=task)
@pytest.mark.parametrize('task',TASKS)
def test_shared_policy_identity_and_measured_dispatch(task):
 from runner.learning_evidence import generate_evidence
 from runner.tests.test_learning_tensor_transforms import request
 r=request();r['assessmentID']='assessment.'+task;r['runtime']['capabilities']={k:{'major':1,'minor':0,'patch':0} for k in (TASKS[task],'attention_mask')};r['fixtures']=[dict(id='attention-shared',inputs=fixture(task))]
 e=generate_evidence(r,submission(task),submission_digest='a'*64)
 assert not e.get('backendError'),e
 assert e['submissionDigest']=='a'*64 and e['fixtures'][0]['fixtureID']=='attention-shared'
 for k,v in expected(TASKS[task],fixture(task)).items():assert e['fixtures'][0]['tensors'][k]['values']==pytest.approx(v['values'],abs=3e-6,rel=3e-5)
@pytest.mark.parametrize('task',TASKS)
@pytest.mark.parametrize('mode',['guided','practice','assessment'])
def test_actual_cli_four_changed_fixtures_and_exact_digest(tmp_path,task,mode):
 import json,hashlib,subprocess,sys
 from pathlib import Path
 from runner.tests.test_learning_tensor_transforms import request
 r=request();r['assessmentID']='assessment.'+task;r['runtime']['capabilities']={k:{'major':1,'minor':0,'patch':0} for k in (TASKS[task],'attention_mask')};r['fixtures']=[dict(id=mode+'-'+str(i),inputs=fixture(task,i%3+1,i+len(mode))) for i in range(4)]
 req=tmp_path/'request.json';sub=tmp_path/'submission.json';req.write_text(json.dumps(r));raw=json.dumps(submission(task),sort_keys=True).encode();sub.write_bytes(raw)
 done=subprocess.run([sys.executable,str(Path(__file__).parents[1]/'main.py'),'--learning-evidence-request',str(req),'--learning-submission',str(sub),'--learning-submission-sha256',hashlib.sha256(raw).hexdigest()],capture_output=True,text=True)
 assert done.returncode==0,done.stdout+done.stderr
 e=json.loads(done.stdout.strip().splitlines()[-1])['evidence'];assert not e.get('backendError'),e
 assert e['submissionDigest']==hashlib.sha256(raw).hexdigest() and len(e['fixtures'])==4
 for f,a in zip(r['fixtures'],e['fixtures']):
  assert a['fixtureID']==f['id'] and a['partitions']=={}
  wanted=expected(TASKS[task],f['inputs']);assert set(a['tensors'])==set(wanted)
  for k,v in wanted.items():assert a['tensors'][k]['values']==pytest.approx(v['values'],abs=3e-6,rel=3e-5)
@pytest.mark.parametrize('task',TASKS)
def test_foreign_or_missing_mask_capability_private_identity_is_unmeasured(task):
 from runner.learning_evidence import generate_evidence
 from runner.tests.test_learning_tensor_transforms import request
 r=request();r['assessmentID']='assessment.'+task;r['runtime']['capabilities']={TASKS[task]:{'major':1,'minor':0,'patch':0}};r['fixtures']=[dict(id='private',inputs=fixture(task))]
 e=generate_evidence(r,submission(task),submission_digest='a'*64);assert e['fixtures']==[] and e['runtime']['capabilities']!=r['runtime']['capabilities']
 r['runtime']['capabilities']['attention_mask']={'major':1,'minor':0,'patch':0};s=submission(task);s['expected']={}
 e=generate_evidence(r,s,submission_digest='a'*64);assert e['fixtures']==[] and e.get('backendError')

def test_shared_transport_preserves_old_sixteen_array_bound_and_exact_attention_keys():
 from runner.learning_evidence import validate_request
 from runner.tests.test_learning_tensor_transforms import request
 r=request();r['fixtures']=[dict(id='old',inputs={str(i):dict(shape=[1],values=[1]) for i in range(17)})]
 with pytest.raises(ValueError,match='fixture'):validate_request(r)
 task='transformer-source-target';r['assessmentID']='assessment.'+task;r['runtime']['capabilities']={k:{'major':1,'minor':0,'patch':0} for k in (TASKS[task],'attention_mask')};r['fixtures']=[dict(id='new',inputs=fixture(task))]
 assert len(r['fixtures'][0]['inputs'])==37;validate_request(r)
 r['fixtures'][0]['inputs']['answer']=dict(shape=[1],values=[1])
 with pytest.raises(ValueError,match='fixture'):validate_request(r)
 r['fixtures'][0]['inputs'].pop('answer');r['fixtures'][0]['inputs'].pop('parameter.encoder.norm.weight')
 with pytest.raises(ValueError,match='key set'):validate_request(r)
 r['fixtures']=[dict(id='new',inputs=fixture(task))];r['fixtures'][0]['inputs']['source']['shape']=[1,2,3]
 with pytest.raises(ValueError,match='geometry'):validate_request(r)

@pytest.mark.parametrize('task',TASKS)
def test_actual_utility_checkpoints_measure_changed_pair_and_padding(task):
 worker=importlib.import_module('runner.learning_attention');s=submission(task)
 mask=next(n for n in s['graph']['nodes'] if n['id']=='mask')
 mask['config']['blocked_queries']=[1];mask['config']['padding_keys']=[2]
 f=fixture(task,2);e={'fixtures':[]}
 worker.execute(s,[dict(id='f',inputs=f)],e,lambda:False,lambda x:(x['shape'],x['values']),task=task)
 actual=e['fixtures'][0]['tensors'];q=3 if TASKS[task]=='transformer_encoder' else 2
 expected=[int(j==2 or (i==1 and j==1)) for _ in range(2) for i in range(q) for j in range(3)]
 assert actual['blockedConnections']==dict(shape=[2,q,3],values=expected)
 assert actual['blockedSourcePositions']==dict(shape=[2,3],values=[0,0,1]*2)
 assert all(type(v) is int for key in ('blockedConnections','blockedSourcePositions') for v in actual[key]['values'])
