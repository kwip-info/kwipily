"""Independent reduction groups, weighted influence, typed buffers and persistence."""
import copy,itertools,json,math
from pathlib import Path
import torch,pytest
from runner.tensor_layout import BatchedLayout
from runner.stateful_normalization_configuration import normalize_stateful_normalization as normalize,resolve_stateful_normalization as resolve
OPS=['batch_norm1d','batch_norm2d','instance_norm1d','instance_norm2d','group_norm']

@pytest.mark.parametrize('op',OPS)
def test_exact_shapes_affine_and_buffer_inventory(op):
 sample=(2,3) if op.endswith('1d') or op=='group_norm' else (2,2,3);c=normalize(op,{'affine':True,'bias':False,**({} if op=='group_norm' else {'track_running_stats':True})});r=resolve(op,c,BatchedLayout(sample));assert r['layout'].sample==sample;assert r['parameterCount']==2;assert set(r['buffers'])==(set() if op=='group_norm' else {'running_mean','running_var','num_batches_tracked'})
 if op!='group_norm':assert r['buffers']['num_batches_tracked']=={'dtype':'int64','shape':[]}

@pytest.mark.parametrize('op,c,sample',[('batch_norm1d',{'num_features':3},(2,3)),('instance_norm1d',{},(2,3,4)),('batch_norm2d',{},(2,3)),('group_norm',{'num_groups':3,'num_channels':4},(4,2)),('group_norm',{'num_groups':2},(2,))])
def test_incompatible_channel_rank_group_refusal(op,c,sample):
 with pytest.raises(ValueError):resolve(op,c,BatchedLayout(sample))

def test_declared_defaults_and_inactive_settings_retained():
 assert normalize('batch_norm1d',{'averaging':'cumulative','momentum':.2})['momentum']==.2
 assert normalize('instance_norm1d',{})['affine'] is False
 assert normalize('group_norm',{'affine':False,'bias':True})['bias'] is True

def coordinate_reference(kind,shape,values,up,gamma,beta,eps,groups=2,stats=None):
 n,c=shape[:2];s=math.prod(shape[2:]);sets=[]
 if stats is not None:
  means,variances=stats;norm=[(v-means[(i//s)%c])/math.sqrt(variances[(i//s)%c]+eps) for i,v in enumerate(values)];out=[norm[i]*gamma[(i//s)%c]+beta[(i//s)%c] for i in range(len(values))];dx=[up[i]*gamma[(i//s)%c]/math.sqrt(variances[(i//s)%c]+eps) for i in range(len(values))]
 else:
  if kind.startswith('batch'):sets=[[((b*c+ch)*s)+k for b in range(n) for k in range(s)] for ch in range(c)]
  elif kind.startswith('instance'):sets=[[((b*c+ch)*s)+k for k in range(s)] for b in range(n) for ch in range(c)]
  else:sets=[[(b*c+ch)*s+k for ch in range(g*(c//groups),(g+1)*(c//groups)) for k in range(s)] for b in range(n) for g in range(groups)]
  norm=[0.]*len(values);out=[0.]*len(values);dx=[0.]*len(values)
  for indices in sets:
   count=len(indices);mean=math.fsum(values[i] for i in indices)/count;var=math.fsum((values[i]-mean)**2 for i in indices)/count;inv=1/math.sqrt(var+eps);q=[(values[i]-mean)*inv for i in indices];u=[up[i]*gamma[(i//s)%c] for i in indices];avg=math.fsum(u)/count;cov=math.fsum(a*b for a,b in zip(u,q))/count
   for i,a,b in zip(indices,q,u):norm[i]=a;out[i]=a*gamma[(i//s)%c]+beta[(i//s)%c];dx[i]=inv*(b-avg-a*cov)
 dg=[math.fsum(up[i]*norm[i] for i in range(len(values)) if (i//s)%c==ch) for ch in range(c)];db=[math.fsum(up[i] for i in range(len(values)) if (i//s)%c==ch) for ch in range(c)]
 return out,dx,dg,db


CASES=[(op,track,averaging) for op in OPS for track,averaging in ([(True,'moving'),(True,'cumulative'),(False,'moving')] if op.startswith('batch') else [(True,None),(False,None)] if op.startswith('instance') else [(False,None)])]
@pytest.fixture(autouse=True)
def isolate_policy(request):
 cs=getattr(request.node,'callspec',None);device=None if cs is None else cs.params.get('device')
 if device not in ('cpu','mps'):yield;return
 prior=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled()
 try:torch.use_deterministic_algorithms(device=='cpu');yield
 finally:torch.use_deterministic_algorithms(prior,warn_only=warn)

def independent_running(kind,shape,values):
 n,c=shape[:2];s=math.prod(shape[2:]);means=[];variances=[]
 for ch in range(c):
  pieces=[[values[(b*c+ch)*s+i] for i in range(s)] for b in range(n)]
  if kind.startswith('batch'):pieces=[sum(pieces,[])]
  m=[math.fsum(p)/len(p) for p in pieces];v=[math.fsum((x-avg)**2 for x in p)/(len(p)-1) for p,avg in zip(pieces,m)];means.append(math.fsum(m)/len(m));variances.append(math.fsum(v)/len(v))
 return means,variances

@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('op,tracking,averaging',CASES)
@pytest.mark.parametrize('affine,bias',[(False,False),(False,True),(True,False),(True,True)])
def test_independent_training_eval_affine_influence_and_two_calibration_buffers(device,op,tracking,averaging,affine,bias):
 from runner.engine import _normalization_layer,_execute_stateful_normalization
 sample=(4,3) if op.endswith('1d') or op=='group_norm' else (4,2,3);shape=(3,*sample);c={'eps':.001,'affine':affine,'bias':bias}
 c.update({'num_channels':4,'num_groups':2} if op=='group_norm' else {'num_features':4,'momentum':.25,'track_running_stats':tracking})
 if averaging is not None:c['averaging']=averaging
 c=normalize(op,c);layer=_normalization_layer(op,c).to(device);values=[(i%17-8)/5 for i in range(math.prod(shape))];up=[(i%11-5)/7 for i in range(len(values))];gamma=[.5,-.75,1.25,.875] if affine else [1.]*4;beta=[.1,-.2,.3,-.4] if affine and bias else [0.]*4
 with torch.no_grad():
  if affine:layer.weight.copy_(torch.tensor(gamma,device=device))
  if affine and bias:layer.bias.copy_(torch.tensor(beta,device=device))
 def measure(training,stats):
  layer.train(training);x=torch.tensor(values,device=device).reshape(shape).requires_grad_();actual=_execute_stateful_normalization(x,op,c,sample,layer);reference=coordinate_reference(op,shape,values,up,gamma,beta,.001,stats=stats);torch.testing.assert_close(actual.cpu(),torch.tensor(reference[0]).reshape(shape),atol=2e-5,rtol=2e-5)
  refs=[reference[1]]+([reference[2]] if affine else [])+([reference[3]] if affine and bias else []);grads=torch.autograd.grad(actual,[x,*layer.parameters()],torch.tensor(up,device=device).reshape(shape))
  for a,b in zip(grads,refs):torch.testing.assert_close(a.cpu(),torch.tensor(b).reshape(a.shape),atol=2e-5,rtol=2e-5)
 measure(True,None)
 expected_mean,expected_var=([0.]*4,[1.]*4)
 if tracking:
  means,variances=independent_running(op,shape,values);weight=1. if averaging=='cumulative' else .25;expected_mean=[weight*v for v in means];expected_var=[1-weight+weight*v for v in variances];torch.testing.assert_close(layer.running_mean.cpu(),torch.tensor(expected_mean),atol=2e-6,rtol=2e-6);torch.testing.assert_close(layer.running_var.cpu(),torch.tensor(expected_var),atol=2e-6,rtol=2e-6);assert layer.num_batches_tracked.item()==(1 if op.startswith('batch') else 0)
 shifted=[v+2 for v in values];_execute_stateful_normalization(torch.tensor(shifted,device=device).reshape(shape),op,c,sample,layer)
 if tracking:
  means,variances=independent_running(op,shape,shifted);weight=.5 if averaging=='cumulative' else .25;expected_mean=[(1-weight)*old+weight*new for old,new in zip(expected_mean,means)];expected_var=[(1-weight)*old+weight*new for old,new in zip(expected_var,variances)];torch.testing.assert_close(layer.running_mean.cpu(),torch.tensor(expected_mean),atol=2e-6,rtol=2e-6);torch.testing.assert_close(layer.running_var.cpu(),torch.tensor(expected_var),atol=2e-6,rtol=2e-6);assert layer.num_batches_tracked.item()==(2 if op.startswith('batch') else 0)
 before={k:v.clone() for k,v in layer.state_dict().items()};measure(False,(expected_mean,expected_var) if tracking else None);assert all(torch.equal(before[k],v) for k,v in layer.state_dict().items())

@pytest.mark.parametrize('op',OPS)
@pytest.mark.parametrize('field,value',[('eps',True),('eps',0),('eps',float('nan')),('eps',float('inf')),('affine',1),('bias',0),('unknown',1)])
def test_strict_common_configuration_refuses(op,field,value):
 with pytest.raises(ValueError):normalize(op,{field:value})

@pytest.mark.parametrize('op',OPS[:-1])
@pytest.mark.parametrize('field,value',[('num_features',True),('num_features',2.5),('num_features',513),('momentum',None),('momentum',True),('momentum',-.1),('momentum',1.1),('track_running_stats',1)])
def test_strict_channel_moving_settings_refuse(op,field,value):
 with pytest.raises(ValueError):normalize(op,{field:value})

@pytest.mark.parametrize('op',OPS)
def test_schema_one_refuses(op):
 from runner.graph_semantics import effective_config
 with pytest.raises(ValueError,match='schema'):effective_config({'type':op,'config':{}},1)

@pytest.mark.parametrize('op',OPS)
@pytest.mark.parametrize('bad',['dtype','nonfinite','sparse','empty','shape','budget'])
def test_runtime_boundary_refuses_before_module_call(op,bad,monkeypatch):
 from runner.engine import _normalization_layer,_execute_stateful_normalization
 sample=(2,3) if op.endswith('1d') or op=='group_norm' else (2,2,3);c=normalize(op,{});layer=_normalization_layer(op,c);x=torch.ones(2,*sample)
 if bad=='dtype':x=x.double()
 elif bad=='nonfinite':x.flatten()[0]=float('nan')
 elif bad=='sparse':x=x.to_sparse()
 elif bad=='empty':x=x[:0]
 elif bad=='shape':x=x.flatten(1)
 else:x=x[:1].expand(100000001,*sample)
 calls=[];monkeypatch.setattr(layer,'forward',lambda *a:calls.append(True))
 with pytest.raises(ValueError):_execute_stateful_normalization(x,op,c,sample,layer)
 assert calls==[]

@pytest.mark.parametrize('op',OPS[:-1])
@pytest.mark.parametrize('field,value',[('running_var',-1.),('num_batches_tracked',-1),('num_batches_tracked',1000000001),('running_mean',float('nan'))])
def test_bad_running_state_refuses_before_update(op,field,value,monkeypatch):
 from runner.engine import _normalization_layer,_execute_stateful_normalization
 c=normalize(op,{'track_running_stats':True});layer=_normalization_layer(op,c);sample=(2,3) if op.endswith('1d') else (2,2,3)
 with torch.no_grad():getattr(layer,field).fill_(value)
 calls=[];monkeypatch.setattr(layer,'forward',lambda *a:calls.append(True))
 with pytest.raises(ValueError):_execute_stateful_normalization(torch.ones(2,*sample),op,c,sample,layer)
 assert calls==[]

@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('op',OPS[:-1])
def test_current_statistics_singleton_refuses_but_tracked_eval_preserves_state(device,op):
 from runner.engine import _normalization_layer,_execute_stateful_normalization
 sample=(2,) if op.startswith('batch') and op.endswith('1d') else (2,1) if op.endswith('1d') else (2,1,1);c=normalize(op,{'track_running_stats':True});layer=_normalization_layer(op,c).to(device);x=torch.ones(1,*sample,device=device);before={k:v.clone() for k,v in layer.state_dict().items()}
 with pytest.raises(ValueError,match='more than one'):_execute_stateful_normalization(x,op,c,sample,layer)
 assert all(torch.equal(before[k],v) for k,v in layer.state_dict().items());layer.eval();assert torch.isfinite(_execute_stateful_normalization(x,op,c,sample,layer)).all();assert all(torch.equal(before[k],v) for k,v in layer.state_dict().items())

@pytest.mark.parametrize('device',['cpu','mps'])
def test_untracked_batch_normalization_does_not_promise_microbatch_invariance(device):
 from runner.engine import _normalization_layer,_execute_stateful_normalization
 c=normalize('batch_norm1d',{'track_running_stats':False});layer=_normalization_layer('batch_norm1d',c).to(device).eval();x=torch.tensor([1.,2.,4.,-2.,5.,8.,3.,-7.],device=device).reshape(2,2,2);full=_execute_stateful_normalization(x,'batch_norm1d',c,(2,2),layer);one=torch.cat([_execute_stateful_normalization(row[None],'batch_norm1d',c,(2,2),layer) for row in x]);assert not torch.allclose(full,one)

def graph(op,config=None):
 sample=(4,2) if op=='group_norm' else (2,2) if op.endswith('1d') else (2,2,2);c={'affine':True,'bias':True}
 if op=='group_norm':c.update(num_channels=4,num_groups=2)
 else:c.update(track_running_stats=True)
 if op.startswith('batch'):c['averaging']='cumulative'
 c.update(config or {});width=math.prod(sample);rows=[('input','input',{'features':2}),('project','linear',{'in_features':2,'out_features':width,'bias':True}),('array','reshape',{'shape':list(sample)}),('norm',op,c),('flat','flatten',{'start_dim':1,'end_dim':-1}),('head','linear',{'in_features':width,'out_features':2,'bias':True}),('output','output',{})]
 return {'schemaVersion':2,'name':'Normalization state','nodes':[{'id':i,'name':'same name','type':t,'config':c} for i,t,c in rows],'edges':[{'id':str(i),'source':a[0],'sourcePort':'out','target':b[0],'targetPort':'in'} for i,(a,b) in enumerate(zip(rows,rows[1:]))]}

@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('op',OPS)
def test_actual_graph_portable_training_eval_gradients_and_buffer_persistence(device,op):
 from runner.engine import GraphModel,export_source,REGISTRY
 model=GraphModel(graph(op),REGISTRY).to(device);code=export_source(model);assert 'runner.' not in code;scope={};exec(code,scope);portable=scope['Model']().to(device);portable.load_state_dict(model.state_dict(),strict=True)
 x=torch.tensor([[1.,-2.],[-.5,3.]],device=device).requires_grad_();y=x.detach().clone().requires_grad_();up=torch.tensor([[.5,-.25],[-.75,.25]],device=device);a=model(x);b=portable(y);torch.testing.assert_close(a,b,atol=0,rtol=0)
 ga=torch.autograd.grad(a,[x,*model.parameters()],up);gb=torch.autograd.grad(b,[y,*portable.parameters()],up)
 for a,b in zip(ga,gb):torch.testing.assert_close(a,b,atol=0,rtol=0)
 for k,v in model.state_dict().items():assert torch.equal(v,portable.state_dict()[k])
 before={k:v.clone() for k,v in model.state_dict().items()};model.eval();portable.eval();torch.testing.assert_close(model(x[:1]),portable(y[:1]),atol=0,rtol=0);assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
 expected={'layers.project.weight','layers.project.bias','layers.norm.weight','layers.norm.bias','layers.head.weight','layers.head.bias'}
 if op!='group_norm':expected|={'layers.norm.running_mean','layers.norm.running_var','layers.norm.num_batches_tracked'}
 assert set(model.state_dict())==expected;wrong=dict(model.state_dict());wrong['layers.norm.fake_buffer']=torch.ones(1,device=device)
 with pytest.raises(RuntimeError):portable.load_state_dict(wrong,strict=True)

@pytest.mark.parametrize('op',OPS)
def test_actual_cpu_resume_parameters_buffers_adam_rng_metrics_and_finetune_lineage(op,tmp_path):
 from runner.engine import run,REGISTRY
 from runner.tests.test_checkpoint_lifecycle import assert_nested_equal
 g=graph(op);options={'sample_limit':32,'batch_size':8,'device':'cpu','registry':REGISTRY,'emit':lambda e:None}
 for name,epochs,resume in [('full',2,None),('first',1,None),('resume',2,tmp_path/'first/checkpoint.pt')]:run(g,tmp_path/name,epochs=epochs,resume_checkpoint=resume,**options)
 a=torch.load(tmp_path/'full/checkpoint.pt',weights_only=True);b=torch.load(tmp_path/'resume/checkpoint.pt',weights_only=True)
 for key in ('modelState','optimizerState','rngState'):assert_nested_equal(a[key],b[key])
 assert (tmp_path/'full/metrics.json').read_bytes()==(tmp_path/'resume/metrics.json').read_bytes()
 run(g,tmp_path/'fine',epochs=1,seed=173,finetune_checkpoint=tmp_path/'first/checkpoint.pt',**options)
 for name in ('full','resume','fine'):
  m=json.loads((tmp_path/name/'manifest.json').read_text());buffers=op!='group_norm';assert ('buffers' in m['lineage']['restoredState'])==(buffers and name!='full');assert ('buffers' in m['lineage']['resetState'])==(buffers and name=='full');assert bool(m['checkpoint']['contract']['schedule'].get('bufferNodeOrder'))==buffers
 assert a['contract']['schedule']['normalizationNodeOrder']==['norm'];assert a['contract']['schedule']['parameterNodeOrder']==['project','norm','head']

@pytest.mark.parametrize('op',OPS)
def test_actual_mps_training_weights_and_buffers_finetune_and_cache_eval(op,tmp_path):
 from runner.engine import run,REGISTRY
 from runner.preview import preview
 g=graph(op);options={'sample_limit':32,'batch_size':8,'device':'mps','registry':REGISTRY,'emit':lambda e:None};run(g,tmp_path/'first',epochs=1,**options);m=json.loads((tmp_path/'first/manifest.json').read_text());assert m['runtime']['device']=='mps' and m['checkpoint']['contract']['guarantee']=='weightsOnly'
 run(g,tmp_path/'fine',epochs=1,seed=173,finetune_checkpoint=tmp_path/'first/checkpoint.pt',**options)
 with pytest.raises(ValueError,match='Exact resume'):run(g,tmp_path/'resume',epochs=2,resume_checkpoint=tmp_path/'first/checkpoint.pt',**options)
 before=preview(g,tmp_path/'before',registry=REGISTRY,device='mps',sample_limit=32,checkpoint=tmp_path/'first/checkpoint.pt');after=preview(g,tmp_path/'after',registry=REGISTRY,device='mps',sample_limit=32,checkpoint=tmp_path/'first/checkpoint.pt',resume_from='flat',cache=before['cachePath']);assert before['previews']['output']==after['previews']['output'];assert before['previews']['norm']['shape'][0]==6

@pytest.mark.parametrize('op',OPS)
def test_authored_generated_and_actual_discovery_exact_descriptor(op,monkeypatch):
 from runner.stateful_normalization_configuration import capability
 from runner.engine import capabilities
 import importlib
 root=Path(__file__).resolve().parents[2];monkeypatch.syspath_prepend(str(root/'runner'));table=importlib.import_module('runner.stateful_normalization_adapters_generated');bundle=json.loads((root/'components/normalization/bundle.json').read_text());entry=next(row for row in bundle['components'] if row['definition']['id']==op);assert entry['runtimeContract']==capability(op)==table.CAPABILITIES[op]()==capabilities()['componentCapabilities'][op];assert table.NORMALIZERS[op]({})==normalize(op,{})

@pytest.mark.parametrize('op',OPS[:-1])
def test_preview_saved_bad_buffer_dtype_refuses(op,tmp_path):
 from runner.engine import run,REGISTRY
 from runner.preview import preview
 run(graph(op),tmp_path/'first',epochs=1,sample_limit=32,batch_size=8,device='cpu',registry=REGISTRY,emit=lambda e:None);p=tmp_path/'first/checkpoint.pt';state=torch.load(p,weights_only=True);state['modelState']['layers.norm.num_batches_tracked']=state['modelState']['layers.norm.num_batches_tracked'].float();bad=tmp_path/'first/bad.pt';torch.save(state,bad)
 with pytest.raises(ValueError,match='dtype'):preview(graph(op),tmp_path/'preview',registry=REGISTRY,device='cpu',checkpoint=bad)

def inference_scope():
 path=Path(__file__).resolve().parents[2]/'Sources/KwipMLApp/Resources/inference.py';scope={'__file__':str(path),'__name__':'reviewed_inference_test'};exec(compile(path.read_text(),str(path),'exec'),scope);return scope

@pytest.mark.parametrize('op',OPS[:-1])
@pytest.mark.parametrize('bad',['none','negativeVariance','negativeCounter','largeCounter','fractionCounter','wrongShape','unknownIntegerState'])
def test_exported_inference_helper_strict_typed_buffer_whitelist(op,bad):
 from runner.engine import _normalization_layer
 c=normalize(op,{'track_running_stats':True});layer=_normalization_layer(op,c);model=torch.nn.ModuleDict({'norm':layer});state=copy.deepcopy(model.state_dict());prefix='norm.'
 if bad=='negativeVariance':state[prefix+'running_var'].fill_(-1)
 elif bad=='negativeCounter':state[prefix+'num_batches_tracked'].fill_(-1)
 elif bad=='largeCounter':state[prefix+'num_batches_tracked'].fill_(1000000001)
 elif bad=='fractionCounter':state[prefix+'num_batches_tracked']=torch.tensor(.5)
 elif bad=='wrongShape':state[prefix+'num_batches_tracked']=torch.tensor([0],dtype=torch.int64)
 elif bad=='unknownIntegerState':state['extra.weight']=torch.ones(1,dtype=torch.int64)
 check=inference_scope()['validate_reviewed_normalization_buffers']
 if bad=='none':assert check(model,state)
 else:
  with pytest.raises(ValueError):check(model,state)

@pytest.mark.parametrize('op',OPS[:-1])
def test_running_overflow_or_instance_nonzero_counter_refused(op):
 from runner.engine import _normalization_layer,_execute_stateful_normalization
 c=normalize(op,{'track_running_stats':True});layer=_normalization_layer(op,c);sample=(2,3) if op.endswith('1d') else (2,2,3)
 if op.startswith('instance'):
  layer.num_batches_tracked.fill_(1)
  with pytest.raises(ValueError,match='counter'):_execute_stateful_normalization(torch.ones(2,*sample),op,c,sample,layer)
 else:
  # Finite source can overflow a tracked variance; it must never publish that state.
  x=torch.full((2,*sample),1e30);x[0].neg_()
  with pytest.raises(ValueError,match='state|statistics|finite'):_execute_stateful_normalization(x,op,c,sample,layer)

@pytest.mark.parametrize('op',OPS)
def test_actual_generated_export_inference_helper_cold_load_typed_state(op,tmp_path):
 import hashlib
 from runner.engine import GraphModel,export_source,REGISTRY
 model=GraphModel(graph(op),REGISTRY);model.train();model(torch.tensor([[1.,-2.],[-.5,3.]]));model.eval();before={k:v.clone() for k,v in model.state_dict().items()}
 template=Path(__file__).resolve().parents[2]/'Sources/KwipMLApp/Resources/inference.py';(tmp_path/'inference.py').write_bytes(template.read_bytes());(tmp_path/'model.py').write_text(export_source(model));torch.save({'checkpointSchemaVersion':1,'modelState':model.state_dict()},tmp_path/'checkpoint.pt');(tmp_path/'input-contract.json').write_text(json.dumps({'schemaVersion':1,'input':{'dtype':'float32','shape':['N',2],'preprocessing':'alreadyPrepared'}}))
 def inventory():
  files=[]
  for name in ('model.py','checkpoint.pt','inference.py','input-contract.json'):
   p=tmp_path/name;files.append({'path':name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
  (tmp_path/'export-manifest.json').write_text(json.dumps({'schemaVersion':1,'kind':'kwipmlTrainedModel','files':files}))
 inventory();scope={'__file__':str(tmp_path/'inference.py'),'__name__':'saved_normalization_inference'};exec(compile(template.read_text(),str(template),'exec'),scope);predictor=scope['load_model']();x=torch.tensor([[1.,-2.],[-.5,3.]]);torch.testing.assert_close(predictor.predict(x,batch_size=1),model(x),atol=2e-6,rtol=2e-6);assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
 bad=copy.deepcopy(model.state_dict());bad['layers.head.weight']=bad['layers.head.weight'].to(torch.int64);torch.save({'checkpointSchemaVersion':1,'modelState':bad},tmp_path/'checkpoint.pt');inventory()
 with pytest.raises(ValueError,match='float32|dtype|typed'):scope['load_model']()

@pytest.mark.parametrize('op',OPS)
def test_cache_binds_changed_running_buffer_or_affine_state(op,tmp_path):
 from runner.engine import run,REGISTRY
 from runner.preview import preview
 g=graph(op);run(g,tmp_path/'first',epochs=1,sample_limit=32,batch_size=8,device='cpu',registry=REGISTRY,emit=lambda e:None);old=tmp_path/'first/checkpoint.pt';before=preview(g,tmp_path/'before',registry=REGISTRY,device='cpu',sample_limit=32,checkpoint=old);state=torch.load(old,weights_only=True)
 key='layers.norm.weight' if op=='group_norm' else 'layers.norm.running_mean';state['modelState'][key]=state['modelState'][key]+.25;changed=tmp_path/'first/changed.pt';torch.save(state,changed)
 with pytest.raises(ValueError,match='cache|Cache'):preview(g,tmp_path/'changed',registry=REGISTRY,device='cpu',sample_limit=32,checkpoint=changed,resume_from='flat',cache=before['cachePath'])

@pytest.mark.parametrize('op',['batch_norm1d','instance_norm1d'])
def test_actual_training_batch_preflight_refuses_before_any_running_update(op,tmp_path,monkeypatch):
 from runner.engine import run,REGISTRY,GraphModel
 g=graph(op);next(n for n in g['nodes'] if n['id']=='project')['config']['out_features']=2;next(n for n in g['nodes'] if n['id']=='array')['config']['shape']=[2] if op.startswith('batch') else [2,1];next(n for n in g['nodes'] if n['id']=='head')['config']['in_features']=2
 observed=[];original=GraphModel.__init__
 def record(self,*a,**k):original(self,*a,**k);observed.append(self)
 monkeypatch.setattr(GraphModel,'__init__',record)
 # Generated32 observations leave26training; batch25 produces a final singleton.
 with pytest.raises(ValueError,match='minibatch|spatial'):run(g,tmp_path/'run',epochs=1,sample_limit=32,batch_size=25,device='cpu',registry=REGISTRY,emit=lambda e:None)
 assert len(observed)==1;norm=observed[0].layers[observed[0].layer_keys['norm']];assert norm.num_batches_tracked.item()==0;assert torch.equal(norm.running_mean,torch.zeros(2));assert torch.equal(norm.running_var,torch.ones(2));assert not (tmp_path/'run/manifest.json').exists()
