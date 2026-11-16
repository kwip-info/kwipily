"""Independent coordinate windows, overlap addition and real composed lifecycle."""
import copy,itertools,json,math
from pathlib import Path
import pytest,torch
from runner.tensor_layout import BatchedLayout
from runner.spatial_patch_configuration import normalize_spatial_patch as normalize,resolve_spatial_patch as resolve

GEOMETRIES=[((2,4,5),(2,3),(1,1),(1,1),(0,0)),((2,5,4),(2,2),(2,1),(2,1),(1,2)),((1,3,3),(3,3),(1,1),(1,1),(1,1)),((1,4,4),(1,1),(3,3),(1,1),(0,0)),((2,2,3),(2,2),(1,1),(1,1),(3,2))]
def settings(k,s,d,p):return {'kernel_size':list(k),'stride':list(s),'dilation':list(d),'padding':list(p)}
def independent_windows(shape,k,s,d,p):
 n,c,h,w=shape;kh,kw=k;sh,sw=s;dh,dw=d;ph,pw=p;oh=(h+2*ph-dh*(kh-1)-1)//sh+1;ow=(w+2*pw-dw*(kw-1)-1)//sw+1
 positions=[]
 for b,ch,ky,kx,y,x in itertools.product(range(n),range(c),range(kh),range(kw),range(oh),range(ow)):
  iy=y*sh-ph+ky*dh;ix=x*sw-pw+kx*dw;positions.append(((b*c+ch)*h+iy)*w+ix if 0<=iy<h and 0<=ix<w else None)
 return (n,c*kh*kw,oh*ow),positions

@pytest.mark.parametrize('sample,k,s,d,p',GEOMETRIES)
def test_independent_static_geometry_and_fold_output_size(sample,k,s,d,p):
 c=settings(k,s,d,p);out,_=independent_windows((2,*sample),k,s,d,p);u=resolve('unfold',c,BatchedLayout(sample));assert u['layout'].sample==out[1:];assert u['workPerObservation']==math.prod(out[1:]);f=resolve('fold',{**c,'output_size':list(sample[-2:])},BatchedLayout(out[1:]));assert f['layout'].sample==sample;assert f['workPerObservation']==math.prod(out[1:])

def test_defaults_are_explicit_and_output_extent_ambiguity_is_not_repaired():
 assert normalize('unfold',{})==settings((2,2),(1,1),(1,1),(0,0))
 for extent in (4,5):assert resolve('fold',{'stride':[2,2],'output_size':[extent,extent]},BatchedLayout((4,4)))['layout'].sample==(1,extent,extent)

@pytest.mark.parametrize('op,c,sample',[('unfold',{'kernel_size':[5,5]},(1,2,2)),('fold',{'output_size':[3,3]},(5,4)),('fold',{'output_size':[3,3]},(4,5))])
def test_impossible_geometry_refuses(op,c,sample):
 with pytest.raises(ValueError):resolve(op,c,BatchedLayout(sample))

@pytest.fixture(autouse=True)
def isolate_device_policy(request):
 device=getattr(request.node,'callspec',None);device=None if device is None else device.params.get('device')
 if device not in ('cpu','mps'):yield;return
 prior=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled()
 try:torch.use_deterministic_algorithms(device=='cpu',warn_only=False);yield
 finally:torch.use_deterministic_algorithms(prior,warn_only=warn)

def execute(op,c,x):
 from runner.engine import _execute_spatial_patch
 r=resolve(op,c,BatchedLayout(tuple(x.shape[1:])));return _execute_spatial_patch(x,op,r['config'],tuple(x.shape[1:]),r['layout'].sample,r['workPerObservation'])

@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('sample,k,s,d,p',GEOMETRIES)
def test_scalar_coordinates_weighted_gradients_overlap_and_unchanged_views(device,sample,k,s,d,p):
 shape=(2,*sample);out,mapping=independent_windows(shape,k,s,d,p);vals=[(i%13-6)/4 for i in range(math.prod(shape))];c=settings(k,s,d,p)
 backing=torch.zeros(*shape[:-1],shape[-1]*2,device=device);backing[...,::2]=torch.tensor(vals,device=device).reshape(shape);x=backing[...,::2].detach().requires_grad_();u=execute('unfold',c,x);uv=[0. if i is None else vals[i] for i in mapping];weights=[(i%7-3)/4 for i in range(len(mapping))];dx=[0.]*len(vals)
 for i,target in enumerate(mapping):
  if target is not None:dx[target]+=weights[i]
 torch.testing.assert_close(u.cpu(),torch.tensor(uv).reshape(out),atol=0,rtol=0);torch.testing.assert_close(torch.autograd.grad(u,x,torch.tensor(weights,device=device).reshape(out))[0].cpu(),torch.tensor(dx).reshape(shape),atol=0,rtol=0);assert x.detach().cpu().flatten().tolist()==vals;assert u.untyped_storage().data_ptr()!=x.untyped_storage().data_ptr()
 columns=[(i%11-5)/4 for i in range(len(mapping))];z=torch.tensor(columns,device=device).reshape(out).requires_grad_();f=execute('fold',{**c,'output_size':list(sample[-2:])},z);fv=[0.]*len(vals);fw=[(i%9-4)/4 for i in range(len(vals))]
 for i,target in enumerate(mapping):
  if target is not None:fv[target]+=columns[i]
 dz=[0. if i is None else fw[i] for i in mapping]
 torch.testing.assert_close(f.cpu(),torch.tensor(fv).reshape(shape),atol=0,rtol=0);torch.testing.assert_close(torch.autograd.grad(f,z,torch.tensor(fw,device=device).reshape(shape))[0].cpu(),torch.tensor(dz).reshape(out),atol=0,rtol=0);assert z.detach().cpu().flatten().tolist()==columns
 overlap=execute('fold',{**c,'output_size':list(sample[-2:])},u.detach());counts=[0]*len(vals)
 for i in mapping:
  if i is not None:counts[i]+=1
 torch.testing.assert_close(overlap.cpu(),torch.tensor([a*b for a,b in zip(vals,counts)]).reshape(shape),atol=0,rtol=0)

@pytest.mark.parametrize('op',['unfold','fold'])
@pytest.mark.parametrize('key',['kernel_size','stride','dilation','padding'])
@pytest.mark.parametrize('value',[2,[1],[1,2,3],[True,1],[1.5,2],[float('inf'),2]])
def test_strict_pairs_refuse(op,key,value):
 with pytest.raises(ValueError):normalize(op,{key:value})

@pytest.mark.parametrize('op,c',[('unfold',{'output_size':[2,2]}),('fold',{'unknown':1}),('fold',{'output_size':[0,2]}),('unfold',{'kernel_size':[4097,2]}),('fold',{'padding':[-1,0]})])
def test_declared_field_scope_and_bounds(op,c):
 with pytest.raises(ValueError):normalize(op,c)

@pytest.mark.parametrize('op,sample,c',[('unfold',(2,3),{}),('fold',(1,2,3),{}),('unfold',(513,3,3),{}),('unfold',(1,4097,3),{}),('fold',(2052,1),{'output_size':[2,2]}),('unfold',(512,4096,4096),{'kernel_size':[1,1]}),('unfold',(512,512,512),{}),('fold',(4,16777216),{'output_size':[4096,4096]})])
def test_rank_channel_spatial_static_budget_refuses(op,sample,c):
 with pytest.raises(ValueError):resolve(op,c,BatchedLayout(sample))

@pytest.mark.parametrize('op',['unfold','fold'])
@pytest.mark.parametrize('problem',['dtype','rank','batch','sparse','nonfinite'])
def test_actual_source_refuses_before_native(op,problem,monkeypatch):
 from runner.engine import _execute_spatial_patch
 sample=(1,3,3) if op=='unfold' else (4,4);c={} if op=='unfold' else {'output_size':[3,3]};r=resolve(op,c,BatchedLayout(sample));x=torch.ones(2,*sample)
 if problem=='dtype':x=x.double()
 if problem=='rank':x=x.flatten(1)
 if problem=='batch':x=x[:0]
 if problem=='sparse':x=x.to_sparse()
 if problem=='nonfinite':x.flatten()[0]=float('nan')
 called=[];monkeypatch.setattr(torch.nn.functional,op,lambda *a,**k:called.append(True))
 with pytest.raises(ValueError):_execute_spatial_patch(x,op,r['config'],sample,r['layout'].sample,r['workPerObservation'])
 assert not called

def test_observation_output_work_budget_before_finite_scan(monkeypatch):
 from runner.engine import _execute_spatial_patch
 called=[];monkeypatch.setattr(torch,'isfinite',lambda *a,**k:called.append(True))
 x=torch.ones(1,1,2,2).expand(25000001,1,2,2)
 with pytest.raises(ValueError,match='budget'):_execute_spatial_patch(x,'unfold',normalize('unfold',{}),(1,2,2),(4,1),4)
 assert not called

@pytest.mark.parametrize('device',['cpu','mps'])
def test_fold_overlap_overflow_refuses_before_capture(device):
 z=torch.full((1,4,4),torch.finfo(torch.float32).max,device=device)
 with pytest.raises(ValueError,match='finite'):execute('fold',{'output_size':[3,3]},z)

@pytest.mark.parametrize('op',['unfold','fold'])
def test_schema_one_refuses(op):
 from runner.graph_semantics import effective_config
 with pytest.raises(ValueError,match='schema'):effective_config({'type':op,'config':{}},1)

PATCH_CONFIG=settings((3,2),(1,1),(1,1),(0,0))
def graph(op):
 sample=(1,4,5) if op=='unfold' else (6,8);config=PATCH_CONFIG if op=='unfold' else {**PATCH_CONFIG,'output_size':[4,5]};out=resolve(op,config,BatchedLayout(sample))['layout'].sample
 rows=[('input','input',{'features':2}),('project','linear',{'in_features':2,'out_features':math.prod(sample),'bias':True}),('array','reshape',{'shape':list(sample)}),('patch',''+op,config),('flat','flatten',{'start_dim':1,'end_dim':-1}),('head','linear',{'in_features':math.prod(out),'out_features':2,'bias':True}),('output','output',{})]
 return {'schemaVersion':2,'name':'Spatial patches','nodes':[{'id':i,'name':'same name','type':t,'config':c} for i,t,c in rows],'edges':[{'id':str(i),'source':a[0],'sourcePort':'out','target':b[0],'targetPort':'in'} for i,(a,b) in enumerate(zip(rows,rows[1:]))]}

@pytest.mark.parametrize('op',['unfold','fold'])
@pytest.mark.parametrize('device',['cpu','mps'])
def test_whole_graph_portable_input_and_parameter_gradients_strict_roundtrip(op,device):
 from runner.engine import GraphModel,REGISTRY,export_source
 model=GraphModel(graph(op),REGISTRY).to(device);code=export_source(model);assert 'runner.' not in code;scope={};exec(code,scope);portable=scope['Model']().to(device);portable.load_state_dict(model.state_dict(),strict=True)
 assert set(model.state_dict())=={'layers.project.weight','layers.project.bias','layers.head.weight','layers.head.bias'}
 x=torch.tensor([[1.,-2.],[-.5,3.]],device=device,requires_grad=True);y=x.detach().clone().requires_grad_();weights=torch.tensor([[.5,-.25],[-.75,.25]],device=device)
 a=model(x);b=portable(y);torch.testing.assert_close(a,b,atol=0,rtol=0)
 ag=torch.autograd.grad(a,[x,*model.parameters()],weights);bg=torch.autograd.grad(b,[y,*portable.parameters()],weights)
 for v,w in zip(ag,bg):torch.testing.assert_close(v,w,atol=0,rtol=0)
 model.eval();torch.testing.assert_close(model(x),a,atol=0,rtol=0)
 state=dict(model.state_dict());state['layers.patch.weight']=torch.ones(1,device=device)
 with pytest.raises(RuntimeError):portable.load_state_dict(state,strict=True)

@pytest.mark.parametrize('op',['unfold','fold'])
def test_exact_cpu_checkpoint_continuation_and_finetuning(op,tmp_path):
 from runner.engine import run,REGISTRY
 from runner.tests.test_checkpoint_lifecycle import assert_nested_equal
 options={'sample_limit':32,'batch_size':8,'device':'cpu','registry':REGISTRY,'emit':lambda event:None};g=graph(op)
 for name,epochs,resume in [('full',2,None),('first',1,None),('resume',2,tmp_path/'first/checkpoint.pt')]:run(g,tmp_path/name,epochs=epochs,resume_checkpoint=resume,**options)
 a=torch.load(tmp_path/'full/checkpoint.pt',weights_only=True);b=torch.load(tmp_path/'resume/checkpoint.pt',weights_only=True)
 for key in ('modelState','optimizerState','rngState'):assert_nested_equal(a[key],b[key])
 assert (tmp_path/'full/metrics.json').read_bytes()==(tmp_path/'resume/metrics.json').read_bytes();assert 'parameterNodeOrder' not in a['contract']['schedule'];assert set(a['modelState'])=={'layers.project.weight','layers.project.bias','layers.head.weight','layers.head.bias'}
 run(g,tmp_path/'fine',epochs=1,seed=173,finetune_checkpoint=tmp_path/'first/checkpoint.pt',**options)
 m=json.loads((tmp_path/'fine/manifest.json').read_text());assert m['checkpoint']['contract']['guarantee']=='epochBoundaryExactCPU';assert m['runtime']['device']=='cpu'

@pytest.mark.parametrize('op',['unfold','fold'])
def test_actual_mps_training_weights_only_capture_cache_and_finetune(op,tmp_path):
 from runner.engine import run,REGISTRY
 from runner.preview import preview
 options={'sample_limit':32,'batch_size':8,'device':'mps','registry':REGISTRY,'emit':lambda event:None};g=graph(op)
 run(g,tmp_path/'first',epochs=1,sample_capture={'schemaVersion':1,'activationValues':True,'predictionRows':True},**options)
 m=json.loads((tmp_path/'first/manifest.json').read_text());assert m['runtime']['device']=='mps';assert m['checkpoint']['contract']['guarantee']=='weightsOnly';assert m['settings']['deterministicAlgorithms'] is False
 run(g,tmp_path/'fine',epochs=1,seed=173,finetune_checkpoint=tmp_path/'first/checkpoint.pt',**options)
 with pytest.raises(ValueError,match='Exact resume'):run(g,tmp_path/'resume',epochs=2,resume_checkpoint=tmp_path/'first/checkpoint.pt',**options)
 before=preview(g,tmp_path/'before',registry=REGISTRY,device='mps',checkpoint=tmp_path/'first/checkpoint.pt');after=preview(g,tmp_path/'after',registry=REGISTRY,device='mps',checkpoint=tmp_path/'first/checkpoint.pt',resume_from='flat',cache=before['cachePath']);assert before['previews']['output']==after['previews']['output'];assert before['previews']['patch']['shape']==[8,*resolve(op,next(n['config'] for n in g['nodes'] if n['id']=='patch'),BatchedLayout((1,4,5) if op=='unfold' else (6,8)))['layout'].sample]

@pytest.mark.parametrize('op',['unfold','fold'])
def test_actual_authored_generated_and_discovered_descriptor_matches(op,monkeypatch):
 from runner.spatial_patch_configuration import capability
 from runner.engine import capabilities,REGISTRY
 monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
 import runner.spatial_patch_adapters_generated as table
 assert table.CAPABILITIES[op]()==capability(op);assert capabilities()['componentCapabilities'][op]==capability(op)
 bundle=json.loads((Path(__file__).resolve().parents[2]/'components/spatial-patches/bundle.json').read_text());assert next(d for d in bundle['components'] if d['definition']['id']==op)['runtimeContract']==capability(op)
 assert table.NORMALIZERS[op]({})==normalize(op,{})

@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('op',['unfold','fold'])
def test_learned_projection_and_head_scalar_values_all_parameter_influence(op,device):
 from runner.engine import GraphModel,REGISTRY
 g=graph(op);model=GraphModel(g,REGISTRY).to(device);sample=(1,4,5) if op=='unfold' else (6,8);width=math.prod(sample);outwidth=48 if op=='unfold' else 20
 w1=[(i%7-3)/16 for i in range(width*2)];b1=[(i%5-2)/16 for i in range(width)];w2=[(i%9-4)/16 for i in range(outwidth*2)];b2=[.125,-.25]
 with torch.no_grad():
  for key,data,shape in [('layers.project.weight',w1,(width,2)),('layers.project.bias',b1,(width,)),('layers.head.weight',w2,(2,outwidth)),('layers.head.bias',b2,(2,))]:model.state_dict()[key].copy_(torch.tensor(data,device=device).reshape(shape))
 xv=[1.,-2.,-.5,3.];up=[.5,-.25,-.75,.25];project=[b1[j]+sum(xv[n*2+k]*w1[j*2+k] for k in range(2)) for n in range(2) for j in range(width)];_,mapping=independent_windows((2,1,4,5),(3,2),(1,1),(1,1),(0,0))
 patch=[0.]* (2*outwidth)
 if op=='unfold':patch=[0. if i is None else project[i] for i in mapping]
 else:
  for i,target in enumerate(mapping):
   if target is not None:patch[target]+=project[i]
 expected=[b2[j]+sum(patch[n*outwidth+k]*w2[j*outwidth+k] for k in range(outwidth)) for n in range(2) for j in range(2)]
 dp=[sum(up[n*2+j]*w2[j*outwidth+k] for j in range(2)) for n in range(2) for k in range(outwidth)];dz=[0.]*len(project)
 if op=='unfold':
  for i,target in enumerate(mapping):
   if target is not None:dz[target]+=dp[i]
 else:dz=[0. if target is None else dp[target] for target in mapping]
 dx=[sum(dz[n*width+j]*w1[j*2+k] for j in range(width)) for n in range(2) for k in range(2)];dw1=[sum(dz[n*width+j]*xv[n*2+k] for n in range(2)) for j in range(width) for k in range(2)];db1=[sum(dz[n*width+j] for n in range(2)) for j in range(width)];dw2=[sum(up[n*2+j]*patch[n*outwidth+k] for n in range(2)) for j in range(2) for k in range(outwidth)];db2=[sum(up[n*2+j] for n in range(2)) for j in range(2)]
 x=torch.tensor(xv,device=device).reshape(2,2).requires_grad_();actual=model(x);torch.testing.assert_close(actual.cpu(),torch.tensor(expected).reshape(2,2),atol=2e-6,rtol=2e-6)
 gradients=torch.autograd.grad(actual,[x,*model.parameters()],torch.tensor(up,device=device).reshape(2,2))
 for actual,values in zip(gradients,[dx,dw1,db1,dw2,db2]):torch.testing.assert_close(actual.cpu(),torch.tensor(values).reshape(actual.shape),atol=2e-6,rtol=2e-6)

def test_actual_fold_output_budget_independent_of_small_incoming_columns(monkeypatch):
 from runner.engine import _execute_spatial_patch
 c={'kernel_size':[1,1],'stride':[4095,4095],'output_size':[4096,4096]};r=resolve('fold',c,BatchedLayout((1,4)));x=torch.ones(7,1,4);calls=[];monkeypatch.setattr(torch,'isfinite',lambda *a,**k:calls.append(True))
 with pytest.raises(ValueError,match='budget'):_execute_spatial_patch(x,'fold',r['config'],(1,4),r['layout'].sample,r['workPerObservation'])
 assert calls==[]

@pytest.mark.parametrize('op',['unfold','fold'])
def test_exact_integral_json_number_representation_and_identity_geometry(op):
 from runner.graph_semantics import effective_config,semantic_graph
 integers={'kernel_size':[2,2],'stride':[1,1],'dilation':[1,1],'padding':[0,0]}
 if op=='fold':integers['output_size']=[2,2]
 numbers={k:[float(v) for v in a] for k,a in integers.items()}
 assert normalize(op,numbers)==normalize(op,integers)
 assert effective_config({'type':op,'config':numbers},2)==integers
