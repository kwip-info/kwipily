"""Independent coordinates and weighted source influence for named tensor routing."""
import math,itertools,copy,json
import pytest,torch
from runner.tensor_layout import BatchedLayout
from runner.tensor_routing_configuration import normalize_tensor_routing as normalize,resolve_tensor_routing as resolve,configured_inputs,parse_equation

@pytest.fixture(autouse=True)
def isolated_runtime_device_determinism(request):
 """Direct adapters use the worker's device policy, then restore process globals."""
 device=getattr(request.node,'callspec',None)
 device=None if device is None else device.params.get('device')
 if device not in ('cpu','mps'):
  yield
  return
 prior=torch.are_deterministic_algorithms_enabled()
 prior_warn_only=torch.is_deterministic_algorithms_warn_only_enabled()
 try:
  torch.use_deterministic_algorithms(device=='cpu',warn_only=False)
  assert torch.are_deterministic_algorithms_enabled()==(device=='cpu')
  assert not torch.is_deterministic_algorithms_warn_only_enabled()
  yield
 finally:
  torch.use_deterministic_algorithms(prior,warn_only=prior_warn_only)

@pytest.mark.parametrize('operation,config,sample,expected',[
 ('split',{'mode':'size','dim':-1,'split_size':2,'sections':[3]},(2,5),[(2,2),(2,2),(2,1)]),
 ('split',{'mode':'sections','dim':2,'split_size':9,'sections':[1,3,1]},(2,5),[(2,1),(2,3),(2,1)]),
 ('split',{'split_size':100},(2,5),[(2,5)]),
 ('chunk',{'chunks':6},(2,13),[(2,3)]*4+[(2,1)]),
 ('chunk',{'chunks':8},(2,3),[(2,1)]*3)])
def test_real_native_part_geometry_before_allocation(operation,config,sample,expected):
 out=resolve(operation,config,[BatchedLayout(sample)])['outputs'];assert list(out)==[f'part{i}' for i in range(len(expected))];assert [v.sample for v in out.values()]==expected

@pytest.mark.parametrize('equation,left,right,expected',[
 ('nab,nbc->nac',(2,3),(3,4),(2,4)),('na,nb->nab',(2,),(3,),(2,3)),('naa->na',(3,3),None,(3,)),('nab->nba',(2,3),None,(3,2)),('nab->na',(2,3),None,(2,)),('nab,nab->nab',(2,3),(1,3),(2,3)),('naab,nac->nbc',(2,2,3),(2,4),(3,4))])
def test_full_einsum_grammar_and_named_dimension_resolution(equation,left,right,expected):
 c={'pattern':'custom','equation':equation};out=resolve('einsum',c,[BatchedLayout(left)]+([BatchedLayout(right)] if right else []));assert out['outputs']['out'].sample==expected;assert configured_inputs('einsum',c)==(('a','b') if right else ('a',))

def test_inactive_fields_do_not_change_identity_but_invalid_fields_still_refuse():
 assert normalize('split',{'split_size':2,'sections':[7]})==normalize('split',{'split_size':2,'sections':[1,1]})
 assert normalize('einsum',{'pattern':'multiply','equation':'na->na'})['equation']=='nab,nbc->nac'
 with pytest.raises(ValueError):normalize('split',{'sections':[True]})

@pytest.mark.parametrize('equation',['ab->ba','nan->na','na,nb,nc->nabc','na','na->','na->nn','na->nz','n...a->na','na\n->na','na->nα','na->na->na','na->n'])
def test_invalid_einsum_notation_is_data_not_executable_code(equation):
 with pytest.raises(ValueError):parse_equation(equation)


def test_gather_shape_contract_different_selected_extent():
 assert resolve('gather',{'dim':2},[BatchedLayout((2,4)),BatchedLayout((2,3))])['outputs']['out'].sample==(2,3)
 with pytest.raises(ValueError):resolve('gather',{'dim':2},[BatchedLayout((2,4)),BatchedLayout((3,3))])

def run_native(op,c,tensors):
 from runner.engine import _execute_tensor_routing
 r=resolve(op,c,[BatchedLayout(tuple(t.shape[1:])) for t in tensors]);outputs={k:v.sample for k,v in r['outputs'].items()};return _execute_tensor_routing(tensors,op,r['config'],[tuple(t.shape[1:]) for t in tensors],outputs,r['workPerObservation'])

@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('op,c,extent,sizes',[('split',{'split_size':2},5,[2,2,1]),('split',{'mode':'sections','sections':[1,3,1]},5,[1,3,1]),('chunk',{'chunks':6},13,[3,3,3,3,1]),('chunk',{'chunks':8},3,[1,1,1])])
def test_ordered_native_views_and_independent_weighted_source_influence(device,op,c,extent,sizes):
 values=[(i%11-5)/4 for i in range(2*2*extent)];store=torch.zeros(2,2,extent*2,device=device);store[...,::2]=torch.tensor(values,device=device).reshape(2,2,extent);x=store[...,::2].detach().requires_grad_();parts=run_native(op,c,[x]);assert len(parts)==len(sizes)
 gradients=[0.]*len(values);loss=0;start=0
 for port,(part,size) in enumerate(zip(parts,sizes)):
  expected=[];weights=[]
  for n,r,column in itertools.product(range(2),range(2),range(size)):
   index=(n*2+r)*extent+start+column;expected.append(values[index]);g=(len(weights)%5-2+port)/4;weights.append(g);gradients[index]+=g
  torch.testing.assert_close(part.cpu(),torch.tensor(expected).reshape(2,2,size),rtol=0,atol=0);assert part.untyped_storage().data_ptr()==x.untyped_storage().data_ptr();loss=loss+(part*torch.tensor(weights,device=device).reshape(part.shape)).sum();start+=size
 torch.testing.assert_close(torch.autograd.grad(loss,x)[0].cpu(),torch.tensor(gradients).reshape(x.shape),rtol=0,atol=0);assert x.detach().cpu().flatten().tolist()==values

@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('dim',[1,2,-1,-2])
def test_gather_repeated_selectors_scalar_coordinates_and_no_index_gradient(device,dim):
 axis=dim%3;sample=(3,3) if axis==1 else (2,3);values=[(i%13-6)/4 for i in range(2*2*4)];indices=[(i*3+1)%2 if axis==1 else (i%3) for i in range(2*math.prod(sample))];x=torch.tensor(values,device=device).reshape(2,2,4).requires_grad_();selector=torch.tensor(indices,dtype=torch.int64,device=device).reshape(2,*sample);actual=run_native('gather',{'dim':dim},[x,selector]);expected=[];dx=[0.]*len(values);weights=[]
 for n,a,b in itertools.product(range(2),*(range(v) for v in sample)):
  i=len(expected);coords=[n,a,b];coords[axis]=indices[i];offset=(coords[0]*2+coords[1])*4+coords[2];expected.append(values[offset]);g=(i%5-2)/4;weights.append(g);dx[offset]+=g
 torch.testing.assert_close(actual.cpu(),torch.tensor(expected).reshape(actual.shape),rtol=0,atol=0);torch.testing.assert_close(torch.autograd.grad(actual,x,torch.tensor(weights,device=device).reshape(actual.shape))[0].cpu(),torch.tensor(dx).reshape(x.shape),rtol=0,atol=0);assert not selector.requires_grad and selector.grad is None

EINSUM=[('nab,nbc->nac',(2,3),(3,2)),('na,nb->nab',(2,),(3,)),('naa->na',(3,3),None),('nab->nba',(2,3),None),('nab->na',(2,3),None),('nab,nab->nab',(2,3),(1,3)),('naab,nac->nbc',(2,2,3),(2,2))]

def scalar_einsum(equation,samples,values,batch):
 inputs,out=equation.split('->');labels=inputs.split(',');sizes={}
 for term,shape in zip(labels,samples):
  for name,extent in zip(term[1:],shape):sizes[name]=max(sizes.get(name,1),extent)
 shape=tuple(sizes[k] for k in out[1:]);expected=[0.]*(batch*math.prod(shape));gradients=[[0.]*len(v) for v in values];weights=[(i%7-3)/4 for i in range(len(expected))]
 def offset(coords,dims):
  i=0
  for c,d in zip(coords,dims):i=i*d+c
  return i
 for n in range(batch):
  for coords in itertools.product(*(range(v) for v in sizes.values())):
   names=dict(zip(sizes,coords));source=[]
   for term,sample in zip(labels,samples):source.append(offset((n,*[0 if extent==1 else names[k] for k,extent in zip(term[1:],sample)]),(batch,*sample)))
   target=offset((n,*[names[k] for k in out[1:]]),(batch,*shape));v=[data[i] for data,i in zip(values,source)];expected[target]+=math.prod(v)
   for operand,i in enumerate(source):gradients[operand][i]+=weights[target]*math.prod([x for j,x in enumerate(v) if j!=operand])
 return (batch,*shape),expected,weights,gradients

@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('equation,a,b',EINSUM)
def test_all_einsum_grammars_independent_coordinates_and_weighted_operand_influence(device,equation,a,b):
 samples=[a]+([b] if b else []);values=[[(i*(j+3)%11-5)/4 for i in range(2*math.prod(s))] for j,s in enumerate(samples)];sources=[torch.tensor(v,device=device).reshape(2,*s).requires_grad_() for s,v in zip(samples,values)];shape,expected,weights,gradients=scalar_einsum(equation,samples,values,2);actual=run_native('einsum',{'pattern':'custom','equation':equation},sources)
 torch.testing.assert_close(actual.cpu(),torch.tensor(expected).reshape(shape),atol=2e-6,rtol=2e-6);g=torch.autograd.grad(actual,sources,torch.tensor(weights,device=device).reshape(shape))
 for source,gradient,wanted in zip(sources,g,gradients):torch.testing.assert_close(gradient.cpu(),torch.tensor(wanted).reshape(source.shape),atol=2e-6,rtol=2e-6)

def graph(op,c=None,part='part0'):
 defaults={'split':{'split_size':2},'chunk':{'chunks':6},'gather':{'dim':2},'einsum':{'pattern':'multiply'}};c=defaults[op] if c is None else c
 def node(i,t,config):return {'id':i,'type':t,'name':'same name','config':config}
 rows=[node('input','input',{'features':2})];wires=[]
 def add(i,t,config,source,port='out',targetport='in'):rows.append(node(i,t,config));wires.append((source,port,i,targetport))
 if op in ('split','chunk'):
  sample=(2,5) if op=='split' else (2,13)
  if c.get('dim',-1)==1:sample=tuple(reversed(sample))
  add('project','linear',{'in_features':2,'out_features':math.prod(sample),'bias':True},'input');add('array','reshape',{'shape':list(sample)},'project');add('operation',op,c,'array');out=resolve(op,c,[BatchedLayout(sample)])['outputs'][part].sample;add('flat','flatten',{'start_dim':1,'end_dim':-1},'operation',part)
 elif op=='gather':
  add('project','linear',{'in_features':2,'out_features':8,'bias':True},'input');add('array','reshape',{'shape':[2,4]},'project');add('pool','max_pool1d',{'kernel_size':2,'stride':2,'padding':0,'dilation':1,'ceil_mode':False,'return_indices':True},'array');rows.append(node('operation','gather',c));wires.extend([('array','out','operation','source'),('pool','indices','operation','indices')]);out=(2,2);add('flat','flatten',{'start_dim':1,'end_dim':-1},'operation')
 else:
  canonical=normalize(op,c);operands,output=parse_equation(canonical['equation']);dimensions={'a':2,'b':3,'c':2}
  layouts=[]
  for i,term in enumerate(operands):
   shape=tuple(dimensions[k] for k in term[1:]);layouts.append(BatchedLayout(shape));add(f'projection{i}','linear',{'in_features':2,'out_features':math.prod(shape),'bias':True},'input');add(f'array{i}','reshape',{'shape':list(shape)},f'projection{i}')
  rows.append(node('operation',op,c));wires.extend((f'array{i}','out','operation','a' if i==0 else 'b') for i in range(len(operands)));out=resolve(op,c,layouts)['outputs']['out'].sample;add('flat','flatten',{'start_dim':1,'end_dim':-1},'operation')
 add('head','linear',{'in_features':math.prod(out),'out_features':2,'bias':True},'flat');add('output','output',{},'head')
 return {'schemaVersion':2,'name':'Ordered routing','nodes':rows,'edges':[{'id':str(i),'source':s,'sourcePort':p,'target':t,'targetPort':q} for i,(s,p,t,q) in enumerate(wires)]}

GRAPH_CASES=[('split',{},'part2'),('split',{'split_size':100},'part0'),('split',{'mode':'sections','sections':[1,3,1]},'part1'),('split',{'dim':1,'split_size':2},'part2'),('chunk',{'chunks':6},'part4'),('gather',{},'out'),*[("einsum",{'pattern':'custom','equation':equation},'out') for equation,a,b in EINSUM]]
@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('op,c,part',GRAPH_CASES)
def test_actual_graph_ordered_port_aliases_and_strict_portable_values_input_gradients(op,c,part,device):
 from runner.engine import GraphModel,REGISTRY,export_source
 model=GraphModel(graph(op,c,part),REGISTRY).to(device);source=export_source(model);assert 'runner.' not in source;scope={};exec(source,scope);portable=scope['Model']().to(device);portable.load_state_dict(model.state_dict(),strict=True);x=torch.tensor([[1.,-2.],[-.5,3.]],device=device,requires_grad=True);y=x.detach().clone().requires_grad_();weights=torch.tensor([[.5,-.25],[-.75,.25]],device=device)
 torch.testing.assert_close(model(x),portable(y),rtol=0,atol=0);torch.testing.assert_close(torch.autograd.grad(model(x),x,weights)[0],torch.autograd.grad(portable(y),y,weights)[0],rtol=0,atol=0);assert set(model.state_dict())==set(portable.state_dict())
 if op in ('split','chunk'):
  _,stages=model(x,inspect=True);assert list(stages['operation'])==list(model.layouts['operation'])

@pytest.mark.parametrize('op,c,part',[('split',{},'part3'),('chunk',{'chunks':6},'part5')])
def test_stale_saved_port_refuses_without_rerouting_or_removing_edges(op,c,part):
 from runner.engine import GraphModel,REGISTRY
 g=graph(op,c,'part0');next(e for e in g['edges'] if e['source']=='operation')['sourcePort']=part;before=copy.deepcopy(g)
 with pytest.raises(ValueError,match='Missing output'):GraphModel(g,REGISTRY)
 assert g==before

@pytest.mark.parametrize('op,c',[('split',{'mode':'bad'}),('split',{'sections':[0]}),('split',{'sections':[1]*9}),('split',{'split_size':True}),('split',{'split_size':0}),('split',{'split_size':100_000_001}),('chunk',{'chunks':0}),('chunk',{'chunks':9}),('chunk',{'chunks':1.5}),('gather',{'dim':0}),('gather',{'dim':True}),('einsum',{'pattern':'custom','equation':'x'*129}),('einsum',{'pattern':'bad'}),('einsum',{'equation':True})])
def test_strict_counts_choices_text_and_inactive_control_refusals(op,c):
 with pytest.raises(ValueError):normalize(op,c)

@pytest.mark.parametrize('op',['split','chunk','gather','einsum'])
def test_no_unknown_fields_or_observation_axes(op):
 with pytest.raises(ValueError):normalize(op,{'code':'eval(me)'})
 if op!='einsum':
  with pytest.raises(ValueError):resolve(op,{'dim':-3},[BatchedLayout((2,3))]* (2 if op=='gather' else 1))


def test_static_einsum_repeated_label_and_work_resource_refusals():
 for eq,shapes in [('naa->na',[(2,3)]),('nab,nbc->nac',[(2,3),(4,2)]),('nab,nbc->nac',[(1000,1000),(1000,1000)])]:
  with pytest.raises(ValueError):resolve('einsum',{'pattern':'custom','equation':eq},[BatchedLayout(s) for s in shapes])
 with pytest.raises(ValueError):resolve('split',{'split_size':1},[BatchedLayout((2,9))])
 with pytest.raises(ValueError):resolve('split',{'mode':'sections','sections':[1,2]},[BatchedLayout((2,5))])


def test_actual_contraction_work_refuses_before_finite_scan_or_native_call(monkeypatch):
 from runner.engine import _execute_tensor_routing
 c={'pattern':'multiply','equation':'nab,nbc->nac'};samples=[(100,100),(100,100)];resolved=resolve('einsum',c,[BatchedLayout(s) for s in samples]);tensors=[torch.zeros(1,*s).expand(101,*s) for s in samples];called=[]
 monkeypatch.setattr(torch,'isfinite',lambda *a,**k:called.append('scan'));monkeypatch.setattr(torch,'einsum',lambda *a,**k:called.append('native'))
 with pytest.raises(ValueError,match='work budget'):_execute_tensor_routing(tensors,'einsum',resolved['config'],samples,{'out':(100,100)},resolved['workPerObservation'])
 assert called==[]

@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('bad',[-1,4])
def test_actual_gather_invalid_selector_refuses_before_native_call(device,bad,monkeypatch):
 x=torch.ones(2,2,4,device=device);indices=torch.zeros(2,2,3,dtype=torch.int64,device=device);indices[0,1,2]=bad;called=[];monkeypatch.setattr(torch,'gather',lambda *a,**k:called.append('native'))
 with pytest.raises(ValueError,match='selectors'):run_native('gather',{'dim':2},[x,indices])
 assert called==[]

@pytest.mark.parametrize('op',['split','chunk','gather','einsum'])
@pytest.mark.parametrize('problem',['dtype','sparse','empty','nonfinite','budget'])
def test_actual_all_sources_boundary_refusals_include_unselected_parts(op,problem):
 from runner.engine import _execute_tensor_routing
 c=normalize(op,{});samples=[(2,4)] if op in ('split','chunk') else [(2,4),(2,3)] if op=='gather' else [(2,3),(3,2)]
 tensors=[torch.ones(2,*s,dtype=torch.int64 if op=='gather' and i==1 else torch.float32) for i,s in enumerate(samples)];r=resolve(op,c,[BatchedLayout(s) for s in samples]);outputs={p:l.sample for p,l in r['outputs'].items()}
 if problem=='dtype':tensors[0]=tensors[0].double()
 elif problem=='sparse':tensors[0]=tensors[0].to_sparse()
 elif problem=='empty':tensors=[t[:0] for t in tensors]
 elif problem=='nonfinite':tensors[0].flatten()[-1]=float('nan')
 else:tensors=[t[:1].expand(100_000_001,*s) for t,s in zip(tensors,samples)]
 with pytest.raises(ValueError):_execute_tensor_routing(tensors,op,c,samples,outputs,r['workPerObservation'])

@pytest.mark.parametrize('op',['gather','einsum'])
@pytest.mark.parametrize('problem',['batch','device'])
def test_actual_binary_batch_and_device_match(op,problem):
 c={};samples=[(2,4),(2,3)] if op=='gather' else [(2,3),(3,2)];tensors=[torch.ones(2,*s,dtype=torch.int64 if op=='gather' and i==1 else torch.float32) for i,s in enumerate(samples)]
 tensors[1]=tensors[1][:1] if problem=='batch' else tensors[1].to('mps')
 with pytest.raises(ValueError,match='observations and devices'):run_native(op,c,tensors)

@pytest.mark.parametrize('device',['cpu','mps'])
def test_einsum_finite_input_overflow_refuses(device):
 a=torch.full((2,2,3),3.4028234663852886e38,device=device);b=torch.full((2,3,2),2.,device=device)
 with pytest.raises(ValueError,match='nonfinite'):run_native('einsum',{},[a,b])

@pytest.mark.parametrize('op,c,part',[('split',{},'part2'),('chunk',{'chunks':6},'part4'),('gather',{},'out'),('einsum',{'pattern':'multiply'},'out')])
@pytest.mark.parametrize('device',['cpu','mps'])
def test_per_member_training_checkpoint_capture_portable_and_named_cache(op,c,part,device,tmp_path):
 from runner.engine import run,GraphModel,REGISTRY
 from runner.preview import preview
 g=graph(op,c,part);run(g,tmp_path/'training',epochs=1,sample_limit=32,batch_size=8,device=device,registry=REGISTRY,emit=lambda event:None,sample_capture={'schemaVersion':1,'activationValues':True,'predictionRows':True})
 manifest=json.loads((tmp_path/'training/manifest.json').read_text());assert manifest['runtime']['device']==device;assert manifest['settings']['deterministicAlgorithms']==(device=='cpu')
 checkpoint=torch.load(tmp_path/'training/checkpoint.pt',weights_only=True,map_location='cpu');model=GraphModel(g,REGISTRY).eval();model.load_state_dict(checkpoint['modelState'],strict=True);scope={};exec((tmp_path/'training/model.py').read_text(),scope);portable=scope['Model']().eval();portable.load_state_dict(checkpoint['modelState'],strict=True);torch.testing.assert_close(model(torch.ones(3,2)),portable(torch.ones(3,2)),rtol=0,atol=0)
 capture=json.loads((tmp_path/'training/activations.json').read_text())['previews']['operation'];assert ('outputs' in capture)==(op in ('split','chunk'))
 first=preview(g,tmp_path/'preview',registry=REGISTRY,device=device,checkpoint=tmp_path/'training/checkpoint.pt');again=preview(g,tmp_path/'again',registry=REGISTRY,device=device,checkpoint=tmp_path/'training/checkpoint.pt',resume_from='flat',cache=first['cachePath']);assert first['previews']['output']==again['previews']['output']
 if op in ('split','chunk'):
  cache=torch.load(first['cachePath'],weights_only=True,map_location='cpu');assert list(cache['activations']['operation'])==list(model.layouts['operation'])

@pytest.mark.parametrize('op',['split','chunk','gather','einsum'])
def test_authored_generated_discovery_exact_descriptor_and_input_policy(op,monkeypatch):
 import importlib
 from pathlib import Path
 from runner.engine import capabilities
 from runner.tensor_routing_configuration import capability
 root=Path(__file__).resolve().parents[2];bundle=json.loads((root/'components/tensor-routing/bundle.json').read_text());authored=next(row['runtimeContract'] for row in bundle['components'] if row['definition']['id']==op);monkeypatch.syspath_prepend(str(root/'runner'));table=importlib.import_module('runner.tensor_routing_adapters_generated');assert authored==capability(op)==table.CAPABILITIES[op]()==capabilities()['componentCapabilities'][op]
 for c in ({},{'pattern':'diagonal'} if op=='einsum' else {}):assert table.CONFIGURED_INPUTS[op](c)==configured_inputs(op,c)

@pytest.mark.parametrize('op',['split','chunk','gather','einsum'])
def test_every_routing_operator_refuses_schema_one(op):
 from runner.graph_semantics import effective_config
 with pytest.raises(ValueError):effective_config({'type':op,'config':{}},1)

@pytest.mark.parametrize('change',[{'minItems':0},{'maxItems':0},{'min':0},{'max':128},{'choices':[]},{'allowsNull':True},{'defaultValue':None}])
def test_actual_registry_text_field_contract_refuses_malformed_metadata(change,tmp_path):
 from runner.engine import GraphModel,REGISTRY
 registry=json.loads(open(REGISTRY).read());entry=next(row for row in registry['components'] if row['id']=='einsum');field=next(f for f in entry['fields'] if f['key']=='equation');field.update(change);path=tmp_path/'malformed-registry.json';path.write_text(json.dumps(registry))
 with pytest.raises(ValueError,match='malformed text'):GraphModel(graph('einsum'),path)
