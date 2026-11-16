"""Independent recurrent controls/geometry and scalar native module evidence."""
import importlib,math
import pytest
from runner.tensor_layout import BatchedLayout
OPS=('rnn','lstm','gru','rnn_cell','lstm_cell','gru_cell')
def helper():return importlib.import_module('runner.recurrent_configuration')
@pytest.mark.parametrize('op',OPS)
def test_defaults_ports_native_parameter_ownership(op):
 h=helper();c=h.normalize_recurrent(op,{})
 assert c['input_size']==2 and c['hidden_size']==2 and c['state_mode']=='zero' and c['bias'] is True
 assert h.configured_input_ports(op,c)==('in',)
 assert h.configured_input_ports(op,{**c,'state_mode':'supplied'})==(('in','hidden','cell') if op.startswith('lstm') else ('in','hidden'))
 r=h.resolve_recurrent(op,c,[BatchedLayout((3,2) if not op.endswith('_cell') else (2,))])
 assert r['outputs']['out'].sample==((3,2) if not op.endswith('_cell') else (2,))
 assert r['outputs']['hidden'].sample==((1,2) if not op.endswith('_cell') else (2,))
 gate=4 if op.startswith('lstm') else 3 if op.startswith('gru') else 1
 assert r['parameterCount']==gate*2*(2+2+2)
 assert [p['name'] for p in h.parameter_shapes(op,c)]==(['weight_ih','weight_hh','bias_ih','bias_hh'] if op.endswith('_cell') else ['weight_ih_l0','weight_hh_l0','bias_ih_l0','bias_hh_l0'])
@pytest.mark.parametrize('op',OPS)
@pytest.mark.parametrize('config',[{'input_size':True},{'hidden_size':0},{'bias':1},{'state_mode':'carry'},{'arbitrary':1}])
def test_malformed_config_refused(op,config):
 with pytest.raises(ValueError):helper().normalize_recurrent(op,config)
@pytest.mark.parametrize('op',['rnn','lstm','gru'])
def test_state_geometry_and_multilayer_bidirectional(op):
 h=helper();c={'input_size':3,'hidden_size':4,'num_layers':2,'bidirectional':True,'state_mode':'supplied'}
 sources=[BatchedLayout((5,3)),BatchedLayout((4,4))]+([BatchedLayout((4,4))] if op=='lstm' else [])
 r=h.resolve_recurrent(op,c,sources);assert r['outputs']['out'].sample==(5,8);assert r['outputs']['hidden'].sample==(4,4)
 names=[x['name'] for x in h.parameter_shapes(op,c)];assert names[:8]==['weight_ih_l0','weight_hh_l0','bias_ih_l0','bias_hh_l0','weight_ih_l0_reverse','weight_hh_l0_reverse','bias_ih_l0_reverse','bias_hh_l0_reverse']
 with pytest.raises(ValueError):h.resolve_recurrent(op,c,[sources[0],BatchedLayout((2,4))]+sources[2:])
 with pytest.raises(ValueError):h.resolve_recurrent(op,c,sources[:1])
def test_projection_and_allocation_guards():
 h=helper();r=h.resolve_recurrent('lstm',{'input_size':3,'hidden_size':4,'proj_size':2,'num_layers':2,'bidirectional':True},[BatchedLayout((3,3))])
 assert {k:v.sample for k,v in r['outputs'].items()}=={'out':(3,4),'hidden':(4,2),'cell':(4,4)}
 assert h.parameter_shapes('lstm',{'input_size':3,'hidden_size':4,'proj_size':2,'bias':False})[-1]=={'name':'weight_hr_l0','shape':(2,4)}
 for op,c,sample in [('lstm',{'proj_size':2},(3,2)),('rnn',{'num_layers':1,'dropout':.1},(3,2)),('gru',{'input_size':4096,'hidden_size':512,'num_layers':8,'bidirectional':True},(4096,4096))]:
  with pytest.raises(ValueError):h.resolve_recurrent(op,c,[BatchedLayout(sample)])

# Plain Python oracle: no adapter/native recurrent operation for expected values.
def scalar_reference(op,c,parameters,x,hidden=None,cell_state=None):
 cell=op.endswith('_cell');n=len(x);t=1 if cell else len(x[0]);l=c.get('num_layers',1);d=2 if c.get('bidirectional',False) else 1;h=c['hidden_size'];p=c.get('proj_size',0) or h
 seq=[[list(row)] if cell else [list(v) for v in row] for row in x];hs=[];cs=[]
 def sigmoid(v):return 1/(1+math.exp(-v))
 def dot(row,v):return sum(a*b for a,b in zip(row,v))
 for layer in range(l):
  outputs=[[[0.]*(d*p) for _ in range(t)] for _ in range(n)]
  for direction in range(d):
   key='' if cell else '_l'+str(layer)+('_reverse' if direction else '')
   wi=parameters['weight_ih'+key];wh=parameters['weight_hh'+key];bi=parameters.get('bias_ih'+key,[0.]*len(wi));bh=parameters.get('bias_hh'+key,[0.]*len(wh));final=[];finalc=[]
   for observation in range(n):
    state=list(hidden[observation] if cell else hidden[observation][layer*d+direction]) if hidden is not None else [0.]*p
    memory=list(cell_state[observation] if cell else cell_state[observation][layer*d+direction]) if cell_state is not None else [0.]*h
    for time in (range(t-1,-1,-1) if direction else range(t)):
     a=[dot(row,seq[observation][time])+b for row,b in zip(wi,bi)];b=[dot(row,state)+v for row,v in zip(wh,bh)]
     if op.startswith('lstm'):
      gates=[a[k]+b[k] for k in range(4*h)];memory=[sigmoid(gates[h+j])*memory[j]+sigmoid(gates[j])*math.tanh(gates[2*h+j]) for j in range(h)];raw=[sigmoid(gates[3*h+j])*math.tanh(memory[j]) for j in range(h)];state=[dot(row,raw) for row in parameters['weight_hr'+key]] if c.get('proj_size') else raw
     elif op.startswith('gru'):
      r=[sigmoid(a[j]+b[j]) for j in range(h)];z=[sigmoid(a[h+j]+b[h+j]) for j in range(h)];candidate=[math.tanh(a[2*h+j]+r[j]*b[2*h+j]) for j in range(h)];state=[(1-z[j])*candidate[j]+z[j]*state[j] for j in range(h)]
     else:
      raw=[a[j]+b[j] for j in range(h)];state=[max(0.,v) if c.get('nonlinearity')=='relu' else math.tanh(v) for v in raw]
     outputs[observation][time][direction*p:(direction+1)*p]=state
    final.append(state);finalc.append(memory)
   hs.append(final);cs.append(finalc)
  seq=outputs
 result={'out':[row[0] for row in seq] if cell else seq,'hidden':[row[0] for row in seq] if cell else [[hs[k][i] for k in range(l*d)] for i in range(n)]}
 if op.startswith('lstm'):result['cell']=[cs[0][i] for i in range(n)] if cell else [[cs[k][i] for k in range(l*d)] for i in range(n)]
 return result

def leaves(value):
 if isinstance(value,list):
  for item in value:yield from leaves(item)
 else:yield value

def scalar_loss(value):return sum(sum(v*(.15+.07*i) for i,v in enumerate(leaves(array))) for array in value.values())

def derivative(value,compute):
 result=[]
 def visit(v):
  for i,item in enumerate(v):
   if isinstance(item,list):visit(item)
   else:
    v[i]=item+1e-5;positive=compute();v[i]=item-1e-5;negative=compute();v[i]=item;result.append((positive-negative)/2e-5)
 visit(value);return result

@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('op',OPS)
@pytest.mark.parametrize('bias',[True,False])
@pytest.mark.parametrize('state_mode',['supplied','zero'])
def test_independent_scalar_outputs_and_every_owned_gradient(op,device,bias,state_mode):
 import torch
 if device=='mps' and not torch.backends.mps.is_available():pytest.fail('Required actual MPS device unavailable')
 previous=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled();torch.use_deterministic_algorithms(device=='cpu')
 try:
  h=helper();c=h.normalize_recurrent(op,{'input_size':2,'hidden_size':2,'state_mode':state_mode,'bias':bias,**({'num_layers':2,'bidirectional':True} if not op.endswith('_cell') else {}),**({'nonlinearity':'relu'} if op in ('rnn','rnn_cell') else {})})
  class_name={'rnn':'RNN','lstm':'LSTM','gru':'GRU','rnn_cell':'RNNCell','lstm_cell':'LSTMCell','gru_cell':'GRUCell'}[op]
  kwargs={k:v for k,v in c.items() if k!='state_mode'}
  if not op.endswith('_cell'):kwargs['batch_first']=True
  module=getattr(torch.nn,class_name)(**kwargs).to(device)
  parameters={}
  with torch.no_grad():
   for index,(name,parameter) in enumerate(module.named_parameters()):
    parameter.copy_(torch.linspace(.025+index*.002,.125+index*.002,parameter.numel(),device=device).reshape(parameter.shape));parameters[name]=parameter.cpu().tolist()
  x=torch.tensor([[.3,-.2]] if op.endswith('_cell') else [[[.3,-.2],[.2,.4]]],device=device,requires_grad=True)
  state=torch.tensor([[.1,.2]] if op.endswith('_cell') else [[[.1,.2]]*4],device=device,requires_grad=True)
  memory=torch.tensor([[.2,.1]] if op.endswith('_cell') else [[[.2,.1]]*4],device=device,requires_grad=True) if op.startswith('lstm') else None
  operands=[x]+([state]+([memory] if memory is not None else []) if state_mode=='supplied' else []);layouts=[BatchedLayout(tuple(v.shape[1:])) for v in operands];resolved=h.resolve_recurrent(op,c,layouts)
  engine=importlib.import_module('runner.engine');actual=engine._execute_recurrent(operands,op,c,[v.sample for v in layouts],{k:v.sample for k,v in resolved['outputs'].items()},resolved['gateElementsPerObservation'],resolved['workPerObservation'],module)
  xp=x.detach().cpu().tolist();hp=state.detach().cpu().tolist() if state_mode=='supplied' else None;cp=memory.detach().cpu().tolist() if memory is not None and state_mode=='supplied' else None
  reference=lambda:scalar_reference(op,c,parameters,xp,hp,cp);expected=reference()
  for key in actual:assert actual[key].detach().cpu().flatten().tolist()==pytest.approx(list(leaves(expected[key])),abs=2e-6,rel=2e-5)
  sum((v*torch.arange(v.numel(),device=device).reshape(v.shape).mul(.07).add(.15)).sum() for v in actual.values()).backward()
  compute=lambda:scalar_loss(reference())
  for tensor,plain in [(x,xp)]+([(state,hp)]+([(memory,cp)] if memory is not None else []) if state_mode=='supplied' else []):assert tensor.grad.cpu().flatten().tolist()==pytest.approx(derivative(plain,compute),abs=3e-6,rel=3e-5)
  for name,parameter in module.named_parameters():assert parameter.grad.cpu().flatten().tolist()==pytest.approx(derivative(parameters[name],compute),abs=3e-6,rel=3e-5)
  if op.endswith('_cell'):assert actual['out'] is actual['hidden']
 finally:torch.use_deterministic_algorithms(previous,warn_only=warn)

@pytest.fixture(autouse=True)
def isolate_actual_device_policy(request):
 import torch
 call=getattr(request.node,'callspec',None);device=call.params.get('device') if call else None
 if device not in ('cpu','mps'):yield;return
 prior=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled()
 try:torch.use_deterministic_algorithms(device=='cpu');yield
 finally:torch.use_deterministic_algorithms(prior,warn_only=warn)

def classifier(op,state_mode='zero',output_port='out',extra=None):
 cell=op.endswith('_cell');c={'input_size':2 if cell else 1,'hidden_size':2,'state_mode':state_mode,**(extra or {})}
 nodes=[{'id':'input','type':'input','config':{'features':2}}];edges=[]
 def edge(a,b,port='in',source='out'):edges.append({'id':str(len(edges)),'source':a,'sourcePort':source,'target':b,'targetPort':port})
 if not cell:nodes.append({'id':'sequence','type':'reshape','config':{'shape':[2,1]}});edge('input','sequence')
 nodes.append({'id':'recurrent','type':op,'config':c});edge('input' if cell else 'sequence','recurrent')
 if state_mode=='supplied':
  if not cell:nodes.append({'id':'initial','type':'reshape','config':{'shape':[1,2]}});edge('input','initial')
  edge('input' if cell else 'initial','recurrent','hidden')
  if op.startswith('lstm'):edge('input' if cell else 'initial','recurrent','cell')
 p=c.get('proj_size',0) or c['hidden_size'];d=2 if c.get('bidirectional',False) else 1;l=c.get('num_layers',1)
 width=c['hidden_size'] if cell else 2*d*p if output_port=='out' else l*d*(c['hidden_size'] if output_port=='cell' else p)
 nodes.extend([{'id':'flatten','type':'flatten','config':{'start_dim':1,'end_dim':-1}},{'id':'head','type':'linear','config':{'in_features':width,'out_features':2,'bias':True}},{'id':'output','type':'output','config':{}}]);edge('recurrent','flatten',source=output_port);edge('flatten','head');edge('head','output')
 return {'schemaVersion':2,'nodes':nodes,'edges':edges}

@pytest.mark.parametrize('op',OPS)
@pytest.mark.parametrize('state_mode',['zero','supplied'])
@pytest.mark.parametrize('device',['cpu','mps'])
def test_genuine_registry_named_graph_and_selfcontained_export(op,state_mode,device):
 import torch
 from runner.engine import GraphModel,export_source,capabilities
 model=GraphModel(classifier(op,state_mode),bundled_dataset=False).to(device).eval();x=torch.tensor([[.4,-.2],[-.3,.1]],device=device,requires_grad=True)
 assert model.nodes['recurrent']['config']==helper().normalize_recurrent(op,classifier(op,state_mode)['nodes'][2 if not op.endswith('_cell') else 1]['config'])
 y=model(x);ns={};code=export_source(model);assert 'runner.' not in code;exec(code,ns);exported=ns['Model']().to(device).eval();exported.load_state_dict(model.state_dict(),strict=True);z=exported(x);torch.testing.assert_close(z,y,rtol=0,atol=0)
 runtime_grad=torch.autograd.grad(y.square().sum(),[x,*model.parameters()]);portable_grad=torch.autograd.grad(z.square().sum(),[x,*exported.parameters()])
 for a,b in zip(runtime_grad,portable_grad):torch.testing.assert_close(a,b,rtol=0,atol=0)
 assert capabilities()['componentCapabilities'][op]==helper().capability(op)
 assert [name.split('layers.recurrent.')[1] for name,_ in model.named_parameters() if name.startswith('layers.recurrent.')]==[p['name'] for p in helper().parameter_shapes(op,model.nodes['recurrent']['config'])]
 for port in ('hidden','cell') if op.startswith('lstm') else ('hidden',):assert GraphModel(classifier(op,state_mode,port),bundled_dataset=False).to(device)(x).shape==(2,2)

@pytest.mark.parametrize('op',OPS)
@pytest.mark.parametrize('problem',['dtype','batch','nonfinite','width','state','parameter','resource'])
def test_execution_refuses_before_native_forward(op,problem,monkeypatch):
 import torch
 from runner.engine import _execute_recurrent
 h=helper();c=h.normalize_recurrent(op,{'state_mode':'supplied'});cell=op.endswith('_cell');samples=[(2,) if cell else (2,2),(2,) if cell else (1,2)]+([(2,) if cell else (1,2)] if op.startswith('lstm') else []);resolved=h.resolve_recurrent(op,c,[BatchedLayout(v) for v in samples])
 name={'rnn':'RNN','lstm':'LSTM','gru':'GRU','rnn_cell':'RNNCell','lstm_cell':'LSTMCell','gru_cell':'GRUCell'}[op];kw={k:v for k,v in c.items() if k!='state_mode'}
 if not cell:kw['batch_first']=True
 module=getattr(torch.nn,name)(**kw);sources=[torch.ones(1,*v) for v in samples];work=resolved['workPerObservation']
 if problem=='dtype':sources[0]=sources[0].double()
 if problem=='batch':sources[0]=sources[0][:0]
 if problem=='nonfinite':sources[0].flatten()[0]=float('nan')
 if problem=='width':sources[0]=torch.ones(1,*samples[0][:-1],3)
 if problem=='state':sources[1]=torch.ones(2,*samples[1])
 if problem=='parameter':next(module.parameters()).data.flatten()[0]=float('inf')
 if problem=='resource':work=100_000_001
 monkeypatch.setattr(module,'forward',lambda *args:pytest.fail('Invalid source reached native recurrent forward'))
 with pytest.raises(ValueError):_execute_recurrent(sources,op,c,samples,{k:v.sample for k,v in resolved['outputs'].items()},resolved['gateElementsPerObservation'],work,module)

@pytest.mark.parametrize('op',OPS)
def test_cpu_exact_resume_and_finetune_actual_owned_parameters(op,tmp_path):
 import json,torch
 from runner.engine import run
 from runner.tests.test_checkpoint_lifecycle import assert_nested_equal
 g=classifier(op);options={'sample_limit':32,'batch_size':8,'device':'cpu','emit':lambda event:None}
 for name,epochs,parent in [('full',2,None),('first',1,None),('resume',2,tmp_path/'first/checkpoint.pt')]:run(g,tmp_path/name,epochs=epochs,resume_checkpoint=parent,**options)
 a=torch.load(tmp_path/'full/checkpoint.pt',weights_only=True);b=torch.load(tmp_path/'resume/checkpoint.pt',weights_only=True)
 for key in ('modelState','optimizerState','rngState'):assert_nested_equal(a[key],b[key])
 assert (tmp_path/'full/metrics.json').read_bytes()==(tmp_path/'resume/metrics.json').read_bytes()
 assert a['contract']['schedule']['recurrentNodeOrder']==['recurrent'];assert a['contract']['schedule']['parameterNodeOrder']==['recurrent','head']
 run(g,tmp_path/'fine',epochs=1,seed=173,finetune_checkpoint=tmp_path/'first/checkpoint.pt',**options)
 fine=json.loads((tmp_path/'fine/manifest.json').read_text());assert fine['lineage']['restoredState']==['parameters'];assert fine['lineage']['resetState']==['optimizer','randomGenerator','epoch','metrics']
 assert {k for k in a['modelState'] if k.startswith('layers.recurrent.')}=={'layers.recurrent.'+p['name'] for p in helper().parameter_shapes(op,g['nodes'][2 if not op.endswith('_cell') else 1]['config'])}

@pytest.mark.parametrize('op',OPS)
def test_mps_actual_training_finetune_and_exact_resume_refusal(op,tmp_path):
 import json
 from runner.engine import run
 options={'sample_limit':32,'batch_size':8,'device':'mps','emit':lambda event:None};g=classifier(op)
 run(g,tmp_path/'first',epochs=1,sample_capture={'schemaVersion':1,'activationValues':True,'predictionRows':True},**options)
 manifest=json.loads((tmp_path/'first/manifest.json').read_text());assert manifest['runtime']['device']=='mps';assert manifest['checkpoint']['contract']['guarantee']=='weightsOnly';assert manifest['settings']['deterministicAlgorithms'] is False
 run(g,tmp_path/'fine',epochs=1,seed=173,finetune_checkpoint=tmp_path/'first/checkpoint.pt',**options)
 with pytest.raises(ValueError,match='Exact resume'):run(g,tmp_path/'resume',epochs=2,resume_checkpoint=tmp_path/'first/checkpoint.pt',**options)

@pytest.mark.parametrize('op',['rnn','lstm','gru'])
def test_registered_dropout_modes_and_cpu_rng_continuation(op,tmp_path):
 import torch
 from runner.engine import run,GraphModel
 from runner.tests.test_checkpoint_lifecycle import assert_nested_equal
 g=classifier(op,extra={'num_layers':2,'dropout':.5});model=GraphModel(g);x=torch.tensor([[.2,-.3],[.4,.5]])
 torch.manual_seed(911);a=model(x);saved=torch.get_rng_state();b=model(x);torch.set_rng_state(saved);replayed=model(x)
 assert not torch.equal(a,b);torch.testing.assert_close(b,replayed,rtol=0,atol=0);model.eval();torch.testing.assert_close(model(x),model(x),rtol=0,atol=0)
 options={'sample_limit':32,'batch_size':8,'device':'cpu','emit':lambda event:None}
 for name,epochs,parent in [('full',2,None),('first',1,None),('resume',2,tmp_path/'first/checkpoint.pt')]:run(g,tmp_path/name,epochs=epochs,resume_checkpoint=parent,**options)
 a=torch.load(tmp_path/'full/checkpoint.pt',weights_only=True);b=torch.load(tmp_path/'resume/checkpoint.pt',weights_only=True)
 for key in ('modelState','optimizerState','rngState'):assert_nested_equal(a[key],b[key])
 assert a['contract']['schedule']['stochasticNodeOrder']==['recurrent']
 assert (tmp_path/'full/metrics.json').read_bytes()==(tmp_path/'resume/metrics.json').read_bytes()

@pytest.mark.parametrize('problem',['missing','disabled','rank','dtype'])
def test_real_graph_conditional_state_admission(problem):
 from runner.engine import GraphModel
 g=classifier('lstm',state_mode='supplied')
 if problem=='missing':g['edges']=[e for e in g['edges'] if e['targetPort']!='cell']
 if problem=='disabled':next(n for n in g['nodes'] if n['id']=='recurrent')['config']['state_mode']='zero'
 if problem=='rank':next(n for n in g['nodes'] if n['id']=='initial')['config']['shape']=[2]
 if problem=='dtype':
  g=classifier('rnn_cell',state_mode='supplied');g['nodes'][0]={'id':'input','type':'index_input','config':{'shape':[2]}}
 with pytest.raises(ValueError):GraphModel(g,bundled_dataset=False)

def test_projected_lstm_cpu_independent_all_gradients_and_export(tmp_path):
 import torch
 from runner.engine import _execute_recurrent,GraphModel,export_source,run
 h=helper();c=h.normalize_recurrent('lstm',{'proj_size':1,'state_mode':'supplied'});module=torch.nn.LSTM(2,2,batch_first=True,proj_size=1)
 parameters={}
 with torch.no_grad():
  for index,(name,p) in enumerate(module.named_parameters()):p.copy_(torch.linspace(.03,.13,p.numel()).reshape(p.shape)+index*.01);parameters[name]=p.tolist()
 x=torch.tensor([[[.2,-.1],[.3,.4]]],requires_grad=True);hidden=torch.tensor([[[.1]]],requires_grad=True);cell=torch.tensor([[[.2,.1]]],requires_grad=True);sources=[x,hidden,cell];layouts=[BatchedLayout(tuple(v.shape[1:])) for v in sources];r=h.resolve_recurrent('lstm',c,layouts)
 actual=_execute_recurrent(sources,'lstm',c,[v.sample for v in layouts],{k:v.sample for k,v in r['outputs'].items()},r['gateElementsPerObservation'],r['workPerObservation'],module)
 xp=x.detach().tolist();hp=hidden.detach().tolist();cp=cell.detach().tolist();reference=lambda:scalar_reference('lstm',c,parameters,xp,hp,cp)
 for key,v in actual.items():assert v.detach().flatten().tolist()==pytest.approx(list(leaves(reference()[key])),abs=2e-6)
 sum((v*torch.arange(v.numel()).reshape(v.shape).mul(.07).add(.15)).sum() for v in actual.values()).backward();compute=lambda:scalar_loss(reference())
 for tensor,plain in [(x,xp),(hidden,hp),(cell,cp)]:assert tensor.grad.flatten().tolist()==pytest.approx(derivative(plain,compute),abs=3e-6)
 for name,p in module.named_parameters():assert p.grad.flatten().tolist()==pytest.approx(derivative(parameters[name],compute),abs=3e-6)
 g=classifier('lstm',extra={'proj_size':1});model=GraphModel(g).eval();scope={};exec(export_source(model),scope);portable=scope['Model']().eval();portable.load_state_dict(model.state_dict(),strict=True);features=torch.tensor([[.2,.4],[-.1,.3]]);torch.testing.assert_close(model(features),portable(features),rtol=0,atol=0)
 run(g,tmp_path/'cpu',epochs=1,sample_limit=32,batch_size=8,device='cpu',emit=lambda event:None)

def test_projected_lstm_mps_refuses_before_training_model_mutation(tmp_path,monkeypatch):
 import torch
 import runner.engine as engine
 g=classifier('lstm',extra={'proj_size':1});scope={};exec(engine.export_source(engine.GraphModel(g)),scope);portable=scope['Model']().to('mps')
 monkeypatch.setattr(engine,'GraphModel',lambda *args,**kwargs:pytest.fail('Projected MPS policy should refuse before creating model'))
 with pytest.raises(ValueError,match='CPU'):engine.run(g,tmp_path/'mps',epochs=1,sample_limit=32,batch_size=8,device='mps',emit=lambda event:None)
 assert not (tmp_path/'mps').exists()
 c=helper().normalize_recurrent('lstm',{'proj_size':1});r=helper().resolve_recurrent('lstm',c,[BatchedLayout((2,2))]);native=torch.nn.LSTM(2,2,batch_first=True,proj_size=1).to('mps')
 with pytest.raises(ValueError,match='CPU'):engine._execute_recurrent([torch.ones(1,2,2,device='mps')],'lstm',c,[(2,2)],{k:v.sample for k,v in r['outputs'].items()},r['gateElementsPerObservation'],r['workPerObservation'],native)
 with pytest.raises(ValueError,match='CPU'):portable(torch.ones(1,2,device='mps'))

@pytest.mark.parametrize('portable',[False,True])
def test_hash_correct_malformed_state_dtype_is_not_silently_cast(portable,tmp_path):
 import torch,json,hashlib
 from runner.engine import GraphModel,export_source,run
 g=classifier('lstm');model=GraphModel(g)
 if portable:
  scope={};exec(export_source(model),scope);target=scope['Model']();state=model.state_dict();state['layers.recurrent.weight_ih_l0']=state['layers.recurrent.weight_ih_l0'].double()
  with pytest.raises(ValueError,match='state'):target.load_state_dict(state,strict=True)
 else:
  run(g,tmp_path/'first',epochs=1,sample_limit=32,batch_size=8,device='cpu',emit=lambda event:None)
  path=tmp_path/'first/checkpoint.pt';checkpoint=torch.load(path,weights_only=True);checkpoint['modelState']['layers.recurrent.weight_ih_l0']=checkpoint['modelState']['layers.recurrent.weight_ih_l0'].double();torch.save(checkpoint,path)
  manifest=json.loads((tmp_path/'first/manifest.json').read_text());manifest['checkpoint']['bytes']=path.stat().st_size;manifest['checkpoint']['sha256']=hashlib.sha256(path.read_bytes()).hexdigest();(tmp_path/'first/manifest.json').write_text(json.dumps(manifest))
  with pytest.raises(ValueError,match='state|dtype'):run(g,tmp_path/'fine',epochs=1,sample_limit=32,batch_size=8,device='cpu',finetune_checkpoint=path,emit=lambda event:None)

@pytest.mark.parametrize('op',OPS)
@pytest.mark.parametrize('device',['cpu','mps'])
def test_strided_inputs_states_explicit_carry_and_no_mutable_forward_state(op,device):
 import torch,copy
 from runner.engine import GraphModel,export_source
 g=classifier(op);cell=op.endswith('_cell');first=next(n for n in g['nodes'] if n['id']=='recurrent')
 second={'id':'next','type':op,'config':{'input_size':2,'hidden_size':2,'state_mode':'supplied'}};g['nodes'].insert(g['nodes'].index(first)+1,second)
 for e in g['edges']:
  if e['source']=='recurrent':e['source']='next'
 for port,source in [('in','out'),('hidden','hidden')]+([('cell','cell')] if op.startswith('lstm') else []):g['edges'].append({'id':str(len(g['edges'])),'source':'recurrent','sourcePort':source,'target':'next','targetPort':port})
 model=GraphModel(g).to(device).eval();features=torch.tensor([[.3,0.,-.1,0.],[.2,0.,.4,0.]],device=device);x=features[:,::2];before=x.clone();weights={k:v.clone() for k,v in model.state_dict().items()}
 y=model(x);repeat=model(x);torch.testing.assert_close(y,repeat,rtol=0,atol=0);assert torch.equal(x,before)
 for key,value in model.state_dict().items():assert torch.equal(value,weights[key])
 scope={};exec(export_source(model),scope);portable=scope['Model']().to(device).eval();portable.load_state_dict(model.state_dict(),strict=True);torch.testing.assert_close(portable(x),y,rtol=0,atol=0)

@pytest.mark.parametrize('op',OPS)
def test_actual_named_state_preview_cache_roundtrip(op,tmp_path):
 from runner.preview import preview
 g=classifier(op,state_mode='supplied');first=preview(g,tmp_path/'before',device='cpu');second=preview(g,tmp_path/'after',device='cpu',resume_from='flatten',cache=first['cachePath'])
 assert first['previews']['output']==second['previews']['output']

@pytest.mark.parametrize('portable',[False,True])
@pytest.mark.parametrize('problem',['missing','extra','shape','sparse','nonfinite'])
def test_exact_composed_state_admission_before_any_copy(portable,problem):
 import torch
 from runner.engine import GraphModel,export_source
 model=GraphModel(classifier('gru'));state={k:v.clone() for k,v in model.state_dict().items()};key='layers.recurrent.weight_hh_l0'
 if problem=='missing':del state[key]
 elif problem=='extra':state['hidden']=torch.zeros(1,2)
 elif problem=='shape':state[key]=torch.zeros(1)
 elif problem=='sparse':state[key]=state[key].to_sparse()
 else:state[key].flatten()[0]=float('nan')
 if portable:scope={};exec(export_source(model),scope);model=scope['Model']()
 before={k:v.clone() for k,v in model.state_dict().items()}
 with pytest.raises(ValueError,match='state'):model.load_state_dict(state,strict=True)
 for name,value in model.state_dict().items():assert torch.equal(value,before[name])

@pytest.mark.parametrize('op',['rnn','lstm','gru'])
def test_mps_registered_multilayer_dropout_training_weights_only_policy(op,tmp_path):
 import json
 from runner.engine import run
 g=classifier(op,extra={'num_layers':2,'bidirectional':True,'dropout':.4});options={'device':'mps','sample_limit':32,'batch_size':8,'emit':lambda e:None}
 run(g,tmp_path/'first',epochs=1,**options);manifest=json.loads((tmp_path/'first/manifest.json').read_text());assert manifest['checkpoint']['contract']['schedule']['stochasticNodeOrder']==['recurrent'];assert manifest['settings']['deterministicAlgorithms'] is False
 run(g,tmp_path/'fine',epochs=1,finetune_checkpoint=tmp_path/'first/checkpoint.pt',**options)
 with pytest.raises(ValueError,match='Exact resume'):run(g,tmp_path/'resume',epochs=2,resume_checkpoint=tmp_path/'first/checkpoint.pt',**options)

def test_projected_cpu_exact_resume_with_adamw_and_scheduler(tmp_path):
 import torch,json
 from runner.engine import run
 from runner.optimizer_configuration import canonical_configuration as optimizer_configuration
 from runner.scheduler_configuration import canonical_configuration as scheduler_configuration
 from runner.tests.test_checkpoint_lifecycle import assert_nested_equal
 g=classifier('lstm',extra={'proj_size':1,'num_layers':2,'bidirectional':True});recipe={'schemaVersion':1,'componentID':'optimizer_adamw','configurationVersion':1,'config':optimizer_configuration('optimizer_adamw',{'lr':.02})};schedule={'schemaVersion':1,'componentID':'scheduler_step_lr','configurationVersion':1,'config':scheduler_configuration('scheduler_step_lr',{'step_size':1,'gamma':.5})}
 options={'device':'cpu','sample_limit':32,'batch_size':8,'emit':lambda e:None,'optimizer_recipe':recipe,'scheduler_recipe':schedule}
 for name,epochs,parent in [('full',2,None),('first',1,None),('resume',2,tmp_path/'first/checkpoint.pt')]:run(g,tmp_path/name,epochs=epochs,resume_checkpoint=parent,**options)
 full=torch.load(tmp_path/'full/checkpoint.pt',weights_only=True);resumed=torch.load(tmp_path/'resume/checkpoint.pt',weights_only=True)
 for key in ('modelState','optimizerState','optimizerUpdates','scheduler','rngState'):assert_nested_equal(full[key],resumed[key])
 assert (tmp_path/'full/metrics.json').read_bytes()==(tmp_path/'resume/metrics.json').read_bytes()
 assert (tmp_path/'full/rate-trace.json').read_bytes()==(tmp_path/'resume/rate-trace.json').read_bytes()
 manifest=json.loads((tmp_path/'full/manifest.json').read_text());names=[p['name'] for p in manifest['provenance']['optimizer']['parameterOwnership']];assert 'layers.recurrent.weight_hr_l0' in names and 'layers.recurrent.weight_hr_l1_reverse' in names

@pytest.mark.parametrize('op',OPS)
def test_actual_authored_generated_descriptor_and_canonical_hash_equivalence(op,monkeypatch):
 import json
 from runner.engine import REGISTRY,capabilities
 from runner.graph_semantics import semantic_graph
 monkeypatch.syspath_prepend(str(__import__('pathlib').Path('runner').resolve()))
 from runner.recurrent_adapters_generated import CAPABILITIES,NORMALIZERS,RESOLVERS,CONFIGURED_INPUTS
 entries=json.loads(REGISTRY.read_text());row=next(x for x in entries['components'] if x['id']==op) if isinstance(entries,dict) else next(x for x in entries if x['id']==op)
 bundle=json.loads(__import__('pathlib').Path('components/recurrent/bundle.json').read_text());authored=next(x for x in bundle['components'] if x['definition']['id']==op)
 assert authored['runtimeContract']==helper().capability(op)==CAPABILITIES[op]()==capabilities()['componentCapabilities'][op]
 assert NORMALIZERS[op]({})==helper().normalize_recurrent(op,{}) and CONFIGURED_INPUTS[op]({})==('in',)
 from tensor_layout import BatchedLayout as WorkerLayout
 assert RESOLVERS[op]({},[WorkerLayout((3,2) if not op.endswith('_cell') else (2,))])['parameterCount']>0
 g=classifier(op);node=next(x for x in g['nodes'] if x['type']==op);before=semantic_graph(g);node['config']=helper().normalize_recurrent(op,node['config']);assert semantic_graph(g)==before
