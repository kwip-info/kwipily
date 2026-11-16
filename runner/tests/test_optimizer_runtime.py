"""Independent scalar multi-step updates, strict options/ownership and lifecycle."""
import copy,json,math
from pathlib import Path
import pytest,torch
from runner.optimizer_configuration import canonical_configuration,validate_recipe,create_optimizer,restore_optimizer,training_recipe_code
IDS=['optimizer_sgd','optimizer_adam','optimizer_adamw','optimizer_rmsprop','optimizer_adagrad']
def recipe(op,config=None):return {'schemaVersion':1,'componentID':op,'configurationVersion':1,'config':config or {}}
@pytest.fixture(autouse=True)
def isolate_device_policy(request):
 calls=getattr(request.node,'callspec',None);device=None if calls is None else calls.params.get('device')
 if device not in ('cpu','mps'):yield;return
 previous=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled()
 try:torch.use_deterministic_algorithms(device=='cpu');yield
 finally:torch.use_deterministic_algorithms(previous,warn_only=warn)

@pytest.mark.parametrize('op',IDS)
def test_canonical_defaults_and_recipe_identity(op):
 c=canonical_configuration(op,{});assert c['lr'] in (.001,.01);assert validate_recipe(recipe(op))==recipe(op,c)
@pytest.mark.parametrize('op',IDS)
def test_unknown_boolean_nonfinite_controls_refuse(op):
 for c in [{'unknown':1},{'lr':True},{'lr':float('inf')},{'lr':-1}]:
  with pytest.raises(ValueError):canonical_configuration(op,c)

def scalar_update(op,c,values,grad,state):
 if grad is None:return values,state
 op=op.removeprefix('optimizer_');s=copy.deepcopy(state);t=s.get('_updates',0)+1;s['_updates']=t;out=[]
 for i,(p,g) in enumerate(zip(values,grad)):
  if c['maximize']:g=-g
  if op in ('adamw',) or (op=='adam' and c['decoupled_weight_decay']):p*=1-c['lr']*c['weight_decay']
  else:g+=c['weight_decay']*p
  def prior(name,initial=0.):return s.setdefault(name,[initial]*len(values))[i]
  def put(name,x):s.setdefault(name,[0.]*len(values))[i]=x;return x
  if op=='sgd':
   if c['momentum']:
    b=put('momentum_buffer',g if t==1 else c['momentum']*prior('momentum_buffer')+(1-c['dampening'])*g);g=g+c['momentum']*b if c['nesterov'] else b
   out.append(p-c['lr']*g)
  elif op in ('adam','adamw'):
   m=put('exp_avg',c['beta1']*prior('exp_avg')+(1-c['beta1'])*g);v=put('exp_avg_sq',c['beta2']*prior('exp_avg_sq')+(1-c['beta2'])*g*g)
   if c['amsgrad']:v=put('max_exp_avg_sq',max(prior('max_exp_avg_sq'),v))
   out.append(p-c['lr']*m/(1-c['beta1']**t)/(math.sqrt(v/(1-c['beta2']**t))+c['eps']))
  elif op=='rmsprop':
   v=put('square_avg',c['alpha']*prior('square_avg')+(1-c['alpha'])*g*g)
   if c['centered']:m=put('grad_avg',c['alpha']*prior('grad_avg')+(1-c['alpha'])*g);v-=m*m
   d=math.sqrt(v)+c['eps'];adjust=g/d
   if c['momentum']:adjust=put('momentum_buffer',c['momentum']*prior('momentum_buffer')+adjust)
   out.append(p-c['lr']*adjust)
  else:
   v=put('sum',prior('sum',c['initial_accumulator_value'])+g*g);out.append(p-c['lr']/(1+(t-1)*c['lr_decay'])*g/(math.sqrt(v)+c['eps']))
 return out,s

CASES=[('optimizer_sgd',{}),('optimizer_sgd',{'momentum':.6,'dampening':.2,'weight_decay':.15,'maximize':True}),('optimizer_sgd',{'momentum':.8,'nesterov':True,'lr':.07}),('optimizer_adam',{}),('optimizer_adam',{'beta1':.4,'beta2':.7,'amsgrad':True,'decoupled_weight_decay':True,'weight_decay':.2,'maximize':True,'eps':.01}),('optimizer_adam',{'weight_decay':.2,'beta1':0.,'beta2':0.}),('optimizer_adamw',{}),('optimizer_adamw',{'beta1':.4,'beta2':.7,'amsgrad':True,'weight_decay':.2,'maximize':True,'eps':.01}),('optimizer_rmsprop',{}),('optimizer_rmsprop',{'alpha':.4,'momentum':.6,'centered':True,'weight_decay':.2,'maximize':True,'eps':.01}),('optimizer_rmsprop',{'alpha':1.,'momentum':1.,'eps':.1}),('optimizer_adagrad',{}),('optimizer_adagrad',{'lr_decay':.3,'initial_accumulator_value':.4,'weight_decay':.2,'maximize':True,'eps':.01})]
@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('op,supplied',CASES)
def test_independent_changing_updates_state_and_no_gradient(device,op,supplied):
 from runner.optimizer_configuration import validate_optimizer_state
 c=canonical_configuration(op,supplied);r=recipe(op,c);params=[torch.nn.Parameter(torch.tensor([.8,-1.2],device=device)),torch.nn.Parameter(torch.tensor([.4],device=device))];opt=create_optimizer(params,r);expected=[[.8,-1.2],[.4]];states=[{},{}];counts=[0,0]
 gradients=[([.3,-.5],[.2]),([-.8,.1],None),([.15,.6],[-.4]),([-.25,-.2],[.7])]
 validate_optimizer_state(opt,opt.state_dict(),r,counts)
 for pair in gradients:
  for i,g in enumerate(pair):params[i].grad=None if g is None else torch.tensor(g,device=device);expected[i],states[i]=scalar_update(op,c,expected[i],g,states[i]);counts[i]+=g is not None
  opt.step();validate_optimizer_state(opt,opt.state_dict(),r,counts)
  for param,w,s,count in zip(params,expected,states,counts):
   torch.testing.assert_close(param.cpu(),torch.tensor(w),atol=3e-5,rtol=3e-5);item=opt.state.get(param,{})
   if op=='optimizer_adagrad' and count==0:assert set(item)=={'step','sum'}
   for key,value in item.items():
    if key=='step':assert value.item()==count and value.device.type=='cpu' and value.dtype==torch.float32
    else:torch.testing.assert_close(value.cpu(),torch.tensor(s[key]),atol=3e-5,rtol=3e-5)

@pytest.mark.parametrize('op',IDS)
@pytest.mark.parametrize('device',['cpu','mps'])
def test_zero_rate_and_frozen_buffer_exclusion(device,op):
 p=torch.nn.Parameter(torch.tensor([1.,-2.],device=device));opt=create_optimizer([p],recipe(op,{'lr':0.}));before=p.detach().clone();p.grad=torch.tensor([.5,-.7],device=device);opt.step();assert torch.equal(p,before)
 for bad in [[],[p,p],[torch.tensor([1.],device=device)],[torch.nn.Parameter(torch.tensor([1.],device=device),requires_grad=False)],[torch.nn.Parameter(torch.tensor([1.],dtype=torch.float64))],[torch.nn.Parameter(torch.tensor([float('nan')]))]]:
  with pytest.raises(ValueError):create_optimizer(bad,recipe(op))

@pytest.mark.parametrize('op,supplied',CASES)
@pytest.mark.parametrize('device',['cpu','mps'])
def test_strict_roundtrip_and_corruption_before_mutation(op,supplied,device):
 from runner.optimizer_configuration import validate_optimizer_state
 p=torch.nn.Parameter(torch.tensor([.8,-1.2],device=device));r=recipe(op,supplied);o=create_optimizer([p],r);p.grad=torch.tensor([.3,-.5],device=device);o.step();state=copy.deepcopy(o.state_dict());target=create_optimizer([p],r);restore_optimizer(target,state,r,[1]);assert target.state_dict()['param_groups']==state['param_groups'];assert all(torch.equal(value,target.state_dict()['state'][ident][key]) for ident,item in state['state'].items() for key,value in item.items());validate_optimizer_state(target,target.state_dict(),r,[1])
 bads=[]
 for edit in ['foreign','options','paramOrder','nonfinite','shape','dtype','negativeSquare','counter','extraKey','missingKey']:
  bad=copy.deepcopy(state);item=bad['state'].get(0,{})
  if edit=='foreign':bad['state'][999]={}
  elif edit=='options':bad['param_groups'][0]['lr']=.2
  elif edit=='paramOrder':bad['param_groups'][0]['params']=[8]
  elif not item:continue
  elif edit=='extraKey':item['madeup']=torch.tensor(0.)
  elif edit=='missingKey':item.pop(next(iter(item)))
  elif edit=='counter':
   if 'step' not in item:continue
   item['step']=torch.tensor(7.)
  elif edit=='negativeSquare':
   key=next((k for k in item if k in ('sum','square_avg','exp_avg_sq','max_exp_avg_sq')),None)
   if key is None:continue
   item[key]=-torch.ones_like(item[key])
  else:
   key=next(iter(item));item[key]=torch.full_like(item[key],float('nan')) if edit=='nonfinite' else torch.zeros(3) if edit=='shape' else item[key].cpu().to(torch.float64)
  bads.append(bad)
 for bad in bads:
  untouched=create_optimizer([p],r);before=copy.deepcopy(untouched.state_dict())
  with pytest.raises(ValueError):restore_optimizer(untouched,bad,r,[1])
  assert untouched.state_dict()['param_groups']==before['param_groups'];assert set(untouched.state_dict()['state'])==set(before['state'])

@pytest.mark.parametrize('op,supplied',CASES)
@pytest.mark.parametrize('device',['cpu','mps'])
def test_exported_fresh_step_matches_scalar(op,supplied,device):
 r=recipe(op,supplied);scope={};exec(training_recipe_code(r),scope);model=torch.nn.Linear(2,1,bias=False).to(device)
 with torch.no_grad():model.weight.copy_(torch.tensor([[.8,-1.2]]))
 o=scope['create_optimizer'](model);model.weight.grad=torch.tensor([[.3,-.5]],device=device);o.step();expected,_=scalar_update(op,canonical_configuration(op,supplied),[.8,-1.2],[.3,-.5],{});torch.testing.assert_close(model.weight.cpu(),torch.tensor([expected]),atol=3e-5,rtol=3e-5)

@pytest.mark.parametrize('op',IDS)
def test_strict_recipe_fields_versions_and_numeric_boundaries(op):
 for edit in [{'schemaVersion':True},{'configurationVersion':2},{'extra':1},{'config':[]},{'componentID':'unknown'}]:
  with pytest.raises(ValueError):validate_recipe({**recipe(op),**edit})
 c=canonical_configuration(op,{});
 for key,value in c.items():
  if type(value) is bool:
   with pytest.raises(ValueError):canonical_configuration(op,{key:1})
  else:
   for bad in [True,float('nan'),float('inf'),-1.]:
    with pytest.raises(ValueError):canonical_configuration(op,{key:bad})
 for bad in [{'nesterov':True},{'nesterov':True,'momentum':.5,'dampening':.1}]:
  with pytest.raises(ValueError):canonical_configuration('optimizer_sgd',bad)

@pytest.mark.parametrize('op',IDS)
def test_actual_cpu_exact_resume_options_ownership_buffers_and_training_export(op,tmp_path):
 from runner.engine import run,REGISTRY
 from runner.tests.test_stateful_normalization_runtime import graph
 from runner.tests.test_checkpoint_lifecycle import assert_nested_equal
 g=graph('batch_norm1d');r=recipe(op,{'lr':.02});paths=[tmp_path/x for x in ['full','first','resumed']]
 for path,epochs,parent in [(paths[0],3,None),(paths[1],1,None),(paths[2],3,paths[1]/'checkpoint.pt')]:run(g,path,epochs=epochs,device='cpu',registry=REGISTRY,sample_limit=32,batch_size=10,optimizer_recipe=r,resume_checkpoint=parent)
 a,b=[torch.load(p/'checkpoint.pt',map_location='cpu',weights_only=True) for p in [paths[0],paths[2]]]
 for key in ['modelState','optimizerState','rngState','optimizerUpdates']:assert_nested_equal(a[key],b[key])
 assert json.loads((paths[0]/'metrics.json').read_text())==json.loads((paths[2]/'metrics.json').read_text())
 manifest=json.loads((paths[2]/'manifest.json').read_text());c=canonical_configuration(op,{'lr':.02});assert manifest['settings']['optimizer']=={'optimizer_sgd':'SGD','optimizer_adam':'Adam','optimizer_adamw':'AdamW','optimizer_rmsprop':'RMSprop','optimizer_adagrad':'Adagrad'}[op];assert manifest['settings']['learningRate']==.02 and manifest['settings']['optimizerRecipe']==recipe(op,c)
 contract=a['contract']['optimizer'];assert contract['recipe']==recipe(op,c);assert len(contract['parameterOwnership'])==6;assert all('running_' not in p['name'] and 'num_batches' not in p['name'] for p in contract['parameterOwnership']);assert 'buffers' in manifest['lineage']['restoredState'];assert (paths[0]/'training_recipe.py').is_file();assert 'OPTIMIZER_RECIPE' not in (paths[0]/'model.py').read_text()
 for altered in [recipe(op,{'lr':.04}),recipe(IDS[(IDS.index(op)+1)%5])]:
  with pytest.raises(ValueError):run(g,tmp_path/('bad'+altered['componentID']),epochs=4,device='cpu',registry=REGISTRY,sample_limit=32,batch_size=10,optimizer_recipe=altered,resume_checkpoint=paths[1]/'checkpoint.pt')

@pytest.mark.parametrize('op',IDS)
def test_actual_mps_weights_only_finetune_and_exact_refusal(op,tmp_path):
 from runner.engine import run,REGISTRY
 from runner.tests.test_stateful_normalization_runtime import graph
 g=graph('batch_norm1d');first=tmp_path/'first';r=recipe(op,{'lr':.02});run(g,first,epochs=1,device='mps',registry=REGISTRY,sample_limit=32,batch_size=10,optimizer_recipe=r)
 with pytest.raises(ValueError):run(g,tmp_path/'bad',epochs=2,device='mps',registry=REGISTRY,sample_limit=32,batch_size=10,optimizer_recipe=r,resume_checkpoint=first/'checkpoint.pt')
 fine=tmp_path/'fine';run(g,fine,epochs=1,device='mps',registry=REGISTRY,sample_limit=32,batch_size=10,optimizer_recipe=recipe(op,{'lr':.01}),finetune_checkpoint=first/'checkpoint.pt');m=json.loads((fine/'manifest.json').read_text());assert m['lineage']['guarantee']=='weightsOnly' and 'buffers' in m['lineage']['restoredState'];assert m['settings']['initialEpoch']==0

@pytest.mark.parametrize('op',IDS)
def test_discovery_explicit_descriptor_identity_and_no_model_registration(op):
 from runner.engine import capabilities,SUPPORTED
 from runner.optimizer_configuration import capability
 assert capabilities()['componentCapabilities'][op]==capability(op);assert op not in SUPPORTED

def test_absent_recipe_preserves_legacy_contract_and_artifact_set(tmp_path):
 from runner.engine import run
 from runner.tests.test_runner import graph
 run(graph(),tmp_path/'legacy',epochs=1,device='cpu',sample_limit=32,batch_size=10);state=torch.load(tmp_path/'legacy/checkpoint.pt',weights_only=True);assert state['contract']['optimizer']=={'name':'Adam','learningRate':.03,'betas':[.9,.999],'eps':1e-8,'weightDecay':0,'amsgrad':False};assert 'optimizerUpdates' not in state;assert not (tmp_path/'legacy/training_recipe.py').exists();assert 'optimizerRecipe' not in json.loads((tmp_path/'legacy/manifest.json').read_text())['settings']

@pytest.mark.parametrize('op',IDS)
def test_fresh_no_gradient_initial_state_and_strict_state_controls(op):
 from runner.optimizer_configuration import validate_optimizer_state
 p=torch.nn.Parameter(torch.tensor([.8,-1.2]));r=recipe(op);o=create_optimizer([p],r);o.step();validate_optimizer_state(o,o.state_dict(),r,[0]);assert set(o.state_dict()['state'])==({0} if op=='optimizer_adagrad' else set())
 for count in [None,[],[True],[-1],[1000000001]]:
  with pytest.raises(ValueError):validate_optimizer_state(o,o.state_dict(),r,count)
 p.grad=torch.tensor([.3,-.5]);o.step();state=copy.deepcopy(o.state_dict());item=state['state'].get(0)
 if item and 'step' in item:
  for v in [torch.tensor(1,dtype=torch.int64),torch.tensor([1.]),torch.tensor(.5),torch.tensor(float('inf'))]:
   bad=copy.deepcopy(state);bad['state'][0]['step']=v
   with pytest.raises(ValueError):restore_optimizer(create_optimizer([p],r),bad,r,[1])

@pytest.mark.parametrize('op',IDS)
def test_invalid_recipe_and_failed_step_do_not_publish_completion(op,tmp_path):
 from runner.engine import run
 from runner.tests.test_runner import graph
 with pytest.raises(ValueError):run(graph(),tmp_path/'bad',epochs=1,device='cpu',optimizer_recipe={**recipe(op),'scheduler':{}},sample_limit=32)
 assert not (tmp_path/'bad').exists()
 # Finite valid controls may overflow actual numerical work, which must refuse.
 with pytest.raises((ValueError,RuntimeError)):run(graph(),tmp_path/'overflow',epochs=1,device='cpu',optimizer_recipe=recipe(op,{'lr':3.4028234663852886e38,'weight_decay':3.4028234663852886e38}),sample_limit=32,batch_size=10)
 assert not (tmp_path/'overflow/manifest.json').exists();assert not (tmp_path/'overflow/checkpoint.pt').exists()

@pytest.mark.parametrize('op',IDS)
def test_finetune_may_change_parent_batch_size_and_optimizer(op,tmp_path):
 from runner.engine import run
 from runner.tests.test_runner import graph
 first=tmp_path/'first';run(graph(),first,epochs=1,device='cpu',sample_limit=32,batch_size=4,optimizer_recipe=recipe(op))
 run(graph(),tmp_path/'fine',epochs=1,device='cpu',sample_limit=32,batch_size=32,optimizer_recipe=recipe(IDS[(IDS.index(op)+1)%5]),finetune_checkpoint=first/'checkpoint.pt');assert (tmp_path/'fine/manifest.json').exists()
 run(graph(),tmp_path/'legacyFine',epochs=1,device='cpu',sample_limit=32,batch_size=32,finetune_checkpoint=first/'checkpoint.pt');assert (tmp_path/'legacyFine/manifest.json').exists()

def test_amsgrad_dominance_and_duplicate_order_refusal():
 r=recipe('optimizer_adam',{'amsgrad':True});params=[torch.nn.Parameter(torch.tensor([1.])),torch.nn.Parameter(torch.tensor([2.]))];o=create_optimizer(params,r)
 for p in params:p.grad=torch.tensor([.5])
 o.step();state=copy.deepcopy(o.state_dict());state['state'][0]['max_exp_avg_sq']=torch.zeros(1)
 with pytest.raises(ValueError):restore_optimizer(create_optimizer(params,r),state,r,[1,1])
 state=copy.deepcopy(o.state_dict());state['param_groups'][0]['params']=[0,0]
 with pytest.raises(ValueError):restore_optimizer(create_optimizer(params,r),state,r,[1,1])

def test_cancelled_explicit_recipe_cannot_publish_checkpoint(tmp_path):
 from runner.engine import run
 from runner.tests.test_runner import graph
 flag=[False]
 def emit(e):
  if e['event']=='epoch':flag[0]=True
 def cancel():
  if flag[0]:raise RuntimeError('controlled cancel')
 with pytest.raises(RuntimeError,match='controlled cancel'):run(graph(),tmp_path/'run',epochs=2,device='cpu',sample_limit=32,optimizer_recipe=recipe('optimizer_adam'),emit=emit,check_cancelled=cancel)
 assert not (tmp_path/'run/manifest.json').exists() and not (tmp_path/'run/checkpoint.pt').exists()

@pytest.mark.parametrize('op',IDS)
def test_explicit_cpu_resume_preserves_stochastic_model_rng(op,tmp_path):
 from runner.engine import run,REGISTRY
 from runner.tests.test_remaining_activation_runtime import graph
 from runner.tests.test_checkpoint_lifecycle import assert_nested_equal
 g=graph('dropout',{'p':.3});r=recipe(op,{'lr':.01});full=tmp_path/'full';first=tmp_path/'first';resumed=tmp_path/'resumed'
 run(g,full,epochs=2,device='cpu',registry=REGISTRY,sample_limit=32,batch_size=10,optimizer_recipe=r);run(g,first,epochs=1,device='cpu',registry=REGISTRY,sample_limit=32,batch_size=10,optimizer_recipe=r);run(g,resumed,epochs=2,device='cpu',registry=REGISTRY,sample_limit=32,batch_size=10,optimizer_recipe=r,resume_checkpoint=first/'checkpoint.pt')
 a,b=[torch.load(p/'checkpoint.pt',weights_only=True) for p in [full,resumed]]
 for key in ['modelState','optimizerState','rngState','optimizerUpdates']:assert_nested_equal(a[key],b[key])
 assert json.loads((full/'metrics.json').read_text())==json.loads((resumed/'metrics.json').read_text())

@pytest.mark.parametrize('op',IDS)
def test_genuine_published_recipe_bundle_generated_descriptor(op,monkeypatch):
 monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'runner'))
 from runner.optimizer_adapters_generated import CAPABILITIES,NORMALIZERS
 from runner.optimizer_configuration import capability
 from runner.engine import REGISTRY
 row=next(c for c in json.loads(REGISTRY.read_text())['components'] if c['id']==op);bundle=json.loads((Path(__file__).resolve().parents[2]/'components/optimizers/bundle.json').read_text());authored=next(c['runtimeContract'] for c in bundle['components'] if c['definition']['id']==op);assert authored==capability(op);assert CAPABILITIES[op]==capability(op) and NORMALIZERS[op]({})==canonical_configuration(op,{})

@pytest.mark.parametrize('op',IDS)
def test_checkpoint_option_boolean_numeric_aliases_refuse(op):
 p=torch.nn.Parameter(torch.tensor([1.]));r=recipe(op);o=create_optimizer([p],r);p.grad=torch.tensor([.5]);o.step();state=copy.deepcopy(o.state_dict())
 for key in ['weight_decay','foreach','maximize']:
  bad=copy.deepcopy(state);bad['param_groups'][0][key]=False if key=='weight_decay' else 0
  with pytest.raises(ValueError):restore_optimizer(create_optimizer([p],r),bad,r,[1])

@pytest.mark.parametrize('op',IDS)
def test_all_numeric_control_valid_edges_and_outside_refusals(op):
 c=canonical_configuration(op,{})
 for key,value in c.items():
  if type(value) is bool:continue
  lower=1.1754943508222875e-38 if key=='eps' else 0.;upper=.9999999403953552 if key in ['beta1','beta2'] else 1. if key in ['momentum','dampening','alpha'] else 3.4028234663852886e38
  assert canonical_configuration(op,{key:lower})[key]==lower and canonical_configuration(op,{key:upper})[key]==upper
  for outside in [lower/2 if lower else -1e-20,upper*2]:
   with pytest.raises(ValueError):canonical_configuration(op,{key:outside})

def test_empty_selected_model_refuses_optimizer_before_success(tmp_path):
 from runner.engine import run
 g={'schemaVersion':1,'name':'No learned settings','nodes':[{'id':'input','type':'input','config':{'features':2}},{'id':'output','type':'output','config':{}}],'edges':[{'id':'e','source':'input','sourcePort':'out','target':'output','targetPort':'in'}]}
 with pytest.raises(ValueError,match='trainable parameters'):run(g,tmp_path/'run',epochs=1,device='cpu',sample_limit=32,optimizer_recipe=recipe('optimizer_adam'))
 assert not (tmp_path/'run/manifest.json').exists()

def test_untouched_adagrad_accumulator_cannot_be_forged():
 p=torch.nn.Parameter(torch.tensor([1.,2.]));r=recipe('optimizer_adagrad',{'initial_accumulator_value':.4});o=create_optimizer([p],r);state=copy.deepcopy(o.state_dict());state['state'][0]['sum']+=.2
 with pytest.raises(ValueError,match='initial value'):restore_optimizer(o,state,r,[0])

def test_shared_learned_parameter_identity_and_frozen_settings_owned_once():
 model=torch.nn.Module();model.first=torch.nn.Linear(2,2);model.second=torch.nn.Linear(2,2);model.second.weight=model.first.weight;model.first.bias.requires_grad_(False);model.register_buffer('summary',torch.zeros(2));parameters=[p for p in model.parameters() if p.requires_grad];o=create_optimizer(parameters,recipe('optimizer_adam'));assert len(o.param_groups)==1 and len(o.param_groups[0]['params'])==2 and len({id(p) for p in o.param_groups[0]['params']})==2;assert id(model.first.bias) not in {id(p) for p in o.param_groups[0]['params']};assert id(model.summary) not in {id(p) for p in o.param_groups[0]['params']}
