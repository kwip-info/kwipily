"""Reviewed versioned optimizer controls, ownership, strict state and fresh export."""
import math
MAXIMUM=3.4028234663852886e38
EPS_MIN=1.1754943508222875e-38
BETA_MAX=.9999999403953552
COUNTER_MAX=1_000_000_000
DEFAULTS={
'optimizer_sgd':dict(lr=.001,momentum=0.,dampening=0.,weight_decay=0.,nesterov=False,maximize=False),
'optimizer_adam':dict(lr=.001,beta1=.9,beta2=.999,eps=1e-8,weight_decay=0.,amsgrad=False,decoupled_weight_decay=False,maximize=False),
'optimizer_adamw':dict(lr=.001,beta1=.9,beta2=.999,eps=1e-8,weight_decay=.01,amsgrad=False,maximize=False),
'optimizer_rmsprop':dict(lr=.01,alpha=.99,eps=1e-8,weight_decay=0.,momentum=0.,centered=False,maximize=False),
'optimizer_adagrad':dict(lr=.01,lr_decay=0.,weight_decay=0.,initial_accumulator_value=0.,eps=1e-10,maximize=False)}
NAMES={'optimizer_sgd':'SGD','optimizer_adam':'Adam','optimizer_adamw':'AdamW','optimizer_rmsprop':'RMSprop','optimizer_adagrad':'Adagrad'}

def canonical_configuration(component_id,supplied):
 if type(component_id) is not str or component_id not in DEFAULTS or type(supplied) is not dict:raise ValueError('Choose a reviewed optimizer and declared controls.')
 defaults=DEFAULTS[component_id]
 if set(supplied)-set(defaults):raise ValueError('Optimizer accepts only its declared controls.')
 c={**defaults,**supplied}
 for key,default in defaults.items():
  value=c[key]
  if type(default) is bool:
   if type(value) is not bool:raise ValueError('Optimizer switches require Boolean values.')
  else:
   lo=EPS_MIN if key=='eps' else 0.;hi=BETA_MAX if key in ('beta1','beta2') else 1. if key in ('momentum','dampening','alpha') else MAXIMUM
   if type(value) not in (int,float) or not lo<=value<=hi:raise ValueError('Optimizer numeric control is outside its finite supported bounds.')
   c[key]=float(value)
 if c.get('nesterov') and (c['momentum']<=0 or c['dampening']!=0):raise ValueError('Look ahead needs positive momentum and zero dampening.')
 return c

def validate_recipe(value):
 if type(value) is not dict or set(value)!={'schemaVersion','componentID','configurationVersion','config'} or type(value['schemaVersion']) is not int or value['schemaVersion']!=1 or type(value['configurationVersion']) is not int or value['configurationVersion']!=1:raise ValueError('Optimizer recipe needs exact schema/configuration version1 and declared fields.')
 return {'schemaVersion':1,'componentID':value['componentID'],'configurationVersion':1,'config':canonical_configuration(value['componentID'],value['config'])}

def constructor_options(recipe):
 r=validate_recipe(recipe);c=dict(r['config']);op=r['componentID'];c.update(foreach=False,differentiable=False)
 if op in ('optimizer_adam','optimizer_adamw'):c['betas']=(c.pop('beta1'),c.pop('beta2'));c.update(fused=False,capturable=False)
 if op=='optimizer_sgd':c['fused']=False
 if op=='optimizer_rmsprop':c['capturable']=False
 if op=='optimizer_adagrad':c['fused']=False
 return c

def create_optimizer(parameters,recipe):
 import torch
 r=validate_recipe(recipe);params=list(parameters)
 if not params or len({id(p) for p in params})!=len(params):raise ValueError('Optimizer needs a nonempty unique ordered trainable parameter list.')
 for p in params:
  if not isinstance(p,torch.nn.Parameter) or not p.requires_grad or p.dtype!=torch.float32 or p.layout!=torch.strided or p.device.type not in ('cpu','mps') or not torch.isfinite(p).all().item():raise ValueError('Optimizer owns only finite dense float32 learned parameters on CPU/MPS; exclude buffers and frozen settings.')
 if len({p.device for p in params})!=1:raise ValueError('Optimizer parameters must share one device.')
 return getattr(torch.optim,NAMES[r['componentID']])(params,**constructor_options(r))

def _state_keys(op,c,count):
 if op=='optimizer_adagrad':return {'step','sum'}
 if count==0:return set()
 if op=='optimizer_sgd':return {'momentum_buffer'} if c['momentum']>0 else set()
 if op in ('optimizer_adam','optimizer_adamw'):return {'step','exp_avg','exp_avg_sq'}|({'max_exp_avg_sq'} if c['amsgrad'] else set())
 return {'step','square_avg'}|({'momentum_buffer'} if c['momentum']>0 else set())|({'grad_avg'} if c['centered'] else set())

def _same_setting(actual,expected):
 if type(expected) is bool:return type(actual) is bool and actual==expected
 if type(expected) in (int,float):return type(actual) in (int,float) and actual==expected
 if isinstance(expected,(list,tuple)):return type(actual) is type(expected) and len(actual)==len(expected) and all(_same_setting(a,b) for a,b in zip(actual,expected))
 return type(actual) is type(expected) and actual==expected

def validate_optimizer_state(optimizer,state,recipe,expected_steps,scheduler_context=None):
 import torch
 r=validate_recipe(recipe)
 if type(optimizer) is not getattr(torch.optim,NAMES[r['componentID']]):raise ValueError('Optimizer implementation differs from the admitted recipe.')
 params=[p for g in optimizer.param_groups for p in g['params']]
 if not params or len({id(p) for p in params})!=len(params) or any(not isinstance(p,torch.nn.Parameter) or not p.requires_grad or p.dtype!=torch.float32 or p.layout!=torch.strided or p.device.type not in ('cpu','mps') or not torch.isfinite(p).all().item() for p in params):raise ValueError('Optimizer live ownership is no longer unique finite dense trainable float32 parameters.')
 expected=optimizer.state_dict()['param_groups'];c=r['config'];op=r['componentID']
 if type(expected_steps) is not list or len(expected_steps)!=len(params) or any(type(n) is not int or not 0<=n<=COUNTER_MAX for n in expected_steps):raise ValueError('Optimizer update counts must identify every ordered parameter.')
 if type(state) is not dict or set(state)!={'state','param_groups'} or type(state['state']) is not dict or type(state['param_groups']) is not list or len(expected)!=1 or len(state['param_groups'])!=1:raise ValueError('Optimizer state needs exactly one declared parameter group.')
 actual=state['param_groups'][0];group=expected[0]
 if type(actual) is not dict or set(actual)!=set(group) or any(not _same_setting(actual[k],group[k]) for k in group) or type(actual['params']) is not list or any(type(i) is not int for i in actual['params']):raise ValueError('Optimizer parameter identities/order/options differ from the admitted recipe.')
 ids=group['params']
 if any(type(i) is not int for i in ids) or any(type(i) is not int for i in state['state']) or len(set(ids))!=len(params) or set(state['state'])-set(ids):raise ValueError('Optimizer state has duplicate or foreign parameter ownership.')
 # Compare constructor flags/options to a fresh admitted instance, not restored mutable groups.
 dummy=torch.nn.Parameter(torch.empty(0,dtype=torch.float32,device=params[0].device))
 fresh=create_optimizer([dummy],r).state_dict()['param_groups'][0];fresh['params']=ids
 if scheduler_context is not None:
  if __package__:
   from .scheduler_configuration import OptimizerContext
  else:
   from scheduler_configuration import OptimizerContext
  if type(scheduler_context) is not OptimizerContext:raise ValueError('Optimizer schedule context must be explicitly admitted.')
  fresh=dict(scheduler_context.group);fresh['params']=ids
 if group!=fresh:raise ValueError('Optimizer live options differ from canonical recipe.')
 for ident,p,count in zip(ids,params,expected_steps):
  expected_keys=_state_keys(op,c,count)
  if scheduler_context is not None and scheduler_context.buffer_indices is not None and op=='optimizer_sgd':expected_keys={'momentum_buffer'} if ident in scheduler_context.buffer_indices else set()
  item=state['state'].get(ident,{})
  if type(item) is not dict or set(item)!=expected_keys:raise ValueError('Optimizer per-parameter state is missing, extra or invented for a no-gradient setting.')
  for key,value in item.items():
   if not isinstance(value,torch.Tensor) or value.layout!=torch.strided or not torch.isfinite(value).all().item():raise ValueError('Optimizer state must be finite dense tensors.')
   if key=='step':
    if value.dtype!=torch.float32 or value.device.type!='cpu' or value.shape!=torch.Size([]) or value.item()!=count:raise ValueError('Optimizer step counter must be an exact CPU float32 scalar matching actual updates.')
   else:
    if value.dtype!=p.dtype or value.shape!=p.shape:raise ValueError('Optimizer remembered tensor shape/dtype differs from its owned parameter.')
    if key in ('exp_avg_sq','max_exp_avg_sq','square_avg','sum') and (value<0).any().item():raise ValueError('Optimizer squared memory cannot be negative.')
  if op=='optimizer_adagrad' and count==0 and not torch.equal(item['sum'],torch.full_like(p,c['initial_accumulator_value'],device=item['sum'].device)):raise ValueError('Untouched Adagrad accumulator differs from its configured initial value.')
  if 'max_exp_avg_sq' in item and (item['max_exp_avg_sq']<item['exp_avg_sq']).any().item():raise ValueError('Largest squared average must dominate its current squared average.')
 return r

def restore_optimizer(optimizer,state,recipe,expected_steps,scheduler_context=None):
 validate_optimizer_state(optimizer,state,recipe,expected_steps,scheduler_context)
 optimizer.load_state_dict(state)
 validate_optimizer_state(optimizer,optimizer.state_dict(),recipe,expected_steps,scheduler_context)

def training_recipe_code(recipe,scheduler_recipe=None):
 if scheduler_recipe is not None:
  if __package__:
   from .scheduler_configuration import validate_compatibility,training_code
  else:
   from scheduler_configuration import validate_compatibility,training_code
  validate_compatibility(scheduler_recipe,recipe)
  code=training_recipe_code(recipe)+training_code(scheduler_recipe)
  if len(code.encode('utf-8'))>8192:raise ValueError('Fresh training recipe exceeds its export bound.')
  return code
 r=validate_recipe(recipe);kwargs=constructor_options(r)
 return '"""Fresh configured optimizer; move/materialize model before constructing it. No exact-resume promise."""\nimport torch\nOPTIMIZER_RECIPE = '+repr(r)+'\ndef create_optimizer(model):\n    parameters = [p for p in model.parameters() if p.requires_grad]\n    if not parameters: raise ValueError("Training needs learned settings")\n    if len({id(p) for p in parameters}) != len(parameters) or any(p.dtype != torch.float32 or p.layout != torch.strided or p.device.type not in ("cpu","mps") or not torch.isfinite(p).all().item() for p in parameters): raise ValueError("Optimizer needs unique finite dense float32 learned settings")\n    if len({p.device for p in parameters}) != 1: raise ValueError("Optimizer needs one device")\n    return torch.optim.'+NAMES[r['componentID']]+'(parameters, **'+repr(kwargs)+')\n'

def capability(component_id):
 if component_id not in DEFAULTS:raise ValueError('Unknown optimizer.')
 return {'version':{'major':1,'minor':0,'patch':0},'adapterID':'kwip.runner.optimizer.'+component_id,'recipeSchemaVersions':[1],'configurationVersions':[1],'graphDTypes':['float32'],'devices':['cpu','mps'],'oneParameterGroup':True,'orderedTrainableParameters':True,'excludesBuffers':True,'foreach':False,'differentiable':False,'cpuResume':'epochBoundaryExactCPU','mpsResume':'weightsOnly','numericMaximum':MAXIMUM,'epsilonMinimum':EPS_MIN,'epsilonMaximum':MAXIMUM,'defaults':dict(DEFAULTS[component_id]),'fractionMaximum':BETA_MAX if component_id in ('optimizer_adam','optimizer_adamw') else 1.,'statePolicy':{'optimizer_sgd':'momentumOnlyNoCounter','optimizer_adam':'adamMomentsOptionalMaximum','optimizer_adamw':'adamMomentsOptionalMaximum','optimizer_rmsprop':'squaredOptionalMomentumAndCentered','optimizer_adagrad':'initializedAccumulator'}[component_id],'counterDType':'float32','counterDevice':'cpu','counterMaximum':COUNTER_MAX}
