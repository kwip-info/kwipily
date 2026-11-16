"""Selected-policy Torch/stdlib portable helpers; never autoexecuted on load."""
import hashlib
import inspect
if __package__:
 from . import gradient_policy,trainability,requested_gradients,output_routing
else:
 import gradient_policy,trainability,requested_gradients,output_routing

def _common():
 names=('exact','integer','identity','number','encoded','budget','validate_request')
 return 'import copy, json, math, re, struct, hashlib\nfrom contextlib import contextmanager\nfrom dataclasses import dataclass\nimport torch\nMAX_RECIPE_BYTES=32768\nMAX_ELEMENTS=65536\n'+''.join(inspect.getsource(getattr(gradient_policy,n))+'\n' for n in names)

def _bounded(source):
 if len(source.encode('utf8'))>131072:raise ValueError('Portable policy helper exceeds131072 bytes.')
 return source

def trainability_recipe_code(recipe,composition=False):
 recipe=trainability.validate_recipe(recipe)
 source=_common()+'TRAINABILITY_RECIPE = '+repr(recipe)+'\n'
 source+=''.join(inspect.getsource(getattr(trainability,n))+'\n' for n in ('canonical_configuration','validate_recipe'))
 source+=_owner_source(composition)
 source+=''.join(inspect.getsource(getattr(trainability,n))+'\n' for n in ('_resolve','resolve_owners','apply_trainability'))
 source+='\n_apply_trainability = apply_trainability\ndef apply_trainability(model):\n    return _apply_trainability(model, TRAINABILITY_RECIPE)\n'
 return _bounded(source)

def _objective_window_source(recipe,scalarization):
 if __package__:
  from . import loss_export,loss_partition
 else:
  import loss_export,loss_partition
 if recipe is None:
  return """
def window_denominator(roles,prediction_shape):
    if set(roles)!={'target'} or roles['target'].dtype!=torch.int64 or list(roles['target'].shape)!=[prediction_shape[0]] or len(prediction_shape)!=2 or prediction_shape[1]!=2 or ((roles['target']<0)|(roles['target']>1)).any().item():raise ValueError('Legacy cross-entropy requires aligned binary class targets.')
    return float(prediction_shape[0])
def objective_contribution(prediction,roles):
    return torch.nn.functional.cross_entropy(prediction,roles['target'],reduction='sum')
"""
 objective_source=loss_export.objective_recipe_code(recipe,scalarization)
 objective_source+='\nfrom dataclasses import dataclass\n'
 for name in ('ObjectivePartitionContribution','objective_partition_contribution','preflight_objective_training_batches'):
  objective_source+=inspect.getsource(getattr(loss_partition,name))+'\n'
 window=inspect.getsource(gradient_policy.window_denominator)
 # Keep the reviewed arithmetic body, substituting its package imports with
 # the separately generated standalone objective namespace.
 window='def admitted_window_denominator(objective_recipe,roles,prediction_shape,scalarization="mean"):\n loss_recipe=validate_recipe\n'+window[window.index(' r=loss_recipe'):]
 objective_source+=window
 return '\n_OBJECTIVE_NAMESPACE={"__name__":__name__}\nexec('+repr(objective_source)+',_OBJECTIVE_NAMESPACE)\n'+"""
def window_denominator(roles,prediction_shape):
    return _OBJECTIVE_NAMESPACE['admitted_window_denominator'](_OBJECTIVE_NAMESPACE['OBJECTIVE_RECIPE'],roles,prediction_shape,_OBJECTIVE_NAMESPACE['OBJECTIVE_SCALARIZATION'])
def objective_contribution(prediction,roles):
    objective=_OBJECTIVE_NAMESPACE['create_training_objective'](prediction.device.type)
    return _OBJECTIVE_NAMESPACE['objective_partition_contribution'](objective,prediction,roles,_OBJECTIVE_NAMESPACE['OBJECTIVE_SCALARIZATION']).numerator
"""

def gradient_policy_code(recipe,objective_recipe=None,scalarization="mean",composition=False,bag_input_contract=None):
 recipe=gradient_policy.validate_recipe(recipe);config={k:recipe['config'][k] for k in ('accumulation_steps','gradient_scale')}
 source=_common()+'GRADIENT_POLICY_SHA256 = '+repr(hashlib.sha256(gradient_policy.encoded(recipe)).hexdigest())+'\nACCUMULATION_CONFIG = '+repr(config)+'\n'
 source+=_owner_source(composition)
 source+=inspect.getsource(output_routing.OutputAddress)+'\n'+inspect.getsource(output_routing.select_output)+'\n'
 source+=inspect.getsource(requested_gradients._isolated)+'\n'
 for name in ('_inputs','_targets','analyze'):
  source+=inspect.getsource(getattr(requested_gradients,'_ordinary'+name if name.startswith('_') else '_ordinary_'+name)).replace('def _ordinary'+name+'(', 'def '+name+'(').replace('def _ordinary_'+name+'(', 'def '+name+'(')+'\n'
 if bag_input_contract is not None:
  source+=_bag_analysis_source(bag_input_contract)
 source+='''
def create_accumulation_state():
    return dict(active=False,count=0,expected=0,denominator=1.,model=None)
def begin_window(model,state,*,denominator,expected_microbatches):
    if state.get('active'):raise ValueError('Finish the current window before resetting gradients.')
    integer(expected_microbatches,1,ACCUMULATION_CONFIG['accumulation_steps'])
    if type(denominator) not in (int,float) or not math.isfinite(denominator) or denominator<=0:raise ValueError('Admit a positive finite full-window denominator before forward.')
    if ACCUMULATION_CONFIG['gradient_scale']=='sum' and denominator!=1:raise ValueError('Summed objective requires denominator1.')
    native_owners(model)
    model.zero_grad(set_to_none=True)
    state.update(active=True,count=0,expected=expected_microbatches,denominator=denominator,model=id(model))
def backward_contribution(numerator,state):
    if not state.get('active') or state['count']>=state['expected']:raise ValueError('No admitted pending microbatch remains.')
    if not isinstance(numerator,torch.Tensor) or numerator.ndim!=0 or not torch.isfinite(numerator).item():raise ValueError('Provide the actual differentiable objective numerator.')
    scalar=numerator if ACCUMULATION_CONFIG['gradient_scale']=='sum' else numerator/state['denominator']
    scalar.backward();state['count']+=1
def finish_window(model,optimizer,state,*,scheduler=None):
    if not state.get('active') or state['count']!=state['expected'] or state['model']!=id(model):raise ValueError('Complete every admitted microbatch before stepping the same model.')
    expected=[p for p in model.parameters() if p.requires_grad];owned=[p for group in optimizer.param_groups for p in group['params']]
    if not expected or [id(p) for p in expected]!=[id(p) for p in owned]:raise ValueError('Optimizer must own the exact ordered trainable model parameters.')
    if any(p.grad is not None and (p.grad.layout!=torch.strided or not torch.isfinite(p.grad).all().item()) for p in model.parameters()):raise ValueError('Sparse or nonfinite accumulated gradients are unsupported.')
    if scheduler is not None and scheduler.optimizer is not optimizer:raise ValueError('Scheduler belongs to another optimizer.')
    if isinstance(scheduler,torch.optim.lr_scheduler.OneCycleLR) and scheduler.last_epoch>=scheduler.total_steps:raise ValueError('Pinned optimizer-update horizon is exhausted.')
    optimizer.step()
    if any(not torch.isfinite(p).all().item() for p in owned):raise ValueError('Optimizer produced nonfinite learned values.')
    if isinstance(scheduler,torch.optim.lr_scheduler.OneCycleLR):scheduler.step()
    model.zero_grad(set_to_none=True);state.update(active=False,count=0,expected=0,model=None)
'''
 source+=_objective_window_source(objective_recipe,scalarization)
 return _bounded(source)


def _owner_source(composition):
 if composition:
  if __package__:from .native_ownership import portable_source
  else:from native_ownership import portable_source
  return portable_source()
 return inspect.getsource(trainability._legacy_native_owners).replace('def _legacy_native_owners(', 'def native_owners(')+'\n'

def _bag_analysis_source(contract):
 # Resolve the exported sibling's packet class and validation; never duplicate it.
 if type(contract) is not dict or set(contract)!=set(('kind','schemaVersion','sourceID','configuration','ports')) or contract['kind']!='bagPacket' or type(contract['schemaVersion']) is not int or contract['schemaVersion']!=1:raise ValueError('Require actual adapter bag input contract.')
 if __package__:from .bag_source import source_layouts
 else:from bag_source import source_layouts
 roles=source_layouts(contract['sourceID'],contract['configuration'])
 expected={role:dict(role=role,dtype='float32' if role=='per_sample_weights' else 'int64') for role in roles}
 if contract['ports']!=expected:raise ValueError('Bag derivative roles must match actual source controls.')
 source="\ntry:\n from .model import execute_bag_source\nexcept ImportError:\n from model import execute_bag_source\n"
 source+='_BAG_INPUT_CONTRACT = '+repr(contract)+'\n'
 source+='def _bag_contract(model):\n return copy.deepcopy(_BAG_INPUT_CONTRACT)\n'
 coordinates=inspect.getsource(requested_gradients._bag_coordinates)
 coordinates=coordinates.replace(" if __package__:from .bag_source import execute_bag_source\n else:from bag_source import execute_bag_source\n",'')
 source+=coordinates+'\n'+inspect.getsource(requested_gradients._targets)+'\n'+inspect.getsource(requested_gradients._bag_analyze)+'\nanalyze = _bag_analyze\n'
 return source
