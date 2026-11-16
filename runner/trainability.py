"""Apply actual native parameter flags atomically before optimizer ownership."""
import torch
if __package__:
 from .gradient_policy import exact,identity,budget
else:
 from gradient_policy import exact,identity,budget

def canonical_configuration(values):
 if type(values) is not dict or set(values)-{'owners'}:raise ValueError('Trainability accepts only owners.')
 owners=values.get('owners',[])
 if type(owners) is not list or len(owners)>128:raise ValueError('Choose at most128 trainability overrides.')
 result=[];seen=set()
 for owner in owners:
  exact(owner,('node','parameter','mode'))
  node=identity(owner['node']);parameter=identity(owner['parameter']);mode=owner['mode']
  if type(mode) is not str or mode not in ('inherit','trainable','frozen'):raise ValueError('Choose inherit, trainable or frozen.')
  if (node,parameter) in seen:raise ValueError('Duplicate trainability owner.')
  seen.add((node,parameter));result.append(dict(node=node,parameter=parameter,mode=mode))
 result.sort(key=lambda o:(o['node'],o['parameter']));c=dict(owners=result)
 budget(dict(schemaVersion=1,configurationVersion=1,componentID='trainability',config=c));return c

def validate_recipe(value):
 exact(value,('schemaVersion','configurationVersion','componentID','config'))
 if type(value['schemaVersion']) is not int or value['schemaVersion']!=1 or type(value['configurationVersion']) is not int or value['configurationVersion']!=1 or value['componentID']!='trainability':raise ValueError('Unsupported trainability recipe identity/version.')
 exact(value['config'],('owners',))
 return dict(schemaVersion=1,configurationVersion=1,componentID='trainability',config=canonical_configuration(value['config']))

def _legacy_native_owners(model):
 if not isinstance(model,torch.nn.Module) or not isinstance(getattr(model,'layers',None),torch.nn.ModuleDict) or type(getattr(model,'layer_keys',None)) is not dict:raise ValueError('Select a materialized model with exact native layer_keys ownership.')
 all_params=list(model.named_parameters(remove_duplicate=False))
 if any(isinstance(p,torch.nn.parameter.UninitializedParameter) for _,p in all_params):raise ValueError('Materialize lazy model parameters before resolving owners.')
 if len({id(p) for _,p in all_params})!=len(all_params):raise ValueError('Aliased native parameters have ambiguous ownership.')
 if sum(p.numel() for _,p in all_params)>200000:raise ValueError('Model exceeds the200000-parameter limit.')
 for _,p in all_params:
  if p.dtype!=torch.float32 or p.layout!=torch.strided or p.device.type not in ('cpu','mps') or not p.numel() or not torch.isfinite(p.detach()).all().item():raise ValueError('Native owners require finite dense float32 CPU/MPS parameters.')
 reverse={v:k for k,v in model.layer_keys.items()}
 if len(reverse)!=len(model.layer_keys) or any(k not in model.layers for k in reverse):raise ValueError('Native layer ownership is ambiguous or missing.')
 result=[]
 for name,p in all_params:
  matched=[]
  for key,node in reverse.items():
   prefix='layers.'+key+'.'
   if name.startswith(prefix):matched.append((node,name[len(prefix):]))
  if len(matched)!=1:raise ValueError('Parameter has no unique expanded-node native ownership.')
  node,relative=matched[0];result.append((dict(node=node,parameter=relative,name=name,shape=list(p.shape),dtype='float32',requiresGrad=bool(p.requires_grad)),p))
 return result

def native_owners(model):
 if not hasattr(model,'composition_root'):return _legacy_native_owners(model)
 if __package__:from .native_ownership import native_owners as shared
 else:from native_ownership import native_owners as shared
 return shared(model)

def _resolve(model,recipe):
 r=validate_recipe(recipe);inventory=native_owners(model);by_owner={(m['node'],m['parameter']):(m,p) for m,p in inventory}
 resolved=[]
 for owner in r['config']['owners']:
  key=(owner['node'],owner['parameter'])
  if key not in by_owner:raise ValueError('Trainability selects an actual native parameter, not a buffer or foreign owner.')
  m,p=by_owner[key];resolved.append((m,p,owner['mode']))
 return inventory,resolved

def resolve_owners(model,recipe):return [dict(m,mode=mode) for m,_,mode in _resolve(model,recipe)[1]]
def apply_trainability(model,recipe):
 inventory,resolved=_resolve(model,recipe)
 # All settings/owners were admitted before changing any flag.
 for _,p,mode in resolved:
  if mode!='inherit':p.requires_grad_(mode=='trainable')
 return [dict(m,requiresGrad=bool(p.requires_grad)) for m,p in inventory]


def capability():
 return {'version': '1.0.0', 'adapterID': 'kwip.runner.trainability', 'recipeSchemaVersion': 1, 'configurationVersion': 1, 'devices': ['cpu', 'mps'], 'dtype': 'float32', 'ownerMaximum': 128, 'recipeByteMaximum': 32768, 'ownerModes': ['inherit', 'trainable', 'frozen'], 'ownerResolution': 'materializedExpandedNodeNativeParameter', 'application': 'beforeOptimizerOwnership', 'moduleMode': 'existingTrainEvalTransitions', 'frozenInference': 'evaluationNoGrad', 'inputDerivativesThroughFrozenParameters': True, 'emptyTrainingOwnership': 'refuse', 'transition': 'freshOrFineTuneFreshOptimizer', 'exactResume': 'identicalPolicyAndOrderedOwnership'}
