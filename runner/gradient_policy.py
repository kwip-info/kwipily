"""Strict optional accumulation/request policy; no training loop or global state."""
import copy
import json
import math
import re
import struct

MAX_RECIPE_BYTES=32768
MAX_ELEMENTS=65536

def exact(value,keys,message='Unexpected policy fields.'):
 if type(value) is not dict or set(value)!=set(keys):raise ValueError(message)
def integer(value,minimum,maximum):
 if type(value) is not int or not minimum<=value<=maximum:raise ValueError('Choose an exact integer within the declared policy bounds.')
 return value
def identity(value):
 if type(value) is not str or not value or '\0' in value:raise ValueError('Choose a nonempty actual model identity without NUL.')
 return value
def number(value):
 if type(value) not in (int,float):raise ValueError('Upstream values require finite float32 numbers, not booleans.')
 try:native=struct.unpack('f',struct.pack('f',value))[0]
 except (OverflowError,struct.error):raise ValueError('Upstream value exceeds float32.') from None
 if not math.isfinite(native) or value!=0 and native==0:raise ValueError('Upstream value is not representable as finite float32.')
 return float(value)
def encoded(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf8')
def budget(value):
 if len(encoded(value))>MAX_RECIPE_BYTES:raise ValueError('Policy exceeds its 32768-byte portable recipe limit.')
 return value

def validate_request(value):
 exact(value,('id','output','partition','row_start','row_count','reduction','upstream','targets','unavailable'))
 if type(value['id']) is not str or re.fullmatch(r'[A-Za-z0-9_-]{1,64}',value['id']) is None:raise ValueError('Request ID must be unique ASCII letters, digits, underscore or hyphen, length1…64.')
 exact(value['output'],('node','port'))
 output={k:identity(v) for k,v in value['output'].items()}
 if value['partition'] not in ('training','validation') or type(value['partition']) is not str:raise ValueError('Choose training or validation partition.')
 start=integer(value['row_start'],0,2**63-1);count=integer(value['row_count'],1,32)
 if start>2**63-1-count:raise ValueError('Requested row range overflows Int64.')
 if value['reduction'] not in ('sum','mean','upstream') or type(value['reduction']) is not str:raise ValueError('Choose sum, mean or upstream derivative reduction.')
 if value['unavailable'] not in ('error','reportUnavailable') or type(value['unavailable']) is not str:raise ValueError('Choose an explicit unavailable-result policy.')
 supplied=value['upstream'];upstream=None
 if value['reduction']=='upstream':
  exact(supplied,('dtype','shape','values'))
  if supplied['dtype']!='float32' or type(supplied['shape']) is not list or not 1<=len(supplied['shape'])<=8:raise ValueError('Upstream requires float32 and1…8 directions.')
  shape=[integer(v,1,4096) for v in supplied['shape']];n=math.prod(shape)
  if shape[0]!=count or n>MAX_ELEMENTS or type(supplied['values']) is not list or len(supplied['values'])!=n:raise ValueError('Upstream shape/count must match selected observations within65536 elements.')
  upstream=dict(dtype='float32',shape=shape,values=[number(v) for v in supplied['values']])
 elif supplied is not None:raise ValueError('Sum/mean requests require upstream null.')
 if type(value['targets']) is not list or not 1<=len(value['targets'])<=32:raise ValueError('Choose1…32 exact derivative targets.')
 targets=[];seen=set()
 for target in value['targets']:
  if type(target) is not dict or target.get('kind') not in ('input','parameter'):raise ValueError('Choose an actual input or native parameter target.')
  exact(target,('kind','node','port') if target['kind']=='input' else ('kind','node','parameter'))
  canonical={k:identity(v) for k,v in target.items()};key=encoded(canonical)
  if key in seen:raise ValueError('Duplicate target in one request.')
  seen.add(key);targets.append(canonical)
 return dict(id=value['id'],output=output,partition=value['partition'],row_start=start,row_count=count,reduction=value['reduction'],upstream=upstream,targets=targets,unavailable=value['unavailable'])

def canonical_configuration(values):
 if type(values) is not dict or set(values)-{'accumulation_steps','gradient_scale','requests'}:raise ValueError('Gradient policy accepts accumulation_steps, gradient_scale and requests.')
 c={**dict(accumulation_steps=1,gradient_scale='mean',requests=[]),**values}
 k=integer(c['accumulation_steps'],1,32)
 if type(c['gradient_scale']) is not str or c['gradient_scale'] not in ('mean','sum'):raise ValueError('Choose mean or sum gradient scaling.')
 if type(c['requests']) is not list or len(c['requests'])>32:raise ValueError('At most32 derivative requests are supported.')
 requests=[validate_request(r) for r in c['requests']]
 if len({r['id'] for r in requests})!=len(requests):raise ValueError('Duplicate derivative request ID.')
 if len({encoded(t) for r in requests for t in r['targets']})>32:raise ValueError('At most32 unique derivative targets are supported.')
 if sum(len(r['upstream']['values']) for r in requests if r['upstream'] is not None)>MAX_ELEMENTS:raise ValueError('Total upstream exceeds65536 elements.')
 result=dict(accumulation_steps=k,gradient_scale=c['gradient_scale'],requests=requests)
 budget(dict(schemaVersion=1,configurationVersion=1,componentID='gradient_policy',config=result));return result

def validate_recipe(value):
 exact(value,('schemaVersion','configurationVersion','componentID','config'))
 if type(value['schemaVersion']) is not int or value['schemaVersion']!=1 or type(value['configurationVersion']) is not int or value['configurationVersion']!=1 or value['componentID']!='gradient_policy':raise ValueError('Unsupported gradient policy recipe identity/version.')
 exact(value['config'],('accumulation_steps','gradient_scale','requests'))
 return budget(dict(schemaVersion=1,configurationVersion=1,componentID='gradient_policy',config=canonical_configuration(value['config'])))

def validate_capture(value=None):
 if value is None:return dict(schemaVersion=1,values=False)
 exact(value,('schemaVersion','values'))
 if type(value['schemaVersion']) is not int or value['schemaVersion']!=1 or type(value['values']) is not bool:raise ValueError('Derivative capture requires schema1 and an explicit boolean values permission.')
 return dict(value)

def validate_compatibility(policy,objective_recipe=None,scalarization='mean'):
 c=validate_recipe(policy)['config']
 if scalarization not in ('mean','sum'):raise ValueError('Unsupported objective scalarization.')
 reduction='mean' if objective_recipe is None else objective_recipe['config']['reduction']
 effective=scalarization if reduction=='none' else 'mean' if reduction in ('mean','batchmean') else reduction
 if c['gradient_scale']!=effective:raise ValueError('Gradient scaling must match the selected objective reduction; choose '+effective+'.')
 return c

def window_denominator(objective_recipe,roles,prediction_shape,scalarization='mean'):
 """Pre-forward target admission for ONE complete window, not local means."""
 import torch
 if __package__:
  from .loss_configuration import validate_recipe as loss_recipe,resolve_objective
  from .loss_partition import preflight_objective_training_batches
  from .tensor_layout import BatchedLayout
 else:
  from loss_configuration import validate_recipe as loss_recipe,resolve_objective
  from loss_partition import preflight_objective_training_batches
  from tensor_layout import BatchedLayout
 r=loss_recipe(objective_recipe);c=r['config'];op=r['componentID']
 if type(prediction_shape) not in (list,tuple) or not 2<=len(prediction_shape)<=8:raise ValueError('Use a complete observation-first prediction shape.')
 shape=[integer(v,1,4096 if i else 100000) for i,v in enumerate(prediction_shape)];n=shape[0]
 layout=BatchedLayout(tuple(shape[1:]))
 # The existing preflight validates typed roles and target values. Mean class
 # weights need an additional check against actual float32-rounded values.
 preflight_objective_training_batches(r,roles,n,layout,training_rows=n)
 metadata={k:dict(dtype=str(v.dtype).removeprefix('torch.'),sampleShape=list(v.shape[1:])) for k,v in roles.items()}
 resolved=resolve_objective(op,c,layout,metadata)
 if scalarization not in ('mean','sum'):raise ValueError('Choose mean or sum scalarization.')
 red=c['reduction']
 if red=='sum' or red=='none' and scalarization=='sum':return 1.
 if red=='none':return float(n*resolved['rawElementsPerObservation'])
 if red=='batchmean' or op in ('loss_ctc','loss_triplet_margin'):return float(n)
 if op=='loss_nll' or op=='loss_cross_entropy' and c['target_mode']=='class_indices':
  selected=roles['target'];selected=selected[selected!=c['ignore_index']]
  d=float(torch.tensor(c['class_weights'],dtype=torch.float32).double()[selected.cpu()].sum().item()) if c['class_weights'] else float(selected.numel())
  if not math.isfinite(d) or not 0<d<=torch.finfo(torch.float32).max:raise ValueError('Window native mean denominator is undefined or overflows float32.')
  return d
 return float(n*resolved['rawElementsPerObservation'])

def scaled_contribution(part,denominator,scale):
 import torch
 if scale not in ('mean','sum') or type(denominator) not in (int,float) or not math.isfinite(denominator) or denominator<=0:raise ValueError('Window requires its admitted positive finite denominator and compatible scale.')
 value=part.numerator if scale=='sum' else part.numerator/denominator
 if not isinstance(value,torch.Tensor) or value.ndim or not torch.isfinite(value).item():raise ValueError('Window contribution must be a finite differentiable scalar.')
 return value


def capability():
 return {'version': '1.0.0', 'adapterID': 'kwip.runner.gradient_policy', 'recipeSchemaVersion': 1, 'configurationVersion': 1, 'devices': ['cpu', 'mps'], 'dtype': 'float32', 'derivativeOrder': 1, 'derivativeLayout': 'dense', 'accumulationMinimum': 1, 'accumulationMaximum': 32, 'requestMaximum': 32, 'uniqueTargetMaximum': 32, 'derivativeElementMaximum': 65536, 'upstreamElementMaximum': 65536, 'rowCountMaximum': 32, 'recipeByteMaximum': 32768, 'derivativeArtifactByteMaximum': 8388608, 'analysisPhase': 'finalSnapshotEvaluation', 'capturePermission': 'derivativeCapture.values', 'unavailableCauses': ['integerInput', 'frozenParameter', 'disconnected', 'recordingDisabled', 'sparseGradient'], 'nativeBackwardFailure': 'refuseWithoutFallback', 'reset': 'windowStart', 'partialWindow': 'flush', 'objectiveScaling': 'exactWindowNumeratorDenominator', 'oneCycleCadence': 'optimizerUpdate', 'continuation': 'cpuEpochBoundaryExactMPSWeightsOnly'}
