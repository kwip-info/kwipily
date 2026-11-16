"""Torch-free strict objective recipes and observation-owned target geometry."""
import copy,json,math,struct
MAX_ELEMENTS=100_000_000
FLOAT_MIN=1.401298464324817e-45
FLOAT_MAX=3.4028234663852886e38
DEFAULTS={
 'loss_cross_entropy':dict(reduction='mean',target_mode='class_indices',class_axis=-1,class_weights=[],ignore_index=-100,label_smoothing=0.),
 'loss_bce_with_logits':dict(reduction='mean',weight_mode='none',weight_shape=[],weight_values=[],pos_weight_mode='none',pos_weight_shape=[],pos_weight_values=[]),
 'loss_mse':dict(reduction='mean',target_broadcast=False),
 'loss_l1':dict(reduction='mean',target_broadcast=False),
 'loss_nll':dict(reduction='mean',class_axis=-1,class_weights=[],ignore_index=-100),
 'loss_kl_div':dict(reduction='mean',log_target=False),
 'loss_ctc':dict(reduction='mean',blank=0,zero_infinity=False,target_layout='padded'),
 'loss_triplet_margin':dict(reduction='mean',margin=1.,p=2.,eps=1e-6,swap=False)}
IDS=tuple(DEFAULTS)

def integer(value,lo,hi):
 if type(value) not in (int,float) or not math.isfinite(value) or value!=int(value) or not lo<=value<=hi:raise ValueError('Objective integer control is outside its supported bounds.')
 return int(value)
def number(value,lo=0.,hi=FLOAT_MAX):
 if type(value) not in (int,float) or not math.isfinite(value) or not lo<=value<=hi:raise ValueError('Objective numeric control must be finite and within its supported bounds.')
 try:rounded=struct.unpack('f',struct.pack('f',value))[0]
 except (OverflowError,struct.error):raise ValueError('Objective numeric control is not representable in float32.') from None
 if not math.isfinite(rounded) or (value!=0 and rounded==0):raise ValueError('Objective numeric control underflows or overflows float32.')
 return float(value)
def shape(value):
 if type(value) is not list or len(value)>7:raise ValueError('Target/sample shape requires at most seven declared directions.')
 return [integer(v,1,4096) for v in value]
def elements(value):
 total=1
 for v in value:
  if v>MAX_ELEMENTS//total:raise ValueError('Objective tensor exceeds its 100M element budget.')
  total*=v
 return total

def canonical_configuration(component_id,supplied):
 if type(component_id) is not str or component_id not in DEFAULTS or type(supplied) is not dict or set(supplied)-set(DEFAULTS[component_id]):raise ValueError('Objective requires a reviewed component and only declared controls.')
 c=copy.deepcopy(DEFAULTS[component_id]);c.update(supplied)
 for key,default in DEFAULTS[component_id].items():
  value=c[key]
  if type(default) is bool:
   if type(value) is not bool:raise ValueError('Objective switches require Boolean values.')
  elif key in ('class_axis','ignore_index','blank'):
   lo,hi=(-7,7) if key=='class_axis' else (-(2**31),2**31-1) if key=='ignore_index' else (0,511)
   c[key]=integer(value,lo,hi)
   if key=='class_axis' and c[key]==0:raise ValueError('Class direction cannot be the batch direction 0.')
  elif key.endswith('_shape'):c[key]=shape(value);elements(c[key])
  elif key in ('class_weights','weight_values','pos_weight_values'):
   if type(value) is not list or len(value)>(512 if key=='class_weights' else 4096):raise ValueError('Objective weight list exceeds its declared bound.')
   c[key]=[number(v) for v in value]
  elif type(default) is float:
   lo=FLOAT_MIN if key in ('margin','p') else 0.;hi=16. if key=='p' else 1. if key=='label_smoothing' else FLOAT_MAX
   c[key]=number(value,lo,hi)
  else:
   choices={'reduction':('none','mean','sum','batchmean') if component_id=='loss_kl_div' else ('none','mean','sum'),'target_mode':('class_indices','probabilities'),'target_layout':('padded','packed'),'weight_mode':('none','configured','mapped'),'pos_weight_mode':('none','configured','mapped')}
   if type(value) is not str or value not in choices[key]:raise ValueError('Objective choice is not supported.')
 if component_id=='loss_cross_entropy' and c['target_mode']=='probabilities' and c['ignore_index']!=-100:raise ValueError('Probability targets cannot use an ignored class index; choose −100.')
 if component_id=='loss_bce_with_logits':
  for prefix in ('weight','pos_weight'):
   vals=c[prefix+'_values'];size=elements(c[prefix+'_shape'])
   if (vals and len(vals)!=size) or (c[prefix+'_mode']=='configured' and len(vals)!=size):raise ValueError('Configured weight values must exactly match their sample shape, including one scalar value for shape [].')
 envelope=dict(schemaVersion=1,componentID=component_id,configurationVersion=1,config=c)
 if len(json.dumps(envelope,allow_nan=False,separators=(',',':')).encode())>32768:raise ValueError('Canonical objective recipe exceeds 32768 UTF8 bytes.')
 return c

def validate_recipe(value):
 if type(value) is not dict or set(value)!={'schemaVersion','componentID','configurationVersion','config'} or type(value['schemaVersion']) is not int or value['schemaVersion']!=1 or type(value['configurationVersion']) is not int or value['configurationVersion']!=1:raise ValueError('Objective recipe requires exact version 1 and its four declared keys.')
 r=dict(schemaVersion=1,componentID=value['componentID'],configurationVersion=1,config=canonical_configuration(value['componentID'],value['config']))
 if len(json.dumps(r,allow_nan=False,separators=(',',':')).encode())>32768:raise ValueError('Canonical objective recipe exceeds 32768 UTF8 bytes.')
 return r

def sample_broadcast(first,second):
 rank=max(len(first),len(second));a=[1]*(rank-len(first))+list(first);b=[1]*(rank-len(second))+list(second)
 if any(x!=y and x!=1 and y!=1 for x,y in zip(a,b)):raise ValueError('Target sample directions cannot broadcast; the observation direction is never broadcast.')
 out=[max(x,y) for x,y in zip(a,b)];elements(out);return out

def resolve_objective(component_id,configuration,prediction_layout,target_roles):
 c=canonical_configuration(component_id,configuration);p=list(prediction_layout.sample)
 if not 1<=len(p)<=7 or any(type(v) is not int or not 1<=v<=4096 for v in p):raise ValueError('Prediction requires rank 2…8 with sample directions 1…4096.')
 elements(p)
 if type(target_roles) is not dict:raise ValueError('Objective target roles must be explicitly declared.')
 roles={}
 for key,value in target_roles.items():
  if type(value) is not dict or set(value)!={'dtype','sampleShape'} or value['dtype'] not in ('float32','int64'):raise ValueError('Target role requires exact dtype and sampleShape fields.')
  s=shape(value['sampleShape']);elements(s);roles[key]={'dtype':value['dtype'],'sampleShape':s}
 expected=set() if component_id=='loss_triplet_margin' else {'target'}
 if component_id=='loss_ctc':expected|={'input_lengths','target_lengths'}
 if component_id=='loss_bce_with_logits':expected|={k for k in ('weight','pos_weight') if c[k+'_mode']=='mapped'}
 if set(roles)!=expected:raise ValueError('Objective target roles are missing or unexpected for this recipe.')
 def require(key,dtype,s=None):
  v=roles[key]
  if v['dtype']!=dtype or (s is not None and v['sampleShape']!=s):raise ValueError('Objective target role dtype/sample shape does not match the selected prediction geometry.')
 raw=p;axis=None;classes=None;work=elements(p)
 if component_id in ('loss_cross_entropy','loss_nll'):
  rank=len(p)+1;a=c['class_axis']
  if not -rank<=a<rank or a%rank==0:raise ValueError('Class direction is outside this prediction or aliases the batch direction.')
  axis=a%rank;classes=p[axis-1]
  if not 2<=classes<=512:raise ValueError('Class direction requires 2…512 classes.')
  if c['class_weights'] and len(c['class_weights'])!=classes:raise ValueError('Class weights must match the actual class count.')
  raw=p[:axis-1]+p[axis:]
  probability=component_id=='loss_cross_entropy' and c['target_mode']=='probabilities'
  require('target','float32' if probability else 'int64',p if probability else raw)
 elif component_id=='loss_bce_with_logits':
  require('target','float32',p)
  for prefix in ('weight','pos_weight'):
   mode=c[prefix+'_mode']
   if mode=='none':continue
   s=c[prefix+'_shape'] if mode=='configured' else roles[prefix]['sampleShape']
   if mode=='mapped':require(prefix,'float32')
   if len(s)>len(p) or sample_broadcast(p,s)!=p:raise ValueError('Weights must broadcast only into existing prediction sample directions, never observations.')
 elif component_id in ('loss_mse','loss_l1'):
  require('target','float32');s=roles['target']['sampleShape']
  if not c['target_broadcast'] and s!=p:raise ValueError('Enable explicit sample broadcasting or provide an exactly matching target.')
  raw=sample_broadcast(p,s) if c['target_broadcast'] else p
 elif component_id=='loss_kl_div':
  require('target','float32',p);classes=p[-1]
  if not 2<=classes<=512:raise ValueError('KL comparison requires 2…512 final-direction probabilities.')
 elif component_id=='loss_ctc':
  if len(p)!=2 or not 2<=p[-1]<=512 or c['blank']>=p[-1]:raise ValueError('CTC prediction requires [observations,time,classes] and a blank within 2…512 classes.')
  require('target','int64');require('input_lengths','int64',[]);require('target_lengths','int64',[])
  if len(roles['target']['sampleShape'])!=1:raise ValueError('CTC prepared targets require one padded sequence direction.')
  raw=[];classes=p[-1];work=p[0]*(2*roles['target']['sampleShape'][0]+1)
 elif component_id=='loss_triplet_margin':
  if len(p)!=2 or p[0]!=3:raise ValueError('Triplet prediction must be [observations,3,width] in anchor/positive/negative order.')
  raw=[]
 if work>MAX_ELEMENTS:raise ValueError('Objective work exceeds 100M; shorten the prediction/target geometry.')
 return dict(configuration=c,targetRoles=roles,rawSampleShape=raw,classAxis=axis,classCount=classes,predictionElementsPerObservation=elements(p),rawElementsPerObservation=elements(raw),workPerObservation=work)

def capability(component_id):
 if component_id not in DEFAULTS:raise ValueError('Unknown objective.')
 return dict(version='1.0.0',adapterID='kwip.runner.objective.'+component_id,recipeSchemaVersions=[1],configurationVersion=1,predictionDTypes=['float32'],devices=['cpu','mps'],predictionRanks=[2,8],sampleDimensionMaximum=4096,classMinimum=2,classMaximum=512,classAxisMinimum=-7,classAxisMaximum=7,excludesBatchClassAxis=True,reductions=['none','mean','sum','batchmean'] if component_id=='loss_kl_div' else ['none','mean','sum'],scalarizations=['mean','sum'],targetRoles=['target','input_lengths','target_lengths'] if component_id=='loss_ctc' else [] if component_id=='loss_triplet_margin' else ['target','weight','pos_weight'] if component_id=='loss_bce_with_logits' else ['target'],maximumTensorElements=MAX_ELEMENTS,maximumWorkElements=MAX_ELEMENTS,recipeMaximumBytes=32768,fixedWeights='buffers',mappedWeights='preparedData',batchBroadcast=False)
