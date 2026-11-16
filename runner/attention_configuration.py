"""Torch-free attention configuration, observation-first geometry and owned state."""
import math
try:
 from .tensor_layout import BatchedLayout
except ImportError:
 from tensor_layout import BatchedLayout
OPERATIONS=frozenset(('multihead_attention','transformer','transformer_encoder','transformer_decoder','scaled_dot_product_attention','attention_mask','rotary_position'))
MAXIMUM=100_000_000
MASK_FIELDS={
 'multihead_attention':(('attention_mask_mode','attn_mask','causal'),('padding_mask_mode','key_padding_mask',None)),
 'transformer':(('source_mask_mode','src_mask','causal'),('target_mask_mode','tgt_mask','causal'),('memory_mask_mode','memory_mask',None),('source_padding_mode','src_key_padding_mask',None),('target_padding_mode','tgt_key_padding_mask',None),('memory_padding_mode','memory_key_padding_mask',None)),
 'transformer_encoder':(('mask_mode','mask','causal'),('padding_mask_mode','src_key_padding_mask',None)),
 'transformer_decoder':(('target_mask_mode','tgt_mask','causal'),('memory_mask_mode','memory_mask',None),('target_padding_mode','tgt_key_padding_mask',None),('memory_padding_mode','memory_key_padding_mask',None)),
 'scaled_dot_product_attention':(('mask_mode','mask','causal'),),
 'attention_mask':(), 'rotary_position':()}
FLOAT_PORTS={'multihead_attention':('query','key','value'),'transformer':('source','target'),'transformer_encoder':('in',),'transformer_decoder':('target','memory'),'scaled_dot_product_attention':('query','key','value'),'attention_mask':('query','key'),'rotary_position':('query','key')}
def _integer(v,name,lo,hi):
 if type(v) not in (int,float) or not math.isfinite(v) or v!=int(v) or not lo<=v<=hi:raise ValueError(f'{name} must be a whole number from {lo} to {hi}.')
 return int(v)
def _number(v,name,lo,hi,positive=False):
 if type(v) not in (int,float) or not math.isfinite(v) or not lo<=v<=hi or positive and v<=0:raise ValueError(f'{name} must be a finite number in the declared range.')
 return float(v)
ROTARY_DEFAULTS={'sequence_axis':-2,'rotation':'full','rotary_dim':2,'pairing':'adjacent','base':10000.,'start_position':0,'position_scale':1.}
def normalize_rotary_position(config):
 if type(config) is not dict or set(config)-set(ROTARY_DEFAULTS):raise ValueError('Rotary position accepts only its declared axis, rotation and position controls.')
 c={**ROTARY_DEFAULTS,**config}
 for key,choices in [('rotation',('full','custom')),('pairing',('adjacent','halves'))]:
  if type(c[key]) is not str or c[key] not in choices:raise ValueError('Choose a supported rotary arrangement, pairing or position source.')
 c['sequence_axis']=_integer(c['sequence_axis'],'Sequence axis',-3,2)
 if c['sequence_axis'] in (0,-1):raise ValueError('Sequence axis cannot be the observation or final feature axis.')
 c['rotary_dim']=_integer(c['rotary_dim'],'Rotated width',2,4096)
 if c['rotary_dim']%2:raise ValueError('Rotated width must be even.')
 c['start_position']=_integer(c['start_position'],'Starting position',0,1_000_000)
 c['base']=_number(c['base'],'Frequency base',1.,1_000_000_000.)
 if c['base']<=1:raise ValueError('Frequency base must be greater than one.')
 c['position_scale']=_number(c['position_scale'],'Position scale',.0001,10000.)
 return c

def resolve_rotary_position(config,layouts):
 c=normalize_rotary_position(config);count=2
 if type(layouts) not in (list,tuple) or len(layouts)!=count or any(not isinstance(x,BatchedLayout) for x in layouts):raise ValueError('Connect exactly the Query and Key tensors; positions belong to Configuration.')
 q,k=(x.sample for x in layouts[:2]);rank=len(q)+1;axis=c['sequence_axis']%rank
 if len(q) not in (2,3) or len(k)!=len(q) or axis not in range(1,rank-1) or any(not 1<=v<=4096 for shape in (q,k) for v in shape):raise ValueError('RoPE requires rank3/4 tensors with an explicit non-batch, non-feature sequence axis and bounded positive extents.')
 t=q[axis-1];width=q[-1];rotated=width if c['rotation']=='full' else c['rotary_dim']
 if k[axis-1]!=t or k[-1]!=width or rotated>width or rotated%2:raise ValueError('Match Query/Key sequence and feature widths; the rotated prefix must fit and be even.')
 if rank==4 and any(shape[2-axis]>32 for shape in (q,k)):raise ValueError('RoPE supports at most32 heads per operand.')
 if (c['start_position']+t-1>1_000_000 or (c['start_position']+t-1)/c['position_scale']>1_000_000):raise ValueError('Sequential/effective positions exceed one million; reduce the offset or increase position scale.')
 physical=sum(math.prod(x) for x in (q,k));work=8*(physical+t*rotated)
 if physical>MAXIMUM or work>MAXIMUM:raise ValueError('Rotary position exceeds its bounded tensor/work allowance.')
 return {'config':c,'outputs':{'query':layouts[0],'key':layouts[1]},'outputDTypes':{'query':'float32','key':'float32'},'parameterCount':0,'outputElementsPerObservation':physical,'scoreElementsPerObservation':0,'maskElementsPerObservation':0,'workPerObservation':work,'axis':axis,'rotated':rotated,'length':t}

def normalize_attention(operation,config):
 if operation=='rotary_position':return normalize_rotary_position(config)
 if operation not in OPERATIONS or not isinstance(config,dict):raise ValueError('Choose a supported attention component and declared configuration.')
 if operation=='attention_mask':c={'mode':'none','blocked_queries':[],'blocked_keys':[],'padding_keys':[]}
 elif operation=='scaled_dot_product_attention':c={'dropout':0.,'scale':None}
 elif operation=='multihead_attention':c={'embed_dim':4,'num_heads':2,'dropout':0.,'bias':True,'kdim':None,'vdim':None,'add_bias_kv':False,'add_zero_attn':False,'return_weights':True,'average_weights':True}
 else:
  c={'d_model':4,'nhead':2,'dim_feedforward':8,'dropout':0.,'bias':True,'activation':'relu','layer_norm_eps':1e-5,'norm_first':False}
  if operation=='transformer':c.update(num_encoder_layers=1,num_decoder_layers=1)
  else:c.update(num_layers=1,final_norm=False)
 for field,_,_ in MASK_FIELDS[operation]:c[field]='none'
 if set(config)-set(c):raise ValueError('Use only the declared attention controls; learned weights and execution state cannot be authored.')
 c.update(config)
 for field in ('embed_dim','d_model'):
  if field in c:c[field]=_integer(c[field],field,1,512)
 for field in ('num_heads','nhead'):
  if field in c:c[field]=_integer(c[field],field,1,32)
 for field in ('kdim','vdim'):
  if field in c and c[field] is not None:c[field]=_integer(c[field],field,1,4096)
 for field in ('num_layers','num_encoder_layers','num_decoder_layers'):
  if field in c:c[field]=_integer(c[field],field,1,8)
 if 'dim_feedforward' in c:c['dim_feedforward']=_integer(c['dim_feedforward'],'dim_feedforward',1,4096)
 for field in ('bias','add_bias_kv','add_zero_attn','return_weights','average_weights','norm_first','final_norm'):
  if field in c and type(c[field]) is not bool:raise ValueError(f'{field} must be a Boolean switch.')
 if 'dropout' in c:c['dropout']=_number(c['dropout'],'dropout',0,1)
 if 'layer_norm_eps' in c:c['layer_norm_eps']=_number(c['layer_norm_eps'],'layer_norm_eps',1.401298464324817e-45,3.4028234663852886e38,True)
 if 'scale' in c and c['scale'] is not None:c['scale']=_number(c['scale'],'scale',1.401298464324817e-45,3.4028234663852886e38,True)
 if 'activation' in c and c['activation'] not in ('relu','gelu'):raise ValueError('Choose relu or gelu activation.')
 if operation not in ('attention_mask','scaled_dot_product_attention'):
  if c.get('embed_dim',c.get('d_model'))%c.get('num_heads',c.get('nhead')):raise ValueError('Model width must divide evenly between comparison groups.')
 for field,_,causal in MASK_FIELDS[operation]:
  if type(c[field]) is not str or c[field] not in (('none','supplied','causal') if causal else ('none','supplied')):raise ValueError(f'Choose a supported {field} arrangement.')
 if operation=='attention_mask':
  if type(c['mode']) is not str or c['mode'] not in ('none','causal','blocked'):raise ValueError('Choose none, causal or blocked connections.')
  for field in ('blocked_queries','blocked_keys','padding_keys'):
   if type(c[field]) is not list or len(c[field])>4096:raise ValueError('Mask coordinate lists support at most 4096 entries.')
   c[field]=[_integer(v,field,0,4095) for v in c[field]]
  pairs=list(zip(c['blocked_queries'],c['blocked_keys']))
  if len(c['blocked_queries'])!=len(c['blocked_keys']) or len(set(pairs))!=len(pairs) or len(set(c['padding_keys']))!=len(c['padding_keys']):raise ValueError('Mask pairs must match in count and contain unique connections; padding keys must be unique.')
  if pairs and c['mode']!='blocked':raise ValueError('Configured blocked connections require blocked mode.')
 if sum(math.prod(p['shape']) for p in parameter_shapes(operation,c,_normalized=True))>2_000_000:raise ValueError('Attention exceeds two million learned settings. Reduce widths or layers.')
 return c
def configured_input_ports(operation,config):
 c=normalize_attention(operation,config)
 return FLOAT_PORTS[operation]+tuple(port for field,port,_ in MASK_FIELDS[operation] if c[field]=='supplied')
def parameter_shapes(operation,config,_normalized=False):
 if operation=='rotary_position':
  if not _normalized:normalize_rotary_position(config)
  return []
 c=config if _normalized else normalize_attention(operation,config);out=[]
 def add(name,shape):out.append({'name':name,'shape':tuple(shape)})
 def mha(prefix,e,k,v,bias,augmentation=False):
  if k==e and v==e:add(prefix+'in_proj_weight',(3*e,e))
  else:
   for role,width in [('q',e),('k',k),('v',v)]:add(prefix+role+'_proj_weight',(e,width))
  if bias:add(prefix+'in_proj_bias',(3*e,))
  if augmentation:
   add(prefix+'bias_k',(1,1,e));add(prefix+'bias_v',(1,1,e))
  add(prefix+'out_proj.weight',(e,e))
  if bias:add(prefix+'out_proj.bias',(e,))
 def norm(prefix,e):
  add(prefix+'weight',(e,))
  if c['bias']:add(prefix+'bias',(e,))
 def stack(prefix,count,decoder):
  e=c['d_model'];f=c['dim_feedforward']
  for i in range(count):
   p=prefix+f'layers.{i}.';mha(p+'self_attn.',e,e,e,c['bias'])
   if decoder:mha(p+'multihead_attn.',e,e,e,c['bias'])
   for n,shape in [('linear1',(f,e)),('linear2',(e,f))]:
    add(p+n+'.weight',shape)
    if c['bias']:add(p+n+'.bias',(shape[0],))
   for j in range(1,4 if decoder else 3):norm(p+f'norm{j}.',e)
 if operation=='multihead_attention':mha('',c['embed_dim'],c['kdim'] or c['embed_dim'],c['vdim'] or c['embed_dim'],c['bias'],c['add_bias_kv'])
 elif operation=='transformer':
  stack('encoder.',c['num_encoder_layers'],False);norm('encoder.norm.',c['d_model']);stack('decoder.',c['num_decoder_layers'],True);norm('decoder.norm.',c['d_model'])
 elif operation in ('transformer_encoder','transformer_decoder'):
  stack('',c['num_layers'],operation=='transformer_decoder')
  if c['final_norm']:norm('norm.',c['d_model'])
 return out
def constructor_options(operation,config):
 c=normalize_attention(operation,config)
 if operation=='multihead_attention':return {k:c[k] for k in ('embed_dim','num_heads','dropout','bias','add_bias_kv','add_zero_attn','kdim','vdim')}|{'batch_first':True}
 if operation=='transformer':return {k:c[k] for k in ('d_model','nhead','num_encoder_layers','num_decoder_layers','dim_feedforward','dropout','activation','layer_norm_eps','norm_first','bias')}|{'batch_first':True}
 if operation in ('transformer_encoder','transformer_decoder'):return {k:c[k] for k in ('d_model','nhead','dim_feedforward','dropout','activation','layer_norm_eps','norm_first','bias')}|{'batch_first':True}
 return {}
def resolve_attention(operation,config,layouts):
 if operation=='rotary_position':return resolve_rotary_position(config,layouts)
 c=normalize_attention(operation,config);ports=configured_input_ports(operation,c)
 if type(layouts) not in (list,tuple) or len(layouts)!=len(ports) or any(not isinstance(x,BatchedLayout) for x in layouts):raise ValueError('Connect exactly the enabled attention inputs and mask ports.')
 source={p:l.sample for p,l in zip(ports,layouts)};floats=[source[p] for p in FLOAT_PORTS[operation]]
 for s in floats:
  if len(s) not in ((2,3) if operation=='attention_mask' else (3,) if operation=='scaled_dot_product_attention' else (2,)) or any(v<1 or v>4096 for v in s):raise ValueError('Use supported positive observation-first sequence dimensions, up to 4096.')
 outputs={};dtypes={};scores=work=masks=0;mask_shapes={}
 def output(port,s,dtype='float32'):outputs[port]=BatchedLayout(s);dtypes[port]=dtype
 def eligible(field,l,s):
  if c.get(field)=='causal' and l!=s:raise ValueError('Causal connections require equal query and source lengths; choose a supplied mask for cross attention.')
 def attention(l,s,e,k=None,v=None,extra=0,project=True):
  nonlocal scores,work
  h=c.get('num_heads',c.get('nhead',floats[0][0] if operation=='scaled_dot_product_attention' else 1));d=e//h if operation!='scaled_dot_product_attention' else e;dv=d if operation!='scaled_dot_product_attention' else floats[2][-1]
  scores+=h*l*(s+extra);work+=h*l*(s+extra)*(d+dv)
  if project:work+=l*e*e+s*(k or e)*e+s*(v or e)*e+l*e*e
 if operation=='attention_mask':
  q,k=floats
  if len(q)!=len(k) or len(q)==3 and (q[0]!=k[0] or q[0]>32):raise ValueError('Mask references need matching ranks and comparison-group counts.')
  l,s=q[-2],k[-2]
  if c['mode']=='causal' and l!=s:raise ValueError('Causal masks need equal reference lengths.')
  if any(v>=l for v in c['blocked_queries']) or any(v>=s for v in c['blocked_keys']+c['padding_keys']):raise ValueError('Mask coordinates must be inside the connected sequence geometry.')
  output('mask',(l,s),'int64');output('padding',(s,),'int64');masks=work=l*s+s
 elif operation=='scaled_dot_product_attention':
  q,k,v=floats;h,l,d=q;hk,s,dk=k;hv,sv,dv=v
  if not h==hk==hv or not 1<=h<=32 or d!=dk or s!=sv or max(d,dv)>512:raise ValueError('SDPA needs equal comparison groups/query-key widths, matching source lengths and supported widths.')
  eligible('mask_mode',l,s);attention(l,s,d,project=False);output('out',(h,l,dv));mask_shapes={'mask':(l,s)}
 elif operation=='multihead_attention':
  q,k,v=floats;l,e=q;s,kw=k;sv,vw=v
  if e!=c['embed_dim'] or kw!=(c['kdim'] or e) or vw!=(c['vdim'] or e) or s!=sv:raise ValueError('Connect exact query/key/value widths and matching key/value sequence lengths.')
  eligible('attention_mask_mode',l,s);extra=int(c['add_bias_kv'])+int(c['add_zero_attn']);attention(l,s,e,kw,vw,extra);output('out',(l,e))
  if c['return_weights']:output('weights',(l,s+extra) if c['average_weights'] else (c['num_heads'],l,s+extra))
  mask_shapes={'attn_mask':(l,s),'key_padding_mask':(s,)}
 else:
  e=c['d_model'];f=c['dim_feedforward']
  if any(x[-1]!=e for x in floats):raise ValueError('Every model sequence must have the declared width.')
  if operation=='transformer_encoder':s=floats[0][0];t=s;enc=c['num_layers'];dec=0
  elif operation=='transformer_decoder':t=floats[0][0];s=floats[1][0];enc=0;dec=c['num_layers']
  else:s=floats[0][0];t=floats[1][0];enc=c['num_encoder_layers'];dec=c['num_decoder_layers']
  for _ in range(enc):attention(s,s,e);work+=2*s*e*f
  for _ in range(dec):attention(t,t,e);attention(t,s,e);work+=2*t*e*f
  output('out',(t,e))
  if operation=='transformer_encoder':eligible('mask_mode',s,s);mask_shapes={'mask':(s,s),'src_key_padding_mask':(s,)}
  else:
   eligible('source_mask_mode',s,s);eligible('target_mask_mode',t,t)
   mask_shapes={'src_mask':(s,s),'tgt_mask':(t,t),'memory_mask':(t,s),'src_key_padding_mask':(s,),'tgt_key_padding_mask':(t,),'memory_key_padding_mask':(s,)}
 for field,port,_ in MASK_FIELDS[operation]:
  if c[field]=='supplied':
   if source[port]!=mask_shapes[port]:raise ValueError('Supplied binary masks must match every observation and exact query/source directions.')
   masks+=math.prod(mask_shapes[port])
  elif c[field]=='causal':
   masks+=math.prod(mask_shapes[port])
 physical=sum(math.prod(x.sample) for x in outputs.values());parameters=sum(math.prod(p['shape']) for p in parameter_shapes(operation,c,_normalized=True))
 if any(math.prod(s)>MAXIMUM for s in source.values()) or any(x>MAXIMUM for x in (physical,scores,masks,work)):raise ValueError('Attention exceeds its tensor, score, mask or computation budget. Reduce observations, widths or sequence lengths.')
 return {'config':c,'outputs':outputs,'outputDTypes':dtypes,'parameterCount':parameters,'outputElementsPerObservation':physical,'scoreElementsPerObservation':scores,'maskElementsPerObservation':masks,'workPerObservation':work}
def capability(operation):
 if operation=='rotary_position':return rotary_position_capability()
 if operation not in OPERATIONS:raise ValueError('Unknown attention component.')
 return {'version':{'major':1,'minor':0,'patch':0},'adapterID':'kwip.runner.graph-v2.'+operation,'graphSchemaVersions':[2],'preservesBatch':True,'graphDTypes':['float32'],'floatInputSampleRanks':[2,3] if operation=='attention_mask' else [3] if operation=='scaled_dot_product_attention' else [2],'maximumParameters':2_000_000,'maximumTensorElements':MAXIMUM,'maximumScoreElements':MAXIMUM,'maximumMaskElements':MAXIMUM,'maximumWorkElements':MAXIMUM,'sequenceMinimum':1,'sequenceMaximum':4096,'modelWidthMaximum':512,'inputWidthMaximum':4096,'headMaximum':32,'layerMaximum':8,'maskDType':'int64','maskConvention':'oneBlockedZeroAllowed','maskBroadcast':'perObservationAcrossHeadsOnly','statePersistence':'none','parameterOrder':'nativeModuleOrder','dropoutPolicy':'notApplicable' if operation=='attention_mask' else 'trainingOnly','utilityMaximumItems':4096,'utilityCoordinateMaximum':4095,'multiheadWeights':'nativeOptional' if operation=='multihead_attention' else 'notApplicable','sdpaGroupedQuery':False,'sdpaMPSDropoutExecution':'deviceLocalPublicOps' if operation=='scaled_dot_product_attention' else 'notApplicable'}

def rotary_position_capability():
 return {'version':{'major':1,'minor':0,'patch':0},'adapterID':'kwip.runner.graph-v2.rotary_position','graphSchemaVersions':[2],'preservesBatch':True,'floatInputSampleRanks':[2,3],'graphDTypes':['float32'],'positionSource':'inlineSequentialOnly','sequenceMaximum':4096,'featureMaximum':4096,'headMaximum':32,'positionMaximum':1_000_000,'effectivePositionMaximum':1_000_000,'frequencyBaseMinimumExclusive':1.,'frequencyBaseMaximum':1_000_000_000.,'positionScaleMinimum':.0001,'positionScaleMaximum':10000.,'pairings':['adjacent','halves'],'partialRotation':'evenPrefixPreserveSuffix','maximumTensorElements':MAXIMUM,'maximumWorkElements':MAXIMUM,'statePersistence':'none','rotationArithmetic':'realFloat32SinCos','sequentialPositions':'configuredOffsetPerCompleteForward'}
