"""Native attention execution with explicit binary mask and allocation admission."""
import math
import torch
if __package__:
 from .attention_configuration import normalize_attention,resolve_attention,configured_input_ports,parameter_shapes,MASK_FIELDS,FLOAT_PORTS,MAXIMUM,constructor_options
 from .tensor_layout import BatchedLayout
else:
 from attention_configuration import normalize_attention,resolve_attention,configured_input_ports,parameter_shapes,MASK_FIELDS,FLOAT_PORTS,MAXIMUM,constructor_options
 from tensor_layout import BatchedLayout
class ScaledDotProductAttention(torch.nn.Module):
 def __init__(self,dropout=0.,scale=None):super().__init__();self.dropout=dropout;self.scale=scale
 def forward(self,q,k,v,mask=None):
  if q.device.type=='mps' and self.training and self.dropout>0:
   scores=torch.matmul(q,k.transpose(-2,-1))*(self.scale if self.scale is not None else 1/math.sqrt(q.shape[-1]))
   if not torch.isfinite(scores).all():raise ValueError('Attention comparison scores must remain finite. Reduce input magnitude or the custom scale.')
   if mask is not None:scores=scores.masked_fill(~mask,float('-inf'))
   probabilities=torch.softmax(scores,dim=-1)
   if not torch.isfinite(probabilities).all():raise ValueError('Attention probabilities must remain finite with an eligible source per query.')
   return torch.matmul(torch.nn.functional.dropout(probabilities,p=self.dropout,training=True),v)
  return torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=self.dropout if self.training else 0.,scale=self.scale,enable_gqa=False)
 def load_state_dict(self,state_dict,strict=True,assign=False):
  validate_attention_state(state_dict,self.state_dict());return super().load_state_dict(state_dict,strict=strict,assign=assign)
def validate_attention_state(state,expected):
 if not isinstance(state,dict) or set(state)!=set(expected):raise ValueError('Attention learned-state names differ from the captured model.')
 for name,value in state.items():
  prior=expected[name]
  if not isinstance(value,torch.Tensor) or value.layout!=torch.strided or value.shape!=prior.shape or value.dtype!=prior.dtype or not torch.isfinite(value).all():raise ValueError('Attention learned-state shape, dtype or finite values are incompatible.')
def create_attention_module(operation,config):
 c=normalize_attention(operation,config);kw=constructor_options(operation,c)
 if operation in ('attention_mask','rotary_position'):return None
 if operation=='scaled_dot_product_attention':return ScaledDotProductAttention(c['dropout'],c['scale'])
 if operation=='multihead_attention':return torch.nn.MultiheadAttention(**kw)
 if operation=='transformer':
  module=torch.nn.Transformer(**kw);module.encoder.enable_nested_tensor=False;module.encoder.use_nested_tensor=False;return module
 norm=torch.nn.LayerNorm(c['d_model'],eps=c['layer_norm_eps'],bias=c['bias']) if c['final_norm'] else None
 if operation=='transformer_encoder':return torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(**kw),c['num_layers'],norm=norm,enable_nested_tensor=False)
 return torch.nn.TransformerDecoder(torch.nn.TransformerDecoderLayer(**kw),c['num_layers'],norm=norm)
def _admit_mask(mask,shape,n,device):
 if not isinstance(mask,torch.Tensor) or mask.layout!=torch.strided or mask.dtype!=torch.int64 or mask.device!=device or tuple(mask.shape)!=(n,*shape) or mask.requires_grad:raise ValueError('Connect exact dense whole-number masks with matching observations and device.')
 if not ((mask==0)|(mask==1)).all():raise ValueError('Mask values must be zero allowed or one blocked.')
 return mask.bool()
def _combined_mask(attention,padding,n,l,s,device):
 if attention is None:blocked=padding[:,None,:].expand(n,l,s) if padding is not None else None
 else:blocked=attention|(padding[:,None,:] if padding is not None else False)
 if blocked is not None and blocked.all(dim=-1).any():raise ValueError('Every query must retain at least one eligible original source. Unblock a connection or padding key.')
 return blocked
def execute_attention(operation,config,sources,layer):
 if operation=='rotary_position':return execute_rotary_position(config,sources,layer)
 c=normalize_attention(operation,config);ports=configured_input_ports(operation,c)
 if type(sources) not in (list,tuple) or len(sources)!=len(ports) or any(not isinstance(x,torch.Tensor) or x.layout!=torch.strided for x in sources):raise ValueError('Connect every enabled dense attention operand exactly once.')
 if sources[0].ndim<2 or sources[0].shape[0]<1:raise ValueError('Attention needs a positive observation count.')
 n=sources[0].shape[0];device=sources[0].device;inputs=dict(zip(ports,sources))
 for port,value in inputs.items():
  if value.device!=device or value.ndim<2 or value.shape[0]!=n or value.dtype!=(torch.float32 if port in FLOAT_PORTS[operation] else torch.int64):raise ValueError('Attention operands need matching observations/device and their declared float32 or int64 type.')
 resolved=resolve_attention(operation,c,[BatchedLayout(tuple(x.shape[1:])) for x in sources])
 for count in [resolved['outputElementsPerObservation'],resolved['scoreElementsPerObservation'],resolved['maskElementsPerObservation'],resolved['workPerObservation'],*[math.prod(x.shape[1:]) for x in sources]]:
  if count>MAXIMUM//n:raise ValueError('Attention exceeds its observation, score, mask or computation budget before allocation.')
 if any(not torch.isfinite(inputs[p]).all() for p in FLOAT_PORTS[operation]):raise ValueError('Attention reference values must be finite.')
 if operation=='attention_mask':
  l,s=sources[0].shape[-2],sources[1].shape[-2];mask=torch.zeros(n,l,s,dtype=torch.int64,device=device);padding=torch.zeros(n,s,dtype=torch.int64,device=device)
  if c['mode']=='causal':mask=torch.triu(torch.ones(l,s,dtype=torch.int64,device=device),diagonal=1).expand(n,l,s).clone()
  for q,k in zip(c['blocked_queries'],c['blocked_keys']):mask[:,q,k]=1
  for k in c['padding_keys']:padding[:,k]=1;mask[:,:,k]=1
  return {'mask':mask,'padding':padding}
 classes={'multihead_attention':torch.nn.MultiheadAttention,'transformer':torch.nn.Transformer,'transformer_encoder':torch.nn.TransformerEncoder,'transformer_decoder':torch.nn.TransformerDecoder,'scaled_dot_product_attention':ScaledDotProductAttention}
 if type(layer) is not classes[operation]:raise ValueError('Attention native module differs from its reviewed constructor.')
 actual=[{'name':name,'shape':tuple(value.shape)} for name,value in layer.named_parameters()]
 if actual!=parameter_shapes(operation,c):raise ValueError('Attention learned parameter names, order or shapes are incompatible.')
 if any(p.dtype!=torch.float32 or p.device!=device or p.layout!=torch.strided or not torch.isfinite(p).all() for p in layer.parameters()):raise ValueError('Attention learned parameters must have finite float32 values on the input device.')
 # Parameter shapes alone do not capture head grouping, dropout, normalization or zero-position augmentation.
 for module in layer.modules():
  if type(module) is torch.nn.MultiheadAttention:
   expected_heads=c.get('num_heads',c.get('nhead'));expected_width=c.get('embed_dim',c.get('d_model'))
   if module.num_heads!=expected_heads or module.embed_dim!=expected_width or module.dropout!=c['dropout'] or not module.batch_first or module.add_zero_attn!=(c.get('add_zero_attn',False)):raise ValueError('Attention comparison groups or constructor options differ from the captured configuration.')
   if operation=='multihead_attention' and (module.kdim!=(c['kdim'] or c['embed_dim']) or module.vdim!=(c['vdim'] or c['embed_dim']) or (module.bias_k is not None)!=c['add_bias_kv']):raise ValueError('Attention source widths or learned extra positions differ from configuration.')
  if isinstance(module,(torch.nn.TransformerEncoderLayer,torch.nn.TransformerDecoderLayer)):
   if module.norm_first!=c['norm_first'] or module.activation is not (torch.nn.functional.relu if c['activation']=='relu' else torch.nn.functional.gelu):raise ValueError('Attention residual order or activation differs from configuration.')
  if type(module) is torch.nn.LayerNorm and module.eps!=c['layer_norm_eps']:raise ValueError('Attention normalization safety differs from configuration.')
  if type(module) is torch.nn.Dropout and module.p!=c['dropout']:raise ValueError('Attention dropout differs from configuration.')

 mask_values={}
 for field,port,_ in MASK_FIELDS[operation]:
  if c[field]=='supplied':mask_values[port]=_admit_mask(inputs[port],tuple(inputs[port].shape[1:]),n,device)
  elif c[field]=='causal':
   if port in ('attn_mask','mask'):length=sources[0].shape[-2]
   elif port=='src_mask':length=sources[0].shape[1]
   else:length=sources[0 if operation=='transformer_decoder' else 1].shape[1]
   mask_values[port]=torch.triu(torch.ones(length,length,dtype=torch.bool,device=device),diagonal=1).expand(n,length,length)
 if operation=='scaled_dot_product_attention':
  q,k,v=sources[:3];blocked=_combined_mask(mask_values.get('mask'),None,n,q.shape[-2],k.shape[-2],device)
  if layer.dropout!=c['dropout'] or layer.scale!=c['scale']:raise ValueError('SDPA mode or scale differs from its captured configuration.')
  result={'out':layer(q,k,v,None if blocked is None else ~blocked[:,None,:,:])}
 else:
  h=c.get('num_heads',c.get('nhead'))
  def admit(attention,padding,l,s):
   _combined_mask(mask_values.get(attention),mask_values.get(padding),n,l,s,device)
  def broadcast(port):
   mask=mask_values.get(port)
   return None if mask is None else mask[:,None,:,:].expand(n,h,*mask.shape[1:]).reshape(n*h,*mask.shape[1:])
  if operation=='multihead_attention':
   q,k,v=sources[:3];admit('attn_mask','key_padding_mask',q.shape[1],k.shape[1]);out,weights=layer(q,k,v,attn_mask=broadcast('attn_mask'),key_padding_mask=mask_values.get('key_padding_mask'),need_weights=c['return_weights'],average_attn_weights=c['average_weights'])
   result={'out':out}
   if c['return_weights']:result['weights']=weights
  elif operation=='transformer_encoder':
   x=sources[0];admit('mask','src_key_padding_mask',x.shape[1],x.shape[1]);result={'out':layer(x,mask=broadcast('mask'),src_key_padding_mask=mask_values.get('src_key_padding_mask'),is_causal=False)}
  elif operation=='transformer_decoder':
   target,memory=sources[:2];admit('tgt_mask','tgt_key_padding_mask',target.shape[1],target.shape[1]);admit('memory_mask','memory_key_padding_mask',target.shape[1],memory.shape[1]);result={'out':layer(target,memory,tgt_mask=broadcast('tgt_mask'),memory_mask=broadcast('memory_mask'),tgt_key_padding_mask=mask_values.get('tgt_key_padding_mask'),memory_key_padding_mask=mask_values.get('memory_key_padding_mask'),tgt_is_causal=False,memory_is_causal=False)}
  else:
   source,target=sources[:2];admit('src_mask','src_key_padding_mask',source.shape[1],source.shape[1]);admit('tgt_mask','tgt_key_padding_mask',target.shape[1],target.shape[1]);admit('memory_mask','memory_key_padding_mask',target.shape[1],source.shape[1]);result={'out':layer(source,target,src_mask=broadcast('src_mask'),tgt_mask=broadcast('tgt_mask'),memory_mask=broadcast('memory_mask'),src_key_padding_mask=mask_values.get('src_key_padding_mask'),tgt_key_padding_mask=mask_values.get('tgt_key_padding_mask'),memory_key_padding_mask=mask_values.get('memory_key_padding_mask'),src_is_causal=False,tgt_is_causal=False,memory_is_causal=False)}
 if set(result)!=set(resolved['outputs']):raise ValueError('Attention returned undeclared or omitted outputs.')
 for port,value in result.items():
  if not isinstance(value,torch.Tensor) or value.layout!=torch.strided or value.device!=device or value.dtype!=torch.float32 or tuple(value.shape)!=(n,*resolved['outputs'][port].sample) or not torch.isfinite(value).all():raise ValueError('Attention output shape, dtype, device or finite values differ from its declared role.')
 return result

def execute_rotary_position(config,sources,layer):
 c=normalize_attention('rotary_position',config);count=2
 if layer is not None or type(sources) not in (list,tuple) or len(sources)!=count or any(not isinstance(x,torch.Tensor) or x.layout!=torch.strided for x in sources):raise ValueError('RoPE needs its exact dense inputs and no learned module.')
 q,k=sources[:2]
 if q.ndim<3 or q.shape[0]<1:raise ValueError('RoPE requires positive observations and sequence/feature axes.')
 n=q.shape[0];device=q.device
 for i,x in enumerate(sources):
  if x.ndim<2 or x.shape[0]!=n or x.device!=device or x.dtype!=torch.float32:raise ValueError('RoPE operands need matching observations/device, float32 Query and Key.')
 resolved=resolve_attention('rotary_position',c,[BatchedLayout(tuple(x.shape[1:])) for x in sources])
 if any(resolved[key]>MAXIMUM//n for key in ('outputElementsPerObservation','workPerObservation')):raise ValueError('RoPE exceeds its batched allocation/work allowance before rotation.')
 if not torch.isfinite(q).all() or not torch.isfinite(k).all():raise ValueError('RoPE reference values must be finite.')
 t=resolved['length'];r=resolved['rotated'];axis=resolved['axis']
 positions=torch.arange(c['start_position'],c['start_position']+t,device=device,dtype=torch.int64).expand(n,t)
 if (positions<0).any() or (positions>1_000_000).any() or (positions.float()/c['position_scale']>1_000_000).any():raise ValueError('Positions and scaled effective positions must be from0 to one million.')
 frequencies=torch.pow(torch.tensor(c['base'],device=device,dtype=torch.float32),-torch.arange(0,r,2,device=device,dtype=torch.float32)/r)
 angles=positions.float().unsqueeze(-1)*frequencies/c['position_scale'];cos=angles.cos();sin=angles.sin()
 def apply(x):
  moved=x.movedim(axis,-2);prefix=moved[...,:r];shape=(n,)+(1,)*(x.ndim-3)+(t,r//2)
  co,si=cos.reshape(shape),sin.reshape(shape)
  if c['pairing']=='adjacent':
   a,b=prefix[...,0::2],prefix[...,1::2]
   rotated=torch.stack((a*co-b*si,a*si+b*co),dim=-1).flatten(-2)
  else:
   a,b=prefix[...,:r//2],prefix[...,r//2:]
   rotated=torch.cat((a*co-b*si,a*si+b*co),dim=-1)
  out=torch.cat((rotated,moved[...,r:]),dim=-1).movedim(-2,axis)
  if not torch.isfinite(out).all():raise ValueError('Rotary output must remain finite; reduce input magnitude.')
  return out
 return {'query':apply(q),'key':apply(k)}
