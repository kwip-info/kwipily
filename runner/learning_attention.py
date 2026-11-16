"""Genuine native CPUfloat64 attention measurements; gold stays outside this worker."""
import math
import torch
try:
 from .attention_configuration import normalize_attention,configured_input_ports,parameter_shapes,constructor_options,resolve_attention
 from .tensor_layout import BatchedLayout
except ImportError:
 from attention_configuration import normalize_attention,configured_input_ports,parameter_shapes,constructor_options,resolve_attention
 from tensor_layout import BatchedLayout
TASKS={'multihead-attention-connections':'multihead_attention','transformer-source-target':'transformer','transformer-encoder-context':'transformer_encoder','transformer-decoder-memory':'transformer_decoder','scaled-attention-similarity':'scaled_dot_product_attention'}
def target(op):
 if op=='multihead_attention':return dict(embed_dim=2,num_heads=1,kdim=2,vdim=2,dropout=0.,bias=True,add_bias_kv=False,add_zero_attn=False,return_weights=True,average_weights=False,attention_mask_mode='supplied',padding_mask_mode='supplied')
 if op=='scaled_dot_product_attention':return dict(dropout=0.,scale=None,mask_mode='supplied')
 c=dict(d_model=2,nhead=1,dim_feedforward=3,dropout=0.,activation='gelu',layer_norm_eps=1e-5,norm_first=True,bias=True)
 if op=='transformer':c.update(num_encoder_layers=1,num_decoder_layers=1,source_mask_mode='none',target_mask_mode='causal',memory_mask_mode='supplied',source_padding_mode='none',target_padding_mode='none',memory_padding_mode='none')
 elif op=='transformer_encoder':c.update(num_layers=1,final_norm=True,mask_mode='supplied',padding_mask_mode='none')
 else:c.update(num_layers=1,final_norm=True,target_mask_mode='causal',memory_mask_mode='supplied',target_padding_mode='none',memory_padding_mode='none')
 return c
def roles(op):return ['query','key','value'] if op in ('multihead_attention','scaled_dot_product_attention') else ['source','target'] if op=='transformer' else ['in'] if op=='transformer_encoder' else ['target','memory']
def samples(op):
 if op=='scaled_dot_product_attention':return {'query':[1,2,2],'key':[1,3,2],'value':[1,3,2]}
 if op=='multihead_attention':return {'query':[2,2],'key':[3,2],'value':[3,2]}
 if op=='transformer':return {'source':[3,2],'target':[2,2]}
 if op=='transformer_encoder':return {'in':[3,2]}
 return {'target':[2,2],'memory':[3,2]}
def outputs(op):return {'result':[2,2],'weights':[1,2,3]} if op=='multihead_attention' else {'result':[1,2,2]} if op=='scaled_dot_product_attention' else {'result':[3,2] if op=='transformer_encoder' else [2,2]}
def mask_roles(op):return ['in','in'] if op=='transformer_encoder' else ['target','source'] if op=='transformer' else ['target','memory'] if op=='transformer_decoder' else ['query','key']
def mask_port(op):return 'memory_mask' if op in ('transformer','transformer_decoder') else 'attn_mask' if op=='multihead_attention' else 'mask'
def configuration(submission,task):
 if task not in TASKS or type(submission) is not dict or set(submission)!={'schemaVersion','kind','graph'} or type(submission['schemaVersion']) is not int or submission['schemaVersion']!=1 or submission['kind']!='attentionGraph':raise ValueError('Use the attention graph; learned tables, fixtures and answers belong to the checker.')
 op=TASKS[task];g=submission['graph']
 if type(g) is not dict or set(g)!={'schemaVersion','name','nodes','edges'} or type(g['schemaVersion']) is not int or g['schemaVersion']!=2 or type(g['name']) is not str or len(g['name'])>256 or type(g['nodes']) is not list or len(g['nodes'])!=len(roles(op))+len(outputs(op))+2 or type(g['edges']) is not list or len(g['edges'])>9:raise ValueError('Connect the supplied sequences, one Attention mask, the requested attention block and every result.')
 nodes={};source={};sink={};operator=None;mask=None
 for node in g['nodes']:
  if type(node) is not dict or set(node)!={'id','type','name','x','y','config'} or type(node['id']) is not str or not 1<=len(node['id'])<=128 or node['id'] in nodes or type(node['name']) is not str or len(node['name'])>256 or any(type(node[k]) not in (int,float) or not math.isfinite(node[k]) for k in ('x','y')):raise ValueError('Malformed attention block identity or position.')
  nodes[node['id']]=node;c=node['config']
  if node['type']=='input':
   if type(c) is not dict or set(c)!={'fixture'} or c['fixture'] not in roles(op) or c['fixture'] in source:raise ValueError('Use each supplied sequence role exactly once.')
   source[c['fixture']]=node['id']
  elif node['type']=='output':
   if type(c) is not dict or set(c)!={'role'} or c['role'] not in outputs(op) or c['role'] in sink:raise ValueError('Use each named result role exactly once.')
   sink[c['role']]=node['id']
  elif node['type']==op and operator is None:operator=node
  elif node['type']=='attention_mask' and mask is None:mask=node
  else:raise ValueError('Use only the requested attention block, mask and role boundaries.')
 if operator is None or mask is None or set(source)!=set(roles(op)) or set(sink)!=set(outputs(op)):raise ValueError('Complete every supplied sequence, mask and result role.')
 c=normalize_attention(op,operator['config']);mc=normalize_attention('attention_mask',mask['config']);t=normalize_attention(op,target(op))
 fixed=['dropout']+(['embed_dim','num_heads','kdim','vdim','add_bias_kv','add_zero_attn','return_weights','average_weights'] if op=='multihead_attention' else [] if op=='scaled_dot_product_attention' else ['d_model','nhead','dim_feedforward']+[k for k in ('num_layers','num_encoder_layers','num_decoder_layers','target_mask_mode') if k in t])
 if any(c[k]!=t[k] for k in fixed) or configured_input_ports(op,c)!=configured_input_ports(op,t):raise ValueError('Keep the supplied widths, layers, no-dropout and named sequence/mask/result geometry for this exercise.')
 expected={(ident,'out',operator['id'],role) for role,ident in source.items()}|{(operator['id'],'out' if role=='result' else role,ident,'in') for role,ident in sink.items()}
 for port,role in zip(('query','key'),mask_roles(op)):expected.add((source[role],'out',mask['id'],port))
 expected.add((mask['id'],'mask',operator['id'],mask_port(op)))
 if op=='multihead_attention':expected.add((mask['id'],'padding',operator['id'],'key_padding_mask'))
 actual=set();ids=set()
 for edge in g['edges']:
  if type(edge) is not dict or set(edge)!={'id','source','sourcePort','target','targetPort'} or type(edge['id']) is not str or not 1<=len(edge['id'])<=128 or edge['id'] in ids or any(type(edge[k]) is not str for k in ('source','sourcePort','target','targetPort')):raise ValueError('Malformed attention port connection.')
  ids.add(edge['id']);actual.add((edge['source'],edge['sourcePort'],edge['target'],edge['targetPort']))
 if actual!=expected or len(g['edges'])!=len(expected):raise ValueError('Connect each supplied role, mask reference, mask output and named result to its matching port.')
 geometry=samples(op);refs=[BatchedLayout(geometry[r]) for r in mask_roles(op)];resolve_attention('attention_mask',mc,refs)
 l=geometry[mask_roles(op)[0]][-2];s=geometry[mask_roles(op)[1]][-2]
 blocked=[[mc['mode']=='causal' and j>i or mc['mode']=='blocked' and (i,j) in set(zip(mc['blocked_queries'],mc['blocked_keys'])) for j in range(s)] for i in range(l)]
 padding=[j in mc['padding_keys'] for j in range(s)]
 blocked=[[b or padding[j] for j,b in enumerate(row)] for row in blocked]
 if any(all(row) for row in blocked):raise ValueError('Every query must retain at least one allowed original source position.')
 layouts=[BatchedLayout(geometry[p] if p in geometry else [s] if p=='key_padding_mask' else [l,s]) for p in configured_input_ports(op,c)];resolve_attention(op,c,layouts)
 return op,c,blocked,padding

def execute(submission,fixtures,evidence,stopped,tensor,*,task):
 if stopped():raise ValueError('Attention lesson cancelled.')
 op,c,blocked,padding=configuration(submission,task);ps={p['name']:list(p['shape']) for p in parameter_shapes(op,target(op))};required=set(roles(op))|{'parameter.'+k for k in ps}|{'upstream.'+r for r in outputs(op)}
 if type(fixtures) is not list or not 1<=len(fixtures)<=32:raise ValueError('Use one to32 bounded checker-owned fixtures.')
 checked=[];ids=set()
 for fixture in fixtures:
  if type(fixture) is not dict or set(fixture)!={'id','inputs'} or type(fixture['id']) is not str or not 1<=len(fixture['id'])<=128 or fixture['id'] in ids or type(fixture['inputs']) is not dict or set(fixture['inputs'])!=required:raise ValueError('All sequences, learned tables and influence arrays must be checker-owned.')
  ids.add(fixture['id']);values={}
  for key,raw in fixture['inputs'].items():
   shape,flat=tensor(raw)
   if not isinstance(shape,(tuple,list)) or any(type(d) is not int or d<=0 for d in shape) or math.prod(shape)>512 or not isinstance(flat,(tuple,list)) or len(flat)!=math.prod(shape) or any(type(v) not in (int,float) or not math.isfinite(v) for v in flat):raise ValueError('Use bounded finite checker-owned arrays.')
   values[key]=(list(shape),list(flat))
  n=values[roles(op)[0]][0][0]
  if n not in (1,2,3):raise ValueError('Use one to three independent observations.')
  expected={**{r:[n]+s for r,s in samples(op).items()},**{'parameter.'+k:s for k,s in ps.items()},**{'upstream.'+r:[n]+s for r,s in outputs(op).items()}}
  if any(values[k][0]!=s for k,s in expected.items()):raise ValueError('A supplied sequence, learned table or result influence has the wrong shape.')
  checked.append((fixture['id'],values,n))
 for ident,values,n in checked:
  if stopped():raise ValueError('Attention lesson cancelled.')
  options=constructor_options(op,c)
  if op=='multihead_attention':module=torch.nn.MultiheadAttention(**options)
  elif op=='transformer':module=torch.nn.Transformer(**options);module.encoder.enable_nested_tensor=False;module.encoder.use_nested_tensor=False
  elif op in ('transformer_encoder','transformer_decoder'):
   layer=(torch.nn.TransformerEncoderLayer if op=='transformer_encoder' else torch.nn.TransformerDecoderLayer)(**options);norm=torch.nn.LayerNorm(2,eps=c['layer_norm_eps'],bias=c['bias']) if c['final_norm'] else None
   module=torch.nn.TransformerEncoder(layer,1,norm=norm,enable_nested_tensor=False) if op=='transformer_encoder' else torch.nn.TransformerDecoder(layer,1,norm=norm)
  else:module=None
  active={}
  if module is not None:
   module=module.to(dtype=torch.float64).eval();active=dict(module.named_parameters())
   if any(k not in ps or list(p.shape)!=ps[k] for k,p in active.items()):raise ValueError('The native lesson learned-table ownership differs from the trusted fixture.')
   with torch.no_grad():
    for k,p in active.items():p.copy_(torch.tensor(values['parameter.'+k][1],dtype=torch.float64).reshape(values['parameter.'+k][0]))
  x={r:torch.tensor(values[r][1],dtype=torch.float64).reshape(values[r][0]).requires_grad_(True) for r in roles(op)}
  mask=torch.tensor(blocked,dtype=torch.bool);pad=torch.tensor(padding,dtype=torch.bool).expand(n,-1);causal=torch.triu(torch.ones(2,2,dtype=torch.bool),1)
  if op=='multihead_attention':out,weights=module(x['query'],x['key'],x['value'],attn_mask=mask,key_padding_mask=pad,need_weights=True,average_attn_weights=False);measured={'result':out,'weights':weights}
  elif op=='scaled_dot_product_attention':measured={'result':torch.nn.functional.scaled_dot_product_attention(x['query'],x['key'],x['value'],attn_mask=~mask,dropout_p=0.,scale=c['scale'])}
  elif op=='transformer_encoder':measured={'result':module(x['in'],mask=mask)}
  elif op=='transformer_decoder':measured={'result':module(x['target'],x['memory'],tgt_mask=causal,memory_mask=mask)}
  else:measured={'result':module(x['source'],x['target'],tgt_mask=causal,memory_mask=mask)}
  loss=sum((v*torch.tensor(values['upstream.'+r][1],dtype=torch.float64).reshape(values['upstream.'+r][0])).sum() for r,v in measured.items());loss.backward()
  for r,v in x.items():measured[r+'Gradient']=v.grad
  for k,s in ps.items():measured['parameter.'+k+'Gradient']=active[k].grad if k in active else torch.zeros(s,dtype=torch.float64)
  if any(v is None or not torch.isfinite(v).all().item() for v in measured.values()):raise ValueError('Attention produced unavailable or nonfinite result/influence.')
  measured['blockedConnections']=mask.to(dtype=torch.int64).unsqueeze(0).expand(n,-1,-1)
  measured['blockedSourcePositions']=pad.to(dtype=torch.int64)
  tables={k:dict(shape=list(v.shape),values=v.detach().reshape(-1).tolist()) for k,v in measured.items()};evidence['maximumTensorElements']=max(evidence.get('maximumTensorElements',0),max(v.numel() for v in measured.values()));evidence['fixtures'].append(dict(fixtureID=ident,tensors=tables,partitions={}))
