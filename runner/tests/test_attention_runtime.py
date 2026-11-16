"""Attention helper TDD: real configuration, typed masks and independent geometry."""
import importlib
import pytest
from runner.tensor_layout import BatchedLayout
OPS=('multihead_attention','transformer','transformer_encoder','transformer_decoder','scaled_dot_product_attention','attention_mask')
def helper():return importlib.import_module('runner.attention_configuration')
@pytest.mark.parametrize('op',OPS)
def test_defaults_are_canonical_and_unknown_authorable_state_refuses(op):
 h=helper();c=h.normalize_attention(op,{})
 assert h.normalize_attention(op,c)==c
 for bad in [{'arbitrary':1},{'weights':[]},{'training':True}]:
  with pytest.raises(ValueError):h.normalize_attention(op,bad)

def test_explicit_mask_utility_coordinates_are_typed_and_bounded():
 h=helper();c=h.normalize_attention('attention_mask',{'mode':'blocked','blocked_queries':[0,1],'blocked_keys':[2,0],'padding_keys':[1]})
 r=h.resolve_attention('attention_mask',c,[BatchedLayout((2,4)),BatchedLayout((3,5))])
 assert {k:v.sample for k,v in r['outputs'].items()}=={'mask':(2,3),'padding':(3,)}
 assert r['outputDTypes']=={'mask':'int64','padding':'int64'} and r['parameterCount']==0
 for bad in [{'mode':'none','blocked_queries':[0],'blocked_keys':[0]},{'mode':'causal','blocked_queries':[0],'blocked_keys':[0]},{'mode':'blocked','blocked_queries':[0],'blocked_keys':[]},{'mode':'blocked','blocked_queries':[0,0],'blocked_keys':[1,1]},{'padding_keys':[1,1]},{'padding_keys':[True]},{'padding_keys':[-1]}]:
  with pytest.raises(ValueError):h.normalize_attention('attention_mask',bad)
 with pytest.raises(ValueError):h.resolve_attention('attention_mask',{'mode':'causal'},[BatchedLayout((2,4)),BatchedLayout((3,4))])
 with pytest.raises(ValueError):h.resolve_attention('attention_mask',{'mode':'blocked','blocked_queries':[2],'blocked_keys':[0]},[BatchedLayout((2,4)),BatchedLayout((3,4))])

@pytest.mark.parametrize('op',OPS)
def test_descriptors_have_no_implicit_state_and_production_float32_only(op):
 h=helper();d=h.capability(op)
 assert d['graphSchemaVersions']==[2] and d['graphDTypes']==['float32']
 assert d['preservesBatch'] is True and d['maximumTensorElements']==100_000_000
 assert d['statePersistence']=='none'

@pytest.mark.parametrize('op',['multihead_attention','transformer','transformer_encoder','transformer_decoder'])
@pytest.mark.parametrize('bias',[True,False])
def test_native_parameter_order_and_shapes_independently_crosschecked(op,bias):
 import torch
 h=helper();c=h.normalize_attention(op,{'bias':bias})
 kw=h.constructor_options(op,c)
 if op=='multihead_attention':m=torch.nn.MultiheadAttention(**kw)
 elif op=='transformer':m=torch.nn.Transformer(**kw)
 elif op=='transformer_encoder':m=torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(**kw),1,enable_nested_tensor=False)
 else:m=torch.nn.TransformerDecoder(torch.nn.TransformerDecoderLayer(**kw),1)
 assert h.parameter_shapes(op,c)==[{'name':k,'shape':tuple(p.shape)} for k,p in m.named_parameters()]

@pytest.mark.parametrize('op',['transformer','transformer_encoder','transformer_decoder'])
def test_stack_custom_final_normalization_and_multiple_layers_geometry(op):
 h=helper();c={'d_model':6,'nhead':3,'dim_feedforward':7,'norm_first':True,'activation':'gelu'}
 if op=='transformer':c.update(num_encoder_layers=2,num_decoder_layers=3);samples=[(4,6),(2,6)]
 elif op=='transformer_encoder':c.update(num_layers=2,final_norm=True);samples=[(4,6)]
 else:c.update(num_layers=3,final_norm=True);samples=[(2,6),(4,6)]
 r=h.resolve_attention(op,c,[BatchedLayout(x) for x in samples]);assert r['outputs']['out'].sample==samples[0 if op!='transformer' else 1]
 assert r['parameterCount']==sum(__import__('math').prod(x['shape']) for x in h.parameter_shapes(op,c))
 assert r['scoreElementsPerObservation']>0 and r['workPerObservation']>r['scoreElementsPerObservation']

@pytest.mark.parametrize('op',['multihead_attention','scaled_dot_product_attention','attention_mask'])
@pytest.mark.parametrize('device',['cpu','mps'])
def test_actual_execution_helper_missing_before_native_modes(op,device):
 import torch
 execution=importlib.import_module('runner.attention_execution')
 layer=execution.create_attention_module(op,{})
 if op=='multihead_attention':sources=[torch.ones(2,2,4,device=device),torch.ones(2,3,4,device=device),torch.ones(2,3,4,device=device)]
 elif op=='scaled_dot_product_attention':sources=[torch.ones(2,2,2,2,device=device),torch.ones(2,2,3,2,device=device),torch.ones(2,2,3,3,device=device)]
 else:sources=[torch.ones(2,2,4,device=device),torch.ones(2,3,4,device=device)]
 if layer is not None:layer=layer.to(device).eval()
 result=execution.execute_attention(op,{},sources,layer)
 assert set(result)==({'mask','padding'} if op=='attention_mask' else {'out','weights'} if op=='multihead_attention' else {'out'})
 assert all(x.device.type==device and torch.isfinite(x).all() for x in result.values())

def test_mask_utility_padding_is_reachable_through_sdpa_mask_output():
 import torch
 from runner.attention_execution import execute_attention
 q=torch.ones(2,2,4);k=torch.ones(2,3,4)
 result=execute_attention('attention_mask',{'padding_keys':[1]},[q,k],None)
 assert result['mask'][:,:,1].eq(1).all()
 assert result['mask'][:,:,[0,2]].eq(0).all() and result['padding'][:,1].eq(1).all()

@pytest.mark.parametrize('changed', [{'num_heads':1},{'add_zero_attn':True},{'dropout':.2}])
def test_native_mha_changed_constructor_controls_refuse_before_forward(changed,monkeypatch):
 import torch
 from runner.attention_execution import create_attention_module,execute_attention
 layer=create_attention_module('multihead_attention',changed)
 monkeypatch.setattr(layer,'forward',lambda *a,**kw:pytest.fail('Mismatched constructor reached native scoring'))
 with pytest.raises(ValueError):execute_attention('multihead_attention',{},[torch.ones(1,2,4)]*3,layer)

# Independent plain-Python attention/stack oracle. No native function computes expected values.
def scalar_attention(q,k,v,heads,blocked=None,scale=None):
 import math
 width=len(q[0])//heads;value_width=len(v[0])//heads;all_weights=[];result=[[0.]*(heads*value_width) for _ in q]
 for head in range(heads):
  weights=[]
  for i,row in enumerate(q):
   score=[sum(row[head*width+d]*key[head*width+d] for d in range(width))*(scale if scale is not None else 1/math.sqrt(width)) for key in k]
   allowed=[j for j in range(len(k)) if blocked is None or not blocked[i][j]]
   peak=max(score[j] for j in allowed);denom=sum(math.exp(score[j]-peak) for j in allowed);prob=[math.exp(score[j]-peak)/denom if j in allowed else 0. for j in range(len(k))];weights.append(prob)
   for d in range(value_width):result[i][head*value_width+d]=sum(prob[j]*v[j][head*value_width+d] for j in range(len(v)))
  all_weights.append(weights)
 return result,all_weights

def scalar_module(operation,c,p,sources,blocked=None):
 import math
 def affine(rows,prefix):
  w=p[prefix+'weight'];b=p.get(prefix+'bias',[0.]*len(w))
  return [[sum(a*v for a,v in zip(row,w[i]))+b[i] for i in range(len(w))] for row in rows]
 def norm(rows,prefix):
  w=p[prefix+'weight'];b=p.get(prefix+'bias',[0.]*len(w));out=[]
  for row in rows:
   mean=sum(row)/len(row);variance=sum((x-mean)**2 for x in row)/len(row)
   out.append([(x-mean)/math.sqrt(variance+c['layer_norm_eps'])*w[i]+b[i] for i,x in enumerate(row)])
  return out
 def add(a,b):return [[x+y for x,y in zip(xrow,yrow)] for xrow,yrow in zip(a,b)]
 def mha(q,k,v,prefix,blocked=None):
  e=c.get('embed_dim',c.get('d_model'));heads=c.get('num_heads',c.get('nhead'))
  if prefix+'in_proj_weight' in p:
   weights=p[prefix+'in_proj_weight'];banks=[weights[i*e:(i+1)*e] for i in range(3)]
  else:banks=[p[prefix+role+'_proj_weight'] for role in ('q','k','v')]
  biases=p.get(prefix+'in_proj_bias',[0.]*(3*e))
  transformed=[]
  for part,rows in enumerate((q,k,v)):transformed.append([[sum(x*w for x,w in zip(row,banks[part][i]))+biases[part*e+i] for i in range(e)] for row in rows])
  q,k,v=transformed;extra=0
  if prefix+'bias_k' in p:k=k+[p[prefix+'bias_k'][0][0]];v=v+[p[prefix+'bias_v'][0][0]];extra+=1
  if operation=='multihead_attention' and c['add_zero_attn']:k=k+[[0.]*e];v=v+[[0.]*e];extra+=1
  if blocked is not None:blocked=[row+[0]*extra for row in blocked]
  mixed,weights=scalar_attention(q,k,v,heads,blocked);out=affine(mixed,prefix+'out_proj.')
  return out,weights
 def feed(rows,prefix):
  hidden=affine(rows,prefix+'linear1.')
  hidden=[[max(0.,x) if c['activation']=='relu' else x*.5*(1+math.erf(x/math.sqrt(2))) for x in row] for row in hidden]
  return affine(hidden,prefix+'linear2.')
 n=len(sources[0]);out=[];weight_rows=[]
 for observation in range(n):
  values=[x[observation] for x in sources]
  if operation=='scaled_dot_product_attention':
   q,k,v=values;heads=[]
   for a,b,d in zip(q,k,v):heads.append(scalar_attention(a,b,d,1,scale=c['scale'])[0])
   out.append(heads);continue
  if operation=='multihead_attention':
   y,w=mha(*values,'',None if blocked is None else blocked[observation]);out.append(y)
   weight_rows.append([[sum(w[h][i][j] for h in range(c['num_heads']))/c['num_heads'] for j in range(len(w[0][0]))] for i in range(len(w[0]))] if c['average_weights'] else w);continue
  def encode(rows,prefix,count):
   for i in range(count):
    layer=prefix+f'layers.{i}.'
    if c['norm_first']:
     z=norm(rows,layer+'norm1.');rows=add(rows,mha(z,z,z,layer+'self_attn.')[0]);rows=add(rows,feed(norm(rows,layer+'norm2.'),layer))
    else:rows=norm(add(rows,mha(rows,rows,rows,layer+'self_attn.')[0]),layer+'norm1.');rows=norm(add(rows,feed(rows,layer)),layer+'norm2.')
   return norm(rows,prefix+'norm.') if prefix+'norm.weight' in p else rows
  def decode(rows,memory,prefix,count):
   for i in range(count):
    layer=prefix+f'layers.{i}.'
    if c['norm_first']:
     z=norm(rows,layer+'norm1.');rows=add(rows,mha(z,z,z,layer+'self_attn.')[0]);rows=add(rows,mha(norm(rows,layer+'norm2.'),memory,memory,layer+'multihead_attn.')[0]);rows=add(rows,feed(norm(rows,layer+'norm3.'),layer))
    else:
     rows=norm(add(rows,mha(rows,rows,rows,layer+'self_attn.')[0]),layer+'norm1.');rows=norm(add(rows,mha(rows,memory,memory,layer+'multihead_attn.')[0]),layer+'norm2.');rows=norm(add(rows,feed(rows,layer)),layer+'norm3.')
   return norm(rows,prefix+'norm.') if prefix+'norm.weight' in p else rows
  if operation=='transformer_encoder':out.append(encode(values[0],'',c['num_layers']))
  elif operation=='transformer_decoder':out.append(decode(values[0],values[1],'',c['num_layers']))
  else:memory=encode(values[0],'encoder.',c['num_encoder_layers']);out.append(decode(values[1],memory,'decoder.',c['num_decoder_layers']))
 result={'out':out}
 if operation=='multihead_attention' and c['return_weights']:result['weights']=weight_rows
 return result

def scalar_leaves(value):
 if isinstance(value,list):
  for i,child in enumerate(value):
   if isinstance(child,list):yield from scalar_leaves(child)
   else:yield value,i

def scalar_objective(outputs):
 total=0.
 for role,values in outputs.items():
  for i,(row,j) in enumerate(scalar_leaves(values)):total+=row[j]*(.13+.031*i+(.07 if role=='weights' else 0))
 return total

@pytest.fixture(autouse=True)
def isolate_actual_device_policy(request):
 import torch
 device=getattr(getattr(request.node,'callspec',None),'params',{}).get('device')
 if device not in ('cpu','mps'):yield;return
 previous=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled()
 try:torch.use_deterministic_algorithms(device=='cpu');yield
 finally:torch.use_deterministic_algorithms(previous,warn_only=warn)

@pytest.mark.parametrize('operation',['multihead_attention','transformer','transformer_encoder','transformer_decoder','scaled_dot_product_attention'])
@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('norm_first',[True,False])
def test_scalar_outputs_and_all_floating_source_and_owned_parameter_gradients(operation,device,norm_first):
 import torch,copy,math
 from runner.attention_execution import create_attention_module,execute_attention
 if operation=='multihead_attention':config={'embed_dim':2,'num_heads':1,'kdim':3,'vdim':2,'average_weights':False,'add_bias_kv':True,'add_zero_attn':True};shapes=[(2,2),(3,3),(3,2)]
 elif operation=='scaled_dot_product_attention':config={'scale':.7};shapes=[(1,2,2),(1,3,2),(1,3,3)]
 else:
  config={'d_model':2,'nhead':1,'dim_feedforward':3,'norm_first':norm_first,'activation':'gelu','layer_norm_eps':.001}
  if operation!='transformer':config['final_norm']=True
  shapes=[(3,2)] if operation=='transformer_encoder' else [(2,2),(3,2)] if operation=='transformer_decoder' else [(3,2),(2,2)]
 c=helper().normalize_attention(operation,config);layer=create_attention_module(operation,c).to(device).eval();p={}
 with torch.no_grad():
  for offset,(name,value) in enumerate(layer.named_parameters()):
   numbers=torch.arange(value.numel(),dtype=torch.float32).reshape(value.shape);numbers=((numbers+offset)%7-3)*.037
   if '.norm' in '.'+name or name.startswith('norm.'):
    if name.endswith('weight'):numbers=numbers+1
   value.copy_(numbers.to(device));p[name]=value.detach().cpu().tolist()
 inputs=[]
 for offset,shape in enumerate(shapes):
  value=((torch.arange(2*math.prod(shape),dtype=torch.float32).reshape(2,*shape)+offset)%11-5)*.071
  inputs.append(value.to(device).requires_grad_())
 source_values=[x.detach().cpu().tolist() for x in inputs];expected=scalar_module(operation,c,p,source_values);actual=execute_attention(operation,c,inputs,layer)
 loss=0
 for role,values in actual.items():
  torch.testing.assert_close(values,torch.tensor(expected[role],device=device),rtol=3e-5,atol=4e-6)
  coefficients=torch.arange(values.numel(),device=device,dtype=torch.float32).reshape(values.shape)*.031+.13+(.07 if role=='weights' else 0);loss=loss+(values*coefficients).sum()
 gradients=torch.autograd.grad(loss,[*inputs,*layer.parameters()]);epsilon=1e-5
 for name,value,actual_gradient in [(None,value,grad) for value,grad in zip(inputs,gradients[:len(inputs)])]+[(name,value,grad) for (name,value),grad in zip(layer.named_parameters(),gradients[len(inputs):])]:
  target=source_values[next(i for i,x in enumerate(inputs) if x is value)] if name is None else p[name]
  measured=[]
  for row,j in scalar_leaves(target):
   original=row[j];row[j]=original+epsilon;plus=scalar_objective(scalar_module(operation,c,p,source_values));row[j]=original-epsilon;minus=scalar_objective(scalar_module(operation,c,p,source_values));row[j]=original;measured.append((plus-minus)/(2*epsilon))
  torch.testing.assert_close(actual_gradient.flatten(),torch.tensor(measured,device=device,dtype=torch.float32),rtol=8e-4,atol=2e-5)

@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('average',[True,False])
@pytest.mark.parametrize('bias',[True,False])
def test_multihead_masks_unequal_widths_augmentations_and_optional_weight_roles(device,average,bias):
 import torch
 from runner.attention_execution import create_attention_module,execute_attention
 op='multihead_attention';c=helper().normalize_attention(op,{'embed_dim':4,'num_heads':2,'kdim':3,'vdim':5,'bias':bias,'add_bias_kv':True,'add_zero_attn':True,'average_weights':average,'attention_mask_mode':'supplied','padding_mask_mode':'supplied'})
 layer=create_attention_module(op,c).to(device).eval();torch.manual_seed(173)
 x=[torch.randn(2,2,4,device=device,requires_grad=True),torch.randn(2,3,3,device=device,requires_grad=True),torch.randn(2,3,5,device=device,requires_grad=True)]
 mask=torch.tensor([[[0,1,0],[0,0,1]],[[1,0,0],[0,1,0]]],device=device,dtype=torch.int64);padding=torch.tensor([[0,1,0],[0,0,1]],device=device,dtype=torch.int64)
 p={k:v.detach().cpu().tolist() for k,v in layer.named_parameters()};combined=(mask.bool()|padding[:,None,:].bool()).cpu().tolist();expected=scalar_module(op,c,p,[v.detach().cpu().tolist() for v in x],combined)
 actual=execute_attention(op,c,[*x,mask,padding],layer)
 for role,value in actual.items():torch.testing.assert_close(value,torch.tensor(expected[role],device=device),rtol=3e-5,atol=3e-6)
 assert actual['weights'].shape==((2,2,5) if average else (2,2,2,5))
 loss=actual['out'].square().sum()+actual['weights'].square().sum();gradients=torch.autograd.grad(loss,[*x,*layer.parameters()]);assert all(torch.isfinite(g).all() for g in gradients)
 without=execute_attention(op,{**c,'return_weights':False},[*x,mask,padding],layer);assert set(without)=={'out'};torch.testing.assert_close(without['out'],actual['out'],rtol=3e-5,atol=3e-6)

@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('scale',[None,.7])
def test_sdpa_binary_mask_inversion_and_different_value_width_independent_scalar(device,scale):
 import torch
 from runner.attention_execution import create_attention_module,execute_attention
 c={'scale':scale,'mask_mode':'supplied'};layer=create_attention_module('scaled_dot_product_attention',c).to(device).eval();torch.manual_seed(41)
 sources=[torch.randn(2,2,2,3,device=device,requires_grad=True),torch.randn(2,2,3,3,device=device,requires_grad=True),torch.randn(2,2,3,4,device=device,requires_grad=True)]
 mask=torch.tensor([[[0,1,0],[1,0,0]],[[1,0,0],[0,1,0]]],device=device,dtype=torch.int64);actual=execute_attention('scaled_dot_product_attention',c,[*sources,mask],layer)['out'];q,k,v=[x.detach().cpu().tolist() for x in sources];blocked=mask.cpu().tolist()
 expected=[[scalar_attention(q[n][h],k[n][h],v[n][h],1,blocked[n],scale)[0] for h in range(2)] for n in range(2)]
 torch.testing.assert_close(actual,torch.tensor(expected,device=device),rtol=3e-5,atol=3e-6)
 gradients=torch.autograd.grad(actual.square().sum(),sources);assert all(torch.isfinite(g).all() for g in gradients)

@pytest.mark.parametrize('op',['multihead_attention','scaled_dot_product_attention'])
@pytest.mark.parametrize('problem',['allblocked','nonbinary','dtype','observation','shape','causal_length','finite','resource'])
def test_masks_and_allocation_reject_before_native_scoring(op,problem,monkeypatch):
 import torch
 from runner.attention_execution import create_attention_module,execute_attention
 c={'attention_mask_mode':'supplied','padding_mask_mode':'supplied','add_bias_kv':True,'add_zero_attn':True} if op=='multihead_attention' else {'mask_mode':'supplied'}
 layer=create_attention_module(op,c);shapes=[(1,2,4),(1,3,4),(1,3,4)] if op=='multihead_attention' else [(1,2,2,2),(1,2,3,2),(1,2,3,3)]
 x=[torch.ones(shape) for shape in shapes];mask=torch.zeros(1,2,3,dtype=torch.int64);padding=torch.zeros(1,3,dtype=torch.int64)
 if problem=='allblocked':mask[0,0]=torch.tensor([1,0,1]);padding[:,1]=1
 if problem=='nonbinary':mask[0,0,0]=2
 if problem=='dtype':mask=mask.float()
 if problem=='observation':mask=mask.expand(2,2,3)
 if problem=='shape':mask=mask[:,:1]
 if problem=='causal_length':c={'attention_mask_mode':'causal'} if op=='multihead_attention' else {'mask_mode':'causal'}
 if problem=='finite':x[0].flatten()[0]=float('nan')
 if problem=='resource':x=[v.expand(10_000_000,*v.shape[1:]) for v in x];mask=mask.expand(10_000_000,2,3);padding=padding.expand(10_000_000,3)
 if op=='scaled_dot_product_attention' and problem=='allblocked':mask[0,0,:]=1
 monkeypatch.setattr(layer,'forward',lambda *a,**k:pytest.fail('Invalid mask/configuration reached native score allocation'))
 operands=x if problem=='causal_length' else x+[mask]+([padding] if op=='multihead_attention' else [])
 with pytest.raises(ValueError):execute_attention(op,c,operands,layer)

@pytest.mark.parametrize('operation',['multihead_attention','transformer','transformer_encoder','transformer_decoder','scaled_dot_product_attention'])
@pytest.mark.parametrize('device',['cpu','mps'])
def test_parent_train_eval_dropout_and_cpu_rng_state_restart(operation,device):
 import torch
 from runner.attention_execution import create_attention_module,execute_attention
 c={'dropout':.5};layer=create_attention_module(operation,c).to(device)
 if operation in ('multihead_attention','scaled_dot_product_attention'):
  shapes=[(2,2,4),(2,3,4),(2,3,4)] if operation=='multihead_attention' else [(2,2,2,3),(2,2,3,3),(2,2,3,4)]
 elif operation=='transformer_encoder':shapes=[(2,3,4)]
 else:shapes=[(2,2,4),(2,3,4)]
 sources=[torch.arange(__import__('math').prod(shape),device=device,dtype=torch.float32).reshape(shape)*.03+.1 for shape in shapes]
 layer.eval();first=execute_attention(operation,c,sources,layer);second=execute_attention(operation,c,sources,layer)
 for role in first:torch.testing.assert_close(first[role],second[role],rtol=0,atol=0)
 layer.train();torch.manual_seed(911);state=torch.get_rng_state() if device=='cpu' else torch.mps.get_rng_state();a=execute_attention(operation,c,sources,layer)
 if device=='cpu':torch.set_rng_state(state)
 else:torch.mps.set_rng_state(state)
 b=execute_attention(operation,c,sources,layer)
 for role in a:torch.testing.assert_close(a[role],b[role],rtol=0 if device=='cpu' else 3e-5,atol=0 if device=='cpu' else 3e-6)
 assert not torch.equal(a['out'],first['out'])
 if operation=='transformer':assert layer.encoder.enable_nested_tensor is False and layer.encoder.use_nested_tensor is False
 if operation=='transformer_encoder':assert layer.enable_nested_tensor is False

@pytest.mark.parametrize('operation',['transformer_encoder','transformer_decoder'])
def test_cloned_prototype_parameters_start_equal_but_are_distinct_owned_state(operation):
 import torch
 from runner.attention_execution import create_attention_module
 layer=create_attention_module(operation,{'num_layers':2})
 first=dict(layer.layers[0].named_parameters());second=dict(layer.layers[1].named_parameters())
 assert set(first)==set(second)
 for name,value in first.items():
  assert value.data_ptr()!=second[name].data_ptr();torch.testing.assert_close(value,second[name],rtol=0,atol=0)

@pytest.mark.parametrize('operation',['multihead_attention','transformer','transformer_encoder','transformer_decoder'])
@pytest.mark.parametrize('problem',['extra','missing','shape','dtype','sparse','nonfinite'])
def test_independent_strict_state_admission_refuses_before_copy(operation,problem):
 import torch
 from runner.attention_execution import create_attention_module,validate_attention_state
 layer=create_attention_module(operation,{});expected={k:v.clone() for k,v in layer.state_dict().items()};changed={k:v.clone() for k,v in expected.items()}
 if operation=='scaled_dot_product_attention':
  changed['invented_parameter']=torch.ones(1)
 else:
  first=next(iter(changed))
  if problem=='extra':changed['invented_parameter']=torch.ones(1)
  elif problem=='missing':del changed[first]
  elif problem=='shape':changed[first]=torch.ones(1)
  elif problem=='dtype':changed[first]=changed[first].double()
  elif problem=='sparse':changed[first]=changed[first].to_sparse()
  else:changed[first].flatten()[0]=float('nan')
 with pytest.raises(ValueError):validate_attention_state(changed,expected)
 for key,value in layer.state_dict().items():assert torch.equal(value,expected[key])
 if operation=='scaled_dot_product_attention':
  assert expected=={}
  with pytest.raises(ValueError):layer.load_state_dict(changed,strict=False)


def test_sdpa_empty_state_roundtrip_has_no_invented_learned_parameter():
 import torch
 from runner.attention_execution import create_attention_module,validate_attention_state
 layer=create_attention_module('scaled_dot_product_attention',{})
 assert layer.state_dict()=={} and list(layer.parameters())==[]
 assert layer.load_state_dict({}).missing_keys==[]
 for value in [{'invented_parameter':torch.ones(1)},None]:
  with pytest.raises(ValueError):validate_attention_state(value,{})
  with pytest.raises(ValueError):layer.load_state_dict(value,strict=False)

@pytest.mark.parametrize('requires_grad',[False,True])
def test_mps_sdpa_dropout_public_path_seeded_mask_scalar_and_gradients(requires_grad):
 import copy,math,torch
 from runner.attention_execution import create_attention_module,execute_attention
 layer=create_attention_module('scaled_dot_product_attention',{'dropout':.5,'scale':.7,'mask_mode':'supplied'}).to('mps').train()
 shapes=[(1,2,2,2),(1,2,3,2),(1,2,3,3)]
 sources=[(torch.arange(math.prod(s),device='mps',dtype=torch.float32).reshape(s)*.027+.13).requires_grad_(requires_grad) for s in shapes]
 blocked=torch.tensor([[[0,1,0],[1,0,0]]],device='mps',dtype=torch.int64)
 torch.manual_seed(173);state=torch.mps.get_rng_state()
 multiplier=torch.nn.functional.dropout(torch.ones(1,2,2,3,device='mps'),p=.5,training=True).cpu().tolist()
 torch.mps.set_rng_state(state)
 actual=execute_attention('scaled_dot_product_attention',{'dropout':.5,'scale':.7,'mask_mode':'supplied'},[*sources,blocked],layer)['out']
 arrays=[s.detach().cpu().tolist() for s in sources]
 def oracle(xs):
  q,k,v=xs;out=[]
  for n in range(1):
   heads=[]
   for h in range(2):
    _,weights=scalar_attention(q[n][h],k[n][h],v[n][h],1,blocked.cpu().tolist()[n],.7)
    heads.append([[sum(weights[0][i][j]*multiplier[n][h][i][j]*v[n][h][j][d] for j in range(3)) for d in range(3)] for i in range(2)])
   out.append(heads)
  return out
 expected=oracle(arrays);torch.testing.assert_close(actual,torch.tensor(expected,device='mps'),rtol=3e-5,atol=3e-6)
 assert actual.requires_grad is requires_grad
 if requires_grad:
  upstream=torch.arange(actual.numel(),device='mps',dtype=torch.float32).reshape(actual.shape)*.031+.13
  actual_grads=torch.autograd.grad((actual*upstream).sum(),sources);coeff=upstream.cpu().tolist()
  def objective(xs):return sum(oracle(xs)[n][h][i][d]*coeff[n][h][i][d] for n in range(1) for h in range(2) for i in range(2) for d in range(3))
  for operand,gradient in enumerate(actual_grads):
   expected_grad=[]
   for leaf,index in scalar_leaves(arrays[operand]):
    original=leaf[index];leaf[index]=original+1e-5;positive=objective(arrays);leaf[index]=original-1e-5;negative=objective(arrays);leaf[index]=original
    expected_grad.append((positive-negative)/2e-5)
   torch.testing.assert_close(gradient.flatten(),torch.tensor(expected_grad,device='mps'),rtol=8e-4,atol=2e-5)

def attention_classifier(operation,masked=False,weights=False,extra=None):
 nodes=[{'id':'input','type':'input','config':{'features':2}},{'id':'learn','type':'linear','config':{'in_features':2,'out_features':8,'bias':True}},{'id':'array','type':'reshape','config':{'shape':[2,2,2] if operation=='scaled_dot_product_attention' else [2,4]}}];edges=[]
 def edge(a,b,port='in',source='out'):edges.append({'id':str(len(edges)),'source':a,'sourcePort':source,'target':b,'targetPort':port})
 edge('input','learn');edge('learn','array');c=dict(extra or {})
 if masked:
  if operation=='multihead_attention':c.update(attention_mask_mode='supplied',padding_mask_mode='supplied')
  elif operation=='transformer_encoder':c.update(mask_mode='supplied',padding_mask_mode='supplied')
  elif operation=='scaled_dot_product_attention':c['mask_mode']='supplied'
  elif operation=='transformer_decoder':c.update(target_mask_mode='supplied',memory_mask_mode='supplied',target_padding_mode='supplied',memory_padding_mode='supplied')
  else:c.update(source_mask_mode='supplied',target_mask_mode='supplied',memory_mask_mode='supplied',source_padding_mode='supplied',target_padding_mode='supplied',memory_padding_mode='supplied')
  nodes.append({'id':'mask','type':'attention_mask','config':{'padding_keys':[1]}});edge('array','mask','query');edge('array','mask','key')
 nodes.append({'id':'attention','type':operation,'config':c})
 from runner.attention_configuration import configured_input_ports,FLOAT_PORTS
 for port in configured_input_ports(operation,c):edge('array' if port in FLOAT_PORTS[operation] else 'mask','attention',port,'out' if port in FLOAT_PORTS[operation] else 'padding' if 'padding' in port else 'mask')
 width=4 if operation=='multihead_attention' and weights else 8
 nodes.extend([{'id':'flat','type':'flatten','config':{'start_dim':1,'end_dim':-1}},{'id':'head','type':'linear','config':{'in_features':width,'out_features':2,'bias':True}},{'id':'output','type':'output','config':{}}]);edge('attention','flat',source='weights' if weights else 'out');edge('flat','head');edge('head','output')
 return {'schemaVersion':2,'nodes':nodes,'edges':edges}

@pytest.mark.parametrize('operation',OPS[:-1])
@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('masked',[False,True])
def test_real_graph_discovery_and_portable_named_output_gradients(operation,device,masked):
 import torch
 from runner.engine import GraphModel,export_source,capabilities
 model=GraphModel(attention_classifier(operation,masked),bundled_dataset=False).to(device).eval()
 x=torch.tensor([[.4,-.2],[-.3,.1]],device=device,requires_grad=True);y=model(x);code=export_source(model)
 assert 'runner.' not in code and 'kwipml' not in code.lower().replace('exported kwipml','exported')
 ns={};exec(code,ns);exported=ns['Model']().to(device).eval();exported.load_state_dict(model.state_dict(),strict=True);z=exported(x)
 torch.testing.assert_close(z,y,rtol=0,atol=0)
 a=torch.autograd.grad(y.square().sum(),[x,*model.parameters()]);b=torch.autograd.grad(z.square().sum(),[x,*exported.parameters()])
 for left,right in zip(a,b):torch.testing.assert_close(left,right,rtol=0,atol=0)
 assert capabilities()['componentCapabilities'][operation]==helper().capability(operation)

@pytest.mark.parametrize('operation',OPS[:-1])
def test_checkpoint_attention_order_and_stochastic_policy_without_registry_allocation(operation):
 import torch
 from runner.checkpoint_lifecycle import contract
 graph=attention_classifier(operation,True,extra={'dropout':.3});order=[n['id'] for n in graph['nodes']]
 c=contract(graph,[torch.ones(4,2)],{'sourceHash':'source','splitHash':'split'},42,2,.03,'cpu','v1',order)
 assert c['schedule']['attentionNodeOrder']==['attention']
 assert c['schedule']['parameterNodeOrder']==(['learn','head'] if operation=='scaled_dot_product_attention' else ['learn','attention','head'])
 assert c['schedule']['stochasticNodeOrder']==['attention']
 assert c['guarantee']=='epochBoundaryExactCPU'

@pytest.mark.parametrize('operation',OPS[:-1])
def test_real_cpu_checkpoint_exact_dropout_resume_and_weights_finetune(operation,tmp_path):
 import torch,json
 from runner.engine import run
 from runner.tests.test_checkpoint_lifecycle import assert_nested_equal
 graph=attention_classifier(operation,True,extra={'dropout':.3});options={'device':'cpu','sample_limit':32,'batch_size':8,'emit':lambda e:None}
 for name,epochs,parent in [('full',2,None),('first',1,None),('resume',2,tmp_path/'first/checkpoint.pt')]:run(graph,tmp_path/name,epochs=epochs,resume_checkpoint=parent,**options)
 full=torch.load(tmp_path/'full/checkpoint.pt',weights_only=True);resumed=torch.load(tmp_path/'resume/checkpoint.pt',weights_only=True)
 for key in ('modelState','optimizerState','rngState'):assert_nested_equal(full[key],resumed[key])
 assert (tmp_path/'full/metrics.json').read_bytes()==(tmp_path/'resume/metrics.json').read_bytes()
 assert full['contract']['schedule']['attentionNodeOrder']==['attention'] and full['contract']['schedule']['stochasticNodeOrder']==['attention']
 run(graph,tmp_path/'fine',epochs=1,finetune_checkpoint=tmp_path/'first/checkpoint.pt',seed=173,**options)
 assert json.loads((tmp_path/'fine/manifest.json').read_text())['lineage']['mode']=='weightsOnlyFineTune'

@pytest.mark.parametrize('operation',OPS[:-1])
def test_real_mps_training_dropout_weights_only_and_resume_refusal(operation,tmp_path):
 import json
 from runner.engine import run
 graph=attention_classifier(operation,True,extra={'dropout':.3});events=[];options={'device':'mps','sample_limit':32,'batch_size':8,'emit':events.append}
 run(graph,tmp_path/'first',epochs=1,sample_capture={'schemaVersion':1,'activationValues':True,'predictionRows':True},**options)
 manifest=json.loads((tmp_path/'first/manifest.json').read_text());assert manifest['runtime']['device']=='mps' and manifest['checkpoint']['contract']['guarantee']=='weightsOnly'
 if operation=='scaled_dot_product_attention':
  expected=[{'nodeID':'attention','operator':operation,'computeDevice':'mps','policy':'deviceLocalPublicOps','phase':'training'}]
  assert manifest['runtime']['attentionExecution']==expected
  assert next(event for event in events if event['event']=='started')['attentionExecution']==expected
 else:assert 'attentionExecution' not in manifest['runtime']
 run(graph,tmp_path/'fine',epochs=1,finetune_checkpoint=tmp_path/'first/checkpoint.pt',seed=173,**options)
 with pytest.raises(ValueError,match='Exact resume'):run(graph,tmp_path/'resume',epochs=2,resume_checkpoint=tmp_path/'first/checkpoint.pt',**options)

@pytest.mark.parametrize('device',['cpu','mps'])
def test_named_mha_weights_can_drive_classifier_with_collision_safe_keys(device):
 import torch
 from runner.engine import GraphModel,export_source
 graph=attention_classifier('multihead_attention',True,weights=True)
 # A component called 'training' conflicts with nn.Module.training, so registration must escape it.
 for node in graph['nodes']:
  if node['id']=='attention':node['id']='training'
 for edge in graph['edges']:
  for key in ('source','target'):
   if edge[key]=='attention':edge[key]='training'
 model=GraphModel(graph).to(device).eval();assert model.layer_keys['training']!='training'
 x=torch.tensor([[.3,-.2],[.1,.4]],device=device);y=model(x);scope={};exec(export_source(model),scope);portable=scope['Model']().to(device).eval();portable.load_state_dict(model.state_dict());torch.testing.assert_close(portable(x),y,rtol=0,atol=0)

@pytest.mark.parametrize('operation',OPS[:-1])
@pytest.mark.parametrize('device',['cpu','mps'])
def test_actual_portable_subprocess_training_dropout_seed_replay(operation,device,tmp_path):
 import os,sys,subprocess,torch
 from runner.engine import GraphModel,export_source
 model=GraphModel(attention_classifier(operation,True,extra={'dropout':.3})).to(device).train()
 x=torch.tensor([[.4,-.2],[-.3,.1]],device=device,requires_grad=True)
 torch.manual_seed(911);state=torch.get_rng_state() if device=='cpu' else torch.mps.get_rng_state()
 y=model(x);grads=torch.autograd.grad(y.square().sum(),[x,*model.parameters()])
 (tmp_path/'model.py').write_text(export_source(model));torch.save({'state':model.state_dict(),'rng':state,'input':x.detach().cpu(),'output':y.detach().cpu(),'gradients':[g.cpu() for g in grads]},tmp_path/'proof.pt')
 script='''import torch
from model import Model
p=torch.load('proof.pt',weights_only=True,map_location='cpu');device=DEVICE
model=Model().to(device).train();model.load_state_dict(p['state']);x=p['input'].to(device).requires_grad_()
if device=='cpu':torch.set_rng_state(p['rng'])
else:torch.mps.set_rng_state(p['rng'])
y=model(x);grads=torch.autograd.grad(y.square().sum(),[x,*model.parameters()])
for a,b in [(y.cpu(),p['output']),*zip([g.cpu() for g in grads],p['gradients'])]:torch.testing.assert_close(a,b,rtol=0 if device=='cpu' else 3e-5,atol=0 if device=='cpu' else 3e-6)
assert model.training
model.eval();assert torch.equal(model(x),model(x))
print('portable independent-process output/gradient/dropout/eval verified')
'''.replace('DEVICE',repr(device))
 (tmp_path/'verify.py').write_text(script)
 result=subprocess.run([sys.executable,'verify.py'],cwd=tmp_path,env={k:v for k,v in os.environ.items() if k!='PYTHONPATH'},capture_output=True,text=True,timeout=30)
 assert result.returncode==0,result.stdout+result.stderr
 assert 'portable independent-process' in result.stdout

@pytest.mark.parametrize('operation',OPS[:-1])
@pytest.mark.parametrize('portable',[False,True])
def test_exact_all_composed_state_types_and_keys_refuse_before_any_copy(operation,portable):
 import torch
 from runner.engine import GraphModel,export_source
 model=GraphModel(attention_classifier(operation));expected={k:v.clone() for k,v in model.state_dict().items()}
 if portable:scope={};exec(export_source(model),scope);model=scope['Model']()
 for problem in ('extra','missing','shape','dtype','nonfinite','sparse'):
  changed={k:v.clone() for k,v in expected.items()};key=next(iter(changed))
  if problem=='extra':changed['invented_attention_state']=torch.ones(1)
  elif problem=='missing':del changed[key]
  elif problem=='shape':changed[key]=torch.ones(1)
  elif problem=='dtype':changed[key]=changed[key].double()
  elif problem=='nonfinite':changed[key].flatten()[0]=float('nan')
  else:changed[key]=changed[key].to_sparse()
  before={k:v.clone() for k,v in model.state_dict().items()}
  with pytest.raises(ValueError):model.load_state_dict(changed,strict=False)
  for name,value in model.state_dict().items():assert torch.equal(value,before[name])

@pytest.mark.parametrize('operation',OPS)
def test_actual_published_descriptor_normalizer_resolver_contract_identity(operation,monkeypatch):
 import json
 from pathlib import Path
 from runner.engine import REGISTRY,capabilities
 monkeypatch.syspath_prepend(str(Path('runner').resolve()))
 from runner.attention_adapters_generated import CAPABILITIES,NORMALIZERS,CONFIGURED_INPUTS
 definition=next(c for c in json.loads(REGISTRY.read_text())['components'] if c['id']==operation)
 authored=next(c for c in json.loads(Path('components/attention/bundle.json').read_text())['components'] if c['definition']['id']==operation)
 assert definition['id']==authored['definition']['id']
 assert authored['runtimeContract']==helper().capability(operation)==CAPABILITIES[operation]==capabilities()['componentCapabilities'][operation]
 assert NORMALIZERS[operation]({})==helper().normalize_attention(operation,{})
 assert CONFIGURED_INPUTS[operation]({})==helper().configured_input_ports(operation,{})

@pytest.mark.parametrize('operation',OPS[:-1])
def test_attention_named_preview_cache_and_semantic_defaults(operation,tmp_path):
 from runner.preview import preview
 from runner.graph_semantics import semantic_graph
 graph=attention_classifier(operation,True);before=semantic_graph(graph)
 node=next(n for n in graph['nodes'] if n['type']==operation);node['config']=helper().normalize_attention(operation,node['config']);assert semantic_graph(graph)==before
 first=preview(graph,tmp_path/'before',device='cpu');second=preview(graph,tmp_path/'after',device='cpu',resume_from='flat',cache=first['cachePath'])
 assert first['previews']['output']==second['previews']['output']
 assert first['previews']['mask']['outputs']['mask']['dtype']=='int64'
 if operation=='multihead_attention':assert 'weights' in first['previews']['attention']['outputs']

def test_utility_head_count_refused_before_mask_allocation(monkeypatch):
 import torch
 from runner.attention_execution import execute_attention
 query=torch.ones(1,33,2,2);key=torch.ones(1,33,3,2)
 def allocated(*args,**kwargs):pytest.fail('Unsupported comparison-group count allocated a mask')
 monkeypatch.setattr(torch,'zeros',allocated)
 with pytest.raises(ValueError,match='comparison'):execute_attention('attention_mask',{},[query,key],None)
