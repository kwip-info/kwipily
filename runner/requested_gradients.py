"""Bounded, recorded final-snapshot derivatives, separate from no-grad inference."""
import copy
import hashlib
import math
import re
from contextlib import contextmanager
import torch
if __package__:
 from .gradient_policy import validate_request,validate_recipe,validate_capture,encoded,MAX_ELEMENTS,exact
 from .trainability import native_owners
 from .output_routing import OutputAddress,select_output
else:
 from gradient_policy import validate_request,validate_recipe,validate_capture,encoded,MAX_ELEMENTS,exact
 from trainability import native_owners
 from output_routing import OutputAddress,select_output

@contextmanager
def _isolated(model):
 modes=[(m,m.training) for m in model.modules()]
 params=[(p,p.detach().clone(),p.grad,None if p.grad is None else p.grad.detach().clone()) for p in model.parameters()]
 buffers=list(model.buffers())
 if sum(b.numel() for b in buffers)>100000000:raise ValueError('Analysis buffer snapshot exceeds existing tensor budget.')
 saved_buffers=[b.detach().clone() for b in buffers]
 cpu=torch.get_rng_state();mps=torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
 try:
  model.eval();yield
 finally:
  with torch.no_grad():
   for p,value,grad,grad_value in params:
    if not torch.equal(p,value):p.copy_(value)
    p.grad=grad
    if grad is not None:grad.copy_(grad_value)
   for b,value in zip(buffers,saved_buffers):
    if not torch.equal(b,value):b.copy_(value)
  for module,mode in modes:module.training=mode
  torch.set_rng_state(cpu)
  if mps is not None:torch.mps.set_rng_state(mps)

def _ordinary_inputs(inputs,device):
 if device not in ('cpu','mps') or device=='mps' and not torch.backends.mps.is_available():raise ValueError('Requested native derivative device is unavailable.')
 if not isinstance(inputs,torch.Tensor) or inputs.layout!=torch.strided or inputs.dtype not in (torch.float32,torch.int64) or not 2<=inputs.ndim<=8 or not 1<=len(inputs)<=32 or inputs.numel()>100000000 or any(d<=0 or d>4096 for d in inputs.shape[1:]):raise ValueError('Analysis requires bounded dense observation-first float32/int64 inputs.')
 if inputs.device.type!=device or not torch.isfinite(inputs).all().item():raise ValueError('Analysis inputs must be finite on the selected device; no implicit fallback.')

def _ordinary_targets(model,inputs,request):
 inventory=native_owners(model);owners={(m['node'],m['parameter']):(m,p) for m,p in inventory}
 resolved=[];seen=set();elements=0
 for target in request['targets']:
  if target['kind']=='input':
   tensor=inputs;name=target['node']+'.'+target['port'];key=('input',id(inputs))
   metadata=dict(name=name,shape=list(inputs.shape),dtype=str(inputs.dtype).removeprefix('torch.'))
  else:
   key=(target['node'],target['parameter'])
   if key not in owners:raise ValueError('Derivative target is not an actual native parameter owner.')
   m,tensor=owners[key];key=('parameter',id(tensor));metadata={k:m[k] for k in ('name','shape','dtype')}
  if key in seen:raise ValueError('Derivative targets alias the same actual tensor.')
  seen.add(key);elements+=tensor.numel();resolved.append((target,metadata,tensor))
 if elements>MAX_ELEMENTS:raise ValueError('Derivative targets exceed65536 returned elements before execution.')
 return resolved,elements

def _ordinary_analyze(model,inputs,request,*,device,recording=True,check_cancelled=lambda:None):
 """Unqualified host result: caller supplies inputs; no archived source claim."""
 request=validate_request(request);_inputs(inputs,device)
 if type(recording) is not bool:raise ValueError('Recording context must be explicit boolean.')
 if len(inputs)!=request['row_count']:raise ValueError('Analysis input rows must match exact selected row count.')
 resolved,_=_targets(model,inputs,request)
 if any(p.device.type!=device for _,p in native_owners(model)):raise ValueError('Model and analysis device differ; no implicit fallback.')
 check_cancelled()
 with _isolated(model),torch.enable_grad() if recording else torch.no_grad():
  x=inputs.detach().clone()
  if x.dtype==torch.float32 and recording:x.requires_grad_(True)
  try:_,values=model(x,inspect=True)
  except TypeError as error:raise ValueError('Analysis requires genuine inspected graph outputs.') from error
  selected=select_output(values,OutputAddress(**request['output']))
  if not isinstance(selected,torch.Tensor) or selected.layout!=torch.strided or selected.dtype!=torch.float32 or selected.device.type!=device or selected.ndim<1 or len(selected)!=len(x) or selected.numel()>100000000 or not torch.isfinite(selected).all().item():raise ValueError('Selected derivative output must be a bounded finite observation-first float32 tensor.')
  upstream=request['upstream']
  if upstream is not None and upstream['shape']!=list(selected.shape):raise ValueError('Upstream must exactly match actual selected output shape; broadcasting is forbidden.')
  eligible=[];results=[]
  for target,meta,tensor in resolved:
   cause=None
   if target['kind']=='input':
    if select_output(values,OutputAddress(target['node'],target['port'])) is not x:raise ValueError('Input target must be the genuine model input endpoint, not an intermediate.')
    tensor=x
    if x.dtype==torch.int64:cause='integerInput'
   elif not tensor.requires_grad:cause='frozenParameter'
   if cause is None and not recording:cause='recordingDisabled'
   results.append(dict(target=copy.deepcopy(target),**meta,status='unavailable',cause=cause))
   if cause is None:eligible.append((len(results)-1,tensor))
  if eligible:
   if not selected.requires_grad:
    gradients=[None]*len(eligible)
   else:
    check_cancelled()
    if request['reduction']=='upstream':
     objective=selected;gradient_output=torch.tensor(upstream['values'],device=device,dtype=torch.float32).reshape(selected.shape)
    else:
     objective=selected.sum() if request['reduction']=='sum' else selected.mean();gradient_output=None
    if not torch.isfinite(objective).all().item():raise ValueError('Derivative scalar became nonfinite.')
    try:gradients=torch.autograd.grad(objective,[tensor for _,tensor in eligible],grad_outputs=gradient_output,allow_unused=True,create_graph=False,retain_graph=False)
    except (RuntimeError,NotImplementedError) as error:raise ValueError('Native first-order derivative failed on '+device+'; no fallback: '+str(error)) from error
   for (index,_),gradient in zip(eligible,gradients):
    if gradient is None:results[index]['cause']='disconnected'
    elif gradient.layout!=torch.strided:results[index]['cause']='sparseGradient'
    elif gradient.dtype!=torch.float32 or gradient.device.type!=device or not torch.isfinite(gradient).all().item():raise ValueError('Derivative result must be finite dense float32 on the selected device.')
    else:
     results[index].pop('cause');results[index]['status']='observed';results[index]['values']=gradient.detach().cpu().flatten().tolist()
  if request['unavailable']=='error' and any(r['status']=='unavailable' for r in results):raise ValueError('Requested derivative is unavailable: '+','.join(r['cause'] for r in results if r['status']=='unavailable'))
  check_cancelled()
  return dict(output=dict(**request['output'],shape=list(selected.shape),dtype='float32'),targets=results)


def _bag_contract(model):
 if not any(n.get('type')=='bag_input' for n in getattr(model,'nodes',{}).values()):return None
 if __package__:from .product_execution import ProductExecutionAdapter
 else:from product_execution import ProductExecutionAdapter
 return ProductExecutionAdapter(model).input_contract

def _bag_coordinates(model,inputs,device,*,bounded=True):
 contract=_bag_contract(model)
 if contract is None:raise ValueError('Actual bag source contract is required.')
 if device not in ('cpu','mps') or device=='mps' and not torch.backends.mps.is_available():raise ValueError('Requested native derivative device is unavailable.')
 if __package__:from .bag_source import execute_bag_source
 else:from bag_source import execute_bag_source
 values=execute_bag_source(inputs,contract['configuration'])
 if any(t.device.type!=device for t in values.values()):raise ValueError('Analysis inputs must be on the selected device; no implicit fallback.')
 rows=inputs.indices.shape[0] if inputs.layout=='fixed' else inputs.offsets.numel()-int(inputs.include_last_offset)
 if bounded and not 1<=rows<=32:raise ValueError('Analysis requires1..32 whole bag observations.')
 return contract,values,rows

def _inputs(inputs,device):
 return _ordinary_inputs(inputs,device)

def _targets(model,inputs,request):
 if _bag_contract(model) is None:return _ordinary_targets(model,inputs,request)
 contract,values,_=_bag_coordinates(model,inputs,inputs.indices.device.type)
 owners={(m['node'],m['parameter']):(m,p) for m,p in native_owners(model)}
 resolved=[];seen=set();elements=0
 for target in request['targets']:
  if target['kind']=='input':
   if target['node']!=contract['sourceID'] or target['port'] not in values:raise ValueError('Input target must name an actual bag source role.')
   tensor=values[target['port']];meta=dict(name=target['node']+'.'+target['port'],shape=list(tensor.shape),dtype=str(tensor.dtype).removeprefix('torch.'))
  else:
   key=(target['node'],target['parameter'])
   if key not in owners:raise ValueError('Derivative target is not an actual native parameter owner.')
   metadata,tensor=owners[key];meta={k:metadata[k] for k in ('name','shape','dtype')}
  if id(tensor) in seen:raise ValueError('Derivative targets alias the same actual tensor.')
  seen.add(id(tensor));elements+=tensor.numel();resolved.append((target,meta,tensor))
 if elements>MAX_ELEMENTS:raise ValueError('Derivative targets exceed65536 returned elements before execution.')
 return resolved,elements

def _bag_analyze(model,inputs,request,*,device,recording=True,check_cancelled=lambda:None):
 """Unqualified host result: caller supplies inputs; no archived source claim."""
 request=validate_request(request);contract,coordinates,rows=_bag_coordinates(model,inputs,device)
 if request['output']['node']==contract['sourceID']:raise ValueError('Choose an ordinary floating model output, not bag coordinates.')
 if type(recording) is not bool:raise ValueError('Recording context must be explicit boolean.')
 if rows!=request['row_count']:raise ValueError('Analysis input rows must match exact selected row count.')
 resolved,_=_targets(model,inputs,request)
 if any(p.device.type!=device for _,p in native_owners(model)):raise ValueError('Model and analysis device differ; no implicit fallback.')
 check_cancelled()
 with _isolated(model),torch.enable_grad() if recording else torch.no_grad():
  clones={role:t.detach().clone() for role,t in coordinates.items()}
  if 'per_sample_weights' in clones and recording:clones['per_sample_weights'].requires_grad_(True)
  x=type(inputs)(inputs.layout,clones['indices'],clones.get('offsets'),clones.get('per_sample_weights'),inputs.include_last_offset)
  try:_,values=model(x,inspect=True)
  except TypeError as error:raise ValueError('Analysis requires genuine inspected graph outputs.') from error
  selected=select_output(values,OutputAddress(**request['output']))
  if not isinstance(selected,torch.Tensor) or selected.layout!=torch.strided or selected.dtype!=torch.float32 or selected.device.type!=device or selected.ndim<1 or len(selected)!=rows or selected.numel()>100000000 or not torch.isfinite(selected).all().item():raise ValueError('Selected derivative output must be a bounded finite observation-first float32 tensor.')
  upstream=request['upstream']
  if upstream is not None and upstream['shape']!=list(selected.shape):raise ValueError('Upstream must exactly match actual selected output shape; broadcasting is forbidden.')
  eligible=[];results=[]
  for target,meta,tensor in resolved:
   cause=None
   if target['kind']=='input':
    if select_output(values,OutputAddress(target['node'],target['port'])) is not clones[target['port']]:raise ValueError('Input target must be the genuine model input endpoint, not an intermediate.')
    tensor=clones[target['port']]
    if tensor.dtype==torch.int64:cause='integerInput'
   elif not tensor.requires_grad:cause='frozenParameter'
   if cause is None and not recording:cause='recordingDisabled'
   results.append(dict(target=copy.deepcopy(target),**meta,status='unavailable',cause=cause))
   if cause is None:eligible.append((len(results)-1,tensor))
  if eligible:
   if not selected.requires_grad:
    gradients=[None]*len(eligible)
   else:
    check_cancelled()
    if request['reduction']=='upstream':
     objective=selected;gradient_output=torch.tensor(upstream['values'],device=device,dtype=torch.float32).reshape(selected.shape)
    else:
     objective=selected.sum() if request['reduction']=='sum' else selected.mean();gradient_output=None
    if not torch.isfinite(objective).all().item():raise ValueError('Derivative scalar became nonfinite.')
    try:gradients=torch.autograd.grad(objective,[tensor for _,tensor in eligible],grad_outputs=gradient_output,allow_unused=True,create_graph=False,retain_graph=False)
    except (RuntimeError,NotImplementedError) as error:raise ValueError('Native first-order derivative failed on '+device+'; no fallback: '+str(error)) from error
   for (index,_),gradient in zip(eligible,gradients):
    if gradient is None:results[index]['cause']='disconnected'
    elif gradient.layout!=torch.strided:results[index]['cause']='sparseGradient'
    elif gradient.dtype!=torch.float32 or gradient.device.type!=device or not torch.isfinite(gradient).all().item():raise ValueError('Derivative result must be finite dense float32 on the selected device.')
    else:
     results[index].pop('cause');results[index]['status']='observed';results[index]['values']=gradient.detach().cpu().flatten().tolist()
  if request['unavailable']=='error' and any(r['status']=='unavailable' for r in results):raise ValueError('Requested derivative is unavailable: '+','.join(r['cause'] for r in results if r['status']=='unavailable'))
  check_cancelled()
  return dict(output=dict(**request['output'],shape=list(selected.shape),dtype='float32'),targets=results)

def analyze(model,inputs,request,*,device,recording=True,check_cancelled=lambda:None):
 function=_bag_analyze if _bag_contract(model) is not None else _ordinary_analyze
 return function(model,inputs,request,device=device,recording=recording,check_cancelled=check_cancelled)

def preflight_requested_gradients(model,prepared_inputs,policy,*,partition_row_ids,device,derivative_capture=None):
 policy=validate_recipe(policy);capture=validate_capture(derivative_capture);requests=policy['config']['requests']
 if requests and not capture['values']:raise ValueError('Enable derivative value capture before running requested analysis.')
 if __package__:from .model_input import observation_count,select_observations
 else:from model_input import observation_count,select_observations
 if any(p.device.type!=device for _,p in native_owners(model)):raise ValueError('Model and analysis device differ; no implicit fallback.')
 planned=[];elements=0;estimated_bytes=len(encoded(policy))+4096
 for request in requests:
  partition=request['partition'];start=request['row_start'];end=start+request['row_count']
  if type(prepared_inputs) is not dict or partition not in prepared_inputs or type(partition_row_ids) is not dict or partition not in partition_row_ids:raise ValueError('Choose an actual verified prepared partition.')
  full=prepared_inputs[partition];ids=partition_row_ids[partition]
  contract=_bag_contract(model)
  if contract is not None:
   _bag_coordinates(model,full,device,bounded=False)
   if request['output']['node']==contract['sourceID']:raise ValueError('Choose an ordinary floating model output, not bag coordinates.')
   address=OutputAddress(**request['output']);layout=select_output(model.layouts,address)
   if not hasattr(layout,'sample') or select_output(model.dtypes,address)!='float32':raise ValueError('Choose an actual ordinary float32 output.')
   if request['upstream'] is not None and request['upstream']['shape']!=[request['row_count'],*layout.sample]:raise ValueError('Upstream must exactly match selected output geometry before execution.')
  if type(ids) is not list or len(ids)!=observation_count(full) or end>2**63-1 or end>observation_count(full):raise ValueError('Requested row range exceeds the verified partition.')
  batch=select_observations(full,list(range(start,end))) if _bag_contract(model) is not None else full[start:end]
  if _bag_contract(model) is not None:_bag_coordinates(model,batch,device)
  else:_inputs(batch,device)
  resolved,n=_targets(model,batch,request);elements+=n
  estimated_bytes+=len(encoded(request))+len(encoded([dict(target=t,**m) for t,m,_ in resolved]))+n*32+4096
  if estimated_bytes>8388608:raise ValueError('Derivative artifact exceeds8MiB before execution.')
  if elements>MAX_ELEMENTS:raise ValueError('Total requested derivative appearances exceed65536 elements before execution.')
  planned.append((request,batch,ids[start:end]))
 return planned

def analyze_requested_gradients(model,prepared_inputs,policy,*,partition_row_ids,source_identity,device,check_cancelled=lambda:None,derivative_capture=None):
 policy=validate_recipe(policy);capture=validate_capture(derivative_capture);requests=policy['config']['requests']
 if requests and not capture['values']:raise ValueError('Enable derivative value capture before running requested analysis.')
 if not requests:return None
 exact(source_identity,('runID','graphSHA256','checkpointSHA256','preparationManifestSHA256','preparationGraphDigest'))
 if type(source_identity['runID']) is not str or not source_identity['runID']:raise ValueError('Bind analysis to an actual saved run.')
 if any(type(v) is not str or re.fullmatch('[0-9a-f]{64}',v) is None for k,v in source_identity.items() if k!='runID'):raise ValueError('Analysis requires exact saved graph/checkpoint/preparation source pins.')
 planned=preflight_requested_gradients(model,prepared_inputs,policy,partition_row_ids=partition_row_ids,device=device,derivative_capture=derivative_capture)
 result=dict(schemaVersion=1,runID=source_identity['runID'],phase='finalSnapshotEvaluation',device=device,dtype='float32',torchVersion=torch.__version__,recipe=policy,source={k:v for k,v in source_identity.items() if k!='runID'},requests=[])
 contract=_bag_contract(model)
 if contract is not None:result.update(schemaVersion=2,modelInput=contract)
 for request,batch,ids in planned:
  measured=analyze(model,batch,request,device=device,check_cancelled=check_cancelled)
  result['requests'].append(dict(request=request,sample=dict(partition=request['partition'],rowStart=request['row_start'],rowCount=request['row_count'],rowIDsSHA256=hashlib.sha256(encoded(ids)).hexdigest()),**measured))
  if contract is not None:
   counts=[batch.indices.shape[1]]*request['row_count'] if batch.layout=='fixed' else (batch.offsets[1:]-batch.offsets[:-1]).cpu().tolist() if batch.include_last_offset else (torch.cat((batch.offsets[1:],batch.offsets.new_tensor([batch.indices.numel()])))-batch.offsets).cpu().tolist()
   result['requests'][-1]['sample']['bagItemCounts']=counts
 if len(encoded(result))>8388608:raise ValueError('Derivative results exceed the8MiB artifact limit; reduce requests.')
 return result
