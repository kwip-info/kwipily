"""Strict native objectives; observation-owned roles, no implicit target repair."""
import copy,math
import torch
from torch import nn
from torch.nn import functional as F
if __package__:
 from .loss_configuration import validate_recipe,resolve_objective,elements,MAX_ELEMENTS
 from .tensor_layout import BatchedLayout
else:
 from loss_configuration import validate_recipe,resolve_objective,elements,MAX_ELEMENTS
 from tensor_layout import BatchedLayout

class LossObjective(nn.Module):
 def __init__(self,recipe,device='cpu',dtype=torch.float32):
  super().__init__();self.recipe=validate_recipe(recipe);self.component_id=self.recipe['componentID'];self.configuration=self.recipe['config'];self._canonical_recipe=copy.deepcopy(self.recipe)
  d=torch.device(device)
  if d.type not in ('cpu','mps') or dtype not in (torch.float32,torch.float64) or (dtype==torch.float64 and d.type!='cpu'):raise ValueError('Objectives support float32 CPU/MPS or explicit CPU float64 lessons; no backend transfer is implicit.')
  self.expected_device=torch.empty(0,device=d).device;self.expected_dtype=dtype
  c=self.configuration
  if c.get('class_weights'):self.register_buffer('class_weights',torch.tensor(c['class_weights'],device=d,dtype=dtype))
  for prefix in ('weight','pos_weight'):
   if c.get(prefix+'_mode')=='configured':self.register_buffer(prefix,torch.tensor(c[prefix+'_values'],device=d,dtype=dtype).reshape(c[prefix+'_shape']))
 def forward(self,prediction,roles,scalarization='mean'):return evaluate_objective(self,prediction,roles,scalarization)
 def load_state_dict(self,state_dict,strict=True,assign=False):
  if self.recipe!=self._canonical_recipe or self.configuration!=self._canonical_recipe['config']:raise ValueError('Objective identity changed after admission.')
  expected=self.state_dict()
  if type(state_dict) not in (dict,__import__('collections').OrderedDict) or set(state_dict)!=set(expected):raise ValueError('Objective state requires exactly its configured fixed buffers.')
  for key,value in state_dict.items():
   values=self.configuration['class_weights'] if key=='class_weights' else self.configuration[key+'_values'];dims=[len(values)] if key=='class_weights' else self.configuration[key+'_shape']
   target=torch.tensor(values,device=self.expected_device,dtype=self.expected_dtype).reshape(dims)
   if not isinstance(value,torch.Tensor) or value.layout!=torch.strided or value.dtype!=target.dtype or value.shape!=target.shape or not torch.isfinite(value).all().item() or not torch.equal(value.to(target.device),target):raise ValueError('Objective fixed buffer differs from canonical recipe shape, dtype or values.')
  return super().load_state_dict(state_dict,strict=True,assign=assign)

def create_objective(recipe,device='cpu',dtype=torch.float32):return LossObjective(recipe,device,dtype)

def _distribution(value,axis,log=False):
 p=value.exp() if log else value
 if (not log and ((p<0).any().item() or (p>1).any().item())) or not torch.allclose(p.sum(axis),torch.ones_like(p.sum(axis)),atol=1e-4,rtol=0.):raise ValueError('Provide normalized probabilities, or use LogSoftmax for finite log-probabilities; objective inputs are never silently normalized.')

def _tensor(value,obj,integer=False):
 dtype=torch.int64 if integer else obj.expected_dtype
 if not isinstance(value,torch.Tensor) or value.layout!=torch.strided or value.dtype!=dtype or value.device!=obj.expected_device or value.numel()>MAX_ELEMENTS or not torch.isfinite(value).all().item():raise ValueError('Objective roles must be finite dense tensors of the exact declared dtype on the prediction device, within 100M elements.')

def evaluate_objective(objective,prediction,roles,scalarization='mean'):
 if type(objective) is not LossObjective or scalarization not in ('mean','sum') or type(roles) is not dict:raise ValueError('Choose a reviewed native objective and mean/sum scalarization.')
 obj=objective;c=obj.configuration;op=obj.component_id;_tensor(prediction,obj)
 if obj.recipe!=obj._canonical_recipe or c!=obj._canonical_recipe['config'] or validate_recipe(obj.recipe)['config']!=c:raise ValueError('Objective live configuration differs from its canonical recipe.')
 for key,value in obj.named_buffers():
  values=c['class_weights'] if key=='class_weights' else c[key+'_values'];dims=[len(values)] if key=='class_weights' else c[key+'_shape']
  expected=torch.tensor(values,device=obj.expected_device,dtype=obj.expected_dtype).reshape(dims)
  if not torch.equal(value,expected):raise ValueError('Objective live fixed buffer differs from its canonical recipe.')
 if not 2<=prediction.ndim<=8 or prediction.shape[0]<=0:raise ValueError('Prediction requires nonempty observations and full rank 2…8.')
 n=prediction.shape[0];metadata_roles={}
 for key,v in roles.items():
  _tensor(v,obj,integer=key in ('input_lengths','target_lengths') or key=='target' and (op in ('loss_nll','loss_ctc') or op=='loss_cross_entropy' and c['target_mode']=='class_indices'))
  if v.ndim<1 or v.shape[0]!=n:raise ValueError('Every target role must preserve the same current observation count.')
  metadata_roles[key]={'dtype':'int64' if v.dtype==torch.int64 else 'float32','sampleShape':list(v.shape[1:])}
 resolved=resolve_objective(op,c,BatchedLayout(tuple(prediction.shape[1:])),metadata_roles)
 if any(resolved[k]>MAX_ELEMENTS//n for k in ('predictionElementsPerObservation','rawElementsPerObservation')):raise ValueError('Current objective batch exceeds its 100M tensor budget.')
 target=roles.get('target');reduction=c['reduction'];ignored=masked=0;active=n;denominator=None;denominator_convention='nativeReduction' if reduction in ('mean','batchmean') else 'unusedDiagnostic'
 if op in ('loss_cross_entropy','loss_nll'):
  axis=resolved['classAxis'];x=prediction.movedim(axis,1);weights=getattr(obj,'class_weights',None)
  prob=op=='loss_cross_entropy' and c['target_mode']=='probabilities'
  if prob:_distribution(target,axis);y=target.movedim(axis,1);denominator=elements(resolved['rawSampleShape'])*n
  else:
   y=target;valid=y!=c['ignore_index'];ignored=int((~valid).sum().item());active=int(valid.sum().item())
   if ((y[valid]<0)|(y[valid]>=resolved['classCount'])).any().item():raise ValueError('Class targets must select an existing class or the configured ignored index.')
   if weights is not None:
    if reduction=='mean':
     denominator=float(weights[y[valid]].sum().item());denominator_convention='nativeEffectiveClassWeight'
     if not math.isfinite(denominator):raise ValueError('Native effective class-weight denominator overflow; reduce class weights or batch size before taking a mean.')
    else:
     # At most 4096 transferred values at once; this is explicitly a bounded
     # diagnostic total, never a replacement native mean denominator.
     denominator=0.;flat=y.reshape(-1)
     for start in range(0,flat.numel(),4096):
      chunk=flat[start:start+4096];chosen=chunk[chunk!=c['ignore_index']]
      denominator=math.fsum((denominator,math.fsum(weights[chosen].detach().cpu().tolist())))
     denominator_convention='diagnosticActualWeightFloat64Total'
   else:denominator=active;denominator_convention='nativeActivePositionCount' if reduction=='mean' else 'diagnosticActivePositionCount'
  if reduction=='mean' and denominator<=0:raise ValueError('Mean objective has no active positive effective denominator; restore valid targets or nonzero class weights.')
  if op=='loss_nll':_distribution(prediction,axis,True);raw=F.nll_loss(x,y,weight=weights,ignore_index=c['ignore_index'],reduction=reduction)
  else:raw=F.cross_entropy(x,y,weight=weights,ignore_index=c['ignore_index'],label_smoothing=c['label_smoothing'],reduction=reduction)
 elif op=='loss_bce_with_logits':
  if ((target<0)|(target>1)).any().item():raise ValueError('Independent label targets must be between 0 and 1.')
  def weight(prefix):
   mode=c[prefix+'_mode']
   if mode=='none':return None
   v=getattr(obj,prefix) if mode=='configured' else roles[prefix]
   if v.requires_grad or (v<0).any().item():raise ValueError('Loss weights are nonnegative fixed/prepared data; native weights do not support learned gradients.')
   s=list(v.shape) if mode=='configured' else list(v.shape[1:]);leading=1 if mode=='configured' else n
   return v.reshape([leading]+[1]*(prediction.ndim-1-len(s))+s).expand_as(prediction)
  raw=F.binary_cross_entropy_with_logits(prediction,target,weight=weight('weight'),pos_weight=weight('pos_weight'),reduction=reduction);denominator=prediction.numel()
 elif op in ('loss_mse','loss_l1'):
  union=resolved['rawSampleShape'];rank=len(union)
  x=prediction.reshape([n]+[1]*(rank-prediction.ndim+1)+list(prediction.shape[1:])).expand([n]+union)
  y=target.reshape([n]+[1]*(rank-target.ndim+1)+list(target.shape[1:])).expand([n]+union)
  raw=(F.mse_loss if op=='loss_mse' else F.l1_loss)(x,y,reduction=reduction);denominator=n*elements(union)
 elif op=='loss_kl_div':
  _distribution(prediction,-1,True);_distribution(target,-1,c['log_target']);raw=F.kl_div(prediction,target,reduction=reduction,log_target=c['log_target']);denominator=n if reduction=='batchmean' else prediction.numel()
 elif op=='loss_ctc':
  _distribution(prediction,-1,True);il=roles['input_lengths'];tl=roles['target_lengths'];time=prediction.shape[1];size=target.shape[1]
  if ((il<1)|(il>time)).any().item() or ((tl<0)|(tl>size)).any().item():raise ValueError('CTC input lengths must be 1…time and target lengths 0…padded width.')
  active_prefix=[];work=0;infeasible=0;impossible_rows=[]
  for row in range(n):
   length=int(tl[row].item());steps=int(il[row].item());y=target[row,:length]
   if ((y<0)|(y>=prediction.shape[-1])|(y==c['blank'])).any().item():raise ValueError('Active CTC labels must be valid nonblank classes; ignored suffixes are not labels.')
   repeat=int((y[1:]==y[:-1]).sum().item());impossible=steps<length+repeat;impossible_rows.append(impossible);infeasible+=impossible;work+=steps*(2*length+1)
   if work>MAX_ELEMENTS:raise ValueError('CTC alignment exceeds 100M active work elements; shorten sequences or batch size.')
   active_prefix.append(y)
  native_target=torch.cat(active_prefix) if c['target_layout']=='packed' else target
  if c['zero_infinity']:
   unmasked=F.ctc_loss(prediction.transpose(0,1),native_target,il,tl,blank=c['blank'],reduction='none',zero_infinity=False)
   impossible_tensor=torch.tensor(impossible_rows,device=prediction.device,dtype=torch.bool)
   if ((~torch.isfinite(unmasked))&(~impossible_tensor)).any().item():raise ValueError('CTC numeric overflow produced a nonfinite feasible alignment; reduce log-probability magnitude. zero_infinity only masks impossible alignments in this contract.')
  raw=F.ctc_loss(prediction.transpose(0,1),native_target,il,tl,blank=c['blank'],reduction=reduction,zero_infinity=c['zero_infinity']);masked=infeasible if c['zero_infinity'] else 0;active=n-masked;denominator=n
 elif op=='loss_triplet_margin':
  a,p,neg=prediction.unbind(1);raw=F.triplet_margin_loss(a,p,neg,margin=c['margin'],p=c['p'],eps=c['eps'],swap=c['swap'],reduction=reduction);denominator=n
 if not torch.isfinite(raw).all().item():raise ValueError('Objective became nonfinite; check alignment feasibility, probability inputs and distance geometry. No value is repaired silently.')
 scalar=raw.mean() if reduction=='none' and scalarization=='mean' else raw.sum() if reduction=='none' else raw
 if not torch.isfinite(scalar).all().item():raise ValueError('Objective scalar became nonfinite; reduce magnitude or repair target geometry.')
 aggregation_denominator=(raw.numel() if scalarization=='mean' else 1) if reduction=='none' else denominator if reduction in ('mean','batchmean') else 1
 aggregation_convention='rawElementMean' if reduction=='none' and scalarization=='mean' else 'sum' if reduction in ('none','sum') else 'nativeBatchMean' if reduction=='batchmean' else 'nativeMean'
 def finite_gradient(gradient):
  if not torch.isfinite(gradient).all().item():raise ValueError('Objective gradient became nonfinite; inspect zero-probability target derivatives, alignment or distance cusps. Inputs are never silently repaired.')
  return gradient
 for value in [prediction]+[v for key,v in roles.items() if key not in ('weight','pos_weight')]:
  if value.requires_grad:value.register_hook(finite_gradient)
 meta=dict(componentID=op,denominatorConvention=denominator_convention,reduction=reduction,scalarization=scalarization if reduction=='none' else None,rawShape=list(raw.shape),observations=n,activeCount=active,ignoredCount=ignored,maskedCount=masked,effectiveScalar=float(scalar.detach().item()),denominator=denominator,device=prediction.device.type,aggregationNumerator=float(scalar.detach().item())*aggregation_denominator,aggregationDenominator=aggregation_denominator,aggregationConvention=aggregation_convention)
 if any(type(value) is float and not math.isfinite(value) for value in meta.values()):raise ValueError('Objective metadata arithmetic became nonfinite; reduce weights, batch size or objective magnitude before reporting aggregates.')
 if op=='loss_ctc':meta.update(infeasibleCount=infeasible,alignmentWork=work,meanConvention='mean of loss / max(active target length, 1)')
 return dict(raw=raw,scalar=scalar,metadata=meta)
