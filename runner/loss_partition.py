"""Objective-owned unreduced partition contributions and pre-update admission.

The public native helper's local mean refusal remains unchanged. Internal
measurement copies request native none without rewriting the original recipe.
"""
import copy,math
from dataclasses import dataclass
import torch
if __package__:
 from .loss_configuration import validate_recipe,resolve_objective,MAX_ELEMENTS
 from .loss_objectives import LossObjective,create_objective,evaluate_objective,_distribution
 from .tensor_layout import BatchedLayout
else:
 from loss_configuration import validate_recipe,resolve_objective,MAX_ELEMENTS
 from loss_objectives import LossObjective,create_objective,evaluate_objective,_distribution
 from tensor_layout import BatchedLayout

@dataclass(frozen=True)
class ObjectivePartitionContribution:
 recipe:dict
 scalarization:str
 raw:torch.Tensor
 numerator:torch.Tensor
 denominator:float
 metadata:dict
 classification:dict|None
 def __getitem__(self,key):return getattr(self,key)

def objective_partition_contribution(objective,prediction,roles,scalarization='mean'):
 if type(objective) is not LossObjective or scalarization not in ('mean','sum'):raise ValueError('Partition contribution requires an admitted objective and mean/sum scalarization.')
 # This verifies actual configured buffers against original constants before
 # copying them. It does not restore foreign/parent state or mutate the recipe.
 objective.load_state_dict(objective.state_dict())
 original=validate_recipe(objective.recipe);measured=copy.deepcopy(original);measured['config']['reduction']='none'
 measuring=create_objective(measured,objective.expected_device,objective.expected_dtype)
 measuring.load_state_dict(objective.state_dict())
 result=evaluate_objective(measuring,prediction,roles,'sum');raw=result['raw'];meta=dict(result['metadata']);op=original['componentID'];c=original['config'];reduction=c['reduction']
 numerator=raw.sum();denominator=1.
 if reduction=='mean':
  if op=='loss_ctc':numerator=(raw/roles['target_lengths'].clamp_min(1).to(raw.dtype)).sum();denominator=float(len(raw))
  else:denominator=float(meta['denominator'])
 elif reduction=='batchmean':denominator=float(len(prediction))
 elif reduction=='none' and scalarization=='mean':denominator=float(raw.numel())
 classification=None
 if op=='loss_nll' or op=='loss_cross_entropy' and c['target_mode']=='class_indices':
  target=roles['target'];valid=target!=c['ignore_index'];selected=prediction.argmax(c['class_axis']);eligible=int(valid.sum().item());correct=int(((selected==target)&valid).sum().item())
  classification=dict(rule='argmax',classAxis=c['class_axis'],ignoreIndex=c['ignore_index'],eligibleCount=eligible,correctCount=correct)
 meta.update(componentID=op,reduction=reduction,scalarization=scalarization if reduction=='none' else None,measurementReduction='none')
 # Do not publish the measurement helper's local scalar as the selected mean.
 for key in ('effectiveScalar','aggregationNumerator','aggregationDenominator','aggregationConvention'):meta.pop(key,None)
 return ObjectivePartitionContribution(copy.deepcopy(original),scalarization,raw,numerator,denominator,meta,classification)

def finish_objective_partition(contributions,canonical_recipe,scalarization='mean'):
 recipe=validate_recipe(canonical_recipe)
 if type(contributions) not in (list,tuple) or not contributions or scalarization not in ('mean','sum'):raise ValueError('Finish a nonempty bounded objective partition with mean/sum scalarization.')
 first=contributions[0]
 if type(first) is not ObjectivePartitionContribution:raise ValueError('Partition requires measured objective contributions.')
 count=elements=ignored=masked=active=0;denominator=0.;numerators=[];raws=[];classification=None;work=0;infeasible=0
 for part in contributions:
  if type(part) is not ObjectivePartitionContribution or part.recipe!=recipe or part.scalarization!=scalarization:raise ValueError('Partition recipe or scalarization differs from the original measurement.')
  raw=part.raw
  if not isinstance(raw,torch.Tensor) or raw.ndim<1 or raw.shape[0]<=0 or raw.device!=first.raw.device or raw.dtype!=first.raw.dtype or raw.shape[1:]!=first.raw.shape[1:] or not torch.isfinite(raw).all().item():raise ValueError('Partition raw geometry/dtype/device or finite values differ.')
  elements+=raw.numel();count+=len(raw)
  if elements>MAX_ELEMENTS:raise ValueError('Partition raw output exceeds the 100M element budget before concatenation.')
  if type(part.denominator) not in (int,float) or not math.isfinite(part.denominator) or part.denominator<0 or not isinstance(part.numerator,torch.Tensor) or part.numerator.ndim!=0 or not torch.isfinite(part.numerator).item():raise ValueError('Partition numerator/denominator is invalid.')
  denominator+=part.denominator;numerators.append(part.numerator);raws.append(raw);meta=part.metadata
  ignored+=meta['ignoredCount'];masked+=meta['maskedCount'];active+=meta['activeCount'];work+=meta.get('alignmentWork',0);infeasible+=meta.get('infeasibleCount',0)
  if part.classification is not None:
   if classification is None:classification={**part.classification,'eligibleCount':0,'correctCount':0}
   if any(classification[k]!=part.classification[k] for k in ('rule','classAxis','ignoreIndex')):raise ValueError('Partition decoding rule differs.')
   for key in ('eligibleCount','correctCount'):classification[key]+=part.classification[key]
 reduction=recipe['config']['reduction'];is_sum=reduction=='sum' or reduction=='none' and scalarization=='sum'
 if is_sum:denominator=1.
 if reduction=='mean' and recipe['componentID'] in ('loss_cross_entropy','loss_nll') and denominator>torch.finfo(first.raw.dtype).max:raise ValueError('Partition native effective class-weight denominator overflow; reduce weights before taking a mean.')
 if denominator<=0:raise ValueError('Partition mean has no active positive effective denominator; provide valid targets or nonzero class weights.')
 numerator=sum(numerators);scalar=numerator/denominator
 if not torch.isfinite(scalar).item():raise ValueError('Partition scalar became nonfinite; inspect objective magnitude.')
 raw=torch.cat(raws,0) if reduction=='none' else scalar
 convention='sum' if is_sum else 'rawElementMean' if reduction=='none' else 'nativeBatchMean' if reduction=='batchmean' else 'nativeMean'
 metadata=dict(componentID=recipe['componentID'],reduction=reduction,scalarization=scalarization if reduction=='none' else None,rawShape=list(raw.shape),observations=count,activeCount=active,ignoredCount=ignored,maskedCount=masked,effectiveScalar=float(scalar.detach().item()),aggregationNumerator=float(numerator.detach().item()),aggregationDenominator=denominator,aggregationConvention=convention,device=scalar.device.type,recipe=copy.deepcopy(recipe))
 if classification is not None:
  classification['accuracy']=classification['correctCount']/classification['eligibleCount'] if classification['eligibleCount'] else None
  metadata['classification']=classification
 if recipe['componentID']=='loss_ctc':metadata.update(alignmentWork=work,infeasibleCount=infeasible,meanConvention='mean of loss / max(active target length, 1)')
 return dict(raw=raw,scalar=scalar,metadata=metadata)

def preflight_objective_training_batches(recipe,training_roles,batch_size,output_layout,*,training_rows=None):
 original=validate_recipe(recipe);op=original['componentID'];c=original['config']
 if type(training_roles) is not dict or type(batch_size) is not int or not 1<=batch_size<=100000:raise ValueError('Training preflight requires exact role tensors and a batch size 1…100000.')
 if training_rows is None:
  if not training_roles:raise ValueError('Training observation count is required for target-free triplets.')
  first=next(iter(training_roles.values()));training_rows=first.shape[0] if isinstance(first,torch.Tensor) and first.ndim else None
 if type(training_rows) is not int or not 1<=training_rows<=100000:raise ValueError('Training preflight requires 1…100000 observations.')
 metadata={}
 for key,value in training_roles.items():
  if not isinstance(value,torch.Tensor) or value.layout!=torch.strided or value.ndim<1 or len(value)!=training_rows or value.dtype not in (torch.float32,torch.int64) or value.numel()>MAX_ELEMENTS or not torch.isfinite(value).all().item():raise ValueError('Training target roles must be aligned finite dense float32/int64 tensors.')
  metadata[key]=dict(dtype='int64' if value.dtype==torch.int64 else 'float32',sampleShape=list(value.shape[1:]))
 resolved=resolve_objective(op,c,output_layout,metadata)
 largest_batch=min(batch_size,training_rows)
 if any(resolved[k]>MAX_ELEMENTS//largest_batch for k in ('predictionElementsPerObservation','rawElementsPerObservation')):raise ValueError('Actual training batch exceeds its objective tensor budget before model execution.')
 target=training_roles.get('target')
 if op=='loss_nll' or op=='loss_cross_entropy' and c['target_mode']=='class_indices':
  valid=target!=c['ignore_index']
  if ((target[valid]<0)|(target[valid]>=resolved['classCount'])).any().item():raise ValueError('Training class targets select an unavailable class.')
 elif op=='loss_cross_entropy':_distribution(target,resolved['classAxis'])
 elif op=='loss_bce_with_logits':
  if ((target<0)|(target>1)).any().item():raise ValueError('Training independent-label targets must be between 0 and 1.')
  for key in ('weight','pos_weight'):
   if key in training_roles and (training_roles[key].requires_grad or (training_roles[key]<0).any().item()):raise ValueError('Training mapped weights must be nonnegative prepared data, never learned settings.')
 elif op=='loss_kl_div':_distribution(target,-1,c['log_target'])
 elif op=='loss_ctc':
  il=training_roles['input_lengths'];tl=training_roles['target_lengths'];time=output_layout.sample[0];classes=output_layout.sample[1];size=target.shape[1]
  if ((il<1)|(il>time)).any().item() or ((tl<0)|(tl>size)).any().item():raise ValueError('Training CTC lengths differ from time or padded target geometry.')
  work=0
  for row in range(training_rows):
   length=int(tl[row].item());active=target[row,:length]
   if ((active<0)|(active>=classes)|(active==c['blank'])).any().item():raise ValueError('Training active CTC labels must be valid nonblank classes.')
   if row%batch_size==0:work=0
   work+=int(il[row].item())*(2*length+1)
   if work>MAX_ELEMENTS:raise ValueError('Training CTC batch exceeds its active alignment work budget.')
 if c['reduction']=='mean' and (op=='loss_nll' or op=='loss_cross_entropy' and c['target_mode']=='class_indices'):
  target=training_roles['target'];classes=resolved['classCount'];valid=target!=c['ignore_index']
  if ((target[valid]<0)|(target[valid]>=classes)).any().item():raise ValueError('Training class targets select an unavailable class.')
  weights=torch.tensor(c['class_weights'],dtype=torch.float64,device='cpu') if c['class_weights'] else None
  for start in range(0,training_rows,batch_size):
   batch=target[start:start+batch_size];chosen=batch[batch!=c['ignore_index']]
   denominator=float(weights[chosen.cpu()].sum().item()) if weights is not None else chosen.numel()
   if not math.isfinite(denominator) or denominator>torch.finfo(torch.float32).max:raise ValueError(f'Training batch {start//batch_size+1} native effective class-weight denominator overflow; reduce weights before running.')
   if denominator<=0:raise ValueError(f'Training batch {start//batch_size+1} has an undefined mean denominator; change batch size, valid targets or class weights before running.')
 return dict(recipe=copy.deepcopy(original),observations=training_rows,batches=math.ceil(training_rows/batch_size),lastBatchSize=training_rows%batch_size or min(batch_size,training_rows),meanDenominatorsDefined=True)
