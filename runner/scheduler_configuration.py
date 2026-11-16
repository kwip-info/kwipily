"""Strict scalar scheduler recipes and independently admitted schedule geometry."""
import math
MAXIMUM=3.4028234663852886e38
POSITIVE_MINIMUM=1.1754943508222875e-38
BETA_MAX=.9999999403953552
TICK_MAXIMUM=10000
TRACE_BYTE_MAXIMUM=4000000
DEFAULTS={
 'scheduler_step_lr':dict(step_size=2,gamma=.1),
 'scheduler_cosine_annealing_lr':dict(T_max=10,eta_min=0.),
 'scheduler_reduce_on_plateau':dict(mode='min',factor=.1,patience=10,threshold=.0001,threshold_mode='rel',cooldown=0,min_lr=0.,eps=1e-8,metric='validation_loss'),
 'scheduler_one_cycle_lr':dict(max_lr=.1,planned_total_batches=10,pct_start=.3,anneal_strategy='cos',cycle_momentum=True,base_momentum=.85,max_momentum=.95,div_factor=25.,final_div_factor=10000.,three_phase=False),
 'scheduler_linear_lr':dict(start_factor=1/3,end_factor=1.,total_iters=5)}
NAMES={'scheduler_step_lr':'StepLR','scheduler_cosine_annealing_lr':'CosineAnnealingLR','scheduler_reduce_on_plateau':'ReduceLROnPlateau','scheduler_one_cycle_lr':'OneCycleLR','scheduler_linear_lr':'LinearLR'}
CHOICES={'mode':('min','max'),'threshold_mode':('rel','abs'),'metric':('validation_loss','validation_accuracy'),'anneal_strategy':('cos','linear')}
def phase_ends(c):
 n=c['planned_total_batches'];first=c['pct_start']*n-1
 return [first,2*c['pct_start']*n-2,n-1] if c['three_phase'] else [first,n-1]
def canonical_configuration(component_id,supplied):
 if type(component_id) is not str or component_id not in DEFAULTS or type(supplied) is not dict or set(supplied)-set(DEFAULTS[component_id]):raise ValueError('Scheduler accepts only its declared controls.')
 c={**DEFAULTS[component_id],**supplied}
 for key,default in DEFAULTS[component_id].items():
  v=c[key]
  if type(default) is bool:
   if type(v) is not bool:raise ValueError('Scheduler switches require Boolean values.')
  elif type(default) is str:
   if type(v) is not str or v not in CHOICES[key]:raise ValueError('Choose a declared scheduler option.')
  elif type(default) is int:
   low=0 if key in ('patience','cooldown') else 3 if key=='planned_total_batches' else 1
   if type(v) is not int or not low<=v<=TICK_MAXIMUM:raise ValueError('Scheduler integer control is outside its supported bounds.')
  else:
   if type(v) not in (int,float) or not -MAXIMUM<=v<=MAXIMUM or not math.isfinite(v):raise ValueError('Scheduler controls need finite numbers.')
   low=POSITIVE_MINIMUM if key in ('div_factor','final_div_factor','start_factor','end_factor','pct_start') else 0.
   high=1. if key in ('start_factor','end_factor','pct_start','factor','base_momentum','max_momentum') else MAXIMUM
   if not low<=v<=high or (key in ('pct_start','factor') and v==1):raise ValueError('Scheduler numeric control is outside its supported bounds.')
   c[key]=float(v)
 if component_id=='scheduler_one_cycle_lr':
  ends=phase_ends(c);starts=[0,*ends[:-1]]
  if any(end-start<=0 for start,end in zip(starts,ends)):raise ValueError('Each scheduled phase must have a positive interpolation span.')
  if c['base_momentum']>c['max_momentum']:raise ValueError('Minimum scheduled momentum cannot exceed its maximum.')
  initial=c['max_lr']/c['div_factor'];minimum=initial/c['final_div_factor']
  if not all(math.isfinite(x) and 0<=x<=MAXIMUM for x in (initial,minimum)):raise ValueError('Derived OneCycle rates exceed supported arithmetic.')
 return c

def validate_recipe(value):
 if type(value) is not dict or set(value)!={'schemaVersion','componentID','configurationVersion','config'} or type(value['schemaVersion']) is not int or value['schemaVersion']!=1 or type(value['configurationVersion']) is not int or value['configurationVersion']!=1:raise ValueError('Scheduler recipe needs exact schema/configuration version1 and declared fields.')
 return dict(schemaVersion=1,componentID=value['componentID'],configurationVersion=1,config=canonical_configuration(value['componentID'],value['config']))

def constructor_options(recipe):
 r=validate_recipe(recipe);c=dict(r['config'])
 if r['componentID']=='scheduler_reduce_on_plateau':c.pop('metric')
 if r['componentID']=='scheduler_one_cycle_lr':c['total_steps']=c.pop('planned_total_batches')
 return c

def capability(component_id):
 if component_id not in DEFAULTS:raise ValueError('Unknown scheduler.')
 return {'version':{'major':1,'minor':0,'patch':0},'adapterID':'kwip.runner.scheduler.'+component_id,'recipeSchemaVersions':[1],'configurationVersions':[1],'workflowSchemaVersions':[7],'graphDTypes':['float32'],'devices':['cpu','mps'],'requiresExplicitOptimizer':True,'oneParameterGroup':True,'optimizerComponentIDs':['optimizer_sgd','optimizer_adam','optimizer_adamw','optimizer_rmsprop','optimizer_adagrad'],'momentumCyclingOptimizerIDs':['optimizer_sgd','optimizer_adam','optimizer_adamw','optimizer_rmsprop'] if component_id=='scheduler_one_cycle_lr' else [],'timing':'afterOptimizerBatch' if component_id=='scheduler_one_cycle_lr' else 'afterCompletedEpochValidation','constructorInitializationSeparate':True,'tickMaximum':TICK_MAXIMUM,'traceByteMaximum':TRACE_BYTE_MAXIMUM,'numericMaximum':MAXIMUM,'positiveMinimum':POSITIVE_MINIMUM,'scheduledBetaMaximum':BETA_MAX,'defaults':dict(DEFAULTS[component_id]),'cpuResume':'epochBoundaryExactCPU','mpsResume':'weightsOnly','fineTune':'freshOptimizerAndScheduler','statePolicy':'strictRecipeAndRecomputedTrace','terminalNextRatePolicy':'finiteUnusedAtExhaustedHorizon' if component_id=='scheduler_one_cycle_lr' else 'nonnegativeFinite','horizonPolicy':'plannedTotalBatches' if component_id=='scheduler_one_cycle_lr' else 'decreasingHalfCycle' if component_id=='scheduler_cosine_annealing_lr' else 'boundedEpochTicks','export':'freshTrainingSequence'}

def validate_compatibility(recipe,optimizer_recipe):
 if __package__:
  from .optimizer_configuration import validate_recipe as optimizer_validate
 else:
  from optimizer_configuration import validate_recipe as optimizer_validate
 if optimizer_recipe is None:raise ValueError('A scheduler requires a connected explicit optimizer recipe.')
 r=validate_recipe(recipe);o=optimizer_validate(optimizer_recipe);c=r['config'];oc=o['config']
 if r['componentID']=='scheduler_cosine_annealing_lr' and c['eta_min']>oc['lr']:raise ValueError('Cosine minimum cannot exceed the initial optimizer rate in the decreasing half-cycle.')
 if r['componentID']=='scheduler_one_cycle_lr' and c['cycle_momentum']:
  if o['componentID']=='optimizer_adagrad':raise ValueError('Adagrad requires momentum cycling off.')
  if o['componentID'] in ('optimizer_adam','optimizer_adamw') and c['max_momentum']>BETA_MAX:raise ValueError('Scheduled first-beta must remain below one.')
  if o['componentID']=='optimizer_sgd' and oc['nesterov'] and (c['base_momentum']<=0 or oc['dampening']!=0):raise ValueError('Look ahead needs positive minimum scheduled momentum and zero dampening.')
 return r,o

def create_scheduler(optimizer,recipe,optimizer_recipe):
 import torch
 r,o=validate_compatibility(recipe,optimizer_recipe)
 if __package__:
  from .optimizer_configuration import NAMES as optimizer_names
 else:
  from optimizer_configuration import NAMES as optimizer_names
 if type(optimizer) is not getattr(torch.optim,optimizer_names[o['componentID']]) or len(optimizer.param_groups)!=1:raise ValueError('Scheduler needs the declared reviewed optimizer and one group.')
 return getattr(torch.optim.lr_scheduler,NAMES[r['componentID']])(optimizer,**constructor_options(r))

def scalar_schedule(recipe,base,tick):
 r=validate_recipe(recipe);c=r['config'];op=r['componentID'];momentum=None
 if type(tick) is not int or not 0<=tick<=TICK_MAXIMUM:raise ValueError('Invalid scheduler tick.')
 if op=='scheduler_step_lr':
  try:rate=base*c['gamma']**(tick//c['step_size'])
  except OverflowError as error:raise ValueError('Derived scheduler rate overflows supported arithmetic.') from error
 elif op=='scheduler_cosine_annealing_lr':
  if tick>c['T_max']:raise ValueError('Cosine continuation exceeds its decreasing half-cycle.')
  rate=c['eta_min']+(base-c['eta_min'])*(1+math.cos(math.pi*tick/c['T_max']))/2
 elif op=='scheduler_linear_lr':rate=base*(c['start_factor']+(c['end_factor']-c['start_factor'])*min(tick,c['total_iters'])/c['total_iters'])
 elif op=='scheduler_one_cycle_lr':
  if tick>c['planned_total_batches']:raise ValueError('OneCycle continuation exceeds its pinned horizon.')
  initial=c['max_lr']/c['div_factor'];minimum=initial/c['final_div_factor'];ends=phase_ends(c);pairs=[(initial,c['max_lr']),(c['max_lr'],minimum)];m_pairs=[(c['max_momentum'],c['base_momentum']),(c['base_momentum'],c['max_momentum'])]
  if c['three_phase']:pairs=[(initial,c['max_lr']),(c['max_lr'],initial),(initial,minimum)];m_pairs=[m_pairs[0],m_pairs[1],(c['max_momentum'],c['max_momentum'])]
  start=0
  for i,end in enumerate(ends):
   if tick<=end or i==len(ends)-1:
    fraction=(tick-start)/(end-start)
    def interpolate(a,b):return b+(a-b)*(1+math.cos(math.pi*fraction))/2 if c['anneal_strategy']=='cos' else a+(b-a)*fraction
    rate=interpolate(*pairs[i]);momentum=interpolate(*m_pairs[i]) if c['cycle_momentum'] else None;break
   start=end
 else:raise ValueError('Plateau rates require actual completed validation metrics.')
 unused=op=='scheduler_one_cycle_lr' and tick==c['planned_total_batches']
 if momentum is not None and (not math.isfinite(momentum) or not 0<=momentum<=1):raise ValueError('Derived scheduled momentum is outside its finite bounds.')
 if not math.isfinite(rate) or abs(rate)>MAXIMUM or (rate<0 and not unused):raise ValueError('Derived scheduler rate exceeds its usable finite bounds.')
 return rate,momentum,unused

def preflight(recipe,optimizer_recipe,epochs,batches,parameter_count):
 r,o=validate_compatibility(recipe,optimizer_recipe)
 if any(type(v) is not int or v<1 for v in (epochs,batches,parameter_count)):raise ValueError('Scheduler allocation/timing needs positive integer counts.')
 batch=r['componentID']=='scheduler_one_cycle_lr';ticks=epochs*batches if batch else epochs;c=r['config']
 if ticks>TICK_MAXIMUM or (batch and ticks>c['planned_total_batches']) or (r['componentID']=='scheduler_cosine_annealing_lr' and ticks>c['T_max']):raise ValueError('Execution exceeds the pinned scheduler horizon.')
 # Decimal indices, keys and finite scalar values are bounded before allocating any record.
 estimate=4096+ticks*((384+parameter_count*24) if batch else 384)
 if estimate>TRACE_BYTE_MAXIMUM:raise ValueError('Required schedule trace exceeds its resource bound before allocation.')
 if r['componentID']!='scheduler_reduce_on_plateau':
  for tick in range(ticks+1):
   _,momentum,_=scalar_schedule(r,o['config']['lr'],tick)
   if momentum is not None and o['componentID'] in ('optimizer_adam','optimizer_adamw') and momentum>BETA_MAX:raise ValueError('Derived scheduled first-beta exceeds its safe bound.')
 return ticks

class OptimizerContext:
 def __init__(self,group,buffer_indices):
  self.group=dict(group);self.buffer_indices=None if buffer_indices is None else frozenset(buffer_indices)

class SchedulerController:
 """Native ticks checked against independent scalar curves and bounded replay."""
 def __init__(self,optimizer,recipe,optimizer_recipe,parameter_count):
  self.recipe,self.optimizer_recipe=validate_compatibility(recipe,optimizer_recipe);self.optimizer=optimizer;self.parameter_count=parameter_count;self.native=create_scheduler(optimizer,self.recipe,self.optimizer_recipe);self.records=[];self.buffer_indices=set();self.ticks=0;self.epoch_used=None;self.plateau_best=None;self.plateau_bad=0;self.plateau_cooldown=0;self.expected_rate=self.optimizer_recipe['config']['lr'];self.initial_group=dict(optimizer.state_dict()['param_groups'][0]);self._trace_bytes=512;self.check_curve();self.initialization={'tick':0,'rate':optimizer.param_groups[0]['lr']}
  if self.is_batch and self.recipe['config']['cycle_momentum']:self.initialization['momentum']=optimizer.param_groups[0].get('betas',(optimizer.param_groups[0].get('momentum'),))[0]
 @property
 def is_batch(self):return self.recipe['componentID']=='scheduler_one_cycle_lr'
 def check_curve(self):
  g=self.optimizer.param_groups[0]
  if self.recipe['componentID']!='scheduler_reduce_on_plateau':
   rate,momentum,_=scalar_schedule(self.recipe,self.optimizer_recipe['config']['lr'],self.ticks)
   if not math.isclose(g['lr'],rate,rel_tol=1e-12,abs_tol=1e-15):raise ValueError('Native scheduler rate differs from independently indexed curve.')
   if momentum is not None:
    actual=g['betas'][0] if 'betas' in g else g['momentum']
    if not math.isclose(actual,momentum,rel_tol=1e-12,abs_tol=1e-15):raise ValueError('Scheduled momentum differs from independently indexed curve.')
  else:
   if g['lr']!=self.expected_rate:raise ValueError('Plateau native rate differs from independently retained metric history.')
   sentinel=math.inf if self.recipe['config']['mode']=='min' else -math.inf
   if self.native.best!=(self.plateau_best if self.plateau_best is not None else sentinel) or self.native.mode_worse!=sentinel or self.native.num_bad_epochs!=self.plateau_bad or self.native.cooldown_counter!=self.plateau_cooldown:raise ValueError('Plateau retained state differs from independent metric transitions.')
  return g['lr']
 def context(self):
  self.check_curve();g=dict(self.initial_group);live=self.optimizer.state_dict()['param_groups'][0];g['lr']=live['lr']
  if self.is_batch and self.recipe['config']['cycle_momentum']:
   if 'betas' in g:g['betas']=(live['betas'][0],g['betas'][1])
   else:g['momentum']=live['momentum']
  return OptimizerContext(g,self.buffer_indices if self.is_batch else None)
 def before_update(self,indices):
  if self.is_batch and self.ticks>=self.recipe['config']['planned_total_batches']:raise ValueError('OneCycle horizon exhausted; terminal next rate is unused.')
  self.check_curve()
  if type(indices) is not list or indices!=sorted(set(indices)) or any(type(i) is not int or not 0<=i<self.parameter_count for i in indices):raise ValueError('Scheduled update ownership must be sorted unique parameter indices.')
  if self.optimizer_recipe['componentID']=='optimizer_sgd' and self.optimizer.param_groups[0]['momentum']>0:self.buffer_indices.update(indices)
 def begin_epoch(self):self.epoch_used=self.check_curve()
 def _plateau(self,value):
  c=self.recipe['config'];best=self.plateau_best
  good=best is None or (value<best*(1-c['threshold']) if c['threshold_mode']=='rel' and c['mode']=='min' else value>best*(1+c['threshold']) if c['threshold_mode']=='rel' else value<best-c['threshold'] if c['mode']=='min' else value>best+c['threshold'])
  if good:self.plateau_best=value;self.plateau_bad=0
  else:self.plateau_bad+=1
  if self.plateau_cooldown>0:self.plateau_cooldown-=1;self.plateau_bad=0
  if self.plateau_bad>c['patience']:
   next_rate=max(self.expected_rate*c['factor'],c['min_lr'])
   if self.expected_rate-next_rate>c['eps']:self.expected_rate=next_rate
   self.plateau_cooldown=c['cooldown'];self.plateau_bad=0
 def tick(self,epoch,batch=None,updated_indices=None,metric_value=None):
  if len(self.records)>=TICK_MAXIMUM:raise ValueError('Scheduler trace exceeds its tick bound.')
  op=self.recipe['componentID'];used=self.check_curve() if self.is_batch else self.epoch_used
  if used is None:raise ValueError('Epoch scheduler needs an actual completed epoch before ticking.')
  g=self.optimizer.param_groups[0];momentum=g.get('betas',(g.get('momentum'),))[0];self.optimizer._opt_called=True
  if op=='scheduler_reduce_on_plateau':
   if type(metric_value) not in (int,float) or not math.isfinite(metric_value):raise ValueError('Plateau needs its finite metric from completed validation.')
   self._plateau(metric_value);self.native.step(metric_value)
  else:self.native.step()
  self.ticks+=1;next_rate=self.check_curve();unused=self.is_batch and self.ticks==self.recipe['config']['planned_total_batches'];record=dict(tick=self.ticks,epoch=epoch,usedRate=used,nextRate=next_rate,nextRateUnused=unused)
  if self.is_batch:
   if type(batch) is not int or batch<1:raise ValueError('Batch scheduler requires an actual local batch index.')
   record.update(batch=batch,updatedParameterIndices=list(updated_indices or []))
   if self.recipe['config']['cycle_momentum']:record.update(usedMomentum=momentum,nextMomentum=g.get('betas',(g.get('momentum'),))[0])
  if op=='scheduler_reduce_on_plateau':record.update(metric=self.recipe['config']['metric'],metricValue=float(metric_value),changed=next_rate!=used)
  self.records.append(record)
  import json
  self._trace_bytes+=len(json.dumps(record,allow_nan=False,indent=2).encode())+128
  if self._trace_bytes>TRACE_BYTE_MAXIMUM:raise ValueError('Schedule trace exceeds its byte bound.')
  return record
 def trace(self):return dict(schemaVersion=1,recipe=self.recipe,timing='afterOptimizerBatch' if self.is_batch else 'afterCompletedEpochValidation',initialization=self.initialization,records=self.records)
 def checkpoint(self):return dict(recipe=self.recipe,state=self.native.state_dict(),ticks=self.ticks,timing=self.trace()['timing'],rateTrace=self.trace())
 def restore(self,payload,optimizer_updates=None):
  import torch
  if type(payload) is not dict or set(payload)!={'recipe','state','ticks','timing','rateTrace'} or not _strict_equal(payload['recipe'],self.recipe):raise ValueError('Scheduler checkpoint envelope differs from declared recipe.')
  trace=payload['rateTrace'];expected_header=self.trace()
  if type(trace) is not dict or set(trace)!=set(expected_header) or any(not _strict_equal(trace[k],expected_header[k]) for k in ('schemaVersion','recipe','timing','initialization')) or type(trace['records']) is not list or len(trace['records'])>TICK_MAXIMUM or type(payload['ticks']) is not int or payload['ticks']!=len(trace['records']) or payload['timing']!=trace['timing']:raise ValueError('Scheduler trace header/timing/ticks differ.')
  import json
  if len(json.dumps(trace,allow_nan=False,indent=2).encode())>TRACE_BYTE_MAXIMUM:raise ValueError('Scheduler trace exceeds byte bound.')
  if __package__:
   from .optimizer_configuration import create_optimizer
  else:
   from optimizer_configuration import create_optimizer
  dummy=torch.nn.Parameter(torch.empty(0,dtype=torch.float32));replay=SchedulerController(create_optimizer([dummy],self.optimizer_recipe),self.recipe,self.optimizer_recipe,self.parameter_count);counts=[0]*self.parameter_count
  for i,record in enumerate(trace['records'],1):
   if type(record) is not dict or type(record.get('epoch')) is not int or record['epoch']<1:raise ValueError('Invalid scheduler record epoch.')
   if self.is_batch:
    indices=record.get('updatedParameterIndices');replay.before_update(indices)
    for index in indices:counts[index]+=1
   else:replay.begin_epoch()
   actual=replay.tick(record['epoch'],record.get('batch'),record.get('updatedParameterIndices'),record.get('metricValue'))
   if not _strict_equal(actual,record):raise ValueError('Scheduler trace differs from independent rate/metric replay.')
  if self.is_batch and optimizer_updates is not None and counts!=optimizer_updates:raise ValueError('Scheduled update ownership differs from actual optimizer counters.')
  expected=replay.native.state_dict()
  if not _strict_equal(expected,payload['state']):raise ValueError('Native scheduler state differs from exact whitelisted replay.')
  for key,value in replay.context().group.items():
   if key!='params':self.optimizer.param_groups[0][key]=value
  self.native.load_state_dict(expected);self.records=list(trace['records']);self.ticks=payload['ticks'];self.buffer_indices=set(replay.buffer_indices);self.expected_rate=replay.expected_rate;self.plateau_best=replay.plateau_best;self.plateau_bad=replay.plateau_bad;self.plateau_cooldown=replay.plateau_cooldown
  self.check_curve()

def _strict_equal(a,b):
 if type(a) is not type(b):return False
 if type(a) is dict:return set(a)==set(b) and all(_strict_equal(a[k],b[k]) for k in a)
 if type(a) in (list,tuple):return len(a)==len(b) and all(_strict_equal(x,y) for x,y in zip(a,b))
 return a==b

def training_code(recipe):
 r=validate_recipe(recipe);kwargs=constructor_options(r)
 return '\n"""Scheduler ticks are separate from constructor initialization. Epoch ticks follow completed validation; OneCycle follows each real optimizer batch. Fresh state only."""\nimport math\nSCHEDULER_RECIPE = '+repr(r)+'\ndef create_scheduler(optimizer):\n    if len(optimizer.param_groups) != 1: raise ValueError("Scheduler requires one optimizer group")\n    return torch.optim.lr_scheduler.'+NAMES[r['componentID']]+'(optimizer, **'+repr(kwargs)+')\ndef _schedule_guard(scheduler, before_update=False):\n    c = SCHEDULER_RECIPE["config"]\n    name = SCHEDULER_RECIPE["componentID"]\n    tick = scheduler.last_epoch\n    limit = c.get("planned_total_batches", c.get("T_max", 10000))\n    if tick > limit or (before_update and tick >= limit): raise ValueError("Pinned schedule horizon is exhausted")\n    rate = scheduler.optimizer.param_groups[0]["lr"]\n    unused = name == "scheduler_one_cycle_lr" and tick == limit and not before_update\n    if not math.isfinite(rate) or abs(rate) > 3.4028234663852886e38 or (rate < 0 and not unused): raise ValueError("Invalid scheduled rate")\ndef training_step(loss, optimizer, scheduler):\n    _schedule_guard(scheduler, before_update=True)\n    if scheduler.optimizer is not optimizer: raise ValueError("Scheduler belongs to another optimizer")\n    optimizer.zero_grad()\n    if not torch.isfinite(loss).all().item(): raise ValueError("Nonfinite training loss")\n    loss.backward()\n    parameters = optimizer.param_groups[0]["params"]\n    if any(p.grad is not None and (p.grad.layout != torch.strided or not torch.isfinite(p.grad).all().item()) for p in parameters): raise ValueError("Nonfinite or sparse gradient")\n    optimizer.step()\n    if any(not torch.isfinite(p).all().item() for p in parameters): raise ValueError("Nonfinite learned settings")\n    if SCHEDULER_RECIPE["componentID"] == "scheduler_one_cycle_lr":\n        scheduler.step()\n        _schedule_guard(scheduler)\ndef completed_epoch_step(scheduler, *, validation_loss=None, validation_accuracy=None):\n    name = SCHEDULER_RECIPE["componentID"]\n    if name == "scheduler_one_cycle_lr": raise ValueError("OneCycle ticks after each optimizer batch, not epochs")\n    _schedule_guard(scheduler, before_update=True)\n    if name == "scheduler_reduce_on_plateau":\n        value = validation_loss if SCHEDULER_RECIPE["config"]["metric"] == "validation_loss" else validation_accuracy\n        if type(value) not in (int,float) or not math.isfinite(value): raise ValueError("The completed validation metric is required")\n        scheduler.step(value)\n    else: scheduler.step()\n    _schedule_guard(scheduler)\n'
