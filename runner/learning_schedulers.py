"""Measured fixture-owned schedule lessons, separate from production float32 training."""
import copy
import math
import torch
if __package__:
 from .scheduler_configuration import canonical_configuration,constructor_options,validate_compatibility,NAMES,MAXIMUM
 from .optimizer_configuration import canonical_configuration as optimizer_configuration
else:
 from scheduler_configuration import canonical_configuration,constructor_options,validate_compatibility,NAMES,MAXIMUM
 from optimizer_configuration import canonical_configuration as optimizer_configuration
TASKS={'step-lr-intervals':'scheduler_step_lr','cosine-lr-curve':'scheduler_cosine_annealing_lr','linear-lr-warmup':'scheduler_linear_lr','plateau-lr-improvement':'scheduler_reduce_on_plateau','one-cycle-lr-batches':'scheduler_one_cycle_lr'}
SGD=dict(lr=.1,momentum=.5,dampening=0.,weight_decay=0.,nesterov=False,maximize=False)
def configuration(submission,task):
 if task not in TASKS or type(submission) is not dict or set(submission)!={'schemaVersion','kind','graph'} or type(submission['schemaVersion']) is not int or submission['schemaVersion']!=1 or submission['kind']!='schedulerLearningGraph':raise ValueError('Use the scheduler lesson graph; do not supply answers or saved state.')
 graph=submission['graph']
 if type(graph) is not dict or set(graph)!={'schemaVersion','name','nodes','edges'} or type(graph['schemaVersion']) is not int or graph['schemaVersion']!=7 or type(graph['name']) is not str or len(graph['name'])>256 or type(graph['nodes']) is not list or len(graph['nodes'])!=3 or type(graph['edges']) is not list or len(graph['edges'])!=2:raise ValueError('Connect the scheduler and supplied SGD to Train.')
 by={}
 for n in graph['nodes']:
  if type(n) is not dict or set(n)!={'id','type','name','x','y','config'} or type(n['id']) is not str or not 1<=len(n['id'])<=128 or n['id'] in by or type(n['name']) is not str or len(n['name'])>256 or any(type(n[k]) not in (int,float) or not math.isfinite(n[k]) for k in ('x','y')):raise ValueError('Malformed scheduler lesson block.')
  by[n['id']]=n
 sources=[n for n in by.values() if n['type']==TASKS[task]];optimizers=[n for n in by.values() if n['type']=='optimizer_sgd'];trains=[n for n in by.values() if n['type']=='train' and n['config']=={}]
 if len(sources)!=1 or len(optimizers)!=1 or len(trains)!=1:raise ValueError('Keep one requested scheduler, supplied SGD and unchanged Train context.')
 oc=optimizers[0]['config']
 if type(oc) is not dict or set(oc)!=set(SGD) or optimizer_configuration('optimizer_sgd',oc)!=SGD:raise ValueError('Keep the public supplied SGD controls unchanged: rate0.1 and momentum0.5.')
 expected={(sources[0]['id'],'out',trains[0]['id'],'scheduler'),(optimizers[0]['id'],'out',trains[0]['id'],'optimizer')};actual=set();ids=set()
 for e in graph['edges']:
  if type(e) is not dict or set(e)!={'id','source','sourcePort','target','targetPort'} or type(e['id']) is not str or not 1<=len(e['id'])<=128 or e['id'] in ids:raise ValueError('Malformed scheduler lesson wire.')
  ids.add(e['id']);actual.add((e['source'],e['sourcePort'],e['target'],e['targetPort']))
 if actual!=expected:raise ValueError('Connect scheduler out to Train scheduler and SGD out to Train optimizer.')
 recipe=dict(schemaVersion=1,componentID=TASKS[task],configurationVersion=1,config=canonical_configuration(TASKS[task],sources[0]['config']))
 validate_compatibility(recipe,dict(schemaVersion=1,componentID='optimizer_sgd',configurationVersion=1,config=SGD))
 c=recipe['config']
 if (TASKS[task]=='scheduler_cosine_annealing_lr' and c['T_max']<4) or (TASKS[task]=='scheduler_one_cycle_lr' and c['planned_total_batches']<8):raise ValueError('The supplied sequence must fit the scheduler horizon.')
 return recipe

def execute(submission,fixtures,evidence,stopped,tensor,*,task):
 if stopped():raise ValueError('Scheduler lesson cancelled.')
 recipe=configuration(submission,task);validated=[]
 if type(fixtures) is not list or not 1<=len(fixtures)<=32:raise ValueError('Use bounded supplied schedule fixtures.')
 shapes={'initial':[3],'gradients':[8,3],'active':[8,2],'epochOfBatch':[8],'metrics':[4]}
 for f in fixtures:
  if type(f) is not dict or set(f)!={'id','inputs'} or type(f['id']) is not str or not 1<=len(f['id'])<=128 or type(f['inputs']) is not dict or set(f['inputs'])!=set(shapes):raise ValueError('Use supplied initial settings, gradients, activity, epoch positions and validation measurements only.')
  vals={k:tensor(v) for k,v in f['inputs'].items()}
  if any(s!=shapes[k] or len(v)!=math.prod(shapes[k]) or any(type(x) not in (int,float) or not math.isfinite(x) for x in v) for k,(s,v) in vals.items()) or any(x not in (0,1) for x in vals['active'][1]) or vals['epochOfBatch'][1]!=[1,1,2,2,3,3,4,4]:raise ValueError('Invalid supplied eight-batch/four-epoch fixture; no native state has been allocated.')
  validated.append((f['id'],vals))
 for ident,vals in validated:
  if stopped():raise ValueError('Scheduler lesson cancelled.')
  def construct(start):
   params=[torch.nn.Parameter(torch.tensor(start[:2],dtype=torch.float64)),torch.nn.Parameter(torch.tensor(start[2:],dtype=torch.float64))]
   opt=torch.optim.SGD(params,**SGD,foreach=False,fused=False,differentiable=False)
   schedule=getattr(torch.optim.lr_scheduler,NAMES[recipe['componentID']])(opt,**constructor_options(recipe))
   return params,opt,schedule
  params,opt,schedule=construct(vals['initial'][1]);tables={};saved=None
  def record(key,v,rows,width):
   if any(not math.isfinite(float(x)) or abs(float(x))>MAXIMUM for x in v):raise ValueError('Schedule produced unsafe rates, settings or remembered quantities.')
   tables.setdefault(key,{'shape':[rows,width],'values':[]})['values'].extend(float(x) for x in v)
  def run_batch(index,params,opt,schedule,resumed=False):
   if stopped():raise ValueError('Scheduler lesson cancelled.')
   used=float(opt.param_groups[0]['lr'])
   if not 0<=used<=MAXIMUM:raise ValueError('A usable scheduled learning rate must be nonnegative finite.')
   for pi,p in enumerate(params):
    values=vals['gradients'][1][index*3:index*3+2] if pi==0 else vals['gradients'][1][index*3+2:index*3+3]
    p.grad=torch.tensor(values,dtype=torch.float64) if vals['active'][1][index*2+pi] else None
   opt.step()
   if recipe['componentID']=='scheduler_one_cycle_lr':schedule.step()
   elif index%2==1:
    if recipe['componentID']=='scheduler_reduce_on_plateau':schedule.step(vals['metrics'][1][index//2])
    else:schedule.step()
   next_rate=float(opt.param_groups[0]['lr'])
   if next_rate<0 and not (recipe['componentID']=='scheduler_one_cycle_lr' and schedule.last_epoch==recipe['config']['planned_total_batches']):raise ValueError('Negative next rate is permitted only as an unused exhausted OneCycle value.')
   prefix='resumed' if resumed else '';rows=4 if resumed else 8
   mapping={'usedRates':'resumedUsedRates','nextRates':'resumedNextRates','scheduleTicks':'resumedTicks','updatedParameters':'resumedParameters','momentumBuffer':'resumedMomentum'}
   values={'usedRates':[used],'nextRates':[next_rate],'scheduleTicks':[schedule.last_epoch],'updatedParameters':[x for p in params for x in p.detach().tolist()],'momentumBuffer':[x for p in params for x in opt.state.get(p,{}).get('momentum_buffer',torch.zeros_like(p)).tolist()]}
   for k,v in values.items():record(mapping[k] if resumed else k,v,rows,3 if k in ('updatedParameters','momentumBuffer') else 1)
  for index in range(8):
   run_batch(index,params,opt,schedule)
   if index==3:saved=([x for p in params for x in p.detach().tolist()],copy.deepcopy(opt.state_dict()),copy.deepcopy(schedule.state_dict()))
  # Reconstruct actual native objects; load saved options only after constructor initialization.
  resumed_params,resumed_opt,resumed_schedule=construct(saved[0]);resumed_opt.load_state_dict(saved[1]);resumed_schedule.load_state_dict(saved[2])
  for index in range(4,8):run_batch(index,resumed_params,resumed_opt,resumed_schedule,True)
  evidence['maximumTensorElements']=max(evidence.get('maximumTensorElements',0),24)
  evidence['fixtures'].append({'fixtureID':ident,'tensors':tables,'partitions':{}})
