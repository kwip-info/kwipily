"""Fixture-owned training-step lessons; optimizer recipes are never forward layers."""
import math
import torch
if __package__:
    from .optimizer_configuration import canonical_configuration
else:
    from optimizer_configuration import canonical_configuration
TASKS={'sgd-momentum-steps':'optimizer_sgd','adam-running-averages':'optimizer_adam','adamw-separate-decay':'optimizer_adamw','rmsprop-squared-memory':'optimizer_rmsprop','adagrad-accumulated-steps':'optimizer_adagrad'}
CLASSES={'optimizer_sgd':torch.optim.SGD,'optimizer_adam':torch.optim.Adam,'optimizer_adamw':torch.optim.AdamW,'optimizer_rmsprop':torch.optim.RMSprop,'optimizer_adagrad':torch.optim.Adagrad}
def configuration(submission,task):
 if task not in TASKS or not isinstance(submission,dict) or set(submission)!={'schemaVersion','kind','graph'} or type(submission['schemaVersion']) is not int or submission['schemaVersion']!=1 or submission['kind']!='optimizerLearningGraph':raise ValueError('Use the reviewed optimizer lesson submission; do not supply answers or state.')
 graph=submission['graph']
 if not isinstance(graph,dict) or set(graph)!={'schemaVersion','name','nodes','edges'} or type(graph['schemaVersion']) is not int or graph['schemaVersion']!=6 or not isinstance(graph['name'],str) or len(graph['name'])>256 or not isinstance(graph['nodes'],list) or len(graph['nodes'])!=2 or not isinstance(graph['edges'],list) or len(graph['edges'])!=1:raise ValueError('Connect exactly one optimizer recipe to the supplied Train context.')
 by={}
 for node in graph['nodes']:
  if not isinstance(node,dict) or set(node)!={'id','type','name','x','y','config'} or not isinstance(node['id'],str) or not 1<=len(node['id'])<=128 or node['id'] in by or not isinstance(node['name'],str) or len(node['name'])>256 or any(type(node[k]) not in (int,float) or not math.isfinite(node[k]) for k in ('x','y')):raise ValueError('Malformed optimizer lesson block.')
  by[node['id']]=node
 recipes=[n for n in by.values() if n['type']==TASKS[task]];trains=[n for n in by.values() if n['type']=='train' and n['config']=={}]
 if len(recipes)!=1 or len(trains)!=1:raise ValueError('Use the requested optimizer and an unchanged fixture-owned Train context.')
 edge=graph['edges'][0]
 if not isinstance(edge,dict) or set(edge)!={'id','source','sourcePort','target','targetPort'} or not isinstance(edge['id'],str) or not 1<=len(edge['id'])<=128 or (edge['source'],edge['sourcePort'],edge['target'],edge['targetPort'])!=(recipes[0]['id'],'out',trains[0]['id'],'optimizer'):raise ValueError('Connect optimizer out to Train optimizer; this connection carries a recipe, not a tensor.')
 return canonical_configuration(TASKS[task],recipes[0]['config'])
def native(component,c,params):
 # Assessment arithmetic uses CPU float64; production construction remains float32-only.
 kw=dict(c);kw['foreach']=False;kw['differentiable']=False
 if component in ('optimizer_adam','optimizer_adamw'):
  kw['betas']=(kw.pop('beta1'),kw.pop('beta2'));kw['capturable']=False;kw['fused']=False
 elif component in ('optimizer_sgd','optimizer_adagrad'):kw['fused']=False
 elif component=='optimizer_rmsprop':kw['capturable']=False
 return CLASSES[component](params,**kw)
def execute(submission,fixtures,evidence,stopped,tensor,*,task):
 if stopped():raise ValueError('Optimizer lesson cancelled.')
 c=configuration(submission,task);component=TASKS[task];validated=[]
 if not isinstance(fixtures,list) or not 1<=len(fixtures)<=32:raise ValueError('Bounded supplied update fixtures required.')
 for fixture in fixtures:
  if not isinstance(fixture,dict) or set(fixture)!={'id','inputs'} or set(fixture['inputs'])!={'initial','gradients','active'}:raise ValueError('Use only supplied starting settings, gradients and active-parameter schedule.')
  vals={k:tensor(v) for k,v in fixture['inputs'].items()}
  if vals['initial'][0]!=[3] or vals['gradients'][0]!=[4,3] or vals['active'][0]!=[4,2] or any(x not in (0,1) for x in vals['active'][1]) or any(not math.isfinite(x) for k,(_,v) in vals.items() for x in v):raise ValueError('Invalid supplied update shapes or parameter activity schedule.')
  validated.append((fixture['id'],vals))
 for ident,vals in validated:
  if stopped():raise ValueError('Optimizer lesson cancelled.')
  start=vals['initial'][1];params=[torch.nn.Parameter(torch.tensor(start[:2],dtype=torch.float64)),torch.nn.Parameter(torch.tensor(start[2:],dtype=torch.float64))];opt=native(component,c,params);counts=[0,0];tables={}
  def record(key,values,width):
   if not all(math.isfinite(float(x)) for x in values):raise ValueError('Optimizer produced nonfinite settings or state.')
   tables.setdefault(key,{'shape':[4,width],'values':[]})['values'].extend(values)
  def state_values(key):return [float(x) for p in params for x in opt.state.get(p,{}).get(key,torch.zeros_like(p)).reshape(-1).tolist()]
  for step in range(4):
   if stopped():raise ValueError('Optimizer lesson cancelled.')
   gradients=vals['gradients'][1][step*3:step*3+3]
   for pi,p in enumerate(params):
    enabled=vals['active'][1][step*2+pi]==1
    p.grad=torch.tensor(gradients[:2] if pi==0 else gradients[2:],dtype=torch.float64) if enabled else None
    if enabled:counts[pi]+=1
   opt.step();record('updatedParameters',[float(x) for p in params for x in p.detach().tolist()],3);record('statePresence',[int(bool(opt.state.get(p,{}))) for p in params],2);record('updateCounts',counts,2)
   if component=='optimizer_sgd':record('momentumBuffer',state_values('momentum_buffer'),3)
   else:
    record('stepCounts',[float(opt.state.get(p,{}).get('step',0)) for p in params],2)
    if component in ('optimizer_adam','optimizer_adamw'):
     for label,key in [('firstAverage','exp_avg'),('squaredAverage','exp_avg_sq'),('largestSquaredAverage','max_exp_avg_sq')]:record(label,state_values(key),3)
    elif component=='optimizer_rmsprop':
     for label,key in [('squaredAverage','square_avg'),('gradientAverage','grad_avg'),('momentumBuffer','momentum_buffer')]:record(label,state_values(key),3)
    else:record('accumulatedSquares',state_values('sum'),3)
  evidence['maximumTensorElements']=max(evidence.get('maximumTensorElements',0),max(len(t['values']) for t in tables.values()))
  evidence['fixtures'].append({'fixtureID':ident,'tensors':tables,'partitions':{}})
