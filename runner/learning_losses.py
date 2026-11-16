"""Checker-owned CPUfloat64 objective measurements; no submitted answers or code."""
import math
import torch
try:
 from .loss_configuration import validate_recipe,resolve_objective
 from .loss_objectives import create_objective,evaluate_objective
 from .tensor_layout import BatchedLayout
except ImportError:
 from loss_configuration import validate_recipe,resolve_objective
 from loss_objectives import create_objective,evaluate_objective
 from tensor_layout import BatchedLayout
TASKS={'cross-entropy-class-choice':'loss_cross_entropy','bce-independent-labels':'loss_bce_with_logits','mse-error-size':'loss_mse','l1-absolute-error':'loss_l1','nll-log-probability-choice':'loss_nll','kl-distribution-comparison':'loss_kl_div','ctc-sequence-alignment':'loss_ctc','triplet-relative-distance':'loss_triplet_margin'}
def source_roles(op):return ['prediction'] if op=='loss_triplet_margin' else ['prediction','target','input_lengths','target_lengths'] if op=='loss_ctc' else ['prediction','target','weight'] if op=='loss_bce_with_logits' else ['prediction','target']
def configuration(submission,task):
 if task not in TASKS or type(submission) is not dict or set(submission)!={'schemaVersion','kind','graph','objectiveSource','inputSources'} or type(submission['schemaVersion']) is not int or submission['schemaVersion']!=1 or submission['kind']!='lossObjectiveGraph':raise ValueError('Use the objective lesson graph and its actual qualified sources; answers belong to the checker.')
 op=TASKS[task];g=submission['graph']
 if type(g) is not dict or set(g)!={'schemaVersion','name','nodes','edges'} or type(g['schemaVersion']) is not int or g['schemaVersion']!=2 or type(g['name']) is not str or len(g['name'])>256 or type(g['nodes']) is not list or not 1<=len(g['nodes'])<=12 or type(g['edges']) is not list or len(g['edges'])>12:raise ValueError('Use the bounded objective blocks without production training settings.')
 inputs={};outputs={};nodes={};recipe_node=None;check=None;normal=None;uses_log=op in ('loss_nll','loss_kl_div','loss_ctc')
 for node in g['nodes']:
  if type(node) is not dict or set(node)!={'id','type','name','x','y','config'} or type(node['id']) is not str or not 1<=len(node['id'])<=128 or node['id'] in nodes or type(node['name']) is not str or len(node['name'])>256 or any(type(node[k]) not in (int,float) or not math.isfinite(node[k]) for k in ('x','y')) or type(node['config']) is not dict:raise ValueError('Use unique bounded objective block identities, configuration and finite positions.')
  nodes[node['id']]=node;c=node['config']
  if node['type']=='input':
   if set(c)!={'fixture'} or type(c['fixture']) is not str or c['fixture'] not in source_roles(op) or c['fixture'] in inputs:raise ValueError('Use each supplied prediction and typed target role once.')
   inputs[c['fixture']]=node['id']
  elif node['type']=='output':
   if set(c)!={'role'} or type(c['role']) is not str or c['role'] not in ('raw','objective','inputInfluence') or c['role'] in outputs:raise ValueError('Connect all three distinct objective result Outputs.')
   outputs[c['role']]=node['id']
  elif node['type']==op and recipe_node is None:recipe_node=node
  elif node['type']=='objective_check' and check is None:
   if set(c)!={'scalarization'} or c['scalarization'] not in ('mean','sum'):raise ValueError('Choose Mean or Sum to combine kept scores.')
   check=node
  elif node['type']=='log_softmax' and uses_log and normal is None:
   if set(c)!={'dim'} or type(c['dim']) not in (int,float) or c['dim']!=-1:raise ValueError('Prepare probabilities visibly with LogSoftmax direction −1.')
   normal=node
  else:raise ValueError('Use only the requested recipe, supplied inputs, objective check and visible probability preparation.')
 if recipe_node is None or check is None or set(inputs)!=set(source_roles(op)) or set(outputs)!={'raw','objective','inputInfluence'} or uses_log and normal is None:raise ValueError('Complete all supplied roles, recipe and objective Outputs.')
 recipe=validate_recipe(dict(schemaVersion=1,componentID=op,configurationVersion=1,config=recipe_node['config']))
 if op=='loss_bce_with_logits' and (recipe['config']['weight_mode']!='mapped' or recipe['config']['pos_weight_mode']=='mapped'):raise ValueError('This task supplies mapped element importance and configured or disabled positive-label importance.')
 if op=='loss_cross_entropy' and recipe['config']['target_mode']!='class_indices':raise ValueError('The supplied desired class choices are integer positions.')
 actual=set();ids=set();targets=set();expected={(recipe_node['id'],'out',check['id'],'objective')}|{(check['id'],r,i,'in') for r,i in outputs.items()}
 for r,i in inputs.items():
  if r=='prediction' and uses_log:expected|={(i,'out',normal['id'],'in'),(normal['id'],'out',check['id'],r)}
  else:expected.add((i,'out',check['id'],r))
 for edge in g['edges']:
  if type(edge) is not dict or set(edge)!={'id','source','sourcePort','target','targetPort'} or type(edge['id']) is not str or not 1<=len(edge['id'])<=128 or edge['id'] in ids or any(type(edge[k]) is not str for k in ('source','sourcePort','target','targetPort')):raise ValueError('Malformed objective connection.')
  ids.add(edge['id']);target=(edge['target'],edge['targetPort'])
  if target in targets:raise ValueError('Each named check port accepts one connection.')
  targets.add(target);actual.add(tuple(edge[k] for k in ('source','sourcePort','target','targetPort')))
 if actual!=expected or len(actual)!=len(g['edges']):raise ValueError('Connect every actual recipe/input/result to its matching named port.')
 if submission['objectiveSource']!={'domain':'control','localID':recipe_node['id']} or submission['inputSources']!={r:{'domain':'model','localID':i} for r,i in inputs.items()}:raise ValueError('Qualified sources must identify the actual connected control and model blocks.')
 return op,recipe,check['config']['scalarization'],uses_log

def execute(submission,fixtures,evidence,stopped,tensor,*,task):
 if stopped():raise ValueError('Objective lesson cancelled.')
 op,recipe,scalarization,uses_log=configuration(submission,task)
 if type(fixtures) is not list or not 1<=len(fixtures)<=32:raise ValueError('Use one to32 bounded checker-owned fixture tables.')
 checked=[];ids=set();integer_roles=set(source_roles(op)[1:]) if op in ('loss_cross_entropy','loss_nll','loss_ctc') else set()
 for fixture in fixtures:
  if type(fixture) is not dict or set(fixture)!={'id','inputs'} or type(fixture['id']) is not str or not 1<=len(fixture['id'])<=128 or fixture['id'] in ids or type(fixture['inputs']) is not dict or set(fixture['inputs'])!=set(source_roles(op)):raise ValueError('Use exactly the checker-owned prediction and typed desired roles.')
  ids.add(fixture['id']);values={}
  for role,item in fixture['inputs'].items():
   shape,flat=tensor(item)
   if type(shape) not in (list,tuple) or not shape or any(type(d) is not int or not 1<=d<=6 for d in shape) or math.prod(shape)>72 or type(flat) not in (list,tuple) or len(flat)!=math.prod(shape) or any(type(v) not in (int,float) or not math.isfinite(v) for v in flat):raise ValueError('Use finite supplied arrays with at most72 values.')
   if role in integer_roles and any(v!=math.trunc(v) or not -(2**63)<=v<2**63 for v in flat):raise ValueError('Desired class choices and sequence lengths require exact int64 values.')
   values[role]=(list(shape),list(flat))
  shape=values['prediction'][0];n=shape[0];sample=shape[1:]
  if n not in (1,2,3) or any(v[0][0]!=n for v in values.values()):raise ValueError('Preserve one to three independent observations in every supplied role.')
  if op=='loss_ctc':valid=len(sample)==2 and 1<=sample[0]<=6 and 2<=sample[1]<=4 and len(values['target'][0])==2 and 1<=values['target'][0][1]<=3
  elif op=='loss_triplet_margin':valid=len(sample)==2 and sample[0]==3 and 1<=sample[1]<=4
  else:valid=len(sample)==1 and 1<=sample[0]<=4
  if not valid:raise ValueError('Use the bounded task prediction directions and target sequence shape.')
  resolve_objective(op,recipe['config'],BatchedLayout(sample),{r:{'dtype':'int64' if r in integer_roles else 'float32','sampleShape':s[1:]} for r,(s,v) in values.items() if r!='prediction'})
  checked.append((fixture['id'],values))
 for ident,values in checked:
  if stopped():raise ValueError('Objective lesson cancelled.')
  x=torch.tensor(values['prediction'][1],dtype=torch.float64).reshape(values['prediction'][0]).requires_grad_(True)
  role_tensors={r:torch.tensor(v,dtype=torch.int64 if r in integer_roles else torch.float64).reshape(s) for r,(s,v) in values.items() if r!='prediction'}
  prediction=x.log_softmax(-1) if uses_log else x
  objective=create_objective(recipe,'cpu',torch.float64).train()
  train=evaluate_objective(objective,prediction,role_tensors,scalarization)
  objective.eval();evaluation=evaluate_objective(objective,prediction,role_tensors,scalarization)
  raw_recipe={**recipe,'config':{**recipe['config'],'reduction':'none'}}
  raw=evaluate_objective(create_objective(raw_recipe,'cpu',torch.float64),prediction,role_tensors,scalarization)['raw']
  influence=torch.autograd.grad(train['scalar'],x)[0]
  format=list(x.shape)+([0,1,2] if op=='loss_triplet_margin' else [])
  for r in source_roles(op)[1:]:s,v=values[r];format += [1 if r in integer_roles else 0,len(s)]+s+v
  measured={'trainingObjective':train['scalar'].reshape(1),'evaluationObjective':evaluation['scalar'].reshape(1),'targetContract':torch.tensor(format,dtype=torch.float64),'reduction':raw,'inputInfluence':influence}
  if any(not torch.isfinite(v).all().item() for v in measured.values()):raise ValueError('Objective score, target contract or input influence is nonfinite.')
  evidence['maximumTensorElements']=max(evidence.get('maximumTensorElements',0),max(v.numel() for v in measured.values()))
  evidence['fixtures'].append(dict(fixtureID=ident,tensors={k:dict(shape=list(v.shape),values=v.detach().reshape(-1).tolist()) for k,v in measured.items()},partitions={}))
