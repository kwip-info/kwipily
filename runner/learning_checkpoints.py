"""Real engine save/reopen/restore assays; no learner paths, tensors or success flags."""
import hashlib,json,math,tempfile
from pathlib import Path
import torch
if __package__:
 from . import engine,checkpoint_lifecycle,dataflow
 from .dataflow_cli import run_dataflow
 from .loss_configuration import validate_recipe
 from .loss_prepared_data import load_prepared_objective
 from .optimizer_configuration import canonical_configuration
 from .scheduler_configuration import canonical_configuration as scheduler_config
else:
 import engine,checkpoint_lifecycle,dataflow
 from dataflow_cli import run_dataflow
 from loss_configuration import validate_recipe
 from loss_prepared_data import load_prepared_objective
 from optimizer_configuration import canonical_configuration
 from scheduler_configuration import canonical_configuration as scheduler_config
TASKS=('checkpoint-restore-weights','checkpoint-continue-training')
KEYS={'trainingInputs':(8,2),'trainingTargets':(8,2),'heldoutInputs':(4,2),'heldoutTargets':(4,2),'seed':(1,)}
def configuration(s,task):
 if task not in TASKS or type(s)is not dict or set(s)!={'schemaVersion','kind','taskID','configuration'} or type(s['schemaVersion'])is not int or s['schemaVersion']!=1 or s['kind']!='checkpointLearning' or s['taskID']!=task:raise ValueError('Use the requested checkpoint lesson controls; do not supply answers, paths or state.')
 c=s['configuration']
 if type(c)is not dict or set(c)!={'trainingMode','parent','epochs','batchSize'} or c['trainingMode']not in ('fresh','fineTune','exactResume') or c['parent']not in ('none','savedLessonRun') or any(type(c[k])is not int or not 1<=c[k]<=8 for k in ('epochs','batchSize')):raise ValueError('Choose a training mode, supplied parent, and whole-number epochs/batch size from 1 to 8.')
 if c['trainingMode']!='fresh' and c['parent']=='none':raise ValueError('Choose the saved lesson run before restoring or continuing.')
 return dict(c)
def affine_parameters(state):
 weights=[v for k,v in state.items()if k.endswith('learn.weight')];bias=[v for k,v in state.items()if k.endswith('learn.bias')]
 if len(weights)!=1 or len(bias)!=1 or weights[0].shape!=(2,2) or bias[0].shape!=(2,) or len(state)!=2:raise ValueError('The saved affine settings differ from the lesson architecture.')
 w,b=weights[0].tolist(),bias[0].tolist();return [w[0][0],w[0][1],b[0],w[1][0],w[1][1],b[1]]
def affine_update(parameters,rows,targets,batch_size=4,epochs=1):
 # Independent scalar MSE derivative, SGD momentum .5; never calls Torch backward/optimizer.
 p=list(parameters);m=[0.]*6;first=True
 for _ in range(epochs):
  for begin in range(0,len(rows),batch_size):
   bx,by=rows[begin:begin+batch_size],targets[begin:begin+batch_size];g=[0.]*6
   for x,y in zip(bx,by):
    for o in range(2):
     j=o*3;d=(p[j]*x[0]+p[j+1]*x[1]+p[j+2]-y[o])/len(bx)
     g[j]+=d*x[0];g[j+1]+=d*x[1];g[j+2]+=d
   m=g if first else [.5*a+b for a,b in zip(m,g)];first=False;p=[a-.1*b for a,b in zip(p,m)]
 return p,m
def difference(a,b):
 """Compare all members first; never hide missing states by zipping subsets."""
 if isinstance(a,torch.Tensor):
  if not isinstance(b,torch.Tensor)or a.shape!=b.shape or a.dtype!=b.dtype:raise ValueError('Restored state shape or type changed.')
  if not torch.isfinite(a).all()or not torch.isfinite(b).all():raise ValueError('Nonfinite saved state.')
  return float((a.to(torch.float64)-b.to(torch.float64)).abs().max())if a.numel()else 0.
 if type(a)is not type(b):raise ValueError('Restored state member type changed.')
 if isinstance(a,dict):
  if set(a)!=set(b):raise ValueError('Restored state members changed.')
  return max((difference(a[k],b[k])for k in a),default=0.)
 if isinstance(a,(list,tuple)):
  if len(a)!=len(b):raise ValueError('Restored state length changed.')
  return max((difference(x,y)for x,y in zip(a,b)),default=0.)
 if type(a)in(int,float):return abs(a-b)
 if a!=b:raise ValueError('Restored state identity changed.')
 return 0.
def prepared(root,inputs,seed,recipe):
 # Real current preparation with explicit checker-owned population boundaries.
 # Supplied scientific rows keep their exact order; independent protected rows
 # never enter the eight training or four validation observations.
 columns=['observation_order','x0','x1','y0','y1']
 xs=inputs['trainingInputs'][1]+inputs['heldoutInputs'][1];ys=inputs['trainingTargets'][1]+inputs['heldoutTargets'][1]
 rows=[xs[i:i+2]+ys[i:i+2]for i in range(0,24,2)]
 rows += [[100.+i,-100.-i,0.,0.]for i in range(4)]
 source=root/'source.csv';source.write_text(','.join(columns)+'\n'+''.join(','.join(str(v)for v in [position,*row])+'\n'for position,row in enumerate(rows)))
 column_config={'rows':[{'source':k,'included':True,'outputName':k,'targetType':'int64'if k=='observation_order'else'float64'}for k in columns],
                'castFailure':'error','trimText':False,'newColumns':'drop','missingPolicy':'error'}
 graph={'schemaVersion':1,'nodes':[{'id':'s','operator':'kwip.data.sourceCSV','config':{'separator':','}},{'id':'c','operator':'kwip.data.columns','config':column_config},{'id':'p','operator':'kwip.data.partition','config':{'strategy':'time','timeColumn':'observation_order','timeEncoding':'epochMilliseconds','trainingEnd':7,'validationEnd':11,'embargoMilliseconds':0,'entityColumns':[],'entityPolicy':'allowAcrossTime'}}],'edges':[{'from':{'node':'s','port':'table'},'to':{'node':'c','port':'table'}},{'from':{'node':'c','port':'table'},'to':{'node':'p','port':'table'}}]}
 path=root/'flow.json';path.write_text(json.dumps(graph));output=root/'prepared';run_dataflow(path,{'s':str(source)},output,lambda _:None)
 mapping={'schemaVersion':1,'componentID':'loss_mse','inputColumns':['x0','x1'],'roles':[{'role':'target','columns':['y0','y1'],'sampleShape':[2],'dtype':'float32'}],'tripletSourceGroups':[]}
 raw=(output/'manifest.json').read_bytes();data=load_prepared_objective(output/'manifest.json','p','training','p','validation',mapping,recipe,expected_manifest_sha256=hashlib.sha256(raw).hexdigest(),expected_graph_digest=dataflow.digest(graph),sample_shape=[2],prediction_shape=[2])
 return mapping,data
def model_graph(stochastic):
 nodes=[{'id':'input','type':'input','config':{'features':2}},{'id':'learn','type':'linear','config':{'in_features':2,'out_features':2,'bias':True}}]
 if stochastic:nodes.append({'id':'noise','type':'dropout','config':{'p':.25}})
 nodes.append({'id':'output','type':'output','config':{}})
 return {'schemaVersion':2,'nodes':nodes,'edges':[{'id':f'e{i}','source':a['id'],'sourcePort':'out','target':b['id'],'targetPort':'in'}for i,(a,b)in enumerate(zip(nodes,nodes[1:]))]}
def execute(submission,fixtures,evidence,stopped,tensor,*,task,registry=None):
 c=configuration(submission,task);validated=[]
 if type(fixtures)is not list or not 1<=len(fixtures)<=4:raise ValueError('Use one to four supplied checkpoint fixtures.')
 for f in fixtures:
  if type(f)is not dict or set(f)!={'id','inputs'} or type(f['inputs'])is not dict or set(f['inputs'])!=set(KEYS):raise ValueError('Use only the supplied rows and seed; saved artifacts are checker-owned.')
  values={k:tensor(v)for k,v in f['inputs'].items()}
  if any(tuple(shape)!=KEYS[k]or len(v)!=math.prod(shape)or any(type(x)not in(int,float)or not math.isfinite(x)for x in v)for k,(shape,v)in values.items()):raise ValueError('Checkpoint fixture shapes or values differ.')
  seed=values['seed'][1][0]
  if seed!=int(seed)or not 0<=seed<=2**32-1:raise ValueError('Invalid supplied lesson seed.')
  validated.append((f['id'],values,int(seed)))
 for ident,inputs,seed in validated:
  def check():
   if stopped():raise ValueError('Checkpoint lesson cancelled or exceeded its budget.')
  check()
  with tempfile.TemporaryDirectory(prefix='kwip-checkpoint-lesson-')as temporary:
   root=Path(temporary);recipe=validate_recipe({'schemaVersion':1,'componentID':'loss_mse','configurationVersion':1,'config':{}});mapping,data=prepared(root,inputs,seed,recipe)
   optimizer={'schemaVersion':1,'componentID':'optimizer_sgd','configurationVersion':1,'config':canonical_configuration('optimizer_sgd',{'lr':.1,'momentum':.5})}
   graph=model_graph(task==TASKS[1]);kwargs=dict(registry=registry,device='cpu',seed=seed,batch_size=4,prepared=data,objective_recipe=recipe,target_mapping=mapping,optimizer_recipe=optimizer,check_cancelled=check)
   if task==TASKS[1]:kwargs['scheduler_recipe']={'schemaVersion':1,'componentID':'scheduler_step_lr','configurationVersion':1,'config':scheduler_config('scheduler_step_lr',{'step_size':1,'gamma':.5})}
   parent=root/'parent';engine.run(graph,parent,epochs=2,**kwargs)
   pinned={p.name:p.read_bytes()for p in parent.iterdir()if p.is_file()};manifest=json.loads(pinned['manifest.json']);saved,_,_=checkpoint_lifecycle.load(parent/'checkpoint.pt',manifest['checkpoint']['contract'],'fineTune',1)
   p=affine_parameters(saved['modelState']);tables={}
   def record(key,values,shape=None):
    values=[float(v)for v in values]
    if not all(math.isfinite(v)for v in values):raise ValueError('Nonfinite measured lifecycle comparison.')
    tables[key]={'shape':shape or [len(values)],'values':values}
   record('savedParameters',p,[2,3])
   child_kwargs={**kwargs,'batch_size':c['batchSize']}
   if c['trainingMode']=='fineTune':child_kwargs['finetune_checkpoint']=parent/'checkpoint.pt'
   elif c['trainingMode']=='exactResume':child_kwargs['resume_checkpoint']=parent/'checkpoint.pt'
   child=root/'child'
   if task==TASKS[0]:
    context={'recipe':recipe,'targetMapping':mapping};restored=engine.GraphModel(graph,registry=registry,bundled_dataset=False,objective_context=context);restored.load_state_dict(saved['modelState']);restored.eval()
    rp=affine_parameters(restored.state_dict());record('restoredParameterDifference',[a-b for a,b in zip(rp,p)],[2,3])
    x=torch.tensor(inputs['heldoutInputs'][1],dtype=torch.float32).reshape(4,2)
    with torch.no_grad():measured=restored(x).reshape(-1).tolist()
    expected=[p[o*3]*row[0]+p[o*3+1]*row[1]+p[o*3+2]for row in x.tolist()for o in range(2)]
    record('restoredPredictionDifference',[a-b for a,b in zip(measured,expected)],[4,2]);record('restoredPredictions',measured,[4,2])
    expected_parameters,memory=affine_update(p,data[0][0].tolist(),data[0][1]['target'].tolist())
    engine.run(graph,child,epochs=c['epochs'],**child_kwargs)
    observed=torch.load(child/'checkpoint.pt',map_location='cpu',weights_only=True);actual=affine_parameters(observed['modelState']);record('childUpdateDifference',[a-b for a,b in zip(actual,expected_parameters)],[2,3]);record('updatedParameters',actual,[2,3])
    states=list(observed['optimizerState']['state'].values());w,b=[s['momentum_buffer'].reshape(-1).tolist()for s in states];observed_memory=[w[0],w[1],b[0],w[2],w[3],b[1]]
    record('freshMemoryDifference',[a-b for a,b in zip(observed_memory,memory)],[2,3]);record('completedEpochs',[saved['epoch'],observed['epoch']])
   else:
    full=root/'full';engine.run(graph,full,epochs=4,**kwargs);events=[];engine.run(graph,child,epochs=c['epochs'],emit=events.append,**child_kwargs)
    started=next(e for e in events if e['event']=='started');epochs=[e['epoch'] for e in events if e['event']=='epoch'];record('continuationEpochs',[started['initialEpoch'],epochs[0],epochs[-1]])
    a=torch.load(full/'checkpoint.pt',map_location='cpu',weights_only=True);b=torch.load(child/'checkpoint.pt',map_location='cpu',weights_only=True)
    keys=('modelState','optimizerState','optimizerUpdates','scheduler','objectiveState','epoch','seed');record('continuationStateDifference',[max(difference(a[k],b[k])for k in keys)])
    record('continuationHistoryDifference',[difference(json.loads((full/'metrics.json').read_bytes()),json.loads((child/'metrics.json').read_bytes()))]);record('continuationRandomDifference',[difference(a['rngState'],b['rngState'])]);record('resumedParameters',affine_parameters(b['modelState']),[2,3]);record('completedEpochs',[saved['epoch'],a['epoch'],b['epoch']])
   record('parentByteDifference',[sum(sum(x!=y for x,y in zip(raw,(parent/name).read_bytes()))+abs(len(raw)-len((parent/name).read_bytes()))for name,raw in pinned.items())])
   evidence['maximumTensorElements']=max(evidence.get('maximumTensorElements',0),sum(len(t['values'])for t in tables.values()),saved['rngState'].numel());check();evidence['fixtures'].append({'fixtureID':ident,'tensors':tables,'partitions':{}})
