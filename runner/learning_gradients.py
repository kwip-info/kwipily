"""Real-engine gradient lessons with a separate scalar arithmetic reference."""
import copy,hashlib,json,math,tempfile,os,stat
from pathlib import Path
import torch
if __package__:
 from . import gradient_policy,trainability,optimizer_configuration
else:
 import gradient_policy,trainability,optimizer_configuration
TASKS=('gradient-selected-influence','gradient-combine-small-groups','trainability-keep-settings','trainability-next-run')
WIDTHS=((2,2,1),(3,2,1),(2,3,2),(3,3,2))

def fixed_request(input_only=False):
 targets=[dict(kind='input',node='input',port='out')]
 if not input_only:targets.append(dict(kind='parameter',node='first',parameter='weight'))
 return dict(id='input-effect' if input_only else 'selected-effect',output=dict(node='second',port='out'),partition='validation',row_start=0,row_count=3,reduction='mean',upstream=None,targets=targets,unavailable='error')
def recipe(kind,config):return dict(schemaVersion=1,configurationVersion=1,componentID=kind,config=config)
def configuration(s,task):
 exact=gradient_policy.exact
 if task not in TASKS:raise ValueError('Unknown gradient exercise.')
 exact(s,('schemaVersion','kind','taskID','graph'))
 if type(s['schemaVersion']) is not int or s['schemaVersion']!=1 or s['kind']!='gradientLearningGraph' or s['taskID']!=task:raise ValueError('Use this versioned gradient exercise, without supplied answers or state.')
 g=s['graph'];exact(g,('schemaVersion','name','nodes','edges'))
 if type(g['schemaVersion']) is not int or g['schemaVersion']!=10 or type(g['name'])is not str or len(g['name'])>256 or type(g['nodes'])is not list or len(g['nodes'])!=4 or type(g['edges'])is not list or len(g['edges'])!=3:raise ValueError('Connect the three real recipe controls to Train.')
 nodes={};types={}
 for n in g['nodes']:
  exact(n,('id','type','name','x','y','config'))
  if type(n['id'])is not str or not n['id'] or len(n['id'])>128 or n['id'] in nodes or type(n['type'])is not str or n['type'] in types or n['type']not in ('gradient_policy','trainability','optimizer_sgd','train') or type(n['name'])is not str or len(n['name'])>256 or any(type(n[k])not in(int,float)or not math.isfinite(n[k])for k in ('x','y')):raise ValueError('Use unique bounded lesson controls and finite positions.')
  nodes[n['id']]=n;types[n['type']]=n
 if types['train']['config']!={}:raise ValueError('The checker supplies the model, rows, parent and Train context.')
 gradient=recipe('gradient_policy',gradient_policy.canonical_configuration(types['gradient_policy']['config']))
 flags=recipe('trainability',trainability.canonical_configuration(types['trainability']['config']))
 gradient_policy.validate_compatibility(gradient)
 for owner in flags['config']['owners']:
  if owner['node']not in ('first','second')or owner['parameter']not in ('weight','bias'):raise ValueError('Choose actual first/second weight or bias owners.')
 for request in gradient['config']['requests']:
  if request['output']['node']not in ('first','second')or request['output']['port']!='out':raise ValueError('Choose an actual supplied layer output.')
  for target in request['targets']:
   if target['kind']=='input':
    if target['node']!='input'or target['port']!='out':raise ValueError('Choose the genuine supplied input.')
   elif target['node']not in ('first','second')or target['parameter']not in ('weight','bias'):raise ValueError('Choose a genuine supplied learned array.')
 if task in TASKS[2:]:
  expected=dict(accumulation_steps=1,gradient_scale='mean',requests=[fixed_request(True)]if task==TASKS[2]else [])
  if gradient['config']!=expected:raise ValueError('Keep the supplied influence/update policy while choosing trainability.')
 elif flags['config']!={'owners':[]}:raise ValueError('Keep the supplied trainability context while choosing gradients.')
 admitted={(n,p):True for n in ('first','second')for p in ('weight','bias')}
 for o in flags['config']['owners']:
  if o['mode']!='inherit':admitted[o['node'],o['parameter']]=o['mode']=='trainable'
 if not any(admitted.values()):raise ValueError('Leave at least one learned array able to train.')
 expected_optimizer=optimizer_configuration.canonical_configuration('optimizer_sgd',dict(lr=.05,momentum=.5 if task==TASKS[3]else 0.,dampening=0.,weight_decay=0.,nesterov=False,maximize=False))
 supplied=types['optimizer_sgd']['config']
 if type(supplied)is not dict or set(supplied)!=set(expected_optimizer)or optimizer_configuration.canonical_configuration('optimizer_sgd',supplied)!=expected_optimizer:raise ValueError('Keep the supplied SGD rate and memory controls.')
 seen=set();ports={'gradient_policy':'gradientPolicy','trainability':'trainability','optimizer_sgd':'optimizer'};ids=set()
 for e in g['edges']:
  exact(e,('id','source','sourcePort','target','targetPort'))
  if type(e['id'])is not str or not e['id']or len(e['id'])>128 or e['id']in ids or e['source']not in nodes or e['target']!=types['train']['id']or e['sourcePort']!='out'or ports.get(nodes[e['source']]['type'])!=e['targetPort']or e['targetPort']in seen:raise ValueError('Connect each recipe out to its matching Train input.')
  ids.add(e['id']);seen.add(e['targetPort'])
 return dict(gradient=gradient,trainability=flags,optimizer=recipe('optimizer_sgd',expected_optimizer))

def mean_derivatives(widths,parameters,rows):
 if len(widths)!=3 or any(type(v)is not int or not 1<=v<=3 for v in widths):raise ValueError('Use bounded affine widths.')
 i,h,o=widths;p=list(parameters);n=len(rows)
 if not 1<=n<=8 or any(len(r)!=i for r in rows)or len(p)!=h*i+h+o*h+o or any(type(v)not in(int,float)or not math.isfinite(v)or abs(v)>1e6 for v in p+sum(rows,[])):raise ValueError('Use complete finite affine arrays and rows.')
 b1=h*i;w2=b1+h;b2=w2+o*h;output=[];dx=[[0.]*i for _ in rows];dw=[[0.]*i for _ in range(h)]
 for r,x in enumerate(rows):
  hidden=[p[b1+j]+sum(p[j*i+k]*x[k]for k in range(i))for j in range(h)]
  output.append([p[b2+k]+sum(p[w2+k*h+j]*hidden[j]for j in range(h))for k in range(o)])
  for j in range(h):
   influence=sum(p[w2+k*h+j]for k in range(o))/(n*o)
   for k in range(i):dx[r][k]+=influence*p[j*i+k];dw[j][k]+=influence*x[k]
 return dict(output=output,input=dx,firstWeight=dw)

def adjustment(widths,parameters,rows,targets,rate=.05,momentum=0,memory=None,freeze_first=False):
 pred=mean_derivatives(widths,parameters,rows)['output'];i,h,o=widths;p=list(parameters);n=len(rows);b1=h*i;w2=b1+h;b2=w2+o*h
 if len(targets)!=n or any(len(r)!=o or any(type(v)not in(int,float)or not math.isfinite(v)for v in r)for r in targets)or type(rate)not in(int,float)or not math.isfinite(rate)or not 0<=rate<=1 or type(momentum)not in(int,float)or not math.isfinite(momentum)or not 0<=momentum<=1 or memory is not None and(len(memory)!=len(p)or any(not math.isfinite(v)for v in memory)):raise ValueError('Use paired targets and finite bounded SGD controls/memory.')
 g=[0.]*len(p);objective=0.
 for r,x in enumerate(rows):
  hidden=[p[b1+j]+sum(p[j*i+k]*x[k]for k in range(i))for j in range(h)];dh=[0.]*h
  for k in range(o):
   error=pred[r][k]-targets[r][k];d=2*error/(n*o);objective+=error*error/(n*o);g[b2+k]+=d
   for j in range(h):g[w2+k*h+j]+=d*hidden[j];dh[j]+=d*p[w2+k*h+j]
  for j in range(h):
   g[b1+j]+=dh[j]
   for k in range(i):g[j*i+k]+=dh[j]*x[k]
 m=[0.]*len(p)if memory is None else list(memory);out=list(p)
 for j in range(len(p)):
  if freeze_first and j<w2:continue
  m[j]=g[j]if momentum==0 or memory is None else momentum*m[j]+g[j];out[j]-=rate*m[j]
 if not all(math.isfinite(v)for v in out+g+m+[objective]):raise ValueError('Independent scalar adjustment became nonfinite.')
 return dict(parameters=out,gradients=g,memory=m if momentum else None,objective=objective)


# Real preparation/engine/loader integration is deliberately separate from scalar expectations.
def _prepared(root,inputs,seed,objective,registry):
 if __package__:
  from . import dataflow
  from .dataflow_cli import run_dataflow
  from .loss_prepared_data import load_prepared_objective
 else:
  import dataflow
  from dataflow_cli import run_dataflow
  from loss_prepared_data import load_prepared_objective
 i=len(inputs['trainingInputs'][0]);o=len(inputs['trainingTargets'][0]);columns=[f'x{k}'for k in range(i)]+[f'y{k}'for k in range(o)]
 # New checker-owned authoring keeps the five supplied Training rows and
 # three supplied Validation rows in their exact declared order. A separately
 # marked neutral Test probe satisfies current three-population preparation;
 # it is never loaded into training, validation or derivative selection. This
 # is not an upgrade or an equivalence claim for retired random Split.
 ordered=[xx+yy for xx,yy in zip(inputs['trainingInputs'],inputs['trainingTargets'])]+[xx+yy for xx,yy in zip(inputs['heldoutInputs'],inputs['heldoutTargets'])]
 if len(ordered)!=8:raise ValueError('Keep the five/three supplied gradient populations.')
 metadata=['_fixture_position','_fixture_role'];source_columns=columns+metadata
 source=root/'source.csv'
 source.write_text(','.join(source_columns)+'\n'+''.join(','.join(str(v)for v in row+[position,'training'if position<5 else 'validation'])+'\n'for position,row in enumerate(ordered))+','.join(str(v)for v in [0.]*(i+o)+[8,'reserved-test-probe'])+'\n')
 graph={'schemaVersion':1,'nodes':[{'id':'s','operator':'kwip.data.sourceCSV','config':{'separator':','}},{'id':'c','operator':'kwip.data.columns','config':{'rows':[{'source':name,'included':True,'outputName':name,'targetType':'int64'if name=='_fixture_position'else 'keep'if name=='_fixture_role'else 'float64'}for name in source_columns],'castFailure':'error','trimText':False,'newColumns':'drop','missingPolicy':'error'}},{'id':'p','operator':'kwip.data.partition','config':{'strategy':'time','timeColumn':'_fixture_position','timeEncoding':'epochMilliseconds','trainingEnd':4,'validationEnd':7,'embargoMilliseconds':0,'entityColumns':[],'entityPolicy':'allowAcrossTime'}}],'edges':[{'from':{'node':'s','port':'table'},'to':{'node':'c','port':'table'}},{'from':{'node':'c','port':'table'},'to':{'node':'p','port':'table'}}]}
 path=root/'flow.json';path.write_text(json.dumps(graph));output=root/'prepared';run_dataflow(path,{'s':str(source)},output,lambda _:None)
 mapping=dict(schemaVersion=1,componentID='loss_mse',inputColumns=columns[:i],roles=[dict(role='target',columns=columns[i:],sampleShape=[o],dtype='float32')],tripletSourceGroups=[])
 raw=(output/'manifest.json').read_bytes();data=load_prepared_objective(output/'manifest.json','p','training','p','validation',mapping,objective,expected_manifest_sha256=hashlib.sha256(raw).hexdigest(),expected_graph_digest=dataflow.digest(graph),sample_shape=[i],prediction_shape=[o])
 return mapping,data

def _model(widths):
 i,h,o=widths
 nodes=[dict(id='input',type='input',config=dict(features=i)),dict(id='first',type='linear',config=dict(in_features=i,out_features=h,bias=True)),dict(id='second',type='linear',config=dict(in_features=h,out_features=o,bias=True)),dict(id='output',type='output',config={})]
 return dict(schemaVersion=2,nodes=nodes,edges=[dict(id=str(k),source=a['id'],sourcePort='out',target=b['id'],targetPort='in')for k,(a,b)in enumerate(zip(nodes,nodes[1:]))])

def _owners(record,widths):
 i,h,o=widths;shapes={('first','weight'):[h,i],('first','bias'):[h],('second','weight'):[o,h],('second','bias'):[o]}
 if type(record)is not dict or set(record)!= {'schemaVersion','phase','epoch','window','optimizerUpdate','rowIndices','owners'}or record['schemaVersion']!=1 or type(record['owners'])is not list:raise ValueError('Missing qualified actual observer record.')
 owners={(v['node'],v['parameter']):v for v in record['owners']}
 if set(owners)!=set(shapes)or len(record['owners'])!=4:raise ValueError('Actual native owner inventory differs from the lesson model.')
 for key,shape in shapes.items():
  v=owners[key]
  if v['shape']!=shape or v['dtype']!='float32'or type(v['requiresGrad'])is not bool or type(v['optimizerMember'])is not bool or len(v['parameterValues'])!=math.prod(shape)or any(not math.isfinite(a)for a in v['parameterValues']):raise ValueError('Actual observer parameter/flags/shape contract differs.')
  grad=v['gradientValues']
  if grad is not None and(len(grad)!=math.prod(shape)or any(not math.isfinite(a)for a in grad)):raise ValueError('Actual observed gradients must remain finite and complete, with None distinct.')
 return owners

def _flatten(owners,key='parameterValues'):
 result=[]
 for ident in [('first','weight'),('first','bias'),('second','weight'),('second','bias')]:
  values=owners[ident][key]
  if values is None:raise ValueError('Missing gradient array is not a measured zero.')
  result+=values
 return result

def _trace(records,widths,epochs,steps):
 windows=(3+steps-1)//steps
 if len(records)!=epochs*windows*3:raise ValueError('Actual observer must include exactly three phases for every update.')
 result=[]
 for index in range(epochs*windows):
  epoch=index//windows+1;window=index%windows+1;start=(window-1)*steps*2;stop=min(start+steps*2,5);group=records[index*3:index*3+3]
  parsed=[]
  for record,phase in zip(group,('windowStart','beforeStep','afterStep')):
   if record['phase']!=phase or record['epoch']!=epoch or record['window']!=window or record['optimizerUpdate']!=index+1 or record['rowIndices']!=list(range(start,stop)):raise ValueError('Observer phase/update/actual prepared-row identity changed.')
   parsed.append(_owners(record,widths))
  if any(v['gradientValues']is not None for v in parsed[0].values()):raise ValueError('Window reset retained gradient buffers; None cannot be fabricated as zeros.')
  result.append((start,stop,group,parsed))
 return result

def _verify_fixtures(fixtures,tensor):
 if type(fixtures)is not list or not 1<=len(fixtures)<=4:raise ValueError('Use one to four supplied finite fixtures.')
 result=[];seen=set()
 for f in fixtures:
  gradient_policy.exact(f,('id','inputs'))
  if type(f['id'])is not str or not f['id']or f['id']in seen:raise ValueError('Use unique fixture identities.')
  seen.add(f['id']);inputs=f['inputs'];gradient_policy.exact(inputs,('trainingInputs','trainingTargets','heldoutInputs','heldoutTargets','seed'))
  raw={k:tensor(v)for k,v in inputs.items()}
  if any(len(raw[k][0])!=2 for k in ('trainingInputs','trainingTargets','heldoutInputs','heldoutTargets')):raise ValueError('Fixture array rank must match the supplied observation/sample shape.')
  i=raw['trainingInputs'][0][-1];o=raw['trainingTargets'][0][-1]
  widths=next((w for w in WIDTHS if w[0]==i and w[2]==o),None)
  if widths is None:raise ValueError('Unknown supplied model geometry.')
  shapes={'trainingInputs':[5,i],'trainingTargets':[5,o],'heldoutInputs':[3,i],'heldoutTargets':[3,o],'seed':[1]}
  for k,(shape,values)in raw.items():
   if list(shape)!=shapes[k]or len(values)!=math.prod(shape)or any(type(v)not in(int,float)or not math.isfinite(v)or k!='seed'and abs(v)>1e6 for v in values):raise ValueError('Fixture tensor shape/value differs from the bounded lesson contract.')
  seed=raw['seed'][1][0]
  if seed!=int(seed)or not 0<=seed<=2**32-1:raise ValueError('Use the supplied uint32 seed.')
  xs=raw['trainingInputs'][1]+raw['heldoutInputs'][1]
  if len({float(torch.tensor(xs[r*i],dtype=torch.float32))for r in range(8)})!=8:raise ValueError('Prepared feature rows overlap at float32 precision.')
  rows={k:[values[r:r+shapes[k][-1]]for r in range(0,len(values),shapes[k][-1])]for k,(shape,values)in raw.items()if k!='seed'}
  result.append((f['id'],widths,rows,int(seed)))
 return result

def _native_forward(widths,parameters,rows,targets=None):
 """Re-evaluate a forward at genuine observed arrays; this is not a trainer or checkpoint."""
 i,h,o=widths;b1=h*i;w2=b1+h;b2=w2+o*h
 x=torch.tensor(rows,dtype=torch.float32);a=torch.tensor(parameters[:b1],dtype=torch.float32).reshape(h,i);b=torch.tensor(parameters[b1:w2],dtype=torch.float32);c=torch.tensor(parameters[w2:b2],dtype=torch.float32).reshape(o,h);d=torch.tensor(parameters[b2:],dtype=torch.float32)
 with torch.no_grad():
  y=torch.nn.functional.linear(torch.nn.functional.linear(x,a,b),c,d)
  result={'output':y.tolist()}
  if targets is not None:result['objective']=float(torch.nn.functional.mse_loss(y,torch.tensor(targets,dtype=torch.float32)))
 return result

def _state_difference(a,b):
 if isinstance(a,torch.Tensor):
  if not isinstance(b,torch.Tensor)or a.shape!=b.shape or a.dtype!=b.dtype:raise ValueError('Native analysis state shape/type changed.')
  return float((a.to(torch.float64)-b.to(torch.float64)).abs().max())if a.numel()else 0.
 if type(a)is not type(b):raise ValueError('Native analysis state member type changed.')
 if isinstance(a,dict):
  if set(a)!=set(b):raise ValueError('Native analysis state members changed.')
  return max((_state_difference(a[k],b[k])for k in a),default=0.)
 if isinstance(a,(list,tuple)):
  if len(a)!=len(b):raise ValueError('Native analysis state length changed.')
  return max((_state_difference(x,y)for x,y in zip(a,b)),default=0.)
 if type(a)in(int,float):return abs(a-b)
 if a!=b:raise ValueError('Native analysis state identity changed.')
 return 0.

_PARENT_SNAPSHOT_MAX_BYTES=32*1024*1024

def _parent_snapshot(root):
 """Hash/compare every retained tiny-parent file, without following symlinks."""
 if root.is_symlink()or not root.is_dir():raise ValueError('Saved lesson parent must be an actual owned directory.')
 result={};total=0;pending=[root]
 while pending:
  directory=pending.pop()
  with os.scandir(directory)as entries:
   for entry in entries:
    path=Path(entry.path);info=path.lstat();relative=path.relative_to(root).as_posix()
    # Segmentation changes the file count without changing the tiny lesson data.
    # Charge paths and entries as well as contents, so empty files/directories
    # cannot turn the bounded byte snapshot into unbounded traversal or metadata.
    total+=512+len(relative.encode('utf-8'))
    if total>_PARENT_SNAPSHOT_MAX_BYTES:raise ValueError('Saved lesson parent exceeds its bounded file inventory.')
    if stat.S_ISLNK(info.st_mode):raise ValueError('Saved lesson parent symlinks are not permitted.')
    if stat.S_ISDIR(info.st_mode):pending.append(path);continue
    if not stat.S_ISREG(info.st_mode)or info.st_size>_PARENT_SNAPSHOT_MAX_BYTES-total:raise ValueError('Saved lesson parent exceeds its bounded file inventory.')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb')as stream:
     opened=os.fstat(stream.fileno())
     if (opened.st_dev,opened.st_ino)!=(info.st_dev,info.st_ino)or not stat.S_ISREG(opened.st_mode):raise ValueError('Saved lesson parent file identity changed.')
     raw=stream.read(info.st_size+1)
    if len(raw)!=info.st_size:raise ValueError('Saved lesson parent file changed while being compared.')
    total+=len(raw);result[relative]=raw
 return result

# Public dispatch; source registry is explicitly propagated in packaged layouts.
def execute(submission,fixtures,evidence,stopped,tensor,*,task,registry=None):
 if __package__:
  from . import engine,checkpoint_lifecycle,requested_gradients
  from .loss_configuration import validate_recipe as loss_recipe
 else:
  import engine,checkpoint_lifecycle,requested_gradients
  from loss_configuration import validate_recipe as loss_recipe
 controls=configuration(submission,task);validated=_verify_fixtures(fixtures,tensor)
 for ident,widths,inputs,seed in validated:
  def check():
   if stopped():raise ValueError('Gradient lesson cancelled or exceeded its budget.')
  check()
  with tempfile.TemporaryDirectory(prefix='kwip-gradient-lesson-')as temporary:
   root=Path(temporary);objective=loss_recipe(recipe('loss_mse',{}));mapping,data=_prepared(root,inputs,seed,objective,registry);graph=_model(widths)
   options=dict(registry=registry,device='cpu',seed=seed,batch_size=2,prepared=data,objective_recipe=objective,target_mapping=mapping,optimizer_recipe=controls['optimizer'],check_cancelled=check)
   if task==TASKS[3]:
    parent=root/'parent';parent_flags=recipe('trainability',dict(owners=[dict(node='first',parameter=p,mode='frozen')for p in ('weight','bias')]))
    engine.run(graph,parent,epochs=1,gradient_policy=recipe('gradient_policy',dict(accumulation_steps=1,gradient_scale='mean',requests=[])),trainability=parent_flags,**options)
    pinned=_parent_snapshot(parent);parent_manifest=json.loads(pinned['manifest.json']);saved,_,_=checkpoint_lifecycle.load(parent/'checkpoint.pt',parent_manifest['checkpoint']['contract'],'fineTune',1)
    options['finetune_checkpoint']=parent/'checkpoint.pt'
   records=[];output=root/'child';engine.run(graph,output,epochs=2 if task==TASKS[1]else 1,gradient_policy=controls['gradient'],trainability=controls['trainability'],derivative_capture=dict(schemaVersion=1,values=task in (TASKS[0],TASKS[2])),gradient_observer=records.append,**options)
   groups=_trace(records,widths,2 if task==TASKS[1]else 1,controls['gradient']['config']['accumulation_steps']);initial=_flatten(groups[0][3][0]);final=_flatten(groups[-1][3][2]);i,h,o=widths;p=len(initial);tables={}
   def record(key,values,shape):
    if len(values)!=math.prod(shape)or not all(math.isfinite(v)for v in values):raise ValueError('Measured comparison must have a complete finite shape.')
    tables[key]=dict(shape=shape,values=[float(v)for v in values])
   xs=data[0][0].tolist();ys=data[0][1]['target'].tolist();validation=data[0][2].tolist()
   expected=list(initial);memory=None;gradient_diffs=[];updates=[];objectives=[];schedule=[]
   for start,stop,raw,owners in groups:
    reference=adjustment(widths,expected,xs[start:stop],ys[start:stop],rate=.05,momentum=.5 if task==TASKS[3]else 0,memory=memory,freeze_first=task==TASKS[2]);memory=reference['memory'];expected=reference['parameters']
    actual=_flatten(owners[2]);updates += [a-b for a,b in zip(actual,expected)]
    if task==TASKS[1]:
     gradients=_flatten(owners[1],'gradientValues');gradient_diffs += [a-b for a,b in zip(gradients,reference['gradients'])]
     measured_objective=_native_forward(widths,_flatten(owners[0]),xs[start:stop],ys[start:stop])['objective'];objectives.append(measured_objective-reference['objective'])
     schedule += [raw[0]['epoch'],math.ceil((stop-start)/2),stop-start,raw[0]['optimizerUpdate']]
   if task==TASKS[1]:
    count=len(groups);record('windowSchedule',schedule,[count,4]);record('windowObjectiveDifference',objectives,[count]);record('windowGradientDifference',gradient_diffs,[count,p]);record('windowResetDifference',[0.]*(count*p),[count,p]);record('updatedSettingsDifference',updates,[count,p])
   elif task==TASKS[3]:
    saved_model=engine.GraphModel(graph,registry=registry,bundled_dataset=False,objective_context={'recipe':objective,'targetMapping':mapping});saved_model.load_state_dict(saved['modelState']);saved_owners=trainability.native_owners(saved_model);saved_params=sum([param.detach().flatten().tolist()for _,param in saved_owners],[])
    record('restoredStartingDifference',[a-b for a,b in zip(initial,saved_params)],[p]);inventory=[]
    for v in groups[0][3][0].values():inventory += [float(not v['requiresGrad']),float(not v['optimizerMember']),float(bool(v['optimizerStateKeys']))]*len(v['parameterValues'])
    record('childOwnerInventoryDifference',inventory,[p,3]);child_manifest=json.loads((output/'manifest.json').read_bytes());child,_,_=checkpoint_lifecycle.load(output/'checkpoint.pt',child_manifest['checkpoint']['contract'],'fineTune',1);state=list(child['optimizerState']['state'].values());actual_memory=sum([s['momentum_buffer'].flatten().tolist()for s in state],[])
    if len(actual_memory)!=p:raise ValueError('Newly enabled owners require complete fresh optimizer memory; no missing-state zeros.')
    record('freshMemoryDifference',[a-b for a,b in zip(actual_memory,memory)],[p]);child_model=engine.GraphModel(graph,registry=registry,bundled_dataset=False,objective_context={'recipe':objective,'targetMapping':mapping});child_model.load_state_dict(child['modelState']);child_model.eval()
    with torch.no_grad():actual_prediction=child_model(data[0][2]).tolist()
    expected_prediction=mean_derivatives(widths,expected,validation)['output'];record('childSettingsDifference',[a-b for a,b in zip(final,expected)]+[a-b for a,b in zip(sum(actual_prediction,[]),sum(expected_prediction,[]))],[p+3*o])
    current_parent=_parent_snapshot(parent)
    if set(pinned)!=set(current_parent):raise ValueError('Saved parent files changed during child execution.')
    record('parentByteDifference',[sum(sum(a!=b for a,b in zip(raw,current_parent[name]))+abs(len(raw)-len(current_parent[name]))for name,raw in pinned.items())],[1])
   else:
    manifest=json.loads((output/'manifest.json').read_bytes());checkpoint,_,_=checkpoint_lifecycle.load(output/'checkpoint.pt',manifest['checkpoint']['contract'],'fineTune',1)
    model=engine.GraphModel(graph,registry=registry,bundled_dataset=False,objective_context={'recipe':objective,'targetMapping':mapping});model.load_state_dict(checkpoint['modelState']);trainability.apply_trainability(model,controls['trainability']);model.eval()
    request=controls['gradient']['config']['requests'][0]if controls['gradient']['config']['requests']else None
    if request is None:raise ValueError('Add an actual final-result influence request before checking this exercise.')
    start=request['row_start'];stop=start+request['row_count'];source=data[0][0]if request['partition']=='training'else data[0][2];selected=source[start:stop]
    optimizer_before=copy.deepcopy(checkpoint['optimizerState']);before={n:(v.detach().clone(),v.grad,None if v.grad is None else v.grad.detach().clone())for n,v in model.named_parameters()};rng=torch.get_rng_state().clone();measured=requested_gradients.analyze(model,selected,request,device='cpu');reference=mean_derivatives(widths,final,validation)
    targets=measured['targets'];actual_input=next((t for t in targets if t['target']['kind']=='input'),None)
    if actual_input is None or actual_input['status']!='observed':raise ValueError('The input influence must be an actual observed derivative.')
    if actual_input['shape']!=[3,i]:raise ValueError('Select all three held-out observations for this comparison.')
    record('inputInfluenceDifference',[a-b for a,b in zip(actual_input['values'],sum(reference['input'],[]))],[3,i])
    if task==TASKS[0]:
     with torch.no_grad():_,values=model(selected,inspect=True)
     from_output=values[request['output']['node']];from_output=from_output['out']if isinstance(from_output,dict)else from_output
     if list(from_output.shape)!=[3,o]:raise ValueError('Select the final result with the requested shape.')
     record('selectedResultDifference',[a-b for a,b in zip(from_output.flatten().tolist(),sum(reference['output'],[]))],[3,o]);parameter=next((t for t in targets if t['target']==dict(kind='parameter',node='first',parameter='weight')),None)
     if parameter is None or parameter['status']!='observed'or parameter['shape']!=[h,i]:raise ValueError('Select the actual first.weight influence.')
     record('parameterInfluenceDifference',[a-b for a,b in zip(parameter['values'],sum(reference['firstWeight'],[]))],[h,i]);buffers=[];state_difference=0.
     for name,param in model.named_parameters():
      value,grad,grad_value=before[name]
      if (grad is None)!=(param.grad is None):raise ValueError('Requested analysis changed gradient presence; None is not zero.')
      buffers += [0.]*param.numel()if grad is None else (param.grad-grad_value).flatten().tolist();state_difference=max(state_difference,float((param.detach()-value).abs().max()))
     record('gradientBufferDifference',buffers,[p]);record('analysisStateDifference',[state_difference,float(torch.count_nonzero(rng!=torch.get_rng_state())),_state_difference(checkpoint['optimizerState'],optimizer_before)],[3])
    else:
     first=h*i+h;record('fixedSettingsDifference',[a-b for a,b in zip(final[:first],initial[:first])],[first]);record('learningSettingsDifference',[a-b for a,b in zip(final[first:],expected[first:])],[p-first]);inventory=[]
     for owner_key,v in groups[-1][3][2].items():
      active=owner_key[0]=='second';inventory += [float(v['requiresGrad']!=active),float(v['optimizerMember']!=active),float(bool(v['optimizerStateKeys']))]*len(v['parameterValues'])
     record('ownerInventoryDifference',inventory,[p,3]);zero=copy.deepcopy(request);zero['reduction']='upstream';zero['upstream']=dict(dtype='float32',shape=[3,o],values=[0.]*(3*o));z=requested_gradients.analyze(model,data[0][2],zero,device='cpu')['targets'][0]
     for param in model.parameters():param.requires_grad_(False)
     with torch.no_grad():frozen_output=model(data[0][2]).flatten().tolist()
     expected_output=sum(reference['output'],[])
     if len(frozen_output)!=len(expected_output)or any(abs(a-b)>2e-6 for a,b in zip(frozen_output,expected_output)):raise ValueError('Frozen inference changed the actual measured final result.')
     disabled=copy.deepcopy(request);disabled['unavailable']='reportUnavailable';off=requested_gradients.analyze(model,data[0][2],disabled,device='cpu',recording=False)['targets'][0]
     if z['status']!='observed'or any(v!=0 for v in z['values'])or off['status']!='unavailable'or off.get('cause')!='recordingDisabled'or 'values'in off:raise ValueError('Measured zero and recording-disabled availability must remain distinct.')
     record('recordingAvailability',[1 if any(v!=0 for v in actual_input['values'])else 0,0,-1],[3])
   evidence['maximumTensorElements']=max(evidence.get('maximumTensorElements',0),sum(len(t['values'])for t in tables.values()));check();evidence['fixtures'].append(dict(fixtureID=ident,tensors=tables,partitions={}))
