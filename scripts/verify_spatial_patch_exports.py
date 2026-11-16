"""Verify exact prepared Unfold/Fold exports using saved weights and scalar coordinates.

Expected arithmetic never imports an engine, adapter or native Unfold/Fold.
Torch reads trusted owned checkpoints and measures saved portable inference only.
"""
import argparse,importlib.util,json,math,sys
from pathlib import Path
if __package__:
 from .verify_tensor_routing_exports import affine,require,compare,sha,read
else:
 from verify_tensor_routing_exports import affine,require,compare,sha,read

def scalar_reference(operation,state,features):
 require(operation in ('unfold','fold'),'Unknown prepared spatial patch example.')
 count,width=(20,48) if operation=='unfold' else (48,20)
 values=affine(state['layers.learn.weight'],state['layers.learn.bias'],features,2,count)
 patch=[0.]*width
 # The selected examples have one channel, 4×5 output grid, 3×2 patches,
 # two patch-start rows and four columns; packing visits patch cells first.
 index=0
 for ky in range(3):
  for kx in range(2):
   for y in range(2):
    for x in range(4):
     position=(y+ky)*5+x+kx
     if operation=='unfold':patch[index]=values[position]
     else:patch[position]+=values[index]
     index+=1
 output=affine(state['layers.head.weight'],state['layers.head.bias'],patch,width,2)
 return {'input':list(features),'learn':values,'array':values,'patch':patch,'flat':patch,'head':output,'output':output}

def sample_shapes(operation):
 require(operation in ('unfold','fold'),'Unknown prepared spatial patch example.')
 return {'input':[2],'learn':[20 if operation=='unfold' else 48],'array':[1,4,5] if operation=='unfold' else [6,8],'patch':[6,8] if operation=='unfold' else [1,4,5],'flat':[48 if operation=='unfold' else 20],'head':[2],'output':[2]}

def verify_graph(graph,operation):
 shapes=sample_shapes(operation);config={'kernel_size':[3,2],'stride':[1,1],'dilation':[1,1],'padding':[0,0]}
 if operation=='fold':config['output_size']=[4,5]
 expected={'input':('input',{'features':2}),'learn':('linear',{'in_features':2,'out_features':shapes['learn'][0],'bias':True}),'array':('reshape',{'shape':shapes['array']}),'patch':(operation,config),'flat':('flatten',{'start_dim':1,'end_dim':-1}),'head':('linear',{'in_features':shapes['flat'][0],'out_features':2,'bias':True}),'output':('output',{})}
 actual={n['id']:(n['type'],n['config']) for n in graph['nodes']}
 require(graph['schemaVersion']==2 and len(actual)==len(graph['nodes']) and actual==expected,'Saved architecture/configuration differs from the independent prepared spatial patch reference.')
 ids=list(expected);wires={(a,'out',b,'in') for a,b in zip(ids,ids[1:])};edges=[(e['source'],e['sourcePort'],e['target'],e['targetPort']) for e in graph['edges']]
 require(len(edges)==len(wires) and set(edges)==wires,'Saved connectivity differs from the independent prepared reference.')
 return ['learn','head']

def verify(args):
 import torch
 package=args.package.resolve();artifacts=(args.artifacts or (args.archive/'training/artifacts' if args.archive else package)).resolve()
 manifest,predictions,activations=[read(artifacts/name) for name in ('manifest.json','predictions.json','activations.json')]
 require(manifest['runID']==predictions['runID']==activations['runID'],'Worker artifact run identities differ.')
 require(manifest['runtime']['device']==args.training_device and not manifest['runtime'].get('operationDevices'),'Unexpected training device or operator fallback.')
 for field,name in [('checkpoint','checkpoint.pt'),('predictionPreview','predictions.json'),('activationPreview','activations.json')]:
  item=manifest[field];require(item['sha256']==sha(artifacts/name) and item['bytes']==(artifacts/name).stat().st_size,'Saved artifact integrity differs: '+name)
 verify_graph(manifest['graph'],args.component);identity={}
 if args.archive:
  capture=read(args.archive/'capture.json');export=read(package/'export-manifest.json')
  require(export['runID']==capture['runID'] and export['requestIdentity']==capture['requestIdentity'] and export['sourceRevision']==capture['revision'],'Export/capture ownership differs.')
  for option,key in [('run_id','runID'),('request','requestIdentity'),('project_id','projectID'),('revision','revision')]:
   value=getattr(args,option)
   if value is not None:require(capture[key]==value,'Unexpected captured '+key)
  require(export['modelManifestSHA256']==sha(artifacts/'manifest.json'),'Export source manifest hash differs.')
  for entry in export['files']:
   p=(package/entry['path']).resolve();require(p.is_relative_to(package) and p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256'],'Export file integrity differs.')
  identity={key:capture[key] for key in ('runID','requestIdentity','projectID','revision','directoryID')}
 for name in ('model.py','checkpoint.pt'):require(sha(package/name)==sha(artifacts/name),'Export differs from saved '+name)
 checkpoint=torch.load(package/'checkpoint.pt',map_location='cpu',weights_only=True);tensors=checkpoint['modelState']
 require(set(tensors)=={'layers.'+node+'.'+key for node in ('learn','head') for key in ('weight','bias')},'Unexpected learned state or invented patch state.')
 require(checkpoint['epoch']==manifest['checkpoint']['completedEpoch'],'Checkpoint epoch differs.')
 state={key:value.tolist() for key,value in tensors.items()};rows=predictions['predictionRows'];require(0<len(rows)<=manifest['evaluation']['count'],'Captured held-out observations unavailable.')
 features=[r['input'] for r in rows];refs=[scalar_reference(args.component,state,row) for row in features];expected=[v for ref in refs for v in ref['output']]
 contract=read(package/'input-contract.json') if (package/'input-contract.json').exists() else None
 if contract:require(contract['input']['dtype']=='float32' and contract['input']['shape']==['N',2],'Unexpected portable input contract.')
 validation=args.validation_table or (args.archive/'preparation/artifacts/node-006.json' if args.archive else None)
 if validation:
  require(contract is not None,'Prepared row verification requires input contract.');table=read(validation);names=[c['name'] for c in table['columns']];indices=[names.index(c) for c in contract['input']['featureColumns']];prepared=[[row[i] for i in indices] for row in table['rows'][:len(rows)]]
  require(len(prepared)==len(rows) and torch.equal(torch.tensor(prepared,dtype=torch.float32),torch.tensor(features,dtype=torch.float32)),'Captured predictions differ from prepared held-out rows.')
  require([int(row[names.index('label')]) for row in table['rows'][:len(rows)]]==[r['label'] for r in rows],'Prepared labels differ.')
 spec=importlib.util.spec_from_file_location('saved_spatial_patch_model',package/'model.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 devices=('cpu','mps') if args.inference_device=='both' else (args.inference_device,);measurements={}
 for device in devices:
  model=module.Model().to(device);model.load_state_dict(tensors,strict=True);model.eval();x=torch.tensor(features,dtype=torch.float32,device=device)
  with torch.inference_mode():actual=model(x).cpu();singles=torch.cat([model(row[None]).cpu() for row in x])
  error=compare(actual.flatten().tolist(),expected,args.absolute_tolerance,args.relative_tolerance,device+' scalar output');batch=compare(singles.flatten().tolist(),actual.flatten().tolist(),args.absolute_tolerance,args.relative_tolerance,device+' microbatch')
  classes=[max(range(2),key=lambda k:r['output'][k]) for r in refs];require(actual.argmax(1).tolist()==classes==[r['predicted'] for r in rows],'Independent class decisions differ.')
  measurements[device]={'scalarMaximumError':error,'batchOneMaximumError':batch,'classesAllMatch':True}
 shapes=sample_shapes(args.component);require(set(activations['previews'])==set(shapes),'Captured stages differ.');captured={}
 for node,p in activations['previews'].items():
  require(p.get('captureStatus','captured')=='captured' and p['shape']==[min(8,len(rows)),*shapes[node]] and p['dtype']=='float32','Captured stage shape/type/policy differs.')
  observed=p['values'];flat=[v for r in refs for v in r[node]];captured[node]={'valuesCompared':len(observed),'scalarMaximumError':compare(observed,flat[:len(observed)],args.absolute_tolerance,args.relative_tolerance,node+' preview')}
 helper=None
 if (package/'inference.py').exists():
  scope={'__file__':str(package/'inference.py'),'__name__':'saved_spatial_patch_inference'};exec(compile((package/'inference.py').read_text(),str(package/'inference.py'),'exec'),scope);predictor=scope['load_model']();x=torch.tensor(features,dtype=torch.float32)
  full=predictor.predict(x,batch_size=len(rows));one=predictor.predict(x,batch_size=1);helper={'scalarMaximumError':compare(full.flatten().tolist(),expected,args.absolute_tolerance,args.relative_tolerance,'inference helper'),'batchOneMaximumError':compare(one.flatten().tolist(),full.flatten().tolist(),args.absolute_tolerance,args.relative_tolerance,'inference helper microbatch')}
 return {'status':'passed','component':args.component,'identity':identity,'workerRunID':manifest['runID'],'trainingDevice':manifest['runtime']['device'],'torch':torch.__version__,'checkpointEpoch':checkpoint['epoch'],'heldOutRowsCompared':len(rows),'inferenceDevices':measurements,'helper':helper,'capturedPreviews':captured,'absoluteTolerance':args.absolute_tolerance,'relativeTolerance':args.relative_tolerance,'sha256':{name:sha(package/name) for name in ('model.py','checkpoint.pt')},'manifestSHA256':sha(artifacts/'manifest.json')}

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--component',required=True,choices=('unfold','fold'));parser.add_argument('--package',required=True,type=Path);parser.add_argument('--archive',type=Path);parser.add_argument('--artifacts',type=Path);parser.add_argument('--validation-table',type=Path);parser.add_argument('--training-device',required=True,choices=('cpu','mps'));parser.add_argument('--inference-device',default='cpu',choices=('cpu','mps','both'));parser.add_argument('--run-id');parser.add_argument('--request');parser.add_argument('--project-id');parser.add_argument('--revision',type=int);parser.add_argument('--absolute-tolerance',type=float,default=2e-5);parser.add_argument('--relative-tolerance',type=float,default=2e-5);parser.add_argument('--output',type=Path);args=parser.parse_args()
 require(math.isfinite(args.absolute_tolerance) and math.isfinite(args.relative_tolerance) and args.absolute_tolerance>=0 and args.relative_tolerance>=0,'Require finite nonnegative tolerances.')
 result=verify(args);text=json.dumps(result,indent=2)+'\n'
 if args.output:args.output.write_text(text)
 print(text,end='')
if __name__=='__main__':main()
