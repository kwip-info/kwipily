import math
import pytest
from scripts.verify_spatial_patch_exports import scalar_reference

def state(values,width):
 return {'layers.learn.weight':[[0.,0.] for _ in values],'layers.learn.bias':values,'layers.head.weight':[[1.]+[0.]*(width-1),[0.]*(width-1)+[1.]],'layers.head.bias':[1.,-1.]}

def test_hand_calculated_packed_channel_patch_coordinates():
 r=scalar_reference('unfold',state(list(range(20)),48),[0.,0.]);expected=[0,1,2,3,5,6,7,8,1,2,3,4,6,7,8,9,5,6,7,8,10,11,12,13,6,7,8,9,11,12,13,14,10,11,12,13,15,16,17,18,11,12,13,14,16,17,18,19]
 assert r['patch']==expected and r['flat']==expected;assert r['output']==[1.,18.];assert r['learn']==r['array']==list(range(20))

def test_hand_calculated_fold_adds_overlaps_without_normalizing():
 r=scalar_reference('fold',state([1.]*48,20),[0.,0.]);assert r['patch']==[1,2,2,2,1,2,4,4,4,2,2,4,4,4,2,1,2,2,2,1];assert r['output']==[2.,0.]

@pytest.mark.parametrize('op,count,width',[('unfold',20,48),('fold',48,20)])
def test_saved_projection_depends_on_both_measurements(op,count,width):
 s=state([1.]*count,width);s['layers.learn.weight'][0]=[2.,-3.];assert scalar_reference(op,s,[2.,1.])['output'][0]==3.

@pytest.mark.parametrize('op,s,x',[('unfold',state([0.],48),[0.,0.]),('fold',state([1.]*48,20),[math.nan,0.]),('unknown',{},[0.,0.])])
def test_bad_state_and_nonfinite_measurements_refuse(op,s,x):
 with pytest.raises(ValueError):scalar_reference(op,s,x)

@pytest.mark.parametrize('op',['unfold','fold'])
def test_actual_saved_portable_models_match_scalar_and_artifact_boundaries(op,tmp_path):
 import json,argparse
 from scripts.verify_spatial_patch_exports import verify,verify_graph
 from runner.tests.test_spatial_patch_runtime import graph
 from runner.engine import run,REGISTRY
 g=graph(op)
 for n in g['nodes']:
  if n['id']=='project':n['id']='learn'
 for e in g['edges']:
  for key in ('source','target'):
   if e[key]=='project':e[key]='learn'
 verify_graph(g,op)
 run(g,tmp_path,epochs=1,sample_limit=32,batch_size=8,device='cpu',registry=REGISTRY,sample_capture={'schemaVersion':1,'activationValues':True,'predictionRows':True})
 args=argparse.Namespace(component=op,package=tmp_path,artifacts=tmp_path,archive=None,training_device='cpu',inference_device='both',validation_table=None,absolute_tolerance=2e-5,relative_tolerance=2e-5)
 result=verify(args);assert result['status']=='passed' and set(result['inferenceDevices'])=={'cpu','mps'} and result['heldOutRowsCompared']>0;assert set(result['capturedPreviews'])=={'input','learn','array','patch','flat','head','output'}
 saved=(tmp_path/'activations.json').read_text();a=json.loads(saved);a['previews']['patch']['shape'][-1]+=1;(tmp_path/'activations.json').write_text(json.dumps(a))
 with pytest.raises(ValueError,match='integrity'):verify(args)
 (tmp_path/'activations.json').write_text(saved)
 saved=(tmp_path/'manifest.json').read_text();a=json.loads(saved);next(n for n in a['graph']['nodes'] if n['id']=='patch')['config']['stride']=[2,1];(tmp_path/'manifest.json').write_text(json.dumps(a))
 with pytest.raises(ValueError,match='architecture'):verify(args)

@pytest.mark.parametrize('op',['unfold','fold'])
def test_exact_graph_identity_refuses_changed_pair_or_connection(op):
 from scripts.verify_spatial_patch_exports import verify_graph
 from runner.tests.test_spatial_patch_runtime import graph
 g=graph(op)
 for n in g['nodes']:
  if n['id']=='project':n['id']='learn'
 for e in g['edges']:
  for key in ('source','target'):
   if e[key]=='project':e[key]='learn'
 assert verify_graph(g,op)==['learn','head']
 g['edges'][0]['sourcePort']='wrong'
 with pytest.raises(ValueError,match='connectivity'):verify_graph(g,op)
