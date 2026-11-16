"""Packet derivatives use whole observations and actual named source coordinates."""
import copy, hashlib
import pytest, torch
from runner import engine, requested_gradients as r
from runner.gradient_portable import gradient_policy_code
from runner.product_execution import ProductExecutionAdapter
from runner.embedding_bag_packet import EmbeddingBagPacket
from runner.tests.test_bag_source_graph import registry, graph, packet
from runner.tests.test_gradient_engine import policy
from runner.tests.test_requested_gradients import request
DEVICES=['cpu','mps']

def model_for(registry,device):
 m=engine.GraphModel(graph(),registry,bundled_dataset=False).to(device)
 with torch.no_grad():m.layers['bag'].weight.copy_(torch.tensor([[9.,9.],[1.,2.],[3.,-1.],[-2.,4.],[3.,4.]],device=device))
 return m

def q(**changes):
 return request('sum',output=dict(node='bag',port='out'),row_count=4,targets=[dict(kind='input',node='source',port='indices'),dict(kind='input',node='source',port='offsets'),dict(kind='input',node='source',port='per_sample_weights'),dict(kind='parameter',node='bag',parameter='weight')],unavailable='reportUnavailable',**changes)

@pytest.mark.parametrize('device',DEVICES)
def test_literal_packet_derivatives_and_no_pollution(registry,device):
 if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
 m=model_for(registry,device);x=packet(device);p=m.layers['bag'].weight;p.grad=torch.full_like(p,7);old=p.grad;m.train();rng=torch.get_rng_state().clone()
 v=r.analyze(m,x,q(),device=device)['targets']
 assert [t.get('cause') for t in v[:2]]==['integerInput','integerInput']
 assert v[2]['values']==pytest.approx([3.,2.,3.,0.,2.,7.,0.])
 assert v[3]['values']==pytest.approx([0.,0.,-.5,-.5,2.,2.,3.,3.,.25,.25])
 assert p.grad is old and torch.equal(old,torch.full_like(old,7)) and m.training and torch.equal(rng,torch.get_rng_state()) and x.per_sample_weights.grad is None
 # Independent finite difference of the literal first bag score.
 scalar=lambda w:3*w+2*2+3*(-1)
 assert v[2]['values'][0]==pytest.approx((scalar(.501)-scalar(.499))/.002)

@pytest.mark.parametrize('terminal',[False,True])
@pytest.mark.parametrize('device',DEVICES)
def test_artifact_whole_bag_selection_and_empty(registry,terminal,device):
 if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
 m=engine.GraphModel(graph(terminal=terminal),registry,bundled_dataset=False).to(device)
 with torch.no_grad():m.layers['bag'].weight.copy_(torch.tensor([[9.,9.],[1.,2.],[3.,-1.],[-2.,4.],[3.,4.]]))
 x=packet(device,terminal=terminal);ids=['a','b','c','d'];selected=q();selected.update(row_start=1,row_count=2)
 pins=dict(runID='saved',graphSHA256='a'*64,checkpointSHA256='b'*64,preparationManifestSHA256='c'*64,preparationGraphDigest='d'*64)
 result=r.analyze_requested_gradients(m,{'validation':x},policy(requests=[selected]),partition_row_ids={'validation':ids},source_identity=pins,device=device,derivative_capture=dict(schemaVersion=1,values=True))
 assert result['schemaVersion']==2 and result['modelInput']==ProductExecutionAdapter(m).input_contract
 item=result['requests'][0];assert item['sample']['bagItemCounts']==[0,2]
 assert item['sample']['rowIDsSHA256']==hashlib.sha256(r.encoded(ids[1:3])).hexdigest()
 assert item['targets'][1]['shape']==[2+int(terminal)] and item['targets'][2]['values']==[0.,2.]
 empty=EmbeddingBagPacket('packed',torch.empty(0,dtype=torch.int64,device=device),torch.tensor([0,0,0] if terminal else [0,0],device=device),torch.empty(0,device=device),terminal)
 req=q();req['row_count']=2
 measured=r.analyze(m,empty,req,device=device)['targets'];assert measured[2]['shape']==[0] and measured[2]['values']==[] and measured[2]['status']=='observed'

def test_preflight_refuses_role_and_source_before_forward(registry):
 m=model_for(registry,'cpu');calls=[];m.register_forward_pre_hook(lambda *args:calls.append(1));req=q();req['targets'][0]['node']='foreign'
 helper=getattr(r,'preflight_requested_gradients',None);assert callable(helper)
 with pytest.raises(ValueError):helper(m,{'validation':packet('cpu')},policy(requests=[req]),partition_row_ids={'validation':list(range(4))},device='cpu',derivative_capture=dict(schemaVersion=1,values=True))
 assert calls==[]

@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('layout',['fixed','packed'])
def test_fixed_packed_signed_upstream_and_terminal_flags(registry,device,layout):
 if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
 for terminal in (False,True):
  m=engine.GraphModel(graph(layout,True,terminal),registry,bundled_dataset=False).to(device)
  with torch.no_grad():m.layers['bag'].weight.copy_(torch.tensor([[9.,9.],[1.,2.],[3.,-1.],[-2.,4.],[3.,4.]],device=device))
  req=q();req['targets']=[t for t in req['targets'] if layout=='packed' or t.get('port')!='offsets'];req.update(reduction='upstream',upstream=dict(dtype='float32',shape=[4,2],values=[2.,-1.,7.,11.,-3.,5.,4.,6.]))
  values=r.analyze(m,packet(device,layout,True,terminal),req,device=device)['targets']
  weights=next(t for t in values if t['target'].get('port')=='per_sample_weights')
  assert weights['shape']==([4,3] if layout=='fixed' else [7])
  assert weights['values']==pytest.approx([0.,7.,0.,0.,0.,0.,0.,26.,0.,36.,0.,0.] if layout=='fixed' else [0.,7.,0.,0.,26.,36.,0.])

@pytest.mark.parametrize('bad',['source','port','capacity','flags','dtype','nonfinite','device','rowids','range','alias'])
def test_preflight_negative_geometry_and_roles_never_forward(registry,bad):
 m=model_for(registry,'cpu');x=packet('cpu');req=q();ids=list(range(4));calls=[];m.register_forward_pre_hook(lambda *args:calls.append(1))
 if bad=='source':req['targets'][0]['node']='wrong'
 elif bad=='port':req['targets'][0]['port']='out'
 elif bad=='capacity':x=EmbeddingBagPacket('packed',x.indices,torch.tensor([0,4,4,5,7]),x.per_sample_weights,True)
 elif bad=='flags':x=EmbeddingBagPacket('packed',x.indices,x.offsets,x.per_sample_weights,False)
 elif bad=='dtype':x=EmbeddingBagPacket('packed',x.indices.float(),x.offsets,x.per_sample_weights,True)
 elif bad=='nonfinite':x.per_sample_weights.detach()[0]=float('nan')
 elif bad=='device':
  if not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
  x=packet('mps')
 elif bad=='rowids':ids.pop()
 elif bad=='range':req['row_start']=2
 else:req['targets'].append(copy.deepcopy(req['targets'][0]))
 with pytest.raises(ValueError):r.preflight_requested_gradients(m,{'validation':x},policy(requests=[req]),partition_row_ids={'validation':ids},device='cpu',derivative_capture=dict(schemaVersion=1,values=True))
 assert calls==[]

def test_frozen_disconnected_recording_and_aggregate_budget(registry):
 m=model_for(registry,'cpu');m.layers['bag'].weight.requires_grad_(False);req=q();v=r.analyze(m,packet('cpu'),req,device='cpu')['targets'];assert v[-1]['cause']=='frozenParameter' and v[2]['status']=='observed'
 v=r.analyze(m,packet('cpu'),req,device='cpu',recording=False)['targets'];assert v[2]['cause']=='recordingDisabled' and v[0]['cause']=='integerInput'
 m.layers['unused']=torch.nn.Linear(1,1,bias=False);m.layer_keys['unused']='unused';req['targets'].append(dict(kind='parameter',node='unused',parameter='weight'))
 assert r.analyze(m,packet('cpu'),req,device='cpu')['targets'][-1]['cause']=='disconnected'
 # An actual native extra owner exceeds the returned element ceiling before forward.
 m.layers['large']=torch.nn.Linear(257,256,bias=False);m.layer_keys['large']='large';req['targets']=[dict(kind='parameter',node='large',parameter='weight')];calls=[];m.register_forward_pre_hook(lambda *args:calls.append(1))
 with pytest.raises(ValueError,match='65536'):r.preflight_requested_gradients(m,{'validation':packet('cpu')},policy(requests=[req]),partition_row_ids={'validation':list(range(4))},device='cpu',derivative_capture=dict(schemaVersion=1,values=True))
 assert calls==[]

def test_portable_relocated_packet_class_and_state(registry,tmp_path):
 import json,os,subprocess,sys
 m=model_for(registry,'cpu');folder=tmp_path/'relocated';folder.mkdir();(folder/'model.py').write_text(engine.export_source(m,gradient_policy_recipe=policy(requests=[q()])));(folder/'gradient_policy.py').write_text(gradient_policy_code(policy(requests=[q()]),bag_input_contract=ProductExecutionAdapter(m).input_contract));torch.save(m.state_dict(),folder/'weights.pt');(folder/'request.json').write_text(json.dumps(q()))
 program='''import torch,json,model,gradient_policy
m=model.Model();m.load_state_dict(torch.load('weights.pt',weights_only=True));m.train();p=m.layers['bag'].weight;p.grad=torch.full_like(p,8);old=p.grad;rng=torch.get_rng_state().clone()
x=model.EmbeddingBagPacket('packed',torch.tensor([1,2,1,0,3,4,0]),torch.tensor([0,3,3,5,7]),torch.tensor([.5,2.,-1.,99.,3.,.25,88.]),True)
v=gradient_policy.analyze(m,x,json.load(open('request.json')),device='cpu')['targets'];assert v[2]['values']==[3.,2.,3.,0.,2.,7.,0.];assert p.grad is old and torch.equal(old,torch.full_like(old,8)) and m.training and torch.equal(rng,torch.get_rng_state());print('relocated packet derivative passed')
'''
 env={k:v for k,v in os.environ.items() if k!='PYTHONPATH'};result=subprocess.run([sys.executable,'-I','-c',"import sys;sys.path.insert(0,'.');"+program],cwd=folder,env=env,text=True,capture_output=True)
 assert result.returncode==0,result.stdout+result.stderr

def test_ordinary_generated_helper_bytes_unchanged():
 # Baseline independently checked against all emitted legacy function bodies at HEAD.
 assert hashlib.sha256(gradient_policy_code(policy()).encode()).hexdigest()=='833e4541c0d594bb89f8e692e4d9ac0eda33e0db9b10d4be4a20ab8229ac9c27'

def test_source_coordinates_cannot_masquerade_as_model_output(registry):
 m=model_for(registry,'cpu');req=q();req['output']=dict(node='source',port='per_sample_weights');calls=[];m.register_forward_pre_hook(lambda *args:calls.append(1))
 with pytest.raises(ValueError,match='ordinary'):r.preflight_requested_gradients(m,{'validation':packet('cpu')},policy(requests=[req]),partition_row_ids={'validation':list(range(4))},device='cpu',derivative_capture=dict(schemaVersion=1,values=True))
 assert calls==[]

@pytest.mark.parametrize('key',['sourceID','ports','configuration','schemaVersion','extra'])
def test_portable_contract_strict_actual_role_configuration(registry,key):
 contract=ProductExecutionAdapter(model_for(registry,'cpu')).input_contract
 if key=='sourceID':contract[key]=''
 elif key=='ports':contract[key]['indices']['dtype']='float32'
 elif key=='configuration':contract[key]['maximum_items_per_bag']=0
 elif key=='schemaVersion':contract[key]=True
 else:contract[key]=1
 with pytest.raises(ValueError):gradient_policy_code(policy(),bag_input_contract=contract)

@pytest.mark.parametrize('fault',['upstream','output','cancel','inspection'])
def test_bag_errors_restore_exact_state_and_rng(registry,fault):
 m=model_for(registry,'cpu');x=packet('cpu');req=q();p=m.layers['bag'].weight;p.grad=torch.full_like(p,13);old=p.grad;weights=p.detach().clone();m.train();rng=torch.get_rng_state().clone();cancel=lambda:None
 if fault=='upstream':req.update(reduction='upstream',upstream=dict(dtype='float32',shape=[4,1],values=[1.]*4))
 elif fault=='output':req['output']['port']='missing'
 elif fault=='cancel':
  count=[0]
  def cancel():
   count[0]+=1
   if count[0]==2:raise ValueError('cancelled')
 else:
  forward=m.forward
  def forged(*args,**kwargs):
   out,values=forward(*args,**kwargs);values['source']['per_sample_weights']=values['source']['per_sample_weights'].clone();return out,values
  m.forward=forged
 with pytest.raises(ValueError):r.analyze(m,x,req,device='cpu',check_cancelled=cancel)
 assert p.grad is old and torch.equal(old,torch.full_like(old,13)) and torch.equal(p,weights) and m.training and torch.equal(rng,torch.get_rng_state())
