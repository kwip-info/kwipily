"""Selected-policy actual engine updates; plain scalar expected SGD arithmetic."""
import copy,json
import pytest,torch
from runner import engine
from runner.gradient_policy import canonical_configuration
from runner.tests.prepared_objective_fixtures import prepared_window_fixture as prepared
from runner.optimizer_configuration import canonical_configuration as optconfig
from runner.tests.test_trainability import recipe as flags,owner

def policy(k=2,scale='mean',requests=None):return dict(schemaVersion=1,configurationVersion=1,componentID='gradient_policy',config=canonical_configuration(dict(accumulation_steps=k,gradient_scale=scale,requests=requests or [])))
def optimizer():return dict(schemaVersion=1,configurationVersion=1,componentID='optimizer_sgd',config=optconfig('optimizer_sgd',dict(lr=.02,momentum=.5)))
def saved(path):return torch.load(path/'checkpoint.pt',weights_only=True)
@pytest.fixture(autouse=True)
def preserve():
 state=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled()
 yield
 torch.use_deterministic_algorithms(state,warn_only=warn)

def oracle(initial,x,y,batch,k,epochs,freeze=False):
 w=initial['layers.learn.weight'].tolist();b=initial['layers.learn.bias'].tolist();vw=[[0.]*len(w[0]) for _ in w];vb=[0.]*len(b)
 for epoch in range(epochs):
  for start in range(0,len(x),batch*k):
   xx=x[start:start+batch*k];yy=y[start:start+batch*k];dw=[[0.]*len(w[0]) for _ in w];db=[0.]*len(b)
   for row,target in zip(xx,yy):
    for o in range(len(w)):
     error=(b[o]+sum(a*z for a,z in zip(w[o],row))-target[o])*2/(len(xx)*len(w));db[o]+=error
     for i in range(len(row)):dw[o][i]+=error*row[i]
   for o in range(len(w)):
    vb[o]=.5*vb[o]+db[o];b[o]-=.02*vb[o]
    if not freeze:
     for i in range(len(w[0])):vw[o][i]=.5*vw[o][i]+dw[o][i];w[o][i]-=.02*vw[o][i]
 return w,b
@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('freeze',[False,True])
def test_real_window_updates_short_flush_and_flags_match_scalar(tmp_path,device,freeze):
 if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
 graph,r,m,data=prepared(tmp_path/'source','loss_mse');torch.manual_seed(41);initial=engine.GraphModel(graph,bundled_dataset=False,objective_context={'recipe':r,'targetMapping':m}).state_dict()
 trace=[];selected_flags=flags(owner('learn','weight')) if freeze else None
 engine.run(graph,tmp_path/'run',epochs=2,seed=41,batch_size=5,device=device,prepared=data,objective_recipe=r,target_mapping=m,optimizer_recipe=optimizer(),gradient_policy=policy(),trainability=selected_flags,gradient_observer=trace.append)
 actual=saved(tmp_path/'run');expected=oracle(initial,data[0][0].tolist(),data[0][1]['target'].tolist(),5,2,2,freeze)
 assert actual['modelState']['layers.learn.weight'].cpu().flatten().tolist()==pytest.approx([v for row in expected[0] for v in row],rel=3e-5,abs=3e-6)
 assert actual['modelState']['layers.learn.bias'].cpu().tolist()==pytest.approx(expected[1],rel=3e-5,abs=3e-6)
 manifest=json.loads((tmp_path/'run/manifest.json').read_text());assert manifest['settings']['batchesPerEpoch']==3
 assert manifest['settings']['optimizerUpdatesPerEpoch']==2 and manifest['settings']['microbatchesPerEpoch']==3
 assert actual['optimizerUpdates']==[4] if freeze else actual['optimizerUpdates']==[4,4]
 assert actual['updateSchedule']==dict(schemaVersion=1,completedEpochs=2,completedMicrobatches=6,completedOptimizerUpdates=4,pendingMicrobatches=0)
 assert [v['phase'] for v in trace]==['windowStart','beforeStep','afterStep']*4
 assert [v['rowIndices'] for v in trace if v['phase']=='windowStart']==[list(range(10)),[10,11]]*2
 assert all(v['gradientValues'] is None for t in trace if t['phase']=='windowStart' for v in t['owners'])
@pytest.mark.parametrize('kind',['mismatch','permission','allFrozen'])
def test_policy_refusal_before_training(tmp_path,kind):
 graph,r,m,data=prepared(tmp_path/'source','loss_mse');options=dict(gradient_policy=policy())
 if kind=='mismatch':options['gradient_policy']=policy(scale='sum')
 if kind=='permission':
  from runner.tests.test_requested_gradients import request
  options['gradient_policy']=policy(requests=[request(output=dict(node='learn',port='out'))])
 if kind=='allFrozen':options['trainability']=flags(owner('learn','weight'),owner('learn','bias'))
 with pytest.raises(ValueError):engine.run(graph,tmp_path/'run',epochs=1,batch_size=5,prepared=data,objective_recipe=r,target_mapping=m,**options)
 assert not (tmp_path/'run/manifest.json').exists()
def test_dedicated_discovery_matches_frozen_literal():
 from runner.gradient_policy import capability as gp
 from runner.trainability import capability as tp
 value=engine.capabilities();assert value['gradientPolicyCapabilities']=={'gradient_policy':gp()};assert value['trainabilityCapabilities']=={'trainability':tp()}

from runner.loss_configuration import IDS
@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('op',IDS)
@pytest.mark.parametrize('mode',['mean','sum','noneMean','noneSum'])
def test_actual_all_eight_window_objectives_flush_and_finite_metadata(tmp_path,device,op,mode):
 if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
 graph,r,m,data=prepared(tmp_path/'source',op,{'reduction':'none' if mode.startswith('none') else mode})
 scale='sum' if mode in ('sum','noneSum') else 'mean';records=[]
 engine.run(graph,tmp_path/'run',epochs=1,seed=41,batch_size=5,device=device,prepared=data,objective_recipe=r,target_mapping=m,objective_scalarization=scale,optimizer_recipe=optimizer(),gradient_policy=policy(scale=scale),gradient_observer=records.append)
 state=saved(tmp_path/'run');assert state['updateSchedule']['completedOptimizerUpdates']==2 and state['updateSchedule']['pendingMicrobatches']==0
 assert len(records)==6 and all(p['gradientValues'] is None for p in records[-1]['owners']) is False
 manifest=json.loads((tmp_path/'run/manifest.json').read_text());json.dumps(manifest,allow_nan=False)
 assert manifest['settings']['objectiveRecipe']==r and manifest['settings']['objectiveScalarization']==scale
 assert ('accuracy' in manifest['evaluation'])==(op in ('loss_cross_entropy','loss_nll'))

def test_observer_detached_mutation_cannot_alter_native_state_and_bound_before_clone(tmp_path,monkeypatch):
 graph,r,m,data=prepared(tmp_path/'source','loss_mse')
 def hostile(record):
  for owner in record['owners']:owner['parameterValues'][:]=[999.]*len(owner['parameterValues'])
 engine.run(graph,tmp_path/'a',epochs=1,seed=41,batch_size=5,prepared=data,objective_recipe=r,target_mapping=m,optimizer_recipe=optimizer(),gradient_policy=policy(),gradient_observer=hostile)
 engine.run(graph,tmp_path/'b',epochs=1,seed=41,batch_size=5,prepared=data,objective_recipe=r,target_mapping=m,optimizer_recipe=optimizer(),gradient_policy=policy())
 assert all(torch.equal(v,saved(tmp_path/'b')['modelState'][k]) for k,v in saved(tmp_path/'a')['modelState'].items())
 # Inject a tiny private limit so pre-clone admission is tested without a long run.
 monkeypatch.setattr(engine,'_GRADIENT_OBSERVER_ELEMENTS',4)
 with pytest.raises(ValueError,match='observer'):engine.run(graph,tmp_path/'limit',epochs=1,seed=41,batch_size=5,prepared=data,objective_recipe=r,target_mapping=m,optimizer_recipe=optimizer(),gradient_policy=policy(),gradient_observer=lambda _:None)
 assert not (tmp_path/'limit/manifest.json').exists()

def test_absent_policy_checkpoint_contract_metrics_and_generated_sources_match_accepted_baseline(tmp_path):
 import importlib.util,subprocess,sys
 from pathlib import Path
 root=Path(__file__).resolve().parents[2]
 source=(root / 'Tests/historical/ee26ff9/runner/engine.py').read_text()
 reference=root/'.local/gradient_accepted_engine_reference.py';reference.write_text(source)
 spec=importlib.util.spec_from_file_location('runner._gradient_accepted_engine_reference',reference);old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
 graph,r,m,data=prepared(tmp_path/'source','loss_mse');kwargs=dict(epochs=2,seed=41,batch_size=5,prepared=data,objective_recipe=r,target_mapping=m,optimizer_recipe=optimizer())
 old.run(graph,tmp_path/'old',**kwargs);engine.run(graph,tmp_path/'new',**kwargs)
 from runner.tests.test_checkpoint_lifecycle import assert_nested_equal
 assert_nested_equal(saved(tmp_path/'old'),saved(tmp_path/'new'))
 for name in ('metrics.json','model.py','objective_recipe.py','training_recipe.py'):assert (tmp_path/'old'/name).read_bytes()==(tmp_path/'new'/name).read_bytes()
 a=json.loads((tmp_path/'old/manifest.json').read_text());b=json.loads((tmp_path/'new/manifest.json').read_text())
 for key in ('settings','runtime','provenance','evaluation','lineage'):assert a[key]==b[key]
 assert not any((tmp_path/'new'/name).exists() for name in ('gradient_policy.py','trainability_recipe.py','derivative-results.json'))
