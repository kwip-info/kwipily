import copy,random,threading,concurrent.futures
import pytest,torch
from runner.self_play_acceleration import CachedRules,BoundedCache,InferenceQueue
from runner.self_play_model import PolicyValueModel
from runner.self_play_search import Search,SearchSettings
from runner.game_rules import Definition
from runner.game_rule_examples import take_away,board_game
from runner.tests.test_self_play import paired_graph,tiny_settings
from runner.self_play_training import train

@pytest.mark.parametrize('rules,n,a',[(take_away(),1,3),(board_game(3,3,3,False),9,9),(board_game(6,7,4,True),42,7)])
def test_cached_rule_equivalence_and_tree_visits(rules,n,a):
 d=Definition(rules);cached=CachedRules(Definition(rules),100000);s=d.initial()
 for turn in range(5):
  assert cached.outcome(s)==d.outcome(s)
  if d.outcome(s)['terminal']:break
  assert cached.legal_actions(s)==d.legal_actions(s)
  action=d.legal_actions(s)[0];expected=d.step(s,action);actual=cached.step(s,action);assert actual==expected
  actual['state'].clear();assert cached.step(s,action)==expected
  s=expected['state']
 model=PolicyValueModel(paired_graph(n,a));solver=Search(cached,model,SearchSettings(simulations=8,reuseTree=True));state=d.initial()
 for turn in range(3):
  result=solver.run(state,random.Random(turn),noise=True);assert sum(result['visits'])==8
  action=solver.catalog[result['actionIndex']];state=d.step(state,action)['state'];solver.advance(result['actionIndex'],state)
  if d.outcome(state)['terminal']:break
 assert cached.cache.bytes<=100000

def test_batched_inference_is_bounded_matches_scalar_and_closes():
 model=PolicyValueModel(paired_graph());observations=[[.1],[.2],[.3],[.4]]
 expected=[model.predict(x) for x in observations]
 q=InferenceQueue(model,batch_size=4,wait_seconds=.02,cache_bytes=1000)
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:actual=list(pool.map(q.predict,observations))
  for (p,v),(pp,vv) in zip(expected,actual):assert p==pytest.approx(pp,abs=1e-6) and v==pytest.approx(vv,abs=1e-6)
  assert q.calls<4
  assert q.predict(observations[0])==actual[0] and q.cache.hits>=1
 finally:q.close()
 assert not q.thread.is_alive()

@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='Actual Apple GPU required')
def test_real_mps_joint_updates_and_campaign_recovery(tmp_path):
 from runner.self_play_campaign import run_slice
 c=tiny_settings();c.update(device='mps',compute={'activeGames':2,'inferenceDevice':'mps','batchSize':8,'batchWait':.002,'cacheBytes':100000},campaign={'enabled':True,'sliceIterations':1,'maximumSeconds':600,'maximumBytes':268435456})
 first=run_slice(tmp_path/'gpu',take_away(),paired_graph(),c,action='start')
 assert first['devices']['learning']=='mps' and first['devices']['qualified']
 second=run_slice(tmp_path/'gpu',take_away(),paired_graph(),c,action='resume')
 assert second['campaign']['complete'] and first['weightsSHA256']!=second['weightsSHA256']
 assert second['campaign']['reproducibility'].startswith('Accelerated')

@pytest.mark.parametrize('device',['cpu','mps'])
def test_uninterrupted_and_sliced_campaign_model_recovery(tmp_path,device):
 if device=='mps' and not torch.backends.mps.is_available():pytest.skip('Actual Apple GPU required')
 from runner.self_play_campaign import run_slice
 from runner.self_play_replay import ReplayStore
 c=tiny_settings();c.update(device=device,campaign=dict(enabled=True,sliceIterations=1,maximumSeconds=600,maximumBytes=268435456))
 if device=='mps':c['compute']=dict(activeGames=2,inferenceDevice='mps',batchSize=8,batchWait=.002,cacheBytes=100000)
 run_slice(tmp_path/'sliced',take_away(),paired_graph(),c,action='start');run_slice(tmp_path/'sliced',take_away(),paired_graph(),c,action='resume')
 whole=copy.deepcopy(c);whole['campaign']['sliceIterations']=2;run_slice(tmp_path/'whole',take_away(),paired_graph(),whole,action='start')
 def weights(root):
  path,_=ReplayStore(root/'replay').checkpoint();return torch.load(path/'model.pt',weights_only=True,map_location='cpu')
 left=weights(tmp_path/'sliced');right=weights(tmp_path/'whole')
 for name in left:
  if device=='cpu':assert torch.equal(left[name],right[name]),name
  else:assert torch.allclose(left[name],right[name],atol=1e-4,rtol=1e-3),name

def test_automatic_cpu_does_not_probe_an_unselected_gpu(monkeypatch):
 from runner.self_play_devices import qualify_devices
 from runner.self_play_training import TrainingSettings
 def unexpected():raise AssertionError('Automatic CPU must not probe GPU support')
 monkeypatch.setattr(torch.backends.mps,'is_available',unexpected)
 c=TrainingSettings(**dict(tiny_settings(),device='auto',compute=dict(inferenceDevice='auto')))
 result=qualify_devices(paired_graph(),c)
 assert result['learning']==result['inference']=='cpu' and 'Automatic' in result['reason']


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='Actual Apple GPU required')
def test_gpu_qualification_checks_selected_inference_batch(monkeypatch):
 from runner import self_play_devices
 from runner.self_play_training import TrainingSettings
 self_play_devices._cache.clear()
 original=PolicyValueModel.predict_batch;seen=[]
 def corrupted(self,observations):
  seen.append((self.device,len(observations),observations))
  result=original(self,observations)
  if self.device=='mps':
   result[0][0][0]+=1.0
  return result
 monkeypatch.setattr(PolicyValueModel,'predict_batch',corrupted)
 c=TrainingSettings(**dict(tiny_settings(),device='mps',compute={'batchSize':7,'inferenceDevice':'mps'}))
 with pytest.raises(ValueError,match='inference parity'):
  self_play_devices.qualify_devices(paired_graph(),c)
 assert any(device=='mps' and count==7 and len({tuple(x) for x in observations})>1 for device,count,observations in seen)
 assert not self_play_devices._cache
