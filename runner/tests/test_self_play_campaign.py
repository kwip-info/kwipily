import copy,json,random
import pytest,torch
from runner.game_rule_examples import take_away,board_game
from runner.tests.test_self_play import paired_graph,tiny_settings
from runner.self_play_campaign import run_slice,read_campaign

def configuration():
 c=tiny_settings();c['iterations']=3;c['campaign']={'enabled':True,'sliceIterations':1,'maximumSeconds':3600,'maximumBytes':268435456};return c

def test_campaign_three_slices_extend_and_exact_resume(tmp_path):
 c=configuration();root=tmp_path/'campaign';rules=take_away();graph=paired_graph()
 first=run_slice(root,rules,graph,c,action='start');assert first['campaign']['iteration']==1
 second=run_slice(root,rules,graph,c,action='resume');assert second['campaign']['iteration']==2
 third=run_slice(root,rules,graph,c,action='resume');assert third['campaign']['complete'] and third['campaign']['iteration']==3
 assert len(third['iterations'])==1 and len(third['measurements'])==2
 index=read_campaign(root);assert len(index['segments'])==3
 old=copy.deepcopy(index);c['iterations']=5
 with pytest.raises(ValueError,match='Extend'):run_slice(root,rules,graph,c,action='resume')
 extended=run_slice(root,rules,graph,c,action='extend');assert extended['campaign']['iteration']==4
 assert read_campaign(root)['segments'][:3]==old['segments']
 assert first['weightsSHA256']!=extended['weightsSHA256']
 assert len(list((root/'replay/chunks').glob('*')))<20

def test_campaign_crash_after_commit_is_idempotent_and_pins_refuse(tmp_path):
 c=configuration();root=tmp_path/'campaign'
 def crash(e):
  if e['phase']=='checkpoint':raise InterruptedError()
 with pytest.raises(InterruptedError):run_slice(root,take_away(),paired_graph(),c,action='start',progress=crash)
 recovered=run_slice(root,take_away(),paired_graph(),c,action='resume')
 assert recovered['campaign']['iteration']==2 and len(read_campaign(root)['segments'])==2
 changed=copy.deepcopy(c);changed['learningRate']=.1
 with pytest.raises(ValueError,match='Fork'):run_slice(root,take_away(),paired_graph(),changed,action='resume')

def test_campaign_heldout_openings_and_multiple_opponents(tmp_path):
 c=configuration();c['evaluation']={'opponents':['initial','random','uniform-search'],'openings':[[{'take':1}]],'seed':8301}
 out=run_slice(tmp_path/'c',take_away(),paired_graph(),c,action='start')
 assert {r['opponent'] for r in out['evaluationGames']}=={'initial','random','uniform-search'}
 assert {r['seat'] for r in out['evaluationGames']}=={0,1}
 assert all(r['openingSHA256'] for r in out['evaluationGames'])
 # Read the actual required replay inventory, not a nonexistent state field.
 def rows(root):
  manifest=json.loads((root/'replay/checkpoints/iteration-01/manifest.json').read_text())
  return [json.loads(line) for chunk in manifest['replay'] for line in (root/'replay/chunks'/chunk['path']).read_text().splitlines()]
 actual=rows(tmp_path/'c');assert actual
 reference=copy.deepcopy(c);reference['evaluation']={}
 run_slice(tmp_path/'without-openings',take_away(),paired_graph(),reference,action='start')
 assert actual==rows(tmp_path/'without-openings')
 assert {p['game'] for p in actual}==set(range(c['games']))
 # Each collected game begins from the initial state, not the held-out prefix.
 for game in range(c['games']):
  first=next(p for p in actual if p['game']==game)
  assert first['observation']==[1.0]
 assert all('openingSHA256' not in p for p in actual)

def test_campaign_fork_resets_replay_and_learning_settings_are_explicit(tmp_path):
 import hashlib
 c=configuration();parent=tmp_path/'parent';first=run_slice(parent,take_away(),paired_graph(),c,action='start')
 changed=copy.deepcopy(c);changed['learningRate']=.0003
 pin=hashlib.sha256((parent/'campaign.json').read_bytes()).hexdigest()
 fork=run_slice(tmp_path/'child',take_away(),paired_graph(),changed,action='fork',parent_root=parent,parent_index=pin)
 assert fork['campaign']['iteration']==1
 assert read_campaign(tmp_path/'child')['parent']['mode']=='weights only; optimizer/replay reset'
 assert fork['iterations'][0]['collectionSHA256']==first['weightsSHA256']
 with pytest.raises(ValueError,match='changed after capture'):run_slice(parent,take_away(),paired_graph(),c,action='resume',expected_index='0'*64)

def test_campaign_large_total_retains_bounded_slice_and_quota_refuses_before_update(tmp_path):
 c=configuration();c['iterations']=1000
 out=run_slice(tmp_path/'long',take_away(),paired_graph(),c,action='start')
 assert out['campaign']['target']==1000 and out['campaign']['iteration']==1 and len(out['measurements'])==2
 small=configuration();small['campaign']['maximumBytes']=67108864
 with pytest.raises(ValueError,match='space for the next'):run_slice(tmp_path/'small',take_away(),paired_graph(),small,action='start')
 assert not (tmp_path/'small/replay/latest.json').exists()

def test_managed_campaign_grant_is_same_project_pinned_and_has_no_replay_copy(tmp_path):
 import types,uuid,os,hashlib
 from runner.self_play_worker import execute
 from runner.tests.test_self_play_worker import capture
 base=tmp_path/'workflow-operations'/str(uuid.uuid4()).upper();store=base/str(uuid.uuid4()).upper()/'campaign';store.mkdir(parents=True,mode=0o700)
 def launch(action,index=None):
  root=base/str(uuid.uuid4()).upper();inner=dict(schemaVersion=1,definition=take_away(),graph=paired_graph(),configuration=configuration(),initialization=None,trainability=None,resume=False,validateOnly=False)
  request=capture(root,inner);stat=store.stat();grant=dict(campaignID=store.parent.name,device=stat.st_dev,inode=stat.st_ino,action=action,expectedIndexSHA256=index,parent=None)
  raw=json.dumps(grant).encode();(root/'input/campaign-grant.json').write_bytes(raw);request['inputFiles'].append(dict(path='input/campaign-grant.json',bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
  return root,types.SimpleNamespace(_path=root,request=request)
 root,authority=launch('start');events=[];execute(authority,events.append,lambda:None)
 assert len(authority.request['inputFiles'])==3
 assert not (root/'artifacts/replay').exists() and (store/'replay/latest.json').exists()
 assert {n for n in ['campaign','opponents','performance'] if (root/f'artifacts/{n}.table.json').exists()}=={'campaign','opponents','performance'}
 old=hashlib.sha256((store/'campaign.json').read_bytes()).hexdigest();root2,next_authority=launch('resume',old)
 value=read_campaign(store);value['note']='changed after capture';(store/'campaign.json').write_text(json.dumps(value))
 with pytest.raises(ValueError,match='changed after capture'):execute(next_authority,lambda e:None,lambda:None)
 assert read_campaign(store)['iteration']==1

def test_checkpoint_retention_uses_numeric_iteration_order(tmp_path):
 from runner.self_play_replay import ReplayStore
 store=ReplayStore(tmp_path/'numeric',checkpoints=2)
 files={n:b'test' for n in ('model.pt','optimizer.pt','rng.pt')}
 m=None
 for iteration in (98,99,100,101):m=store.commit(iteration,{},[],files,{},m)
 assert store.checkpoint()[1]['iteration']==101
 assert {p.name for p in (store.root/'checkpoints').iterdir()}=={'iteration-100','iteration-101'}

def test_fork_refuses_busy_parent_and_old_capture(tmp_path):
 import fcntl,os,hashlib
 c=configuration();parent=tmp_path/'parent';run_slice(parent,take_away(),paired_graph(),c,action='start')
 pin=hashlib.sha256((parent/'campaign.json').read_bytes()).hexdigest();fd=os.open(parent/'campaign.lock',os.O_RDWR)
 try:
  fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
  with pytest.raises(BlockingIOError):run_slice(tmp_path/'child',take_away(),paired_graph(),c,action='fork',parent_root=parent,parent_index=pin)
 finally:os.close(fd)
 assert not (tmp_path/'child/initial-model.pt').exists()

def test_pause_commits_only_current_iteration_inside_larger_slice(tmp_path):
 c=configuration();c['campaign']['sliceIterations']=3
 first=run_slice(tmp_path/'paused',take_away(),paired_graph(),c,action='start',pause_requested=lambda:True)
 assert first['campaign']['iteration']==1 and not first['campaign']['complete']
 assert len(first['iterations'])==1
 resumed=run_slice(tmp_path/'paused',take_away(),paired_graph(),c,action='resume')
 assert resumed['campaign']['iteration']==3 and resumed['campaign']['complete']

def test_native_grant_never_accepts_internal_unpinned_test_sentinel(tmp_path):
 import uuid
 from runner.self_play_campaign_grant import campaign_directory
 operation=tmp_path/'workflow-operations'/str(uuid.uuid4()).upper()/str(uuid.uuid4()).upper()
 owned=operation.parent/str(uuid.uuid4()).upper()/'campaign';owned.mkdir(mode=0o700,parents=True);s=owned.stat()
 grant=dict(campaignID=owned.parent.name,device=s.st_dev,inode=s.st_ino,action='resume',expectedIndexSHA256='unrestricted-test',parent=None)
 with pytest.raises(ValueError,match='journal hash'):campaign_directory(operation,grant)
 grant['expectedIndexSHA256']='0'*64;grant['device']=True
 with pytest.raises(ValueError,match='directory pin'):campaign_directory(operation,grant)
