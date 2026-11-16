"""Independent rule/session expectations, not a copy of the episode implementation."""
import copy
import hashlib
import json
import pytest
from runner.game_rule_examples import take_away, board_game
def placement():return board_game(3,3,3,False)
def gravity():return board_game(6,7,4,True)
from runner.game_episodes import run_episodes, configuration, SessionError, FrozenGraphPolicy

class Policy:
    identity='a'*64
    def __init__(self,values):self.values=values;self.calls=0
    def logits(self,observation):self.calls+=1;return list(self.values)
    def snapshot_digest(self):return self.identity

def cfg(**values):return configuration({'episodes':2,**values})

def test_scalar_policy_wins_known_sequence_and_source_is_unchanged():
    game=take_away();game['state']['remaining']['initial']=5
    original=copy.deepcopy(game);p=Policy([0,0,10])
    out=run_episodes(game,cfg(seats='fixed'),{'A':p,'B':Policy([10,0,0])})
    assert game==original
    assert [m['action'] for m in out['moves'][:3]]==[{'take':3},{'take':1},{'take':1}]
    assert out['episodes'][0]['winner']=='A'
    assert out['summary']['terminal']==2 and out['summary']['failed']==0
    assert p.calls>=2 and out['identity']['definitionSHA256']==hashlib.sha256(json.dumps(game,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def test_stable_actions_masked_not_renumbered_and_greedy_ties():
    game=take_away();game['state']['remaining']['initial']=2
    out=run_episodes(game,cfg(episodes=1,seats='fixed',captureDecisions=True),{'A':Policy([0,10,100]),'B':Policy([0,0,0])})
    m=out['moves'][0];assert m['actionIndex']==1 and m['action']=={'take':2}
    assert m['decision']['legalMask']==[True,True,False]
    assert m['decision']['probabilities']==[0,1,0]
    tied=run_episodes(game,cfg(episodes=1,seats='fixed'),{'A':Policy([1,1,1]),'B':Policy([1,1,1])})
    assert tied['moves'][0]['actionIndex']==0

@pytest.mark.parametrize('factory,width',[(take_away,3),(placement,9),(gravity,7)])
def test_distinct_rules_share_episode_engine(factory,width):
    out=run_episodes(factory(),cfg(maxMoves=128),{'A':Policy([0]*width),'B':Policy([0]*width)})
    assert len(out['episodes'])==2 and out['summary']['terminal']==2
    assert all(m['status']=='applied' for m in out['moves'])
    assert {e['aSeat'] for e in out['episodes']}=={0,1}

def test_seeded_random_legal_is_replayable_without_global_rng_changes():
    import random
    before=random.getstate()
    c=cfg(policies={'A':{'source':'random'},'B':{'source':'random'}})
    a=run_episodes(placement(),c,{})
    b=run_episodes(placement(),c,{})
    assert a['moves']==b['moves'] and random.getstate()==before
    c['seed']+=1
    assert run_episodes(placement(),c,{})['moves']!=a['moves']

def test_move_limit_is_not_draw_and_not_credited_as_a_win():
    out=run_episodes(placement(),cfg(maxMoves=1),{'A':Policy([0]*9),'B':Policy([0]*9)})
    assert out['summary']['truncated']==2 and out['summary']['draws']==0 and out['summary']['terminal']==0
    assert all(e['status']=='moveLimit' and e['winner'] is None for e in out['episodes'])

@pytest.mark.parametrize('change',[{'episodes':3},{'episodes':0},{'maxMoves':0},{'unexpected':1},{'episodeSeconds':float('nan')},{'policies':{'A':{'source':'sameA'}}}])
def test_invalid_config_refused(change):
    with pytest.raises(SessionError):cfg(**change)

def test_bad_logits_refused_before_playing_and_missing_model_does_not_become_random():
    for values in ([0,1], [0,1,float('nan')], [0,1,float('inf')]):
        with pytest.raises(SessionError):run_episodes(take_away(),cfg(),{'A':Policy(values),'B':Policy([0]*3)})
    with pytest.raises(SessionError):run_episodes(take_away(),cfg(),{})

def test_nonterminal_no_legal_actions_is_failure_not_draw():
    game=take_away();game['legal']='False'
    with pytest.raises(SessionError):run_episodes(game,cfg(),{'A':Policy([0]*3),'B':Policy([0]*3)})

def test_cancellation_propagates_and_output_budget_is_not_silently_sampled():
    def cancel():raise InterruptedError('cancelled')
    with pytest.raises(InterruptedError):run_episodes(take_away(),cfg(),{'A':Policy([0]*3),'B':Policy([0]*3)},check_cancelled=cancel)
    with pytest.raises(SessionError,match='artifact'):run_episodes(gravity(),cfg(),{'A':Policy([0]*7),'B':Policy([0]*7)},maximum_bytes=100)

def test_mutable_policy_is_rejected_as_a_failed_inference_contract():
    class Bad(Policy):
        def snapshot_digest(self):return str(self.calls).zfill(64)
    with pytest.raises(SessionError,match='changed'):run_episodes(take_away(),cfg(),{'A':Bad([0]*3),'B':Policy([0]*3)})


def test_authored_failure_refuses_session():
    game=take_away();game['tests'][0]['expect']={'terminal':True}
    with pytest.raises(SessionError,match='tests failed'):run_episodes(game,cfg(),{'A':Policy([0]*3),'B':Policy([0]*3)})

def numeric_graph():
    return dict(schemaVersion=2,nodes=[dict(id='x',type='input',name='Observation',x=0,y=0,config={'features':1}),dict(id='l',type='linear',name='Logits',x=100,y=0,config={'in_features':1,'out_features':3,'bias':True}),dict(id='y',type='output',name='Policy',x=200,y=0,config={})],edges=[dict(id='a',source='x',sourcePort='out',target='l',targetPort='in'),dict(id='b',source='l',sourcePort='out',target='y',targetPort='in')])

def test_actual_torch_model_is_deterministic_frozen_and_layout_independent():
    import torch
    graph=numeric_graph();before=torch.get_rng_state().clone()
    a=FrozenGraphPolicy(graph,seed=73);b=FrozenGraphPolicy(graph,seed=73)
    assert torch.equal(before,torch.get_rng_state())
    assert a.identity==b.identity and a.logits([2])==b.logits([2])
    pin=a.snapshot_digest();out=run_episodes(take_away(),cfg(policies={'B':{'source':'sameA'}}),{'A':a})
    assert out['summary']['terminal']==2 and a.snapshot_digest()==pin
    assert not a.model.training and all(not p.requires_grad for p in a.model.parameters())
    moved=copy.deepcopy(graph);moved['nodes'][0].update(name='Renamed',x=999,y=12)
    assert FrozenGraphPolicy(moved,seed=73).identity==a.identity
    saved=FrozenGraphPolicy(graph,state=a.model.state_dict())
    assert saved.snapshot_digest()==pin and saved.logits([2])==a.logits([2])

def test_authored_extra_turn_is_not_forced_to_alternate():
    game=take_away();game['transitions'][0]['updates']['to_move']='state.to_move';game['terminal'][0]['winner']='state.to_move'
    game['tests']=[dict(name='Same actor retains turn',actions=[{'take':1}],expect={'state':{'remaining':20,'to_move':0}})]
    out=run_episodes(game,cfg(seats='fixed'),{'A':Policy([0,0,1]),'B':Policy([1,0,0])})
    assert all(m['player']==0 and m['entrant']=='A' for m in out['moves'])
    assert out['summary']['aWins']==2

def test_time_limit_does_not_commit_slow_policy_action_or_credit_draw():
    t=[0.]
    class Slow(Policy):
        def logits(self,observation):
            self.calls+=1
            if self.calls>1:t[0]+=2
            return self.values
    out=run_episodes(take_away(),cfg(episodes=1,seats='fixed',episodeSeconds=1),{'A':Slow([0,0,1]),'B':Policy([0,0,0])},clock=lambda:t[0])
    assert out['moves']==[] and out['episodes'][0]['status']=='timeLimit' and out['summary']['draws']==0

def test_initial_terminal_reward_is_authored_not_zero():
    game=take_away();game['state']['remaining']['initial']=0
    out=run_episodes(game,cfg(),{'A':Policy([0]*3),'B':Policy([0]*3)})
    assert out['summary']['terminal']==2 and out['moves']==[]
    assert all(e['rewards']==[-1,1] for e in out['episodes'])

def test_saved_tensor_shapes_and_dtypes_are_exact_not_silently_cast():
    import torch
    p=FrozenGraphPolicy(numeric_graph());state=dict(p.model.state_dict());key=next(iter(state));state[key]=state[key].double()
    with pytest.raises(SessionError,match='dtype'):FrozenGraphPolicy(numeric_graph(),state=state)

def test_named_observation_binding_is_explicit():
    graph=numeric_graph();graph['namedInputs']={'observation':'x'}
    with pytest.raises(SessionError,match='named'):FrozenGraphPolicy(graph)
    p=FrozenGraphPolicy(graph,input_name='observation')
    assert len(p.logits([.5]))==3

def test_batch_normalization_buffers_stay_frozen_at_batch_one():
    graph=numeric_graph();graph['nodes'].insert(2,dict(id='bn',type='batch_norm1d',name='BN',x=0,y=0,config={'num_features':3,'eps':1e-5,'momentum':.1,'affine':True,'track_running_stats':True}))
    graph['edges'][1]['target']='bn';graph['edges'].append(dict(id='c',source='bn',sourcePort='out',target='y',targetPort='in'))
    p=FrozenGraphPolicy(graph);pin=p.snapshot_digest();run_episodes(take_away(),cfg(policies={'B':{'source':'sameA'}}),{'A':p})
    assert p.snapshot_digest()==pin

def test_cpu_mps_numeric_policy_uses_same_frozen_state():
    import torch
    if not torch.backends.mps.is_available():pytest.skip('Actual MPS device unavailable.')
    cpu=FrozenGraphPolicy(numeric_graph(),seed=29);gpu=FrozenGraphPolicy(numeric_graph(),state=cpu.model.state_dict(),device='mps')
    assert gpu.snapshot_digest()==cpu.snapshot_digest()
    torch.testing.assert_close(torch.tensor(gpu.logits([.5])),torch.tensor(cpu.logits([.5])),rtol=1e-5,atol=1e-6)
    pin=gpu.snapshot_digest();run_episodes(take_away(),cfg(device='mps',policies={'B':{'source':'sameA'}}),{'A':gpu});assert gpu.snapshot_digest()==pin

@pytest.mark.parametrize('factory,width',[(take_away,3),(placement,9),(gravity,7)])
def test_exact_recorded_transition_replay_and_tamper_refusal(factory,width):
    from runner.game_episodes import GameEnvironment
    out=run_episodes(factory(),cfg(),{'A':Policy([0]*width),'B':Policy([0]*width)})
    rows=[r for r in out['moves'] if r['episode']==1]
    replay=GameEnvironment.verify_replay(factory(),rows)
    assert replay['revision']==len(rows) and replay['state']==rows[-1]['after']
    corrupt=copy.deepcopy(rows);corrupt[0]['after']={}
    with pytest.raises(SessionError):GameEnvironment.verify_replay(factory(),corrupt)

def test_environment_stale_duplicate_and_illegal_step_are_atomic():
    from runner.game_episodes import GameEnvironment
    from runner.game_rules import Definition
    env=GameEnvironment(Definition(take_away()),'episode1');before=env.snapshot()
    env.step(0,expected_revision=0,expected_state_sha256=before['stateSHA256']);after=env.snapshot()
    with pytest.raises(SessionError,match='Stale'):env.step(0,expected_revision=0,expected_state_sha256=before['stateSHA256'])
    assert env.snapshot()==after
    with pytest.raises(SessionError):env.step(99,expected_revision=1,expected_state_sha256=after['stateSHA256'])
    assert env.snapshot()==after
