"""Independent math/actor/gradient controls for generic self-play."""
import copy,math,random
import pytest
from runner.game_rule_examples import take_away,board_game
from runner.self_play_search import Search, SearchSettings
from runner.self_play_model import PolicyValueModel, joint_loss


def paired_graph(n=1,a=3):
    specs=[('input','input',{'features':n}),('shared','linear',{'in_features':n,'out_features':8,'bias':True}),('hidden','tanh',{}),('policy','linear',{'in_features':8,'out_features':a,'bias':True}),('scalar','linear',{'in_features':8,'out_features':1,'bias':True}),('value','tanh',{}),('output','output',{'task':'gamePolicyValue','taskVersion':1})]
    edges=[('input','shared','in'),('shared','hidden','in'),('hidden','policy','in'),('hidden','scalar','in'),('scalar','value','in'),('policy','output','policy'),('value','output','value')]
    return dict(schemaVersion=2,name='Joint test',nodes=[dict(id=i,type=t,config=c) for i,t,c in specs],edges=[dict(id=str(k),source=s,sourcePort='out',target=t,targetPort=p) for k,(s,t,p) in enumerate(edges)])

class Zero:
    def __init__(self,a):self.a=a;self.calls=0
    def predict(self,x):self.calls+=1;return [0.]*self.a,0.

def test_joint_heads_real_shared_trunk_gradients_and_safe_roundtrip():
    import torch
    model=PolicyValueModel(paired_graph(),seed=17)
    x=torch.tensor([[.5],[.25]])
    out=model.forward(x)
    assert set(out)=={'policy','value'} and out['policy'].shape==(2,3) and out['value'].shape==(2,1)
    loss,parts=joint_loss(out,torch.tensor([[True,True,False],[True,True,True]]),torch.tensor([[1.,0,0],[0,1.,0]]),torch.tensor([[1.],[-1.]]))
    loss.backward()
    for key in ('shared','policy','scalar'):
        g=model.model.layers[key].weight.grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum()>0
    assert all(math.isfinite(v) for v in parts.values())
    restored=PolicyValueModel(paired_graph(),state=copy.deepcopy(model.model.state_dict()))
    assert torch.equal(out['policy'],restored.forward(x)['policy'])

def test_joint_loss_independent_numeric_and_illegal_gradient():
    import torch
    logits=torch.tensor([[0.,math.log(3),100.]],requires_grad=True);value=torch.tensor([[.5]],requires_grad=True)
    loss,parts=joint_loss({'policy':logits,'value':value},torch.tensor([[True,True,False]]),torch.tensor([[1.,0,0]]),torch.tensor([[1.]]))
    assert loss.item()==pytest.approx(math.log(4)+.25)
    loss.backward();assert logits.grad[0].tolist()==pytest.approx([-.75,.75,0]) and value.grad.item()==pytest.approx(-1.)
    with pytest.raises(ValueError):joint_loss({'policy':logits,'value':value},torch.tensor([[False]*3]),torch.tensor([[1.,0,0]]),torch.tensor([[1.]]))

def test_search_prefers_immediate_win_and_never_mutates_played_state():
    from runner.game_rules import Definition
    rules=take_away();definition=Definition(rules);state={'remaining':3,'to_move':0};before=copy.deepcopy(state)
    search=Search(definition,Zero(3),SearchSettings(simulations=32))
    result=search.run(state,random.Random(17),noise=False)
    assert result['actionIndex']==2 and sum(result['visits'])==32 and result['probabilities'][2]>result['probabilities'][0]
    assert state==before and result['legalMask']==[True,True,True]

@pytest.mark.parametrize('game,a',[(take_away(),3),(board_game(3,3,3,False),9),(board_game(6,7,4,True),7)])
def test_search_different_definitions_same_engine(game,a):
    from runner.game_rules import Definition
    d=Definition(game);s=Search(d,Zero(a),SearchSettings(simulations=2))
    out=s.run(d.initial(),random.Random(8),noise=False)
    assert len(out['visits'])==a and sum(out['visits'])==2 and out['legalMask'][out['actionIndex']]

def test_terminal_search_never_calls_model_and_extra_turn_does_not_flip_value():
    from runner.game_rules import Definition
    rules=take_away();rules['transitions'][0]['updates']['to_move']='state.to_move';rules['terminal'][0]['winner']='state.to_move'
    p=Zero(3);d=Definition(rules);s=Search(d,p,SearchSettings(simulations=16))
    out=s.run({'remaining':2,'to_move':0},random.Random(1),noise=False)
    assert out['q'][1]>0 and out['q'][0]>0
    calls=p.calls;end=s.run({'remaining':0,'to_move':0},random.Random(1),noise=False)
    assert end['terminal'] and p.calls==calls

def test_seeded_noise_bounded_nodes_cancellation_and_nonfinite_refusal():
    from runner.game_rules import Definition
    d=Definition(take_away());s=Search(d,Zero(3),SearchSettings(simulations=8,maximumNodes=3))
    a=s.run(d.initial(),random.Random(9),noise=True);b=s.run(d.initial(),random.Random(9),noise=True)
    assert a==b and a['nodes']<=3
    class Bad(Zero):
        def predict(self,x):return [float('nan')]*3,0.
    with pytest.raises(ValueError):Search(d,Bad(3),SearchSettings()).run(d.initial(),random.Random(1))
    def stop():raise InterruptedError('cancelled')
    with pytest.raises(InterruptedError):Search(d,Zero(3),SearchSettings(),check_cancelled=stop).run(d.initial(),random.Random(1))

def tiny_settings():
    from runner.self_play_training import TrainingSettings
    from dataclasses import asdict
    return asdict(TrainingSettings(iterations=2,games=2,updates=2,batch=4,evaluationPairs=1,maxMoves=32,search=SearchSettings(simulations=2).value()))

def test_real_training_changes_weights_and_resume_is_exact(tmp_path):
    from runner.self_play_training import train
    from runner.self_play_replay import ReplayStore
    c=tiny_settings();graph=paired_graph();rules=take_away()
    whole=train(rules,graph,c,tmp_path/'whole')
    def interrupt(event):
        if event['phase']=='checkpoint' and event['iteration']==1:raise InterruptedError('simulated interruption after committed iteration')
    with pytest.raises(InterruptedError):train(rules,graph,c,tmp_path/'resume',progress=interrupt)
    resumed=train(rules,graph,c,tmp_path/'resume',resume=True)
    assert whole['weightsSHA256']==resumed['weightsSHA256'] and whole['iterations']==resumed['iterations'] and whole['episodes']==resumed['episodes'] and len(whole['iterations'])==2
    assert whole['iterations'][0]['collectionSHA256']!=whole['weightsSHA256']
    assert all(e['terminal']+e['truncated']+e['failed']==2 for e in whole['iterations'])
    assert len(whole['evaluationGames'])==8
    _,cp=ReplayStore(tmp_path/'resume').checkpoint()
    assert sum(e['count'] for e in cp['replay'])>0
    wrong=copy.deepcopy(c);wrong['learningRate']=.01
    with pytest.raises(ValueError,match='Resume pins'):train(rules,graph,wrong,tmp_path/'resume',resume=True)

def test_replay_capacity_corruption_and_immutable_boundary(tmp_path):
    from runner.self_play_replay import ReplayStore,CapacityError
    s=ReplayStore(tmp_path/'store',maximum_positions=1)
    with s.exclusive():
        m=s.commit(1,{},[{'test':1}],{'model.pt':b'weights','optimizer.pt':b'optimizer','rng.pt':b'rng'},{})
        assert s.sample(m,2,random.Random(1))==[{'test':1}]*2
        with pytest.raises(CapacityError):s.commit(2,{},[{'test':2}],{}, {},m)
        with pytest.raises(ValueError):s.commit(1,{},[],{}, {},m)
    chunk=s.root/'chunks'/m['replay'][0]['path'];chunk.write_bytes(b'corrupted')
    with pytest.raises(ValueError):s.checkpoint()

def test_truncated_games_never_become_draw_targets(tmp_path):
    from runner.self_play_training import train
    c=tiny_settings();c['maxMoves']=1
    with pytest.raises(ValueError,match='No terminal games'):train(take_away(),paired_graph(),c,tmp_path/'truncated')
    assert not (tmp_path/'truncated/latest.json').exists()

def test_rolling_retention_preserves_pinned_resume_chunks(tmp_path):
    from runner.self_play_replay import ReplayStore
    s=ReplayStore(tmp_path/'store',maximum_positions=1,retention='rolling',checkpoints=1)
    first=s.commit(1,{},[{'v':1}],{'model.pt':b'weights','optimizer.pt':b'optimizer','rng.pt':b'rng'},{});(s.root/'checkpoints/iteration-01/PINNED').write_text('owner pin')
    second=s.commit(2,{},[{'v':2}],{'model.pt':b'weights','optimizer.pt':b'optimizer','rng.pt':b'rng'},{},first)
    assert s.sample(second,1,random.Random(1))==[{'v':2}]
    assert s.checkpoint('iteration-01')[1]==first
    assert len(list((s.root/'chunks').iterdir()))==2

def test_paired_export_code_matches_both_heads_and_legacy_refuses_joint_task():
    import torch
    from runner.engine import export_source,GraphModel
    graph=paired_graph();model=PolicyValueModel(graph);source=export_source(model.model);ns={};exec(source,ns)
    other=ns['Model']();other.load_state_dict(model.model.state_dict());x=torch.tensor([[.7]])
    for key in ('policy','value'):assert torch.equal(other(x)[key],model.forward(x)[key])
    with pytest.raises(ValueError,match='Self-play'):GraphModel(graph,bundled_dataset=False)
    from runner.game_episodes import FrozenGraphPolicy
    frozen=FrozenGraphPolicy(graph,state=model.model.state_dict());assert frozen.logits([.7])==model.predict([.7])[0]


def test_custom_named_input_train_resume_and_frozen_reuse(tmp_path):
    from runner.self_play_training import train
    graph=paired_graph();graph['namedInputs']={'observation':'input'}
    model=PolicyValueModel(graph)
    assert model.frozen().predict([.5])==model.predict([.5])
    from runner.engine import export_source
    import torch
    namespace={};exec(export_source(model.model),namespace)
    exported=namespace['Model']();exported.load_state_dict(model.model.state_dict())
    x=torch.tensor([[.5]])
    for key in ('policy','value'):
        assert torch.equal(exported({'observation':x})[key],model.forward(x)[key])
    c=tiny_settings()
    whole=train(take_away(),graph,c,tmp_path/'whole')
    def stop(event):
        if event['phase']=='checkpoint':raise InterruptedError('checkpoint')
    with pytest.raises(InterruptedError):train(take_away(),graph,c,tmp_path/'resume',progress=stop)
    resumed=train(take_away(),graph,c,tmp_path/'resume',resume=True)
    assert resumed['weightsSHA256']==whole['weightsSHA256']


def test_rolling_capacity_explains_atomic_headroom_and_preserves_last_boundary(tmp_path):
    from runner.self_play_replay import ReplayStore,CapacityError
    store=ReplayStore(tmp_path,maximum_positions=4,maximum_bytes=65536,retention='rolling',checkpoints=1)
    manifest=store.commit(1,{},[{'text':'a'*9000,'row':i} for i in range(4)],{'model.pt':b'a','optimizer.pt':b'a','rng.pt':b'a'}, {})
    with pytest.raises(CapacityError,match='Rolling needs temporary headroom'):
        store.admit([{'text':'b'*9000,'row':i} for i in range(4)],manifest)
    assert store.checkpoint()[1]==manifest
