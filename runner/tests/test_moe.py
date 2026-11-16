import copy
import torch
import pytest
from runner.moe import RoutedExperts, default_config
import json
from runner.engine import GraphModel, export_source


def graph(c):
    return dict(schemaVersion=2,nodes=[dict(id='input',type='input',config={'features':c['input_features']}),dict(id='mixture',type='mixture_experts',config=c),dict(id='output',type='output',config={})],edges=[dict(id='a',source='input',sourcePort='out',target='mixture',targetPort='in'),dict(id='b',source='mixture',sourcePort='out',target='output',targetPort='in')])


@pytest.mark.parametrize('experts',[2,3,4])
@pytest.mark.parametrize('weighting',['normalized','probability'])
def test_graph_code_and_gradients(experts,weighting):
    c=default_config(3,2,experts);c['weighting']=weighting;c['top_k']=min(2,experts)
    m=GraphModel(graph(c),bundled_dataset=False);x=torch.randn(5,3)
    source=export_source(m);namespace={};exec(source,namespace)
    exported=namespace['Model']();exported.load_state_dict(m.state_dict(),strict=True)
    torch.testing.assert_close(m(x),exported(x))
    expected=torch.autograd.grad(m(x).square().mean(),tuple(m.parameters()),allow_unused=True)
    actual=torch.autograd.grad(exported(x).square().mean(),tuple(exported.parameters()),allow_unused=True)
    for a,b in zip(actual,expected):
        if a is None or b is None:assert a is None and b is None
        else:torch.testing.assert_close(a,b)


def test_graph_training_and_independent_reload(tmp_path):
    c=default_config();m=GraphModel(graph(c));optimizer=torch.optim.AdamW(m.parameters(),lr=.02)
    x=torch.randn(64,2);target=(x[:,0]>0).long()
    from runner.moe import auxiliary_loss
    initial=torch.nn.functional.cross_entropy(m(x),target).item()
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True);y=m(x);loss=torch.nn.functional.cross_entropy(y,target)+auxiliary_loss(m);loss.backward();optimizer.step()
    assert torch.nn.functional.cross_entropy(m(x),target).item()<initial*.5
    from safetensors.torch import save_file,load_file
    save_file(m.state_dict(),str(tmp_path/'model.safetensors'))
    independent=GraphModel(graph(c));independent.load_state_dict(load_file(str(tmp_path/'model.safetensors')))
    torch.testing.assert_close(m(x),independent(x))


def test_mps_forward_backward_if_available():
    if not torch.backends.mps.is_available():pytest.skip('Apple GPU unavailable')
    # Match engine.run's qualified MPS policy; earlier CPU tests enable strict determinism.
    before=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(False)
        c=default_config();cpu=RoutedExperts(c);mps=RoutedExperts(c).to('mps');mps.load_state_dict(cpu.state_dict())
        x=torch.randn(17,2);expected=cpu(x);actual=mps(x.to('mps'))
        torch.testing.assert_close(actual.cpu(),expected,atol=2e-5,rtol=2e-4)
        (actual.square().mean()+mps.auxiliary_loss()).backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in mps.parameters())
    finally:torch.use_deterministic_algorithms(before,warn_only=warn)


def test_real_engine_receipts_progress_and_resume(tmp_path):
    from runner.engine import run
    c=default_config();c['capture']='sample'
    events=[]
    run(graph(c),tmp_path/'first',epochs=2,seed=42,device='cpu',batch_size=64,emit=events.append)
    manifest=json.loads((tmp_path/'first'/'manifest.json').read_text())
    assert set(manifest['routingTables'])=={'routing','routingSample'}
    rows=json.loads((tmp_path/'first'/'routing.json').read_text())['rows']
    assert {row[1] for row in rows}=={'training','inference'}
    points=[p for p in events if p['event']=='trainingProgress' and p['schemaVersion']==2]
    assert points and all(p['moe']['trainingObjective']==p['trainingLoss']+p['moe']['balancePenalty']+p['moe']['routerZPenalty'] for p in points)
    run(graph(c),tmp_path/'resume',epochs=4,seed=42,device='cpu',batch_size=64,resume_checkpoint=tmp_path/'first'/'checkpoint.pt',emit=lambda _:None)
    run(graph(c),tmp_path/'whole',epochs=4,seed=42,device='cpu',batch_size=64,emit=lambda _:None)
    resumed=torch.load(tmp_path/'resume'/'checkpoint.pt',weights_only=True);whole=torch.load(tmp_path/'whole'/'checkpoint.pt',weights_only=True)
    for key in whole['modelState']:torch.testing.assert_close(resumed['modelState'][key],whole['modelState'][key],atol=0,rtol=0)


@pytest.mark.parametrize('mutation',[lambda c:c.update(top_k=5),lambda c:c.update(temperature=float('nan')),lambda c:c.update(mask=1),lambda c:c.update(version=2),lambda c:c.update(expert_2=c['expert_1'])])
def test_strict_refusals(mutation):
    c=default_config();mutation(c)
    with pytest.raises(ValueError):RoutedExperts(c)

def test_sparse_matches_independent_dense_and_router_gradient():
    c=default_config(3,2,experts=3);m=RoutedExperts(c);x=torch.randn(7,3,requires_grad=True)
    y=m(x)
    logits=m.router(x);p=torch.softmax(logits,dim=-1)
    ids=torch.argsort(logits,dim=-1,descending=True,stable=True)[:,:2]
    w=p.gather(1,ids);w=w/w.sum(-1,keepdim=True)
    reference=sum(expert(x)*((ids==i)*w).sum(-1,keepdim=True) for i,expert in enumerate(m.experts.values()))
    torch.testing.assert_close(y,reference)
    (y.square().mean()+m.auxiliary_loss()).backward()
    assert m.router.layers['gate'].weight.grad.abs().sum()>0

def test_mask_ties_and_capture_counts():
    c=default_config(3,2);c['mask']=True;m=RoutedExperts(c)
    for p in m.router.parameters():p.data.zero_()
    x=torch.randn(2,4,3);mask=torch.tensor([[1,1,0,0],[1,0,1,0]])
    out=m(x,mask)
    assert torch.count_nonzero(out[mask==0])==0
    summary=m.routing_summary()
    assert sum(v['assignments'] for v in summary)==8
    assert summary[0]['assignments']==4 and summary[1]['assignments']==4
    assert all(v['assignments']==0 for v in summary[2:])

def test_unselected_frozen_and_invalid():
    c=default_config(2,2);c['top_k']=1;c['weighting']='probability'
    m=RoutedExperts(c)
    for p in m.router.parameters():p.data.zero_()
    optimizer=torch.optim.AdamW(m.parameters(),lr=.01,weight_decay=.1)
    before=copy.deepcopy(m.state_dict());optimizer.zero_grad(set_to_none=True)
    y=m(torch.randn(5,2));(y.square().mean()+m.auxiliary_loss()).backward();optimizer.step()
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items() if k.startswith('experts.expert_b.'))
    with pytest.raises(ValueError):m(torch.randn(2,3))

def test_capture_off_masks_top1_and_extreme_temperature():
    c=default_config();c.update(mask=True,capture='off',top_k=1,temperature=.0001)
    m=RoutedExperts(c)
    x=torch.randn(5,2);mask=torch.zeros(5,dtype=torch.int64)
    y=m(x,mask);(y.square().mean()+m.auxiliary_loss()).backward()
    assert torch.count_nonzero(y)==0 and not m.routing_summary() and not m._samples
    with pytest.raises(ValueError):m(x)
    for parameter in m.router.parameters():parameter.data.zero_()
    m.zero_grad(set_to_none=True);m(x,torch.ones(5,dtype=torch.int64)).sum().backward()
    assert m.router.layers['gate'].weight.grad.abs().sum()==0
    c['capture']='sample';m=RoutedExperts(c);m.capture_enabled=False
    m(x,torch.ones(5,dtype=torch.int64));assert not m._samples and not m._phases

def test_actual_dispatch_skips_unselected_and_frozen_is_unchanged():
    c=default_config();c['top_k']=1
    frozen=json.loads(c['expert_1']);frozen['trainable']=False;c['expert_1']=json.dumps(frozen)
    m=RoutedExperts(c)
    for p in m.router.parameters():p.data.zero_()
    seen=[]
    handles=[e.register_forward_hook(lambda _,_args,_out,k=k:seen.append(k)) for k,e in m.experts.items()]
    before=copy.deepcopy(m.state_dict());o=torch.optim.AdamW(m.parameters(),lr=.01,weight_decay=.1)
    o.zero_grad(set_to_none=True);(m(torch.randn(5,2)).square().mean()+m.auxiliary_loss()).backward();o.step()
    assert seen==['expert_a']
    assert all(torch.equal(before[k],v) for k,v in m.state_dict().items() if k.startswith('experts.'))
    for h in handles:h.remove()

def test_synthetic_preflight_and_cosmetic_identity():
    from runner.moe import preflight,scientific_config
    c=default_config();before=copy.deepcopy(c);assert preflight(c)['optimizerUpdates']==0 and c==before
    e=json.loads(c['expert_1']);e['name']='New label';e['graph']['nodes'][0].update(name='New input',x=900,y=300);e['graph']['edges'][0]['id']='cosmetic';c['expert_1']=json.dumps(e)
    assert scientific_config(c)==scientific_config(before)
    def cancel():raise InterruptedError('cancel')
    with pytest.raises(InterruptedError):preflight(c,cancel)

def test_real_dropout_resume_and_capture_do_not_change_learning(tmp_path):
    from runner.engine import run
    c=default_config();e=json.loads(c['expert_1']);g=e['graph'];g['nodes'].insert(1,dict(id='drop',type='dropout',config={'p':.2}));g['edges'][0]['source']='drop';g['edges'].append(dict(id='input-drop',source='input',sourcePort='out',target='drop',targetPort='in'));c['expert_1']=json.dumps(e)
    run(graph(c),tmp_path/'part',epochs=1,seed=42,device='cpu',batch_size=64,emit=lambda _:None)
    run(graph(c),tmp_path/'resume',epochs=3,seed=42,device='cpu',batch_size=64,resume_checkpoint=tmp_path/'part'/'checkpoint.pt',emit=lambda _:None)
    c['capture']='off'
    run(graph(c),tmp_path/'whole',epochs=3,seed=42,device='cpu',batch_size=64,emit=lambda _:None)
    # Capture is scientific configuration for continuation but does not alter gradients/RNG.
    a=torch.load(tmp_path/'resume'/'checkpoint.pt',weights_only=True);b=torch.load(tmp_path/'whole'/'checkpoint.pt',weights_only=True)
    for k in a['modelState']:torch.testing.assert_close(a['modelState'][k],b['modelState'][k],rtol=0,atol=0)
    assert not json.loads((tmp_path/'whole'/'manifest.json').read_text()).get('routingTables')

def test_real_paired_policy_value_moe_selfplay(tmp_path):
    from runner.tests.test_self_play import paired_graph,tiny_settings
    from runner.self_play_training import train
    from runner.game_rule_examples import take_away
    g=paired_graph();g['nodes'][1].update(type='mixture_experts',config=default_config(1,8))
    c=tiny_settings();result=train(take_away(),g,c,tmp_path/'game')
    assert len(result['iterations'])==2 and result['measurements'][-1]['trainingObjective']>0
    assert result['routingTables']['routing']['rows']

@pytest.mark.parametrize('k',[1,2,3])
@pytest.mark.parametrize('weighting',['normalized','probability'])
def test_all_routing_choices_match_dense_oracle(k,weighting):
    c=default_config(3,2,3);c.update(top_k=k,weighting=weighting)
    m=RoutedExperts(c);x=torch.randn(2,5,3);flat=x.reshape(-1,3)
    logits=m.router(flat);prob=torch.softmax(logits/c['temperature'],dim=-1)
    chosen=torch.argsort(logits,dim=-1,descending=True,stable=True)[:,:k]
    weights=prob.gather(1,chosen)
    if weighting=='normalized':weights=weights/weights.sum(-1,keepdim=True)
    dense=sum(e(flat)*((chosen==i)*weights).sum(-1,keepdim=True) for i,e in enumerate(m.experts.values()))
    torch.testing.assert_close(m(x),dense.reshape(2,5,2))

def test_conditional_optimizer_resume_refuses_changed_ledger(tmp_path):
    from runner.engine import run
    g=graph(default_config());run(g,tmp_path/'first',epochs=1,seed=42,device='cpu',emit=lambda _:None)
    parent=torch.load(tmp_path/'first/checkpoint.pt',weights_only=True);parent['conditionalOptimizerUpdates'][0]+=1
    torch.save(parent,tmp_path/'first/checkpoint.pt')
    with pytest.raises(ValueError):run(g,tmp_path/'next',epochs=2,seed=42,device='cpu',resume_checkpoint=tmp_path/'first/checkpoint.pt',emit=lambda _:None)

def test_selfplay_worker_routing_outputs_are_real_standard_tables(tmp_path):
    from runner.tests.test_self_play_worker import capture
    from runner.tests.test_self_play import paired_graph,tiny_settings
    from runner.self_play_worker import execute
    from runner.game_rule_examples import take_away
    from runner.dataflow_table_reader import read_table_artifact
    import uuid,types
    g=paired_graph();g['nodes'][1].update(type='mixture_experts',config=default_config(1,8))
    root=tmp_path/'workflow-operations'/str(uuid.uuid4()).upper()/str(uuid.uuid4()).upper()
    inner=dict(schemaVersion=1,definition=take_away(),graph=g,configuration=tiny_settings(),initialization=None,trainability=None,resume=False,validateOnly=False)
    request=capture(root,inner);events=[];execute(types.SimpleNamespace(_path=root,request=request),events.append,lambda:None)
    files={row['path']:row for row in events[-1]['files']}
    for port in ('routing','routingSample'):
        path='artifacts/'+port+'.table.json';read_table_artifact(root/path,files[path]['sha256'],lambda:None)


@pytest.mark.parametrize('parameter_free_experts', [False, True])
def test_parameter_free_router_all_masked_is_zero_without_parameter_updates(parameter_free_experts):
    identity = dict(schemaVersion=2,
                    nodes=[dict(id='input', type='input', config={'features':2}), dict(id='output', type='output', config={})],
                    edges=[dict(id='e',source='input',sourcePort='out',target='output',targetPort='in')])
    c=default_config(2,2,experts=2);c.update(mask=True, router=json.dumps(identity))
    if parameter_free_experts:
        for i in range(1,5):
            expert=json.loads(c['expert_'+str(i)]);expert['graph']=identity;c['expert_'+str(i)]=json.dumps(expert)
    m=RoutedExperts(c)
    x=torch.ones(2,2);mask=torch.zeros(2,dtype=torch.int64)
    before=copy.deepcopy(m.state_dict());y=m(x,mask)
    (y.sum()+m.auxiliary_loss()).backward()
    assert torch.count_nonzero(y)==0 and m.auxiliary_loss().item()==0
    assert all(p.grad is None for p in m.parameters())
    if list(m.parameters()):
        optimizer=torch.optim.AdamW(m.parameters(),lr=.1,weight_decay=.1);optimizer.step()
    assert all(torch.equal(before[k],v) for k,v in m.state_dict().items())
    # The source-derived standalone exporter must carry the same zero semantics.
    from runner.moe_export import helpers
    export_config=dict(c,mask=False)
    source=helpers(GraphModel(graph(export_config),bundled_dataset=False,architecture_only=True))
    namespace={'torch':torch,'nn':torch.nn};exec(source,namespace)
    standalone=namespace['RoutedExperts'](c)
    out=standalone(x,mask);(out.sum()+standalone.auxiliary_loss()).backward()
    assert torch.equal(out,y)
