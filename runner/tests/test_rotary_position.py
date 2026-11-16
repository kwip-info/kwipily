"""RoPE uses independently calculated real rotations, not its own output as oracle."""
import math
import pytest
import torch
from runner.attention_configuration import normalize_attention,resolve_attention
from runner.attention_execution import execute_attention
from runner.tensor_layout import BatchedLayout


def rotate(q,k,config=None):
    c=normalize_attention('rotary_position',config or {})
    return execute_attention('rotary_position',c,[q,k],None)


def oracle(x,positions,axis,width,pairing,base,scale):
    moved=x.movedim(axis,-2);out=moved.clone()
    for b in range(x.shape[0]):
        for t in range(moved.shape[-2]):
            for j in range(width//2):
                a,z=(2*j,2*j+1) if pairing=='adjacent' else (j,j+width//2)
                angle=float(positions[b,t])*base**(-2*j/width)/scale
                left,right=moved[b,...,t,a],moved[b,...,t,z]
                out[b,...,t,a]=left*math.cos(angle)-right*math.sin(angle)
                out[b,...,t,z]=left*math.sin(angle)+right*math.cos(angle)
    return out.movedim(-2,axis)


@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('shape,axis',[((2,3,6),1),((2,2,3,6),2),((2,3,2,6),1)])
@pytest.mark.parametrize('pairing',['adjacent','halves'])
def test_real_rotations_partial_suffix_and_gradients(device,shape,axis,pairing):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
    q=(torch.arange(math.prod(shape),device=device,dtype=torch.float32).reshape(shape)/13-2).requires_grad_()
    k=(q.detach().flip(-1)/2).requires_grad_()
    pos=torch.arange(3,device=device).expand(2,3)+2
    config=dict(sequence_axis=axis,rotation='custom',rotary_dim=4,pairing=pairing,base=1234.,position_scale=2.,start_position=2)
    result=rotate(q,k,config)
    assert set(result)=={'query','key'}
    for name,x in [('query',q),('key',k)]:
        expected=oracle(x,pos,axis,4,pairing,1234.,2.)
        torch.testing.assert_close(result[name],expected,atol=3e-5,rtol=3e-5)
        torch.testing.assert_close(result[name][...,4:],x[...,4:],atol=0,rtol=0)
        torch.testing.assert_close(result[name][...,:4].square().sum(-1),x[...,:4].square().sum(-1),atol=3e-4,rtol=3e-5)
        upstream=torch.linspace(-1,2,x.numel(),device=device).reshape(x.shape)
        actual=torch.autograd.grad(result[name],x,upstream,retain_graph=True)[0]
        expected_grad=torch.autograd.grad(expected,x,upstream,retain_graph=True)[0]
        torch.testing.assert_close(actual,expected_grad,atol=3e-5,rtol=3e-5)
        assert result[name].device==x.device and result[name].dtype==torch.float32


@pytest.mark.parametrize('bad',[{'base':1},{'base':True},{'base':float('nan')},{'rotation':'custom','rotary_dim':3},{'sequence_axis':0},{'sequence_axis':-1},{'position_scale':0},{'start_position':-1},{'start_position':True},{'pairing':'unknown'},{'position_source':'unknown'},{'weights':True}])
def test_invalid_configuration_refuses(bad):
    with pytest.raises(ValueError):normalize_attention('rotary_position',bad)


def test_runtime_type_shape_value_and_observation_refusals():
    q=torch.ones(2,3,4);k=q.clone()
    for args in [(q.double(),k,{}),(q,k[:1],{}),(q,k[:,:,:3],{}),(q,k,{'rotation':'custom','rotary_dim':6}),(q,k,{'sequence_axis':2}),(q,k,{'position_source':'supplied'}),(q,k,{'start_position':1_000_000}),(q*float('nan'),k,{})]:
        with pytest.raises(ValueError):rotate(args[0],args[1],args[2])


def test_zero_position_head_independence_common_shift_and_causality():
    torch.manual_seed(42);q=torch.randn(2,2,5,4);k=torch.randn(2,2,5,4);v=torch.randn(2,2,5,3)
    original=(q.clone(),k.clone(),v.clone())
    r=rotate(q,k);shifted=rotate(q,k,{'start_position':31})
    torch.testing.assert_close(r['query'][:,:,0],q[:,:,0],atol=0,rtol=0)
    torch.testing.assert_close(r['query']@r['key'].transpose(-2,-1),shifted['query']@shifted['key'].transpose(-2,-1),atol=2e-5,rtol=2e-5)
    a=torch.nn.functional.scaled_dot_product_attention(r['query'],r['key'],v,is_causal=True)
    changed_q=q.clone();changed_k=k.clone();changed_v=v.clone();changed_q[:,:,3:]+=7;changed_k[:,:,3:]-=3;changed_v[:,:,3:]*=9
    changed=rotate(changed_q,changed_k)
    b=torch.nn.functional.scaled_dot_product_attention(changed['query'],changed['key'],changed_v,is_causal=True)
    torch.testing.assert_close(a[:,:,:3],b[:,:,:3],atol=0,rtol=0)
    for before,after in zip(original,(q,k,v)):torch.testing.assert_close(before,after,atol=0,rtol=0)
    unequal=rotate(q,k[:,:1]);assert unequal['key'].shape==(2,1,5,4)


def rotary_graph(config=None):
    from runner.tests.test_attention_runtime import attention_classifier
    g=attention_classifier('scaled_dot_product_attention',extra={'mask_mode':'causal'})
    g['nodes'].append({'id':'rotary','type':'rotary_position','config':config or {}})
    g['edges']=[e for e in g['edges'] if not(e['target']=='attention' and e['targetPort'] in ('query','key'))]
    for role in ('query','key'):
        g['edges'].append({'id':'rope-in-'+role,'source':'array','sourcePort':'out','target':'rotary','targetPort':role})
        g['edges'].append({'id':'rope-out-'+role,'source':'rotary','sourcePort':role,'target':'attention','targetPort':role})
    return g


@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('pairing',['adjacent','halves'])
def test_actual_graph_export_state_and_value_bypass(device,pairing):
    from runner.engine import GraphModel,REGISTRY,export_source
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
    model=GraphModel(rotary_graph({'pairing':pairing}),REGISTRY).to(device).eval()
    assert 'rotary' not in model.layer_keys,'RoPE must not acquire learned-module state'
    x=torch.tensor([[.2,-.4],[3.,1.]],device=device,requires_grad=True)
    actual,stages=model(x,inspect=True)
    assert set(stages['rotary'])=={'query','key'}
    assert not any('rotary' in key for key in model.state_dict())
    state={key:value.clone() for key,value in model.state_dict().items()}
    scope={};exec(export_source(model),scope)
    exported=scope['Model']().to(device).eval();exported.load_state_dict(state,strict=True)
    y=x.detach().clone().requires_grad_();replayed=exported(y)
    torch.testing.assert_close(replayed,actual,atol=0,rtol=0)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(),x)[0],torch.autograd.grad(replayed.sum(),y)[0],atol=0,rtol=0)
    for key,value in state.items():torch.testing.assert_close(model.state_dict()[key],value,atol=0,rtol=0)


def test_real_resume_and_changed_configuration_refusal(tmp_path):
    from runner.engine import run,REGISTRY
    from runner.tests.test_checkpoint_lifecycle import state,assert_nested_equal
    graph=rotary_graph();opts=dict(registry=REGISTRY,device='cpu',sample_limit=128,batch_size=64)
    full,first,resumed=(tmp_path/name for name in ('full','first','resumed'))
    run(graph,full,epochs=4,**opts);run(graph,first,epochs=2,**opts)
    parent=(first/'checkpoint.pt').read_bytes()
    run(graph,resumed,epochs=4,resume_checkpoint=first/'checkpoint.pt',**opts)
    for key in ('modelState','optimizerState','rngState'):assert_nested_equal(state(full)[key],state(resumed)[key])
    with pytest.raises(ValueError,match='semantic architecture'):
        run(rotary_graph({'base':1234}),tmp_path/'bad',epochs=4,resume_checkpoint=first/'checkpoint.pt',**opts)
    assert (first/'checkpoint.pt').read_bytes()==parent
