"""Scalar LazyLinear checks after its existing validated materialization path."""
import copy
import math
import pytest
import torch
from torch.nn.parameter import UninitializedParameter
from runner.engine import GraphModel,export_source
from runner.tests.test_linear_axes_independent import scalar_reference


def graph(rank,bias,inputs=3,outputs=2):
    prefix=[2 if i%2==0 else 1 for i in range(rank-2)] if max(inputs,outputs)<512 else [1]*(rank-2)
    width=math.prod(prefix)*inputs
    specs=[('input','input',{'features':2}),('expand','linear',{'in_features':2,'out_features':min(width,512),'bias':True})]
    links=[('input','expand','in')];last='expand';resolved=min(width,512)
    while resolved<width:
        ident=f'double{resolved}';specs.append((ident,'concat',{}))
        links.extend([(last,ident,'a'),(last,ident,'b')]);last=ident;resolved*=2
    assert resolved==width
    specs += [('shape','reshape',{'shape':prefix+[inputs]}),('tested','lazy_linear',{'out_features':outputs,'bias':bias}),
              ('flat','flatten',{'start_dim':1,'end_dim':-1}),('head','linear',{'in_features':math.prod(prefix)*outputs,'out_features':2,'bias':False}),('output','output',{})]
    for a,b in [(last,'shape'),('shape','tested'),('tested','flat'),('flat','head'),('head','output')]:links.append((a,b,'in'))
    return {'schemaVersion':2,'nodes':[{'id':i,'type':t,'name':i,'config':c} for i,t,c in specs],
            'edges':[{'id':str(i),'source':a,'sourcePort':'out','target':b,'targetPort':p} for i,(a,b,p) in enumerate(links)]},prefix


def ready(device):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('Actual MPS unavailable; no device acceptance')


def measure(device,rank,bias,batch,inputs=3,outputs=2):
    ready(device);authored,prefix=graph(rank,bias,inputs,outputs);original=copy.deepcopy(authored)
    model=GraphModel(authored).to(device);layer=model.layers['tested']
    assert isinstance(layer,torch.nn.LazyLinear) and isinstance(layer.weight,UninitializedParameter)
    assert 'in_features' not in model.nodes['tested']['config']
    model.materialize_lazy_parameters()
    assert layer.in_features==inputs and tuple(layer.weight.shape)==(outputs,inputs)
    state={k:v.clone() for k,v in model.state_dict().items()};cpu_rng=torch.get_rng_state().clone()
    mps_rng=torch.mps.get_rng_state().clone() if device=='mps' else None
    model.materialize_lazy_parameters()
    assert torch.equal(torch.get_rng_state(),cpu_rng)
    if mps_rng is not None:assert torch.equal(torch.mps.get_rng_state(),mps_rng)
    assert all(torch.equal(v,state[k]) for k,v in model.state_dict().items())
    assert authored==original
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([(i%11-5)/32 for i in range(inputs*outputs)],device=device).reshape(outputs,inputs))
        if bias:layer.bias.copy_(torch.tensor([(j%5-2)/16 for j in range(outputs)],device=device))
    shape=(batch,*prefix,inputs);out_shape=(*shape[:-1],outputs)
    storage=torch.empty((*shape[:-1],inputs*2),device=device)
    storage[...,::2]=torch.tensor([(i%17-8)/16 for i in range(math.prod(shape))],device=device).reshape(shape)
    storage[...,1::2]=-999
    x=storage[...,::2].detach().requires_grad_();assert not x.is_contiguous()
    up=torch.tensor([(i%13-6)/32 for i in range(math.prod(out_shape))],device=device).reshape(out_shape)
    values=lambda t:t.detach().cpu().flatten().tolist()
    oracle=scalar_reference(values(x),values(up),values(layer.weight),values(layer.bias) if bias else None,inputs,outputs)
    expected=[torch.tensor(v,device=device).reshape(s) for v,s in zip(oracle,[out_shape,shape,(outputs,inputs),(outputs,)])]
    source=export_source(model);assert 'nn.LazyLinear(' in source
    namespace={};exec(source,namespace);exported=namespace['Model']().to(device)
    assert isinstance(exported.layers['tested'].weight,UninitializedParameter)
    exported.load_state_dict(model.state_dict(),strict=True)
    assert not isinstance(exported.layers['tested'].weight,UninitializedParameter)
    atol,rtol=(1e-6,1e-5) if device=='cpu' else (1e-5,1e-4)
    before_x=x.detach().clone()
    for current in (layer,exported.layers['tested']):
        before={k:v.clone() for k,v in current.state_dict().items()}
        rng=torch.get_rng_state().clone();gpu_rng=torch.mps.get_rng_state().clone() if device=='mps' else None
        current.train();actual=current(x);derivatives=torch.autograd.grad(actual,(x,*current.parameters()),up)
        assert torch.equal(torch.get_rng_state(),rng)
        if gpu_rng is not None:assert torch.equal(torch.mps.get_rng_state(),gpu_rng)
        torch.testing.assert_close(actual,expected[0],atol=atol,rtol=rtol)
        for measured,target in zip(derivatives,expected[1:]):torch.testing.assert_close(measured,target,atol=atol,rtol=rtol)
        assert actual.shape==out_shape and actual.dtype==torch.float32
        assert torch.isfinite(actual).all() and all(torch.isfinite(v).all() for v in derivatives)
        assert set(before)==({'weight','bias'} if bias else {'weight'})
        assert sum(p.numel() for p in current.parameters())==outputs*(inputs+int(bias))
        assert all(p.grad is None for p in current.parameters())
        current.eval()
        with torch.no_grad():assert torch.equal(current(x),actual)
        assert all(torch.equal(v,before[k]) for k,v in current.state_dict().items())
        assert torch.equal(x,before_x)
        if batch>1:
            changed=x.detach().clone();changed[0]+=1
            assert torch.equal(current(changed)[1:],actual[1:])
    # Exercise complete generated architecture, initialized state, and every gradient.
    public=torch.tensor([[.25,-.5]]*batch,device=device,requires_grad=True)
    actual=model(public);copied=exported(public)
    torch.testing.assert_close(actual,copied,atol=atol,rtol=rtol)
    ga=torch.autograd.grad(actual,(public,*model.parameters()),torch.ones_like(actual))
    gb=torch.autograd.grad(copied,(public,*exported.parameters()),torch.ones_like(copied))
    for a,b in zip(ga,gb):torch.testing.assert_close(a,b,atol=atol,rtol=rtol)


@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('rank',range(2,9))
@pytest.mark.parametrize('bias',[False,True])
@pytest.mark.parametrize('batch',[1,3])
def test_scalar_leading_axes_materialization_and_export_state(device,rank,bias,batch):
    measure(device,rank,bias,batch)


@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('bias',[False,True])
@pytest.mark.parametrize('inputs,outputs',[(1,1),(1,512),(4096,1)])
def test_rank3_safe_inferred_input_output_endpoints(device,bias,inputs,outputs):
    measure(device,3,bias,3,inputs,outputs)
