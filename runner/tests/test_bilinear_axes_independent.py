"""Independent scalar Bilinear oracle over every supported product rank."""
import math
import pytest
import torch
from runner.engine import GraphModel,export_source


def graph(rank,bias,h1=3,h2=2,outputs=2):
    prefix=[2 if i%2==0 else 1 for i in range(rank-2)] if max(h1,h2,outputs)<512 else [1]*(rank-2)
    count=math.prod(prefix)
    specs=[('input','input',{'features':2}),('left','linear',{'in_features':2,'out_features':count*h1,'bias':True}),
           ('right','linear',{'in_features':2,'out_features':count*h2,'bias':False}),
           ('a','reshape',{'shape':prefix+[h1]}),('b','reshape',{'shape':prefix+[h2]}),
           ('tested','bilinear',{'in1_features':h1,'in2_features':h2,'out_features':outputs,'bias':bias}),
           ('flat','flatten',{'start_dim':1,'end_dim':-1}),('head','linear',{'in_features':count*outputs,'out_features':2,'bias':False}),('output','output',{})]
    links=[('input','left','in'),('input','right','in'),('left','a','in'),('right','b','in'),('a','tested','a'),('b','tested','b'),('tested','flat','in'),('flat','head','in'),('head','output','in')]
    return {'schemaVersion':2,'nodes':[{'id':i,'type':t,'name':i,'config':c} for i,t,c in specs],
            'edges':[{'id':str(i),'source':a,'sourcePort':'out','target':b,'targetPort':p} for i,(a,b,p) in enumerate(links)]},prefix


def oracle(a,b,w,bias,g,h1,h2,outputs):
    rows=len(a)//h1
    weight=lambda o,i,j:w[(o*h1+i)*h2+j]
    y=[math.fsum(a[r*h1+i]*weight(o,i,j)*b[r*h2+j] for i in range(h1) for j in range(h2))+(bias[o] if bias is not None else 0.) for r in range(rows) for o in range(outputs)]
    da=[math.fsum(g[r*outputs+o]*weight(o,i,j)*b[r*h2+j] for o in range(outputs) for j in range(h2)) for r in range(rows) for i in range(h1)]
    db=[math.fsum(g[r*outputs+o]*weight(o,i,j)*a[r*h1+i] for o in range(outputs) for i in range(h1)) for r in range(rows) for j in range(h2)]
    dw=[math.fsum(g[r*outputs+o]*a[r*h1+i]*b[r*h2+j] for r in range(rows)) for o in range(outputs) for i in range(h1) for j in range(h2)]
    dc=[math.fsum(g[r*outputs+o] for r in range(rows)) for o in range(outputs)]
    return y,da,db,dw,dc


def ready(device):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('Actual MPS unavailable; no device acceptance')


def operand(shape,device,shift):
    storage=torch.empty((*shape[:-1],shape[-1]*2),device=device)
    storage[...,::2]=torch.tensor([(i%13-6+shift)/16 for i in range(math.prod(shape))],device=device).reshape(shape)
    storage[...,1::2]=-999
    result=storage[...,::2].detach().requires_grad_();assert not result.is_contiguous()
    return result


def measure(device,rank,bias,batch,h1=3,h2=2,outputs=2):
    ready(device);value,prefix=graph(rank,bias,h1,h2,outputs);model=GraphModel(value).to(device)
    layer=model.layers['tested']
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([(i%11-5)/32 for i in range(layer.weight.numel())],device=device).reshape(layer.weight.shape))
        if bias:layer.bias.copy_(torch.tensor([(i%5-2)/16 for i in range(outputs)],device=device))
    a=operand((batch,*prefix,h1),device,0);b=operand((batch,*prefix,h2),device,2)
    shape=(batch,*prefix,outputs);g=torch.tensor([(i%7-3)/16 for i in range(math.prod(shape))],device=device).reshape(shape)
    values=lambda t:t.detach().cpu().flatten().tolist()
    expected=oracle(values(a),values(b),values(layer.weight),values(layer.bias) if bias else None,values(g),h1,h2,outputs)
    shapes=[shape,tuple(a.shape),tuple(b.shape),tuple(layer.weight.shape),(outputs,)]
    targets=[torch.tensor(v,device=device).reshape(s) for v,s in zip(expected,shapes)]
    namespace={};exec(export_source(model),namespace);exported=namespace['Model']().to(device);exported.load_state_dict(model.state_dict(),strict=True)
    atol,rtol=(1e-6,1e-5) if device=='cpu' else (1e-5,1e-4)
    original_a=a.detach().clone();original_b=b.detach().clone()
    for current in (layer,exported.layers['tested']):
        state={k:v.clone() for k,v in current.state_dict().items()}
        current.train();y=current(a,b);gradients=torch.autograd.grad(y,(a,b,*current.parameters()),g)
        torch.testing.assert_close(y,targets[0],atol=atol,rtol=rtol)
        for observed,target in zip(gradients,targets[1:]):torch.testing.assert_close(observed,target,atol=atol,rtol=rtol)
        assert y.shape==shape and y.dtype==torch.float32 and torch.isfinite(y).all()
        assert all(torch.isfinite(v).all() for v in gradients)
        assert set(state)==({'weight','bias'} if bias else {'weight'})
        assert sum(p.numel() for p in current.parameters())==outputs*(h1*h2+int(bias))
        assert all(p.grad is None for p in current.parameters())
        current.eval()
        with torch.no_grad():assert torch.equal(current(a,b),y)
        assert all(torch.equal(v,state[k]) for k,v in current.state_dict().items())
        assert torch.equal(a,original_a) and torch.equal(b,original_b)
        if batch>1:
            changed=a.detach().clone();changed[0]+=1
            assert torch.equal(current(changed,b)[1:],y[1:])
            changed=b.detach().clone();changed[0]-=1
            assert torch.equal(current(a,changed)[1:],y[1:])
    return model,exported


@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('rank',range(2,9))
@pytest.mark.parametrize('bias',[False,True])
@pytest.mark.parametrize('batch',[1,3])
def test_scalar_output_two_input_and_shared_parameter_gradients(device,rank,bias,batch):
    measure(device,rank,bias,batch)


@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('bias',[False,True])
@pytest.mark.parametrize('h1,h2,outputs',[(1,1,1),(512,1,1),(1,512,1),(1,1,512)])
def test_rank3_independent_width_endpoints(device,bias,h1,h2,outputs):
    measure(device,3,bias,3,h1,h2,outputs)


@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('rank',[5,8])
@pytest.mark.parametrize('bias',[False,True])
def test_high_rank_whole_graph_state_and_export_gradients(device,rank,bias):
    model,exported=measure(device,rank,bias,3)
    x=operand((3,2),device,0)
    actual=model(x);copied=exported(x)
    atol,rtol=(1e-6,1e-5) if device=='cpu' else (1e-5,1e-4)
    torch.testing.assert_close(actual,copied,atol=atol,rtol=rtol)
    up=torch.tensor([[.5,-.25]]*3,device=device)
    ga=torch.autograd.grad(actual,(x,*model.parameters()),up)
    gb=torch.autograd.grad(copied,(x,*exported.parameters()),up)
    for a,b in zip(ga,gb):torch.testing.assert_close(a,b,atol=atol,rtol=rtol)
