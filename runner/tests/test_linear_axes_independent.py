"""Independent scalar Linear values and derivatives over leading dimensions.

Existing rank-2 matrix and lifecycle suites cover training/resume policies.
These supplemental checks do not widen the shipped dtype or width contract.
"""
import math
import pytest
import torch
from runner.engine import GraphModel, export_source


def graph(rank, bias, inputs=3, outputs=2):
    # Boundary widths use one leading location so intermediary widths stay <=512.
    leading=[2]*(rank-2) if max(inputs,outputs)<512 else [1]*(rank-2)
    sample=leading+[inputs]; out=math.prod(leading)*outputs
    specs=[('input','input',{'features':2}),('expand','linear',{'in_features':2,'out_features':math.prod(sample),'bias':True}),
           ('shape','reshape',{'shape':sample}),('tested','linear',{'in_features':inputs,'out_features':outputs,'bias':bias}),
           ('flat','flatten',{'start_dim':1,'end_dim':-1}),('head','linear',{'in_features':out,'out_features':2,'bias':False}),('output','output',{})]
    return {'schemaVersion':2,'nodes':[{'id':i,'type':t,'name':i,'x':0,'y':0,'config':c} for i,t,c in specs],
            'edges':[{'id':str(i),'source':a[0],'sourcePort':'out','target':b[0],'targetPort':'in'} for i,(a,b) in enumerate(zip(specs,specs[1:]))]},sample


def scalar_reference(x,g,w,b,inputs,outputs):
    y=[];dx=[];dw=[[] for _ in w];db=[[] for _ in range(outputs)]
    for start in range(0,len(x),inputs):
        row=x[start:start+inputs];up=g[(start//inputs)*outputs:(start//inputs+1)*outputs]
        y.extend(math.fsum(row[k]*w[j*inputs+k] for k in range(inputs))+(b[j] if b is not None else 0.) for j in range(outputs))
        dx.extend(math.fsum(up[j]*w[j*inputs+k] for j in range(outputs)) for k in range(inputs))
        for j in range(outputs):
            db[j].append(up[j])
            for k in range(inputs):dw[j*inputs+k].append(up[j]*row[k])
    return y,dx,[math.fsum(v) for v in dw],[math.fsum(v) for v in db]


def measure(device,rank,bias,batch,inputs=3,outputs=2):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('Actual MPS unavailable; no device acceptance')
    g,sample=graph(rank,bias,inputs,outputs);model=GraphModel(g).to(device)
    layer=model.layers[model.layer_keys['tested']]
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([(i%11-5)/32 for i in range(inputs*outputs)],device=device).reshape(outputs,inputs))
        if bias:layer.bias.copy_(torch.tensor([(j%5-2)/16 for j in range(outputs)],device=device))
    shape=(batch,*sample);out_shape=(*shape[:-1],outputs);count=math.prod(shape)
    storage=torch.empty((*shape[:-1],inputs*2),device=device)
    storage[...,::2]=torch.tensor([(i%17-8)/16 for i in range(count)],device=device).reshape(shape)
    storage[...,1::2]=999
    x=storage[...,::2].detach().requires_grad_();assert not x.is_contiguous()
    up=torch.tensor([(i%13-6)/32 for i in range(math.prod(out_shape))],device=device).reshape(out_shape)
    ref=scalar_reference(x.detach().cpu().flatten().tolist(),up.cpu().flatten().tolist(),layer.weight.detach().cpu().flatten().tolist(),layer.bias.detach().cpu().tolist() if bias else None,inputs,outputs)
    expected=[torch.tensor(v,device=device).reshape(s) for v,s in zip(ref,[out_shape,shape,(outputs,inputs),(outputs,)])]
    namespace={};exec(export_source(model),namespace);exported=namespace['Model']().to(device);exported.load_state_dict(model.state_dict(),strict=True)
    layers=[layer,exported.layers[model.layer_keys['tested']]]
    atol,rtol=(1e-6,1e-5) if device=='cpu' else (1e-5,1e-4)
    for current in layers:
        before={k:v.clone() for k,v in current.state_dict().items()}
        current.train();actual=current(x);derivatives=torch.autograd.grad(actual,(x,*current.parameters()),up)
        torch.testing.assert_close(actual,expected[0],atol=atol,rtol=rtol)
        for measured,target in zip(derivatives,expected[1:]):torch.testing.assert_close(measured,target,atol=atol,rtol=rtol)
        assert actual.shape==out_shape and actual.dtype==torch.float32 and torch.isfinite(actual).all()
        assert set(current.state_dict())==({'weight','bias'} if bias else {'weight'})
        assert sum(p.numel() for p in current.parameters())==inputs*outputs+(outputs if bias else 0)
        assert all(p.grad is None for p in current.parameters())
        current.eval()
        with torch.no_grad():inferred=current(x)
        assert torch.equal(inferred,actual)
        assert all(torch.equal(v,before[k]) for k,v in current.state_dict().items())
        # A batch row must never influence another row, even with many sample axes.
        if batch>1:
            changed=x.detach().clone();changed[0]+=1
            assert torch.equal(current(changed)[1:],actual[1:])
    # Execute the complete generated graph as well as its scalar-verified module.
    public=torch.tensor([[.25,-.5]]*batch,device=device,requires_grad=True)
    actual=model(public);copied=exported(public)
    torch.testing.assert_close(actual,copied,atol=atol,rtol=rtol)
    upstream=torch.ones_like(actual)
    ga=torch.autograd.grad(actual,(public,*model.parameters()),upstream)
    gb=torch.autograd.grad(copied,(public,*exported.parameters()),upstream)
    for a,b in zip(ga,gb):torch.testing.assert_close(a,b,atol=atol,rtol=rtol)


@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('rank',range(2,9))
@pytest.mark.parametrize('bias',[False,True])
@pytest.mark.parametrize('batch',[1,3])
def test_leading_axes_scalar_values_all_gradients_export_and_mode(device,rank,bias,batch):
    measure(device,rank,bias,batch)


@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('bias',[False,True])
@pytest.mark.parametrize('inputs,outputs',[(1,1),(1,512),(512,1)])
def test_rank3_width_endpoints(device,bias,inputs,outputs):
    measure(device,3,bias,3,inputs,outputs)
