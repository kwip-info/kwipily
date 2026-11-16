"""Stdlib indexed grouped cross-correlation and all original scalar gradients."""
import itertools
import math
import pytest
import torch
from runner.engine import GraphModel,export_source


def coordinates(shape):return itertools.product(*(range(n) for n in shape))
def index(coord,shape):
    result=0
    for i,n in zip(coord,shape):result=result*n+i
    return result


def configuration(dim,case,bias):
    c=dict(in_channels=2,out_channels=4,kernel_size=[2]*dim,stride=[1]*dim,dilation=[1]*dim,
           groups=2,bias=bias,padding_policy='same',padding=[0]*dim,padding_mode=case if case in ('zeros','reflect','replicate','circular') else 'zeros')
    if case=='valid-dilation':c.update(padding_policy='valid',dilation=[2]+[1]*(dim-1))
    elif case=='explicit-stride':c.update(padding_policy='explicit',padding=[1]*dim,stride=[2]+[1]*(dim-1),groups=1)
    elif case=='explicit-circular':c.update(padding_policy='explicit',padding=[1]*dim,padding_mode='circular')
    return c


def geometry(spatial,c):
    effective=[d*(k-1)+1 for d,k in zip(c['dilation'],c['kernel_size'])]
    left=[p if c['padding_policy']=='explicit' else (e-1)//2 if c['padding_policy']=='same' else 0 for p,e in zip(c['padding'],effective)]
    right=[p if c['padding_policy']=='explicit' else e-1-l if c['padding_policy']=='same' else 0 for p,e,l in zip(c['padding'],effective,left)]
    out=tuple((n+l+r-e)//s+1 for n,l,r,e,s in zip(spatial,left,right,effective,c['stride']))
    return out,left


def scalar_reference(xshape,x,w,b,g,c):
    spatial=xshape[2:];out,left=geometry(spatial,c);shape=(xshape[0],c['out_channels'],*out)
    per_in=c['in_channels']//c['groups'];per_out=c['out_channels']//c['groups']
    wshape=(c['out_channels'],per_in,*c['kernel_size'])
    dx=[[] for _ in x];dw=[[] for _ in w];db=[[] for _ in range(c['out_channels'])];y=[]
    for n,o,*point in coordinates(shape):
        oi=index((n,o,*point),shape);terms=[];group=o//per_out
        for channel in range(per_in):
            for kernel in coordinates(c['kernel_size']):
                source=[p*s-l+k*d for p,s,l,k,d in zip(point,c['stride'],left,kernel,c['dilation'])]
                valid=True
                for axis,(v,length) in enumerate(zip(source,spatial)):
                    if 0<=v<length:continue
                    if c['padding_mode']=='zeros':valid=False;break
                    if c['padding_mode']=='replicate':source[axis]=max(0,min(length-1,v))
                    elif c['padding_mode']=='circular':source[axis]=v%length
                    else:source[axis]=-v if v<0 else 2*length-2-v
                if not valid:continue
                xi=index((n,group*per_in+channel,*source),xshape);wi=index((o,channel,*kernel),wshape)
                terms.append(x[xi]*w[wi]);dx[xi].append(g[oi]*w[wi]);dw[wi].append(g[oi]*x[xi])
        y.append(math.fsum(terms)+(b[o] if b is not None else 0.));db[o].append(g[oi])
    return shape,y,[math.fsum(v) for v in dx],[math.fsum(v) for v in dw],[math.fsum(v) for v in db]


def graph(dim,c):
    spatial=(5,4,3)[:dim];out,_=geometry(spatial,c);settings=dict(c)
    if dim==1:
        for k in ('kernel_size','stride','dilation','padding'):settings[k]=settings[k][0]
    specs=[('input','input',{'features':2}),('expand','linear',{'in_features':2,'out_features':2*math.prod(spatial),'bias':True}),
           ('shape','reshape',{'shape':[2,*spatial]}),('conv',f'conv{dim}d',settings),('flat','flatten',{'start_dim':1,'end_dim':-1}),
           ('head','linear',{'in_features':4*math.prod(out),'out_features':2,'bias':True}),('output','output',{})]
    return {'schemaVersion':2,'nodes':[{'id':i,'type':t,'name':i,'config':s} for i,t,s in specs],
            'edges':[{'id':str(i),'source':a[0],'sourcePort':'out','target':b[0],'targetPort':'in'} for i,(a,b) in enumerate(zip(specs,specs[1:]))]},spatial


@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('dim',[1,2,3])
@pytest.mark.parametrize('case',['zeros','reflect','replicate','circular','valid-dilation','explicit-stride','explicit-circular'])
@pytest.mark.parametrize('bias',[False,True])
@pytest.mark.parametrize('batch',[1,3])
def test_scalar_grouped_cross_correlation_all_gradients_and_generated_export(device,dim,case,bias,batch):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('Actual MPS unavailable; no device acceptance')
    c=configuration(dim,case,bias);authored,spatial=graph(dim,c);model=GraphModel(authored).to(device);layer=model.layers['conv']
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([(i%11-5)/32 for i in range(layer.weight.numel())],device=device).reshape(layer.weight.shape))
        if bias:layer.bias.copy_(torch.tensor([-.125,0,.125,.25],device=device))
    xshape=(batch,2,*spatial);storage=torch.empty((*xshape[:-1],xshape[-1]*2),device=device)
    storage[...,::2]=torch.tensor([(i%13-6)/16 for i in range(math.prod(xshape))],device=device).reshape(xshape);storage[...,1::2]=-999
    x=storage[...,::2].detach().requires_grad_();assert not x.is_contiguous()
    out,_=geometry(spatial,c);shape=(batch,4,*out);g=torch.tensor([(i%7-3)/16 for i in range(math.prod(shape))],device=device).reshape(shape)
    values=lambda t:t.detach().cpu().flatten().tolist()
    ref=scalar_reference(xshape,values(x),values(layer.weight),values(layer.bias) if bias else None,values(g),c)
    expected=[torch.tensor(v,device=device).reshape(s) for v,s in zip(ref[1:],[shape,xshape,layer.weight.shape,(4,)])]
    source=export_source(model);namespace={};exec(source,namespace);exported=namespace['Model']().to(device);exported.load_state_dict(model.state_dict(),strict=True)
    atol,rtol=(1e-6,1e-5) if device=='cpu' else (1e-5,1e-4)
    before_x=x.detach().clone()
    for current in (layer,exported.layers['conv']):
        before={k:v.clone() for k,v in current.state_dict().items()}
        actual=model.execute_node('conv',{'shape':x},None) if current is layer else current(x)
        measured=torch.autograd.grad(actual,(x,*current.parameters()),g)
        torch.testing.assert_close(actual,expected[0],atol=atol,rtol=rtol)
        for a,b in zip(measured,expected[1:]):torch.testing.assert_close(a,b,atol=atol,rtol=rtol)
        assert actual.shape==shape and actual.dtype==torch.float32
        assert torch.isfinite(actual).all() and all(torch.isfinite(v).all() for v in measured)
        current.eval()
        with torch.no_grad():assert torch.equal(current(x),actual)
        assert torch.equal(x,before_x) and all(torch.equal(v,before[k]) for k,v in current.state_dict().items())
        assert set(before)==({'weight','bias'} if bias else {'weight'})
        assert sum(p.numel() for p in current.parameters())==4*((2//c['groups'])*math.prod(c['kernel_size'])+int(bias))
        if batch>1:
            changed=x.detach().clone();changed[0]+=1
            assert torch.equal(current(changed)[1:],actual[1:])
    # Actual generated complete graph and every surrounding parameter derivative.
    public=torch.tensor([[.25,-.5]]*batch,device=device,requires_grad=True)
    actual=model(public);copied=exported(public)
    torch.testing.assert_close(actual,copied,atol=atol,rtol=rtol)
    ga=torch.autograd.grad(actual,(public,*model.parameters()),torch.ones_like(actual))
    gb=torch.autograd.grad(copied,(public,*exported.parameters()),torch.ones_like(copied))
    for a,b in zip(ga,gb):torch.testing.assert_close(a,b,atol=atol,rtol=rtol)
