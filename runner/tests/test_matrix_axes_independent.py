"""Independent indexed scalar matrix products and broadcast gradient reductions."""
import itertools
import math
import pytest
import torch
from runner.matrix_multiply import matrix_multiply
from runner.engine import GraphModel,export_source


def coords(shape):return itertools.product(*(range(n) for n in shape))
def index(coord,shape):
    value=0
    for c,n in zip(coord,shape):value=value*n+c
    return value


def scalar_reference(ashape,bshape,a,b,g):
    prefix_rank=max(len(ashape),len(bshape))-3
    ap=(1,)*(prefix_rank-len(ashape)+3)+ashape[1:-2]
    bp=(1,)*(prefix_rank-len(bshape)+3)+bshape[1:-2]
    prefix=tuple(max(x,y) for x,y in zip(ap,bp));m,k=ashape[-2:];n=bshape[-1]
    shape=(ashape[0],*prefix,m,n);out=[];da=[[] for _ in a];db=[[] for _ in b]
    for row,*parts in coords((ashape[0],*prefix)):
        acoord=tuple(0 if size==1 else v for size,v in zip(ap,parts))
        bcoord=tuple(0 if size==1 else v for size,v in zip(bp,parts))
        acoord=acoord[len(ap)-len(ashape[1:-2]):];bcoord=bcoord[len(bp)-len(bshape[1:-2]):]
        for i,j in coords((m,n)):
            terms=[];out_index=index((row,*parts,i,j),shape)
            for p in range(k):
                ai=index((row,*acoord,i,p),ashape);bi=index((row,*bcoord,p,j),bshape)
                terms.append(a[ai]*b[bi]);da[ai].append(g[out_index]*b[bi]);db[bi].append(g[out_index]*a[ai])
            out.append(math.fsum(terms))
    return shape,out,[math.fsum(v) for v in da],[math.fsum(v) for v in db]


def shapes(rank,mode):
    prefix=(2,)*(rank-3)
    if mode=='equal':return prefix+(2,3),prefix+(3,2)
    if mode=='singleton':return tuple(1 if i%2 else 2 for i in range(rank-3))+(2,3),tuple(2 if i%2 else 1 for i in range(rank-3))+(3,2)
    return prefix+(2,3),(3,2)


def ready(device):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('Actual MPS unavailable; no device acceptance')


def operand(shape,device,shift):
    storage=torch.empty((*shape[:-1],shape[-1]*2),device=device)
    storage[...,::2]=torch.tensor([(i%13-6+shift)/16 for i in range(math.prod(shape))],device=device).reshape(shape)
    storage[...,1::2]=-999
    out=storage[...,::2].detach().requires_grad_();assert not out.is_contiguous()
    return out


def measure(device,batch,left,right):
    ready(device);a=operand((batch,*left),device,0);b=operand((batch,*right),device,2)
    before=(a.detach().clone(),b.detach().clone())
    y=matrix_multiply(a,b)
    g=torch.tensor([(i%11-5)/16 for i in range(y.numel())],device=device).reshape(y.shape)
    values=lambda t:t.detach().cpu().flatten().tolist()
    shape,expected,da,db=scalar_reference(tuple(a.shape),tuple(b.shape),values(a),values(b),values(g))
    atol,rtol=(1e-6,1e-5) if device=='cpu' else (1e-5,1e-4)
    torch.testing.assert_close(y,torch.tensor(expected,device=device).reshape(shape),atol=atol,rtol=rtol)
    gradients=torch.autograd.grad(y,(a,b),g)
    for current,wanted,s in zip(gradients,[da,db],[a.shape,b.shape]):torch.testing.assert_close(current,torch.tensor(wanted,device=device).reshape(s),atol=atol,rtol=rtol)
    assert tuple(y.shape)==shape and y.dtype==torch.float32
    assert torch.isfinite(y).all() and all(torch.isfinite(v).all() for v in gradients)
    assert torch.equal(a,before[0]) and torch.equal(b,before[1])
    assert a.grad is None and b.grad is None
    if batch>1:
        changed=a.detach().clone();changed[0]+=1
        assert torch.equal(matrix_multiply(changed,b)[1:],y[1:])
        isolated=torch.autograd.grad(matrix_multiply(a,b)[0].sum(),(a,b))
        assert all(torch.count_nonzero(v[1:])==0 for v in isolated)


@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('batch',[1,3])
@pytest.mark.parametrize('rank,mode',[(r,m) for r in range(3,9) for m in (['equal'] if r==3 else ['equal','singleton','shorter'])])
def test_indexed_product_and_broadcast_reduced_both_input_gradients(device,batch,rank,mode):
    measure(device,batch,*shapes(rank,mode))


@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('left,right',[((1,1),(1,1)),((1,512),(512,1)),((512,1),(1,1)),((1,1),(1,512))])
def test_safe_degenerate_and_wide_matrix_controls(device,left,right):
    measure(device,3,left,right)


def graph(left,right):
    rank=max(len(left),len(right));ap=(1,)*(rank-len(left))+left;bp=(1,)*(rank-len(right))+right
    product=tuple(max(x,y) for x,y in zip(ap[:-2],bp[:-2]))+(left[-2],right[-1])
    specs=[('input','input',{'features':2}),('left','linear',{'in_features':2,'out_features':math.prod(left),'bias':True}),
           ('right','linear',{'in_features':2,'out_features':math.prod(right),'bias':False}),('a','reshape',{'shape':list(left)}),
           ('b','reshape',{'shape':list(right)}),('product','matmul',{}),('flat','flatten',{'start_dim':1,'end_dim':-1}),
           ('head','linear',{'in_features':math.prod(product),'out_features':2,'bias':False}),('output','output',{})]
    links=[('input','left','in'),('input','right','in'),('left','a','in'),('right','b','in'),('a','product','a'),('b','product','b'),('product','flat','in'),('flat','head','in'),('head','output','in')]
    return {'schemaVersion':2,'nodes':[{'id':i,'type':t,'name':i,'config':c} for i,t,c in specs],
            'edges':[{'id':str(i),'source':a,'sourcePort':'out','target':b,'targetPort':p} for i,(a,b,p) in enumerate(links)]}


@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('rank,mode',[(5,'singleton'),(8,'shorter')])
def test_real_graph_scalar_stage_and_generated_state_forward_all_gradients(device,rank,mode):
    ready(device);model=GraphModel(graph(*shapes(rank,mode))).to(device)
    with torch.no_grad():
        for p in model.parameters():p.copy_(torch.tensor([(i%7-3)/32 for i in range(p.numel())],device=device).reshape(p.shape))
    x=operand((3,2),device,0);actual,values=model(x,inspect=True)
    a,b=values['a'],values['b'];y=values['product'];up=torch.tensor([(i%11-5)/16 for i in range(y.numel())],device=device).reshape(y.shape)
    vector=lambda t:t.detach().cpu().flatten().tolist()
    shape,wanted,da,db=scalar_reference(tuple(a.shape),tuple(b.shape),vector(a),vector(b),vector(up))
    atol,rtol=(1e-6,1e-5) if device=='cpu' else (1e-5,1e-4)
    torch.testing.assert_close(y,torch.tensor(wanted,device=device).reshape(shape),atol=atol,rtol=rtol)
    for observed,expected,s in zip(torch.autograd.grad(y,(a,b),up,retain_graph=True),[da,db],[a.shape,b.shape]):torch.testing.assert_close(observed,torch.tensor(expected,device=device).reshape(s),atol=atol,rtol=rtol)
    source=export_source(model);assert 'runner' not in source
    namespace={};exec(source,namespace);exported=namespace['Model']().to(device);exported.load_state_dict(model.state_dict(),strict=True)
    assert set(model.layer_keys)=={'left','right','head'} # matrix product owns no parameters
    copied=exported(x)
    torch.testing.assert_close(actual,copied,atol=atol,rtol=rtol)
    ga=torch.autograd.grad(actual,(x,*model.parameters()),torch.ones_like(actual))
    gb=torch.autograd.grad(copied,(x,*exported.parameters()),torch.ones_like(copied))
    for a,b in zip(ga,gb):torch.testing.assert_close(a,b,atol=atol,rtol=rtol)


def test_batch_and_sample_broadcast_refusal_and_preallocation_budget(monkeypatch):
    from runner.tensor_layout import BatchedLayout
    with pytest.raises(ValueError):matrix_multiply(torch.ones(3,2,3),torch.ones(1,3,2))
    with pytest.raises(ValueError):matrix_multiply(torch.ones(1,2,2,3),torch.ones(1,3,3,2))
    with pytest.raises(ValueError):BatchedLayout((2,3)).matmul(BatchedLayout((4,2)))
    a=torch.ones(1,1,1).expand(2,10000,1);b=torch.ones(1,1,1).expand(2,1,10000)
    monkeypatch.setattr(torch,'matmul',lambda *a,**k:(_ for _ in ()).throw(AssertionError('allocated before budget')))
    with pytest.raises(ValueError,match='budget'):matrix_multiply(a,b)
