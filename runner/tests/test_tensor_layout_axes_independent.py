"""Scalar coordinate oracles for batch-preserving tensor layout operations."""
import itertools
import math
import pytest
import torch
from runner.tensor_layout import apply_layout
from runner.engine import GraphModel, export_source


def coordinates(shape):
    return itertools.product(*(range(n) for n in shape))


def offset(coord,shape):
    result=0
    for i,n in zip(coord,shape):result=result*n+i
    return result


def reference(shape,operation,config):
    """Return output shape and its source indices without tensor operations."""
    rank=len(shape)
    if operation=='transpose':
        a=config['dim0']%rank;b=config['dim1']%rank
        out=list(shape);out[a],out[b]=out[b],out[a]
        mapping=[]
        for coord in coordinates(out):
            original=list(coord);original[a],original[b]=original[b],original[a]
            mapping.append(offset(original,shape))
        return tuple(out),mapping
    if operation=='flatten':
        a=config['start_dim']%rank;b=config['end_dim']%rank
        out=shape[:a]+(math.prod(shape[a:b+1]),)+shape[b+1:]
    else:
        sample=list(config['shape'])
        if -1 in sample:sample[sample.index(-1)]=math.prod(shape[1:])//math.prod(n for n in sample if n!=-1)
        out=(shape[0],*sample)
    return tuple(out),list(range(math.prod(shape)))


def cases():
    result=[]
    for rank in range(2,9):
        sample=tuple(2 if i%2==0 else 3 for i in range(rank-1))
        specs=[('reshape',{'shape':[math.prod(sample)]}),('reshape',{'shape':[1,-1]}),
               ('transpose',{'dim0':1,'dim1':rank-1}),('transpose',{'dim0':-1,'dim1':-1}),
               ('flatten',{'start_dim':1,'end_dim':-1}),('flatten',{'start_dim':1,'end_dim':1}),
               ('flatten',{'start_dim':1,'end_dim':min(2,rank-1)}),('flatten',{'start_dim':max(1,rank-2),'end_dim':-1})]
        if rank in (5,8):specs += [('transpose',{'dim0':2,'dim1':rank-2}),('flatten',{'start_dim':2,'end_dim':rank-2})]
        for op,c in specs:
            if (sample,op,c) not in result:result.append((sample,op,c))
    return result


def ready(device):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('Actual MPS unavailable; no device acceptance')


def exercise(device,batch,sample,operation,config,dtype=torch.float32):
    ready(device);shape=(batch,*sample);n=math.prod(shape)
    storage=torch.empty((*shape[:-1],shape[-1]*2),device=device,dtype=dtype)
    storage[...,::2]=torch.tensor(list(range(n)),device=device,dtype=dtype).reshape(shape)
    storage[...,1::2]=-999
    x=storage[...,::2].detach();assert not x.is_contiguous()
    if dtype==torch.float32:x.requires_grad_()
    before=x.detach().clone();out_shape,mapping=reference(shape,operation,config)
    y=apply_layout(x,operation,config)
    assert tuple(y.shape)==out_shape and y.dtype==dtype
    expected=torch.tensor(mapping,device=device,dtype=dtype).reshape(out_shape)
    assert torch.equal(y,expected)
    assert torch.equal(x,before)
    # Equivalent negative aliases must describe precisely the same operation.
    if operation!='reshape':
        aliases={k:(v%len(shape))-len(shape) for k,v in config.items()}
        assert torch.equal(apply_layout(x,operation,aliases),y)
    if dtype==torch.float32:
        g=torch.tensor([(i%23-11)/8 for i in range(n)],device=device).reshape(out_shape)
        dx,=torch.autograd.grad(y,x,g)
        expected_dx=[0.]*n
        for output_index,input_index in enumerate(mapping):expected_dx[input_index]=(output_index%23-11)/8
        assert torch.equal(dx,torch.tensor(expected_dx,device=device).reshape(shape))
        assert x.grad is None # no optimizer or parameter state belongs to these transforms
    if batch>1:
        changed=x.detach().clone();changed[0]+=10000
        assert torch.equal(apply_layout(changed,operation,config)[1:],y[1:])


@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('batch',[1,3])
@pytest.mark.parametrize('sample,operation,config',cases())
def test_coordinate_order_arbitrary_gradients_and_aliases(device,batch,sample,operation,config):
    exercise(device,batch,sample,operation,config)


@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('rank',[3,8])
@pytest.mark.parametrize('operation,config',[('reshape',{'shape':[1,-1]}),('transpose',{'dim0':1,'dim1':-1}),('flatten',{'start_dim':1,'end_dim':-1})])
def test_integer_forward_only(device,rank,operation,config):
    exercise(device,3,(2,)*(rank-1),operation,config,torch.int64)


def compound():
    specs=[('input','input',{'features':2}),('project','linear',{'in_features':2,'out_features':12,'bias':True}),
           ('shape','reshape',{'shape':[2,1,2,3]}),('swap','transpose',{'dim0':1,'dim1':-1}),
           ('partial','flatten',{'start_dim':2,'end_dim':3}),('flat','reshape',{'shape':[-1]}),
           ('head','linear',{'in_features':12,'out_features':2,'bias':False}),('output','output',{})]
    return {'schemaVersion':2,'nodes':[{'id':i,'type':t,'name':i,'config':c} for i,t,c in specs],
            'edges':[{'id':str(i),'source':a[0],'sourcePort':'out','target':b[0],'targetPort':'in'} for i,(a,b) in enumerate(zip(specs,specs[1:]))]}


@pytest.mark.parametrize('device',['cpu','mps'])
def test_compound_graph_export_scalar_coordinates_and_input_gradient(device):
    ready(device);model=GraphModel(compound()).to(device)
    with torch.no_grad():
        for p in model.parameters():p.copy_(torch.tensor([(i%9-4)/32 for i in range(p.numel())],device=device).reshape(p.shape))
    public=torch.tensor([[.5,-.25],[-.75,.125],[1.,-.5]],device=device,requires_grad=True)
    first=model.layers['project'];head=model.layers['head']
    w=first.weight.detach().cpu().tolist();b=first.bias.detach().cpu().tolist();h=head.weight.detach().cpu().tolist()
    shape=(3,12);mapping=list(range(36))
    for ident in ('shape','swap','partial','flat'):
        node=model.nodes[ident];shape,indices=reference(shape,node['type'],node['config']);mapping=[mapping[i] for i in indices]
    px=public.detach().cpu().tolist()
    projected=[math.fsum(px[row][k]*w[j][k] for k in range(2))+b[j] for row in range(3) for j in range(12)]
    ordered=[projected[i] for i in mapping]
    expected=[math.fsum(ordered[row*12+j]*h[o][j] for j in range(12)) for row in range(3) for o in range(2)]
    up=[[.25,-.5],[.125,.75],[-.25,.5]];dx=[]
    for row in range(3):
        routed=[0.]*12
        for j in range(12):routed[mapping[row*12+j]-row*12]=math.fsum(up[row][o]*h[o][j] for o in range(2))
        dx.extend(math.fsum(routed[j]*w[j][k] for j in range(12)) for k in range(2))
    namespace={};exec(export_source(model),namespace);exported=namespace['Model']().to(device);exported.load_state_dict(model.state_dict(),strict=True)
    assert set(model.layer_keys)=={'project','head'} # transforms own no state
    gradients=[]
    for current in (model,exported):
        actual=current(public)
        assert torch.equal(actual,torch.tensor(expected,device=device).reshape(3,2))
        measured=torch.autograd.grad(actual,(public,*current.parameters()),torch.tensor(up,device=device))
        assert torch.equal(measured[0],torch.tensor(dx,device=device).reshape(3,2))
        gradients.append(measured)
    assert all(torch.equal(a,b) for a,b in zip(*gradients))


def test_negative_batch_alias_and_rank_dependent_axes_are_rejected():
    for rank in (2,8):
        x=torch.zeros((1,)+(2,)*(rank-1))
        for axis in (-rank,rank):
            for operation,config in [('transpose',{'dim0':axis,'dim1':1}),('flatten',{'start_dim':axis,'end_dim':-1})]:
                with pytest.raises(ValueError):apply_layout(x,operation,config)
