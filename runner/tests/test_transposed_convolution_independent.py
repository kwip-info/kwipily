"""Independent stdlib scatter-add values and all transposed-convolution gradients."""
import itertools
import math
import pytest
import torch
from runner.engine import GraphModel, export_source


def coordinates(shape):
    return itertools.product(*(range(n) for n in shape))


def index(coord, shape):
    offset = 0
    for coordinate, extent in zip(coord, shape):
        offset = offset * extent + coordinate
    return offset


def configuration(dim, case, bias):
    c = dict(in_channels=2, out_channels=4, kernel_size=[3]*dim,
             stride=[1]*dim, padding=[0]*dim, output_padding=[0]*dim,
             dilation=[1]*dim, groups=2, bias=bias)
    if case == 'crop':
        c.update(padding=[2]+[1]*(dim-1), groups=1)
    elif case == 'stride-extent':
        c.update(kernel_size=[3]*dim, stride=[2]*dim, padding=[1]*dim,
                 output_padding=[1]*dim, groups=1)
    elif case == 'dilated-extent':
        # output_padding == stride is legal when it is smaller than dilation.
        c.update(kernel_size=[2]*dim, dilation=[2]*dim, padding=[1]*dim,
                 output_padding=[1]*dim)
    elif case == 'stride-dilation':
        c.update(kernel_size=[2]*dim, stride=[2]*dim, dilation=[2]*dim,
                 padding=[1]*dim, output_padding=[1]*dim)
    elif case == 'anisotropic':
        c.update(kernel_size=[2,3,2][:dim], stride=[2,1,2][:dim],
                 dilation=[1,2,1][:dim], padding=[1,1,0][:dim],
                 output_padding=[1,0,1][:dim], groups=1)
    return c


def geometry(spatial, c):
    return tuple((n-1)*s - 2*p + d*(k-1) + o + 1
                 for n,s,p,d,k,o in zip(spatial,c['stride'],c['padding'],
                                        c['dilation'],c['kernel_size'],c['output_padding']))


def scalar_reference(xshape, x, w, b, upstream, c):
    shape = (xshape[0], c['out_channels'], *geometry(xshape[2:], c))
    per_in = c['in_channels']//c['groups']
    per_out = c['out_channels']//c['groups']
    wshape = (c['in_channels'], per_out, *c['kernel_size'])
    y = [[] for _ in upstream]
    dx = [[] for _ in x]
    dw = [[] for _ in w]
    db = [[] for _ in range(c['out_channels'])]
    # An input contributes to multiple output positions; overlaps sum, not replace.
    for n, channel, *point in coordinates(xshape):
        xi = index((n,channel,*point),xshape)
        group = channel//per_in
        for local_out in range(per_out):
            output_channel = group*per_out + local_out
            for kernel in coordinates(c['kernel_size']):
                destination = tuple(v*s-p+k*d for v,s,p,k,d in
                                    zip(point,c['stride'],c['padding'],kernel,c['dilation']))
                if any(v < 0 or v >= extent for v,extent in zip(destination,shape[2:])):
                    continue
                yi = index((n,output_channel,*destination),shape)
                wi = index((channel,local_out,*kernel),wshape)
                y[yi].append(x[xi]*w[wi])
                dx[xi].append(upstream[yi]*w[wi])
                dw[wi].append(upstream[yi]*x[xi])
    values = []
    for coordinate in coordinates(shape):
        yi = index(coordinate,shape)
        values.append(math.fsum(y[yi]) + (b[coordinate[1]] if b is not None else 0.))
        db[coordinate[1]].append(upstream[yi])
    return shape, values, [math.fsum(v) for v in dx], [math.fsum(v) for v in dw], [math.fsum(v) for v in db]


def graph(dim, c):
    spatial = (3,2,2)[:dim]
    settings = dict(c)
    if dim == 1:
        for key in ('kernel_size','stride','padding','output_padding','dilation'):
            settings[key] = settings[key][0]
    specs = [('input','input',{'features':2}),
             ('expand','linear',{'in_features':2,'out_features':2*math.prod(spatial),'bias':True}),
             ('shape','reshape',{'shape':[2,*spatial]}),
             ('transpose',f'conv_transpose{dim}d',settings),
             ('flat','flatten',{'start_dim':1,'end_dim':-1}),
             ('head','linear',{'in_features':4*math.prod(geometry(spatial,c)),'out_features':2,'bias':True}),
             ('output','output',{})]
    return {'schemaVersion':2,'nodes':[{'id':i,'type':t,'name':i,'config':s} for i,t,s in specs],
            'edges':[{'id':str(i),'source':a[0],'sourcePort':'out','target':b[0],'targetPort':'in'}
                     for i,(a,b) in enumerate(zip(specs,specs[1:]))]}, spatial


@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('dim',[1,2,3])
@pytest.mark.parametrize('case',['overlap','crop','stride-extent','dilated-extent','stride-dilation','anisotropic'])
@pytest.mark.parametrize('bias',[False,True])
@pytest.mark.parametrize('batch',[1,3])
def test_independent_scatter_all_gradients_real_worker_and_export(device,dim,case,bias,batch):
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('Actual MPS unavailable; no MPS acceptance')
    c = configuration(dim,case,bias)
    authored,spatial = graph(dim,c)
    model = GraphModel(authored).to(device)
    layer = model.layers['transpose']
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([(i%11-5)/32 for i in range(layer.weight.numel())],device=device).reshape(layer.weight.shape))
        if bias:
            layer.bias.copy_(torch.tensor([-.125,0,.125,.25],device=device))
    xshape = (batch,2,*spatial)
    storage = torch.empty((*xshape[:-1],xshape[-1]*2),device=device)
    storage[...,::2] = torch.tensor([(i%13-6)/16 for i in range(math.prod(xshape))],device=device).reshape(xshape)
    storage[...,1::2] = -999
    x = storage[...,::2].detach().requires_grad_()
    assert not x.is_contiguous()
    shape = (batch,4,*geometry(spatial,c))
    upstream = torch.tensor([(i%7-3)/16 for i in range(math.prod(shape))],device=device).reshape(shape)
    values = lambda t: t.detach().cpu().flatten().tolist()
    reference = scalar_reference(xshape,values(x),values(layer.weight),values(layer.bias) if bias else None,values(upstream),c)
    expected = [torch.tensor(v,device=device).reshape(s) for v,s in
                zip(reference[1:],[shape,xshape,layer.weight.shape,(4,)])]
    namespace = {}
    exec(export_source(model),namespace)
    exported = namespace['Model']().to(device)
    exported.load_state_dict(model.state_dict(),strict=True)
    atol,rtol = (1e-6,1e-5) if device == 'cpu' else (1e-5,1e-4)
    before_x = x.detach().clone()
    for current in (layer,exported.layers['transpose']):
        before = {key:value.clone() for key,value in current.state_dict().items()}
        actual = model.execute_node('transpose',{'shape':x},None) if current is layer else current(x)
        gradients = torch.autograd.grad(actual,(x,*current.parameters()),upstream)
        torch.testing.assert_close(actual,expected[0],atol=atol,rtol=rtol)
        for measured,independent in zip(gradients,expected[1:]):
            torch.testing.assert_close(measured,independent,atol=atol,rtol=rtol)
        assert actual.shape == shape and actual.dtype == torch.float32
        assert torch.isfinite(actual).all() and all(torch.isfinite(v).all() for v in gradients)
        current.eval()
        with torch.no_grad():
            assert torch.equal(current(x),actual)
        assert torch.equal(x,before_x)
        assert all(torch.equal(value,before[key]) for key,value in current.state_dict().items())
        assert set(before) == ({'weight','bias'} if bias else {'weight'})
        assert sum(p.numel() for p in current.parameters()) == 2*(4//c['groups'])*math.prod(c['kernel_size']) + int(bias)*4
        if batch > 1:
            changed = x.detach().clone()
            changed[0] += 1
            assert torch.equal(current(changed)[1:],actual[1:])
    # A real entire graph has ordinary trainable outer layers; all derivatives survive export/state load.
    public = torch.tensor([[.25,-.5]]*batch,device=device,requires_grad=True)
    actual,copied = model(public),exported(public)
    torch.testing.assert_close(actual,copied,atol=atol,rtol=rtol)
    first = torch.autograd.grad(actual,(public,*model.parameters()),torch.ones_like(actual))
    second = torch.autograd.grad(copied,(public,*exported.parameters()),torch.ones_like(copied))
    for a,b in zip(first,second):
        torch.testing.assert_close(a,b,atol=atol,rtol=rtol)


def test_output_padding_extends_scatter_domain_instead_of_appending_zero_cells():
    c = dict(in_channels=1,out_channels=1,kernel_size=[3],stride=[2],padding=[1],
             output_padding=[1],dilation=[1],groups=1,bias=False)
    # Last kernel contribution is retained when the output extent grows.
    shape,y,dx,dw,db = scalar_reference((1,1,2),[2,3],[5,7,11],None,[1,2,3,4],c)
    assert shape == (1,1,4) and y == [14,37,21,33]
    assert dx == [29,75] and dw == [6,11,16] and db == [10]
