"""All supported sample axes, independent stabilized scalar values and VJPs.

Existing test_softmax owns lifecycle/discovery. No runtime behavior is added here.
"""
import itertools
import math
import pytest
import torch
from runner.engine import GraphModel, export_source, validate
from runner.tests.test_softmax import registry


def document(rank, axis):
    sample = [2]*(rank-1); width = math.prod(sample)
    specs = [('input','input',{'features':2}), ('project','linear',{'in_features':2,'out_features':width,'bias':True}), ('shape','reshape',{'shape':sample})]
    if rank >= 3: specs.append(('permuted','transpose',{'dim0':1,'dim1':2}))
    specs += [('probabilities','softmax',{'dim':axis}), ('flat','flatten',{'start_dim':1,'end_dim':-1}), ('head','linear',{'in_features':width,'out_features':2,'bias':False}), ('output','output',{})]
    return {'schemaVersion':2,'name':'Sample-axis normalization','nodes':[{'id':i,'type':t,'name':i,'x':0,'y':0,'config':c} for i,t,c in specs],
            'edges':[{'id':str(i),'source':a[0],'sourcePort':'out','target':b[0],'targetPort':'in'} for i,(a,b) in enumerate(zip(specs,specs[1:]))]}


def scalar_slices(values, shape, axis, upstream):
    """Only stdlib indexing/exp/sums; expected derivatives use reference p.

    Rows are never merged across the batch dimension. All slice sums and the
    coupled weighted derivative are computed without tensor operator or autograd.
    """
    axis %= len(shape)
    strides = [math.prod(shape[i+1:]) for i in range(len(shape))]
    other = [i for i in range(len(shape)) if i != axis]
    probabilities = [0.]*len(values); gradients = [0.]*len(values)
    for coordinate in itertools.product(*(range(shape[i]) for i in other)):
        base = sum(index*strides[dimension] for dimension,index in zip(other,coordinate))
        indices = [base+i*strides[axis] for i in range(shape[axis])]
        peak = max(values[i] for i in indices)
        exponentials = [math.exp(values[i]-peak) for i in indices]
        denominator = sum(exponentials)
        shares = [v/denominator for v in exponentials]
        weighted = sum(p*upstream[i] for p,i in zip(shares,indices))
        for p,i in zip(shares,indices):
            probabilities[i] = p; gradients[i] = p*(upstream[i]-weighted)
    return probabilities,gradients


CASES = [(rank, axis) for rank in range(2,9) for positive in range(1,rank) for axis in (positive,positive-rank)]


@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('batch',[1,3])
@pytest.mark.parametrize('rank,axis',CASES)
def test_every_sample_axis_alias_independent_forward_coupled_gradients_and_export(device,batch,rank,axis,registry):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('Actual MPS unavailable')
    width = 2**(rank-1); model = GraphModel(document(rank,axis),registry).to(device)
    weights = [[1. if j%2==0 else -1.,(j%3-1)*0.25] for j in range(width)]
    biases = [(j-width/2)*0.125 for j in range(width)]
    head = [[math.sin(j*0.17) for j in range(width)],[(j%5-2)*0.25 for j in range(width)]]
    with torch.no_grad():
        model.layers['project'].weight.copy_(torch.tensor(weights,device=device));model.layers['project'].bias.copy_(torch.tensor(biases,device=device));model.layers['head'].weight.copy_(torch.tensor(head,device=device))
    storage = torch.empty((3,4),device=device);storage[:,::2]=torch.tensor([[0.5,-0.7],[1000.,-1000.],[-2.,3.]],device=device);storage[:,1::2]=999
    x = storage[:batch,::2].detach().requires_grad_();assert not x.is_contiguous()
    output,stages = model(x,inspect=True)
    source = stages['permuted' if rank>=3 else 'shape']; actual = stages['probabilities']
    if rank>=3:assert not source.is_contiguous()
    shape = tuple(source.shape); observed = source.detach().cpu().flatten().tolist()
    upstream_values = [math.sin(i*0.37)+(i%3)*0.2 for i in range(source.numel())]
    expected,gradient = scalar_slices(observed,shape,axis,upstream_values)
    torch.testing.assert_close(actual,torch.tensor(expected,device=device).reshape(shape),atol=2e-6,rtol=2e-5)
    upstream = torch.tensor(upstream_values,device=device).reshape(shape)
    measured = torch.autograd.grad(actual,source,upstream,retain_graph=True)[0]
    torch.testing.assert_close(measured,torch.tensor(gradient,device=device).reshape(shape),atol=3e-6,rtol=3e-5)
    assert actual.shape==source.shape and actual.dtype==torch.float32
    assert bool(torch.isfinite(actual).all()) and bool(((actual>=0)&(actual<=1)).all())
    assert not any('probabilities' in name for name in model.state_dict())
    # Independently route the head's weighted derivative back through the
    # transpose and affine projection, then compare exported input gradients.
    coefficients = [0.6,-0.2]
    head_upstream = [sum(coefficients[q]*head[q][i%width] for q in range(2)) for i in range(len(observed))]
    _,hidden_gradient = scalar_slices(observed,shape,axis,head_upstream)
    gx = [[0.,0.] for _ in range(batch)]
    for coordinates in itertools.product(*(range(d) for d in shape)):
        flat_index = sum(coordinates[d]*math.prod(shape[d+1:]) for d in range(rank))
        original = list(coordinates)
        if rank>=3:original[1],original[2]=original[2],original[1]
        channel = sum(original[d]*math.prod(shape[d+1:]) for d in range(1,rank))
        for feature in range(2):gx[coordinates[0]][feature] += hidden_gradient[flat_index]*weights[channel][feature]
    measured_input = torch.autograd.grad((output*torch.tensor(coefficients,device=device)).sum(),x)[0]
    expected_input = torch.tensor(gx,device=device)
    torch.testing.assert_close(measured_input,expected_input,atol=3e-5,rtol=3e-5)
    namespace = {};code = export_source(model);exec(code,namespace)
    exported = namespace['Model']().to(device);exported.load_state_dict(model.state_dict(),strict=True)
    assert f'dim={axis}' in code
    torch.testing.assert_close(exported(x),output,atol=0,rtol=0)
    torch.testing.assert_close(torch.autograd.grad((exported(x)*torch.tensor(coefficients,device=device)).sum(),x)[0],expected_input,atol=3e-5,rtol=3e-5)
    for row in range(batch):torch.testing.assert_close(model(x[row:row+1]),output[row:row+1],atol=2e-5,rtol=2e-5)


@pytest.mark.parametrize('rank',range(2,9))
@pytest.mark.parametrize('axis_kind',['batch','negative-batch','beyond-positive','beyond-negative'])
def test_rank_specific_batch_alias_and_outside_axes_refused_before_allocation(rank,axis_kind,registry):
    axis = {'batch':0,'negative-batch':-rank,'beyond-positive':rank,'beyond-negative':-rank-1}[axis_kind]
    with pytest.raises(ValueError):validate(document(rank,axis),registry)


@pytest.mark.parametrize('axis',[True,False,None,'1',1.5,float('nan'),float('inf'),{},[],10**400])
def test_malformed_axis_types_refused(axis,registry):
    with pytest.raises(ValueError):validate(document(3,axis),registry)


def test_integer_source_and_external_double_dtype_refused(registry):
    g = document(3,-1)
    g['nodes'] = [node for node in g['nodes'] if node['id'] not in ('project','shape','permuted')]
    g['nodes'][0].update(type='index_input',config={'shape':[2,2]})
    g['edges'] = [{'id':str(i),'source':a['id'],'sourcePort':'out','target':b['id'],'targetPort':'in'} for i,(a,b) in enumerate(zip(g['nodes'],g['nodes'][1:]))]
    with pytest.raises(ValueError,match='floating'):validate(g,registry)
    model = GraphModel(document(3,-1),registry)
    with pytest.raises(RuntimeError,match='dtype|Float|Double'):model(torch.tensor([[1.,2.]],dtype=torch.float64))
