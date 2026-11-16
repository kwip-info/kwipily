"""Uncoupled scalar LayerNorm checks across every supported trailing suffix.

The existing lifecycle/discovery suites remain authoritative for those surfaces.
Expected values and derivatives here use stdlib sums and population variance;
no autograd or worker normalization produces the reference.
"""
import math
import pytest
import torch
from runner.engine import GraphModel, REGISTRY


CASES = [(rank, suffix) for rank in range(2,9) for suffix in range(1,rank)]


def graph(rank,suffix,affine,bias,epsilon):
    sample=[2]*(rank-1);width=math.prod(sample)
    specs=[('input','input',{'features':2}),('expand','linear',{'in_features':2,'out_features':width,'bias':True}),
           ('shape','reshape',{'shape':sample}),('norm','layer_norm',{'normalized_shape':[2]*suffix,'eps':epsilon,'elementwise_affine':affine,'bias':bias}),
           ('flat','flatten',{'start_dim':1,'end_dim':-1}),('head','linear',{'in_features':width,'out_features':2,'bias':False}),('output','output',{})]
    return {'schemaVersion':2,'name':'Trailing groups','nodes':[{'id':i,'type':t,'name':i,'x':0,'y':0,'config':c} for i,t,c in specs],
            'edges':[{'id':str(i),'source':a[0],'sourcePort':'out','target':b[0],'targetPort':'in'} for i,(a,b) in enumerate(zip(specs,specs[1:]))]}


def scalar_reference(values,upstream,count,epsilon,gamma,beta):
    output=[];gradient=[];weight_gradient=[0.]*count;bias_gradient=[0.]*count
    for start in range(0,len(values),count):
        row=values[start:start+count];g=upstream[start:start+count]
        mean=math.fsum(row)/count
        centered=[v-mean for v in row]
        variance=math.fsum(v*v for v in centered)/count
        inverse=1/math.sqrt(variance+epsilon)
        z=[v*inverse for v in centered]
        q=[g[j]*gamma[j] for j in range(count)]
        mean_q=math.fsum(q)/count;mean_qz=math.fsum(q[j]*z[j] for j in range(count))/count
        output.extend(z[j]*gamma[j]+beta[j] for j in range(count))
        gradient.extend(inverse*(q[j]-mean_q-z[j]*mean_qz) for j in range(count))
        for j in range(count):
            weight_gradient[j]+=g[j]*z[j];bias_gradient[j]+=g[j]
    return output,gradient,weight_gradient,bias_gradient


def prepared_case(device,rank,suffix,affine,bias,conditioning):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('Actual MPS unavailable')
    batch={'ordinary':3,'near-constant':1,'large-offset':2}[conditioning]
    epsilon=1e-3 if conditioning=='near-constant' else 1e-4
    sample=[2]*(rank-1);shape=(batch,*sample);count=2**suffix
    model=GraphModel(graph(rank,suffix,affine,bias,epsilon),REGISTRY).to(device)
    layer=model.layers[model.layer_keys['norm']]
    gamma=[0.5+j/count for j in range(count)] if affine else [1.]*count
    beta=[(j%3-1)*0.25 for j in range(count)] if affine and bias else [0.]*count
    with torch.no_grad():
        if affine:layer.weight.copy_(torch.tensor(gamma,device=device).reshape([2]*suffix))
        if affine and bias:layer.bias.copy_(torch.tensor(beta,device=device).reshape([2]*suffix))
    size=math.prod(shape)
    if conditioning=='ordinary':raw=[math.sin(i*0.23)+(i%5-2)*0.3 for i in range(size)]
    elif conditioning=='near-constant':raw=[1.+(i%5-2)/8192 for i in range(size)]
    else:raw=[1024.+(i%7-3)*0.25 for i in range(size)]
    # Sampling every other last-axis element gives a genuine noncontiguous
    # normalization input at every tested rank, including batch size one.
    storage=torch.empty((*shape[:-1],shape[-1]*2),device=device)
    storage[...,::2]=torch.tensor(raw,device=device).reshape(shape);storage[...,1::2]=999
    x=storage[...,::2].detach().requires_grad_();assert not x.is_contiguous()
    upstream=torch.tensor([math.sin(i*0.17)+(i%7-3)*0.3 for i in range(size)],device=device).reshape(shape)
    # Read actual float32-representable inputs; decimal source values can round.
    observed=x.detach().cpu().flatten().tolist();g=upstream.cpu().flatten().tolist()
    expected,gx,gw,gb=scalar_reference(observed,g,count,epsilon,gamma,beta)
    return layer,x,upstream,(expected,gx,gw,gb)


@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('rank,suffix',CASES)
@pytest.mark.parametrize('affine,bias',[(False,False),(False,True),(True,False),(True,True)])
@pytest.mark.parametrize('conditioning',['ordinary','near-constant'])
def test_all_ranks_suffixes_conditioning_and_affine_gradients(device,rank,suffix,affine,bias,conditioning):
    layer,x,upstream,(expected,gx,gw,gb)=prepared_case(device,rank,suffix,affine,bias,conditioning)
    shape=tuple(x.shape);batch=shape[0]
    actual=layer(x)
    targets=[x]+list(layer.parameters())
    measured=torch.autograd.grad(actual,targets,upstream)
    assert actual.shape==x.shape and actual.dtype==torch.float32
    assert torch.isfinite(actual).all() and all(torch.isfinite(value).all() for value in measured)
    # These are bounded fixture tolerances, not a promise about arbitrary
    # ill-conditioned float32 data. Offset and increments are representable.
    torch.testing.assert_close(actual,torch.tensor(expected,device=device).reshape(shape),atol=3e-5,rtol=3e-5)
    torch.testing.assert_close(measured[0],torch.tensor(gx,device=device).reshape(shape),atol=3e-4,rtol=3e-5)
    assert len(list(layer.parameters()))==int(affine)*(1+int(bias))
    if affine:
        torch.testing.assert_close(measured[1],torch.tensor(gw,device=device).reshape([2]*suffix),atol=3e-4,rtol=3e-5)
        if bias:torch.testing.assert_close(measured[2],torch.tensor(gb,device=device).reshape([2]*suffix),atol=3e-5,rtol=3e-5)
    layer.eval();torch.testing.assert_close(layer(x),actual,atol=0,rtol=0)
    for row in range(batch):torch.testing.assert_close(layer(x[row:row+1]),actual[row:row+1],atol=3e-5,rtol=3e-5)


@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('rank,suffix',CASES)
@pytest.mark.parametrize('affine,bias',[(False,False),(False,True),(True,False),(True,True)])
def test_large_offset_characterization_direct_backend_parity_centered_and_double_controls(device,rank,suffix,affine,bias):
    """Pinned-backend characterization, not a general float32 accuracy promise.

    Original strict run:92 CPU failures/580 passes, retained at
    .local/layer-norm-conditioning-red.log. This fixture uses offset1024 and
    exact quarter-step variations, eps1e-4, two rows and suffix sizes2..128.
    Measured CPU maxima: forward2.30e-5, input-grad0.01098, weight-grad0.01196.
    Conservative CPU fixture bounds below allow0.02 for those derivatives;
    they do not replace the ordinary/nearconstant matrix's strict tolerances.
    MPS measured gradient maxima were below2.7e-6. Centering and double controls
    independently establish the source of the cancellation limitation.
    """
    layer,x,upstream,reference=prepared_case(device,rank,suffix,affine,bias,'large-offset')
    expected,gx,gw,gb=reference;shape=tuple(x.shape)
    actual=layer(x);measured=torch.autograd.grad(actual,[x]+list(layer.parameters()),upstream)
    direct=torch.nn.LayerNorm([2]*suffix,eps=1e-4,elementwise_affine=affine,bias=bias).to(device)
    direct.load_state_dict(layer.state_dict(),strict=True)
    direct_output=direct(x);direct_grad=torch.autograd.grad(direct_output,[x]+list(direct.parameters()),upstream)
    torch.testing.assert_close(actual,direct_output,atol=0,rtol=0)
    for a,b in zip(measured,direct_grad):torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert actual.shape==x.shape and actual.dtype==torch.float32
    assert torch.isfinite(actual).all() and all(torch.isfinite(value).all() for value in measured)
    def maximum_error(value,flat):
        return max(abs(a-b) for a,b in zip(value.detach().cpu().flatten().tolist(),flat))
    bound=0.02 if device=='cpu' else 3e-4
    assert maximum_error(actual,expected) <= (1e-4 if device=='cpu' else 3e-5)
    assert maximum_error(measured[0],gx) <= bound
    if affine:assert maximum_error(measured[1],gw) <= bound
    if affine and bias:assert maximum_error(measured[2],gb) <= 3e-5
    # Same represented data, explicit constant shift: useful numerical control,
    # not an automatic app repair or a fitted-data preprocessing claim.
    centered=(x.detach()-1024).requires_grad_()
    centered_output=direct(centered)
    centered_grad=torch.autograd.grad(centered_output,[centered]+list(direct.parameters()),upstream)
    torch.testing.assert_close(centered_output,torch.tensor(expected,device=device).reshape(shape),atol=3e-5,rtol=3e-5)
    torch.testing.assert_close(centered_grad[0],torch.tensor(gx,device=device).reshape(shape),atol=3e-4,rtol=3e-5)
    if affine:torch.testing.assert_close(centered_grad[1],torch.tensor(gw,device=device).reshape([2]*suffix),atol=3e-4,rtol=3e-5)
    if affine and bias:torch.testing.assert_close(centered_grad[2],torch.tensor(gb,device=device).reshape([2]*suffix),atol=3e-5,rtol=3e-5)
    # CPU float64 is solely a reference control; shipped graphs remain float32
    # and this test does not imply MPS float64 support.
    wide=torch.nn.LayerNorm([2]*suffix,eps=1e-4,elementwise_affine=affine,bias=bias,dtype=torch.float64)
    wide.load_state_dict({k:v.detach().cpu().double() for k,v in layer.state_dict().items()},strict=True)
    wide_input=x.detach().cpu().double().requires_grad_();wide_output=wide(wide_input)
    wide_grad=torch.autograd.grad(wide_output,[wide_input]+list(wide.parameters()),upstream.detach().cpu().double())
    torch.testing.assert_close(wide_output,torch.tensor(expected,dtype=torch.float64).reshape(shape),atol=1e-8,rtol=1e-7)
    torch.testing.assert_close(wide_grad[0],torch.tensor(gx,dtype=torch.float64).reshape(shape),atol=1e-8,rtol=1e-7)
    if affine:torch.testing.assert_close(wide_grad[1],torch.tensor(gw,dtype=torch.float64).reshape([2]*suffix),atol=1e-8,rtol=1e-7)
    if affine and bias:torch.testing.assert_close(wide_grad[2],torch.tensor(gb,dtype=torch.float64).reshape([2]*suffix),atol=1e-8,rtol=1e-7)
