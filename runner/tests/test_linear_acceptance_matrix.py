"""Bounded product graph acceptance against independently evaluated matrix algebra."""
import json
import pytest
import torch
from runner.engine import GraphModel
from runner.product_execution import ProductExecutionAdapter

DEVICES=['cpu','mps']
WIDTHS=[(1,1),(2,3),(3,2),(512,512)]
TOLERANCES={'cpu':(1e-5,1e-6),'mps':(1e-4,1e-5)}

def graph(widths,bias):
    specs=[('input','input',{'features':2})]
    last=2
    for index,width in enumerate([*widths,2]):
        specs.append((f'linear{index}','linear',{'in_features':last,'out_features':width,'bias':bias}));last=width
    specs.append(('output','output',{}))
    return {'schemaVersion':2,'nodes':[{'id':i,'type':t,'name':i,'x':0,'y':0,'config':c} for i,t,c in specs],
            'edges':[{'id':str(i),'source':a[0],'sourcePort':'out','target':b[0],'targetPort':'in'} for i,(a,b) in enumerate(zip(specs,specs[1:]))]}

def ready(device):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('Actual MPS unavailable; not device acceptance')

def oracle(model,x,parameters):
    value=x
    for ident in model.order:
        node=model.nodes[ident]
        if node['type']=='linear':
            key=model.layer_keys[ident];value=value @ parameters[f'layers.{key}.weight'].T
            if node['config']['bias']:value=value+parameters[f'layers.{key}.bias']
    return value

def measure(device,widths,bias,batch):
    ready(device);torch.manual_seed(8042)
    model=GraphModel(graph(widths,bias)).to(device)
    # Low-magnitude deterministic parameters keep the matrix conditioning bounded.
    with torch.no_grad():
        for parameter in model.parameters():parameter.mul_(.3)
    adapter=ProductExecutionAdapter(model)
    x=torch.randn(batch,4,device=device)[:,::2].requires_grad_()
    independent={name:value.detach().clone().requires_grad_() for name,value in model.named_parameters()}
    reference_x=x.detach().clone().requires_grad_();actual=adapter.forward(x);expected=oracle(model,reference_x,independent)
    upstream=torch.linspace(-.7,.9,batch*2,device=device).reshape(batch,2)
    actual_grad=torch.autograd.grad(actual,(x,*model.parameters()),upstream)
    expected_grad=torch.autograd.grad(expected,(reference_x,*independent.values()),upstream)
    rtol,atol=TOLERANCES[device]
    torch.testing.assert_close(actual,expected,rtol=rtol,atol=atol)
    for a,b in zip(actual_grad,expected_grad):torch.testing.assert_close(a,b,rtol=rtol,atol=atol)
    relative=max(float(((a-b).abs()/b.abs().clamp_min(torch.finfo(torch.float32).eps)).max().detach().cpu()) for a,b in [(actual,expected),*zip(actual_grad,expected_grad)])
    return {'device':device,'widths':list(widths),'bias':bias,'batch':batch,'noncontiguous':not x.is_contiguous(),
            'maximumRelativeError':relative,
            'maximumOutputAbsoluteError':float((actual-expected).abs().max().detach().cpu()),
            'maximumGradientAbsoluteError':max(float((a-b).abs().max().detach().cpu()) for a,b in zip(actual_grad,expected_grad))}

@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('widths',WIDTHS)
@pytest.mark.parametrize('bias',[False,True])
@pytest.mark.parametrize('batch',[1,7])
def test_forward_backward_matrix(device,widths,bias,batch):
    result=measure(device,widths,bias,batch)
    assert result['noncontiguous']

@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('bias',[False,True])
def test_seed_and_explicit_reset(device,bias):
    ready(device)
    torch.manual_seed(100);a=GraphModel(graph((3,2),bias)).to(device)
    torch.manual_seed(100);b=GraphModel(graph((3,2),bias)).to(device)
    assert all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters()))
    before={k:v.clone() for k,v in a.state_dict().items()}
    torch.manual_seed(200)
    for layer in a.layers.values():layer.reset_parameters()
    torch.manual_seed(200)
    for layer in b.layers.values():layer.reset_parameters()
    assert all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters()))
    assert any(not torch.equal(value,before[key]) for key,value in a.state_dict().items())
    assert all((layer.bias is not None)==bias for layer in a.layers.values())

@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('bias',[False,True])
def test_branched_gradient_accumulation_and_parameter_ownership(device,bias):
    ready(device)
    g=graph((2,),bias)
    g['nodes']=[g['nodes'][0],g['nodes'][1],{**g['nodes'][1],'id':'other'}, {'id':'sum','type':'add','name':'sum','x':0,'y':0,'config':{}},g['nodes'][-1]]
    wires=[('input','linear0','in'),('input','other','in'),('linear0','sum','a'),('other','sum','b'),('sum','output','in')]
    g['edges']=[{'id':str(i),'source':a,'sourcePort':'out','target':b,'targetPort':p} for i,(a,b,p) in enumerate(wires)]
    torch.manual_seed(19);model=GraphModel(g).to(device);adapter=ProductExecutionAdapter(model)
    x=torch.randn(7,2,device=device,requires_grad=True);upstream=torch.randn(7,2,device=device)
    out=adapter.forward(x);dx,*grads=torch.autograd.grad(out,(x,*model.parameters()),upstream)
    left,right=model.layers['linear0'],model.layers['other'];rtol,atol=TOLERANCES[device]
    torch.testing.assert_close(dx,upstream @ left.weight + upstream @ right.weight,rtol=rtol,atol=atol)
    assert left.weight.data_ptr()!=right.weight.data_ptr()
    for (name,_),gradient in zip(model.named_parameters(),grads):
        expected=upstream.sum(0) if name.endswith('.bias') else upstream.T @ x
        torch.testing.assert_close(gradient,expected,rtol=rtol,atol=atol)

if __name__=='__main__':
    results=[measure(device,widths,bias,batch) for device in DEVICES if device=='cpu' or torch.backends.mps.is_available() for widths in WIDTHS for bias in (False,True) for batch in (1,7)]
    print(json.dumps({'tolerances':TOLERANCES,'cases':results},indent=2))
