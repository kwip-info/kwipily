import copy
import json
import pytest
import torch
from runner.engine import GraphModel, REGISTRY, export_source
from runner.tests.test_softmax import graph as softmax_graph

@pytest.fixture
def registry(tmp_path):
    data=json.loads(REGISTRY.read_text())
    data['components'].append({'id':'layer_norm','title':'Layer normalization','inputs':['in'],'outputs':['out'],'fields':[{'key':'eps','title':'Epsilon','kind':'number','defaultValue':1e-5,'min':1.1754943508222875e-38,'max':3.4028234663852886e38}]})
    path=tmp_path/'registry.json';path.write_text(json.dumps(data));return path

def graph(shape,affine,bias):
    value=softmax_graph();value['nodes'][3].update(type='layer_norm',config={'normalized_shape':shape,'eps':1e-4,'elementwise_affine':affine,'bias':bias})
    return value

@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('shape',[[3],[2,3]])
@pytest.mark.parametrize('affine,bias',[(False,False),(False,True),(True,False),(True,True)])
def test_formula_gradients_state_and_export(registry,device,shape,affine,bias):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
    torch.manual_seed(53)
    model=GraphModel(graph(shape,affine,bias),registry).to(device)
    layer=model.layers[model.layer_keys['probabilities']]
    if affine:
        with torch.no_grad():
            layer.weight.copy_(torch.linspace(.5,1.5,layer.weight.numel(),device=device).reshape(shape))
            if bias:layer.bias.fill_(.25)
    x=torch.tensor([[.2,-.4],[3.,1.],[-2.,.1]],device=device,requires_grad=True)
    result,stages=model(x,inspect=True)
    hidden=stages['shape'];axes=tuple(range(hidden.ndim-len(shape),hidden.ndim))
    center=hidden-hidden.mean(axes,keepdim=True)
    normalized=center*torch.rsqrt((center*center).mean(axes,keepdim=True)+1e-4)
    expected=normalized*(layer.weight if affine else 1)+(layer.bias if affine and bias else 0)
    torch.testing.assert_close(stages['probabilities'],expected,atol=2e-5,rtol=2e-5)
    upstream=torch.linspace(-2,3,hidden.numel(),device=device).reshape(hidden.shape)
    targets=[hidden]+list(layer.parameters())
    actual_grad=torch.autograd.grad(stages['probabilities'],targets,upstream,retain_graph=True)
    formula_grad=torch.autograd.grad(expected,targets,upstream,retain_graph=True)
    for actual,oracle in zip(actual_grad,formula_grad):torch.testing.assert_close(actual,oracle,atol=3e-5,rtol=3e-5)
    assert len(list(layer.parameters()))==int(affine)*(1+int(bias))
    torch.testing.assert_close(model(x[:1]),result[:1],atol=2e-5,rtol=2e-5)
    model.eval();torch.testing.assert_close(model(x),result,atol=0,rtol=0)
    scope={};exec(export_source(model),scope)
    exported=scope['Model']().to(device);exported.load_state_dict(model.state_dict(),strict=True)
    torch.testing.assert_close(exported(x),result,atol=0,rtol=0)

@pytest.mark.parametrize('shape',[[2],[6],[1,2,3]])
def test_reject_wrong_trailing_shape(registry,shape):
    with pytest.raises(ValueError,match='trailing|batch'):GraphModel(graph(shape,True,True),registry)

@pytest.mark.parametrize('device',['cpu','mps'])
def test_constant_inputs_and_reserved_node_key(registry,device):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
    value=graph([2,3],True,True)
    value['nodes'][3]['id']='training'
    for edge in value['edges']:
        for endpoint in ('source','target'):
            if edge[endpoint]=='probabilities':edge[endpoint]='training'
    model=GraphModel(value,registry).to(device)
    layer=model.layers[model.layer_keys['training']]
    assert model.layer_keys['training']!='training'
    x=torch.full((2,2,3),7.,device=device,requires_grad=True)
    actual=layer(x)
    torch.testing.assert_close(actual,torch.zeros_like(actual),atol=2e-5,rtol=0)
    upstream=torch.arange(12.,device=device).reshape(x.shape)
    gradient=torch.autograd.grad(actual,x,upstream)[0]
    expected=(upstream-upstream.mean((1,2),keepdim=True))/1e-2
    torch.testing.assert_close(gradient,expected,atol=2e-4,rtol=2e-5)
    scope={};exec(export_source(model),scope)
    restored=scope['Model']().to(device);restored.load_state_dict(model.state_dict(),strict=True)
    sample=torch.tensor([[1.,2.]],device=device)
    torch.testing.assert_close(restored(sample),model(sample),atol=0,rtol=0)

@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('epsilon',[1.1754943508222875e-38,3.4028234663852886e38])
def test_epsilon_bounds_remain_finite_for_constant_inputs(registry,device,epsilon):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
    value=graph([3],False,False);value['nodes'][3]['config']['eps']=epsilon
    model=GraphModel(value,registry).to(device)
    x=torch.zeros((2,2,3),device=device,requires_grad=True)
    layer=model.layers[model.layer_keys['probabilities']]
    y=layer(x)
    gradient=torch.autograd.grad(y,x,torch.arange(12.,device=device).reshape(x.shape))[0]
    assert torch.isfinite(y).all() and torch.isfinite(gradient).all()


def test_input_and_affine_gradients_match_finite_differences(registry):
    model=GraphModel(graph([2,3],True,True),registry).double()
    layer=model.layers[model.layer_keys['probabilities']]
    x=torch.tensor([[[.2,-.7,1.1],[2.,-.3,.4]]],dtype=torch.float64,requires_grad=True)
    upstream=torch.tensor([[[2.,-1.,.5],[-.4,3.,1.]]],dtype=torch.float64)
    targets=[x,layer.weight,layer.bias]
    analytical=torch.autograd.grad((layer(x)*upstream).sum(),targets)
    delta=1e-6
    for target,expected in zip(targets,analytical):
        for index in range(target.numel()):
            with torch.no_grad():
                flat=target.view(-1);original=flat[index].item()
                flat[index]=original+delta;plus=(layer(x)*upstream).sum().item()
                flat[index]=original-delta;minus=(layer(x)*upstream).sum().item()
                flat[index]=original
            assert (plus-minus)/(2*delta)==pytest.approx(expected.view(-1)[index].item(),abs=2e-8,rel=2e-7)

@pytest.mark.parametrize('width,expected_parameters', [(309, 1_999_883), (310, 2_000_400)])
def test_normalization_budget_before_parameter_allocation(registry, width, expected_parameters):
    from runner.engine import validate
    from unittest.mock import patch
    specs = [('input', 'input', {'features': 2}), ('expand', 'linear', {'in_features': 2, 'out_features': 512, 'bias': True})]
    specs += [(f'wide{i}', 'linear', {'in_features': 512, 'out_features': 512, 'bias': True}) for i in range(7)]
    specs += [('reduce', 'linear', {'in_features': 512, 'out_features': width, 'bias': True}),
              ('norm', 'layer_norm', {'normalized_shape': [width], 'eps': 1e-5, 'elementwise_affine': True, 'bias': True}),
              ('head', 'linear', {'in_features': width, 'out_features': 2, 'bias': True}), ('output', 'output', {})]
    flow = {'schemaVersion': 2, 'nodes': [{'id': i, 'type': t, 'config': c} for i, t, c in specs],
            'edges': [{'id': str(i), 'source': specs[i][0], 'sourcePort': 'out', 'target': specs[i+1][0], 'targetPort': 'in'} for i in range(len(specs)-1)]}
    if expected_parameters <= 2_000_000:
        model = GraphModel(flow, registry)
        assert sum(p.numel() for p in model.parameters()) == expected_parameters
    else:
        with patch('runner.engine.nn.Linear', side_effect=AssertionError('Allocated before budget validation')):
            with pytest.raises(ValueError, match='2 million parameters'):
                GraphModel(flow, registry)
    flow['nodes'][-3]['config']['elementwise_affine'] = False
    validate(flow, registry)
    model = GraphModel(flow, registry)
    assert sum(p.numel() for p in model.parameters()) == expected_parameters - 2 * width
