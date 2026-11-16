"""Independent scalar/analytic GELU references for missing higher-rank modes.

Existing test_gelu covers rank-two training/export/lifecycle. This file closes
higher-rank and uncoupled derivative evidence without broadening the adapter.
"""
import math
import pytest
import torch
from runner.engine import GraphModel, export_source, validate
from runner.tests.test_gelu import graph, registry



def scalar_reference(value, mode):
    """Independent double scalar value and analytic first derivative.

    No torch operators or differentiation are used to produce expected values.
    Approximate mode differentiates its polynomial/tanh expression explicitly.
    """
    if mode:
        scale = math.sqrt(2/math.pi)
        u = scale*(value+0.044715*value**3)
        gate = math.tanh(u)
        slope = scale*(1+3*0.044715*value**2)
        return 0.5*value*(1+gate), 0.5*(1+gate)+0.5*value*(1-gate**2)*slope
    share = 0.5*(1+math.erf(value/math.sqrt(2)))
    density = math.exp(-0.5*value**2)/math.sqrt(2*math.pi)
    return value*share, share+value*density


def higher_graph(mode, sample):
    document = graph(mode)
    document['nodes'].insert(2, {'id': 'shape', 'type': 'reshape', 'name': 'Higher-rank features', 'x': 0, 'y': 0, 'config': {'shape': sample}})
    document['nodes'].insert(4, {'id': 'flat', 'type': 'flatten', 'name': 'Classifier features', 'x': 0, 'y': 0, 'config': {'start_dim': 1, 'end_dim': -1}})
    document['edges'] = [{'id': str(i), 'source': left['id'], 'sourcePort': 'out', 'target': right['id'], 'targetPort': 'in'} for i, (left, right) in enumerate(zip(document['nodes'], document['nodes'][1:]))]
    return document


def identity(model):
    with torch.no_grad():
        for name in ('first', 'last'):
            layer = model.layers[name]
            layer.weight.copy_(torch.eye(2, device=layer.weight.device))
            if layer.bias is not None: layer.bias.zero_()


@pytest.mark.parametrize('device', ['cpu', 'mps'])
@pytest.mark.parametrize('mode', [False, True])
@pytest.mark.parametrize('sample', [[1, 2], [1, 1, 1, 1, 1, 1, 2]])
@pytest.mark.parametrize('batch', [1, 6])
def test_higher_rank_values_analytic_derivative_export_and_batch_independence(device, mode, sample, batch, registry):
    if device == 'mps' and not torch.backends.mps.is_available(): pytest.skip('Actual MPS unavailable')
    raw = [-1000., 0., -20., 20., -3., 3., -1e-6, 1e-6, -0.5, 0.5, -1., 1.]
    storage = torch.empty((6, 4), device=device)
    storage[:, ::2] = torch.tensor(raw, device=device).reshape(6, 2); storage[:, 1::2] = 999
    x = storage[:batch, ::2].detach().requires_grad_(); assert not x.is_contiguous()
    values = raw[:batch*2]
    expected = torch.tensor([scalar_reference(v, mode)[0] for v in values], device=device).reshape(batch, 2)
    derivative = torch.tensor([scalar_reference(v, mode)[1] for v in values], device=device).reshape(batch, 2)
    model = GraphModel(higher_graph(mode, sample), registry).to(device); identity(model)
    output, stages = model(x, inspect=True)
    torch.testing.assert_close(output, expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(torch.autograd.grad(output.sum(), x)[0], derivative, atol=3e-6, rtol=3e-5)
    assert stages['activation'].shape == (batch, *sample)
    assert stages['activation'].dtype == torch.float32
    assert output[0, 1].item() == 0
    # Exactly zero has slope 0.5 in both modes; negative derivatives around -1
    # distinguish GELU from clipping, a sign mask, or a probability activation.
    assert scalar_reference(0., mode)[1] == 0.5
    assert scalar_reference(-1., mode)[1] < 0
    namespace = {}; source = export_source(model); exec(source, namespace)
    exported = namespace['Model']().to(device); exported.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(exported(x), output, atol=0, rtol=0)
    torch.testing.assert_close(torch.autograd.grad(exported(x).sum(), x)[0], derivative, atol=3e-6, rtol=3e-5)
    model.eval(); torch.testing.assert_close(model(x), output, atol=0, rtol=0)
    for row in range(batch):
        torch.testing.assert_close(model(x[row:row+1]), expected[row:row+1], atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize('device', ['cpu', 'mps'])
@pytest.mark.parametrize('mode', [False, True])
def test_independent_affine_chain_input_and_parameter_gradients(device, mode, registry):
    if device == 'mps' and not torch.backends.mps.is_available(): pytest.skip('Actual MPS unavailable')
    w = [[0.7, -0.3], [0.2, 0.9]]; b = [0.1, -0.2]; v = [[1.1, -0.4], [0.3, 0.8]]
    rows = [[-1., 0.5], [0.7, 2.]]; coefficients = [0.6, -0.2]
    model = GraphModel(graph(mode), registry).to(device)
    with torch.no_grad():
        model.layers['first'].weight.copy_(torch.tensor(w, device=device)); model.layers['first'].bias.copy_(torch.tensor(b, device=device)); model.layers['last'].weight.copy_(torch.tensor(v, device=device))
    x = torch.tensor(rows, device=device, requires_grad=True)
    output = model(x); (output*torch.tensor(coefficients, device=device)).sum().backward()
    gx = []; gw = [[0.,0.],[0.,0.]]; gb = [0.,0.]; gv = [[0.,0.],[0.,0.]]; expected = []
    for row in rows:
        pairs = [scalar_reference(sum(w[j][k]*row[k] for k in range(2))+b[j], mode) for j in range(2)]
        hidden = [p[0] for p in pairs]
        expected.append([sum(v[q][j]*hidden[j] for j in range(2)) for q in range(2)])
        dz = [sum(coefficients[q]*v[q][j] for q in range(2))*pairs[j][1] for j in range(2)]
        gx.append([sum(dz[j]*w[j][k] for j in range(2)) for k in range(2)])
        for j in range(2):
            gb[j] += dz[j]
            for k in range(2): gw[j][k] += dz[j]*row[k]
        for q in range(2):
            for j in range(2): gv[q][j] += coefficients[q]*hidden[j]
    for actual, reference in [(output,expected),(x.grad,gx),(model.layers['first'].weight.grad,gw),(model.layers['first'].bias.grad,gb),(model.layers['last'].weight.grad,gv)]:
        torch.testing.assert_close(actual, torch.tensor(reference, device=device), atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize('mode',[False,True])
def test_integer_graph_and_external_double_dtype_refusal(mode, registry):
    document = graph(mode)
    document['nodes'][0].update(type='index_input', config={'shape':[2]})
    document['nodes'] = [node for node in document['nodes'] if node['id']!='first']
    document['edges'] = [edge for edge in document['edges'] if edge['source']!='first']
    document['edges'][0]['target']='activation'
    with pytest.raises(ValueError,match='floating'): validate(document,registry)
    model=GraphModel(graph(mode),registry)
    with pytest.raises(RuntimeError,match='dtype|Float|Double'):
        model(torch.tensor([[1.,2.]],dtype=torch.float64))


@pytest.mark.parametrize('mode',[False,True])
@pytest.mark.parametrize('sample', [[0,2], [3], [1,1,1,1,1,1,1,2]])
def test_invalid_higher_rank_geometry_refused(mode,sample,registry):
    with pytest.raises(ValueError):validate(higher_graph(mode,sample),registry)
