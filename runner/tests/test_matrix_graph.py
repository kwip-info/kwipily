"""Matrix adapter integration uses a private registry until native delivery is complete."""
import json
import pytest
import torch
from runner.engine import GraphModel, REGISTRY, export_source


@pytest.fixture
def registry(tmp_path):
    value = json.loads(REGISTRY.read_text())
    value['components'] = [c for c in value['components'] if c['id'] != 'matmul']
    value['components'].append(dict(id='matmul', title='Matrix multiply', category='Tensor math',
                                    inputs=['a', 'b'], outputs=['out'], fields=[]))
    path = tmp_path / 'registry.json'
    path.write_text(json.dumps(value))
    return path


def graph():
    specs = [('input', 'input', {'features': 2}),
             ('left', 'linear', {'in_features': 2, 'out_features': 24, 'bias': True}),
             ('right', 'linear', {'in_features': 2, 'out_features': 20, 'bias': True}),
             ('a', 'reshape', {'shape': [2, 3, 4]}),
             ('b', 'reshape', {'shape': [4, 5]}), ('product', 'matmul', {}),
             ('flat', 'flatten', {'start_dim': 1, 'end_dim': -1}),
             ('head', 'linear', {'in_features': 30, 'out_features': 2, 'bias': True}),
             ('output', 'output', {})]
    links = [('input', 'left', 'in'), ('input', 'right', 'in'), ('left', 'a', 'in'),
             ('right', 'b', 'in'), ('a', 'product', 'a'), ('b', 'product', 'b'),
             ('product', 'flat', 'in'), ('flat', 'head', 'in'), ('head', 'output', 'in')]
    return dict(schemaVersion=2, nodes=[dict(id=i, type=t, config=c) for i, t, c in specs],
                edges=[dict(id=str(i), source=a, target=b, sourcePort='out', targetPort=p)
                       for i, (a, b, p) in enumerate(links)])


@pytest.mark.parametrize('rows', [1, 2, 5])
def test_graph_export_values_and_all_gradients(registry, rows):
    torch.manual_seed(209)
    model = GraphModel(graph(), registry)
    x = torch.randn(rows, 2, requires_grad=True)
    actual, values = model(x, inspect=True)
    a = model.layers['left'](x).reshape(rows, 2, 3, 4)
    b = model.layers['right'](x).reshape(rows, 4, 5)
    product = torch.stack([a[i] @ b[i] for i in range(rows)])
    expected = model.layers['head'](product.flatten(1))
    torch.testing.assert_close(values['product'], product)
    torch.testing.assert_close(actual, expected)
    source = export_source(model)
    assert 'runner' not in source
    namespace = {}
    exec(source, namespace)
    exported = namespace['Model']()
    exported.load_state_dict(model.state_dict())
    torch.testing.assert_close(exported(x), expected)
    reference = torch.autograd.grad(expected.square().sum(), (x, *model.parameters()), retain_graph=True)
    native = torch.autograd.grad(actual.square().sum(), (x, *model.parameters()))
    portable = torch.autograd.grad(exported(x).square().sum(), (x, *exported.parameters()))
    for wanted, observed, transferred in zip(reference, native, portable):
        torch.testing.assert_close(observed, wanted)
        torch.testing.assert_close(transferred, wanted)


@pytest.mark.parametrize('version,config', [(1, {}), (2, {'transpose': True})])
def test_matrix_graph_rejects_unversioned_or_unrecognized_configuration(registry, version, config):
    value = graph()
    value['schemaVersion'] = version
    value['nodes'][5]['config'] = config
    with pytest.raises(ValueError):
        GraphModel(value, registry)


def test_discovery_describes_exact_matrix_execution_boundary():
    from runner.engine import capabilities
    assert capabilities()['componentCapabilities']['matrixMultiply'] == {
        'version': {'major': 1, 'minor': 0, 'patch': 0},
        'adapterID': 'kwip.runner.graph-v2.matrixMultiply',
        'graphSchemaVersions': [2], 'maximumRank': 8, 'preservesBatch': True,
        'minimumSampleRank': 2, 'sampleBroadcasting': True,
        'maximumOutputElements': 100_000_000, 'graphDTypes': ['float32'],
    }
