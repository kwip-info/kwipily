"""Check Swift-expanded composite against explicit PyTorch reference.

First export fixtures with CompositeFixtureExportTests and
KWIPML_COMPOSITE_FIXTURE_DIR, then run this script with that directory.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runner.engine import GraphModel, export_source


def verify(directory):
    directory = Path(directory)
    flat = json.loads((directory / 'flat.json').read_text())
    reference = json.loads((directory / 'reference.json').read_text())
    origins = json.loads((directory / 'origins.json').read_text())
    actual, expected = GraphModel(flat).double(), GraphModel(reference).double()
    # Match ownership by explicit origin, never initialization/topological order.
    for flat_id, key in actual.layer_keys.items():
        original = origins[flat_id]['localNodeID']
        actual.layers[key].load_state_dict(expected.layers[expected.layer_keys[original]].state_dict())
    x = torch.tensor([[.2, -.3], [1.1, .7], [-.8, .5]], dtype=torch.float64, requires_grad=True)
    a, b = actual(x), expected(x)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    ga = torch.autograd.grad(a.square().sum(), (x, *actual.parameters()))
    gb = torch.autograd.grad(b.square().sum(), (x, *expected.parameters()))
    torch.testing.assert_close(ga[0], gb[0], rtol=0, atol=0)
    actual_gradients = dict(zip([n for n, _ in actual.named_parameters()], ga[1:]))
    expected_gradients = dict(zip([n for n, _ in expected.named_parameters()], gb[1:]))
    for flat_id, key in actual.layer_keys.items():
        ref_key = expected.layer_keys[origins[flat_id]['localNodeID']]
        for parameter in ('weight', 'bias'):
            torch.testing.assert_close(actual_gradients[f'layers.{key}.{parameter}'], expected_gradients[f'layers.{ref_key}.{parameter}'], rtol=0, atol=0)
    namespace = {}
    exec(compile(export_source(actual), '<generated-model>', 'exec'), namespace)
    exported = namespace['Model']().double()
    exported.load_state_dict(actual.state_dict())
    torch.testing.assert_close(exported(x), b, rtol=0, atol=0)
    return {'status': 'passed', 'dtype': 'float64', 'device': 'cpu', 'forward': 'exact', 'inputGradient': 'exact', 'parameterGradients': 'exact', 'exportedForward': 'exact', 'torchVersion': torch.__version__}


if __name__ == '__main__':
    print(json.dumps(verify(sys.argv[1]), indent=2))
