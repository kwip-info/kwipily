import pytest
import torch
from runner.tensor_layout import BatchedLayout, apply_layout


def test_batch_shape_and_axis_contract():
    a = BatchedLayout((3, 4))
    assert a.reshape((2, -1)).sample == (2, 6)
    assert a.transpose(1, -1).sample == (4, 3)
    assert a.flatten(1, -1).sample == (12,)
    assert a.linear(4, 5).sample == (3, 5)
    assert a.concat(BatchedLayout((7, 4)), 1).sample == (10, 4)
    for action in [lambda: a.transpose(0, 1), lambda: a.reshape((5, -1)),
                   lambda: a.reshape((-1, -1)), lambda: a.add(BatchedLayout((1, 4))),
                   lambda: a.flatten(2, 1), lambda: a.transpose(True, 1)]:
        with pytest.raises(ValueError): action()


@pytest.mark.parametrize('shape', [(), (0,), (-1,), (True,), (1.0,), (100_000_001,), (1,) * 8])
def test_bounded_concrete_sample(shape):
    with pytest.raises(ValueError): BatchedLayout(shape)


def test_transpose_changes_value_order_and_backward_routes_axes():
    x = torch.arange(24., requires_grad=True).reshape(2, 3, 4)
    y = apply_layout(x, 'transpose', {'dim0': 1, 'dim1': -1})
    assert y.shape == (2, 4, 3)
    assert y[0].tolist() == [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]]
    gradient, = torch.autograd.grad((y * torch.arange(12.).reshape(1, 4, 3)).sum(), x)
    assert gradient[0].tolist() == [[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]]
    flattened = apply_layout(y, 'flatten', {'start_dim': 1, 'end_dim': -1})
    assert flattened[0].tolist() == [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]


def test_reshape_retains_logical_order_and_handles_noncontiguous_inputs():
    x = torch.arange(24., requires_grad=True).reshape(2, 3, 4)
    y = apply_layout(x, 'reshape', {'shape': [2, -1]})
    assert y.shape == (2, 2, 6)
    assert y[0].flatten().tolist() == list(range(12))
    gradient, = torch.autograd.grad(y.sum(), x)
    assert torch.equal(gradient, torch.ones_like(x))
    transposed = x.transpose(1, 2)
    assert not transposed.is_contiguous()
    z = apply_layout(transposed, 'reshape', {'shape': [-1]})
    assert z[0].tolist() == [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
    with pytest.raises(ValueError): apply_layout(x, 'reshape', {'shape': [12], 'extra': 1})


def test_configuration_normalizes_json_integer_numbers_before_execution():
    from runner.tensor_layout import normalize_configuration
    assert normalize_configuration('reshape', {'shape': [2.0, -1.0]}) == {'shape': [2, -1]}
    normalized = normalize_configuration('transpose', {'dim0': 1.0, 'dim1': -1.0})
    assert all(type(v) is int for v in normalized.values())
    x = torch.arange(24.).reshape(2, 3, 4)
    assert apply_layout(x, 'reshape', {'shape': [2.0, -1.0]}).shape == (2, 2, 6)
    assert torch.equal(apply_layout(x, 'transpose', normalized), x.transpose(1, -1))


@pytest.mark.parametrize('operation,config', [
    ('reshape', {'shape': []}), ('reshape', {'shape': [True, 2]}),
    ('reshape', {'shape': [float('nan')]}), ('reshape', {'shape': [float('inf')]}),
    ('reshape', {'shape': [2.5]}), ('reshape', {'shape': [-1, -1]}),
    ('reshape', {'shape': [100_000_000, 2]}), ('reshape', {'shape': [1] * 8}),
    ('reshape', {'shape': [0, 2]}), ('reshape', {'shape': (2, 3)}),
    ('reshape', {'shape': [2], 'extra': 1}), ('transpose', {'dim0': 0, 'dim1': 1}),
    ('transpose', {'dim0': 8, 'dim1': 1}), ('transpose', {'dim0': True, 'dim1': 1}),
    ('flatten', {'start_dim': 1}), ('unknown', {}),
])
def test_configuration_rejects_invalid_settings_without_input(operation, config):
    from runner.tensor_layout import normalize_configuration
    with pytest.raises(ValueError): normalize_configuration(operation, config)
