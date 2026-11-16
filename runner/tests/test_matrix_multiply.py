import pytest
import torch
from runner.tensor_layout import BatchedLayout
from runner.matrix_multiply import matrix_multiply


@pytest.mark.parametrize('left,right', [
    ((3, 4), (4, 5)), ((2, 3, 4), (4, 5)),
    ((3, 4), (2, 4, 5)), ((2, 1, 3, 4), (7, 4, 5)),
])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_values_and_gradients_match_independent_observations(left, right, dtype):
    generator = torch.Generator().manual_seed(318)
    a = torch.randn((2,) + left, generator=generator, dtype=dtype, requires_grad=True)
    b = torch.randn((2,) + right, generator=generator, dtype=dtype, requires_grad=True)
    actual = matrix_multiply(a, b)
    expected = torch.stack([a[i] @ b[i] for i in range(2)])
    torch.testing.assert_close(actual, expected)
    weights = torch.randn(actual.shape, generator=generator, dtype=dtype)
    actual_grad = torch.autograd.grad((actual * weights).sum(), (a, b), retain_graph=True)
    expected_grad = torch.autograd.grad((expected * weights).sum(), (a, b))
    for x, y in zip(actual_grad, expected_grad):
        torch.testing.assert_close(x, y)
    assert BatchedLayout(left).matmul(BatchedLayout(right)).sample == tuple(actual.shape[1:])


def test_no_cross_observation_gradient_and_noncontiguous_operands():
    a = torch.arange(48., dtype=torch.float64).reshape(2, 2, 4, 3).transpose(-1, -2).requires_grad_()
    b = torch.arange(40., dtype=torch.float64).reshape(2, 5, 4).transpose(-1, -2).requires_grad_()
    assert not a.is_contiguous() and not b.is_contiguous()
    y = matrix_multiply(a, b)
    torch.testing.assert_close(y, torch.stack([a[i] @ b[i] for i in range(2)]))
    da, db = torch.autograd.grad(y[0].sum(), (a, b))
    assert torch.count_nonzero(da[1]) == torch.count_nonzero(db[1]) == 0
    assert torch.count_nonzero(da[0]) > 0 and torch.count_nonzero(db[0]) > 0


@pytest.mark.parametrize('left,right', [((3, 4), (3, 5)), ((2, 3, 4), (7, 4, 5)),
                                         ((100_000, 1), (1, 100_000)), ((4,), (4, 5))])
def test_invalid_shapes_rejected_before_allocation(left, right):
    with pytest.raises(ValueError):
        BatchedLayout(left).matmul(BatchedLayout(right))


@pytest.mark.parametrize('a,b', [
    (torch.ones(2, 3, 4), torch.ones(1, 4, 5)),
    (torch.ones(2, 3, 4), torch.ones(2, 4, 5, dtype=torch.float64)),
    (torch.ones(2, 3, 4, dtype=torch.int64), torch.ones(2, 4, 5, dtype=torch.int64)),
    (torch.ones(0, 3, 4), torch.ones(0, 4, 5)),
    (torch.ones(2, 4), torch.ones(2, 4, 5)),
    (None, torch.ones(2, 4, 5)),
    (torch.ones(2, 3, 4).to_sparse(), torch.ones(2, 4, 5)),
])
def test_invalid_runtime_contract(a, b):
    with pytest.raises(ValueError):
        matrix_multiply(a, b)


def test_finite_difference_gradient_with_broadcasted_heads():
    generator = torch.Generator().manual_seed(901)
    a = torch.randn(2, 2, 2, 3, generator=generator, dtype=torch.float64, requires_grad=True)
    b = torch.randn(2, 3, 2, generator=generator, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(matrix_multiply, (a, b))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='Apple GPU unavailable')
def test_mps_value_and_backward_parity():
    generator = torch.Generator().manual_seed(772)
    a = torch.randn(2, 3, 2, 4, generator=generator, requires_grad=True)
    b = torch.randn(2, 4, 5, generator=generator, requires_grad=True)
    expected = torch.stack([a[i] @ b[i] for i in range(2)])
    expected_grad = torch.autograd.grad(expected.square().sum(), (a, b))
    gpu_a = a.detach().to('mps').requires_grad_()
    gpu_b = b.detach().to('mps').requires_grad_()
    actual = matrix_multiply(gpu_a, gpu_b)
    actual_grad = torch.autograd.grad(actual.square().sum(), (gpu_a, gpu_b))
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-5)
    for x, y in zip(actual_grad, expected_grad):
        torch.testing.assert_close(x.cpu(), y, rtol=1e-4, atol=1e-5)
    with pytest.raises(ValueError, match='device'):
        matrix_multiply(a, gpu_b)


def test_total_output_budget_includes_observation_count():
    # Expanded views keep this rejection test tiny; no large tensor allocation is needed.
    a = torch.ones(1, 1, 1).expand(2, 10_000, 1)
    b = torch.ones(1, 1, 1).expand(2, 1, 10_000)
    with pytest.raises(ValueError, match='budget'):
        matrix_multiply(a, b)
