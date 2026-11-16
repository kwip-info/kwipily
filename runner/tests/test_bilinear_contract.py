import pytest
from runner.graph_semantics import effective_config
from runner.tensor_layout import BatchedLayout


def config():
    return dict(in1_features=3, in2_features=5, out_features=2, bias=True)


def test_schema_two_canonicalizes_integral_widths_and_preserves_bias():
    for bias in (True, False):
        value = {'in1_features': 3.0, 'in2_features': 5, 'out_features': 2, 'bias': bias}
        result = effective_config({'type': 'bilinear', 'config': value}, 2)
        assert result == {**value, 'in1_features': 3}
        assert type(result['in1_features']) is int
        assert result['bias'] is bias


@pytest.mark.parametrize('patch', [
    {'in1_features': True}, {'in2_features': 0}, {'out_features': 513},
    {'in1_features': 1.5}, {'in2_features': float('inf')}, {'bias': 1},
    {'extra': 1}, {'in1_features': 512, 'in2_features': 512, 'out_features': 512},
])
def test_invalid_configuration_refused(patch):
    with pytest.raises(ValueError):
        effective_config({'type': 'bilinear', 'config': {**config(), **patch}}, 2)


def test_explicit_fields_and_schema_required():
    for field in config():
        value = config(); del value[field]
        with pytest.raises(ValueError):
            effective_config({'type': 'bilinear', 'config': value}, 2)
    with pytest.raises(ValueError):
        effective_config({'type': 'bilinear', 'config': config()}, 1)


@pytest.mark.parametrize('prefix', [(), (4,), (2, 1), (1, 2, 1, 2, 1, 2)])
def test_layout_preserves_matching_prefix_with_independent_widths(prefix):
    left, right = BatchedLayout(prefix + (3,)), BatchedLayout(prefix + (5,))
    assert left.bilinear(right, 3, 5, 2).sample == prefix + (2,)


@pytest.mark.parametrize('left,right,widths', [
    ((1, 3), (4, 5), (3, 5, 2)), ((3,), (1, 5), (3, 5, 2)),
    ((4,), (5,), (3, 5, 2)), ((3,), (4,), (3, 5, 2)),
    ((3,), (5,), (True, 5, 2)), ((3,), (5,), (3, 5, False)),
    ((1000000, 3), (1000000, 5), (3, 5, 512)),
])
def test_layout_rejects_broadcast_width_mismatch_and_output_overflow(left, right, widths):
    with pytest.raises(ValueError):
        BatchedLayout(left).bilinear(BatchedLayout(right), *widths)


def test_exact_parameter_limit_accepts_no_bias_but_rejects_added_bias():
    value = {'in1_features': 500, 'in2_features': 400, 'out_features': 10, 'bias': False}
    assert effective_config({'type': 'bilinear', 'config': value}, 2) == value
    with pytest.raises(ValueError, match='parameter budget'):
        effective_config({'type': 'bilinear', 'config': {**value, 'bias': True}}, 2)
