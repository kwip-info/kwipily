import pytest
from runner.tensor_layout import BatchedLayout


def api():
    from runner.lazy_linear_configuration import normalize_lazy_linear, resolve_lazy_linear
    return normalize_lazy_linear, resolve_lazy_linear


def test_preserves_prefix_infers_last_axis_and_does_not_save_width():
    normalize, resolve = api()
    for bias in (False, True):
        config = normalize({'out_features': 7.0, 'bias': bias})
        assert config == {'out_features': 7, 'bias': bias}
        result = resolve(config, BatchedLayout((2, 3, 11)))
        assert result == {'inputFeatures': 11, 'outputFeatures': 7, 'bias': bias,
                          'parameterCount': 77 + (7 if bias else 0), 'layout': BatchedLayout((2, 3, 7))}
        assert 'in_features' not in config


@pytest.mark.parametrize('bad', [None, {}, {'out_features': True, 'bias': True},
    {'out_features': 2.5, 'bias': True}, {'out_features': 0, 'bias': True},
    {'out_features': 513, 'bias': True}, {'out_features': 2, 'bias': 1},
    {'out_features': 2, 'bias': True, 'in_features': 3}])
def test_strict_configuration(bad):
    normalize, _ = api()
    with pytest.raises(ValueError): normalize(bad)


def test_inferred_width_and_parameter_bounds():
    _, resolve = api()
    assert resolve({'out_features': 488, 'bias': True}, BatchedLayout((4096,)))['parameterCount'] == 1999336
    with pytest.raises(ValueError): resolve({'out_features': 489, 'bias': True}, BatchedLayout((4096,)))
    with pytest.raises(ValueError): resolve({'out_features': 2, 'bias': True}, BatchedLayout((4097,)))
