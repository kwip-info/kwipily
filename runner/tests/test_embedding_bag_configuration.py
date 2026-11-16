"""Pure configuration tests; no device execution or capability grant."""
import copy
import math
import pytest
from runner.embedding_bag_configuration import DEFAULTS, normalize_embedding_bag, configured_input_ports


def config(**changes):
    return {**DEFAULTS, **changes}


def test_defaults_and_input_immutability():
    supplied = config()
    before = copy.deepcopy(supplied)
    result = normalize_embedding_bag(supplied)
    assert result == supplied == before
    assert result is not supplied
    assert configured_input_ports(supplied) == ('indices',)


@pytest.mark.parametrize('layout,weighted,ports', [
    ('fixed', False, ('indices',)),
    ('fixed', True, ('indices', 'per_sample_weights')),
    ('packed', False, ('indices', 'offsets')),
    ('packed', True, ('indices', 'offsets', 'per_sample_weights')),
])
def test_configured_ports(layout, weighted, ports):
    assert configured_input_ports(config(layout=layout, mode='sum', per_sample_weights=weighted)) == ports


@pytest.mark.parametrize('mode', ['sum', 'mean', 'max'])
@pytest.mark.parametrize('layout', ['fixed', 'packed'])
def test_all_layout_reductions_and_terminal_flag_retained(mode, layout):
    supplied = config(mode=mode, layout=layout, include_last_offset=True, freeze=True)
    assert normalize_embedding_bag(supplied) == supplied


@pytest.mark.parametrize('padding,expected', [(-16, 0), (-1, 15), (0, 0), (15, 15), (None, None)])
def test_native_padding_canonicalization(padding, expected):
    assert normalize_embedding_bag(config(padding_idx=padding))['padding_idx'] == expected


def test_integral_dimensions_and_numeric_controls_canonicalize():
    result = normalize_embedding_bag(config(num_embeddings=5., embedding_dim=3., max_norm=2, norm_type=1))
    assert type(result['num_embeddings']) is int and result['num_embeddings'] == 5
    assert type(result['embedding_dim']) is int and result['embedding_dim'] == 3
    assert type(result['max_norm']) is float and type(result['norm_type']) is float


@pytest.mark.parametrize('changes', [
    {'num_embeddings': 0}, {'num_embeddings': True}, {'num_embeddings': 1.5},
    {'num_embeddings': float('inf')}, {'num_embeddings': 10**1000},
    {'embedding_dim': 0}, {'embedding_dim': False}, {'embedding_dim': 2.1},
    {'num_embeddings': 100_000_000, 'embedding_dim': 2},
    {'padding_idx': -17}, {'padding_idx': 16}, {'padding_idx': True}, {'padding_idx': .5},
    {'max_norm': 0}, {'max_norm': -1}, {'max_norm': True}, {'max_norm': float('nan')},
    {'max_norm': 1e-50}, {'max_norm': 1e40}, {'max_norm': '2'},
    {'norm_type': 0}, {'norm_type': None}, {'norm_type': False}, {'norm_type': float('inf')},
    {'norm_type': 1e-50}, {'norm_type': 1e40},
    {'mode': 'median'}, {'mode': None}, {'layout': 'ragged'}, {'layout': None},
    {'scale_grad_by_freq': 1}, {'sparse': 0}, {'freeze': None},
    {'include_last_offset': 'true'}, {'per_sample_weights': 1},
    {'mode': 'max', 'sparse': True}, {'mode': 'max', 'scale_grad_by_freq': True},
    {'mode': 'mean', 'per_sample_weights': True}, {'mode': 'max', 'per_sample_weights': True},
    {'mode': 'sum', 'sparse': True, 'scale_grad_by_freq': True},
])
def test_invalid_types_ranges_and_native_combinations_refuse(changes):
    with pytest.raises(ValueError):
        normalize_embedding_bag(config(**changes))
    with pytest.raises(ValueError):
        configured_input_ports(config(**changes))


@pytest.mark.parametrize('field', list(DEFAULTS))
def test_every_field_required(field):
    supplied = config(); del supplied[field]
    with pytest.raises(ValueError): normalize_embedding_bag(supplied)


@pytest.mark.parametrize('supplied', [None, [], True, {'foreign': 1}, {**DEFAULTS, 'foreign': 1}])
def test_exact_object_and_keys(supplied):
    with pytest.raises(ValueError): normalize_embedding_bag(supplied)


def test_static_product_boundary_without_allocation():
    assert normalize_embedding_bag(config(num_embeddings=100_000_000, embedding_dim=1))['num_embeddings'] == 100_000_000
    assert normalize_embedding_bag(config(num_embeddings=1, embedding_dim=100_000_000))['embedding_dim'] == 100_000_000


@pytest.mark.parametrize('changes', [
    {'mode': 'sum', 'sparse': True}, {'mode': 'mean', 'sparse': True},
    {'mode': 'sum', 'scale_grad_by_freq': True},
    {'mode': 'sum', 'per_sample_weights': True, 'sparse': True},
    {'mode': 'sum', 'per_sample_weights': True, 'freeze': True},
    {'max_norm': 1.401298464324817e-45, 'norm_type': 3.4028234663852886e38},
])
def test_valid_controls_are_retained_without_device_or_optimizer_claim(changes):
    assert normalize_embedding_bag(config(**changes)) == config(**changes)
