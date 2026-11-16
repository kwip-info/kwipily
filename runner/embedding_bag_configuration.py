"""Torch-free EmbeddingBag configuration1 and configured-input availability.

This admits static configuration, not table allocation, packet values, a device or
an optimizer. Construction additionally requires the shared aggregate2M model
parameter guard, selected initialization200k limit, and packet/runtime budgets.
"""
import math
import struct

MAX_ELEMENTS = 100_000_000
FLOAT32_MAX = 3.4028234663852886e38
DEFAULTS = dict(num_embeddings=16, embedding_dim=4, mode='mean', padding_idx=None,
                max_norm=None, norm_type=2., scale_grad_by_freq=False, sparse=False,
                include_last_offset=False, freeze=False, layout='fixed', per_sample_weights=False)
FIELDS = frozenset(DEFAULTS)


def _integer(value, minimum, maximum, name):
    if type(value) not in (int, float) or not minimum <= value <= maximum or value != int(value):
        raise ValueError(f'{name} requires an integer from {minimum} to {maximum}.')
    return int(value)


def _positive(value, name):
    if type(value) not in (int, float) or not 0 < value <= FLOAT32_MAX:
        raise ValueError(f'{name} requires a positive finite float32-representable number.')
    number = float(value)
    if not math.isfinite(number) or struct.unpack('f', struct.pack('f', number))[0] == 0:
        raise ValueError(f'{name} must not underflow to zero in float32.')
    return number


def normalize_embedding_bag(config):
    """Validate complete declared fields; return a fresh canonical dictionary."""
    if type(config) is not dict or set(config) != FIELDS:
        raise ValueError('EmbeddingBag requires exactly its declared configuration1 fields.')
    count = _integer(config['num_embeddings'], 1, MAX_ELEMENTS, 'num_embeddings')
    width = _integer(config['embedding_dim'], 1, MAX_ELEMENTS // count, 'embedding_dim')
    if type(config['mode']) is not str or config['mode'] not in ('mean', 'sum', 'max'):
        raise ValueError('Choose mean, sum or max bag reduction.')
    if type(config['layout']) is not str or config['layout'] not in ('fixed', 'packed'):
        raise ValueError('Choose fixed rows or packed bags.')
    for name in ('scale_grad_by_freq', 'sparse', 'include_last_offset', 'freeze', 'per_sample_weights'):
        if type(config[name]) is not bool:
            raise ValueError(f'{name} requires a Boolean value.')
    padding = config['padding_idx']
    if padding is not None:
        padding = _integer(padding, -count, count - 1, 'padding_idx') % count
    maximum = config['max_norm']
    if maximum is not None:
        maximum = _positive(maximum, 'max_norm')
    norm = _positive(config['norm_type'], 'norm_type')
    if config['mode'] == 'max' and (config['sparse'] or config['scale_grad_by_freq']):
        raise ValueError('Max bag reduction does not support sparse or frequency-scaled gradients.')
    if config['per_sample_weights'] and config['mode'] != 'sum':
        raise ValueError('Per-item weights require sum bag reduction.')
    if config['sparse'] and config['scale_grad_by_freq']:
        raise ValueError('Sparse bag gradients do not support frequency scaling.')
    return {**config, 'num_embeddings': count, 'embedding_dim': width,
            'padding_idx': padding, 'max_norm': maximum, 'norm_type': norm}


def configured_input_ports(config):
    """Ordered real inputs; packet validation separately checks their values."""
    config = normalize_embedding_bag(config)
    return ('indices',) + (('offsets',) if config['layout'] == 'packed' else ()) + (
        ('per_sample_weights',) if config['per_sample_weights'] else ())
