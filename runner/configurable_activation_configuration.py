"""Strict out-of-place, parameterless schema-2 activation contracts."""
if __package__:
    from .tensor_layout import BatchedLayout
else:
    from tensor_layout import BatchedLayout

MAXIMUM = 3.4028234663852886e38
PARAMETERS = {'silu': None, 'leaky_relu': ('negative_slope', .01), 'elu': ('alpha', 1.)}


def normalize_configuration(kind, config):
    if kind not in PARAMETERS:
        raise ValueError('Unknown configurable activation.')
    parameter = PARAMETERS[kind]
    allowed = set() if parameter is None else {parameter[0]}
    if type(config) is not dict or set(config) - allowed:
        raise ValueError(f'{kind}: only its declared coefficient is configurable; in-place mutation and unknown settings are not supported.')
    if parameter is None:
        return {}
    key, default = parameter
    value = config.get(key, default)
    # Bounds reject NaN/infinity/overflow before conversion, including huge Python ints.
    if type(value) not in (int, float) or not -MAXIMUM <= value <= MAXIMUM:
        raise ValueError(f'{kind}: {key} must be a finite signed float32-range number, excluding Boolean values.')
    return {key: float(value)}


def resolve_configuration(kind, config, layout):
    normalize_configuration(kind, config)
    if not isinstance(layout, BatchedLayout):
        raise ValueError(f'{kind}: requires a batch-preserving layout with 1..7 sample axes.')
    return {'layout': layout, 'parameterCount': 0}


def capability(kind):
    if kind not in PARAMETERS:
        raise ValueError('Unknown configurable activation.')
    parameter = PARAMETERS[kind]
    configuration = {} if parameter is None else {parameter[0]: {'default': parameter[1], 'minimum': -MAXIMUM, 'maximum': MAXIMUM}}
    return {'version': {'major': 1, 'minor': 0, 'patch': 0},
            'adapterID': 'kwip.runner.graph-v2.' + kind, 'graphSchemaVersions': [2],
            'graphDTypes': ['float32'], 'configuration': configuration, 'outputDTypes': {'out': 'float32'},
            'maximumRank': 8, 'preservesBatch': True, 'preservesShape': True,
            'parameterless': True, 'deterministic': True, 'inplace': False}
