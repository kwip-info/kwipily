"""Allocation-free legacy-compatible Tanh contract for dense float32 stages."""
if __package__:
    from .tensor_layout import BatchedLayout
else:
    from tensor_layout import BatchedLayout


def normalize_tanh(config):
    if type(config) is not dict or config:
        raise ValueError('Tanh requires an empty configuration; each value is transformed independently without an axis or learned parameters.')
    return {}


def resolve_tanh(config, layout):
    normalize_tanh(config)
    if not isinstance(layout, BatchedLayout):
        raise ValueError('Tanh requires a batch-preserving tensor layout with 1..7 sample axes.')
    return {'layout': layout, 'parameterCount': 0}


def capability():
    return {'version': {'major': 1, 'minor': 1, 'patch': 0},
            'adapterID': 'kwip.runner.graph-v2.tanh', 'graphSchemaVersions': [1, 2],
            'graphDTypes': ['float32'], 'configuration': {}, 'outputDTypes': {'out': 'float32'},
            'maximumRank': 8, 'preservesBatch': True, 'preservesShape': True,
            'parameterless': True, 'deterministic': True}
