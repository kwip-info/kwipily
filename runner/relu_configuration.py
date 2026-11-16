"""Allocation-free legacy-compatible ReLU contract for dense float32 stages."""
if __package__:
    from .tensor_layout import BatchedLayout
else:
    from tensor_layout import BatchedLayout


def normalize_relu(config):
    if type(config) is not dict or config:
        raise ValueError('ReLU requires an empty configuration; each value is clipped independently without an axis, learned parameters or in-place mutation.')
    return {}


def resolve_relu(config, layout):
    normalize_relu(config)
    if not isinstance(layout, BatchedLayout):
        raise ValueError('ReLU requires a batch-preserving tensor layout with 1..7 sample axes.')
    return {'layout': layout, 'parameterCount': 0}


def capability():
    return {'version': {'major': 1, 'minor': 1, 'patch': 0},
            'adapterID': 'kwip.runner.graph-v2.relu', 'graphSchemaVersions': [1, 2],
            'graphDTypes': ['float32'], 'configuration': {}, 'outputDTypes': {'out': 'float32'},
            'maximumRank': 8, 'preservesBatch': True, 'preservesShape': True,
            'parameterless': True, 'deterministic': True, 'inplace': False}
