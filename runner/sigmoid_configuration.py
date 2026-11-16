"""Allocation-free parameterless Sigmoid contract for dense float32 graph stages."""
if __package__:
    from .tensor_layout import BatchedLayout
else:
    from tensor_layout import BatchedLayout


def normalize_sigmoid(config):
    if type(config) is not dict or config:
        raise ValueError('Sigmoid requires an empty configuration; each value is transformed independently without an axis or learned parameters.')
    return {}


def resolve_sigmoid(config, layout):
    normalize_sigmoid(config)
    if not isinstance(layout, BatchedLayout):
        raise ValueError('Sigmoid requires a batch-preserving tensor layout with 1..7 sample axes.')
    return {'layout':layout, 'parameterCount':0}


def capability():
    return {'version':{'major':1,'minor':0,'patch':0},
            'adapterID':'kwip.runner.graph-v2.sigmoid','graphSchemaVersions':[2],
            'graphDTypes':['float32'],'configuration':{},'outputDTypes':{'out':'float32'},
            'maximumRank':8,'preservesBatch':True,'preservesShape':True,
            'parameterless':True,'deterministic':True}
