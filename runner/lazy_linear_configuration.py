"""Authored LazyLinear choices and allocation-free resolution from an upstream layout."""
if __package__:
    from .tensor_layout import BatchedLayout
else:
    from tensor_layout import BatchedLayout


def normalize_lazy_linear(config):
    if type(config) is not dict or set(config) != {'out_features', 'bias'}:
        raise ValueError('LazyLinear requires output features and bias; input width is inferred.')
    width = config['out_features']
    if type(width) not in (int, float) or not 1 <= width <= 512 or width != int(width):
        raise ValueError('LazyLinear output features must be an integer from 1 to 512.')
    if type(config['bias']) is not bool:
        raise ValueError('LazyLinear bias must be a boolean.')
    return {'out_features': int(width), 'bias': config['bias']}


def resolve_lazy_linear(config, layout):
    config = normalize_lazy_linear(config)
    if not isinstance(layout, BatchedLayout):
        raise ValueError('LazyLinear requires a validated batched layout.')
    width = layout.sample[-1]
    if not 1 <= width <= 4096:
        raise ValueError('LazyLinear inferred input features must be from 1 to 4096.')
    output = config['out_features']
    count = output * (width + int(config['bias']))
    if count > 2_000_000:
        raise ValueError('LazyLinear exceeds the two-million parameter budget.')
    return {'inputFeatures': width, 'outputFeatures': output, 'bias': config['bias'],
            'parameterCount': count, 'layout': BatchedLayout((*layout.sample[:-1], output))}


def capability():
    """Fresh JSON descriptor for the reviewed, explicitly materialized adapter."""
    return {
        'version': {'major': 1, 'minor': 0, 'patch': 0},
        'adapterID': 'kwip.runner.graph-v2.lazy-linear',
        'graphSchemaVersions': [2], 'maximumRank': 8, 'preservesBatch': True,
        'minimumSampleRank': 1, 'maximumSampleElements': 100_000_000,
        'maximumParameters': 2_000_000, 'inputFeatureMinimum': 1,
        'inputFeatureMaximum': 4096, 'outputFeatureMinimum': 1,
        'outputFeatureMaximum': 512, 'biasModes': [False, True],
        'graphDTypes': ['float32'], 'inputWidthInferred': True,
        'materializationPolicy': 'validatedShapeBeforeOptimizer',
    }
