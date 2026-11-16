"""Allocation-free, batch-preserving adaptive-pooling contracts.

Adaptive bins may overlap and may be larger than their input. Null 2D axes keep
that input dimension. Device-specific execution choices belong to the adapter.
"""
if __package__:
    from .tensor_layout import BatchedLayout
else:
    from tensor_layout import BatchedLayout

_OPERATIONS=frozenset({'adaptive_avg_pool1d','adaptive_avg_pool2d','adaptive_max_pool2d'})
_TITLES={'adaptive_avg_pool1d':'AdaptiveAvgPool1d','adaptive_avg_pool2d':'AdaptiveAvgPool2d','adaptive_max_pool2d':'AdaptiveMaxPool2d'}


def normalize_adaptive_pooling(operation,config):
    if operation not in _OPERATIONS:raise ValueError('Unsupported adaptive pooling operation.')
    title=_TITLES[operation]
    keys={'output_size'} if operation.endswith('1d') else {'output_height','output_width'}
    if operation=='adaptive_max_pool2d':keys.add('return_indices')
    if type(config) is not dict or set(config)!=keys:
        raise ValueError(f'{title} requires exactly its declared output size settings'+(' and return indices.' if 'return_indices' in keys else '.'))
    result={}
    for key in sorted(keys-{'return_indices'}):
        value=config[key]
        if value is None and key!='output_size':result[key]=None;continue
        if type(value) not in (int,float) or not 1<=value<=4096 or value!=int(value):
            raise ValueError(f'{title} {key} must be an integer from 1 to 4096'+(' or null to keep that input axis.' if key!='output_size' else '.'))
        result[key]=int(value)
    if 'return_indices' in keys:
        if type(config['return_indices']) is not bool:raise ValueError(f'{title} return indices must be Boolean.')
        result['return_indices']=config['return_indices']
    return result


def torch_output_size(operation,config):
    config=normalize_adaptive_pooling(operation,config)
    return config['output_size'] if operation.endswith('1d') else (config['output_height'],config['output_width'])


def resolve_adaptive_pooling(operation,config,layout):
    config=normalize_adaptive_pooling(operation,config);title=_TITLES[operation]
    axes=('length',) if operation.endswith('1d') else ('height','width')
    if not isinstance(layout,BatchedLayout) or len(layout.sample)!=len(axes)+1:
        raise ValueError(f'{title} requires sample shape [channels, {", ".join(axes)}].')
    if not 1<=layout.sample[0]<=512 or any(not 1<=v<=4096 for v in layout.sample[1:]):
        raise ValueError(f'{title} requires channels 1–512 and spatial sizes 1–4096.')
    raw=(config['output_size'],) if operation.endswith('1d') else (config['output_height'],config['output_width'])
    output=tuple(length if size is None else size for length,size in zip(layout.sample[1:],raw))
    try:resolved=BatchedLayout((layout.sample[0],*output))
    except ValueError as error:raise ValueError(f'{title} output exceeds the 100M elements per observation budget. Reduce output sizes or channels.') from error
    return {'layout':resolved,'indicesLayout':resolved if config.get('return_indices',False) else None}


def capability(operation):
    if operation not in _OPERATIONS:raise ValueError('Unsupported adaptive pooling operation.')
    one=operation.endswith('1d');maximum=operation=='adaptive_max_pool2d'
    result={'version':{'major':1,'minor':0,'patch':0},'adapterID':f'kwip.runner.graph-v2.{operation}',
            'graphSchemaVersions':[2],'sampleRank':2 if one else 3,'preservesBatch':True,'graphDTypes':['float32'],
            'channelMinimum':1,'channelMaximum':512,'spatialMinimum':1,'spatialMaximum':4096,
            'spatialAxes':['length'] if one else ['height','width'],'outputSizeMinimum':1,'outputSizeMaximum':4096,
            'nullOutputKeepsInputAxis':not one,'outputDTypes':{'out':'float32'},'outputElementMaximum':100_000_000}
    if maximum:result.update(returnIndicesModes=[False,True],indicesMeaning='flattenedSpatialPosition',outputDTypes={'out':'float32','indices':'int64'})
    return result
