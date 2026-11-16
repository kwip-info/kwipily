"""Shared spatial max-pooling contract; individual axes use the reviewed 1D geometry."""
if __package__:
    from .maxpool1d_configuration import normalize_maxpool1d,resolve_maxpool1d
    from .tensor_layout import BatchedLayout
else:
    from maxpool1d_configuration import normalize_maxpool1d,resolve_maxpool1d
    from tensor_layout import BatchedLayout

_FIELDS=frozenset({'kernel_size','stride','padding','dilation','ceil_mode','return_indices'})
_SPATIAL=frozenset({'kernel_size','stride','padding','dilation'})


def axis_config(config,axis):
    return {key:(value[axis] if key in _SPATIAL and value is not None else value) for key,value in config.items()}


def normalize_spatial_maxpool(config,axis_names):
    title=f'MaxPool{len(axis_names)}d'
    if type(config) is not dict or set(config)!=_FIELDS:
        raise ValueError(f'{title} requires exactly kernel, stride, padding, dilation, ceiling mode and return indices.')
    for key in _SPATIAL:
        if key=='stride' and config[key] is None:continue
        if type(config[key]) is not list or len(config[key])!=len(axis_names):
            raise ValueError(f'{title} {key} requires exactly [{", ".join(axis_names)}].')
    axes=[]
    for axis,name in enumerate(axis_names):
        try:axes.append(normalize_maxpool1d(axis_config(config,axis)))
        except ValueError as error:raise ValueError(f'{title} {name}: {error}') from error
    return {key:([axis[key] for axis in axes] if key in _SPATIAL and config[key] is not None else axes[0][key]) for key in config}


def resolve_spatial_maxpool(config,layout,axis_names):
    title=f'MaxPool{len(axis_names)}d'
    config=normalize_spatial_maxpool(config,axis_names)
    if not isinstance(layout,BatchedLayout) or len(layout.sample)!=len(axis_names)+1:
        raise ValueError(f'{title} requires sample shape [channels, {", ".join(axis_names)}].')
    output=[layout.sample[0]]
    for axis,name in enumerate(axis_names):
        try:result=resolve_maxpool1d(axis_config(config,axis),BatchedLayout((layout.sample[0],layout.sample[axis+1])))
        except ValueError as error:raise ValueError(f'{title} {name}: {error}') from error
        output.append(result['layout'].sample[1])
    resolved=BatchedLayout(tuple(output))
    return {'layout':resolved,'indicesLayout':resolved if config['return_indices'] else None}


def spatial_capability(axes):
    return {
        'version': {'major':1,'minor':0,'patch':0}, 'adapterID':f'kwip.runner.graph-v2.max_pool{len(axes)}d',
        'graphSchemaVersions':[2], 'sampleRank':len(axes)+1, 'preservesBatch':True, 'graphDTypes':['float32'],
        'channelMinimum':1,'channelMaximum':512,'spatialMinimum':1,'spatialMaximum':4096,'spatialAxes':list(axes),'indicesMeaning':'flattenedSpatialPosition',
        'kernelMinimum':1,'kernelMaximum':4096,'strideMinimum':1,'strideMaximum':4096,
        'dilationMinimum':1,'dilationMaximum':4096,'paddingMinimum':0,'paddingMaximum':2048,
        'paddingKernelDivisor':2,'nullStrideFollowsKernel':True,'ceilModes':[False,True],
        'returnIndicesModes':[False,True],'outputDTypes':{'out':'float32','indices':'int64'},
    }
