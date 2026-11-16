"""Allocation-free MaxPool1d settings and declared values/indices geometry."""
if __package__:
    from .tensor_layout import BatchedLayout
else:
    from tensor_layout import BatchedLayout

_FIELDS = frozenset({'kernel_size','stride','padding','dilation','ceil_mode','return_indices'})


def normalize_maxpool1d(config):
    if type(config) is not dict or set(config) != _FIELDS:
        raise ValueError('MaxPool1d requires exactly kernel, stride, padding, dilation, ceiling mode and return indices.')
    result = {}
    for key in ('kernel_size','stride','padding','dilation'):
        value = config[key]
        if key == 'stride' and value is None:
            result[key] = None
            continue
        low = 0 if key == 'padding' else 1
        high = result['kernel_size']//2 if key == 'padding' else 4096
        if type(value) not in (int,float) or not low <= value <= high or value != int(value):
            raise ValueError(f'MaxPool1d {key} must be an integer from {low} to {high}.')
        result[key] = int(value)
    for key in ('ceil_mode','return_indices'):
        if type(config[key]) is not bool:
            raise ValueError(f'MaxPool1d {key} must be Boolean.')
        result[key] = config[key]
    return result


def resolve_maxpool1d(config, layout):
    config = normalize_maxpool1d(config)
    if not isinstance(layout,BatchedLayout) or len(layout.sample) != 2:
        raise ValueError('MaxPool1d requires channel and length sample axes [C,L].')
    channels,length = layout.sample
    if not 1 <= channels <= 512 or not 1 <= length <= 4096:
        raise ValueError('MaxPool1d requires channels1–512 and length1–4096.')
    stride = config['stride'] or config['kernel_size']
    numerator = length + 2*config['padding'] - config['dilation']*(config['kernel_size']-1) - 1
    output = (numerator + (stride-1 if config['ceil_mode'] else 0))//stride + 1
    if config['ceil_mode'] and (output-1)*stride >= length+config['padding']:
        output -= 1
    if output <= 0:
        raise ValueError('MaxPool1d produces no output positions. Reduce kernel/dilation or adjust padding and ceiling mode.')
    resolved = BatchedLayout((channels,output))
    return {'layout':resolved,'indicesLayout':resolved if config['return_indices'] else None}


def capability():
    return {
        'version': {'major':1,'minor':0,'patch':0}, 'adapterID':'kwip.runner.graph-v2.max_pool1d',
        'graphSchemaVersions':[2], 'sampleRank':2, 'preservesBatch':True, 'graphDTypes':['float32'],
        'channelMinimum':1,'channelMaximum':512,'lengthMinimum':1,'lengthMaximum':4096,
        'kernelMinimum':1,'kernelMaximum':4096,'strideMinimum':1,'strideMaximum':4096,
        'dilationMinimum':1,'dilationMaximum':4096,'paddingMinimum':0,'paddingMaximum':2048,
        'paddingKernelDivisor':2,'nullStrideFollowsKernel':True,'ceilModes':[False,True],
        'returnIndicesModes':[False,True],'outputDTypes':{'out':'float32','indices':'int64'},
    }
