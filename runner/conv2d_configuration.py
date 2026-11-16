"""Authored paired Conv2d choices and allocation-free channel/height/width geometry."""
if __package__:
    from .tensor_layout import BatchedLayout
    from .convolution_geometry import resolve_axis
else:
    from tensor_layout import BatchedLayout
    from convolution_geometry import resolve_axis

_FIELDS = frozenset({'in_channels','out_channels','kernel_size','stride','dilation','groups',
                     'bias','padding_policy','padding','padding_mode'})


def _integer(value, minimum, maximum, key):
    if type(value) not in (int,float) or not minimum <= value <= maximum or value != int(value):
        raise ValueError(f'Conv2d {key} requires integers from {minimum} to {maximum}.')
    return int(value)


def normalize_conv2d(config):
    if type(config) is not dict or set(config) != _FIELDS:
        raise ValueError('Conv2d requires exactly its channel, paired geometry, group, bias and padding settings.')
    result = {key:_integer(config[key],1,512,key) for key in ('in_channels','out_channels','groups')}
    for key in ('kernel_size','stride','dilation','padding'):
        value = config[key]
        if type(value) is not list or len(value) != 2:
            raise ValueError(f'Conv2d {key} must be an explicit [height,width] pair, not a scalar.')
        result[key] = [_integer(v,0 if key == 'padding' else 1,4096,key) for v in value]
    if type(config['bias']) is not bool:
        raise ValueError('Conv2d bias must be a boolean.')
    result['bias'] = config['bias']
    for key, choices in (('padding_policy',('explicit','valid','same')),
                         ('padding_mode',('zeros','reflect','replicate','circular'))):
        value = config[key]
        if type(value) is not str or value not in choices:
            raise ValueError(f'Conv2d {key} must be one of: '+', '.join(choices)+'.')
        result[key] = value
    if result['padding_policy'] != 'explicit' and result['padding'] != [0,0]:
        raise ValueError('Conv2d valid/same policies require explicit padding [0,0].')
    if result['padding_policy'] == 'same' and result['stride'] != [1,1]:
        raise ValueError('Conv2d same padding requires stride [1,1].')
    if result['in_channels'] % result['groups'] or result['out_channels'] % result['groups']:
        raise ValueError('Conv2d input and output channels must both be divisible by groups.')
    fan_in = (result['in_channels']//result['groups']) * result['kernel_size'][0] * result['kernel_size'][1]
    if result['out_channels'] * (fan_in + int(result['bias'])) > 2_000_000:
        raise ValueError('Conv2d exceeds the two-million parameter budget.')
    return result


def resolve_conv2d(config, layout):
    c = normalize_conv2d(config)
    if not isinstance(layout,BatchedLayout) or len(layout.sample) != 3:
        raise ValueError('Conv2d requires exactly channel, height and width sample axes [C,H,W].')
    channels,height,width = layout.sample
    if channels != c['in_channels']:
        raise ValueError('Conv2d input channels must match the authored configuration.')
    resolved = [resolve_axis(length,c['kernel_size'][i],c['stride'][i],c['dilation'][i],c['padding'][i],
                             c['padding_policy'],c['padding_mode'],operation='Conv2d')
                for i,length in enumerate((height,width))]
    out_height,top,bottom = resolved[0]
    out_width,left,right = resolved[1]
    if channels * (height+top+bottom) * (width+left+right) > 100_000_000:
        raise ValueError('Conv2d padded sample volume exceeds 100 million elements.')
    output = BatchedLayout((c['out_channels'],out_height,out_width))
    fan_in = (channels//c['groups']) * c['kernel_size'][0] * c['kernel_size'][1]
    work = out_height * out_width * c['out_channels'] * fan_in
    if work > 128_000_000:
        raise ValueError('Conv2d exceeds 128 million multiply-adds per observation.')
    return {'layout':output,'parameterCount':c['out_channels']*(fan_in+int(c['bias'])),
            'paddingLeft':left,'paddingRight':right,'paddingTop':top,'paddingBottom':bottom,
            'multiplyAdds':work}


def capability():
    """Fresh exact discovery for explicit [height,width] spatial configuration."""
    return {
        'version': {'major': 1, 'minor': 0, 'patch': 0},
        'adapterID': 'kwip.runner.graph-v2.conv2d', 'graphSchemaVersions': [2],
        'sampleRank': 3, 'preservesBatch': True, 'graphDTypes': ['float32'],
        'spatialParameterLength': 2,
        'channelMinimum': 1, 'channelMaximum': 512,
        'lengthMinimum': 1, 'lengthMaximum': 4096,
        'kernelMinimum': 1, 'kernelMaximum': 4096,
        'strideMinimum': 1, 'strideMaximum': 4096,
        'dilationMinimum': 1, 'dilationMaximum': 4096,
        'groupMinimum': 1, 'groupMaximum': 512,
        'paddingMinimum': 0, 'paddingMaximum': 4096,
        'maximumParameters': 2_000_000, 'maximumMultiplyAdds': 128_000_000,
        'maximumSampleElements': 100_000_000, 'maximumPaddedSampleElements': 100_000_000,
        'paddingPolicies': ['explicit', 'valid', 'same'],
        'paddingModes': ['zeros', 'reflect', 'replicate', 'circular'],
        'biasModes': [False, True],
    }
