"""Explicit D/H/W Conv3d configuration and bounded, allocation-free geometry."""
import math
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
        raise ValueError(f'Conv3d {key} requires integers from {minimum} to {maximum}.')
    return int(value)


def normalize_conv3d(config):
    if type(config) is not dict or set(config) != _FIELDS:
        raise ValueError('Conv3d requires exactly its channel, spatial triple, group, bias and padding settings.')
    result = {key:_integer(config[key],1,512,key) for key in ('in_channels','out_channels','groups')}
    for key in ('kernel_size','stride','dilation','padding'):
        value = config[key]
        if type(value) is not list or len(value) != 3:
            raise ValueError(f'Conv3d {key} must be an explicit [depth,height,width] triple, not a scalar.')
        result[key] = [_integer(v,0 if key == 'padding' else 1,4096,key) for v in value]
    if type(config['bias']) is not bool:
        raise ValueError('Conv3d bias must be a boolean.')
    result['bias'] = config['bias']
    for key,choices in (('padding_policy',('explicit','valid','same')),
                        ('padding_mode',('zeros','reflect','replicate','circular'))):
        value = config[key]
        if type(value) is not str or value not in choices:
            raise ValueError(f'Conv3d {key} must be one of: '+', '.join(choices)+'.')
        result[key] = value
    if result['padding_policy'] != 'explicit' and result['padding'] != [0,0,0]:
        raise ValueError('Conv3d valid/same policies require explicit padding [0,0,0].')
    if result['padding_policy'] == 'same' and result['stride'] != [1,1,1]:
        raise ValueError('Conv3d same padding requires stride [1,1,1].')
    if result['in_channels'] % result['groups'] or result['out_channels'] % result['groups']:
        raise ValueError('Conv3d input and output channels must both be divisible by groups.')
    fan_in = (result['in_channels']//result['groups']) * math.prod(result['kernel_size'])
    if result['out_channels'] * (fan_in + int(result['bias'])) > 2_000_000:
        raise ValueError('Conv3d exceeds the two-million parameter budget.')
    return result


def resolve_conv3d(config, layout):
    c = normalize_conv3d(config)
    if not isinstance(layout,BatchedLayout) or len(layout.sample) != 4:
        raise ValueError('Conv3d requires channel, depth, height and width sample axes [C,D,H,W].')
    channels,*spatial = layout.sample
    if channels != c['in_channels']:
        raise ValueError('Conv3d input channels must match the authored configuration.')
    axes = [resolve_axis(length,c['kernel_size'][i],c['stride'][i],c['dilation'][i],c['padding'][i],
                         c['padding_policy'],c['padding_mode'],operation='Conv3d')
            for i,length in enumerate(spatial)]
    if channels * math.prod(length+axis[1]+axis[2] for length,axis in zip(spatial,axes)) > 100_000_000:
        raise ValueError('Conv3d padded sample volume exceeds 100 million elements.')
    output = BatchedLayout((c['out_channels'],*(axis[0] for axis in axes)))
    fan_in = (channels//c['groups']) * math.prod(c['kernel_size'])
    work = math.prod(output.sample) * fan_in
    if work > 128_000_000:
        raise ValueError('Conv3d exceeds 128 million multiply-adds per observation.')
    return {'layout':output,'parameterCount':c['out_channels']*(fan_in+int(c['bias'])),
            'paddingFront':axes[0][1],'paddingBack':axes[0][2],
            'paddingTop':axes[1][1],'paddingBottom':axes[1][2],
            'paddingLeft':axes[2][1],'paddingRight':axes[2][2],'multiplyAdds':work}


def capability():
    """Fresh exact discovery for explicit [depth,height,width] spatial configuration."""
    return {
        'version': {'major': 1, 'minor': 0, 'patch': 0},
        'adapterID': 'kwip.runner.graph-v2.conv3d', 'graphSchemaVersions': [2],
        'sampleRank': 4, 'preservesBatch': True, 'graphDTypes': ['float32'],
        'spatialParameterLength': 3,
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
