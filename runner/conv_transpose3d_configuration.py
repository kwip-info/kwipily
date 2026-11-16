"""Bounded triple transposed convolution; explicit [depth,height,width] settings, zeros only."""
import math

if __package__:
    from .tensor_layout import BatchedLayout
else:
    from tensor_layout import BatchedLayout

_FIELDS = frozenset({'in_channels','out_channels','kernel_size','stride','padding',
                     'output_padding','dilation','groups','bias'})
_SPATIAL = ('kernel_size','stride','padding','output_padding','dilation')


def _integer(value, minimum, maximum, key):
    if type(value) not in (int,float) or not minimum <= value <= maximum or value != int(value):
        raise ValueError(f'ConvTranspose3d {key} must contain integers from {minimum} to {maximum}.')
    return int(value)


def normalize_conv_transpose3d(config):
    if type(config) is not dict or set(config) != _FIELDS:
        raise ValueError('ConvTranspose3d requires exactly channel, kernel, stride, padding, output padding, dilation, groups and bias settings.')
    result = {key:_integer(config[key],1,512,key) for key in ('in_channels','out_channels','groups')}
    for key in _SPATIAL:
        values = config[key]
        if type(values) is not list or len(values) != 3:
            raise ValueError(f'ConvTranspose3d {key} requires exactly [depth,height,width].')
        minimum = 0 if key in ('padding','output_padding') else 1
        result[key] = [_integer(value,minimum,4096,key) for value in values]
    if type(config['bias']) is not bool:
        raise ValueError('ConvTranspose3d bias must be a boolean.')
    result['bias'] = config['bias']
    if result['in_channels'] % result['groups'] or result['out_channels'] % result['groups']:
        raise ValueError('ConvTranspose3d input and output channels must both be divisible by groups.')
    for axis in range(3):
        if not (result['output_padding'][axis] < result['stride'][axis] or
                result['output_padding'][axis] < result['dilation'][axis]):
            raise ValueError('ConvTranspose3d output padding must be smaller than stride or dilation on each axis.')
    count = result['in_channels'] * (result['out_channels']//result['groups']) * math.prod(result['kernel_size']) + int(result['bias'])*result['out_channels']
    if count > 2_000_000:
        raise ValueError('ConvTranspose3d exceeds the two-million parameter budget.')
    return result


def resolve_conv_transpose3d(config, layout):
    c = normalize_conv_transpose3d(config)
    if not isinstance(layout,BatchedLayout) or len(layout.sample) != 4:
        raise ValueError('ConvTranspose3d requires exactly channel, depth, height and width sample axes [C,D,H,W].')
    channels,*spatial = layout.sample
    if channels != c['in_channels'] or any(not 1 <= length <= 4096 for length in spatial):
        raise ValueError('ConvTranspose3d input channels must match and spatial lengths must be from 1 to 4096.')
    if math.prod(layout.sample) > 100_000_000:
        raise ValueError('ConvTranspose3d input sample volume exceeds 100 million elements.')
    uncropped = [(length-1)*c['stride'][axis] + c['dilation'][axis]*(c['kernel_size'][axis]-1)
                 + c['output_padding'][axis] + 1 for axis,length in enumerate(spatial)]
    output = [length-2*c['padding'][axis] for axis,length in enumerate(uncropped)]
    if any(not 1 <= length <= 4096 for length in output):
        raise ValueError('ConvTranspose3d output lengths must be from 1 to 4096; review kernel, stride, dilation, padding and output padding.')
    # Output is a subset of the uncropped extent. Diagnose its bound first;
    # uncropped allocation can independently exceed the budget despite heavy cropping.
    if c['out_channels']*math.prod(output) > 100_000_000:
        raise ValueError('ConvTranspose3d output sample volume exceeds 100 million elements.')
    if c['out_channels']*math.prod(uncropped) > 100_000_000:
        raise ValueError('ConvTranspose3d uncropped output volume exceeds 100 million elements.')
    weight_count = channels * (c['out_channels']//c['groups']) * math.prod(c['kernel_size'])
    work = math.prod(spatial)*weight_count
    if work > 128_000_000:
        raise ValueError('ConvTranspose3d exceeds 128 million multiply-adds per observation.')
    return {'layout':BatchedLayout((c['out_channels'],*output)),
            'parameterCount':weight_count+int(c['bias'])*c['out_channels'],
            'uncroppedOutputShape':uncropped,'multiplyAdds':work}


def capability():
    """Fresh discovery for bounded triple transposed-convolution configuration."""
    return {
        'version': {'major': 1, 'minor': 0, 'patch': 0},
        'adapterID': 'kwip.runner.graph-v2.conv_transpose3d', 'graphSchemaVersions': [2],
        'sampleRank': 4, 'preservesBatch': True, 'graphDTypes': ['float32'],
        'spatialParameterKind': 'triple', 'spatialParameterLength': 3,
        'channelMinimum': 1, 'channelMaximum': 512,
        'lengthMinimum': 1, 'lengthMaximum': 4096,
        'kernelMinimum': 1, 'kernelMaximum': 4096,
        'strideMinimum': 1, 'strideMaximum': 4096,
        'dilationMinimum': 1, 'dilationMaximum': 4096,
        'groupMinimum': 1, 'groupMaximum': 512,
        'paddingMinimum': 0, 'paddingMaximum': 4096,
        'outputPaddingMinimum': 0, 'outputPaddingMaximum': 4096,
        'maximumParameters': 2_000_000, 'maximumMultiplyAdds': 128_000_000,
        'maximumSampleElements': 100_000_000, 'maximumUncroppedSampleElements': 100_000_000,
        'configurationKeys': ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding',
                              'output_padding', 'dilation', 'groups', 'bias'],
        'paddingMode': 'zeros', 'outputPaddingRule': 'lessThanStrideOrDilation',
        'supportsDynamicOutputSize': False, 'biasModes': [False, True],
    }
