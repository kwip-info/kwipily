"""Bounded transposed-convolution geometry; explicit scalar output padding, zeros only."""
if __package__:
    from .tensor_layout import BatchedLayout
else:
    from tensor_layout import BatchedLayout

_FIELDS = frozenset({'in_channels','out_channels','kernel_size','stride','padding',
                     'output_padding','dilation','groups','bias'})


def normalize_conv_transpose1d(config):
    if type(config) is not dict or set(config) != _FIELDS:
        raise ValueError('ConvTranspose1d requires exactly channel, kernel, stride, padding, output padding, dilation, groups and bias settings.')
    result = {}
    for key in _FIELDS - {'bias'}:
        value = config[key]
        minimum = 0 if key in ('padding','output_padding') else 1
        maximum = 512 if key in ('in_channels','out_channels','groups') else 4096
        if type(value) not in (int,float) or not minimum <= value <= maximum or value != int(value):
            raise ValueError(f'ConvTranspose1d {key} must be an integer from {minimum} to {maximum}.')
        result[key] = int(value)
    if type(config['bias']) is not bool:
        raise ValueError('ConvTranspose1d bias must be a boolean.')
    result['bias'] = config['bias']
    if result['in_channels'] % result['groups'] or result['out_channels'] % result['groups']:
        raise ValueError('ConvTranspose1d input and output channels must both be divisible by groups.')
    if not (result['output_padding'] < result['stride'] or result['output_padding'] < result['dilation']):
        raise ValueError('ConvTranspose1d output padding must be smaller than stride or dilation.')
    count = result['in_channels'] * (result['out_channels']//result['groups']) * result['kernel_size'] + int(result['bias'])*result['out_channels']
    if count > 2_000_000:
        raise ValueError('ConvTranspose1d exceeds the two-million parameter budget.')
    return result


def resolve_conv_transpose1d(config, layout):
    c = normalize_conv_transpose1d(config)
    if not isinstance(layout,BatchedLayout) or len(layout.sample) != 2:
        raise ValueError('ConvTranspose1d requires exactly channel and length sample axes [C,L].')
    channels,length = layout.sample
    if channels != c['in_channels'] or not 1 <= length <= 4096:
        raise ValueError('ConvTranspose1d input channels must match and length must be from 1 to 4096.')
    uncropped = (length-1)*c['stride'] + c['dilation']*(c['kernel_size']-1) + c['output_padding'] + 1
    output_length = uncropped - 2*c['padding']
    if not 1 <= output_length <= 4096:
        raise ValueError('ConvTranspose1d output length must be from 1 to 4096; review kernel, stride, dilation, padding and output padding.')
    # Defense in depth: currently implied by Cout<=512, output<=4096 and padding<=4096.
    if c['out_channels']*uncropped > 100_000_000:
        raise ValueError('ConvTranspose1d uncropped output volume exceeds 100 million elements.')
    weight_count = channels * (c['out_channels']//c['groups']) * c['kernel_size']
    work = length * weight_count
    if work > 128_000_000:
        raise ValueError('ConvTranspose1d exceeds 128 million multiply-adds per observation.')
    return {'layout':BatchedLayout((c['out_channels'],output_length)),
            'parameterCount':weight_count+int(c['bias'])*c['out_channels'],
            'uncroppedOutputLength':uncropped,'multiplyAdds':work}


def capability():
    """Fresh discovery for bounded scalar transposed-convolution configuration."""
    return {
        'version': {'major': 1, 'minor': 0, 'patch': 0},
        'adapterID': 'kwip.runner.graph-v2.conv_transpose1d', 'graphSchemaVersions': [2],
        'sampleRank': 2, 'preservesBatch': True, 'graphDTypes': ['float32'],
        'spatialParameterKind': 'scalar',
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
