"""Authored depthwise specialization of the validated grouped Conv2d contract."""
if __package__:
    from .conv2d_configuration import normalize_conv2d, resolve_conv2d
else:
    from conv2d_configuration import normalize_conv2d, resolve_conv2d

_FIELDS = frozenset({'in_channels','channel_multiplier','kernel_size','stride','dilation',
                     'bias','padding_policy','padding','padding_mode'})


def conv2d_config(config):
    """Return fresh canonical grouped settings without mutating the authored document."""
    if type(config) is not dict or set(config) != _FIELDS:
        raise ValueError('DepthwiseConv2d requires exactly input channels, channel multiplier, kernel, stride, dilation, bias and padding settings; output channels and groups are derived.')
    counts = {}
    for key in ('in_channels','channel_multiplier'):
        value = config[key]
        if type(value) not in (int,float) or not 1 <= value <= 512 or value != int(value):
            raise ValueError(f'DepthwiseConv2d {key} must be an integer from 1 to 512.')
        counts[key] = int(value)
    output_channels = counts['in_channels'] * counts['channel_multiplier']
    if output_channels > 512:
        raise ValueError('DepthwiseConv2d input channels times channel multiplier must not exceed 512 output channels.')
    derived = {key:value for key,value in config.items() if key != 'channel_multiplier'}
    derived.update(in_channels=counts['in_channels'],out_channels=output_channels,groups=counts['in_channels'])
    return normalize_conv2d(derived)


def normalize_depthwise_conv2d(config):
    derived = conv2d_config(config)
    result = {key:value for key,value in derived.items() if key not in ('out_channels','groups')}
    result['channel_multiplier'] = derived['out_channels']//derived['in_channels']
    return result


def resolve_depthwise_conv2d(config, layout):
    return resolve_conv2d(conv2d_config(config),layout)


def capability():
    """Fresh exact discovery for explicit [height,width] spatial configuration."""
    return {
        'version': {'major': 1, 'minor': 0, 'patch': 0},
        'adapterID': 'kwip.runner.graph-v2.depthwise_conv2d', 'graphSchemaVersions': [2],
        'sampleRank': 3, 'preservesBatch': True, 'graphDTypes': ['float32'],
        'spatialParameterLength': 2,
        'channelMinimum': 1, 'channelMaximum': 512,
        'lengthMinimum': 1, 'lengthMaximum': 4096,
        'kernelMinimum': 1, 'kernelMaximum': 4096,
        'strideMinimum': 1, 'strideMaximum': 4096,
        'dilationMinimum': 1, 'dilationMaximum': 4096,
        'channelMultiplierMinimum': 1, 'channelMultiplierMaximum': 512,
        'outputChannelMaximum': 512, 'groupsRule': 'inputChannels',
        'outputChannelsRule': 'inputChannelsTimesMultiplier',
        'configurationKeys': ['in_channels', 'channel_multiplier', 'kernel_size', 'stride', 'dilation',
                              'bias', 'padding_policy', 'padding', 'padding_mode'],
        'paddingMinimum': 0, 'paddingMaximum': 4096,
        'maximumParameters': 2_000_000, 'maximumMultiplyAdds': 128_000_000,
        'maximumSampleElements': 100_000_000, 'maximumPaddedSampleElements': 100_000_000,
        'paddingPolicies': ['explicit', 'valid', 'same'],
        'paddingModes': ['zeros', 'reflect', 'replicate', 'circular'],
        'biasModes': [False, True],
    }
