"""Allocation-free, dense Conv1d configuration and per-observation shape bounds."""
if __package__:
    from .tensor_layout import BatchedLayout
    from .convolution_geometry import resolve_axis
else:
    from tensor_layout import BatchedLayout
    from convolution_geometry import resolve_axis

_FIELDS = frozenset({'in_channels','out_channels','kernel_size','stride','dilation','groups',
                     'bias','padding_policy','padding','padding_mode'})


def normalize_conv1d(config):
    if type(config) is not dict or set(config) != _FIELDS:
        raise ValueError('Conv1d requires exactly its channel, kernel, stride, dilation, group, bias and padding settings.')
    result = {}
    for key in ('in_channels','out_channels','kernel_size','stride','dilation','groups','padding'):
        value = config[key]
        minimum = 0 if key == 'padding' else 1
        maximum = 512 if key in ('in_channels','out_channels','groups') else 4096
        if type(value) not in (int,float) or not minimum <= value <= maximum or value != int(value):
            raise ValueError(f'Conv1d {key} must be an integer from {minimum} to {maximum}.')
        result[key] = int(value)
    if type(config['bias']) is not bool:
        raise ValueError('Conv1d bias must be a boolean.')
    result['bias'] = config['bias']
    for key, choices in (('padding_policy',('explicit','valid','same')),
                         ('padding_mode',('zeros','reflect','replicate','circular'))):
        value = config[key]
        if type(value) is not str or value not in choices:
            raise ValueError(f'Conv1d {key} must be one of: '+', '.join(choices)+'.')
        result[key] = value
    if result['padding_policy'] != 'explicit' and result['padding'] != 0:
        raise ValueError('Conv1d valid/same policies require explicit padding to be zero.')
    if result['padding_policy'] == 'same' and result['stride'] != 1:
        raise ValueError('Conv1d same padding requires stride 1.')
    if result['in_channels'] % result['groups'] or result['out_channels'] % result['groups']:
        raise ValueError('Conv1d input and output channels must both be divisible by groups.')
    count = result['out_channels'] * ((result['in_channels'] // result['groups']) * result['kernel_size'] + int(result['bias']))
    if count > 2_000_000:
        raise ValueError('Conv1d exceeds the two-million parameter budget.')
    return result


def resolve_conv1d(config, layout):
    config = normalize_conv1d(config)
    if not isinstance(layout, BatchedLayout) or len(layout.sample) != 2:
        raise ValueError('Conv1d requires exactly channel and length sample axes [C,L].')
    channels, length = layout.sample
    if channels != config['in_channels'] or not 1 <= length <= 4096:
        raise ValueError('Conv1d input channels must match and length must be from 1 to 4096.')
    output_length, left, right = resolve_axis(
        length, config['kernel_size'], config['stride'], config['dilation'], config['padding'],
        config['padding_policy'], config['padding_mode'], operation='Conv1d')
    fan_in = (channels // config['groups']) * config['kernel_size']
    parameters = config['out_channels'] * (fan_in + int(config['bias']))
    work = output_length * config['out_channels'] * fan_in
    if work > 128_000_000:
        raise ValueError('Conv1d exceeds 128 million multiply-adds per observation.')
    return {'layout': BatchedLayout((config['out_channels'], output_length)),
            'parameterCount': parameters, 'paddingLeft': left, 'paddingRight': right,
            'multiplyAdds': work}


def capability():
    """Fresh exact discovery of the reviewed dense channel/length adapter."""
    return {
        'version': {'major': 1, 'minor': 0, 'patch': 0},
        'adapterID': 'kwip.runner.graph-v2.conv1d', 'graphSchemaVersions': [2],
        'sampleRank': 2, 'preservesBatch': True, 'graphDTypes': ['float32'],
        'channelMinimum': 1, 'channelMaximum': 512,
        'lengthMinimum': 1, 'lengthMaximum': 4096,
        'kernelMinimum': 1, 'kernelMaximum': 4096,
        'strideMinimum': 1, 'strideMaximum': 4096,
        'dilationMinimum': 1, 'dilationMaximum': 4096,
        'groupMinimum': 1, 'groupMaximum': 512,
        'paddingMinimum': 0, 'paddingMaximum': 4096,
        'maximumParameters': 2_000_000, 'maximumMultiplyAdds': 128_000_000,
        'paddingPolicies': ['explicit', 'valid', 'same'],
        'paddingModes': ['zeros', 'reflect', 'replicate', 'circular'],
        'biasModes': [False, True],
    }
