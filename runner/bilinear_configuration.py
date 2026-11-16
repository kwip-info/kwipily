"""Canonical two-input learned interaction configuration; no implicit broadcasting."""
FIELDS = {'in1_features', 'in2_features', 'out_features', 'bias'}
MAX_PARAMETERS = 2_000_000


def normalize_bilinear(config):
    if type(config) is not dict or set(config) != FIELDS:
        raise ValueError('Bilinear requires in1_features, in2_features, out_features and bias.')
    result = {}
    for name in ('in1_features', 'in2_features', 'out_features'):
        value = config[name]
        if type(value) not in (int, float) or not 1 <= value <= 512 or value != int(value):
            raise ValueError(f'Bilinear {name} must be an integer from 1 to 512.')
        result[name] = int(value)
    if type(config['bias']) is not bool:
        raise ValueError('Bilinear bias must be a boolean.')
    result['bias'] = config['bias']
    count = result['out_features'] * (result['in1_features'] * result['in2_features'] + int(result['bias']))
    if count > MAX_PARAMETERS:
        raise ValueError('Bilinear exceeds the two-million parameter budget; reduce feature widths.')
    return result


def capability():
    return {
        'version': {'major': 1, 'minor': 0, 'patch': 0},
        'adapterID': 'kwip.runner.graph-v2.bilinear',
        'graphSchemaVersions': [2], 'maximumRank': 8, 'preservesBatch': True,
        'minimumSampleRank': 1, 'sampleBroadcasting': False,
        'maximumSampleElements': 100_000_000, 'maximumParameters': MAX_PARAMETERS,
        'featureMinimum': 1, 'featureMaximum': 512,
        'biasModes': [False, True], 'graphDTypes': ['float32'],
    }
