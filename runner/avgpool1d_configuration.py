"""Strict average-pooling configuration; shared window geometry, distinct averaging semantics."""
if __package__:
    from .maxpool1d_configuration import normalize_maxpool1d, resolve_maxpool1d
else:
    from maxpool1d_configuration import normalize_maxpool1d, resolve_maxpool1d

_FIELDS = {'kernel_size','stride','padding','ceil_mode','count_include_pad'}


def normalize_avgpool1d(config):
    if type(config) is not dict or set(config) != _FIELDS:
        raise ValueError('AvgPool1d requires exactly kernel, stride, padding, ceiling mode and count include pad.')
    if type(config['count_include_pad']) is not bool:
        raise ValueError('AvgPool1d count include pad must be Boolean.')
    try:
        shared = normalize_maxpool1d({k:v for k,v in config.items() if k!='count_include_pad'} | {'dilation':1,'return_indices':False})
    except ValueError as error:
        raise ValueError(str(error).replace('MaxPool1d','AvgPool1d')) from error
    return {k:shared[k] for k in ('kernel_size','stride','padding','ceil_mode')} | {'count_include_pad':config['count_include_pad']}


def resolve_avgpool1d(config, layout):
    config=normalize_avgpool1d(config)
    try:
        return resolve_maxpool1d({k:v for k,v in config.items() if k!='count_include_pad'} | {'dilation':1,'return_indices':False}, layout)
    except ValueError as error:
        raise ValueError(str(error).replace('MaxPool1d','AvgPool1d')) from error


def capability():
    return {
        'version':{'major':1,'minor':0,'patch':0},'adapterID':'kwip.runner.graph-v2.avg_pool1d',
        'graphSchemaVersions':[2],'sampleRank':2,'preservesBatch':True,'graphDTypes':['float32'],
        'channelMinimum':1,'channelMaximum':512,'lengthMinimum':1,'lengthMaximum':4096,
        'kernelMinimum':1,'kernelMaximum':4096,'strideMinimum':1,'strideMaximum':4096,
        'paddingMinimum':0,'paddingMaximum':2048,'paddingKernelDivisor':2,
        'nullStrideFollowsKernel':True,'ceilModes':[False,True],'countIncludePadModes':[False,True],
        'outputDTypes':{'out':'float32'},
    }
