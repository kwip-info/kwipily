"""Strict average-pooling settings; shared spatial windows, distinct divisors."""
if __package__:
    from .maxpool2d_configuration import normalize_maxpool2d, resolve_maxpool2d, capability as max_capability
else:
    from maxpool2d_configuration import normalize_maxpool2d, resolve_maxpool2d, capability as max_capability

_FIELDS=frozenset({'kernel_size','stride','padding','ceil_mode','count_include_pad','divisor_override'})
DIVISOR_MAXIMUM=2_147_483_647


def normalize_avgpool2d(config):
    if type(config) is not dict or set(config)!=_FIELDS:
        raise ValueError('AvgPool2d requires exactly kernel, stride, padding, ceiling mode, count include pad and divisor override.')
    if type(config['count_include_pad']) is not bool:
        raise ValueError('AvgPool2d count include pad must be Boolean.')
    divisor=config['divisor_override']
    if divisor is not None:
        if type(divisor) not in (int,float) or not -DIVISOR_MAXIMUM<=divisor<=DIVISOR_MAXIMUM or divisor==0 or divisor!=int(divisor):
            raise ValueError('AvgPool2d divisor override must be null or a nonzero integer from -2147483647 to 2147483647.')
        divisor=int(divisor)
    shared={k:v for k,v in config.items() if k not in ('count_include_pad','divisor_override')}|{'dilation':[1,1],'return_indices':False}
    try:shared=normalize_maxpool2d(shared)
    except ValueError as error:raise ValueError(str(error).replace('MaxPool2d','AvgPool2d').replace('MaxPool1d','AvgPool2d')) from error
    return {k:shared[k] for k in ('kernel_size','stride','padding','ceil_mode')}|{'count_include_pad':config['count_include_pad'],'divisor_override':divisor}


def resolve_avgpool2d(config,layout):
    config=normalize_avgpool2d(config)
    shared={k:v for k,v in config.items() if k not in ('count_include_pad','divisor_override')}|{'dilation':[1,1],'return_indices':False}
    try:return resolve_maxpool2d(shared,layout)
    except ValueError as error:raise ValueError(str(error).replace('MaxPool2d','AvgPool2d').replace('MaxPool1d','AvgPool2d')) from error


def capability():
    descriptor=max_capability()
    for key in ('dilationMinimum','dilationMaximum','returnIndicesModes','indicesMeaning'):descriptor.pop(key)
    descriptor.update(adapterID='kwip.runner.graph-v2.avg_pool2d',countIncludePadModes=[False,True],
                      divisorOverrideMinimum=-DIVISOR_MAXIMUM,divisorOverrideMaximum=DIVISOR_MAXIMUM,
                      divisorOverrideExcludesZero=True,nullDivisorFollowsWindow=True,outputDTypes={'out':'float32'})
    return descriptor
