"""Bounded learned, logarithmic and parent-mode activation configuration."""
if __package__:
    from .tensor_layout import BatchedLayout
else:
    from tensor_layout import BatchedLayout
MAXIMUM=3.4028234663852886e38
OPERATIONS=frozenset({'prelu','log_softmax','dropout'})

def normalize_remaining_activation(operation,config):
    defaults={'prelu':{'num_parameters':1,'init':.25},'log_softmax':{'dim':-1},'dropout':{'p':.5}}
    if type(operation) is not str or operation not in defaults or type(config) is not dict or set(config)-set(defaults[operation]):raise ValueError('Activation requires only its declared configuration fields.')
    c={**defaults[operation],**config}
    def integer(value,lo,hi):
        if type(value) not in (int,float) or not lo<=value<=hi or value!=int(value):raise ValueError('Activation directions/counts require exact bounded integers, not Boolean values.')
        return int(value)
    def number(value,lo,hi):
        if type(value) not in (int,float) or not lo<=value<=hi:raise ValueError('Activation coefficient must be a finite bounded number, not Boolean.')
        return float(value)
    if operation=='prelu':return {'num_parameters':integer(c['num_parameters'],1,4096),'init':number(c['init'],-MAXIMUM,MAXIMUM)}
    if operation=='log_softmax':
        dim=integer(c['dim'],-8,7)
        if dim==0:raise ValueError('LogSoftmax direction excludes the observation batch.')
        return {'dim':dim}
    return {'p':number(c['p'],0,1)}

def resolve_remaining_activation(operation,config,layout):
    c=normalize_remaining_activation(operation,config)
    if not isinstance(layout,BatchedLayout):raise ValueError('Activation requires a concrete batch-preserving layout.')
    if operation=='prelu' and c['num_parameters'] not in (1,layout.sample[0]):raise ValueError('PReLU slope count must be one or match the first sample direction (channels).')
    if operation=='log_softmax':layout.softmax(c['dim'])
    return {'layout':layout,'config':c,'parameterCount':c['num_parameters'] if operation=='prelu' else 0}

def capability(operation):
    if type(operation) is not str or operation not in OPERATIONS:raise ValueError('Unknown activation.')
    d={'version':{'major':1,'minor':2,'patch':0},'adapterID':'kwip.runner.graph-v2.'+operation,'graphSchemaVersions':[2],'maximumRank':8,'preservesBatch':True,'graphDTypes':['float32'],'maximumTensorElements':100_000_000}
    if operation=='prelu':d.update(parameterCountMinimum=1,parameterCountMaximum=4096,initMinimum=-MAXIMUM,initMaximum=MAXIMUM,channelAxis=1,learnedParameters=True,zeroInputGradient='slope')
    elif operation=='log_softmax':d.update(axisMinimum=-8,axisMaximum=7,excludesBatch=True,outputKind='logarithmicProbabilities')
    else:d.update(probabilityMinimum=0,probabilityMaximum=1,modeSource='parentModule',stochasticTraining=True,evaluationIdentity=True,checkpointGuarantees={'cpu':'epochBoundaryExactCPU','mps':'weightsOnly'})
    return d
