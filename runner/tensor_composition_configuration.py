"""Strict, allocation-free composition settings with historical concat/add defaults."""
if __package__:
    from .tensor_layout import BatchedLayout
else:
    from tensor_layout import BatchedLayout

OPERATIONS=frozenset({'concat','add','stack','permute'})
ALPHA_MAXIMUM=3.4028234663852886e38


def normalize_tensor_composition(operation,config,dtype='float32'):
    if type(operation) is not str or operation not in OPERATIONS or dtype not in ('float32','int64'):
        raise ValueError('Tensor composition requires a reviewed operation and float32 or int64 values.')
    if type(config) is not dict:raise ValueError(f'{operation} requires declared configuration fields.')
    if not config and operation in ('concat','add'):
        config={'dim':-1} if operation=='concat' else {'alpha':1.0,'sample_broadcast':False}
    fields={'dim'} if operation in ('concat','stack') else {'alpha','sample_broadcast'} if operation=='add' else {'dims'}
    if set(config)!=fields:raise ValueError(f'{operation} requires exactly {sorted(fields)}.')
    def axis(value):
        if type(value) not in (int,float) or not -8<=value<=7 or value!=int(value):raise ValueError('Directions must be exact integers from -8 to 7, never booleans or fractions.')
        return int(value)
    if operation in ('concat','stack'):
        dim=axis(config['dim'])
        if dim==0:raise ValueError('Direction zero is the observation batch; choose a sample direction.')
        return {'dim':dim}
    if operation=='add':
        alpha=config['alpha'];broadcast=config['sample_broadcast']
        if type(alpha) not in (int,float) or not -ALPHA_MAXIMUM<=alpha<=ALPHA_MAXIMUM:
            raise ValueError('Second source multiplier must be a finite signed float32 number.')
        if type(broadcast) is not bool:raise ValueError('Sample broadcasting must be Boolean.')
        if dtype=='int64' and (alpha!=1 or broadcast):raise ValueError('Integer Add supports only multiplier 1 and equal sample shapes.')
        return {'alpha':float(alpha),'sample_broadcast':broadcast}
    raw=config['dims']
    if type(raw) is not list or not 2<=len(raw)<=8:raise ValueError('Direction order requires a complete list of 2–8 tensor axes, observations first.')
    dims=[axis(value) for value in raw];rank=len(dims)
    if any(not -rank<=value<rank for value in dims):raise ValueError('Direction order contains an axis outside its complete rank.')
    normalized=[value%rank for value in dims]
    if normalized[0]!=0 or set(normalized)!=set(range(rank)):raise ValueError('Direction order must use every axis exactly once and keep observations first.')
    return {'dims':dims}


def resolve_tensor_composition(operation,config,left,right=None,dtype='float32'):
    config=normalize_tensor_composition(operation,config,dtype=dtype)
    if not isinstance(left,BatchedLayout) or (operation!='permute' and not isinstance(right,BatchedLayout)):
        raise ValueError('Composition inputs require concrete batched sample layouts.')
    if operation=='permute':
        if right is not None:raise ValueError('Permute has one input only.')
        return left.permute(config['dims'])
    if operation=='concat':return left.concat(right,config['dim'])
    if operation=='stack':return left.stack(right,config['dim'])
    return left.add(right,broadcast=config['sample_broadcast'])


def capability(operation):
    if operation not in OPERATIONS:raise ValueError('Unknown tensor composition operation.')
    result={'version':{'major':1,'minor':0,'patch':0},'adapterID':'kwip.runner.graph-v2.'+operation,
            'graphSchemaVersions':[1,2] if operation in ('concat','add') else [2],
            'maximumRank':8,'preservesBatch':True,'graphDTypes':['float32','int64'],
            'maximumTensorElements':100_000_000,'inputCount':1 if operation=='permute' else 2}
    if operation in ('concat','stack'):
        result.update(axisMinimum=-8,axisMaximum=7,excludesBatch=True,inputOrder=['a','b'])
        if operation=='stack':result.update(maximumInputRank=7,newAxisSize=2)
    elif operation=='add':
        result.update(alphaMinimum=-ALPHA_MAXIMUM,alphaMaximum=ALPHA_MAXIMUM,sampleBroadcastModes=[False,True],
                      broadcastBatch=False,integerAlpha=1,integerBroadcast=False)
    else:result.update(axisOrderMinimumItems=2,axisOrderMaximumItems=8,batchAxisFirst=True)
    return result
