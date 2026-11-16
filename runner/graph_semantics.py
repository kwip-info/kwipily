"""Versioned configuration validation and legacy-compatible semantic identity."""
import math
if __package__:
    from .moe import normalize as normalize_moe, scientific_config as moe_scientific_config
    from .model_output_configuration import validate_model_output_config
    from .attention_configuration import normalize_attention, OPERATIONS as ATTENTION_OPERATIONS
    from .recurrent_configuration import normalize_recurrent
    from .stateful_normalization_configuration import normalize_stateful_normalization
    from .spatial_patch_configuration import normalize_spatial_patch
    from .tensor_routing_configuration import normalize_tensor_routing
    from .remaining_activation_configuration import normalize_remaining_activation
    from .configurable_activation_configuration import normalize_configuration as normalize_activation
    from .relu_configuration import normalize_relu
    from .tanh_configuration import normalize_tanh
    from .sigmoid_configuration import normalize_sigmoid
    from .avgpool2d_configuration import normalize_avgpool2d
    from .adaptive_pooling_configuration import normalize_adaptive_pooling
    from .avgpool1d_configuration import normalize_avgpool1d
    from .maxpool1d_configuration import normalize_maxpool1d
    from .maxpool3d_configuration import normalize_maxpool3d
    from .maxpool2d_configuration import normalize_maxpool2d
    from .tensor_layout import normalize_configuration, BatchedLayout
    from .tensor_composition_configuration import normalize_tensor_composition
    from .embedding_configuration import normalize_embedding
    from .embedding_bag_configuration import normalize_embedding_bag
    from .bag_source import normalize_bag_source
    from .lazy_linear_configuration import normalize_lazy_linear
    from .bilinear_configuration import normalize_bilinear
    from .conv1d_configuration import normalize_conv1d
    from .conv2d_configuration import normalize_conv2d
    from .depthwise_conv2d_configuration import normalize_depthwise_conv2d
    from .conv3d_configuration import normalize_conv3d
    from .conv_transpose1d_configuration import normalize_conv_transpose1d
    from .conv_transpose2d_configuration import normalize_conv_transpose2d
    from .conv_transpose3d_configuration import normalize_conv_transpose3d
else:
    from moe import normalize as normalize_moe, scientific_config as moe_scientific_config
    from model_output_configuration import validate_model_output_config
    from attention_configuration import normalize_attention, OPERATIONS as ATTENTION_OPERATIONS
    from recurrent_configuration import normalize_recurrent
    from stateful_normalization_configuration import normalize_stateful_normalization
    from spatial_patch_configuration import normalize_spatial_patch
    from tensor_routing_configuration import normalize_tensor_routing
    from remaining_activation_configuration import normalize_remaining_activation
    from configurable_activation_configuration import normalize_configuration as normalize_activation
    from relu_configuration import normalize_relu
    from tanh_configuration import normalize_tanh
    from sigmoid_configuration import normalize_sigmoid
    from avgpool2d_configuration import normalize_avgpool2d
    from adaptive_pooling_configuration import normalize_adaptive_pooling
    from avgpool1d_configuration import normalize_avgpool1d
    from maxpool1d_configuration import normalize_maxpool1d
    from maxpool3d_configuration import normalize_maxpool3d
    from maxpool2d_configuration import normalize_maxpool2d
    from tensor_layout import normalize_configuration, BatchedLayout
    from tensor_composition_configuration import normalize_tensor_composition
    from embedding_configuration import normalize_embedding
    from embedding_bag_configuration import normalize_embedding_bag
    from bag_source import normalize_bag_source
    from lazy_linear_configuration import normalize_lazy_linear
    from bilinear_configuration import normalize_bilinear
    from conv1d_configuration import normalize_conv1d
    from conv2d_configuration import normalize_conv2d
    from depthwise_conv2d_configuration import normalize_depthwise_conv2d
    from conv3d_configuration import normalize_conv3d
    from conv_transpose1d_configuration import normalize_conv_transpose1d
    from conv_transpose2d_configuration import normalize_conv_transpose2d
    from conv_transpose3d_configuration import normalize_conv_transpose3d

CONVOLUTION_NORMALIZERS={"conv1d":normalize_conv1d,"conv2d":normalize_conv2d,"conv3d":normalize_conv3d,"depthwise_conv2d":normalize_depthwise_conv2d}

TRANSPOSE_CONVOLUTION_NORMALIZERS={"conv_transpose1d":normalize_conv_transpose1d,"conv_transpose2d":normalize_conv_transpose2d,"conv_transpose3d":normalize_conv_transpose3d}

TRANSFORMS={"reshape", "transpose", "flatten"}

FIELDS={'input':{'features'},'linear':{'in_features','out_features'},'relu':set(),'tanh':set(),'gelu':{'approximate_tanh'},'softmax':{'dim'},'concat':set(),'add':set(),'output':set()}

def effective_config(node,version):
    if type(version) is not int or version not in (1,2):raise ValueError('Unsupported graph schema; expected version 1 or 2.')
    kind=node.get('type');config=node.get('config')
    if kind=='mixture_experts':
        if version!=2:raise ValueError('Mixture of experts requires graph schema2.')
        return normalize_moe(config)
    if kind=='input' and type(config) is dict and set(config)=={'shape'}:
        if version!=2:raise ValueError('Shaped floating input requires graph schema2.')
        dimensions=config['shape']
        if type(dimensions) is not list or not 1<=len(dimensions)<=7 or any(type(n) is not int or n<=0 for n in dimensions):raise ValueError('Floating input shape requires positive integer sample dimensions.')
        BatchedLayout(tuple(dimensions))
        return {'shape':list(dimensions)}
    if kind=='output':
        settings=validate_model_output_config(config)
        return settings if config else {}
    if kind in ('bag_input','embedding_bag'):
        if version!=2:raise ValueError('Bag input and EmbeddingBag require graph schema2; migrate this graph before connecting bag sources.')
        return normalize_bag_source(config) if kind=='bag_input' else normalize_embedding_bag(config)
    if kind in ATTENTION_OPERATIONS:
        if version!=2:raise ValueError('Attention components require graph schema version2.')
        return normalize_attention(kind,config)
    if kind in ('rnn','lstm','gru','rnn_cell','lstm_cell','gru_cell'):
        if version!=2:raise ValueError('Recurrent components require graph schema version2.')
        return normalize_recurrent(kind,config)
    if kind in ('batch_norm1d','batch_norm2d','instance_norm1d','instance_norm2d','group_norm'):
        if version!=2:raise ValueError(f'{kind} requires graph schema version2.')
        return normalize_stateful_normalization(kind,config)
    if kind in ('unfold','fold'):
        if version!=2:raise ValueError(f'{kind} requires graph schema version 2.')
        return normalize_spatial_patch(kind,config)
    if kind in ('split','chunk','gather','einsum'):
        if version!=2:raise ValueError(f'{kind} requires graph schema version 2.')
        return normalize_tensor_routing(kind,config)
    if kind in ('prelu','log_softmax','dropout'):
        if version!=2:raise ValueError(f'{kind} requires graph schema version 2.')
        return normalize_remaining_activation(kind,config)
    if kind in ('concat','add','stack','permute'):
        if kind in ('stack','permute') and version!=2:raise ValueError(f'{kind} requires graph schema version 2.')
        return normalize_tensor_composition(kind,config)
    if kind in ('silu','leaky_relu','elu'):
        if version!=2:raise ValueError(f'{kind}: requires graph schema version 2.')
        return normalize_activation(kind,config)
    if kind=='relu':return normalize_relu(config)
    if kind=='tanh':return normalize_tanh(config)
    if kind=='sigmoid':
        if version!=2:raise ValueError('Sigmoid requires graph schema version 2.')
        return normalize_sigmoid(config)
    if kind in ('avg_pool2d','adaptive_avg_pool1d','adaptive_avg_pool2d','adaptive_max_pool2d'):
        if version!=2:raise ValueError(f'{kind}: requires graph schema version 2.')
        return normalize_avgpool2d(config) if kind=='avg_pool2d' else normalize_adaptive_pooling(kind,config)
    if kind in ('avg_pool1d','max_pool1d','max_pool2d','max_pool3d'):
        if version!=2:raise ValueError('Max pooling requires graph schema version 2.')
        return {'avg_pool1d':normalize_avgpool1d,'max_pool1d':normalize_maxpool1d,'max_pool2d':normalize_maxpool2d,'max_pool3d':normalize_maxpool3d}[kind](config)
    if kind in TRANSPOSE_CONVOLUTION_NORMALIZERS:
        if version!=2:raise ValueError(f'{kind.title()} requires graph schema version 2.')
        return TRANSPOSE_CONVOLUTION_NORMALIZERS[kind](config)
    if kind in CONVOLUTION_NORMALIZERS:
        if version!=2:raise ValueError(f'{kind.title()} requires graph schema version 2.')
        return CONVOLUTION_NORMALIZERS[kind](config)
    if kind=='lazy_linear':
        if version!=2:raise ValueError('LazyLinear requires graph schema version 2.')
        return normalize_lazy_linear(config)
    if kind=='bilinear':
        if version!=2:raise ValueError('Bilinear requires graph schema version 2.')
        return normalize_bilinear(config)
    if kind=='matmul':
        if version!=2 or type(config) is not dict or config:
            raise ValueError('Matrix multiplication requires schema 2 and no configuration fields; reshape or transpose with explicit nodes.')
        return {}
    if kind=='index_input':
        if version!=2 or type(config) is not dict or set(config)!={'shape'}:
            raise ValueError('Index input requires schema 2 and an explicit sample shape.')
        shape=config['shape']
        if type(shape) is not list:raise ValueError('Index input shape must be a list.')
        canonical=[]
        for dimension in shape:
            if type(dimension) not in (int,float) or not 1<=dimension<=100_000_000 or dimension!=int(dimension):
                raise ValueError('Index input dimensions must be positive bounded integers.')
            canonical.append(int(dimension))
        BatchedLayout(tuple(canonical))
        return {'shape':canonical}
    if kind=='embedding':
        if version!=2:raise ValueError('Embedding requires graph schema version 2.')
        return normalize_embedding(config)
    if kind=='rms_norm':
        if version!=2 or type(config) is not dict or set(config)!={'normalized_shape','eps','elementwise_affine'}:
            raise ValueError('RMS normalization requires schema2, trailing shape, explicit epsilon and learned scale only.')
        normalized=effective_config({**node,'type':'layer_norm','config':{**config,'bias':False}},version)
        normalized.pop('bias')
        return normalized
    if kind=='layer_norm':
        if version!=2:raise ValueError('Layer normalization requires graph schema version 2.')
        if not isinstance(config,dict) or set(config)!={'normalized_shape','eps','elementwise_affine','bias'}:
            raise ValueError('Layer normalization requires normalized_shape, eps, elementwise_affine and bias.')
        shape=config['normalized_shape']
        if type(shape) is not list or not 1<=len(shape)<=7:
            raise ValueError('Layer normalization requires 1..7 trailing sample dimensions.')
        canonical=[];count=1
        for dimension in shape:
            if type(dimension) not in (int,float) or not 1<=dimension<=100_000_000//count or dimension!=int(dimension):
                raise ValueError('Normalized dimensions must be positive integers with at most 100M elements.')
            canonical.append(int(dimension));count*=int(dimension)
        epsilon=config['eps']
        # The graph runner executes float32 on CPU/MPS. Exclude subnormals whose
        # handling varies by device, and values that overflow that dtype.
        if type(epsilon) not in (int,float) or not 1.1754943508222875e-38<=epsilon<=3.4028234663852886e38:
            raise ValueError('Layer normalization epsilon must be a positive normal float32 value (1.1754943508222875e-38 to 3.4028234663852886e38).')
        if type(config['elementwise_affine']) is not bool or type(config['bias']) is not bool:
            raise ValueError('Layer normalization affine and bias options must be booleans.')
        return {**config,'normalized_shape':canonical,'eps':float(epsilon)}
    if kind in TRANSFORMS:
        if version!=2:raise ValueError('Tensor transforms require graph schema version 2.')
        return normalize_configuration(kind,config)
    if kind not in FIELDS or not isinstance(config,dict):raise ValueError('Unsupported component or malformed configuration.')
    if kind=='softmax':
        if version!=2:raise ValueError('Softmax requires graph schema version 2.')
        if set(config)!={'dim'}:raise ValueError('Softmax requires exactly a dim configuration.')
        axis=config['dim']
        if type(axis) not in (int,float) or not -8<=axis<=7 or axis!=int(axis) or axis==0:
            raise ValueError('Softmax axis must be an integer sample axis from -8 to 7, excluding zero.')
        return {'dim':int(axis)}
    if kind=='gelu':
        if version!=2:raise ValueError('GELU requires graph schema version 2.')
        if set(config)!={'approximate_tanh'} or type(config['approximate_tanh']) is not bool:raise ValueError('GELU requires exactly a boolean approximate_tanh configuration.')
        return dict(config)
    keys=FIELDS[kind] | ({'bias'} if kind=='linear' and version==2 else set())
    if set(config)!=keys:raise ValueError(f'{node.get("id",kind)}: configuration fields must exactly match {sorted(keys)}.')
    for key in FIELDS[kind]:
        value=config[key]
        if type(value) not in (int,float) or not math.isfinite(value) or int(value)!=value or not 1<=value<=512:raise ValueError(f'{key} must be an integer from 1 to 512.')
    if kind=='linear' and version==2 and type(config['bias']) is not bool:raise ValueError('Linear bias must be a boolean.')
    result=dict(config)
    for key in FIELDS[kind]:result[key]=int(result[key])
    if kind=='linear':result.setdefault('bias',True)
    return result

def semantic_graph(graph):
    nodes=[]
    for node in graph['nodes']:
        config=effective_config(node,graph['schemaVersion'])
        if node['type']=='mixture_experts':config=moe_scientific_config(config)
        if node['type']=='linear' and config['bias'] is True:del config['bias']
        if node['type']=='concat' and config['dim']==-1:del config['dim']
        if node['type']=='add':
            if config['alpha']==1:del config['alpha']
            if config['sample_broadcast'] is False:del config['sample_broadcast']
        nodes.append({'id':node['id'],'type':node['type'],'config':config})
    result={'schemaVersion':1,'nodes':sorted(nodes,key=lambda n:n['id']),'edges':sorted([{k:e[k] for k in ('source','sourcePort','target','targetPort')} for e in graph['edges']],key=lambda e:(e['source'],e['sourcePort'],e['target'],e['targetPort']))}
    if 'compositionPlan' in graph:result['compositionPlan']=graph['compositionPlan']
    if 'namedInputs' in graph:result['namedInputs']=dict(graph['namedInputs'])
    return result
