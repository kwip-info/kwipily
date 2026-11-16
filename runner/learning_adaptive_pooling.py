"""Strict bounded float64 lesson execution; expected results belong to scalar references."""
import copy
import math
import torch
if __package__:
    from .learning_tensor_transforms import compile_graph as compile_path
    from .tensor_layout import apply_layout, BatchedLayout
    from .avgpool2d_configuration import normalize_avgpool2d, resolve_avgpool2d
    from .adaptive_pooling_configuration import normalize_adaptive_pooling, resolve_adaptive_pooling, torch_output_size
else:
    from learning_tensor_transforms import compile_graph as compile_path
    from tensor_layout import apply_layout, BatchedLayout
    from avgpool2d_configuration import normalize_avgpool2d, resolve_avgpool2d
    from adaptive_pooling_configuration import normalize_adaptive_pooling, resolve_adaptive_pooling, torch_output_size

_OPERATIONS=frozenset({'avg_pool2d','adaptive_avg_pool1d','adaptive_avg_pool2d','adaptive_max_pool2d'})


def sample_shape(operation):
    return [2,6,6] if operation=='avg_pool2d' else [2,5] if operation=='adaptive_avg_pool1d' else [2,5,4]


def output_shape(operation):
    return [2,4,3] if operation=='avg_pool2d' else [2,3] if operation=='adaptive_avg_pool1d' else [2,3,4]


def compile_graph(submission,operation):
    if operation not in _OPERATIONS or type(submission) is not dict or set(submission)!={'schemaVersion','kind','graph'} or type(submission['schemaVersion']) is not int or submission['schemaVersion']!=1 or submission['kind']!='adaptivePoolingGraph':
        raise ValueError('Submit a reviewed adaptivePoolingGraph, never code or measured answers.')
    surrogate=copy.deepcopy(submission);surrogate['kind']='tensorGraph';graph=surrogate.get('graph')
    if type(graph) is not dict or type(graph.get('nodes')) is not list:raise ValueError('Malformed adaptive pooling graph.')
    pools={}
    for node in graph['nodes']:
        if type(node) is dict and node.get('type')==operation:
            ident=node.get('id')
            if type(ident) is not str:raise ValueError('Pooling component requires a string identity.')
            pools[ident]=normalize_avgpool2d(node.get('config')) if operation=='avg_pool2d' else normalize_adaptive_pooling(operation,node.get('config'))
            node['type']='reshape';node['config']={'shape':sample_shape(operation)}
    if len(pools)!=1:raise ValueError('Use exactly one connected pooling block of the requested kind.')
    operations,count=compile_path(surrogate,include_ids=True)
    return [(ident,operation,pools[ident]) if ident in pools else (ident,kind,config) for ident,kind,config in operations],count


def execute(submission,fixtures,evidence,stopped,parse_tensor,*,operation):
    operations,count=compile_graph(submission,operation)
    for fixture in fixtures:
        if stopped():return
        if type(fixture) is not dict or type(fixture.get('inputs')) is not dict or set(fixture['inputs'])!={'x','upstream'}:raise ValueError('Pooling fixtures contain input and output influence only.')
        shape,values=parse_tensor(fixture['inputs']['x']);gshape,gvalues=parse_tensor(fixture['inputs']['upstream'])
        for dims,vals in [(shape,values),(gshape,gvalues)]:
            if not dims or not all(type(n) is int and n>0 for n in dims) or math.prod(dims)!=len(vals) or not all(type(v) in (int,float) and math.isfinite(v) for v in vals):raise ValueError('Invalid or nonfinite pooling fixture.')
        if len(shape)!=len(sample_shape(operation))+1 or shape[1:]!=sample_shape(operation) or shape[0] not in (1,2) or gshape!=[shape[0],*output_shape(operation)]:raise ValueError('Fixture differs from the public pooling shape contract.')
        layout=BatchedLayout(tuple(shape[1:]));largest=len(values)
        for _,kind,config in operations:
            if kind==operation:
                layout=(resolve_avgpool2d(config,layout) if operation=='avg_pool2d' else resolve_adaptive_pooling(operation,config,layout))['layout']
            elif kind=='reshape':layout=layout.reshape(config['shape'])
            elif kind=='transpose':layout=layout.transpose(config['dim0'],config['dim1'])
            else:layout=layout.flatten(config['start_dim'],config['end_dim'])
            largest=max(largest,shape[0]*math.prod(layout.sample))
        if [shape[0],*layout.sample]!=gshape:raise ValueError('Output shape differs from the task influence.')
        evidence['maximumTensorElements']=max(evidence['maximumTensorElements'],len(values)*(count+3)+len(gvalues)*4+largest)
        if stopped():return
        x=torch.tensor(values,dtype=torch.float64).reshape(shape).requires_grad_(True);output=x;indices=None
        for _,kind,config in operations:
            if stopped():return
            if kind==operation:
                if operation=='avg_pool2d':output=torch.nn.functional.avg_pool2d(output,**config)
                elif operation=='adaptive_max_pool2d':
                    result=torch.nn.functional.adaptive_max_pool2d(output,torch_output_size(operation,config),return_indices=config['return_indices'])
                    if config['return_indices']:output,indices=result
                    else:output=result
                else:output=getattr(torch.nn.functional,operation)(output,torch_output_size(operation,config))
            else:
                output=apply_layout(output,kind,config)
                if indices is not None:indices=apply_layout(indices,kind,config)
        upstream=torch.tensor(gvalues,dtype=torch.float64).reshape(gshape)
        if stopped():return
        gradient=torch.autograd.grad((output*upstream).sum(),x)[0]
        def measured(t):
            if not torch.isfinite(t).all():raise ValueError('Nonfinite measured pooling result.')
            return {'shape':list(t.shape),'values':t.detach().reshape(-1).tolist()}
        tensors={'output':measured(output),'inputGradient':measured(gradient)}
        if indices is not None:tensors['indices']=measured(indices)
        if stopped():return
        evidence['fixtures'].append({'fixtureID':fixture['id'],'tensors':tensors,'partitions':{}})
