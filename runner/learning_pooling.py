"""Bounded real pooling execution. Expected values come from a separate scalar oracle."""
import copy
import math
import torch
if __package__:
    from .learning_tensor_transforms import compile_graph as compile_path
    from .tensor_layout import apply_layout, BatchedLayout
    from .pooling_adapters_generated import RESOLVERS
    from .graph_semantics import effective_config
else:
    from learning_tensor_transforms import compile_graph as compile_path
    from tensor_layout import apply_layout, BatchedLayout
    from pooling_adapters_generated import RESOLVERS
    from graph_semantics import effective_config


def compile_graph(submission, dimensions, average):
    if dimensions not in (1,2,3) or (average and dimensions!=1) or type(submission) is not dict or set(submission)!={'schemaVersion','kind','graph'} or type(submission['schemaVersion']) is not int or submission['schemaVersion']!=1 or submission['kind']!='poolingGraph':
        raise ValueError('Submit a reviewed poolingGraph, never code or measured answers.')
    surrogate=copy.deepcopy(submission);surrogate['kind']='tensorGraph';graph=surrogate.get('graph')
    if type(graph) is not dict or type(graph.get('nodes')) is not list:raise ValueError('Malformed pooling graph.')
    wanted='avg_pool1d' if average else f'max_pool{dimensions}d';pools={}
    for node in graph['nodes']:
        if type(node) is dict and node.get('type')==wanted:
            pools[node.get('id')]=effective_config(node,2)
            node['type']='reshape';node['config']={'shape':[2]+[6 if average else 4]*dimensions}
    if len(pools)!=1:raise ValueError('Use exactly one connected pooling block of the requested kind.')
    operations,count=compile_path(surrogate,include_ids=True)
    return [(ident,wanted,pools[ident]) if ident in pools else (ident,kind,config) for ident,kind,config in operations],count,wanted


def execute(submission,fixtures,evidence,stopped,parse_tensor,*,dimensions,average=False):
    operations,count,wanted=compile_graph(submission,dimensions,average)
    size=6 if average else 4;extent=4 if average else 3
    def axes(v):return [v] if type(v) is int else v
    for fixture in fixtures:
        if stopped():return
        if set(fixture['inputs'])!={'x','upstream'}:raise ValueError('Pooling fixtures contain input and output influence only.')
        shape,values=parse_tensor(fixture['inputs']['x']);gshape,gvalues=parse_tensor(fixture['inputs']['upstream'])
        for dims,vals in [(shape,values),(gshape,gvalues)]:
            if not all(type(n) is int and n>0 for n in dims) or math.prod(dims)!=len(vals) or not all(math.isfinite(v) for v in vals):raise ValueError('Invalid or nonfinite pooling fixture.')
        if len(shape)!=dimensions+2 or shape[1:]!=[2]+[size]*dimensions or shape[0] not in (1,2) or gshape!=[shape[0],2]+[extent]*dimensions:raise ValueError('Fixture differs from the public pooling shape contract.')
        layout=BatchedLayout(tuple(shape[1:]));largest=len(values)
        for _,kind,config in operations:
            if kind==wanted:
                if axes(config['kernel_size'])!=[3]*dimensions or axes(config['stride'])!=[2]*dimensions or axes(config['padding'])!=[1]*dimensions or not config['ceil_mode']:raise ValueError('Configure the supplied window and include the final partial window.')
                if not average and (not config['return_indices'] or axes(config['dilation'])!=[1]*dimensions):raise ValueError('Expose winning indices and use dilation 1.')
                layout=RESOLVERS[kind](config,layout)['layout']
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
            if kind==wanted:
                kwargs={k:v for k,v in config.items() if k in ('kernel_size','stride','padding','ceil_mode','dilation','return_indices','count_include_pad')}
                result=getattr(torch.nn.functional,'avg_pool1d' if average else f'max_pool{dimensions}d')(output,**kwargs)
                if average:output=result
                else:output,indices=result
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
