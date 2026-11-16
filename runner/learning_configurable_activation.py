"""Real bounded activation measurements, independent of scalar lesson references."""
import copy
import math
import torch
if __package__:
    from .learning_tensor_transforms import compile_graph as compile_path
    from .tensor_layout import apply_layout,BatchedLayout
    from .graph_semantics import effective_config
else:
    from learning_tensor_transforms import compile_graph as compile_path
    from tensor_layout import apply_layout,BatchedLayout
    from graph_semantics import effective_config


def compile_graph(submission,component):
    if component not in ('silu','leaky_relu','elu') or type(submission) is not dict or set(submission)!={'schemaVersion','kind','graph'} or type(submission['schemaVersion']) is not int or submission['schemaVersion']!=1 or submission['kind']!='configurableActivationGraph':raise ValueError('Submit a reviewed configurableActivationGraph, never code or measured answers.')
    surrogate=copy.deepcopy(submission);surrogate['kind']='tensorGraph';graph=surrogate.get('graph')
    if type(graph) is not dict or type(graph.get('nodes')) is not list:raise ValueError('Malformed activation graph.')
    activations={}
    for node in graph['nodes']:
        if type(node) is dict and node.get('type')==component:
            activations[node.get('id')]=effective_config(node,2)
            node['type']='reshape';node['config']={'shape':[2,3]}
    if len(activations)!=1:raise ValueError('Use exactly one connected activation of the requested kind.')
    operations,count=compile_path(surrogate,include_ids=True)
    return [(ident,component,activations[ident]) if ident in activations else (ident,kind,config) for ident,kind,config in operations],count


def execute(submission,fixtures,evidence,stopped,parse_tensor,*,component):
    operations,count=compile_graph(submission,component)
    for fixture in fixtures:
        if stopped():return
        if set(fixture['inputs'])!={'x','upstream'}:raise ValueError('Activation fixtures require input and output influence only.')
        shape,values=parse_tensor(fixture['inputs']['x']);gshape,gvalues=parse_tensor(fixture['inputs']['upstream'])
        for dims,vals in [(shape,values),(gshape,gvalues)]:
            if not all(type(n) is int and n>0 for n in dims) or math.prod(dims)!=len(vals) or not all(math.isfinite(v) for v in vals):raise ValueError('Invalid or nonfinite activation fixture.')
        if len(shape)!=3 or shape[0] not in (1,2) or shape[1:]!=[2,3] or gshape!=shape:raise ValueError('Fixture differs from the public activation array contract.')
        layout=BatchedLayout(tuple(shape[1:]));largest=len(values)
        for _,kind,config in operations:
            if kind==component:pass
            elif kind=='reshape':layout=layout.reshape(config['shape'])
            elif kind=='transpose':layout=layout.transpose(config['dim0'],config['dim1'])
            else:layout=layout.flatten(config['start_dim'],config['end_dim'])
            largest=max(largest,shape[0]*math.prod(layout.sample))
        if [shape[0],*layout.sample]!=gshape:raise ValueError('Output array differs from task influence; preserve the input layout.')
        evidence['maximumTensorElements']=max(evidence['maximumTensorElements'],len(values)*(count+3)+len(gvalues)*3+largest)
        if stopped():return
        x=torch.tensor(values,dtype=torch.float64).reshape(shape).requires_grad_(True);output=x
        for _,kind,config in operations:
            if stopped():return
            if kind==component:
                if kind=='silu':output=torch.nn.functional.silu(output,inplace=False)
                elif kind=='leaky_relu':output=torch.nn.functional.leaky_relu(output,negative_slope=config['negative_slope'],inplace=False)
                else:output=torch.nn.functional.elu(output,alpha=config['alpha'],inplace=False)
            else:output=apply_layout(output,kind,config)
        if not torch.isfinite(output).all():raise ValueError('Nonfinite activation output.')
        if stopped():return
        upstream=torch.tensor(gvalues,dtype=torch.float64).reshape(gshape)
        gradient=torch.autograd.grad((output*upstream).sum(),x)[0]
        def measured(t):
            if not torch.isfinite(t).all():raise ValueError('Nonfinite activation influence.')
            return {'shape':list(t.shape),'values':t.detach().reshape(-1).tolist()}
        tensors={'output':measured(output),'inputGradient':measured(gradient)}
        if stopped():return
        evidence['fixtures'].append({'fixtureID':fixture['id'],'tensors':tensors,'partitions':{}})
