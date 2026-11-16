"""Actual bounded shared-window execution; supplied kernels and independent grading."""
import copy
import math
import torch
if __package__:
    from .learning_tensor_transforms import compile_graph as compile_path
    from .tensor_layout import apply_layout, BatchedLayout
    from .conv1d_configuration import resolve_conv1d
    from .conv2d_configuration import resolve_conv2d
    from .conv3d_configuration import resolve_conv3d
    from .conv_transpose1d_configuration import resolve_conv_transpose1d
    from .conv_transpose2d_configuration import resolve_conv_transpose2d
    from .conv_transpose3d_configuration import resolve_conv_transpose3d
    from .graph_semantics import effective_config
else:
    from learning_tensor_transforms import compile_graph as compile_path
    from tensor_layout import apply_layout, BatchedLayout
    from conv1d_configuration import resolve_conv1d
    from conv2d_configuration import resolve_conv2d
    from conv3d_configuration import resolve_conv3d
    from conv_transpose1d_configuration import resolve_conv_transpose1d
    from conv_transpose2d_configuration import resolve_conv_transpose2d
    from conv_transpose3d_configuration import resolve_conv_transpose3d
    from graph_semantics import effective_config


def compile_graph(submission,dimensions,*,transposed=False):
    if dimensions not in (1,2,3) or type(submission) is not dict or set(submission)!={'schemaVersion','kind','graph'} or type(submission['schemaVersion']) is not int or submission['schemaVersion']!=1 or submission['kind']!='convolutionGraph':raise ValueError('Submit a reviewed convolutionGraph, never code or measured answers.')
    surrogate=copy.deepcopy(submission);surrogate['kind']='tensorGraph';graph=surrogate.get('graph')
    if type(graph) is not dict or type(graph.get('nodes')) is not list:raise ValueError('Malformed shared-window graph.')
    convolutions={};wanted=f'conv_transpose{dimensions}d' if transposed else f'conv{dimensions}d'
    for node in graph['nodes']:
        if type(node) is dict and node.get('type')==wanted:
            convolutions[node.get('id')]=effective_config(node,2)
            node['type']='reshape';node['config']={'shape':[2]+[4]*dimensions}
    if len(convolutions)!=1:raise ValueError('Use exactly one connected convolution of the requested dimension.')
    operations,count=compile_path(surrogate,include_ids=True)
    return [(ident,wanted,convolutions[ident]) if ident in convolutions else (ident,kind,config) for ident,kind,config in operations],count


def execute(submission,fixtures,evidence,stopped,parse_tensor,*,dimensions):
    return _execute(submission,fixtures,evidence,stopped,parse_tensor,dimensions=dimensions,transposed=False)


def execute_transposed(submission,fixtures,evidence,stopped,parse_tensor,*,dimensions):
    return _execute(submission,fixtures,evidence,stopped,parse_tensor,dimensions=dimensions,transposed=True)


def _execute(submission,fixtures,evidence,stopped,parse_tensor,*,dimensions,transposed):
    operations,node_count=compile_graph(submission,dimensions,transposed=transposed)
    functional=getattr(torch.nn.functional,f'conv_transpose{dimensions}d' if transposed else f'conv{dimensions}d')
    def axes(value):return [value] if type(value) is int else value
    for fixture in fixtures:
        if stopped():return
        if set(fixture['inputs'])!={'x','weight','bias','upstream'}:raise ValueError('Shared-window fixtures require input, kernel, bias and output influence.')
        parsed={key:parse_tensor(value) for key,value in fixture['inputs'].items()}
        for shape,values in parsed.values():
            if not all(type(n) is int and n>0 for n in shape) or math.prod(shape)!=len(values) or not all(math.isfinite(v) for v in values):raise ValueError('Convolution fixture has invalid shapes or nonfinite values.')
        shape,values=parsed['x'];wshape,wvalues=parsed['weight'];bshape,bvalues=parsed['bias'];gshape,gvalues=parsed['upstream']
        input_extent=3 if transposed else 4;kernel_extent=3 if transposed else 2;output_extent=8 if transposed else 3
        if shape[1:]!=[2]+[input_extent]*dimensions or len(shape)!=dimensions+2 or not 1<=shape[0]<=2 or wshape!=[2,2]+[kernel_extent]*dimensions or bshape!=[2] or gshape!=[shape[0],2]+[output_extent]*dimensions:raise ValueError('Convolution fixtures differ from the public shared-window contract.')
        layout=BatchedLayout(tuple(shape[1:]));largest=len(values)
        for _,kind,config in operations:
            if kind.startswith('conv'):
                if config['in_channels']!=2 or config['out_channels']!=2 or config['groups']!=1 or not config['bias'] or axes(config['kernel_size'])!=[kernel_extent]*dimensions:raise ValueError('Configure two channels, shared kernel sizes2, groups1 and enabled bias to use the supplied kernel.')
                resolvers={1:resolve_conv_transpose1d,2:resolve_conv_transpose2d,3:resolve_conv_transpose3d} if transposed else {1:resolve_conv1d,2:resolve_conv2d,3:resolve_conv3d}
                layout=resolvers[dimensions](config,layout)['layout']
            elif kind=='reshape':layout=layout.reshape(config['shape'])
            elif kind=='transpose':layout=layout.transpose(config['dim0'],config['dim1'])
            else:layout=layout.flatten(config['start_dim'],config['end_dim'])
            largest=max(largest,shape[0]*math.prod(layout.sample))
        if [shape[0],*layout.sample]!=gshape:raise ValueError('Output shape differs from the task influence; check stride, dilation, padding and layout.')
        evidence['maximumTensorElements']=max(evidence['maximumTensorElements'],sum(len(v) for _,v in parsed.values())*3+largest*node_count)
        if stopped():return
        x=torch.tensor(values,dtype=torch.float64).reshape(shape).requires_grad_(True)
        weight=torch.tensor(wvalues,dtype=torch.float64).reshape(wshape).requires_grad_(True)
        bias=torch.tensor(bvalues,dtype=torch.float64).reshape(bshape).requires_grad_(True)
        output=x
        for _,kind,config in operations:
            if stopped():return
            if kind.startswith('conv'):
                if config['in_channels']!=2 or config['out_channels']!=2 or config['groups']!=1 or not config['bias'] or axes(config['kernel_size'])!=[kernel_extent]*dimensions:raise ValueError('Configure two channels, shared kernel sizes2, groups1 and enabled bias to use the supplied kernel.')
                if transposed:
                    output=functional(output,weight,bias,stride=config['stride'],padding=config['padding'],output_padding=config['output_padding'],dilation=config['dilation'],groups=config['groups'])
                else:
                    pad=config['padding'] if config['padding_policy']=='explicit' else config['padding_policy']
                    if config['padding_mode']!='zeros' and any(axes(config['padding'])):raise ValueError('This task requires zero boundary padding; configure padding amount0.')
                    output=functional(output,weight,bias,stride=config['stride'],padding=pad,dilation=config['dilation'],groups=config['groups'])
            else:output=apply_layout(output,kind,config)
        if list(output.shape)!=gshape:raise ValueError('Output shape differs from the task influence; check stride, dilation, padding and layout.')
        if not torch.isfinite(output).all():raise ValueError('Nonfinite convolution output.')
        upstream=torch.tensor(gvalues,dtype=torch.float64).reshape(gshape)
        gradients=torch.autograd.grad((output*upstream).sum(),[x,weight,bias])
        def measured(tensor):
            if not torch.isfinite(tensor).all():raise ValueError('Nonfinite convolution influence.')
            return {'shape':list(tensor.shape),'values':tensor.detach().reshape(-1).tolist()}
        tensors={'output':measured(output),'inputGradient':measured(gradients[0]),'kernelGradient':measured(gradients[1]),'biasGradient':measured(gradients[2])}
        if stopped():return
        evidence['fixtures'].append({'fixtureID':fixture['id'],'tensors':tensors,'partitions':{}})
