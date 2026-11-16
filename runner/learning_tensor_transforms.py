"""Fixture-shaped, parameter-free transform graphs for learning assessments."""
import torch
if __package__:
    from .tensor_layout import apply_layout, normalize_configuration, BatchedLayout
    from .graph_semantics import effective_config
else:
    from tensor_layout import apply_layout, normalize_configuration, BatchedLayout
    from graph_semantics import effective_config


def compile_graph(submission, *, allowed_operations=frozenset({'reshape','transpose','flatten'}), include_ids=False):
    if type(submission) is not dict or set(submission)!={'schemaVersion','kind','graph'} or type(submission['schemaVersion']) is not int or submission['schemaVersion']!=1 or submission['kind']!='tensorGraph':
        raise ValueError('Submit a tensorGraph, never authored results or code.')
    graph=submission['graph']
    if type(graph) is not dict or set(graph)-{'schemaVersion','name','nodes','edges'} or type(graph.get('schemaVersion')) is not int or graph['schemaVersion']!=2:
        raise ValueError('Tensor lesson graph requires schema 2.')
    nodes,edges=graph.get('nodes'),graph.get('edges')
    if type(nodes) is not list or not 2<=len(nodes)<=32 or type(edges) is not list or len(edges)!=len(nodes)-1:
        raise ValueError('Connect 2..32 components in one input-to-output transform path.')
    by_id={};configs={}
    for node in nodes:
        if type(node) is not dict or not {'id','type','config'}<=set(node) or set(node)-{'id','type','config','name','x','y'}:
            raise ValueError('Malformed tensor component.')
        ident=node['id'];kind=node['type'];config=node['config']
        if type(ident) is not str or not 1<=len(ident)<=128 or ident in by_id or type(kind) is not str:
            raise ValueError('Tensor component identities must be unique bounded strings.')
        if kind in ('input','output'):
            if config!={} or type(config) is not dict:raise ValueError('Lesson boundaries use fixture shapes and have no configuration.')
            configs[ident]={}
        else:
            if kind not in allowed_operations:raise ValueError('This operation is not allowed in the requested lesson.')
            configs[ident]=effective_config(node,2) if kind in ('softmax','layer_norm','embedding') else normalize_configuration(kind,config)
        by_id[ident]=node
    inputs=[i for i,n in by_id.items() if n['type']=='input'];outputs=[i for i,n in by_id.items() if n['type']=='output']
    if len(inputs)!=1 or len(outputs)!=1:raise ValueError('Exactly one fixture Input and one Output are required.')
    successors={};predecessors={};edge_ids=set()
    for edge in edges:
        if type(edge) is not dict or set(edge)!={'id','source','sourcePort','target','targetPort'}:raise ValueError('Malformed tensor connection.')
        ident=edge['id'];source=edge['source'];target=edge['target']
        if type(ident) is not str or not ident or ident in edge_ids or type(source) is not str or type(target) is not str or source not in by_id or target not in by_id:
            raise ValueError('Invalid tensor connection identity.')
        if edge['sourcePort']!='out' or edge['targetPort']!='in' or source in successors or target in predecessors or source==outputs[0] or target==inputs[0]:
            raise ValueError('Connect each tensor output to one declared input along the lesson path.')
        edge_ids.add(ident);successors[source]=target;predecessors[target]=source
    order=[];current=inputs[0]
    while current not in order:
        order.append(current)
        if current==outputs[0]:break
        if current not in successors:raise ValueError('Incomplete transform path.')
        current=successors[current]
    if len(order)!=len(nodes) or order[-1]!=outputs[0]:raise ValueError('Cycle or disconnected tensor components.')
    operations=[(i,by_id[i]['type'],configs[i]) if include_ids else (by_id[i]['type'],configs[i]) for i in order[1:-1]]
    return operations,len(nodes)


def execute(submission,fixtures,evidence,stopped,parse_tensor, *, allowed_operations=frozenset({'reshape','transpose','flatten'})):
    operations,node_count=compile_graph(submission,allowed_operations=allowed_operations)
    for fixture in fixtures:
        if stopped():break
        if set(fixture['inputs'])!={'x','upstream'}:raise ValueError('Tensor fixtures require input and upstream gradient.')
        shape,values=parse_tensor(fixture['inputs']['x']);gshape,gvalues=parse_tensor(fixture['inputs']['upstream'])
        if len(shape)<2:raise ValueError('Preserve a leading batch and at least one sample axis.')
        evidence['maximumTensorElements']=max(evidence['maximumTensorElements'],len(values)*(node_count+3)+len(gvalues))
        if stopped():break
        x=torch.tensor(values,dtype=torch.float64).reshape(shape).requires_grad_(True)
        output=x
        for kind,config in operations:
            if stopped():return
            if kind=='softmax':
                BatchedLayout(tuple(output.shape[1:])).softmax(config['dim'])
                output=torch.softmax(output,dim=config['dim'])
            else:output=apply_layout(output,kind,config)
        if list(output.shape)!=gshape:raise ValueError('Output shape differs from the upstream gradient fixture.')
        upstream=torch.tensor(gvalues,dtype=torch.float64).reshape(gshape)
        gradient,=torch.autograd.grad((output*upstream).sum(),x)
        if stopped():break
        def measured(t):return {'shape':list(t.shape),'values':t.detach().reshape(-1).tolist()}
        evidence['fixtures'].append({'fixtureID':fixture['id'],'tensors':{'output':measured(output),'inputGradient':measured(gradient)},'partitions':{}})
