"""Bounded two-fixture graph execution for tensor composition learning; no submitted answers/code."""
import math
import torch
if __package__:
    from .tensor_layout import BatchedLayout, apply_layout
    from .tensor_composition_configuration import normalize_tensor_composition, resolve_tensor_composition
    from .graph_semantics import effective_config
else:
    from tensor_layout import BatchedLayout, apply_layout
    from tensor_composition_configuration import normalize_tensor_composition, resolve_tensor_composition
    from graph_semantics import effective_config


def compile_graph(submission,operation):
    if (type(submission) is not dict or set(submission)!={'schemaVersion','kind','graph'}
            or type(submission['schemaVersion']) is not int or submission['schemaVersion']!=1
            or submission['kind']!='tensorCompositionGraph'):
        raise ValueError('Submit a tensorCompositionGraph, never authored answers or code.')
    graph=submission['graph']
    if (type(graph) is not dict or set(graph)-{'schemaVersion','name','nodes','edges'}
            or type(graph.get('schemaVersion')) is not int or graph['schemaVersion']!=2):
        raise ValueError('Tensor composition lesson graph requires schema 2.')
    nodes,edges=graph.get('nodes'),graph.get('edges')
    if type(nodes) is not list or not (3 if operation=='permute' else 4)<=len(nodes)<=32 or type(edges) is not list or len(edges)>64:
        raise ValueError('Tensor composition lessons require 4..32 nodes and at most 64 connections.')
    by_id={};configs={};bindings={};outputs=[]
    for node in nodes:
        if type(node) is not dict or not {'id','type','config'}<=set(node) or set(node)-{'id','type','config','name','x','y'}:
            raise ValueError('Malformed tensor composition lesson component.')
        ident,kind,config=node['id'],node['type'],node['config']
        if type(ident) is not str or not 1<=len(ident)<=128 or ident in by_id or type(kind) is not str or type(config) is not dict:
            raise ValueError('Invalid tensor composition component identity or configuration.')
        if kind=='input':
            if set(config)!={'fixture'} or config['fixture'] not in (('left',) if operation=='permute' else ('left','right')) or config['fixture'] in bindings:
                raise ValueError('Bind exactly one input to each left/right fixture.')
            bindings[config['fixture']]=ident;configs[ident]=config
        elif kind=='output':
            if config:raise ValueError('Output has no configuration.')
            outputs.append(ident);configs[ident]={}
        elif kind==operation:
            configs[ident]=normalize_tensor_composition(operation,config)
        elif kind in ('reshape','transpose','flatten'):
            configs[ident]=effective_config(node,2)
        else:raise ValueError('Unsupported tensor composition lesson operation.')
        by_id[ident]=node
    if set(bindings)!=({'left'} if operation=='permute' else {'left','right'}) or len(outputs)!=1:
        raise ValueError('Use left/right fixture inputs and one output.')
    incoming={i:{} for i in by_id};successors={i:[] for i in by_id};edge_ids=set()
    for edge in edges:
        if type(edge) is not dict or set(edge)!={'id','source','sourcePort','target','targetPort'}:
            raise ValueError('Malformed tensor composition connection.')
        ident,a,b,port=edge['id'],edge['source'],edge['target'],edge['targetPort']
        if (type(ident) is not str or not ident or ident in edge_ids or type(a) is not str or type(b) is not str
                or a not in by_id or b not in by_id or edge['sourcePort']!='out' or by_id[a]['type']=='output'):
            raise ValueError('Invalid tensor composition connection identity or source.')
        kind=by_id[b]['type'];ports=[] if kind=='input' else ['a','b'] if kind in ('concat','add','stack') else ['in']
        if port not in ports or port in incoming[b]:raise ValueError('Connect each declared input exactly once.')
        incoming[b][port]=a;successors[a].append(b);edge_ids.add(ident)
    for i,node in by_id.items():
        wanted=0 if node['type']=='input' else 2 if node['type'] in ('concat','add','stack') else 1
        if len(incoming[i])!=wanted:raise ValueError('Incomplete tensor composition graph.')
        if node['type']!='output' and not successors[i]:raise ValueError('Unused tensor composition component.')
    pending={i:len(ports) for i,ports in incoming.items()};queue=[i for i,n in pending.items() if n==0];order=[]
    while queue:
        i=queue.pop(0);order.append(i)
        for child in successors[i]:
            pending[child]-=1
            if pending[child]==0:queue.append(child)
    if len(order)!=len(nodes):raise ValueError('Tensor composition lesson graph contains a cycle.')
    if sum(n['type']==operation for n in nodes)!=1:raise ValueError('Use exactly one connected requested tensor composition component.')
    return by_id,configs,bindings,outputs[0],incoming,order


_TARGET_SHAPES={'concat':((2,3),(1,3)),'add':((2,3),(3,)),'stack':((2,3),(2,3)),'permute':((2,3,4),None)}


def execute(submission,fixtures,evidence,stopped,parse_tensor,*,operation):
    if operation not in _TARGET_SHAPES:raise ValueError('Unknown tensor composition lesson.')
    nodes,configs,bindings,output_id,incoming,order=compile_graph(submission,operation)
    expected_roles={'left','upstream'} if operation=='permute' else {'left','right','upstream'}
    for fixture in fixtures:
        if stopped():return
        if set(fixture['inputs'])!=expected_roles:raise ValueError('Use only the declared operand and output influence fixtures.')
        parsed={key:parse_tensor(value) for key,value in fixture['inputs'].items()}
        for shape,values in parsed.values():
            if not shape or len(shape)>8 or any(type(n) is not int or not 1<=n<=4096 for n in shape) or math.prod(shape)!=len(values) or len(values)>4096 or not all(type(v) in (int,float) and math.isfinite(v) for v in values):raise ValueError('Invalid or nonfinite tensor composition fixture.')
        rows=parsed['left'][0][0]
        if not 1<=rows<=3 or tuple(parsed['left'][0][1:])!=_TARGET_SHAPES[operation][0]:raise ValueError('Left fixture differs from the public task shape.')
        if operation!='permute' and (parsed['right'][0][0]!=rows or tuple(parsed['right'][0][1:])!=_TARGET_SHAPES[operation][1]):raise ValueError('Right fixture differs from the public task shape or observation count.')
        layouts={bindings[key]:BatchedLayout(tuple(parsed[key][0][1:])) for key in bindings};total=sum(len(values) for shape,values in parsed.values())
        for ident in order:
            kind=nodes[ident]['type'];ports=incoming[ident];config=configs[ident]
            if kind=='input':continue
            if kind==operation:
                left=layouts[ports['in' if operation=='permute' else 'a']]
                right=None if operation=='permute' else layouts[ports['b']]
                layout=resolve_tensor_composition(operation,config,left,right)
                if isinstance(layout,dict):layout=layout['layout']
            else:
                layout=layouts[ports['in']]
                if kind=='reshape':layout=layout.reshape(config['shape'])
                elif kind=='transpose':layout=layout.transpose(config['dim0'],config['dim1'])
                elif kind=='flatten':layout=layout.flatten(config['start_dim'],config['end_dim'])
            size=rows*math.prod(layout.sample)
            if size>4096:raise ValueError('Tensor composition lesson exceeds bounded output allocation.')
            total+=size;layouts[ident]=layout
        if [rows,*layouts[output_id].sample]!=parsed['upstream'][0]:raise ValueError('Measured output shape differs from the task output influence shape.')
        evidence['maximumTensorElements']=max(evidence['maximumTensorElements'],3*total)
        if stopped():return
        values={bindings[key]:torch.tensor(parsed[key][1],dtype=torch.float64).reshape(parsed[key][0]).requires_grad_(True) for key in bindings}
        operands=tuple(values[bindings[key]] for key in ('left','right') if key in bindings)
        for ident in order:
            if stopped():return
            kind=nodes[ident]['type'];ports=incoming[ident];config=configs[ident]
            if kind=='input':continue
            if kind=='concat':values[ident]=torch.cat((values[ports['a']],values[ports['b']]),dim=config['dim'])
            elif kind=='stack':values[ident]=torch.stack((values[ports['a']],values[ports['b']]),dim=config['dim'])
            elif kind=='permute':values[ident]=values[ports['in']].permute(config['dims'])
            elif kind=='add':
                a,b=values[ports['a']],values[ports['b']]
                rank=max(a.ndim,b.ndim)
                a=a.reshape((rows,)+(1,)*(rank-a.ndim)+tuple(a.shape[1:]));b=b.reshape((rows,)+(1,)*(rank-b.ndim)+tuple(b.shape[1:]))
                values[ident]=torch.add(a,b,alpha=config['alpha'])
            elif kind=='output':values[ident]=values[ports['in']]
            else:values[ident]=apply_layout(values[ports['in']],kind,config)
        result=values[output_id];upstream=torch.tensor(parsed['upstream'][1],dtype=torch.float64).reshape(parsed['upstream'][0])
        gradients=torch.autograd.grad((result*upstream).sum(),operands)
        if stopped():return
        def measured(value):
            if not torch.isfinite(value).all():raise ValueError('Nonfinite tensor composition measurements.')
            return {'shape':list(value.shape),'values':value.detach().reshape(-1).tolist()}
        tensors={'output':measured(result),'leftGradient':measured(gradients[0])}
        if len(gradients)==2:tensors['rightGradient']=measured(gradients[1])
        evidence['fixtures'].append({'fixtureID':fixture['id'],'tensors':tensors,'partitions':{}})
