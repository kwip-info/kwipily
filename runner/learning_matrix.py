"""Bounded two-fixture graph execution for matrix learning; no submitted answers/code."""
import math
import torch
if __package__:
    from .tensor_layout import BatchedLayout, apply_layout
    from .matrix_multiply import matrix_multiply
    from .graph_semantics import effective_config
else:
    from tensor_layout import BatchedLayout, apply_layout
    from matrix_multiply import matrix_multiply
    from graph_semantics import effective_config


def compile_graph(submission):
    if (type(submission) is not dict or set(submission)!={'schemaVersion','kind','graph'}
            or type(submission['schemaVersion']) is not int or submission['schemaVersion']!=1
            or submission['kind']!='matrixGraph'):
        raise ValueError('Submit a matrixGraph, never authored answers or code.')
    graph=submission['graph']
    if (type(graph) is not dict or set(graph)-{'schemaVersion','name','nodes','edges'}
            or type(graph.get('schemaVersion')) is not int or graph['schemaVersion']!=2):
        raise ValueError('Matrix lesson graph requires schema 2.')
    nodes,edges=graph.get('nodes'),graph.get('edges')
    if type(nodes) is not list or not 4<=len(nodes)<=32 or type(edges) is not list or len(edges)>64:
        raise ValueError('Matrix lessons require 4..32 nodes and at most 64 connections.')
    by_id={};configs={};bindings={};outputs=[]
    for node in nodes:
        if type(node) is not dict or not {'id','type','config'}<=set(node) or set(node)-{'id','type','config','name','x','y'}:
            raise ValueError('Malformed matrix lesson component.')
        ident,kind,config=node['id'],node['type'],node['config']
        if type(ident) is not str or not 1<=len(ident)<=128 or ident in by_id or type(kind) is not str or type(config) is not dict:
            raise ValueError('Invalid matrix component identity or configuration.')
        if kind=='input':
            if set(config)!={'fixture'} or config['fixture'] not in ('left','right') or config['fixture'] in bindings:
                raise ValueError('Bind exactly one input to each left/right fixture.')
            bindings[config['fixture']]=ident;configs[ident]=config
        elif kind=='output':
            if config:raise ValueError('Output has no configuration.')
            outputs.append(ident);configs[ident]={}
        elif kind in ('matmul','reshape','transpose','flatten'):
            configs[ident]=effective_config(node,2)
        else:raise ValueError('Unsupported matrix lesson operation.')
        by_id[ident]=node
    if set(bindings)!={'left','right'} or len(outputs)!=1:
        raise ValueError('Use left/right fixture inputs and one output.')
    incoming={i:{} for i in by_id};successors={i:[] for i in by_id};edge_ids=set()
    for edge in edges:
        if type(edge) is not dict or set(edge)!={'id','source','sourcePort','target','targetPort'}:
            raise ValueError('Malformed matrix connection.')
        ident,a,b,port=edge['id'],edge['source'],edge['target'],edge['targetPort']
        if (type(ident) is not str or not ident or ident in edge_ids or type(a) is not str or type(b) is not str
                or a not in by_id or b not in by_id or edge['sourcePort']!='out' or by_id[a]['type']=='output'):
            raise ValueError('Invalid matrix connection identity or source.')
        kind=by_id[b]['type'];ports=[] if kind=='input' else ['a','b'] if kind=='matmul' else ['in']
        if port not in ports or port in incoming[b]:raise ValueError('Connect each declared input exactly once.')
        incoming[b][port]=a;successors[a].append(b);edge_ids.add(ident)
    for i,node in by_id.items():
        wanted=0 if node['type']=='input' else 2 if node['type']=='matmul' else 1
        if len(incoming[i])!=wanted:raise ValueError('Incomplete matrix graph.')
        if node['type']!='output' and not successors[i]:raise ValueError('Unused matrix component.')
    pending={i:len(ports) for i,ports in incoming.items()};queue=[i for i,n in pending.items() if n==0];order=[]
    while queue:
        i=queue.pop(0);order.append(i)
        for child in successors[i]:
            pending[child]-=1
            if pending[child]==0:queue.append(child)
    if len(order)!=len(nodes):raise ValueError('Matrix lesson graph contains a cycle.')
    if not any(n['type']=='matmul' for n in nodes):raise ValueError('Use a matrix multiplication component.')
    return by_id,configs,bindings,outputs[0],incoming,order


def execute(submission,fixtures,evidence,stopped,parse_tensor):
    nodes,configs,bindings,output_id,incoming,order=compile_graph(submission)
    for fixture in fixtures:
        if stopped():return
        if set(fixture['inputs'])!={'left','right','upstream'}:raise ValueError('Matrix fixtures require left, right and upstream tensors.')
        parsed={key:parse_tensor(value) for key,value in fixture['inputs'].items()}
        ashape,av=parsed['left'];bshape,bv=parsed['right'];gshape,gv=parsed['upstream']
        if len(ashape)<3 or len(bshape)<3 or ashape[0]!=bshape[0]:raise ValueError('Fixture matrices require matching independent observations.')
        rows=ashape[0];layouts={bindings['left']:BatchedLayout(tuple(ashape[1:])),bindings['right']:BatchedLayout(tuple(bshape[1:]))}
        total=len(av)+len(bv)+len(gv);work=0
        for i in order:
            kind=nodes[i]['type'];ports=incoming[i];c=configs[i]
            if kind=='input':continue
            if kind=='matmul':
                layout=layouts[ports['a']].matmul(layouts[ports['b']])
                work+=rows*math.prod(layout.sample)*layouts[ports['a']].sample[-1]
            else:
                layout=layouts[ports['in']]
                if kind=='reshape':layout=layout.reshape(c['shape'])
                elif kind=='transpose':layout=layout.transpose(c['dim0'],c['dim1'])
                elif kind=='flatten':layout=layout.flatten(c['start_dim'],c['end_dim'])
            size=rows*math.prod(layout.sample)
            if size>4096 or work>100_000:raise ValueError('Matrix lesson exceeds bounded tensor or scalar-product work.')
            total+=size;layouts[i]=layout
        if [rows,*layouts[output_id].sample]!=gshape:raise ValueError('Matrix output shape differs from upstream.')
        evidence['maximumTensorElements']=max(evidence['maximumTensorElements'],3*total)
        if stopped():return
        values={bindings[key]:torch.tensor(parsed[key][1],dtype=torch.float64).reshape(parsed[key][0]).requires_grad_(True) for key in ('left','right')}
        left,right=values[bindings['left']],values[bindings['right']]
        for i in order:
            if stopped():return
            kind=nodes[i]['type'];ports=incoming[i]
            if kind=='input':continue
            if kind=='matmul':values[i]=matrix_multiply(values[ports['a']],values[ports['b']])
            elif kind=='output':values[i]=values[ports['in']]
            else:values[i]=apply_layout(values[ports['in']],kind,configs[i])
        output=values[output_id]
        da,db=torch.autograd.grad((output*torch.tensor(gv,dtype=torch.float64).reshape(gshape)).sum(),(left,right))
        if stopped():return
        def measured(t):
            if not torch.isfinite(t).all():raise ValueError('Matrix assessment produced nonfinite measurements.')
            return {'shape':list(t.shape),'values':t.detach().reshape(-1).tolist()}
        evidence['fixtures'].append({'fixtureID':fixture['id'],'tensors':{'output':measured(output),'leftGradient':measured(da),'rightGradient':measured(db)},'partitions':{}})
