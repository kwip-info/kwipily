"""Named-port bounded routing lessons. Expected answers remain in the trusted assessment plan."""
import math
import torch
if __package__:
    from .tensor_layout import BatchedLayout, apply_layout
    from .tensor_routing_configuration import normalize_tensor_routing,configured_inputs,resolve_tensor_routing
    from .tensor_composition_configuration import normalize_tensor_composition,resolve_tensor_composition
    from .graph_semantics import effective_config
else:
    from tensor_layout import BatchedLayout,apply_layout
    from tensor_routing_configuration import normalize_tensor_routing,configured_inputs,resolve_tensor_routing
    from tensor_composition_configuration import normalize_tensor_composition,resolve_tensor_composition
    from graph_semantics import effective_config
TASKS={'split-size-order':('split',{'x':(2,5)}),'split-sections-order':('split',{'x':(2,5)}),'chunk-uneven-parts':('chunk',{'x':(2,13)}),'gather-repeated-positions':('gather',{'x':(2,4),'indices':(2,3)}),'einsum-combine-tables':('einsum',{'left':(2,3),'right':(3,2)}),'einsum-diagonal-view':('einsum',{'x':(3,3)})}

def compile_graph(submission,task):
    operation,shapes=TASKS[task]
    if type(submission) is not dict or set(submission)!={'schemaVersion','kind','graph'} or type(submission['schemaVersion']) is not int or submission['schemaVersion']!=1 or submission['kind']!='tensorRoutingGraph':raise ValueError('Submit only a tensorRoutingGraph, without answers, seeds or code.')
    graph=submission['graph']
    if type(graph) is not dict or set(graph)-{'schemaVersion','name','nodes','edges'} or type(graph.get('schemaVersion')) is not int or graph['schemaVersion']!=2:raise ValueError('Routing lessons require schema 2.')
    nodes,edges=graph.get('nodes'),graph.get('edges')
    if type(nodes) is not list or not 3<=len(nodes)<=32 or type(edges) is not list or len(edges)>64:raise ValueError('Connect the declared inputs, requested block and Output within lesson bounds.')
    by_id={};configs={};bindings={};outputs=[];ports={}
    for node in nodes:
        if type(node) is not dict or not {'id','type','config'}<=set(node) or set(node)-{'id','type','config','name','x','y'}:raise ValueError('Malformed routing component.')
        ident,kind,c=node['id'],node['type'],node['config']
        if type(ident) is not str or not 1<=len(ident)<=128 or ident in by_id or type(kind) is not str or type(c) is not dict:raise ValueError('Invalid component identity or settings.')
        if any(k in node and (type(node[k]) not in (int,float) or not math.isfinite(node[k])) for k in ('x','y')):raise ValueError('Invalid canvas coordinates.')
        if kind=='input':
            if set(c)!={'fixture'} or c['fixture'] not in shapes or c['fixture'] in bindings:raise ValueError('Bind each declared fixture once.')
            bindings[c['fixture']]=ident;configs[ident]=c;ports[ident]=()
        elif kind=='output':
            if c:raise ValueError('Output has no settings.')
            outputs.append(ident);configs[ident]={};ports[ident]=('in',)
        elif kind==operation:configs[ident]=normalize_tensor_routing(kind,c);ports[ident]=configured_inputs(kind,configs[ident])
        elif kind=='concat':configs[ident]=normalize_tensor_composition(kind,c);ports[ident]=('a','b')
        elif kind in ('reshape','transpose','flatten'):configs[ident]=effective_config(node,2);ports[ident]=('in',)
        else:raise ValueError('Unsupported routing lesson block.')
        by_id[ident]=node
    if set(bindings)!=set(shapes) or len(outputs)!=1 or sum(n['type']==operation for n in nodes)!=1:raise ValueError('Connect the declared inputs, exactly one requested block and one Output.')
    incoming={i:{} for i in by_id};successors={i:[] for i in by_id};ids=set()
    for edge in edges:
        if type(edge) is not dict or set(edge)!={'id','source','sourcePort','target','targetPort'}:raise ValueError('Malformed named-port connection.')
        eid,a,ap,b,bp=(edge[k] for k in ('id','source','sourcePort','target','targetPort'))
        if type(eid) is not str or not eid or eid in ids or type(a) is not str or type(b) is not str or a not in by_id or b not in by_id:raise ValueError('Invalid connection identity.')
        declared=tuple(f'part{i}' for i in range(8)) if by_id[a]['type'] in ('split','chunk') else ('out',)
        if by_id[a]['type']=='output' or ap not in declared or bp not in ports[b] or bp in incoming[b]:raise ValueError('Use available source ports and each configured input exactly once.')
        incoming[b][bp]=(a,ap);successors[a].append(b);ids.add(eid)
    for i in by_id:
        if set(incoming[i])!=set(ports[i]) or (by_id[i]['type']!='output' and not successors[i]):raise ValueError('Complete all configured inputs and remove unused blocks.')
    pending={i:len(p) for i,p in incoming.items()};queue=[i for i,v in pending.items() if not v];order=[]
    while queue:
        i=queue.pop(0);order.append(i)
        for child in successors[i]:
            pending[child]-=1
            if not pending[child]:queue.append(child)
    if len(order)!=len(nodes):raise ValueError('Remove cycles from the lesson.')
    return by_id,configs,bindings,outputs[0],incoming,ports,order

def execute(submission,fixtures,evidence,stopped,parse_tensor,*,task):
    if task not in TASKS:raise ValueError('Unknown routing lesson.')
    operation,shapes=TASKS[task];nodes,configs,bindings,out,incoming,ports,order=compile_graph(submission,task)
    for fixture in fixtures:
        if stopped():return
        if set(fixture['inputs'])!=set(shapes)|{'upstream'}:raise ValueError('Use only the trusted task fixtures.')
        parsed={key:parse_tensor(value) for key,value in fixture['inputs'].items()}
        for shape,values in parsed.values():
            if not shape or len(shape)>8 or any(type(n) is not int or not 1<=n<=4096 for n in shape) or math.prod(shape)!=len(values) or len(values)>4096 or not all(type(v) in (int,float) and math.isfinite(v) for v in values):raise ValueError('Invalid finite bounded fixture.')
        rows=parsed[next(iter(shapes))][0][0]
        if not 1<=rows<=3 or any(parsed[role][0]!=[rows,*shape] for role,shape in shapes.items()):raise ValueError('Fixture shapes or observation counts differ from the task.')
        if 'indices' in parsed:
            if any(type(v) is bool or v!=int(v) or v<0 for v in parsed['indices'][1]):raise ValueError('Selectors must be nonnegative whole positions.')
        layouts={(bindings[r],'out'):BatchedLayout(tuple(shape)) for r,shape in shapes.items()};dtypes={(bindings[r],'out'):('int64' if r=='indices' else 'float64') for r in shapes};total=sum(len(v) for s,v in parsed.values())
        for ident in order:
            kind=nodes[ident]['type'];c=configs[ident]
            if kind=='input':continue
            addresses=[incoming[ident][p] for p in ports[ident]]
            if any(a not in layouts for a in addresses):raise ValueError('Selected part is unavailable for the current shape/settings.')
            ins=[layouts[a] for a in addresses];types=[dtypes[a] for a in addresses]
            if kind==operation:
                if types != (['float64','int64'] if kind=='gather' else ['float64']*len(ins)):raise ValueError('Routing source types differ from configured ports.')
                resolved=resolve_tensor_routing(kind,c,ins);outs=resolved['outputs']
                if kind=='gather':
                    # The selector branch can only rearrange its fixture positions; its values
                    # remain unchanged. Bound those values against the resolved number source,
                    # not the source fixture's original direction before layout transforms.
                    extent=ins[0].sample[ins[0]._axis(c['dim'])]
                    if any(v>=extent for v in parsed['indices'][1]):raise ValueError('Selectors exceed the actual connected source direction.')
                if rows*resolved['workPerObservation']>100_000_000:raise ValueError('Einsum work exceeds the task budget.')
            elif kind=='concat':
                if types!=['float64','float64']:raise ValueError('Concatenate requires floating lesson sources.')
                outs={'out':resolve_tensor_composition(kind,c,*ins)}
            else:
                layout=ins[0]
                if kind=='reshape':layout=layout.reshape(c['shape'])
                elif kind=='transpose':layout=layout.transpose(c['dim0'],c['dim1'])
                elif kind=='flatten':layout=layout.flatten(c['start_dim'],c['end_dim'])
                outs={'out':layout}
            for port,layout in outs.items():
                size=rows*math.prod(layout.sample)
                if size>4096:raise ValueError('Lesson output exceeds bounded allocation.')
                total+=size;layouts[ident,port]=layout;dtypes[ident,port]=types[0]
        if parsed['upstream'][0]!=[rows,*layouts[out,'out'].sample] or dtypes[out,'out']!='float64':raise ValueError('Measured output differs from task influence shape or type.')
        evidence['maximumTensorElements']=max(evidence['maximumTensorElements'],3*total)
        if stopped():return
        roots={r:torch.tensor(parsed[r][1],dtype=torch.int64 if r=='indices' else torch.float64).reshape(parsed[r][0]).requires_grad_(r!='indices') for r in shapes};values={(bindings[r],'out'):v for r,v in roots.items()}
        for ident in order:
            if stopped():return
            kind=nodes[ident]['type'];c=configs[ident]
            if kind=='input':continue
            ins=[values[incoming[ident][p]] for p in ports[ident]]
            if kind=='split':
                result=torch.split(ins[0],c['split_size'] if c['mode']=='size' else c['sections'],dim=c['dim']);outs={f'part{i}':v for i,v in enumerate(result)}
            elif kind=='chunk':outs={f'part{i}':v for i,v in enumerate(torch.chunk(ins[0],c['chunks'],dim=c['dim']))}
            elif kind=='gather':outs={'out':torch.gather(ins[0],c['dim'],ins[1])}
            elif kind=='einsum':outs={'out':torch.einsum(c['equation'],*ins)}
            elif kind=='concat':outs={'out':torch.cat(ins,dim=c['dim'])}
            elif kind=='output':outs={'out':ins[0]}
            else:outs={'out':apply_layout(ins[0],kind,c)}
            values.update({(ident,p):v for p,v in outs.items()})
        y=values[out,'out'];upstream=torch.tensor(parsed['upstream'][1],dtype=torch.float64).reshape(parsed['upstream'][0]);floating={r:t for r,t in roots.items() if r!='indices'}
        grads=torch.autograd.grad(y,tuple(floating.values()),grad_outputs=upstream)
        if stopped():return
        def captured(value):
            if not torch.isfinite(value).all():raise ValueError('Nonfinite routing output or source influence.')
            return {'shape':list(value.shape),'values':value.detach().reshape(-1).tolist()}
        tensors={'output':captured(y)}
        for (role,root),grad in zip(floating.items(),grads):tensors['inputGradient' if role=='x' else role+'Gradient']=captured(grad)
        evidence['fixtures'].append({'fixtureID':fixture['id'],'tensors':tensors,'partitions':{}})
