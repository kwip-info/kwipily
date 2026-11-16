"""Bounded state-free spatial patch lessons; expected values are trusted-plan-owned."""
import math
import torch
if __package__:
    from .spatial_patch_configuration import normalize_spatial_patch,resolve_spatial_patch
    from .tensor_layout import BatchedLayout,apply_layout
    from .graph_semantics import effective_config
else:
    from spatial_patch_configuration import normalize_spatial_patch,resolve_spatial_patch
    from tensor_layout import BatchedLayout,apply_layout
    from graph_semantics import effective_config
TASKS={'unfold-overlapping-windows':('unfold',(2,3,4)),'unfold-padded-spaced-windows':('unfold',(2,3,4)),'fold-overlap-addition':('fold',(8,20)),'fold-uncovered-positions':('fold',(8,4))}
def compile_graph(submission,operation):
    if type(submission) is not dict or set(submission)!={'schemaVersion','kind','graph'} or type(submission['schemaVersion']) is not int or submission['schemaVersion']!=1 or submission['kind']!='spatialPatchGraph':raise ValueError('Submit a spatialPatchGraph without answers, seeds, state or code.')
    graph=submission['graph']
    if type(graph) is not dict or set(graph)-{'schemaVersion','name','nodes','edges'} or type(graph.get('schemaVersion')) is not int or graph['schemaVersion']!=2:raise ValueError('Spatial patch lessons require schema 2.')
    nodes,edges=graph.get('nodes'),graph.get('edges')
    if type(nodes) is not list or not 3<=len(nodes)<=32 or type(edges) is not list or len(edges)>31:raise ValueError('Connect Input, the requested block and Output within lesson bounds.')
    by_id={};configs={};sources=[];outputs=[]
    for node in nodes:
        if type(node) is not dict or not {'id','type','config'}<=set(node) or set(node)-{'id','type','config','name','x','y'}:raise ValueError('Malformed spatial patch component.')
        ident,kind,c=node['id'],node['type'],node['config']
        if type(ident) is not str or not 1<=len(ident)<=128 or ident in by_id or type(c) is not dict:raise ValueError('Invalid component identity or configuration.')
        if any(k in node and (type(node[k]) not in (int,float) or not math.isfinite(node[k])) for k in ('x','y')):raise ValueError('Invalid canvas position.')
        if kind in ('input','output'):
            if c:raise ValueError('Lesson boundaries have no configuration.')
            (sources if kind=='input' else outputs).append(ident);configs[ident]={}
        elif kind==operation:configs[ident]=normalize_spatial_patch(kind,c)
        elif kind in ('reshape','transpose','flatten'):configs[ident]=effective_config(node,2)
        else:raise ValueError('Unsupported spatial patch lesson block.')
        by_id[ident]=node
    if len(sources)!=1 or len(outputs)!=1 or sum(n['type']==operation for n in nodes)!=1:raise ValueError('Use one Input, one requested patch block and one Output.')
    incoming={};successors={i:[] for i in by_id};ids=set()
    for e in edges:
        if type(e) is not dict or set(e)!={'id','source','sourcePort','target','targetPort'}:raise ValueError('Malformed patch connection.')
        eid,a,b=e['id'],e['source'],e['target']
        if type(eid) is not str or not eid or eid in ids or type(a) is not str or type(b) is not str or a not in by_id or b not in by_id or e['sourcePort']!='out' or e['targetPort']!='in' or by_id[a]['type']=='output' or by_id[b]['type']=='input' or b in incoming:raise ValueError('Connect each declared input once using its out/in ports.')
        incoming[b]=a;successors[a].append(b);ids.add(eid)
    if any(i not in incoming for i in by_id if i!=sources[0]) or any(not successors[i] for i in by_id if i!=outputs[0]):raise ValueError('Complete all connections and remove unused blocks.')
    pending={i:int(i in incoming) for i in by_id};queue=[sources[0]];order=[]
    while queue:
        i=queue.pop(0);order.append(i)
        for child in successors[i]:
            pending[child]-=1
            if not pending[child]:queue.append(child)
    if len(order)!=len(nodes):raise ValueError('Remove cycles from the patch graph.')
    return by_id,configs,sources[0],outputs[0],incoming,order

def execute(submission,fixtures,evidence,stopped,parse_tensor,*,task):
    if task not in TASKS:raise ValueError('Unknown spatial patch lesson.')
    operation,sample=TASKS[task];nodes,configs,source,out,incoming,order=compile_graph(submission,operation)
    for fixture in fixtures:
        if stopped():return
        if set(fixture['inputs'])!={'x','upstream'}:raise ValueError('Use only the trusted number-source and output-influence fixtures.')
        parsed={key:parse_tensor(value) for key,value in fixture['inputs'].items()}
        for shape,values in parsed.values():
            if not shape or len(shape)>8 or any(type(v) is not int or not 1<=v<=4096 for v in shape) or math.prod(shape)!=len(values) or len(values)>4096 or any(type(v) not in (int,float) or not math.isfinite(v) for v in values):raise ValueError('Invalid finite bounded patch fixture.')
        shape,x=parsed['x'];rows=shape[0]
        if not 1<=rows<=3 or tuple(shape[1:])!=sample:raise ValueError('Source fixture differs from the public task shape.')
        layouts={source:BatchedLayout(sample)};total=sum(len(v) for s,v in parsed.values())
        for ident in order:
            kind=nodes[ident]['type'];c=configs[ident]
            if kind=='input':continue
            layout=layouts[incoming[ident]]
            if kind==operation:
                resolved=resolve_spatial_patch(kind,c,layout);layout=resolved['layout']
                if rows*resolved['workPerObservation']>100_000_000:raise ValueError('Spatial extraction/addition work exceeds the budget.')
            elif kind=='reshape':layout=layout.reshape(c['shape'])
            elif kind=='transpose':layout=layout.transpose(c['dim0'],c['dim1'])
            elif kind=='flatten':layout=layout.flatten(c['start_dim'],c['end_dim'])
            size=rows*math.prod(layout.sample)
            if size>4096:raise ValueError('Patch output exceeds bounded lesson allocation.')
            total+=size;layouts[ident]=layout
        if parsed['upstream'][0]!=[rows,*layouts[out].sample]:raise ValueError('Measured output differs from the task influence shape.')
        evidence['maximumTensorElements']=max(evidence['maximumTensorElements'],3*total)
        if stopped():return
        root=torch.tensor(x,dtype=torch.float64).reshape(shape).requires_grad_(True);values={source:root}
        for ident in order:
            if stopped():return
            kind=nodes[ident]['type'];c=configs[ident]
            if kind=='input':continue
            value=values[incoming[ident]]
            if kind=='unfold':values[ident]=torch.nn.functional.unfold(value,**c)
            elif kind=='fold':values[ident]=torch.nn.functional.fold(value,**c)
            elif kind=='output':values[ident]=value
            else:values[ident]=apply_layout(value,kind,c)
        y=values[out];upstream=torch.tensor(parsed['upstream'][1],dtype=torch.float64).reshape(parsed['upstream'][0]);gradient,=torch.autograd.grad(y,(root,),grad_outputs=upstream)
        if stopped():return
        def captured(value):
            if not torch.isfinite(value).all():raise ValueError('Nonfinite spatial patch output or influence.')
            return {'shape':list(value.shape),'values':value.detach().reshape(-1).tolist()}
        evidence['fixtures'].append({'fixtureID':fixture['id'],'tensors':{'output':captured(y),'inputGradient':captured(gradient)},'partitions':{}})
