"""Trusted stateful normalization teaching sequence; no authored buffers or mode overrides."""
import math
import torch
if __package__:
    from .stateful_normalization_configuration import normalize_stateful_normalization,resolve_stateful_normalization
    from .tensor_layout import BatchedLayout,apply_layout
    from .graph_semantics import effective_config
else:
    from stateful_normalization_configuration import normalize_stateful_normalization,resolve_stateful_normalization
    from tensor_layout import BatchedLayout,apply_layout
    from graph_semantics import effective_config
TASKS={
 'batch-norm1d-current-group':('batch_norm1d',(3,4),False,False),
 'batch-norm1d-remembered-summary':('batch_norm1d',(3,4),True,True),
 'batch-norm2d-current-group':('batch_norm2d',(3,2,3),False,False),
 'batch-norm2d-remembered-summary':('batch_norm2d',(3,2,3),True,True),
 'instance-norm1d-individual-patterns':('instance_norm1d',(3,4),False,False),
 'instance-norm1d-remembered-summary':('instance_norm1d',(3,4),True,True),
 'instance-norm2d-individual-grids':('instance_norm2d',(3,2,3),False,False),
 'instance-norm2d-remembered-summary':('instance_norm2d',(3,2,3),True,True),
 'group-norm-channel-teams':('group_norm',(4,3),False,False),
 'group-norm-channel-adjustments':('group_norm',(4,3),False,True)}
def compile_graph(submission,operation):
    submission={k:v for k,v in submission.items() if k not in ('parameters','gradientNodeID')}
    if type(submission) is not dict or set(submission)!={'schemaVersion','kind','graph'} or type(submission['schemaVersion']) is not int or submission['schemaVersion']!=1 or submission['kind']!='statefulNormalizationGraph':raise ValueError('Submit a statefulNormalizationGraph without answers, seeds, state or code.')
    graph=submission['graph']
    if type(graph) is not dict or set(graph)-{'schemaVersion','name','nodes','edges'} or type(graph.get('schemaVersion')) is not int or graph['schemaVersion']!=2:raise ValueError('Spatial normalization lessons require schema 2.')
    nodes,edges=graph.get('nodes'),graph.get('edges')
    if type(nodes) is not list or not 3<=len(nodes)<=32 or type(edges) is not list or len(edges)>31:raise ValueError('Connect Input, the requested block and Output within lesson bounds.')
    by_id={};configs={};sources=[];outputs=[]
    for node in nodes:
        if type(node) is not dict or not {'id','type','config'}<=set(node) or set(node)-{'id','type','config','name','x','y'}:raise ValueError('Malformed spatial normalization component.')
        ident,kind,c=node['id'],node['type'],node['config']
        if type(ident) is not str or not 1<=len(ident)<=128 or ident in by_id or type(c) is not dict:raise ValueError('Invalid component identity or configuration.')
        if any(k in node and (type(node[k]) not in (int,float) or not math.isfinite(node[k])) for k in ('x','y')):raise ValueError('Invalid canvas position.')
        if kind in ('input','output'):
            if c:raise ValueError('Lesson boundaries have no configuration.')
            (sources if kind=='input' else outputs).append(ident);configs[ident]={}
        elif kind==operation:configs[ident]=normalize_stateful_normalization(kind,c)
        elif kind in ('reshape','transpose','flatten'):configs[ident]=effective_config(node,2)
        else:raise ValueError('Unsupported spatial normalization lesson block.')
        by_id[ident]=node
    if len(sources)!=1 or len(outputs)!=1 or sum(n['type']==operation for n in nodes)!=1:raise ValueError('Use one Input, one requested normalization block and one Output.')
    incoming={};successors={i:[] for i in by_id};ids=set()
    for e in edges:
        if type(e) is not dict or set(e)!={'id','source','sourcePort','target','targetPort'}:raise ValueError('Malformed normalization connection.')
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
    if len(order)!=len(nodes):raise ValueError('Remove cycles from the normalization graph.')
    return by_id,configs,sources[0],outputs[0],incoming,order

def execute(submission,fixtures,evidence,stopped,parse_tensor,*,task):
    if task not in TASKS:raise ValueError('Unknown normalization objective.')
    if type(submission) is not dict or set(submission)!={'schemaVersion','kind','graph','parameters','gradientNodeID'}:raise ValueError('Provide graph and active learned parameters only; no buffers, answers, modes or seeds.')
    operation,sample,remembered,_=TASKS[task];nodes,configs,source,out,incoming,order=compile_graph(submission,operation)
    active=next(i for i in order if nodes[i]['type']==operation);config=configs[active];parameters=submission['parameters'];target=submission['gradientNodeID']
    if type(parameters) is not dict or set(parameters)!=({active} if config['affine'] else set()) or (target!=active if config['affine'] else target is not None):raise ValueError('Provide parameters for exactly the affine-enabled block and select it for parameter influence.')
    parsed_parameters={}
    if config['affine']:
        channel=config['num_channels'] if operation=='group_norm' else config['num_features'];names={'weight','bias'} if config['bias'] else {'weight'}
        if type(parameters[active]) is not dict or set(parameters[active])!=names:raise ValueError('Provide exactly active per-channel scale and optional shift.')
        for name,t in parameters[active].items():
            if type(t) is not dict or set(t)!={'shape','values'}:raise ValueError('Channel parameters contain only shape and values.')
            dims,values=parse_tensor(t)
            if dims!=[channel] or len(values)!=channel or any(type(v) not in (int,float) or not math.isfinite(v) for v in values):raise ValueError('Learned scale/shift must be finite channel vectors.')
            parsed_parameters[name]=(dims,values)
    for fixture in fixtures:
        if stopped():return
        required={'x','upstream','calibrationA','calibrationB'} if remembered else {'x','upstream'}
        if set(fixture['inputs'])!=required:raise ValueError('Use only the trusted source/influence and declared calibration sequence.')
        if any(type(t) is not dict or set(t)!={'shape','values'} for t in fixture['inputs'].values()):raise ValueError('Trusted tensor fixtures contain only shape and values.')
        parsed={k:parse_tensor(t) for k,t in fixture['inputs'].items()};total=3*sum(len(v) for dims,v in parsed.values())+3*sum(len(v) for dims,v in parsed_parameters.values())
        for key,(shape,values) in parsed.items():
            if not shape or len(shape)>8 or any(type(v) is not int or not 1<=v<=4096 for v in shape) or math.prod(shape)!=len(values) or len(values)>4096 or any(type(v) not in (int,float) or not math.isfinite(v) for v in values):raise ValueError('Malformed finite bounded fixture.')
            if not 1<=shape[0]<=3 or tuple(shape[1:])!=sample:raise ValueError('Fixture source shape differs from the objective.')
            if key=='upstream':continue
            layouts={source:BatchedLayout(sample)}
            for ident in order:
                kind=nodes[ident]['type'];c=configs[ident]
                if kind=='input':continue
                layout=layouts[incoming[ident]]
                if kind==operation:
                    resolved=resolve_stateful_normalization(kind,c,layout);layout=resolved['layout'];spatial=math.prod(layout.sample[1:])
                    uses_current=key!='x' or not remembered or not c.get('track_running_stats',False)
                    if uses_current and ((kind.startswith('batch') and shape[0]*spatial<=1) or (kind.startswith('instance') and spatial<=1)):raise ValueError('Current normalization statistics need more positions or observations.')
                elif kind=='reshape':layout=layout.reshape(c['shape'])
                elif kind=='transpose':layout=layout.transpose(c['dim0'],c['dim1'])
                elif kind=='flatten':layout=layout.flatten(c['start_dim'],c['end_dim'])
                if shape[0]*math.prod(layout.sample)>4096:raise ValueError('Normalization output exceeds lesson allocation.')
                total+=3*shape[0]*math.prod(layout.sample);layouts[ident]=layout
            if key=='x' and parsed['upstream'][0]!=[shape[0],*layouts[out].sample]:raise ValueError('Output and influence shapes differ.')
        channels=sample[0];total+=4*channels+2 if remembered else 0
        evidence['maximumTensorElements']=max(evidence['maximumTensorElements'],total)
        if stopped():return
        c=config.copy();c.pop('averaging',None)
        if operation.startswith('batch') and config['averaging']=='cumulative':c['momentum']=None
        cls={'batch_norm1d':torch.nn.BatchNorm1d,'batch_norm2d':torch.nn.BatchNorm2d,'instance_norm1d':torch.nn.InstanceNorm1d,'instance_norm2d':torch.nn.InstanceNorm2d,'group_norm':torch.nn.GroupNorm}[operation]
        module=cls(**c,dtype=torch.float64)
        with torch.no_grad():
            for name,(dims,vals) in parsed_parameters.items():getattr(module,name).copy_(torch.tensor(vals,dtype=torch.float64).reshape(dims))
        def forward(value):
            values={source:value}
            for ident in order:
                if stopped():return None
                kind=nodes[ident]['type']
                if kind=='input':continue
                x=values[incoming[ident]];values[ident]=module(x) if kind==operation else x if kind=='output' else apply_layout(x,kind,configs[ident])
            return values[out]
        if remembered:
            module.train()
            with torch.no_grad():
                for key in ('calibrationA','calibrationB'):
                    dims,vals=parsed[key]
                    if forward(torch.tensor(vals,dtype=torch.float64).reshape(dims)) is None:return
            module.eval()
        else:module.train()
        def captured(t):
            if not torch.isfinite(t).all():raise ValueError('Nonfinite normalization output/influence/state.')
            return {'shape':list(t.shape) or [1],'values':t.detach().reshape(-1).tolist()}
        result={}
        def state(prefix):
            names={'running_mean':'RunningMean','running_var':'RunningVariance','num_batches_tracked':'BatchesTracked'}
            for name,label in names.items():
                t=getattr(module,name,None)
                if t is None:raise ValueError('This objective requires remembered statistics; enable tracking.')
                result[prefix+label if prefix else label[0].lower()+label[1:]]=captured(t)
        if remembered:state('')
        dims,vals=parsed['x'];root=torch.tensor(vals,dtype=torch.float64).reshape(dims).requires_grad_(True);y=forward(root)
        if y is None:return
        upstream=torch.tensor(parsed['upstream'][1],dtype=torch.float64).reshape(parsed['upstream'][0]);targets=[root];names=['inputGradient']
        if config['affine']:
            targets.append(module.weight);names.append('scaleGradient')
            if config['bias']:targets.append(module.bias);names.append('biasGradient')
        gradients=torch.autograd.grad(y,targets,grad_outputs=upstream);result['output']=captured(y);result.update({name:captured(value) for name,value in zip(names,gradients)})
        module.eval()
        with torch.no_grad():evaluation=forward(root.detach())
        if evaluation is None or stopped():return
        result['evaluationOutput']=captured(evaluation)
        if remembered:state('postEval')
        evidence['fixtures'].append({'fixtureID':fixture['id'],'tensors':result,'partitions':{}})
