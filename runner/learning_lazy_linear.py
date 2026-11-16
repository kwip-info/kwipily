"""Measured inferred-width/state task; no learner answers or seeded output oracle."""
import copy
import math
import torch
if __package__:
    from .learning_tensor_transforms import compile_graph as compile_path
    from .tensor_layout import apply_layout, BatchedLayout
    from .graph_semantics import effective_config
else:
    from learning_tensor_transforms import compile_graph as compile_path
    from tensor_layout import apply_layout, BatchedLayout
    from graph_semantics import effective_config


def compile_graph(submission):
    if type(submission) is not dict or set(submission)!={'schemaVersion','kind','graph'} or type(submission['schemaVersion']) is not int or submission['schemaVersion']!=1 or submission['kind']!='lazyLinearGraph':
        raise ValueError('Submit a lazyLinearGraph, never authored evidence or code.')
    surrogate=copy.deepcopy(submission);surrogate['kind']='tensorGraph'
    graph=surrogate.get('graph')
    if type(graph) is not dict or type(graph.get('nodes')) is not list:raise ValueError('Malformed inferred-width graph.')
    lazy={}
    for node in graph['nodes']:
        if type(node) is dict and node.get('type')=='lazy_linear':
            config=effective_config(node,2)
            lazy[node.get('id')]=config
            node['type']='reshape';node['config']={'shape':[2,3]}
    if len(lazy)!=1:raise ValueError('Use exactly one connected LazyLinear.')
    operations,count=compile_path(surrogate,include_ids=True)
    return [(ident,'lazy_linear',lazy[ident]) if ident in lazy else (ident,kind,config) for ident,kind,config in operations],count


def execute(submission,fixtures,evidence,stopped,parse_tensor):
    operations,node_count=compile_graph(submission)
    for fixture in fixtures:
        if stopped():return
        if set(fixture['inputs'])!={'x'}:raise ValueError('Inferred-width fixtures require only public-shaped measurements.')
        shape,values=parse_tensor(fixture['inputs']['x'])
        if len(shape)!=3 or shape[1:]!=[2,3] or not 1<=shape[0]<=3 or len(values)!=math.prod(shape) or not all(math.isfinite(v) for v in values):
            raise ValueError('This task requires finite measurements shaped [batch,2,3].')
        # Resolve allocation bounds before any parameter/RNG allocation.
        layout=BatchedLayout(tuple(shape[1:]));width=None;config=None
        for _,kind,settings in operations:
            if kind=='lazy_linear':
                width=layout.sample[-1];config=settings
                if not 1<=width<=4096 or (width+int(settings['bias']))*settings['out_features']>2_000_000:raise ValueError('Lazy parameter budget exceeded.')
                layout=BatchedLayout((*layout.sample[:-1],settings['out_features']))
            elif kind=='reshape':layout=layout.reshape(settings['shape'])
            elif kind=='transpose':layout=layout.transpose(settings['dim0'],settings['dim1'])
            else:layout=layout.flatten(settings['start_dim'],settings['end_dim'])
        parameters=(width+int(config['bias']))*config['out_features']
        evidence['maximumTensorElements']=max(evidence['maximumTensorElements'],parameters+len(values)*(node_count+3)+shape[0]*math.prod(layout.sample)*3)
        if stopped():return
        layer=torch.nn.LazyLinear(config['out_features'],bias=config['bias'])
        # Same single shape-only forward as production materialize_lazy_parameters.
        with torch.no_grad():layer(torch.zeros((1,width),dtype=torch.float32))
        def forward(module):
            out=torch.tensor(values,dtype=torch.float32).reshape(shape)
            for _,kind,settings in operations:
                out=module(out) if kind=='lazy_linear' else apply_layout(out,kind,settings)
                if not torch.isfinite(out).all():raise ValueError('Nonfinite inferred-width output.')
            return out.detach().clone()
        before={k:v.detach().clone() for k,v in layer.state_dict().items()}
        rng=torch.get_rng_state().clone()
        first=forward(layer);second=forward(layer)
        repeat=all(torch.equal(v,layer.state_dict()[k]) for k,v in before.items()) and torch.equal(first,second)
        reload=torch.nn.LazyLinear(config['out_features'],bias=config['bias'])
        reload.load_state_dict(before,strict=True)
        restored=forward(reload)
        state_ok=all(torch.equal(v,reload.state_dict()[k]) for k,v in before.items()) and torch.equal(first,restored)
        rng_ok=torch.equal(rng,torch.get_rng_state())
        actual_count=sum(p.numel() for p in layer.parameters())
        count_ok=actual_count==width*config['out_features']+(config['out_features'] if config['bias'] else 0)
        def vector(v):return {'shape':[len(v)],'values':v}
        tensors={'output-shape':vector(list(first.shape)),'inferred-width':vector([layer.weight.shape[1]]),'weight-shape':vector(list(layer.weight.shape)),
                 'parameter-count':vector([int(count_ok)]),'repeat-state':vector([int(repeat)]),'reload-state':vector([int(state_ok)]),'rng-preserved':vector([int(rng_ok)])}
        if stopped():return
        evidence['fixtures'].append({'fixtureID':fixture['id'],'tensors':tensors,'partitions':{}})
