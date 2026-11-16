"""Measured bounded Bilinear graphs. Inputs/parameters come only from private fixtures."""
import math
import torch
import torch.nn.functional as F
if __package__:
    from .tensor_layout import BatchedLayout, apply_layout, normalize_configuration
else:
    from tensor_layout import BatchedLayout, apply_layout, normalize_configuration


def compile_graph(submission, bias_enabled):
    if (type(submission) is not dict or set(submission) != {'schemaVersion','kind','graph'}
            or type(submission['schemaVersion']) is not int or submission['schemaVersion'] != 1
            or submission['kind'] != 'bilinearGraph'):
        raise ValueError('Submit a bilinearGraph, never authored answers or code.')
    graph = submission['graph']
    if (type(graph) is not dict or set(graph)-{'schemaVersion','name','nodes','edges'}
            or type(graph.get('schemaVersion')) is not int or graph['schemaVersion'] != 2):
        raise ValueError('Bilinear lesson graph requires schema 2.')
    nodes, edges = graph.get('nodes'), graph.get('edges')
    if type(nodes) is not list or not 5 <= len(nodes) <= 32 or type(edges) is not list or len(edges) > 64:
        raise ValueError('Bilinear lessons require 5..32 components and at most 64 connections.')
    expected = {'left','right','weight'} | ({'bias'} if bias_enabled else set())
    by_id, configs, bindings, outputs, operators = {}, {}, {}, [], []
    for node in nodes:
        if type(node) is not dict or not {'id','type','config'} <= set(node) or set(node)-{'id','type','config','name','x','y'}:
            raise ValueError('Malformed Bilinear component.')
        ident, kind, config = node['id'], node['type'], node['config']
        if type(ident) is not str or not 1 <= len(ident) <= 128 or ident in by_id or type(kind) is not str or type(config) is not dict:
            raise ValueError('Invalid Bilinear component identity/configuration.')
        if kind == 'input':
            binding = config.get('fixture')
            if set(config) != {'fixture'} or type(binding) is not str or binding not in expected or binding in bindings:
                raise ValueError('Bind exactly one Input to each active fixture operand.')
            bindings[binding] = ident; configs[ident] = config
        elif kind == 'output':
            if config: raise ValueError('Output has no configuration.')
            outputs.append(ident); configs[ident] = {}
        elif kind == 'bilinear':
            if set(config) != {'bias'} or type(config['bias']) is not bool or config['bias'] is not bias_enabled:
                raise ValueError('Bilinear bias must be a boolean matching the selected assessment.')
            operators.append(ident); configs[ident] = config
        elif kind in ('reshape','transpose','flatten'):
            configs[ident] = normalize_configuration(kind, config)
        else: raise ValueError('Unsupported Bilinear lesson operation.')
        by_id[ident] = node
    if set(bindings) != expected or len(outputs) != 1 or len(operators) != 1:
        raise ValueError('Use all active fixture Inputs, exactly one Bilinear, and one Output.')
    incoming = {i:{} for i in by_id}; successors = {i:[] for i in by_id}; edge_ids = set()
    ports = ['a','b','weight'] + (['bias'] if bias_enabled else [])
    for edge in edges:
        if type(edge) is not dict or set(edge) != {'id','source','sourcePort','target','targetPort'}:
            raise ValueError('Malformed Bilinear connection.')
        ident, a, b, port = edge['id'], edge['source'], edge['target'], edge['targetPort']
        if (type(ident) is not str or not 1 <= len(ident) <= 128 or ident in edge_ids
                or type(a) is not str or type(b) is not str or a not in by_id or b not in by_id
                or edge['sourcePort'] != 'out' or by_id[a]['type'] == 'output'):
            raise ValueError('Invalid Bilinear connection identity or source.')
        kind = by_id[b]['type']; allowed = [] if kind == 'input' else ports if kind == 'bilinear' else ['in']
        if type(port) is not str or port not in allowed or port in incoming[b]:
            raise ValueError('Connect each declared input exactly once.')
        incoming[b][port] = a; successors[a].append(b); edge_ids.add(ident)
    for i, node in by_id.items():
        wanted = 0 if node['type'] == 'input' else len(ports) if node['type'] == 'bilinear' else 1
        if len(incoming[i]) != wanted: raise ValueError('Incomplete Bilinear graph.')
        if node['type'] != 'output' and not successors[i]: raise ValueError('Unused Bilinear component.')
    pending = {i:len(v) for i,v in incoming.items()}; queue = [i for i,v in pending.items() if v == 0]; order = []
    while queue:
        ident = queue.pop(0); order.append(ident)
        for child in successors[ident]:
            pending[child] -= 1
            if pending[child] == 0: queue.append(child)
    if len(order) != len(nodes): raise ValueError('Bilinear graph contains a cycle.')
    return by_id, configs, bindings, outputs[0], incoming, order


def transformed_shape(shape, kind, config):
    if len(shape) < 2: raise ValueError('Transforms require a protected leading axis and sample axes.')
    layout = BatchedLayout(tuple(shape[1:]))
    if kind == 'reshape': layout = layout.reshape(config['shape'])
    elif kind == 'transpose': layout = layout.transpose(config['dim0'], config['dim1'])
    else: layout = layout.flatten(config['start_dim'], config['end_dim'])
    return [shape[0], *layout.sample]


def execute(submission, fixtures, evidence, stopped, parse_tensor, *, bias_enabled):
    nodes, configs, bindings, output_id, incoming, order = compile_graph(submission, bias_enabled)
    active = ['left','right','weight'] + (['bias'] if bias_enabled else [])
    for fixture in fixtures:
        if stopped(): return
        if set(fixture['inputs']) != set(active) | {'upstream'}:
            raise ValueError('Bilinear fixtures require exactly the active operands and upstream gradient.')
        parsed = {key:parse_tensor(value) for key,value in fixture['inputs'].items()}
        shapes = {bindings[key]:parsed[key][0] for key in active}
        total = sum(len(v[1]) for v in parsed.values())
        for ident in order:
            if stopped(): return
            kind = nodes[ident]['type']; ports = incoming[ident]
            if kind == 'input': continue
            if kind == 'bilinear':
                a, b, w = (shapes[ports[p]] for p in ('a','b','weight'))
                if (len(a) < 2 or len(b) < 2 or a[:-1] != b[:-1] or len(w) != 3
                        or a[-1] != w[1] or b[-1] != w[2]):
                    raise ValueError('Bilinear requires exact operand prefixes, matching widths and weight [O,H1,H2]; broadcasting is unsupported.')
                if bias_enabled and shapes[ports['bias']] != [w[0]]:
                    raise ValueError('Bilinear bias must have shape [O].')
                # One forward and three differentiated contractions; bias summation adds no contraction.
                if 4 * math.prod(a[:-1]) * math.prod(w) > 100_000:
                    raise ValueError('Bilinear lesson exceeds 100,000 forward/backward contraction terms.')
                shape = [*a[:-1], w[0]]
            elif kind == 'output': shape = shapes[ports['in']]
            else: shape = transformed_shape(shapes[ports['in']], kind, configs[ident])
            size = math.prod(shape)
            if size > 4096: raise ValueError('Bilinear intermediate exceeds 4096 elements.')
            shapes[ident] = shape; total += size
        if shapes[output_id] != parsed['upstream'][0]:
            raise ValueError('Bilinear graph output shape differs from upstream gradient.')
        evidence['maximumTensorElements'] = max(evidence['maximumTensorElements'], 3 * total)
        if stopped(): return
        leaves = {key:torch.tensor(parsed[key][1],dtype=torch.float64).reshape(parsed[key][0]).requires_grad_(True) for key in active}
        values = {bindings[key]:value for key,value in leaves.items()}
        for ident in order:
            if stopped(): return
            kind = nodes[ident]['type']; ports = incoming[ident]
            if kind == 'input': continue
            if kind == 'bilinear':
                values[ident] = F.bilinear(values[ports['a']], values[ports['b']], values[ports['weight']], values[ports['bias']] if bias_enabled else None)
            elif kind == 'output': values[ident] = values[ports['in']]
            else: values[ident] = apply_layout(values[ports['in']], kind, configs[ident])
        upstream = torch.tensor(parsed['upstream'][1],dtype=torch.float64).reshape(parsed['upstream'][0])
        gradients = torch.autograd.grad((values[output_id] * upstream).sum(), tuple(leaves.values()))
        if stopped(): return
        def measured(value):
            if not torch.isfinite(value).all(): raise ValueError('Bilinear assessment produced nonfinite measurements.')
            return {'shape':list(value.shape),'values':value.detach().reshape(-1).tolist()}
        tensors = {'output':measured(values[output_id])}
        tensors.update({key+'Gradient':measured(value) for key,value in zip(active,gradients)})
        evidence['fixtures'].append({'fixtureID':fixture['id'],'tensors':tensors,'partitions':{}})
