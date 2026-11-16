"""Measured bag lessons using real row preparation and the production native adapter.

Private sourceRows are rectangular table cells, not a dense model input. Lengths
collate those rows into genuine packed arrays, including zero-item packets.
Only floating table/weight sources have derivatives; IDs/offsets never do.
"""
import math
import torch
if __package__:
    from .embedding_bag import CheckedEmbeddingBag
    from .embedding_bag_configuration import normalize_embedding_bag, configured_input_ports
    from .embedding_bag_packet import EmbeddingBagPacket
    from .bag_source import normalize_bag_source, execute_bag_source
else:
    from embedding_bag import CheckedEmbeddingBag
    from embedding_bag_configuration import normalize_embedding_bag, configured_input_ports
    from embedding_bag_packet import EmbeddingBagPacket
    from bag_source import normalize_bag_source, execute_bag_source


def compile_graph(submission):
    if type(submission) is not dict or set(submission)!={'schemaVersion','kind','graph'} or type(submission['schemaVersion']) is not int or submission['schemaVersion']!=1 or submission['kind']!='embeddingBagGraph':
        raise ValueError('Submit an embeddingBagGraph construction, never authored answers or code.')
    g=submission['graph']
    if type(g) is not dict or set(g)!={'schemaVersion','name','nodes','edges'} or type(g['schemaVersion']) is not int or g['schemaVersion']!=2 or type(g['name']) is not str or type(g['nodes']) is not list or len(g['nodes'])!=3 or type(g['edges']) is not list:
        raise ValueError('Use one Bag Input, one EmbeddingBag and one Output without training settings.')
    nodes={}
    for n in g['nodes']:
        if type(n) is not dict or set(n)!={'id','type','name','x','y','config'} or type(n['id']) is not str or not 1<=len(n['id'])<=128 or n['id'] in nodes or type(n['name']) is not str or any(type(n[k]) not in (int,float) or not math.isfinite(n[k]) for k in ('x','y')):
            raise ValueError('Component identities and positions must be finite and unique.')
        nodes[n['id']]=n
    if sorted(n['type'] for n in nodes.values())!=['bag_input','embedding_bag','output']:
        raise ValueError('Use the real Bag Input and EmbeddingBag components.')
    source=next(n for n in nodes.values() if n['type']=='bag_input')
    bag=next(n for n in nodes.values() if n['type']=='embedding_bag')
    out=next(n for n in nodes.values() if n['type']=='output')
    sc=normalize_bag_source(source['config']);bc=normalize_embedding_bag(bag['config'])
    if out['config']!={} or any(sc[k]!=bc[k] for k in ('layout','per_sample_weights','include_last_offset')):
        raise ValueError('Source roles and bag layout/weight/offset controls must agree.')
    required={(source['id'],r,bag['id'],r) for r in configured_input_ports(bc)}|{(bag['id'],'out',out['id'],'in')}
    actual=set();ids=set()
    for e in g['edges']:
        if type(e) is not dict or set(e)!={'id','source','sourcePort','target','targetPort'} or any(type(v) is not str or not v for v in e.values()) or e['id'] in ids:
            raise ValueError('Connections require unique identities and exact named roles.')
        ids.add(e['id']);actual.add((e['source'],e['sourcePort'],e['target'],e['targetPort']))
    if len(g['edges'])!=len(required) or actual!=required:
        raise ValueError('Connect every enabled source role to its matching bag input, then bag output to Output.')
    return sc,bc


def _tensor(raw,parse_tensor):
    shape,values=parse_tensor(raw)
    if type(shape) not in (list,tuple) or not shape or len(shape)>3 or any(type(d) is not int or not 1<=d<=4096 for d in shape) or math.prod(shape)>4096 or len(values)!=math.prod(shape) or any(type(v) not in (int,float) or not math.isfinite(v) or abs(v)>3.4028234663852886e38 for v in values):
        raise ValueError('Lesson source cells must be bounded finite numeric tables.')
    return list(shape),values


def execute(submission,fixtures,evidence,stopped,parse_tensor):
    sc,bc=compile_graph(submission)
    for fixture in fixtures:
        if stopped():return
        inputs=fixture['inputs'];roles={'sourceRows','table','upstream'}|({'lengths'} if sc['layout']=='packed' else set())|({'sourceWeights'} if sc['per_sample_weights'] else set())
        if set(inputs)!=roles:raise ValueError('Bag fixtures require exact row-source roles, not user answers.')
        parsed={k:_tensor(v,parse_tensor) for k,v in inputs.items()}
        shape,rows=parsed['sourceRows'];ts,table=parsed['table'];gs,upstream=parsed['upstream']
        if len(shape)!=2 or shape[1]>sc['maximum_items_per_bag'] or ts!=[bc['num_embeddings'],bc['embedding_dim']] or gs!=[shape[0],bc['embedding_dim']]:
            raise ValueError('Source, table and per-bag influence shapes must agree.')
        if any(v!=int(v) or not 0<=v<bc['num_embeddings'] for v in rows):raise ValueError('Every source ID cell must select an existing table row, including unused cells.')
        lengths=[shape[1]]*shape[0]
        if 'lengths' in parsed:
            ls,lv=parsed['lengths']
            if ls!=[shape[0]] or any(v!=int(v) or not 0<=v<=shape[1] for v in lv):raise ValueError('Every packed row length must be a bounded whole number, including zero.')
            lengths=list(map(int,lv))
        if 'sourceWeights' in parsed and parsed['sourceWeights'][0]!=shape:raise ValueError('Item weights must match every source ID cell.')
        evidence['maximumTensorElements']=max(evidence['maximumTensorElements'],8*sum(len(v) for _,v in parsed.values())+sum(lengths)*3)
        if stopped():return
        ids=torch.tensor(rows,dtype=torch.int64).reshape(shape)
        source_weights=torch.tensor(parsed['sourceWeights'][1],dtype=torch.float32).reshape(shape).requires_grad_(True) if 'sourceWeights' in parsed else None
        offsets=None
        if sc['layout']=='packed':
            positions=[r*shape[1]+c for r,length in enumerate(lengths) for c in range(length)]
            select=torch.tensor(positions,dtype=torch.int64)
            ids=ids.reshape(-1).index_select(0,select)
            weights=source_weights.reshape(-1).index_select(0,select) if source_weights is not None else None
            starts=[0]
            for length in lengths:starts.append(starts[-1]+length)
            offsets=torch.tensor(starts if sc['include_last_offset'] else starts[:-1],dtype=torch.int64)
        else:weights=source_weights
        packet=EmbeddingBagPacket(sc['layout'],ids,offsets,weights,sc['include_last_offset'])
        execute_bag_source(packet,sc)
        layer=CheckedEmbeddingBag(bc)
        with torch.no_grad():layer.weight.copy_(torch.tensor(table,dtype=torch.float32).reshape(ts))
        output=layer(packet)
        tensors={'output':output,'tableAfterForward':layer.weight}
        differentiable=([layer.weight] if layer.weight.requires_grad else [])+([source_weights] if source_weights is not None else [])
        if differentiable:
            gradients=torch.autograd.grad((output*torch.tensor(upstream,dtype=torch.float32).reshape(gs)).sum(),differentiable)
            names=(['tableGradient'] if layer.weight.requires_grad else [])+(['sourceWeightsGradient'] if source_weights is not None else [])
            tensors.update(zip(names,(g.to_dense() if g.is_sparse else g for g in gradients)))
        if stopped():return
        measured={}
        for name,t in tensors.items():
            if not torch.isfinite(t).all().item():raise ValueError('Bag assessment produced nonfinite measurements.')
            measured[name]={'shape':list(t.shape),'values':t.detach().reshape(-1).tolist()}
        evidence['fixtures'].append({'fixtureID':fixture['id'],'tensors':measured,'partitions':{}})
