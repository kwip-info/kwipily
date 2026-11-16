"""Measured lookup lessons with fixture-owned tables and discrete input coordinates."""
import math
import torch
if __package__:
    from .learning_tensor_transforms import compile_graph
    from .tensor_layout import BatchedLayout, apply_layout
else:
    from learning_tensor_transforms import compile_graph
    from tensor_layout import BatchedLayout, apply_layout


def execute(submission,fixtures,evidence,stopped,parse_tensor):
    if type(submission) is not dict or set(submission)!={'schemaVersion','kind','graph'} or type(submission['schemaVersion']) is not int or submission['schemaVersion']!=1 or submission['kind']!='embeddingGraph':
        raise ValueError('Submit an embeddingGraph, never authored answers or executable code.')
    operations,node_count=compile_graph({**submission,'kind':'tensorGraph'},allowed_operations=frozenset({'embedding','reshape','transpose','flatten'}))
    lookups=[config for kind,config in operations if kind=='embedding']
    if len(lookups)!=1:raise ValueError('The lookup lesson requires exactly one embedding using the fixture table.')
    config=lookups[0]
    if config['freeze'] or (config['sparse'] and config['scale_grad_by_freq']):
        raise ValueError('This assessment requires a trainable table and compatible gradient settings.')
    for fixture in fixtures:
        if stopped():return
        if set(fixture['inputs'])!={'indices','table','upstream'}:raise ValueError('Lookup fixtures require indices, table and upstream.')
        shape,indices=parse_tensor(fixture['inputs']['indices'])
        tshape,table=parse_tensor(fixture['inputs']['table'])
        gshape,upstream=parse_tensor(fixture['inputs']['upstream'])
        if not 2<=len(shape)<=7 or tshape!=[config['num_embeddings'],config['embedding_dim']]:raise ValueError('Lookup table or batched index shape does not match the submission.')
        if any(type(i) not in (int,float) or not math.isfinite(i) or i!=int(i) or not 0<=i<config['num_embeddings'] for i in indices):
            raise ValueError('Lookup fixture coordinates must be exact integers selecting existing rows.')
        layout=BatchedLayout(tuple(shape[1:]))
        maximum=len(indices)
        for kind,c in operations:
            if kind=='embedding':layout=BatchedLayout((*layout.sample,c['embedding_dim']))
            elif kind=='reshape':layout=layout.reshape(c['shape'])
            elif kind=='transpose':layout=layout.transpose(c['dim0'],c['dim1'])
            elif kind=='flatten':layout=layout.flatten(c['start_dim'],c['end_dim'])
            maximum=max(maximum,shape[0]*math.prod(layout.sample))
        if [shape[0],*layout.sample]!=gshape:raise ValueError('Lookup output shape differs from upstream.')
        evidence['maximumTensorElements']=max(evidence['maximumTensorElements'],maximum*(node_count+3)+len(upstream)+3*len(table))
        if stopped():return
        output=torch.tensor(indices,dtype=torch.int64).reshape(shape)
        weights=torch.tensor(table,dtype=torch.float64).reshape(tshape).requires_grad_(True)
        for kind,c in operations:
            if stopped():return
            if kind=='embedding':
                output=torch.nn.functional.embedding(output,weights,padding_idx=c['padding_idx'],max_norm=c['max_norm'],norm_type=c['norm_type'],scale_grad_by_freq=c['scale_grad_by_freq'],sparse=c['sparse'])
            else:output=apply_layout(output,kind,c)
        gradient,=torch.autograd.grad((output*torch.tensor(upstream,dtype=torch.float64).reshape(gshape)).sum(),weights)
        if gradient.is_sparse:gradient=gradient.to_dense()
        if stopped():return
        def measured(t):
            if not torch.isfinite(t).all():raise ValueError('Lookup assessment produced nonfinite measurements.')
            return {'shape':list(t.shape),'values':t.detach().reshape(-1).tolist()}
        evidence['fixtures'].append({'fixtureID':fixture['id'],'tensors':{'output':measured(output),'tableAfterForward':measured(weights),'tableGradient':measured(gradient)},'partitions':{}})
