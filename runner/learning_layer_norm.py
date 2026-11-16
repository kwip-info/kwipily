"""Bounded fixture-shaped normalization submissions; all outputs are measured."""
import torch
if __package__:
    from .learning_tensor_transforms import compile_graph
    from .tensor_layout import BatchedLayout, apply_layout
else:
    from learning_tensor_transforms import compile_graph
    from tensor_layout import BatchedLayout, apply_layout

OPERATIONS=frozenset({'layer_norm','reshape','transpose','flatten'})


def compile_submission(submission, parse_tensor):
    if type(submission) is not dict or set(submission)!={'schemaVersion','kind','graph','parameters','gradientNodeID'} or type(submission['schemaVersion']) is not int or submission['schemaVersion']!=1 or submission['kind']!='layerNormGraph':
        raise ValueError('Submit a layerNormGraph with explicit parameters and gradient target, never authored results or code.')
    operations,count=compile_graph({'schemaVersion':1,'kind':'tensorGraph','graph':submission['graph']},allowed_operations=OPERATIONS,include_ids=True)
    active={ident:config for ident,kind,config in operations if kind=='layer_norm' and config['elementwise_affine']}
    parameters=submission['parameters'];target=submission['gradientNodeID']
    if type(parameters) is not dict or set(parameters)!=set(active):
        raise ValueError('Provide parameters for exactly the affine-enabled normalization components.')
    if target is not None and (type(target) is not str or target not in active):
        raise ValueError('Gradient target must identify an affine-enabled normalization component.')
    parsed={};parameter_count=0
    for ident,config in active.items():
        values=parameters[ident];names={'weight','bias'} if config['bias'] else {'weight'}
        if type(values) is not dict or set(values)!=names:
            raise ValueError('Normalization parameters must contain exactly the active weight/bias tensors.')
        parsed[ident]={}
        for name in sorted(names):
            shape,flat=parse_tensor(values[name])
            if shape!=config['normalized_shape']:raise ValueError('Parameter shape must equal normalized_shape.')
            parsed[ident][name]=(shape,flat);parameter_count+=len(flat)
    return operations,count,parsed,parameter_count,target


def execute(submission,fixtures,evidence,stopped,parse_tensor):
    operations,node_count,parameters,parameter_count,target=compile_submission(submission,parse_tensor)
    evidence['maximumTensorElements']=parameter_count*3
    if stopped():return
    for fixture in fixtures:
        if stopped():return
        if set(fixture['inputs'])!={'x','upstream'}:raise ValueError('Normalization fixtures require x and upstream.')
        shape,values=parse_tensor(fixture['inputs']['x']);gshape,gvalues=parse_tensor(fixture['inputs']['upstream'])
        if len(shape)<2:raise ValueError('Preserve batch and sample axes.')
        layout=BatchedLayout(tuple(shape[1:]))
        for _,kind,config in operations:
            if kind=='layer_norm':layout=layout.layer_norm(config['normalized_shape'])
            elif kind=='reshape':layout=layout.reshape(config['shape'])
            elif kind=='transpose':layout=layout.transpose(config['dim0'],config['dim1'])
            elif kind=='flatten':layout=layout.flatten(config['start_dim'],config['end_dim'])
        if [shape[0],*layout.sample]!=gshape:raise ValueError('Output shape differs from upstream fixture.')
        evidence['maximumTensorElements']=max(evidence['maximumTensorElements'],len(values)*(node_count+3)+len(gvalues)+3*parameter_count)
        if stopped():return
        # Double precision is an assessment oracle domain, not a new production device/dtype claim.
        x=torch.tensor(values,dtype=torch.float64).reshape(shape).requires_grad_(True)
        weights={ident:{name:torch.tensor(flat,dtype=torch.float64).reshape(dims).requires_grad_(True) for name,(dims,flat) in tensors.items()} for ident,tensors in parameters.items()}
        output=x
        for ident,kind,config in operations:
            if stopped():return
            if kind=='layer_norm':
                state=weights.get(ident,{})
                output=torch.nn.functional.layer_norm(output,config['normalized_shape'],state.get('weight'),state.get('bias'),config['eps'])
            else:output=apply_layout(output,kind,config)
        upstream=torch.tensor(gvalues,dtype=torch.float64).reshape(gshape)
        targets=[x];names=['inputGradient']
        if target is not None:
            targets.append(weights[target]['weight']);names.append('scaleGradient')
            if 'bias' in weights[target]:targets.append(weights[target]['bias']);names.append('biasGradient')
        gradients=torch.autograd.grad((output*upstream).sum(),targets)
        if stopped():return
        def measured(t):
            if not torch.isfinite(t).all():raise ValueError('Normalization assessment produced nonfinite measured values.')
            return {'shape':list(t.shape),'values':t.detach().reshape(-1).tolist()}
        tensors={'output':measured(output)}
        tensors.update({name:measured(value) for name,value in zip(names,gradients)})
        evidence['fixtures'].append({'fixtureID':fixture['id'],'tensors':tensors,'partitions':{}})
