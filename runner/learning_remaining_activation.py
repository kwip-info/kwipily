"""Strict CPU float64 lesson measurement; canonical masks originate only from upstream capture."""
import copy
import hashlib
import json
import math
from pathlib import Path
import torch
if __package__:
    from .learning_tensor_transforms import compile_graph as compile_path
    from .remaining_activation_configuration import normalize_remaining_activation,resolve_remaining_activation
    from .tensor_layout import BatchedLayout,apply_layout
else:
    from learning_tensor_transforms import compile_graph as compile_path
    from remaining_activation_configuration import normalize_remaining_activation,resolve_remaining_activation
    from tensor_layout import BatchedLayout,apply_layout

TASKS={'prelu-shared-slope':'prelu','prelu-channel-slopes':'prelu','logsoftmax-sample-scores':'log_softmax','dropout-training-pattern':'dropout','dropout-evaluation-pattern':'dropout'}
BANK_PATH=Path(__file__).resolve().parent/'fixtures/remaining_activation_dropout_masks_v1.json'
BANK_SHA256='5a96044ae7f6286b354d6cb43cd6bec4e040a15b55bffc01ecaa57bbc1056538'

def canonical_bank():
    if str(torch.__version__)!='2.14.1':raise ValueError('Canonical dropout masks require runtime Torch 2.14.1; refresh reviewed lesson evidence, never silently regenerate masks.')
    raw=BANK_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=BANK_SHA256:raise ValueError('Canonical dropout mask integrity differs from its reviewed resource.')
    bank=json.loads(raw)
    if bank['maskVersion']!='kwip.dropout.cpu-float64.v1' or bank['torchVersion']!=str(torch.__version__) or bank['dtype']!='float64' or bank['device']!='cpu' or bank['p']!=.5:raise ValueError('Canonical dropout mask runtime/version differs.')
    return bank

def finite_tensor(value,parse_tensor):
    if type(value) is not dict or set(value)!={'shape','values'}:raise ValueError('Provide only tensor shape and values.')
    shape,values=parse_tensor(value)
    if type(shape) is not list or type(values) is not list or not 1<=len(shape)<=8 or any(type(n) is not int or not 1<=n<=4096 for n in shape) or math.prod(shape)!=len(values) or len(values)>4096 or not all(type(v) in (int,float) and math.isfinite(v) for v in values):raise ValueError('Use bounded finite tensors with positive integer directions.')
    return shape,values

def compile_submission(submission,task,parse_tensor):
    if task not in TASKS or type(submission) is not dict or set(submission)!={'schemaVersion','kind','graph','parameters','gradientNodeID'} or type(submission['schemaVersion']) is not int or submission['schemaVersion']!=1 or submission['kind']!='remainingActivationGraph':raise ValueError('Submit a reviewed remainingActivationGraph; authored masks, seeds and answers are refused.')
    operation=TASKS[task];surrogate=copy.deepcopy(submission);graph=surrogate['graph'];surrogate={'schemaVersion':1,'kind':'tensorGraph','graph':graph}
    if type(graph) is not dict or type(graph.get('nodes')) is not list:raise ValueError('Malformed activation graph.')
    active={}
    for node in graph['nodes']:
        if type(node) is not dict:raise ValueError('Malformed component.')
        for key in ('x','y'):
            if key in node and (type(node[key]) not in (int,float) or not math.isfinite(node[key])):raise ValueError('Use finite canvas coordinates.')
        if node.get('type')==operation:
            active[node.get('id')]=normalize_remaining_activation(operation,node.get('config'))
            node['type']='reshape';node['config']={'shape':[2,3]}
    if len(active)!=1:raise ValueError('Use exactly one requested activation block.')
    operations,count=compile_path(surrogate,include_ids=True)
    operations=[(ident,operation,active[ident]) if ident in active else (ident,kind,config) for ident,kind,config in operations]
    parameters=submission['parameters'];target=submission['gradientNodeID'];parsed={};parameter_count=0
    if type(parameters) is not dict:raise ValueError('Provide an explicit parameter map.')
    if operation=='prelu':
        if set(parameters)!=set(active) or type(target) is not str or target not in active:raise ValueError('Supply slopes for exactly the PReLU node and select it for slope influence.')
        for ident,config in active.items():
            tensors=parameters[ident]
            if type(tensors) is not dict or set(tensors)!={'weight'}:raise ValueError('PReLU state contains only its learned slope weight.')
            shape,values=finite_tensor(tensors['weight'],parse_tensor)
            if shape!=[config['num_parameters']]:raise ValueError('Slope count must match the configured parameter count.')
            parsed[ident]=(shape,values);parameter_count+=len(values)
    elif parameters or target is not None:raise ValueError('Parameterless tasks have no supplied state or gradient target.')
    return operations,count,parsed,parameter_count

def execute(submission,fixtures,evidence,stopped,parse_tensor,*,task):
    operation=TASKS.get(task)
    operations,count,parameters,parameter_count=compile_submission(submission,task,parse_tensor)
    for fixture in fixtures:
        if stopped():return
        wanted={'x','upstream','maskBankIndex'} if operation=='dropout' else {'x','upstream'}
        if set(fixture['inputs'])!=wanted:raise ValueError('Fixture fields differ from the reviewed activation task.')
        shape,values=finite_tensor(fixture['inputs']['x'],parse_tensor);gshape,gvalues=finite_tensor(fixture['inputs']['upstream'],parse_tensor)
        if len(shape)!=3 or shape[0] not in (2,3,4) or shape[1:]!=[2,3]:raise ValueError('Use the public two-channel, three-position sample with 2..4 observations.')
        entry=None
        if operation=='dropout':
            index_shape,index_values=finite_tensor(fixture['inputs']['maskBankIndex'],parse_tensor)
            if index_shape!=[1] or len(index_values)!=1 or index_values[0]!=int(index_values[0]) or not 0<=index_values[0]<12:raise ValueError('Invalid canonical mask bank selection.')
            entry=canonical_bank()['entries'][int(index_values[0])]
            if entry['shape']!=shape:raise ValueError('Canonical mask observation count differs from the fixture.')
        layout=BatchedLayout(tuple(shape[1:]));largest=len(values)
        for _,kind,config in operations:
            if kind==operation:
                if entry is not None and [shape[0],*layout.sample]!=entry['shape']:raise ValueError('The reviewed dropout mask requires the declared component layout.')
                layout=resolve_remaining_activation(kind,config,layout)['layout']
            elif kind=='reshape':layout=layout.reshape(config['shape'])
            elif kind=='transpose':layout=layout.transpose(config['dim0'],config['dim1'])
            else:layout=layout.flatten(config['start_dim'],config['end_dim'])
            largest=max(largest,shape[0]*math.prod(layout.sample))
        if [shape[0],*layout.sample]!=gshape:raise ValueError('Output shape differs from the supplied output influence.')
        evidence['maximumTensorElements']=max(evidence['maximumTensorElements'],3*parameter_count+len(values)*(count+5)+len(gvalues)*3+largest*3)
        if stopped():return
        x=torch.tensor(values,dtype=torch.float64).reshape(shape).requires_grad_(True)
        weights={ident:torch.tensor(vals,dtype=torch.float64).reshape(dims).requires_grad_(True) for ident,(dims,vals) in parameters.items()}
        upstream=torch.tensor(gvalues,dtype=torch.float64).reshape(gshape)
        def forward(training):
            output=x
            for ident,kind,config in operations:
                if stopped():return None
                if kind=='prelu':output=torch.nn.functional.prelu(output,weights[ident])
                elif kind=='log_softmax':output=torch.nn.functional.log_softmax(output,dim=config['dim'])
                elif kind=='dropout':output=torch.nn.functional.dropout(output,p=config['p'],training=training,inplace=False)
                else:output=apply_layout(output,kind,config)
            return output
        def measure(training):
            if entry is None:output=forward(training)
            else:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(entry['seed']);output=forward(training)
            if output is None:return None
            targets=[x]+list(weights.values())
            gradients=torch.autograd.grad((output*upstream).sum(),targets)
            result={'output':output,'inputGradient':gradients[0]}
            if weights:result['slopeGradient']=gradients[1]
            return result
        result=measure(task!='dropout-evaluation-pattern')
        if result is None or stopped():return
        if task=='dropout-evaluation-pattern':
            training=measure(True)
            if training is None or stopped():return
            result.update(trainingOutput=training['output'],trainingInputGradient=training['inputGradient'])
        def captured(t):
            if not torch.isfinite(t).all():raise ValueError('Nonfinite activation output or influence.')
            return {'shape':list(t.shape),'values':t.detach().reshape(-1).tolist()}
        evidence['fixtures'].append({'fixtureID':fixture['id'],'tensors':{key:captured(value) for key,value in result.items()},'partitions':{}})
