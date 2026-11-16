"""Registry-validated CPU graph compiler and reproducible two-moons experiment."""
from __future__ import annotations
import inspect
import warnings
import hashlib
import json
import math
import keyword
import re
import unicodedata
import os
from pathlib import Path
import platform
import tempfile
import uuid
import torch
from torch import nn
if __package__:
    from .module_composition_plan import validate_plan, Invocation, ExecutionFrame, PrimitiveInvocation, GraphContainer, OrderedContainer, NamedContainer
    from .optimizer_configuration import validate_recipe as validate_optimizer_recipe, create_optimizer as create_reviewed_optimizer, restore_optimizer as restore_reviewed_optimizer, validate_optimizer_state, training_recipe_code, capability as optimizer_capability, NAMES as OPTIMIZER_NAMES
    from .scheduler_configuration import validate_recipe as validate_scheduler_recipe, SchedulerController, preflight as scheduler_preflight, capability as scheduler_capability, NAMES as SCHEDULER_NAMES
    from .attention_configuration import resolve_attention, configured_input_ports as attention_inputs, capability as attention_capability, OPERATIONS as ATTENTION_OPERATIONS, FLOAT_PORTS as ATTENTION_FLOAT_PORTS
    from .attention_execution import create_attention_module, execute_attention
    from .recurrent_configuration import resolve_recurrent, configured_input_ports as recurrent_inputs, capability as recurrent_capability
    from .stateful_normalization_configuration import resolve_stateful_normalization, capability as stateful_normalization_capability
    from .spatial_patch_configuration import resolve_spatial_patch, capability as spatial_patch_capability
    from .tensor_routing_configuration import configured_inputs as routing_inputs, resolve_tensor_routing, capability as routing_capability
    from .remaining_activation_configuration import resolve_remaining_activation, capability as remaining_activation_capability
    from .adaptive_pooling_configuration import torch_output_size
    from .tensor_composition_configuration import resolve_tensor_composition, capability as composition_capability
    from .configurable_activation_configuration import resolve_configuration as resolve_activation, capability as activation_capability
    from .relu_configuration import resolve_relu, capability as relu_capability
    from .tanh_configuration import resolve_tanh, capability as tanh_capability
    from .sigmoid_configuration import resolve_sigmoid, capability as sigmoid_capability
    from .pooling_adapters_generated import RESOLVERS as POOLING_RESOLVERS, CAPABILITIES as POOLING_CAPABILITIES
    from .output_routing import OutputAddress, select_output, summarize_stage
    from .artifact_storage import ArtifactStorage
    from .worker_resources import configure_torch
    from .execution_device import resolve_device
    from .embedding import CheckedEmbedding, capability as embedding_capability
    from .embedding_bag import CheckedEmbeddingBag
    from .embedding_bag_configuration import configured_input_ports as bag_input_ports
    from .embedding_bag_packet import EmbeddingBagPacket
    from .bag_source import BagCoordinateLayout, source_layouts as bag_source_layouts, execute_bag_source
    from .gradient_capture import GradientCapture, validate_config as validate_gradient_config, capability as gradient_capability, failed_capability as failed_gradient_capability
    from .product_execution import ProductExecutionAdapter, execution_contract
    from .tensor_layout import BatchedLayout, apply_layout
    from .bilinear_configuration import capability as bilinear_capability
    from .conv1d_configuration import resolve_conv1d, capability as conv1d_capability
    from .conv2d_configuration import resolve_conv2d, capability as conv2d_capability
    from .depthwise_conv2d_configuration import resolve_depthwise_conv2d, conv2d_config, capability as depthwise_conv2d_capability
    from .conv3d_configuration import resolve_conv3d, capability as conv3d_capability
    from .conv_transpose1d_configuration import resolve_conv_transpose1d, capability as conv_transpose1d_capability
    from .conv_transpose2d_configuration import resolve_conv_transpose2d, capability as conv_transpose2d_capability
    from .conv_transpose3d_configuration import resolve_conv_transpose3d, capability as conv_transpose3d_capability
    from .lazy_linear_configuration import resolve_lazy_linear, capability as lazy_linear_capability
    from .matrix_multiply import matrix_multiply, capability as matrix_capability
    from .graph_semantics import effective_config, TRANSFORMS
    from .data import csv_dataset, inspect_data, select_split
else:
    from module_composition_plan import validate_plan, Invocation, ExecutionFrame, PrimitiveInvocation, GraphContainer, OrderedContainer, NamedContainer
    from optimizer_configuration import validate_recipe as validate_optimizer_recipe, create_optimizer as create_reviewed_optimizer, restore_optimizer as restore_reviewed_optimizer, validate_optimizer_state, training_recipe_code, capability as optimizer_capability, NAMES as OPTIMIZER_NAMES
    from scheduler_configuration import validate_recipe as validate_scheduler_recipe, SchedulerController, preflight as scheduler_preflight, capability as scheduler_capability, NAMES as SCHEDULER_NAMES
    from attention_configuration import resolve_attention, configured_input_ports as attention_inputs, capability as attention_capability, OPERATIONS as ATTENTION_OPERATIONS, FLOAT_PORTS as ATTENTION_FLOAT_PORTS
    from attention_execution import create_attention_module, execute_attention
    from recurrent_configuration import resolve_recurrent, configured_input_ports as recurrent_inputs, capability as recurrent_capability
    from stateful_normalization_configuration import resolve_stateful_normalization, capability as stateful_normalization_capability
    from spatial_patch_configuration import resolve_spatial_patch, capability as spatial_patch_capability
    from tensor_routing_configuration import configured_inputs as routing_inputs, resolve_tensor_routing, capability as routing_capability
    from remaining_activation_configuration import resolve_remaining_activation, capability as remaining_activation_capability
    from adaptive_pooling_configuration import torch_output_size
    from tensor_composition_configuration import resolve_tensor_composition, capability as composition_capability
    from configurable_activation_configuration import resolve_configuration as resolve_activation, capability as activation_capability
    from relu_configuration import resolve_relu, capability as relu_capability
    from tanh_configuration import resolve_tanh, capability as tanh_capability
    from sigmoid_configuration import resolve_sigmoid, capability as sigmoid_capability
    from pooling_adapters_generated import RESOLVERS as POOLING_RESOLVERS, CAPABILITIES as POOLING_CAPABILITIES
    from output_routing import OutputAddress, select_output, summarize_stage
    from artifact_storage import ArtifactStorage
    from worker_resources import configure_torch
    from execution_device import resolve_device
    from embedding import CheckedEmbedding, capability as embedding_capability
    from embedding_bag import CheckedEmbeddingBag
    from embedding_bag_configuration import configured_input_ports as bag_input_ports
    from embedding_bag_packet import EmbeddingBagPacket
    from bag_source import BagCoordinateLayout, source_layouts as bag_source_layouts, execute_bag_source
    from gradient_capture import GradientCapture, validate_config as validate_gradient_config, capability as gradient_capability, failed_capability as failed_gradient_capability
    from product_execution import ProductExecutionAdapter, execution_contract
    from tensor_layout import BatchedLayout, apply_layout
    from bilinear_configuration import capability as bilinear_capability
    from conv1d_configuration import resolve_conv1d, capability as conv1d_capability
    from conv2d_configuration import resolve_conv2d, capability as conv2d_capability
    from depthwise_conv2d_configuration import resolve_depthwise_conv2d, conv2d_config, capability as depthwise_conv2d_capability
    from conv3d_configuration import resolve_conv3d, capability as conv3d_capability
    from conv_transpose1d_configuration import resolve_conv_transpose1d, capability as conv_transpose1d_capability
    from conv_transpose2d_configuration import resolve_conv_transpose2d, capability as conv_transpose2d_capability
    from conv_transpose3d_configuration import resolve_conv_transpose3d, capability as conv_transpose3d_capability
    from lazy_linear_configuration import resolve_lazy_linear, capability as lazy_linear_capability
    from matrix_multiply import matrix_multiply, capability as matrix_capability
    from graph_semantics import effective_config, TRANSFORMS
    from data import csv_dataset, inspect_data, select_split

REGISTRY = Path(__file__).resolve().parents[1] / 'Sources/KwipMLCore/Resources/components.json'
RECIPE_VERSION = 2
CONVOLUTION_RESOLVERS = {'conv1d':resolve_conv1d,'conv2d':resolve_conv2d,'conv3d':resolve_conv3d,'depthwise_conv2d':resolve_depthwise_conv2d}
CONVOLUTION_LAYERS = {'conv1d':nn.Conv1d,'conv2d':nn.Conv2d,'conv3d':nn.Conv3d,'depthwise_conv2d':nn.Conv2d}
TRANSPOSE_CONVOLUTION_RESOLVERS = {'conv_transpose1d':resolve_conv_transpose1d,'conv_transpose2d':resolve_conv_transpose2d,'conv_transpose3d':resolve_conv_transpose3d}
TRANSPOSE_CONVOLUTION_LAYERS = {'conv_transpose1d':nn.ConvTranspose1d,'conv_transpose2d':nn.ConvTranspose2d,'conv_transpose3d':nn.ConvTranspose3d}
POOLING_FUNCTIONS = {'avg_pool1d':torch.nn.functional.avg_pool1d,'max_pool1d':torch.nn.functional.max_pool1d,'max_pool2d':torch.nn.functional.max_pool2d,'max_pool3d':torch.nn.functional.max_pool3d}
REVIEWED_NEW_POOLING = frozenset({'avg_pool2d','adaptive_avg_pool1d','adaptive_avg_pool2d','adaptive_max_pool2d'})
POOLING_FUNCTIONS.update({kind:getattr(torch.nn.functional,kind) for kind in REVIEWED_NEW_POOLING})
STATEFUL_NORMALIZATION_LAYERS={'batch_norm1d':nn.BatchNorm1d,'batch_norm2d':nn.BatchNorm2d,'instance_norm1d':nn.InstanceNorm1d,'instance_norm2d':nn.InstanceNorm2d,'group_norm':nn.GroupNorm}
STATEFUL_NORMALIZATION_OPERATIONS=frozenset(STATEFUL_NORMALIZATION_LAYERS)
SPATIAL_PATCH_OPERATIONS=frozenset({'unfold','fold'})
ROUTING_OPERATIONS=frozenset({'split','chunk','gather','einsum'})
RECURRENT_LAYERS={'rnn':nn.RNN,'lstm':nn.LSTM,'gru':nn.GRU,'rnn_cell':nn.RNNCell,'lstm_cell':nn.LSTMCell,'gru_cell':nn.GRUCell}
RECURRENT_OPERATIONS=frozenset(RECURRENT_LAYERS)
SUPPORTED = {'bag_input','embedding_bag'} | set(ATTENTION_OPERATIONS) | set(RECURRENT_OPERATIONS) | set(STATEFUL_NORMALIZATION_OPERATIONS) | {'unfold','fold','split','chunk','gather','einsum','prelu','log_softmax','dropout','silu','leaky_relu','elu','avg_pool1d','max_pool3d','max_pool2d','max_pool1d','depthwise_conv2d','conv_transpose3d','conv_transpose2d','conv_transpose1d','conv3d','conv2d','conv1d','lazy_linear','bilinear','matmul','index_input','embedding','input','linear','relu','tanh','sigmoid','gelu','softmax','layer_norm','concat','add','output'} | TRANSFORMS | REVIEWED_NEW_POOLING | {'stack','permute'}

if __package__:
    from .loss_configuration import validate_recipe as validate_objective_recipe, resolve_objective, capability as objective_capability
    from .loss_prepared_data import validate_target_mapping
    from .loss_objectives import create_objective, evaluate_objective
    from .loss_partition import objective_partition_contribution, finish_objective_partition, preflight_objective_training_batches
else:
    from loss_configuration import validate_recipe as validate_objective_recipe, resolve_objective, capability as objective_capability
    from loss_prepared_data import validate_target_mapping
    from loss_objectives import create_objective, evaluate_objective
    from loss_partition import objective_partition_contribution, finish_objective_partition, preflight_objective_training_batches


def _objective_context(value):
    if value is None:return None
    if type(value) is not dict or set(value)!={'recipe','targetMapping'}:raise ValueError('Objective context requires exact recipe and typed targetMapping.')
    recipe=validate_objective_recipe(value['recipe'])
    return {'recipe':recipe,'targetMapping':validate_target_mapping(value['targetMapping'],recipe)}

TRAINING_STORAGE_BYTES=343000000
_GRADIENT_OBSERVER_ELEMENTS=65536
def _objective_prediction_rows(inputs, outputs, targets, raw, *, include_raw, decode_axis, header, byte_limit=1_000_000):
    """Measure requested JSON before constructing full nested prediction arrays.

    Shape admission precedes decoding/conversion. Numeric spelling is measured in
    bounded chunks; neither pass truncates or silently omits a requested row.
    """
    count=min(200,len(inputs))
    shapes={'input':tuple(inputs.shape[1:]),'output':tuple(outputs.shape[1:]),
            'targets':{key:tuple(value.shape[1:]) for key,value in targets.items()}}
    if include_raw:shapes['objectiveOutput']=tuple(raw.shape[1:])
    if decode_axis is not None:
        shape=list(outputs.shape[1:]);del shape[decode_axis]
        shapes['decoded']=tuple(shape)
    refusal='Prediction preview exceeds its 1 MB artifact budget. Turn off prediction-row capture or reduce the output shape.'
    def array_structure(shape):
        elements=math.prod(shape)
        if not shape:return 0
        arrays=1;prefix=1
        for dimension in shape[:-1]:prefix*=dimension;arrays+=prefix
        return 2*arrays+max(0,elements-1)
    def skeleton(value):
        return {key:skeleton(item) for key,item in value.items()} if isinstance(value,dict) else 0
    # Scalar placeholders contribute one byte each. Replace them with array
    # punctuation and at least one byte per actual scalar using shape only.
    def minimum_extra(value):
        if isinstance(value,dict):return sum(minimum_extra(item) for item in value.values())
        return array_structure(value)+math.prod(value)-1
    row_base=len(json.dumps(skeleton(shapes),sort_keys=True,separators=(',',':')).encode())
    header_bytes=len(json.dumps({**header,'predictionRows':[]},sort_keys=True,separators=(',',':'),allow_nan=False).encode())
    minimum=header_bytes+max(0,count-1)+count*(row_base+minimum_extra(shapes))
    if minimum>byte_limit:raise ValueError(refusal)
    tensors={'input':inputs,'output':outputs,'targets':targets}
    if include_raw:tensors['objectiveOutput']=raw
    if decode_axis is not None:tensors['decoded']=outputs[:count].argmax(decode_axis+1 if decode_axis>=0 else decode_axis)
    def numeric_extra(value):
        if isinstance(value,dict):return sum(numeric_extra(item) for item in value.values())
        flat=value[:count].detach().reshape(-1)
        extra=0
        for start in range(0,flat.numel(),4096):
            for number in flat[start:start+4096].cpu().tolist():
                extra+=len(json.dumps(number,allow_nan=False,separators=(',',':')))-1
        return extra
    if minimum+numeric_extra(tensors)>byte_limit:raise ValueError(refusal)
    def materialize(value,row):
        if isinstance(value,dict):return {key:materialize(item,row) for key,item in value.items()}
        return value[row].detach().cpu().tolist()
    return [materialize(tensors,row) for row in range(count)]


TRAINING_ARTIFACT_LIMITS={'graph.json':2000000,'metrics.json':4000000,'model.py':2000000,
    'checkpoint.pt':100000000,'activations.json':2000000,'predictions.json':1000000,'manifest.json':20000000,
    'gradient-summary.json':8000000,'composite/source.json':2000000,'training_recipe.py':8192,'rate-trace.json':4000000,'objective_recipe.py':131072,'initialization_recipe.py':131072,'gradient_policy.py':131072,'trainability_recipe.py':131072,'derivative-results.json':8388608}


def _pooling_kwargs(kind, config):
    if kind.startswith('adaptive_'):
        arguments={'output_size':torch_output_size(kind,config)}
        if 'return_indices' in config:arguments['return_indices']=config['return_indices']
        return arguments
    return config


def _execute_reviewed_pooling(value, kind, arguments, sample, output_sample):
    """Strict dense float32 adapter with an explicit differentiable device policy."""
    if kind not in ('avg_pool2d','adaptive_avg_pool1d','adaptive_avg_pool2d','adaptive_max_pool2d'):
        raise ValueError('Unknown reviewed pooling adapter.')
    if not isinstance(value,torch.Tensor) or value.layout != torch.strided or value.dtype != torch.float32:
        raise ValueError(f'{kind} requires dense float32 values.')
    if value.ndim != len(sample)+1 or value.shape[0]<=0 or tuple(value.shape[1:]) != tuple(sample):
        raise ValueError(f'{kind} requires a nonempty batch with the exact authored sample shape.')
    output_elements=math.prod(output_sample)*(2 if arguments.get('return_indices',False) else 1)
    if value.numel()>100_000_000 or value.shape[0]>100_000_000//output_elements:
        raise ValueError(f'{kind} exceeds the input/output allocation budget; reduce batch or output sizes.')
    if not torch.isfinite(value).all().item():
        raise ValueError(f'{kind} requires finite input values.')
    fallback=(value.device.type=='mps' and kind in ('adaptive_avg_pool1d','adaptive_avg_pool2d')
              and any(a % b != 0 for a,b in zip(sample[1:],output_sample[1:])))
    if fallback:
        warnings.warn(f'{kind}: CPU fallback for nondivisible adaptive average bins; values and gradients return to MPS.',RuntimeWarning,stacklevel=2)
    # Explicit shape policy, never a catch-all exception fallback. Device copies
    # remain differentiable so influence reaches the original input/model.
    result=getattr(torch.nn.functional,kind)(value.to('cpu') if fallback else value,**arguments)
    if fallback:result=result.to(value.device)
    return result



def _execute_tensor_composition(left,right,operation,config,left_sample,right_sample,output_sample):
    """Batch-independent dense composition with explicit sample broadcasting."""
    if operation not in ('concat','add','stack','permute'):
        raise ValueError('Unknown tensor composition adapter.')
    unary=operation=='permute'
    sources=[left] if unary else [left,right]
    samples=[left_sample] if unary else [left_sample,right_sample]
    for value,sample in zip(sources,samples):
        if not isinstance(value,torch.Tensor) or value.layout!=torch.strided or value.dtype not in (torch.float32,torch.int64):
            raise ValueError(f'{operation} requires dense float32 or int64 values.')
        if value.ndim!=len(sample)+1 or value.shape[0]<=0 or tuple(value.shape[1:])!=tuple(sample):
            raise ValueError(f'{operation} requires a nonempty batch and the exact authored sample shape.')
        if value.numel()>100_000_000:
            raise ValueError(f'{operation} exceeds its input allocation budget.')
    if not unary and (left.dtype!=right.dtype or left.device!=right.device or left.shape[0]!=right.shape[0]):
        raise ValueError('Composition requires matching types, devices and observation batches; observations never broadcast.')
    if left.shape[0]>100_000_000//math.prod(output_sample):
        raise ValueError(f'{operation} exceeds its output allocation budget; reduce batch or sample dimensions.')
    if any(value.dtype==torch.float32 and not torch.isfinite(value).all().item() for value in sources):
        raise ValueError(f'{operation} requires finite input values, including a zero-multiplied source.')
    if operation=='concat':result=torch.cat((left,right),dim=config['dim'])
    elif operation=='stack':result=torch.stack((left,right),dim=config['dim'])
    elif operation=='permute':result=left.permute(config['dims'])
    else:
        if left.dtype==torch.int64 and (config['alpha']!=1 or config['sample_broadcast']):
            raise ValueError('Integer Add permits only multiplier 1 and equal shapes.')
        if config['sample_broadcast']:
            count=len(output_sample)
            left=left.reshape((left.shape[0],)+(1,)*(count-len(left_sample))+tuple(left_sample))
            right=right.reshape((right.shape[0],)+(1,)*(count-len(right_sample))+tuple(right_sample))
        result=torch.add(left,right,alpha=1 if left.dtype==torch.int64 else config['alpha'])
    if result.dtype==torch.float32 and not torch.isfinite(result).all().item():
        raise ValueError(f'{operation} produced nonfinite values; reduce the multiplier or input magnitude.')
    return result

def validate(graph, registry=None, *, include_layouts=False, include_types=False, bundled_dataset=True, objective_context=None, training_required=True):
    """Validate before constructing parameters; return topological order and widths."""
    objective_context=_objective_context(objective_context)
    if type(training_required) is not bool:raise ValueError('Choose an explicit training validation requirement.')
    if type(bundled_dataset) is not bool:raise ValueError('Choose an explicit dataset validation context.')
    registry = json.loads(Path(registry or REGISTRY).read_text())
    definitions = {c['id']: c for c in registry['components']}
    if not isinstance(graph,dict) or type(graph.get('schemaVersion')) is not int or graph.get('schemaVersion') not in (1,2):
        raise ValueError('Unsupported graph schema; expected version 1 or 2.')
    nodes,edges = graph.get('nodes'),graph.get('edges')
    if not isinstance(nodes,list) or not 2 <= len(nodes) <= 128 or not isinstance(edges,list) or len(edges)>256:
        raise ValueError('Graph requires 2–128 nodes and at most 256 connections.')
    by_id={}; widths={}; incoming={}; successors={}
    for node in nodes:
        if not isinstance(node,dict): raise ValueError('Malformed component.')
        ident=node.get('id'); kind=node.get('type')
        if not isinstance(ident,str) or not ident or len(ident)>128 or ident in by_id or '.' in ident:
            raise ValueError('Component IDs must be unique nonempty strings without dots (maximum 128 characters).')
        if kind not in SUPPORTED or kind not in definitions: raise ValueError(f'Unsupported component type: {kind}')
        if kind in ('bag_input','embedding_bag'):
            if graph['schemaVersion']!=2:
                raise ValueError('Bag input and EmbeddingBag require graph schema2; migrate this graph before connecting bag sources.')
            expected_inputs=[] if kind=='bag_input' else ['indices','offsets','per_sample_weights']
            expected_outputs=['indices','offsets','per_sample_weights'] if kind=='bag_input' else ['out']
            if definitions[kind].get('inputs')!=expected_inputs or definitions[kind].get('outputs')!=expected_outputs:
                raise ValueError('Bag registry must declare exactly its reviewed named input/output ceilings.')
        config=node.get('config',{})
        if not isinstance(config,dict): raise ValueError(f'{ident}: malformed configuration.')
        config=effective_config(node,graph['schemaVersion'])
        for position in ('x','y'):
            if position in node and (isinstance(node[position],bool) or not isinstance(node[position],(int,float)) or not math.isfinite(node[position])): raise ValueError(f'{ident}: position must be finite numeric coordinates.')
        for field in definitions[kind]['fields']:
            if field.get('kind')=='text':
                minimum,maximum=field.get('minItems'),field.get('maxItems')
                if (type(minimum) is not int or type(maximum) is not int or not 1<=minimum<=maximum<=4096
                    or any(field.get(key) is not None for key in ('min','max','choices')) or field.get('allowsNull',False) is not False
                    or type(field.get('defaultValue')) is not str or not minimum<=len(field['defaultValue'])<=maximum):raise ValueError(f'{ident}: malformed text registry field.')
                if type(config.get(field['key'])) is not str or not minimum<=len(config[field['key']])<=maximum:raise ValueError(f'{ident}: text requires its declared character bounds.')
                continue
            if field.get('kind')=='choice':
                choices=field.get('choices')
                if (type(choices) is not list or not 1<=len(choices)<=64
                        or any(type(option) is not dict or set(option)!={'value','title'}
                               or any(type(option[key]) is not str or not option[key].strip() or len(option[key])>128 for key in ('value','title')) for option in choices)
                        or any(field.get(key) is not None for key in ('min','max','minItems','maxItems'))
                        or field.get('allowsNull',False) is not False):
                    raise ValueError(f'{ident}: malformed categorical registry field.')
                allowed=[option['value'] for option in choices]
                if len(set(allowed))!=len(allowed) or type(field.get('defaultValue')) is not str or field['defaultValue'] not in allowed:
                    raise ValueError(f'{ident}: categorical choices need unique values and a declared default.')
                if type(config.get(field['key'])) is not str or config[field['key']] not in allowed:
                    raise ValueError(f"{ident}: {field['title']} must match a declared categorical choice.")
                continue
            if field.get('kind')=='boolean': continue
            value=config.get(field['key'],field['defaultValue'])
            if value is None and field.get('allowsNull') is True and field.get('kind','integer') in ('integer','number','integerList'):continue
            if field.get('kind')=='integerList':
                if type(value) is not list or not field['minItems']<=len(value)<=field['maxItems'] or any(type(v) is not int or not field['min']<=v<=field['max'] for v in value):
                    raise ValueError(f"{ident}: {field['title']} must be an integer list within its declared length and element bounds.")
                continue
            if field.get('kind')=='number':
                if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not field['min']<=value<=field['max']:
                    raise ValueError(f"{ident}: {field['title']} must be a finite number from {field['min']} to {field['max']}.")
                continue
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or int(value)!=value or not field['min']<=value<=field['max']:
                raise ValueError(f"{ident}: {field['title']} must be an integer from {field['min']} to {field['max']}.")
        by_id[ident]=node; incoming[ident]={}; successors[ident]=[]
    if sum(n['type'] in ('input','index_input','bag_input') for n in nodes)!=1 or sum(n['type']=='output' for n in nodes)!=1:
        raise ValueError('This flow requires exactly one Input and one Output.')
    def input_ports(ident):
        node=by_id[ident]
        if node['type']=='embedding_bag':return bag_input_ports(effective_config(node,graph['schemaVersion']))
        return attention_inputs(node['type'],effective_config(node,graph['schemaVersion'])) if node['type'] in ATTENTION_OPERATIONS else recurrent_inputs(node['type'],effective_config(node,graph['schemaVersion'])) if node['type'] in RECURRENT_OPERATIONS else routing_inputs(node['type'],effective_config(node,graph['schemaVersion'])) if node['type'] in ROUTING_OPERATIONS else definitions[node['type']]['inputs']
    edge_ids=set()
    for edge in edges:
        if not isinstance(edge,dict): raise ValueError('Malformed connection.')
        eid=edge.get('id'); s=edge.get('source'); t=edge.get('target'); sp=edge.get('sourcePort'); tp=edge.get('targetPort')
        if not isinstance(eid,str) or not eid or eid in edge_ids: raise ValueError('Connection IDs must be unique and nonempty.')
        edge_ids.add(eid)
        if s not in by_id or t not in by_id: raise ValueError('Connection references a missing component.')
        source_outputs=bag_source_layouts(s,effective_config(by_id[s],graph['schemaVersion'])) if by_id[s]['type']=='bag_input' else definitions[by_id[s]['type']]['outputs']
        if sp not in source_outputs or tp not in input_ports(t):
            raise ValueError('Connection references an invalid port.')
        if tp in incoming[t]: raise ValueError(f'{t}: input port {tp} already connected.')
        incoming[t][tp]=OutputAddress(s,sp); successors[s].append(t)
    for ident,node in by_id.items():
        if set(incoming[ident])!=set(input_ports(ident)): raise ValueError(f'{ident}: connect every required input.')
        if node['type']!='output' and not successors[ident]: raise ValueError(f'{ident}: disconnected component; connect it to the output.')
    pending={i:len(p) for i,p in incoming.items()}; ready=[i for i,n in pending.items() if n==0]; order=[]
    while ready:
        ident=ready.pop(0); order.append(ident)
        for t in successors[ident]:
            pending[t]-=1
            if pending[t]==0: ready.append(t)
    if len(order)!=len(nodes): raise ValueError('Cycles are unsupported. Remove the circular connection.')
    parameters=0; frozen_parameters=0; layouts={}; dtypes={}
    for ident in order:
        node=by_id[ident]; kind=node['type']
        settings=effective_config(node,graph['schemaVersion'])
        inputs=[select_output(layouts,incoming[ident][p]) for p in input_ports(ident)]
        input_types=[select_output(dtypes,incoming[ident][p]) for p in input_ports(ident)]
        dtype=input_types[0] if input_types else 'float32'
        if kind!='embedding_bag' and any(isinstance(value,BagCoordinateLayout) for value in inputs):
            raise ValueError(f'{ident}: ordinary tensor operators cannot consume bag-coordinate source roles.')
        if kind=='bag_input':
            layouts[ident]=bag_source_layouts(ident,settings)
            dtypes[ident]={role:'float32' if role=='per_sample_weights' else 'int64' for role in layouts[ident]}
            continue
        if (kind in ATTENTION_OPERATIONS or kind in RECURRENT_OPERATIONS or kind in STATEFUL_NORMALIZATION_OPERATIONS or kind in SPATIAL_PATCH_OPERATIONS or kind in REVIEWED_NEW_POOLING or kind in ('prelu','log_softmax','dropout','silu','leaky_relu','elu','avg_pool1d','max_pool3d','max_pool2d','max_pool1d','depthwise_conv2d','conv_transpose3d','conv_transpose2d','conv_transpose1d','conv3d','conv2d','conv1d','bilinear','linear','lazy_linear','relu','tanh','sigmoid','gelu','softmax','layer_norm')) and dtype!='float32':
            raise ValueError(f'{ident}: {kind} requires floating-point features, not integer indices.')
        if kind in ('bilinear','add','concat','stack','matmul') and len(set(input_types))!=1:
            raise ValueError(f'{ident}: mixed tensor types require an explicit conversion.')
        if kind in ATTENTION_OPERATIONS:
            required_types=['float32' if p in ATTENTION_FLOAT_PORTS[kind] else 'int64' for p in input_ports(ident)]
            if input_types!=required_types:raise ValueError('Attention inputs require their declared float32 references and int64 masks.')
        if kind in RECURRENT_OPERATIONS and any(t!='float32' for t in input_types):raise ValueError('Recurrent inputs and states all require float32.')
        if kind in ROUTING_OPERATIONS:
            required_types=['float32','int64'] if kind=='gather' else ['float32']*len(inputs)
            if input_types!=required_types:raise ValueError(f'{ident}: routing requires its declared floating sources and integer selectors.')
        if kind=='embedding_bag':
            names=bag_input_ports(settings)
            if any(not isinstance(value,BagCoordinateLayout) for value in inputs):
                raise ValueError('EmbeddingBag requires named bag-coordinate source inputs.')
            first=inputs[0]
            if (any(value.source_id!=first.source_id or value.role!=name for name,value in zip(names,inputs))
                    or any((value.layout,value.maximum_items_per_bag,value.per_sample_weights,value.include_last_offset)!=(first.layout,first.maximum_items_per_bag,first.per_sample_weights,first.include_last_offset) for value in inputs)
                    or (first.layout,first.per_sample_weights,first.include_last_offset)!=(settings['layout'],settings['per_sample_weights'],settings['include_last_offset'])):
                raise ValueError('EmbeddingBag inputs require coherent roles from one matching bag source.')
            shape=BatchedLayout((settings['embedding_dim'],));dtype='float32'
            count=settings['num_embeddings']*settings['embedding_dim'];parameters+=count
            if settings['freeze']:frozen_parameters+=count
        elif kind=='index_input':
            shape=BatchedLayout(tuple(settings['shape']));dtype='int64'
        elif kind=='embedding':
            if dtype!='int64':raise ValueError(f'{ident}: embedding requires integer indices.')
            if settings['sparse'] and settings['scale_grad_by_freq']:raise ValueError('Embedding sparse gradients do not support frequency scaling.')
            shape=BatchedLayout(inputs[0].sample+(settings['embedding_dim'],));dtype='float32'
            parameters+=settings['num_embeddings']*settings['embedding_dim']
            if settings['freeze']:frozen_parameters+=settings['num_embeddings']*settings['embedding_dim']
        elif kind=='input':
            shape=BatchedLayout((settings['features'],))
            if bundled_dataset and shape.sample!=(2,): raise ValueError('Bundled two-moons data has exactly 2 input features.')
        elif kind=='linear':
            actual=inputs[0].sample[-1]
            if actual!=settings['in_features']: raise ValueError(f"{ident}: receives {actual} features but expects {settings['in_features']}. Change Input features or insert a projection.")
            shape=inputs[0].linear(settings['in_features'],settings['out_features'])
            parameters+=(actual+int(settings['bias']))*settings['out_features']
        elif kind in POOLING_RESOLVERS:
            shape=POOLING_RESOLVERS[kind](settings,inputs[0])['layout']
        elif kind in TRANSPOSE_CONVOLUTION_RESOLVERS:
            resolved=TRANSPOSE_CONVOLUTION_RESOLVERS[kind](settings,inputs[0])
            shape=resolved['layout'];parameters+=resolved['parameterCount']
        elif kind in CONVOLUTION_RESOLVERS:
            resolved=CONVOLUTION_RESOLVERS[kind](settings,inputs[0])
            shape=resolved['layout'];parameters+=resolved['parameterCount']
        elif kind=='lazy_linear':
            resolved=resolve_lazy_linear(settings,inputs[0])
            shape=resolved['layout'];parameters+=resolved['parameterCount']
        elif kind=='bilinear':
            shape=inputs[0].bilinear(inputs[1],settings['in1_features'],settings['in2_features'],settings['out_features'])
            parameters+=settings['out_features']*(settings['in1_features']*settings['in2_features']+int(settings['bias']))
        elif kind=='layer_norm':
            shape=inputs[0].layer_norm(settings['normalized_shape'])
            if settings['elementwise_affine']:
                parameters+=math.prod(settings['normalized_shape'])*(1+int(settings['bias']))
        elif kind in ATTENTION_OPERATIONS:
            attention=resolve_attention(kind,settings,inputs);shape=next(iter(attention['outputs'].values()));parameters+=attention['parameterCount']
        elif kind in RECURRENT_OPERATIONS:
            recurrent=resolve_recurrent(kind,settings,inputs);shape=recurrent['outputs']['out'];parameters+=recurrent['parameterCount']
        elif kind in STATEFUL_NORMALIZATION_OPERATIONS:
            resolved=resolve_stateful_normalization(kind,settings,inputs[0]);shape=resolved['layout'];parameters+=resolved['parameterCount']
        elif kind in SPATIAL_PATCH_OPERATIONS:
            shape=resolve_spatial_patch(kind,settings,inputs[0])['layout']
        elif kind in ROUTING_OPERATIONS:
            routing=resolve_tensor_routing(kind,settings,inputs);shape=next(iter(routing['outputs'].values()))
        elif kind in ('prelu','log_softmax','dropout'):
            resolved=resolve_remaining_activation(kind,settings,inputs[0]);shape=resolved['layout'];parameters+=resolved['parameterCount']
        elif kind in ('silu','leaky_relu','elu'): shape=resolve_activation(kind,settings,inputs[0])['layout']
        elif kind=='relu': shape=resolve_relu(settings,inputs[0])['layout']
        elif kind=='tanh': shape=resolve_tanh(settings,inputs[0])['layout']
        elif kind=='sigmoid': shape=resolve_sigmoid(settings,inputs[0])['layout']
        elif kind=='softmax': shape=inputs[0].softmax(settings['dim'])
        elif kind=='reshape': shape=inputs[0].reshape(settings['shape'])
        elif kind=='transpose': shape=inputs[0].transpose(settings['dim0'],settings['dim1'])
        elif kind=='flatten': shape=inputs[0].flatten(settings['start_dim'],settings['end_dim'])
        elif kind in ('concat','add','stack','permute'): shape=resolve_tensor_composition(kind,settings,inputs[0],inputs[1] if len(inputs)>1 else None,dtype=dtype)
        elif kind=='matmul':
            if dtype!='float32':raise ValueError(f'{ident}: matrix multiplication requires floating-point features.')
            shape=inputs[0].matmul(inputs[1])
        else: shape=inputs[0]
        if shape.sample[-1]>4096: raise ValueError('Intermediate width exceeds the POC limit of 4096.')
        if kind=='output':
            if objective_context is None:
                if shape.sample!=(2,):raise ValueError('Classification output must have exactly 2 features.')
            else:
                if dtype!='float32':raise ValueError('Explicit objective prediction requires float32 output.')
                recipe=objective_context['recipe'];roles={r['role']:{'dtype':r['dtype'],'sampleShape':r['sampleShape']} for r in objective_context['targetMapping']['roles']}
                resolve_objective(recipe['componentID'],recipe['config'],shape,roles)
        if kind in ATTENTION_OPERATIONS:
            layouts[ident]=attention['outputs'];dtypes[ident]=attention['outputDTypes']
        elif kind in RECURRENT_OPERATIONS:
            layouts[ident]=recurrent['outputs'];dtypes[ident]={port:'float32' for port in recurrent['outputs']}
        elif kind in ROUTING_OPERATIONS:
            outputs=routing['outputs']
            if any(v.sample[-1]>4096 for v in outputs.values()):raise ValueError('Intermediate width exceeds the POC limit of 4096.')
            layouts[ident]=outputs if kind in ('split','chunk') else outputs['out']
            dtypes[ident]={port:'float32' for port in outputs} if kind in ('split','chunk') else 'float32'
        elif kind in POOLING_RESOLVERS and settings.get('return_indices',False):
            layouts[ident]={'out':shape,'indices':shape};dtypes[ident]={'out':dtype,'indices':'int64'}
        else:
            layouts[ident]=shape;dtypes[ident]=dtype
        if len(shape.sample)==1: widths[ident]=shape.sample[0]
    if training_required and parameters - frozen_parameters == 0: raise ValueError('Add a component with trainable parameters before running.')
    if parameters>2_000_000: raise ValueError('Model exceeds the POC limit of 2 million parameters.')
    if "compositionPlan" in graph:
        validate_plan(graph["compositionPlan"],graph,{i:{"inputs":input_ports(i),"outputs":list(layouts[i]) if isinstance(layouts[i],dict) else (["out"] if by_id[i]["type"]!="output" else [])} for i in order})
    if include_types:return order,widths,layouts,dtypes
    return (order,widths,layouts) if include_layouts else (order,widths)


def _execute_remaining_activation(value,operation,config,sample,layer=None):
    if not isinstance(value,torch.Tensor) or value.layout!=torch.strided or value.dtype!=torch.float32:
        raise ValueError('Activation requires dense float32 values.')
    if value.ndim!=len(sample)+1 or value.shape[0]<=0 or tuple(value.shape[1:])!=tuple(sample):
        raise ValueError('Activation requires a nonempty batch with its exact authored sample shape.')
    if value.numel()>100_000_000:raise ValueError('Activation exceeds its input/output allocation budget.')
    if not torch.isfinite(value).all().item():raise ValueError('Activation requires finite input values.')
    if operation=='log_softmax':result=torch.nn.functional.log_softmax(value,dim=config['dim'])
    elif operation in ('prelu','dropout'):
        if operation=='prelu':
            if layer.weight.dtype!=value.dtype or layer.weight.device!=value.device:raise ValueError('PReLU slopes and inputs require matching float32 devices.')
            if not torch.isfinite(layer.weight).all().item():raise ValueError('PReLU requires finite learned slopes.')
        result=layer(value)
    else:raise ValueError('Unknown remaining activation.')
    if not torch.isfinite(result).all().item():raise ValueError('Activation produced nonfinite values; reduce input or coefficient magnitude.')
    return result


def _normalization_layer(operation,config):
    cls=STATEFUL_NORMALIZATION_LAYERS[operation];kwargs={key:config[key] for key in ('eps','affine','bias')}
    if operation=='group_norm':return cls(config['num_groups'],config['num_channels'],**kwargs)
    return cls(config['num_features'],momentum=None if config.get('averaging')=='cumulative' else config['momentum'],track_running_stats=config['track_running_stats'],**kwargs)


def _validate_normalization_state(layer,operation,config):
    channels=config.get('num_features',config.get('num_channels'));expected={}
    if config['affine']:
        expected['weight']=('float32',(channels,))
        if config['bias']:expected['bias']=('float32',(channels,))
    if config.get('track_running_stats',False):expected.update(running_mean=('float32',(channels,)),running_var=('float32',(channels,)),num_batches_tracked=('int64',()))
    actual=layer.state_dict()
    if set(actual)!=set(expected):raise ValueError('Normalization state differs from configured parameters and buffers.')
    for key,(dtype,shape) in expected.items():
        value=actual[key]
        if value.layout!=torch.strided or str(value.dtype).removeprefix('torch.')!=dtype or tuple(value.shape)!=shape or not bool(torch.isfinite(value).all()):raise ValueError('Normalization state requires exact finite typed parameters/buffers.')
        if key=='running_var' and bool((value<0).any()):raise ValueError('Normalization running variance cannot be negative.')
        if key=='num_batches_tracked' and (value.item()<0 or value.item()>1_000_000_000 or (operation.startswith('instance') and value.item()!=0)):raise ValueError('Normalization counter differs from its bounded tracking policy.')


def _execute_stateful_normalization(value,operation,config,sample,layer):
    if not isinstance(value,torch.Tensor) or value.layout!=torch.strided or value.dtype!=torch.float32 or value.ndim!=len(sample)+1 or value.shape[0]<=0 or tuple(value.shape[1:])!=tuple(sample):raise ValueError('Normalization needs dense float32 values with its declared observation/channel/spatial layout.')
    if value.numel()>100_000_000:raise ValueError('Normalization exceeds the100-million source/output budget.')
    current=operation=='group_norm' or layer.training or not config.get('track_running_stats',False)
    spatial=math.prod(sample[1:])
    if current and operation.startswith('batch') and value.shape[0]*spatial<=1:raise ValueError('Batch normalization needs more than one value per channel; increase batch size or spatial positions without dropping observations.')
    if current and operation.startswith('instance') and spatial<=1:raise ValueError('Instance normalization needs more than one spatial position; add spatial positions or use saved tracked evaluation.')
    _validate_normalization_state(layer,operation,config)
    if layer.training and operation.startswith('batch') and config.get('track_running_stats',False) and layer.num_batches_tracked.item()>=1_000_000_000:raise ValueError('Normalization tracked-call counter is at its supported limit.')
    if not bool(torch.isfinite(value).all()):raise ValueError('Normalization source must be finite; inspect feature scaling.')
    result=layer(value)
    _validate_normalization_state(layer,operation,config)
    if tuple(result.shape)!=tuple(value.shape) or not bool(torch.isfinite(result).all()):raise ValueError('Normalization output/statistics are nonfinite; revise feature scaling or epsilon.')
    return result


def _preflight_normalization_batches(model,train_rows,validation_rows,batch_size):
    for ident in model.order:
        node=model.nodes[ident];kind=node['type']
        if kind not in STATEFUL_NORMALIZATION_OPERATIONS:continue
        c=node['config'];sample=select_output(model.layouts,model.incoming[ident]['in']).sample;spatial=math.prod(sample[1:])
        for label,rows,size,current in [('Training',train_rows,batch_size,True),('Validation',validation_rows,1024,not c.get('track_running_stats',False))]:
            if not current:continue
            if kind.startswith('instance') and spatial<=1:raise ValueError(f'{label} {ident}: instance normalization needs more than one spatial position; add positions without skipping observations.')
            if kind.startswith('batch') and spatial<=1 and (min(size,rows)<=1 or rows%size==1):raise ValueError(f'{label} {ident}: the actual last minibatch has one value per channel; change batch size or add spatial positions without dropping observations.')


def _execute_spatial_patch(value,operation,config,sample,output_sample,work):
    if not isinstance(value,torch.Tensor) or value.layout!=torch.strided or value.dtype!=torch.float32 or tuple(value.shape[1:])!=tuple(sample) or value.ndim!=len(sample)+1 or value.shape[0]<=0:raise ValueError('Spatial patches require a dense float32 array with its declared observation/sample shape.')
    batch=value.shape[0]
    if value.numel()>100_000_000 or batch>100_000_000//math.prod(output_sample) or batch>100_000_000//work:raise ValueError('Spatial patches exceed the runtime source/output/extraction budget.')
    if not bool(torch.isfinite(value).all()):raise ValueError('Spatial patches require finite source values.')
    kwargs={key:tuple(config[key]) for key in ('kernel_size','dilation','padding','stride')}
    result=torch.nn.functional.unfold(value,**kwargs) if operation=='unfold' else torch.nn.functional.fold(value,output_size=tuple(config['output_size']),**kwargs)
    if tuple(result.shape)!=(batch,*output_sample) or not bool(torch.isfinite(result).all()):raise ValueError('Spatial patch output differs from its finite declared geometry.')
    return result


def _execute_tensor_routing(sources,operation,config,samples,outputs,work):
    if operation not in ('split','chunk','gather','einsum') or len(sources)!=len(samples):raise ValueError('Unknown routing adapter or source count.')
    batch=None;device=None
    for i,(value,sample) in enumerate(zip(sources,samples)):
        dtype=torch.int64 if operation=='gather' and i==1 else torch.float32
        if not isinstance(value,torch.Tensor) or value.layout!=torch.strided or value.dtype!=dtype:raise ValueError('Routing requires dense float32 sources and int64 Gather selectors.')
        if value.ndim!=len(sample)+1 or value.shape[0]<=0 or tuple(value.shape[1:])!=tuple(sample):raise ValueError('Routing requires nonempty observations and exact authored sample shapes.')
        if value.numel()>100_000_000:raise ValueError('Routing exceeds its source allocation budget.')
        if batch is not None and (value.shape[0]!=batch or value.device!=device):raise ValueError('Routing observations and devices must match without broadcasting.')
        batch,device=value.shape[0],value.device
    if batch>100_000_000//sum(math.prod(s) for s in outputs.values()):raise ValueError('Routing exceeds its output allocation budget.')
    if work and batch>100_000_000//work:raise ValueError('Einsum exceeds its contraction work budget.')
    if any(value.dtype==torch.float32 and not torch.isfinite(value).all().item() for value in sources):raise ValueError('Routing requires finite values in every source, including unused parts.')
    if operation=='split':result=torch.split(sources[0],config['split_size'] if config['mode']=='size' else config['sections'],dim=config['dim'])
    elif operation=='chunk':result=torch.chunk(sources[0],config['chunks'],dim=config['dim'])
    elif operation=='gather':
        index=sources[1];extent=sources[0].shape[config['dim']]
        if not ((index>=0)&(index<extent)).all().item():raise ValueError('Gather selectors must be nonnegative positions within the selected source extent.')
        result=torch.gather(sources[0],config['dim'],index,sparse_grad=False)
    else:result=torch.einsum(config['equation'],*sources)
    values=result if isinstance(result,tuple) else (result,)
    if len(values)!=len(outputs) or any(tuple(v.shape)!=(batch,*sample) for v,sample in zip(values,outputs.values())):raise ValueError('Native routing outputs differ from their declared ordered ports.')
    if any(not torch.isfinite(v).all().item() for v in values):raise ValueError('Routing produced nonfinite values; reduce source magnitudes.')
    return result


def _validate_recurrent_model_state(state,expected):
    """Exact typed recurrent composed-state admission; never rely on Torch casting."""
    if not isinstance(state,dict) or set(state)!=set(expected):raise ValueError('Recurrent model state keys differ from the captured architecture.')
    for key,wanted in expected.items():
        value=state[key]
        if not isinstance(value,torch.Tensor) or value.layout!=torch.strided or tuple(value.shape)!=tuple(wanted.shape) or value.dtype!=wanted.dtype or not torch.isfinite(value).all().item():raise ValueError('Recurrent model state shapes, dtypes and finite dense values must match exactly.')


def _recurrent_kwargs(operation,config):
    result={k:v for k,v in config.items() if k!='state_mode'}
    if not operation.endswith('_cell'):result['batch_first']=True
    return result


def _execute_recurrent(sources,operation,config,samples,outputs,gate_elements,work,layer):
    """Self-contained strict explicit-state boundary, also embedded in portable exports."""
    if not sources or any(not isinstance(v,torch.Tensor) for v in sources):raise ValueError('Recurrent execution requires tensor inputs and states.')
    cell=operation.endswith('_cell');lstm=operation.startswith('lstm');batch=sources[0].shape[0] if sources[0].ndim else 0
    if operation=='lstm' and config.get('proj_size',0) and sources[0].device.type=='mps':raise ValueError('Projected LSTM requires CPU; choose CPU or projection width zero before running.')
    names=['in'] if config['state_mode']=='zero' else ['in','hidden','cell'] if lstm else ['in','hidden']
    if len(sources)!=len(names) or len(samples)!=len(names) or batch<=0:raise ValueError('Recurrent execution requires every declared input/state and a nonempty batch.')
    for value,sample in zip(sources,samples):
        if value.layout!=torch.strided or value.dtype!=torch.float32 or value.device!=sources[0].device or tuple(value.shape)!=(batch,*sample) or value.numel()>100_000_000:raise ValueError('Recurrent input/state must be finite dense float32 on one device with exact observation-first shapes.')
    physical=sum(math.prod(sample) for key,sample in outputs.items() if not(cell and key=='hidden'))
    if any(batch>100_000_000//extent for extent in (physical,gate_elements,work)):raise ValueError('Recurrent source/output, gate storage or recurrence work exceeds its allocation budget.')
    if any(not torch.isfinite(v).all().item() for v in sources):raise ValueError('Recurrent input/state contains nonfinite values.')
    expected={'rnn':'RNN','lstm':'LSTM','gru':'GRU','rnn_cell':'RNNCell','lstm_cell':'LSTMCell','gru_cell':'GRUCell'}[operation]
    if type(layer) is not getattr(torch.nn,expected):raise ValueError('Recurrent learned state belongs to a different component.')
    for key in ('input_size','hidden_size','bias','num_layers','bidirectional','dropout','nonlinearity','proj_size'):
        if key in config and getattr(layer,key,None)!=config[key]:raise ValueError('Recurrent learned-state configuration differs from the captured component.')
    if not cell and not layer.batch_first:raise ValueError('Recurrent sequence observations must remain first.')
    if any(p.dtype!=torch.float32 or p.layout!=torch.strided or p.device!=sources[0].device or not torch.isfinite(p).all().item() for p in layer.parameters()):raise ValueError('Recurrent learned parameters must be finite dense float32 on the selected device.')
    h=config['hidden_size'];p=config.get('proj_size',0) or h;g=4 if lstm else 3 if operation.startswith('gru') else 1;directions=2 if config.get('bidirectional',False) else 1
    expected_parameters=[]
    for index in range(config.get('num_layers',1)):
        width=config['input_size'] if index==0 else directions*p
        for direction in range(directions):
            suffix='' if cell else '_l'+str(index)+('_reverse' if direction else '')
            for name,shape in [('weight_ih',(g*h,width)),('weight_hh',(g*h,p))]+([('bias_ih',(g*h,)),('bias_hh',(g*h,))] if config['bias'] else [])+([('weight_hr',(p,h))] if config.get('proj_size',0) else []):expected_parameters.append((name+suffix,shape))
    if [(name,tuple(value.shape)) for name,value in layer.named_parameters()]!=expected_parameters:raise ValueError('Recurrent learned parameter ownership or shapes differ from the configured module.')

    state=None
    if len(sources)>1:
        hidden=sources[1] if cell else sources[1].permute(1,0,2).contiguous()
        if lstm:state=(hidden,sources[2] if cell else sources[2].permute(1,0,2).contiguous())
        else:state=hidden
    native=layer(sources[0],state) if state is not None else layer(sources[0])
    if cell:
        hidden,memory=native if lstm else (native,None)
        result={'out':hidden,'hidden':hidden}
        if lstm:result['cell']=memory
    else:
        sequence,state=native;hidden,memory=state if lstm else (state,None)
        result={'out':sequence,'hidden':hidden.permute(1,0,2)}
        if lstm:result['cell']=memory.permute(1,0,2)
    if set(result)!=set(outputs) or any(value.dtype!=torch.float32 or value.device!=sources[0].device or value.layout!=torch.strided or tuple(value.shape)!=(batch,*outputs[key]) or not torch.isfinite(value).all().item() for key,value in result.items()):raise ValueError('Recurrent outputs/state differ from the finite observation-first contract.')
    return result

class GraphModel(nn.Module):
    def __init__(self,graph,registry=None,*,bundled_dataset=True,objective_context=None):
        super().__init__()
        self.objective_context=_objective_context(objective_context)
        self.order,self.widths,self.layouts,self.dtypes=validate(graph,registry,include_types=True,bundled_dataset=bundled_dataset,objective_context=self.objective_context)
        self.nodes={n['id']:({**n,'config':effective_config(n,graph['schemaVersion'])} if n['type'] in ATTENTION_OPERATIONS | RECURRENT_OPERATIONS | TRANSFORMS | REVIEWED_NEW_POOLING | ROUTING_OPERATIONS | SPATIAL_PATCH_OPERATIONS | STATEFUL_NORMALIZATION_OPERATIONS | {'prelu','log_softmax','dropout','concat','add','stack','permute','silu','leaky_relu','elu','avg_pool1d','max_pool3d','max_pool2d','max_pool1d','depthwise_conv2d','conv_transpose3d','conv_transpose2d','conv_transpose1d','conv3d','conv2d','conv1d','lazy_linear','bilinear','softmax','layer_norm','embedding','index_input','bag_input','embedding_bag'} else n) for n in graph['nodes']}
        self.incoming={i:{} for i in self.order}
        for e in graph['edges']: self.incoming[e['target']][e['targetPort']]=OutputAddress(e['source'],e['sourcePort'])
        self.layers=nn.ModuleDict()
        self.layer_keys={}
        for ident in self.order:
            n=self.nodes[ident]
            if n['type'] in ATTENTION_OPERATIONS-{'attention_mask'} or n['type'] in RECURRENT_OPERATIONS or n['type'] in STATEFUL_NORMALIZATION_OPERATIONS or n['type'] in ('prelu','dropout','depthwise_conv2d','conv_transpose3d','conv_transpose2d','conv_transpose1d','conv3d','conv2d','conv1d','bilinear','linear','lazy_linear','layer_norm','embedding','embedding_bag'):
                key=ident
                if hasattr(self.layers,key):
                    key='kwip_'+hashlib.sha256(ident.encode()).hexdigest()
                    while key in self.nodes or hasattr(self.layers,key): key='_'+key
                self.layer_keys[ident]=key
                c=effective_config(n,graph['schemaVersion'])
                if n['type'] in ATTENTION_OPERATIONS:self.layers[key]=create_attention_module(n['type'],c)
                elif n['type'] in RECURRENT_OPERATIONS:self.layers[key]=RECURRENT_LAYERS[n['type']](**_recurrent_kwargs(n['type'],c))
                elif n['type'] in STATEFUL_NORMALIZATION_OPERATIONS:self.layers[key]=_normalization_layer(n['type'],c)
                elif n['type']=='prelu':self.layers[key]=nn.PReLU(c['num_parameters'],c['init'])
                elif n['type']=='dropout':self.layers[key]=nn.Dropout(c['p'],inplace=False)
                elif n['type']=='linear':self.layers[key]=nn.Linear(c['in_features'],c['out_features'],bias=c['bias'])
                elif n['type'] in TRANSPOSE_CONVOLUTION_LAYERS:
                    convolution=TRANSPOSE_CONVOLUTION_LAYERS[n['type']]
                    self.layers[key]=convolution(c['in_channels'],c['out_channels'],c['kernel_size'],stride=c['stride'],padding=c['padding'],output_padding=c['output_padding'],groups=c['groups'],bias=c['bias'],dilation=c['dilation'])
                elif n['type'] in CONVOLUTION_LAYERS:
                    if n['type']=='depthwise_conv2d':c=conv2d_config(c)
                    convolution=CONVOLUTION_LAYERS[n['type']]
                    padding=c['padding'] if c['padding_policy']=='explicit' else c['padding_policy']
                    self.layers[key]=convolution(c['in_channels'],c['out_channels'],c['kernel_size'],stride=c['stride'],padding=padding,dilation=c['dilation'],groups=c['groups'],bias=c['bias'],padding_mode=c['padding_mode'])
                elif n['type']=='lazy_linear':self.layers[key]=nn.LazyLinear(c['out_features'],bias=c['bias'])
                elif n['type']=='bilinear':self.layers[key]=nn.Bilinear(c['in1_features'],c['in2_features'],c['out_features'],bias=c['bias'])
                elif n['type']=='embedding':self.layers[key]=CheckedEmbedding(c)
                elif n['type']=='embedding_bag':self.layers[key]=CheckedEmbeddingBag(c)
                else:self.layers[key]=nn.LayerNorm(c['normalized_shape'],eps=c['eps'],elementwise_affine=c['elementwise_affine'],bias=c['bias'])
        self.output_id=next(i for i in self.order if self.nodes[i]['type']=='output')
        if 'compositionPlan' in graph:self._install_composition(graph)

    def _install_composition(self,graph):
        # Transfer each once-created operation into exactly one native leaf.
        plan=graph['compositionPlan'];self._composition_plan=json.loads(json.dumps(plan));self._composition_rows={r['id']:r for r in plan['containers']}
        self._composition_ports={i:{'inputs':list(self.incoming[i]),'outputs':list(self.layouts[i]) if isinstance(self.layouts[i],dict) else ['out']} for i in self.order}
        admitted=validate_plan(plan,graph,self._composition_ports)
        self._composition_schedules=dict(admitted.schedules)
        operations={i:self.layers[k] for i,k in self.layer_keys.items()}
        del self.layers
        self.layers={};self.layer_keys={};self._composition_containers={}
        def build(ident,path):
            row=self._composition_rows[ident];children=[]
            for child in row['children']:
                nativepath=path+('.blocks.' if row['kind']=='graph' else '.')+child['key']
                if child['kind']=='container':module=build(child['id'],nativepath)
                else:
                    node=child['id'];operation=operations.get(node);module=PrimitiveInvocation(self,node,operation)
                    if operation is not None:
                        key=nativepath+'.operation';self.layer_keys[node]=key;self.layers[key]=operation
                children.append((child['key'],module))
            if row['kind']=='graph':module=GraphContainer(self,ident,children)
            elif row['kind']=='sequential':module=nn.Sequential(__import__('collections').OrderedDict(children))
            elif row['kind']=='orderedList':module=OrderedContainer(self,ident,[v for _,v in children])
            else:module=NamedContainer(self,ident,children)
            self._composition_containers[ident]=module
            return module
        self.composition_root=build('root','composition_root')

    def _composition_leaf(self,ident,call):
        frame=call.frame;names=self._composition_ports[ident]['inputs']
        if len(call.values)!=len(names):raise ValueError('Composition invocation input arity differs from configured ports.')
        values={}
        for name,value in zip(names,call.values):
            address=self.incoming[ident][name];values.setdefault(address.node,{})[address.port]=value
        if frame.observer is not None:frame.observer('started',ident)
        result=self.execute_node(ident,values,frame.x);frame.values[ident]=result
        if frame.observer is not None:frame.observer('completed',ident)
        return Invocation(tuple(select_output({ident:result},OutputAddress(ident,p)) for p in self._composition_ports[ident]['outputs']),frame)

    def _composition_container(self,ident,call):
        row=self._composition_rows[ident];frame=call.frame
        if len(call.values)!=len(row['inputs']):raise ValueError('Composition boundary input arity differs from its admitted interface.')
        # Passed boundaries are real values, including fanout; retain named siblings.
        for entry,value in zip(row['inputs'],call.values):
            for target in entry['targets']:
                address=self.incoming[target['node']][target['port']]
                existing=frame.values.get(address.node)
                if ident=='root' and self.nodes[address.node]['type']=='bag_input':
                    if type(value) is not dict or address.port not in value:raise ValueError('Root bag invocation requires each actual named coordinate role.')
                    if not isinstance(existing,dict):raise ValueError('Root bag source must execute before its owning container.')
                    existing[address.port]=value[address.port]
                elif isinstance(existing,dict):existing[address.port]=value
                else:frame.values[address.node]=value
        container=self._composition_containers[ident]
        for index in self._composition_schedules[ident]:
            child=row['children'][index]
            if child['kind']=='node':addresses=list(self.incoming[child['id']].values())
            else:addresses=[self.incoming[e['targets'][0]['node']][e['targets'][0]['port']] for e in self._composition_rows[child['id']]['inputs']]
            packet=Invocation(tuple(select_output(frame.values,a) for a in addresses),frame)
            module=container.blocks[child['key']] if row['kind']=='graph' else (container[index] if row['kind']=='orderedList' else container[child['key']])
            module(packet)
        return Invocation(tuple(select_output(frame.values,OutputAddress(e['source']['node'],e['source']['port'])) for e in row['outputs']),frame)


    def load_state_dict(self,state_dict,strict=True,assign=False):
        if any(node['type'] in RECURRENT_OPERATIONS | ATTENTION_OPERATIONS for node in self.nodes.values()):_validate_recurrent_model_state(state_dict,self.state_dict())
        return super().load_state_dict(state_dict,strict=strict,assign=assign)

    def materialize_lazy_parameters(self):
        """Initialize only lazy layers on their current device/dtype, before consumers scan state.

        One row with the inferred last dimension is sufficient; no source data,
        intermediate activations or other stateful modules participate.
        """
        for ident in self.order:
            if self.nodes[ident]['type'] != 'lazy_linear': continue
            layer = self.layers[self.layer_keys[ident]]
            upstream = select_output(self.layouts,self.incoming[ident]['in'])
            resolved = resolve_lazy_linear(self.nodes[ident]['config'], upstream)
            if isinstance(layer, nn.LazyLinear):
                with torch.no_grad():
                    layer(torch.zeros((1, resolved['inputFeatures']), device=layer.weight.device, dtype=layer.weight.dtype))
            expected = (resolved['outputFeatures'], resolved['inputFeatures'])
            if tuple(layer.weight.shape) != expected or (layer.bias is not None) != resolved['bias'] or (layer.bias is not None and tuple(layer.bias.shape) != (expected[0],)):
                raise ValueError(f'{ident}: LazyLinear state does not match the inferred graph shape; reload compatible weights.')

    def operation_device_disclosures(self,device):
        """Bounded per-node device policy, emitted once and retained with the run."""
        if device!='mps':return []
        result=[]
        for ident in self.order:
            node=self.nodes[ident];kind=node['type']
            if kind not in ('adaptive_avg_pool1d','adaptive_avg_pool2d'):continue
            source=select_output(self.layouts,self.incoming[ident]['in']).sample
            target=select_output(self.layouts,OutputAddress(ident,'out')).sample
            if any(a % b != 0 for a,b in zip(source[1:],target[1:])):
                result.append({'nodeID':ident,'operator':kind,'computeDevice':'cpu','returnDevice':'mps','reason':'nondivisibleAdaptiveAverageBins'})
        return result

    def execute_node(self,ident,values,x):
        kind=self.nodes[ident]['type']; ports=self.incoming[ident]
        if kind=='bag_input':return execute_bag_source(x,self.nodes[ident]['config'])
        if kind=='embedding_bag':
            config=self.nodes[ident]['config']
            packet=EmbeddingBagPacket(config['layout'],select_output(values,ports['indices']),
                select_output(values,ports['offsets']) if config['layout']=='packed' else None,
                select_output(values,ports['per_sample_weights']) if config['per_sample_weights'] else None,
                config['include_last_offset'])
            return self.layers[self.layer_keys[ident]](packet)
        if kind in TRANSFORMS: return apply_layout(select_output(values,ports['in']),kind,self.nodes[ident]['config'])
        if kind=='index_input':
            shape=self.nodes[ident]['config']['shape']
            if x.dtype!=torch.int64 or list(x.shape[1:])!=shape or x.ndim!=len(shape)+1 or x.shape[0]<=0:
                raise ValueError('Index input requires int64 indices with the declared sample shape and a nonempty batch.')
            return x
        if kind=='input': return x
        if kind in POOLING_RESOLVERS:
            value=select_output(values,ports['in']);sample=select_output(self.layouts,ports['in']).sample
            if value.ndim!=len(sample)+1 or value.shape[0]<=0 or tuple(value.shape[1:])!=sample:
                raise ValueError(f'{kind.replace("max_pool","MaxPool").replace("avg_pool","AvgPool")} requires a nonempty batch with the exact authored sample shape.')
            settings=self.nodes[ident]['config']
            result=(_execute_reviewed_pooling(value,kind,_pooling_kwargs(kind,settings),sample,select_output(self.layouts,OutputAddress(ident,'out')).sample)
                    if kind in REVIEWED_NEW_POOLING else POOLING_FUNCTIONS[kind](value,**settings))
            return {'out':result[0],'indices':result[1]} if self.nodes[ident]['config'].get('return_indices',False) else result
        if kind in CONVOLUTION_LAYERS or kind in TRANSPOSE_CONVOLUTION_LAYERS:
            value=select_output(values,ports['in']);sample=select_output(self.layouts,ports['in']).sample
            if value.ndim!=len(sample)+1 or value.shape[0]<=0 or tuple(value.shape[1:])!=sample:
                constructor=CONVOLUTION_LAYERS.get(kind) or TRANSPOSE_CONVOLUTION_LAYERS[kind]
                raise ValueError(f'{ident}: {constructor.__name__} requires a nonempty batch with the exact authored sample shape.')
        if kind=='lazy_linear' and select_output(values,ports['in']).shape[-1]!=select_output(self.layouts,ports['in']).sample[-1]:
            raise ValueError(f'{ident}: LazyLinear input width differs from the authored graph.')
        if kind in ('depthwise_conv2d','conv_transpose3d','conv_transpose2d','conv_transpose1d','conv3d','conv2d','conv1d','linear','lazy_linear','layer_norm','embedding'): return self.layers[self.layer_keys[ident]](select_output(values,ports['in']))
        if kind in ATTENTION_OPERATIONS:
            names=attention_inputs(kind,self.nodes[ident]['config']);sources=[select_output(values,ports[name]) for name in names]
            return execute_attention(kind,self.nodes[ident]['config'],sources,self.layers[self.layer_keys[ident]] if kind!='attention_mask' else None)
        if kind in RECURRENT_OPERATIONS:
            names=recurrent_inputs(kind,self.nodes[ident]['config']);sources=[select_output(values,ports[name]) for name in names];layouts=[select_output(self.layouts,ports[name]) for name in names];r=resolve_recurrent(kind,self.nodes[ident]['config'],layouts)
            return _execute_recurrent(sources,kind,r['config'],[v.sample for v in layouts],{k:v.sample for k,v in r['outputs'].items()},r['gateElementsPerObservation'],r['workPerObservation'],self.layers[self.layer_keys[ident]])
        if kind in STATEFUL_NORMALIZATION_OPERATIONS:
            value=select_output(values,ports['in']);sample=select_output(self.layouts,ports['in']).sample
            return _execute_stateful_normalization(value,kind,self.nodes[ident]['config'],sample,self.layers[self.layer_keys[ident]])
        if kind in SPATIAL_PATCH_OPERATIONS:
            value=select_output(values,ports['in']);sample=select_output(self.layouts,ports['in']).sample;resolved=resolve_spatial_patch(kind,self.nodes[ident]['config'],BatchedLayout(sample))
            return _execute_spatial_patch(value,kind,resolved['config'],sample,resolved['layout'].sample,resolved['workPerObservation'])
        if kind in ROUTING_OPERATIONS:
            names=routing_inputs(kind,self.nodes[ident]['config']);sources=[select_output(values,ports[name]) for name in names];samples=[select_output(self.layouts,ports[name]).sample for name in names]
            resolved=resolve_tensor_routing(kind,self.nodes[ident]['config'],[select_output(self.layouts,ports[name]) for name in names]);outputs={port:layout.sample for port,layout in resolved['outputs'].items()}
            result=_execute_tensor_routing(sources,kind,resolved['config'],samples,outputs,resolved['workPerObservation'])
            return dict(zip(outputs,result)) if kind in ('split','chunk') else result
        if kind in ('prelu','log_softmax','dropout'):
            value=select_output(values,ports['in']);sample=select_output(self.layouts,ports['in']).sample
            return _execute_remaining_activation(value,kind,self.nodes[ident]['config'],sample,self.layers[self.layer_keys[ident]] if kind!='log_softmax' else None)
        if kind=='bilinear': return self.layers[self.layer_keys[ident]](select_output(values,ports['a']),select_output(values,ports['b']))
        if kind=='relu': return torch.relu(select_output(values,ports['in']))
        if kind=='tanh': return torch.tanh(select_output(values,ports['in']))
        if kind in ('silu','leaky_relu','elu'):
            return getattr(torch.nn.functional,kind)(select_output(values,ports['in']),**self.nodes[ident]['config'],inplace=False)
        if kind=='sigmoid': return torch.sigmoid(select_output(values,ports['in']))
        if kind=='softmax': return torch.softmax(select_output(values,ports['in']),dim=self.nodes[ident]['config']['dim'])
        if kind=='gelu': return torch.nn.functional.gelu(select_output(values,ports['in']),approximate='tanh' if self.nodes[ident]['config']['approximate_tanh'] else 'none')
        if kind in ('concat','add','stack','permute'):
            unary=kind=='permute'
            left=select_output(values,ports['in'] if unary else ports['a']);right=None if unary else select_output(values,ports['b'])
            sample=select_output(self.layouts,ports['in'] if unary else ports['a']).sample
            other=None if unary else select_output(self.layouts,ports['b']).sample
            output=select_output(self.layouts,OutputAddress(ident,'out')).sample
            return _execute_tensor_composition(left,right,kind,self.nodes[ident]['config'],sample,other,output)
        if kind=='matmul': return matrix_multiply(select_output(values,ports['a']),select_output(values,ports['b']))
        return select_output(values,ports['in'])

    def forward(self,x,inspect=False,operation_observer=None):
        if operation_observer is None:operation_observer=getattr(self,'_capture_operation_observer',None)
        if hasattr(self,'composition_root'):
            frame=ExecutionFrame(x,operation_observer)
            source=next(i for i in self.order if self.nodes[i]['type'] in ('input','index_input','bag_input'))
            if operation_observer is not None:operation_observer('started',source)
            frame.values[source]=self.execute_node(source,{},x)
            if operation_observer is not None:operation_observer('completed',source)
            self.composition_root(Invocation((frame.values[source],),frame))
            if operation_observer is not None:operation_observer('started',self.output_id)
            frame.values[self.output_id]=self.execute_node(self.output_id,frame.values,x)
            if operation_observer is not None:operation_observer('completed',self.output_id)
            return (frame.values[self.output_id],frame.values) if inspect else frame.values[self.output_id]
        values={}
        for ident in self.order:
            if operation_observer is not None:operation_observer('started',ident)
            values[ident]=self.execute_node(ident,values,x)
            if operation_observer is not None:operation_observer('completed',ident)
        return (values[self.output_id],values) if inspect else values[self.output_id]


def moons_source():
    """Fixed source observations independent of model or partition seed."""
    gen=torch.Generator().manual_seed(42)
    theta=torch.linspace(0,math.pi,500)
    x=torch.cat((torch.stack((torch.cos(theta),torch.sin(theta)),1),torch.stack((1-torch.cos(theta),.5-torch.sin(theta)),1)))
    x=x+torch.randn(x.shape,generator=gen)*.12
    y=torch.cat((torch.zeros(500,dtype=torch.long),torch.ones(500,dtype=torch.long)))
    return x,y


def load_data(dataset_csv=None,feature_columns=None,target_column=None,split_seed=42,sample_limit=0,validation_percent=20):
    if dataset_csv: return csv_dataset(dataset_csv,feature_columns,target_column,split_seed,sample_limit,validation_percent)
    x,y=moons_source()
    data,selection=select_split(x,y,split_seed,sample_limit,validation_percent)
    source_hash=hashlib.sha256(x.numpy().tobytes()+y.numpy().tobytes()).hexdigest()
    selection['splitHash']=hashlib.sha256(json.dumps({'sourceHash':source_hash,'partitionHash':selection['splitHash']},sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return data,{'dataset':'kwipml-two-moons-v2','sourceSeed':42,'sourceHash':source_hash,'noise':.12,**selection}


def dataset(seed=42):
    """Backward-compatible helper: argument is now explicitly the partition seed."""
    return load_data(split_seed=seed)[0]


def export_source(model,initialization_recipe=None,trainability_recipe=None,gradient_policy_recipe=None,*,_composition_blocks=None):
    """Readable tensor locals; serialized module keys remain checkpoint-compatible."""
    has_bags=any(node['type'] in ('bag_input','embedding_bag') for node in model.nodes.values())
    if hasattr(model,'composition_root'):return _composition_export_source(model,trainability_recipe)
    aliases={};used={'self','x','torch','nn'}
    if has_bags:used.update(('offsets','per_sample_weights','inspect'))
    for ident in model.order:
        node=model.nodes[ident];label=node.get('name')
        if not isinstance(label,str) or not label.strip():label=node['type']
        label=unicodedata.normalize('NFKD',label).encode('ascii','ignore').decode()
        base=re.sub(r'[^a-zA-Z0-9_]+','_',label).strip('_').lower()[:64] or node['type']
        if base[0].isdigit() or keyword.iskeyword(base) or base in {'self','x','torch','nn'}:base='node_'+base
        candidate=base;suffix=2
        while candidate in used:
            candidate=f'{base}_{suffix}';suffix+=1
        used.add(candidate);aliases[ident]=candidate
    for ident in model.order:
        if isinstance(model.layouts[ident],dict):
            original=aliases[ident];ports={}
            for port in model.layouts[ident]:
                if port=='out':ports[port]=original;continue
                base=original+'_'+port;candidate=base;suffix=2
                while candidate in used:candidate=f'{base}_{suffix}';suffix+=1
                used.add(candidate);ports[port]=candidate
            aliases[ident]=ports
    lines=['"""Exported KwipML architecture. Load checkpoint[\"modelState\"] with load_state_dict."""','import torch','from torch import nn','','class Model(nn.Module):','    def __init__(self):','        super().__init__()','        self.layers = nn.ModuleDict()']
    if has_bags:
        if __package__:
            from .bag_export import standalone_bag_helpers
        else:
            from bag_export import standalone_bag_helpers
        insertion=lines.index('class Model(nn.Module):')
        lines[insertion:insertion]=[standalone_bag_helpers(),'']
    if initialization_recipe is not None or trainability_recipe is not None or gradient_policy_recipe is not None:
        lines.append('        self.layer_keys = '+repr(dict(sorted(model.layer_keys.items()))))
    if any(node['type'] in REVIEWED_NEW_POOLING for node in model.nodes.values()):
        lines[1:1]=['import math','import warnings']
        insertion=lines.index('class Model(nn.Module):')
        lines[insertion:insertion]=[inspect.getsource(_execute_reviewed_pooling),'']
    if any(node['type'] in ('concat','add','stack','permute') for node in model.nodes.values()):
        if 'import math' not in lines:lines.insert(1,'import math')
        insertion=lines.index('class Model(nn.Module):')
        lines[insertion:insertion]=[inspect.getsource(_execute_tensor_composition),'']
    if any(node['type'] in ('prelu','log_softmax','dropout') for node in model.nodes.values()):
        insertion=lines.index('class Model(nn.Module):');lines[insertion:insertion]=[inspect.getsource(_execute_remaining_activation),'']
    if any(node['type'] in ATTENTION_OPERATIONS for node in model.nodes.values()):
        if __package__:
            from . import attention_configuration,attention_execution,tensor_layout
        else:
            import attention_configuration,attention_execution,tensor_layout
        configuration=inspect.getsource(attention_configuration)
        configuration=configuration[:configuration.index('try:')]+configuration[configuration.index('OPERATIONS='):configuration.index('def capability(')]
        execution=inspect.getsource(attention_execution)
        execution=execution[:execution.index('if __package__:')]+execution[execution.index('class ScaledDotProductAttention'):]
        insertion=lines.index('class Model(nn.Module):')
        lines[insertion:insertion]=[inspect.getsource(tensor_layout),configuration,execution,inspect.getsource(_validate_recurrent_model_state),'']
    if any(node['type'] in RECURRENT_OPERATIONS for node in model.nodes.values()):
        if 'import math' not in lines:lines.insert(1,'import math')
        insertion=lines.index('class Model(nn.Module):');lines[insertion:insertion]=[inspect.getsource(_validate_recurrent_model_state),inspect.getsource(_execute_recurrent),'']
    if any(node['type'] in STATEFUL_NORMALIZATION_OPERATIONS for node in model.nodes.values()):
        if 'import math' not in lines:lines.insert(1,'import math')
        insertion=lines.index('class Model(nn.Module):');lines[insertion:insertion]=[inspect.getsource(_validate_normalization_state),inspect.getsource(_execute_stateful_normalization),'']
    if any(node['type'] in SPATIAL_PATCH_OPERATIONS for node in model.nodes.values()):
        if 'import math' not in lines:lines.insert(0,'import math')
        insertion=lines.index('class Model(nn.Module):');lines[insertion:insertion]=[inspect.getsource(_execute_spatial_patch),'']
    if any(node['type'] in ROUTING_OPERATIONS for node in model.nodes.values()):
        if 'import math' not in lines:lines.insert(1,'import math')
        insertion=lines.index('class Model(nn.Module):');lines[insertion:insertion]=[inspect.getsource(_execute_tensor_routing),'']
    # Human-readable aliases identify layers without registering duplicate modules.
    layer_aliases={model.layer_keys[ident]:aliases[ident] for ident in model.layer_keys}
    authored_types={model.layer_keys[ident]:model.nodes[ident]['type'] for ident in model.layer_keys}
    if 'lazy_linear' in authored_types.values():
        lines.insert(1, '# Move to the intended device/dtype and run a shape-valid dry forward before creating an optimizer.')
    for ident,layer in model.layers.items():
        lines.append(f'        # {layer_aliases[ident]}: checkpoint key retained')
        if authored_types[ident] in ATTENTION_OPERATIONS:
            node=next(n for n in model.nodes.values() if model.layer_keys.get(n['id'])==ident)
            lines.append(f'        self.layers[{ident!r}] = create_attention_module({node["type"]!r}, {node["config"]!r})')
        elif authored_types[ident] in RECURRENT_OPERATIONS:
            kind=authored_types[ident];node=next(n for n in model.nodes.values() if n['type']==kind and model.layer_keys.get(n['id'])==ident);kwargs=_recurrent_kwargs(kind,node['config'])
            lines.append(f'        self.layers[{ident!r}] = nn.{RECURRENT_LAYERS[kind].__name__}(**{kwargs!r})')
        elif authored_types[ident] in STATEFUL_NORMALIZATION_OPERATIONS:
            kind=authored_types[ident];constructor=STATEFUL_NORMALIZATION_LAYERS[kind].__name__;arguments=f'{layer.num_groups}, {layer.num_channels}' if kind=='group_norm' else str(layer.num_features)
            extra='' if kind=='group_norm' else f', momentum={layer.momentum!r}, track_running_stats={layer.track_running_stats}'
            lines.append(f'        self.layers[{ident!r}] = nn.{constructor}({arguments}, eps={layer.eps!r}, affine={layer.affine}, bias={layer.bias is not None}{extra})')
        elif authored_types[ident]=='prelu':
            lines.append(f'        self.layers[{ident!r}] = nn.PReLU({layer.num_parameters}, init={model.nodes[next(node for node,key in model.layer_keys.items() if key==ident)]["config"]["init"]!r})')
        elif authored_types[ident]=='dropout':
            lines.append(f'        self.layers[{ident!r}] = nn.Dropout(p={layer.p!r}, inplace=False)')
        elif authored_types[ident]=='lazy_linear':
            lines.append(f'        self.layers[{ident!r}] = nn.LazyLinear({layer.out_features}, bias={layer.bias is not None})')
        elif authored_types[ident] in TRANSPOSE_CONVOLUTION_LAYERS:
            constructor=TRANSPOSE_CONVOLUTION_LAYERS[authored_types[ident]].__name__
            lines.append(f'        self.layers[{ident!r}] = nn.{constructor}({layer.in_channels}, {layer.out_channels}, {layer.kernel_size!r}, stride={layer.stride!r}, padding={layer.padding!r}, output_padding={layer.output_padding!r}, groups={layer.groups}, bias={layer.bias is not None}, dilation={layer.dilation!r})')
        elif authored_types[ident] in CONVOLUTION_LAYERS:
            constructor=CONVOLUTION_LAYERS[authored_types[ident]].__name__
            lines.append(f'        self.layers[{ident!r}] = nn.{constructor}({layer.in_channels}, {layer.out_channels}, {layer.kernel_size!r}, stride={layer.stride!r}, padding={layer.padding!r}, dilation={layer.dilation!r}, groups={layer.groups}, bias={layer.bias is not None}, padding_mode={layer.padding_mode!r})')
        elif authored_types[ident]=='embedding_bag':
            node=next(node for node,key in model.layer_keys.items() if key==ident)
            lines.append(f'        self.layers[{ident!r}] = CheckedEmbeddingBag({model.nodes[node]["config"]!r})')
            lines.append(f'        self.layers[{ident!r}].weight.requires_grad_({layer.weight.requires_grad})')
        elif isinstance(layer,nn.Embedding):
            lines.append(f'        self.layers[{ident!r}] = nn.Embedding({layer.num_embeddings}, {layer.embedding_dim}, padding_idx={layer.padding_idx!r}, max_norm={layer.max_norm!r}, norm_type={layer.norm_type!r}, scale_grad_by_freq={layer.scale_grad_by_freq}, sparse={layer.sparse})')
            lines.append(f'        self.layers[{ident!r}].weight.requires_grad_({layer.weight.requires_grad})')
        elif isinstance(layer,nn.Bilinear):
            lines.append(f'        self.layers[{ident!r}] = nn.Bilinear({layer.in1_features}, {layer.in2_features}, {layer.out_features}, bias={layer.bias is not None})')
        elif isinstance(layer,nn.LayerNorm):
            lines.append(f'        self.layers[{ident!r}] = nn.LayerNorm({list(layer.normalized_shape)!r}, eps={layer.eps!r}, elementwise_affine={layer.elementwise_affine}, bias={layer.bias is not None})')
        else:lines.append(f'        self.layers[{ident!r}] = nn.Linear({layer.in_features}, {layer.out_features}, bias={layer.bias is not None})')
    if any(node['type'] in RECURRENT_OPERATIONS | ATTENTION_OPERATIONS for node in model.nodes.values()):
        lines += ['', '    def load_state_dict(self, state_dict, strict=True, assign=False):', '        _validate_recurrent_model_state(state_dict, self.state_dict())', '        return super().load_state_dict(state_dict, strict=strict, assign=assign)']
    if trainability_recipe is not None:
        for name,p in model.named_parameters():lines.append('        dict(self.named_parameters())['+repr(name)+'].requires_grad_('+repr(bool(p.requires_grad))+')')
    lines+=['','    def forward(self, x, offsets=None, per_sample_weights=None, inspect=False):' if has_bags else '    def forward(self, x, inspect=False):' if gradient_policy_recipe is not None else '    def forward(self, x):']
    if has_bags:
        source=next(node for node in model.nodes.values() if node['type']=='bag_input')
        lines.append('        x = _coerce_bag_packet(x, offsets, per_sample_weights, '+repr(source['config'])+')')
    for ident in model.order:
        block_start=len(lines)
        kind=model.nodes[ident]['type'];ports=model.incoming[ident]
        unary=select_output(aliases,ports['in']) if 'in' in ports else None
        if kind in CONVOLUTION_LAYERS or kind in TRANSPOSE_CONVOLUTION_LAYERS:
            sample=select_output(model.layouts,ports['in']).sample
            constructor=CONVOLUTION_LAYERS.get(kind) or TRANSPOSE_CONVOLUTION_LAYERS[kind]
            lines += [f'        if {unary}.ndim != {len(sample)+1} or {unary}.shape[0] <= 0 or {unary}.shape[1:] != {sample!r}:', f"            raise ValueError('{constructor.__name__} requires a nonempty batch with the exact authored sample shape.')"]
        if kind in POOLING_RESOLVERS:
            sample=select_output(model.layouts,ports['in']).sample
            lines += [f'        if {unary}.ndim != {len(sample)+1} or {unary}.shape[0] <= 0 or {unary}.shape[1:] != {sample!r}:', f"            raise ValueError('{kind.replace("max_pool","MaxPool").replace("avg_pool","AvgPool")} requires a nonempty batch with the exact authored sample shape.')"]
        if kind=='lazy_linear':
            width=select_output(model.layouts,ports['in']).sample[-1]
            lines += [f'        if {unary}.shape[-1] != {width}:', "            raise ValueError('LazyLinear input width differs from the authored graph.')"]
        if kind=='embedding':
            layer=model.layers[model.layer_keys[ident]]
            lines += [f'        if {unary}.numel() > {100_000_000//layer.embedding_dim} or {unary}.min().item() < 0 or {unary}.max().item() >= {layer.num_embeddings}:', "            raise ValueError('Embedding indices are out of range or exceed the output allocation budget.')"]
        if kind=='bag_input':
            expr=f'execute_bag_source(x, {model.nodes[ident]["config"]!r}).values()'
        elif kind=='embedding_bag':
            config=model.nodes[ident]['config']
            ids=select_output(aliases,ports['indices'])
            offsets=select_output(aliases,ports['offsets']) if config['layout']=='packed' else 'None'
            weights=select_output(aliases,ports['per_sample_weights']) if config['per_sample_weights'] else 'None'
            expr=f'self.layers[{model.layer_keys[ident]!r}](EmbeddingBagPacket({config["layout"]!r}, {ids}, {offsets}, {weights}, {config["include_last_offset"]!r}))'
        elif kind=='index_input':
            shape=model.nodes[ident]['config']['shape']
            lines += [f'        if x.dtype != torch.int64 or x.ndim != {len(shape)+1} or list(x.shape[1:]) != {shape!r} or x.shape[0] <= 0:', "            raise ValueError('Index input requires int64 indices with the declared sample shape and a nonempty batch.')"]
            expr='x'
        elif kind=='input':expr='x'
        elif kind in ('depthwise_conv2d','conv_transpose3d','conv_transpose2d','conv_transpose1d','conv3d','conv2d','conv1d','linear','lazy_linear','layer_norm','embedding'):expr=f'self.layers[{model.layer_keys[ident]!r}]({unary})'
        elif kind in POOLING_RESOLVERS:
            settings=model.nodes[ident]['config']
            if kind in REVIEWED_NEW_POOLING:
                sample=select_output(model.layouts,ports['in']).sample
                output_sample=select_output(model.layouts,OutputAddress(ident,'out')).sample
                expr=f'_execute_reviewed_pooling({unary}, {kind!r}, {_pooling_kwargs(kind,settings)!r}, {sample!r}, {output_sample!r})'
            else:expr=f'torch.nn.functional.{kind}({unary}, '+', '.join(f'{key}={value!r}' for key,value in settings.items())+')'
        elif kind=='bilinear':expr=f"self.layers[{model.layer_keys[ident]!r}]({select_output(aliases,ports['a'])}, {select_output(aliases,ports['b'])})"
        elif kind in ATTENTION_OPERATIONS:
            names=attention_inputs(kind,model.nodes[ident]['config']);arguments=', '.join(select_output(aliases,ports[name]) for name in names)
            layer=f'self.layers[{model.layer_keys[ident]!r}]' if kind!='attention_mask' else 'None'
            expr=f'execute_attention({kind!r}, {model.nodes[ident]["config"]!r}, [{arguments}], {layer}).values()'
        elif kind in RECURRENT_OPERATIONS:
            names=recurrent_inputs(kind,model.nodes[ident]['config']);layouts=[select_output(model.layouts,ports[name]) for name in names];r=resolve_recurrent(kind,model.nodes[ident]['config'],layouts);arguments=', '.join(select_output(aliases,ports[name]) for name in names)
            samples=[v.sample for v in layouts];outputs={k:v.sample for k,v in r['outputs'].items()}
            expr=f'_execute_recurrent([{arguments}], {kind!r}, {r["config"]!r}, {samples!r}, {outputs!r}, {r["gateElementsPerObservation"]!r}, {r["workPerObservation"]!r}, self.layers[{model.layer_keys[ident]!r}]).values()'
        elif kind in STATEFUL_NORMALIZATION_OPERATIONS:
            sample=select_output(model.layouts,ports['in']).sample
            expr=f'_execute_stateful_normalization({unary}, {kind!r}, {model.nodes[ident]["config"]!r}, {sample!r}, self.layers[{model.layer_keys[ident]!r}])'
        elif kind in SPATIAL_PATCH_OPERATIONS:
            sample=select_output(model.layouts,ports['in']).sample;resolved=resolve_spatial_patch(kind,model.nodes[ident]['config'],BatchedLayout(sample))
            expr=f'_execute_spatial_patch({unary}, {kind!r}, {resolved["config"]!r}, {sample!r}, {resolved["layout"].sample!r}, {resolved["workPerObservation"]!r})'
        elif kind in ROUTING_OPERATIONS:
            names=routing_inputs(kind,model.nodes[ident]['config']);layouts=[select_output(model.layouts,ports[name]) for name in names];resolved=resolve_tensor_routing(kind,model.nodes[ident]['config'],layouts)
            arguments=', '.join(select_output(aliases,ports[name]) for name in names)
            samples=[v.sample for v in layouts];outputs={port:layout.sample for port,layout in resolved['outputs'].items()}
            expr=f'_execute_tensor_routing([{arguments}], {kind!r}, {resolved["config"]!r}, {samples!r}, {outputs!r}, {resolved["workPerObservation"]!r})'
        elif kind in ('prelu','log_softmax','dropout'):
            sample=select_output(model.layouts,ports['in']).sample
            layer=f'self.layers[{model.layer_keys[ident]!r}]' if kind!='log_softmax' else 'None'
            expr=f'_execute_remaining_activation({unary}, {kind!r}, {model.nodes[ident]["config"]!r}, {sample!r}, {layer})'
        elif kind in ('silu','leaky_relu','elu'):
            settings=model.nodes[ident]['config']
            arguments=''.join(f', {key}={value!r}' for key,value in settings.items())
            expr=f'torch.nn.functional.{kind}({unary}{arguments}, inplace=False)'
        elif kind in ('relu','tanh','sigmoid'):expr=f'torch.{kind}({unary})'
        elif kind=='gelu':expr=f"torch.nn.functional.gelu({unary}, approximate={'tanh' if model.nodes[ident]['config']['approximate_tanh'] else 'none'!r})"
        elif kind=='softmax':expr=f"torch.softmax({unary}, dim={model.nodes[ident]['config']['dim']})"
        elif kind=='reshape':
            shape=model.nodes[ident]['config']['shape']
            expr=f'{unary}.reshape({unary}.shape[0], '+', '.join(map(str,shape))+')'
        elif kind=='transpose':
            config=model.nodes[ident]['config']
            expr=f"{unary}.transpose({config['dim0']}, {config['dim1']})"
        elif kind=='flatten':
            config=model.nodes[ident]['config']
            expr=f"{unary}.flatten(start_dim={config['start_dim']}, end_dim={config['end_dim']})"
        elif kind in ('concat','add','stack','permute'):
            unary=kind=='permute'
            left=select_output(aliases,ports['in'] if unary else ports['a']);right='None' if unary else select_output(aliases,ports['b'])
            sample=select_output(model.layouts,ports['in'] if unary else ports['a']).sample
            other=None if unary else select_output(model.layouts,ports['b']).sample
            output=select_output(model.layouts,OutputAddress(ident,'out')).sample
            expr=f'_execute_tensor_composition({left}, {right}, {kind!r}, {model.nodes[ident]["config"]!r}, {sample!r}, {other!r}, {output!r})'
        elif kind=='matmul':
            a,b=select_output(aliases,ports['a']),select_output(aliases,ports['b'])
            left,right=select_output(model.layouts,ports['a']).sample,select_output(model.layouts,ports['b']).sample
            rank=max(len(left),len(right))
            left=(1,)*(rank-len(left))+left
            right=(1,)*(rank-len(right))+right
            max_rows=100_000_000//math.prod(model.layouts[ident].sample)
            lines += [f'        if {a}.shape[0] != {b}.shape[0] or not 0 < {a}.shape[0] <= {max_rows}:', "            raise ValueError('Matrix multiplication requires equal positive batches within its output allocation budget.')"]
            lines.append('        # Broadcast sample axes after the observation batch, preserving independent rows.')
            expr=f'torch.matmul({a}.reshape(({a}.shape[0],) + {left!r}), {b}.reshape(({b}.shape[0],) + {right!r}))'
        else:expr=unary
        target=aliases[ident]
        if isinstance(target,dict):target=', '.join(target.values())+(',' if len(target)==1 else '')
        lines.append(f'        {target} = {expr}')
        if _composition_blocks is not None:_composition_blocks[ident]=(lines[block_start:].copy(),aliases.copy())
    if gradient_policy_recipe is not None or has_bags:
        values=[]
        for ident,alias in aliases.items():
            expression=('{'+', '.join(repr(port)+': '+variable for port,variable in alias.items())+'}') if isinstance(alias,dict) else alias
            values.append(repr(ident)+': '+expression)
        lines.append('        if inspect:return '+aliases[model.output_id]+', {'+', '.join(values)+'}')
    lines.append(f'        return {aliases[model.output_id]}')
    result='\n'.join(lines)+'\n'
    if has_bags and len(result.encode('utf8'))>2_000_000:raise ValueError('Bag exported model exceeds the existing source bound.')
    return result


def _composition_export_source(model,trainability_recipe):
    """Reuse reviewed primitive export bodies inside the genuine native hierarchy."""
    import types
    from collections import OrderedDict
    if __package__:
        from . import module_composition_plan as containers
        from .output_routing import OutputAddress,select_output
    else:
        import module_composition_plan as containers
        from output_routing import OutputAddress,select_output
    proxy=types.SimpleNamespace(order=model.order,nodes=model.nodes,incoming=model.incoming,layouts=model.layouts,output_id=model.output_id)
    proxy.layer_keys={node:'operation_'+str(index) for index,node in enumerate(i for i in model.order if i in model.layer_keys)}
    proxy.layers=OrderedDict((key,model.layers[model.layer_keys[node]]) for node,key in proxy.layer_keys.items())
    blocks={};base=export_source(proxy,initialization_recipe=True,gradient_policy_recipe=True,_composition_blocks=blocks)
    has_bags=any(node['type']=='bag_input' for node in model.nodes.values())
    signature='    def forward(self, x, offsets=None, per_sample_weights=None, inspect=False):' if has_bags else '    def forward(self, x, inspect=False):'
    prefix=base[:base.index(signature)]
    # Install before any generated load_state_dict method, still inside __init__.
    boundary=prefix.find('\n    def load_state_dict(')
    if boundary<0:boundary=len(prefix.rstrip())
    setup=["        self.order = "+repr(model.order),"        self.nodes = "+repr(model.nodes),"        self.layouts = "+repr({i:{p:None for p in v} if isinstance(v,dict) else None for i,v in model.layouts.items()}),"        self.output_id = "+repr(model.output_id),"        self.incoming = {node:{port:OutputAddress(*address) for port,address in ports.items()} for node,ports in "+repr({i:{p:(a.node,a.port) for p,a in ports.items()} for i,ports in model.incoming.items()})+".items()}","        self._install_composition(COMPOSITION_GRAPH)"]
    if trainability_recipe is not None:
        setup.extend('        dict(self.named_parameters())['+repr(name)+'].requires_grad_('+repr(bool(p.requires_grad))+')' for name,p in model.named_parameters())
    prefix=prefix[:boundary].rstrip()+'\n'+'\n'.join(setup)+'\n'+prefix[boundary:]
    sources=['import weakref','from dataclasses import dataclass',inspect.getsource(OutputAddress),inspect.getsource(select_output)]
    sources.extend(inspect.getsource(getattr(containers,name)) for name in ('Invocation','ExecutionFrame','PrimitiveInvocation','GraphContainer','OrderedContainer','NamedContainer'))
    sources.append('COMPOSITION_GRAPH = '+repr({'compositionPlan':model._composition_plan}))
    constructor=inspect.getsource(GraphModel._install_composition)
    constructor=constructor.replace('        admitted=validate_plan(plan,graph,self._composition_ports)\n','')
    constructor=constructor.replace('        self._composition_schedules=dict(admitted.schedules)','        self._composition_schedules='+repr(model._composition_schedules))
    # The plan is a generated pinned constant, admitted before source generation.
    constructor=constructor.replace("json.loads(json.dumps(plan))", "__import__('copy').deepcopy(plan)")
    execute=['    def execute_node(self,ident,values,x):']
    for ident in model.order:
        body,aliases=blocks[ident];execute.append('        if ident == '+repr(ident)+':')
        bound=set()
        for address in model.incoming[ident].values():
            alias=aliases[address.node];name=alias[address.port] if isinstance(alias,dict) else alias
            if name not in bound:execute.append('            '+name+' = select_output(values,OutputAddress('+repr(address.node)+','+repr(address.port)+'))');bound.add(name)
        for line in body:
            for node,key in proxy.layer_keys.items():line=line.replace('self.layers['+repr(key)+']','self.layers['+repr(model.layer_keys[node])+']')
            execute.append('    '+line)
        target=aliases[ident]
        expression=('{'+', '.join(repr(port)+': '+variable for port,variable in target.items())+'}' if isinstance(target,dict) else target)
        execute.append('            return '+expression)
    execute.append("        raise ValueError('Unknown composition primitive identity.')")
    forward=inspect.getsource(GraphModel.forward)
    if has_bags:
        source=next(node for node in model.nodes.values() if node['type']=='bag_input')
        forward=forward.replace('def forward(self,x,inspect=False,operation_observer=None):','def forward(self,x,offsets=None,per_sample_weights=None,inspect=False,operation_observer=None):\n        x = _coerce_bag_packet(x,offsets,per_sample_weights,'+repr(source['config'])+')')
    dispatch=inspect.getsource(GraphModel._composition_container)
    if not has_bags:
        forward=forward.replace("('input','index_input','bag_input')","('input','index_input')")
        bag_dispatch="                if ident=='root' and self.nodes[address.node]['type']=='bag_input':\n                    if type(value) is not dict or address.port not in value:raise ValueError('Root bag invocation requires each actual named coordinate role.')\n                    if not isinstance(existing,dict):raise ValueError('Root bag source must execute before its owning container.')\n                    existing[address.port]=value[address.port]\n                elif isinstance(existing,dict):existing[address.port]=value"
        dispatch=dispatch.replace(bag_dispatch,'                if isinstance(existing,dict):existing[address.port]=value')
    methods=[constructor,inspect.getsource(GraphModel._composition_leaf),dispatch,forward,'\n'.join(execute)]
    insertion=prefix.index('class Model(nn.Module):')
    result=prefix[:insertion]+'\n'.join(sources)+'\n\n'+prefix[insertion:]+'\n'+'\n'.join(methods)+'\n'
    if len(result.encode('utf8'))>2_000_000:raise ValueError('Composition exported model exceeds its existing source bound.')
    return result


def atomic_json(path,value):
    atomic_bytes(path,json.dumps(value,indent=2,allow_nan=False).encode())


def atomic_bytes(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_bytes(data); os.replace(tmp,path)


def initialization_capability():
    return {'version': {'major': 1, 'minor': 0, 'patch': 0}, 'adapterID': 'kwip.runner.initialization', 'recipeSchemaVersions': [1], 'configurationVersions': [1], 'algorithms': ['xavier_uniform', 'kaiming_uniform', 'normal', 'zeros'], 'seedMinimum': 0, 'seedMaximum': 9223372036854775807, 'seedDerivation': 'kwip.initialization.owner-seed.v1', 'generatorPlacement': 'cpu-float32', 'devices': ['cpu', 'mps'], 'parameterDTypes': ['float32'], 'maximumRecipeBytes': 32768, 'maximumRules': 128, 'maximumModelParameters': 200000, 'ownerPolicy': 'expandedNodeRelativeNativeParameter', 'ownerOrder': 'nodeThenParameter', 'aliasPolicy': 'rejectDuplicateParameterIdentity', 'fanPolicy': 'nativeStoredDimensions', 'applicationTiming': 'freshAfterLazyBeforeOptimizer', 'restoredWeights': 'preserve', 'candidateCommit': 'validateAllBeforeCopy', 'preservesParameterIdentity': True, 'initializesBuffers': False, 'standaloneHelperMaximumBytes': 131072}


def capabilities():
    if __package__:
        from .bag_source import capability as bag_source_capability
        from .embedding_bag import capability as embedding_bag_capability
    else:
        from bag_source import capability as bag_source_capability
        from embedding_bag import capability as embedding_bag_capability
    if __package__:
        from .module_composition_capability import capability as native_composition_capability
        from .gradient_policy import capability as policy_capability
        from .trainability import capability as trainability_capability
    else:
        from module_composition_capability import capability as native_composition_capability
        from gradient_policy import capability as policy_capability
        from trainability import capability as trainability_capability
    if __package__:
        from .checkpoint_lifecycle import capability as lifecycle_capability
    else:
        from checkpoint_lifecycle import capability as lifecycle_capability
    devices=['cpu']+(['mps'] if torch.backends.mps.is_available() else [])
    return {'event':'capabilities','devices':devices,'defaultDevice':devices[-1],'torchVersion':str(torch.__version__),'executionContract':execution_contract(str(torch.__version__),devices),'gradientCapture':gradient_capability(),'failedGradientCapture':failed_gradient_capability(),'lifecycleCapabilities':{'checkpoint_lifecycle':lifecycle_capability()},'initializationCapabilities':{'initialization':initialization_capability()},'gradientPolicyCapabilities':{'gradient_policy':policy_capability()},'trainabilityCapabilities':{'trainability':trainability_capability()},'compositionCapabilities':{'moduleComposition':native_composition_capability()},'componentCapabilities':{'bag_input':bag_source_capability(),'embedding_bag':embedding_bag_capability(),**{kind:objective_capability(kind) for kind in ('loss_cross_entropy','loss_bce_with_logits','loss_mse','loss_l1','loss_nll','loss_kl_div','loss_ctc','loss_triplet_margin')},**{kind:attention_capability(kind) for kind in ATTENTION_OPERATIONS},**{kind:recurrent_capability(kind) for kind in RECURRENT_OPERATIONS},**{kind:scheduler_capability(kind) for kind in SCHEDULER_NAMES},**{kind:optimizer_capability(kind) for kind in OPTIMIZER_NAMES},**{kind:stateful_normalization_capability(kind) for kind in STATEFUL_NORMALIZATION_OPERATIONS},**{kind:spatial_patch_capability(kind) for kind in SPATIAL_PATCH_OPERATIONS},**{kind:routing_capability(kind) for kind in ROUTING_OPERATIONS},**{kind:remaining_activation_capability(kind) for kind in ('prelu','log_softmax','dropout')},**{kind:composition_capability(kind) for kind in ('concat','add','stack','permute')},**{kind:activation_capability(kind) for kind in ('silu','leaky_relu','elu')},'relu':relu_capability(),'tanh':tanh_capability(),'sigmoid':sigmoid_capability(),**{key: capability() for key,capability in POOLING_CAPABILITIES.items()},'depthwise_conv2d':depthwise_conv2d_capability(),'conv_transpose3d':conv_transpose3d_capability(),'conv_transpose2d':conv_transpose2d_capability(),'conv_transpose1d':conv_transpose1d_capability(),'conv3d':conv3d_capability(),'conv2d':conv2d_capability(),'conv1d':conv1d_capability(),'lazyLinear':lazy_linear_capability(),'bilinear':bilinear_capability(),'matrixMultiply':matrix_capability(),'embedding':embedding_capability(),'layerNorm':{'version':{'major':1,'minor':0,'patch':0},'adapterID':'kwip.runner.graph-v2.layer_norm','graphSchemaVersions':[2],'maximumRank':8,'preservesBatch':True,'epsilonMinimum':1.1754943508222875e-38,'epsilonMaximum':3.4028234663852886e38,'elementwiseAffineModes':[False,True],'biasModes':[False,True],'varianceCorrection':0,'usesCurrentInputStatistics':True},'softmax':{'version':{'major':1,'minor':0,'patch':0},'adapterID':'kwip.runner.graph-v2.softmax','graphSchemaVersions':[2],'maximumRank':8,'preservesBatch':True},'tensorLayout':{'version':{'major':1,'minor':0,'patch':0},'adapterID':'kwip.runner.graph-v2.tensor-layout','graphSchemaVersions':[2],'operations':['reshape','transpose','flatten'],'maximumRank':8,'preservesBatch':True},'gelu':{'version':{'major':1,'minor':0,'patch':0},'adapterID':'kwip.runner.graph-v2.gelu','graphSchemaVersions':[2],'approximateTanhModes':[False,True]}}}


def cpu_state(value):
    if isinstance(value,torch.Tensor): return value.detach().cpu()
    if isinstance(value,dict): return {k:cpu_state(v) for k,v in value.items()}
    if isinstance(value,list): return [cpu_state(v) for v in value]
    if isinstance(value,tuple): return tuple(cpu_state(v) for v in value)
    return value


def run(graph,output,epochs=100,seed=42,emit=lambda event:None,registry=None,device='auto',dataset_csv=None,feature_columns=None,target_column=None,batch_size=1024,sample_limit=0,split_seed=42,validation_percent=20,learning_rate=.03,prepared=None,resume_checkpoint=None,finetune_checkpoint=None,composite_source=None,gradient_capture=None,check_cancelled=lambda:None,sample_capture=None,optimizer_recipe=None,scheduler_recipe=None,objective_recipe=None,target_mapping=None,objective_scalarization='mean',initialization_recipe=None,*,gradient_policy=None,trainability=None,derivative_capture=None,gradient_observer=None):
    check_cancelled()
    if __package__:
        from .model_input import observation_count,move_to_device
        from .bag_training import input_slice,prediction_rows as bag_prediction_rows
    else:
        from model_input import observation_count,move_to_device
        from bag_training import input_slice,prediction_rows as bag_prediction_rows
    bag_source=any(node.get('type')=='bag_input' for node in graph['nodes'])
    if __package__:
        from . import initialization
    else:
        import initialization
    if __package__:
        from . import gradient_policy as gradient_policy_module,trainability as trainability_module
    else:
        import gradient_policy as gradient_policy_module,trainability as trainability_module
    gradient_policy=gradient_policy_module.validate_recipe(gradient_policy) if gradient_policy is not None else None
    trainability=trainability_module.validate_recipe(trainability) if trainability is not None else None
    derivative_capture=gradient_policy_module.validate_capture(derivative_capture)
    if gradient_policy is not None and gradient_policy['config']['requests'] and not derivative_capture['values']:
        raise ValueError('Enable derivative value capture before running requested analysis.')
    if gradient_policy is None and derivative_capture['values']:raise ValueError('Derivative capture requires a selected gradient policy.')
    if gradient_observer is not None and not callable(gradient_observer):raise ValueError('Trusted gradient observer must be callable.')
    if initialization_recipe is not None:initialization_recipe=initialization.validate_recipe(initialization_recipe)
    if (objective_recipe is None)!=(target_mapping is None):raise ValueError('Explicit objective requires its exact typed target mapping.')
    if objective_scalarization not in ('mean','sum') or objective_recipe is None and objective_scalarization!='mean':raise ValueError('Objective scalarization requires an explicit recipe and mean/sum selection.')
    objective_context=_objective_context({'recipe':objective_recipe,'targetMapping':target_mapping}) if objective_recipe is not None else None
    if objective_context is not None:
        objective_recipe=objective_context['recipe'];target_mapping=objective_context['targetMapping']
        if prepared is None:raise ValueError('Explicit objectives require typed prepared partitions; select and prepare target roles.')
    if gradient_policy is not None:gradient_policy_module.validate_compatibility(gradient_policy,objective_recipe,objective_scalarization)
    optimizer_recipe=validate_optimizer_recipe(optimizer_recipe) if optimizer_recipe is not None else None
    scheduler_recipe=validate_scheduler_recipe(scheduler_recipe) if scheduler_recipe is not None else None
    if scheduler_recipe is not None and optimizer_recipe is None:raise ValueError('A scheduler requires an explicit optimizer recipe.')
    if optimizer_recipe is not None:learning_rate=optimizer_recipe['config']['lr']
    gradient_capture=validate_gradient_config(gradient_capture)
    if __package__:
        from .capture_policy import resolve_training_sample_capture_policy
    else:
        from capture_policy import resolve_training_sample_capture_policy
    explicit_sample_capture=sample_capture is not None
    sample_capture=resolve_training_sample_capture_policy(sample_capture)
    if objective_context is None:execution_order,_=validate(graph,registry,bundled_dataset=prepared is None)
    else:
        execution_order,_,objective_layouts,_=validate(graph,registry,bundled_dataset=False,include_types=True,objective_context=objective_context)
        objective_output=objective_layouts[next(n['id'] for n in graph['nodes'] if n['type']=='output')]
    for node in graph['nodes']:
        if node['type'] in ('embedding','embedding_bag'):
            config=effective_config(node,graph['schemaVersion'])
            if config['sparse'] and not config['freeze']:
                raise ValueError('The current Adam training recipe does not support sparse embedding gradients; choose dense gradients.')
    composite_provenance=None
    if composite_source is not None:
        if __package__:
            from .composite_source import checked_source
        else:
            from composite_source import checked_source
        composite_provenance=checked_source(*composite_source,graph,registry=registry or REGISTRY,objective_context=objective_context)
    if resume_checkpoint and finetune_checkpoint:raise ValueError('Choose either exact resume or weights-only fine-tuning, never both')
    if isinstance(epochs,bool) or not isinstance(epochs,int) or not 1<=epochs<=10000: raise ValueError('Epochs must be an integer from 1 to 10000.')
    if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<=2**32-1: raise ValueError('Seed must be an integer from 0 to 4294967295.')
    if type(batch_size) is not int or not 1<=batch_size<=100000: raise ValueError('Batch size must be an integer from 1 to 100000.')
    if optimizer_recipe is None and (isinstance(learning_rate,bool) or not isinstance(learning_rate,(int,float)) or not math.isfinite(learning_rate) or not 0<learning_rate<=1): raise ValueError('Learning rate must be finite and greater than 0, at most 1.')
    check_cancelled()
    configure_torch(torch)
    actual_device=resolve_device(device)
    if actual_device=='mps' and any(n['type']=='lstm' and effective_config(n,graph['schemaVersion'])['proj_size'] for n in graph['nodes']):raise ValueError('Projected LSTM requires CPU; choose CPU or projection width zero before running.')
    if prepared is None:
        data,provenance=load_data(dataset_csv,feature_columns,target_column,split_seed,sample_limit,validation_percent)
    else:
        data,provenance,_=prepared
    if __package__:
        from . import checkpoint_lifecycle as lifecycle
    else:
        import checkpoint_lifecycle as lifecycle
    check_cancelled()
    training_count=observation_count(data[0]);validation_count=observation_count(data[2])
    contract_data=data
    if objective_context is not None:
        if provenance.get('objectiveRecipe')!=objective_recipe or provenance.get('objectiveMapping')!=target_mapping:raise ValueError('Prepared objective recipe/mapping differs from the selected immutable binding.')
        if len(data)!=4 or type(data[1]) is not dict or type(data[3]) is not dict:raise ValueError('Explicit objective requires aligned typed role dictionaries, never binary labels.')
        training_admission=json.loads(json.dumps(objective_recipe))
        if gradient_policy is not None:training_admission['config']['reduction']='none'
        preflight_objective_training_batches(training_admission,data[1],batch_size,objective_output,training_rows=training_count)
        # Admission-only none measurement avoids inventing a local validation
        # mean. Global denominator admission follows below before model state.
        admission_recipe=json.loads(json.dumps(objective_recipe));admission_recipe['config']['reduction']='none'
        preflight_objective_training_batches(admission_recipe,data[3],min(1024,validation_count),objective_output,training_rows=validation_count)
        op=objective_recipe['componentID'];cfg=objective_recipe['config'];eligible=op=='loss_nll' or op=='loss_cross_entropy' and cfg['target_mode']=='class_indices'
        if eligible:
            valid=data[3]['target']!=cfg['ignore_index'];eligible_count=int(valid.sum().item())
            if cfg['reduction']=='mean':
                denominator=float(torch.tensor(cfg['class_weights'])[data[3]['target'][valid]].sum().item()) if cfg['class_weights'] else eligible_count
                if not math.isfinite(denominator) or denominator<=0:raise ValueError('Validation partition mean denominator is undefined or overflows; revise targets/class weights before running.')
        if scheduler_recipe is not None and scheduler_recipe['config'].get('metric')=='validation_accuracy' and (not eligible or eligible_count==0):raise ValueError('Validation accuracy is unavailable for this objective/target contract; choose validation_loss before training.')
        role_order=[r['role'] for r in target_mapping['roles']]
        contract_data=(data[0],*[data[1][role] for role in role_order],data[2],*[data[3][role] for role in role_order])
    microbatches=math.ceil(training_count/batch_size)
    accumulation_steps=gradient_policy['config']['accumulation_steps'] if gradient_policy is not None else 1
    updates_per_epoch=math.ceil(microbatches/accumulation_steps)
    window_denominators={}
    if gradient_policy is not None:
        for start in range(0,training_count,batch_size*accumulation_steps):
            stop=min(start+batch_size*accumulation_steps,training_count)
            window_denominators[start]=gradient_policy_module.window_denominator(objective_recipe,{k:v[start:stop] for k,v in data[1].items()},[stop-start,*objective_output.sample],objective_scalarization) if objective_context is not None else float(stop-start)
    update_cadence=dict(schemaVersion=1,unit='optimizerUpdate',microbatchesPerEpoch=microbatches,optimizerUpdatesPerEpoch=updates_per_epoch,accumulationSteps=accumulation_steps)
    checkpoint_contract=lifecycle.contract(graph,contract_data,provenance,seed,batch_size,learning_rate,actual_device,RECIPE_VERSION,execution_order)
    explicit_model=None;parameter_ownership=None
    if optimizer_recipe is not None or initialization_recipe is not None or trainability is not None or 'compositionPlan' in graph:
        torch.manual_seed(seed);torch.use_deterministic_algorithms(actual_device=='cpu')
        explicit_model=GraphModel(graph,registry,bundled_dataset=prepared is None,**({'objective_context':objective_context} if objective_context is not None else {})).to(actual_device);explicit_model.materialize_lazy_parameters()
    trainability_inventory=trainability_module.apply_trainability(explicit_model,trainability) if trainability is not None else None
    if trainability_inventory is not None and not any(owner['requiresGrad'] for owner in trainability_inventory):raise ValueError('Train requires at least one trainable native parameter; keep all-frozen models for inference.')
    if optimizer_recipe is not None:
        parameter_ownership=[{'name':name,'shape':list(parameter.shape),'dtype':str(parameter.dtype).removeprefix('torch.')} for name,parameter in explicit_model.named_parameters() if parameter.requires_grad]
        checkpoint_contract=lifecycle.contract(graph,contract_data,provenance,seed,batch_size,learning_rate,actual_device,RECIPE_VERSION,execution_order,optimizer_recipe,parameter_ownership)
        provenance={**provenance,'optimizer':{'recipe':optimizer_recipe,'parameterOwnership':parameter_ownership,'implementation':OPTIMIZER_NAMES[optimizer_recipe['componentID']],'torchVersion':str(torch.__version__)}}
    if scheduler_recipe is not None:
        scheduler_preflight(scheduler_recipe,optimizer_recipe,epochs,updates_per_epoch,len(parameter_ownership))
        checkpoint_contract['scheduler']={'schemaVersion':1,'recipe':scheduler_recipe}
        provenance={**provenance,'scheduler':{'recipe':scheduler_recipe,'implementation':SCHEDULER_NAMES[scheduler_recipe['componentID']],'torchVersion':str(torch.__version__)}}
    if objective_context is not None:
        checkpoint_contract['objective']={'schemaVersion':1,'recipe':objective_recipe,'targetMapping':target_mapping,'scalarization':objective_scalarization}
        provenance={**provenance,'objective':{'recipe':objective_recipe,'targetMapping':target_mapping,'scalarization':objective_scalarization,'torchVersion':str(torch.__version__)}}
    if initialization_recipe is not None:
        initialization_contract={'schemaVersion':1,'recipe':initialization_recipe,'resolvedOwners':initialization.resolve_owners(explicit_model,initialization_recipe),'generatorPlacement':'cpu-float32','seedDerivation':initialization.SEED_VERSION}
        checkpoint_contract['initialization']=initialization_contract
        provenance={**provenance,'initialization':{**initialization_contract,'torchVersion':str(torch.__version__),'application':'notAppliedOnRestore' if resume_checkpoint or finetune_checkpoint else 'applied'}}
    if gradient_policy is not None:
        schedule=dict(microbatchesPerEpoch=microbatches,optimizerUpdatesPerEpoch=updates_per_epoch,accumulationSteps=accumulation_steps,gradientScale=gradient_policy['config']['gradient_scale'])
        checkpoint_contract['gradientPolicy']=dict(schemaVersion=1,recipe=gradient_policy,schedule=schedule,checkpointBoundary='completedEpochNoPendingGradients')
        provenance={**provenance,'gradientPolicy':dict(schemaVersion=1,recipe=gradient_policy,implementation='kwip.runner.gradient_policy',torchVersion=str(torch.__version__),schedule=schedule)}
    if trainability is not None:
        ownership=[owner['name'] for owner in trainability_inventory if owner['requiresGrad']]
        checkpoint_contract['trainability']=dict(schemaVersion=1,recipe=trainability,parameterOwnership=trainability_inventory,optimizerParameterNames=ownership)
        provenance={**provenance,'trainability':dict(schemaVersion=1,recipe=trainability,implementation='kwip.runner.trainability',torchVersion=str(torch.__version__),parameterOwnership=trainability_inventory,optimizerParameterNames=ownership)}
    continuation=None;history=[];lineage={'mode':'newTraining','parentRunID':None,'restoredState':[],'resetState':['parameters','optimizer','randomGenerator','epoch','metrics']};initial_epoch=0
    if 'compositionPlan' in graph:
        if __package__:from .native_ownership import composition_identity
        else:from native_ownership import composition_identity
        checkpoint_contract['composition']=composition_identity(explicit_model)
    if checkpoint_contract.get('schedule',{}).get('bufferNodeOrder'):lineage['resetState'].insert(1,'buffers')
    if resume_checkpoint or finetune_checkpoint:
        continuation,history,lineage=lifecycle.load(resume_checkpoint or finetune_checkpoint,checkpoint_contract,'exactResume' if resume_checkpoint else 'weightsOnlyFineTune',epochs)
        if resume_checkpoint:initial_epoch=continuation['epoch']
    check_cancelled()
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()): raise ValueError('Output directory must be empty; use a new directory for each run.')
    if explicit_model is None:
        torch.manual_seed(seed); torch.use_deterministic_algorithms(actual_device=='cpu')
        model=GraphModel(graph,registry,bundled_dataset=prepared is None,**({'objective_context':objective_context} if objective_context is not None else {})).to(actual_device)
        model.materialize_lazy_parameters()
    else:model=explicit_model
    _preflight_normalization_batches(model,training_count,validation_count,batch_size)
    if gradient_policy is not None and gradient_policy['config']['requests']:
        if prepared is None:raise ValueError('Requested derivatives require verified prepared training/validation partitions.')
        if __package__:from .requested_gradients import preflight_requested_gradients
        else:from requested_gradients import preflight_requested_gradients
        if bag_source:
            preflight_requested_gradients(model,{'training':move_to_device(data[0],actual_device),'validation':move_to_device(data[2],actual_device)},gradient_policy,partition_row_ids={'training':provenance['trainingRowIDs'],'validation':provenance['validationRowIDs']},device=actual_device,derivative_capture=derivative_capture)
        else:
            target_elements=0;native={(meta['node'],meta['parameter']):p for meta,p in trainability_module.native_owners(model)}
            for request in gradient_policy['config']['requests']:
                full=data[0] if request['partition']=='training' else data[2]
                if request['row_start']+request['row_count']>observation_count(full):raise ValueError('Requested derivative row range exceeds verified prepared partition.')
                address=OutputAddress(**request['output']);layout=select_output(model.layouts,address);dtype=select_output(model.dtypes,address)
                if dtype!='float32':raise ValueError('Requested derivative output must be floating point.')
                if request['upstream'] is not None and request['upstream']['shape']!=[request['row_count'],*layout.sample]:raise ValueError('Upstream shape differs from selected output.')
                for target in request['targets']:
                    if target['kind']=='input':
                        if target['node'] not in model.nodes or model.nodes[target['node']]['type']!='input' or target['port']!='out':raise ValueError('Requested input target is not the genuine model input endpoint.')
                        target_elements+=request['row_count']*math.prod(full.shape[1:])
                    else:
                        key=(target['node'],target['parameter'])
                        if key not in native:raise ValueError('Requested parameter target is not an actual native owner.')
                        target_elements+=native[key].numel()
                if target_elements>65536:raise ValueError('Requested derivative target appearances exceed65536 elements before training.')

    if initialization_recipe is not None and continuation is None:
        initialization.apply_initialization(model,initialization_recipe,check_cancelled=check_cancelled)
    owned_parameters=[parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer=create_reviewed_optimizer(owned_parameters,optimizer_recipe) if optimizer_recipe is not None else torch.optim.Adam(owned_parameters,lr=learning_rate)
    optimizer_updates=[0]*len(owned_parameters)
    scheduler=SchedulerController(optimizer,scheduler_recipe,optimizer_recipe,len(owned_parameters)) if scheduler_recipe is not None else None
    if continuation is not None:
        expected=model.state_dict();saved=continuation['modelState']
        if saved.keys()!=expected.keys() or any(saved[key].shape!=value.shape or saved[key].dtype!=value.dtype for key,value in expected.items()):raise ValueError('Checkpoint weight names, shape or dtype are incompatible')
        parent_initialization=continuation['contract'].get('initialization')
        if parent_initialization is not None and initialization.resolve_owners(model,parent_initialization['recipe'])!=parent_initialization['resolvedOwners']:
            raise ValueError('Parent initialization ownership differs from the actual selected model.')
        parent_optimizer=continuation['contract'].get('optimizer',{})
        if 'recipe' in parent_optimizer:
            parent_recipe=validate_optimizer_recipe(parent_optimizer['recipe']);parent_updates=continuation.get('optimizerUpdates')
            parent_owned_parameters=owned_parameters
            if 'trainability' in continuation['contract'] or trainability is not None:
                actual_parameters=dict(model.named_parameters())
                parent_names=[owner['name'] for owner in parent_optimizer['parameterOwnership']]
                if any(name not in actual_parameters for name in parent_names):raise ValueError('Parent optimizer ownership is not present in the actual model.')
                parent_owned_parameters=[actual_parameters[name] for name in parent_names]
            saved_flags=[(p,p.requires_grad) for p in model.parameters()]
            try:
                for p in parent_owned_parameters:p.requires_grad_(True)
                parent_instance=create_reviewed_optimizer(parent_owned_parameters,parent_recipe);parent_context=None
                if 'scheduler' in continuation['contract']:
                    parent_controller=SchedulerController(parent_instance,continuation['contract']['scheduler']['recipe'],parent_recipe,len(parent_owned_parameters));parent_controller.restore(continuation.get('scheduler'),parent_updates);parent_context=parent_controller.context()
                validate_optimizer_state(parent_instance,continuation.get('optimizerState'),parent_recipe,parent_updates,parent_context)
            finally:
                for p,flag in saved_flags:p.requires_grad_(flag)
        model.load_state_dict(continuation['modelState'],strict=True)
        for ident in model.order:
            if model.nodes[ident]['type'] in STATEFUL_NORMALIZATION_OPERATIONS:_validate_normalization_state(model.layers[model.layer_keys[ident]],model.nodes[ident]['type'],model.nodes[ident]['config'])
        if resume_checkpoint:
            if optimizer_recipe is not None:
                optimizer_updates=list(continuation['optimizerUpdates'])
                if scheduler is not None:scheduler.restore(continuation.get('scheduler'),optimizer_updates)
                restore_reviewed_optimizer(optimizer,continuation['optimizerState'],optimizer_recipe,optimizer_updates,scheduler.context() if scheduler else None)
            else:lifecycle.restore_optimizer(optimizer,continuation['optimizerState'],initial_epoch*updates_per_epoch)
            torch.set_rng_state(continuation['rngState'])
    if scheduler is not None:
        key='restoredState' if resume_checkpoint else 'resetState'
        lineage[key].extend(name for name in ('scheduler','rateTrace') if name not in lineage[key])
    for policy_key in ('gradientPolicy','trainability'):
        if policy_key in checkpoint_contract:
            key='restoredState' if resume_checkpoint else 'resetState'
            if policy_key not in lineage[key]:lineage[key].append(policy_key)
    observer_elements=0
    def observe_gradient(phase,epoch,window,start,stop):
        nonlocal observer_elements
        if gradient_observer is None:return
        native=trainability_module.native_owners(model)
        required=sum(p.numel()*(1+int(p.grad is not None)) for _,p in native)
        if required>_GRADIENT_OBSERVER_ELEMENTS-observer_elements:raise ValueError('Trusted gradient observer exceeds65536 cumulative values.')
        observer_elements+=required;members={id(p) for p in owned_parameters};owners=[]
        for metadata,p in native:
            if p.grad is not None and p.grad.layout!=torch.strided:raise ValueError('Dense gradient observation does not densify sparse updates.')
            owners.append(dict(metadata,optimizerMember=id(p) in members,optimizerStateKeys=sorted(optimizer.state.get(p,{})),parameterValues=p.detach().cpu().flatten().tolist(),gradientValues=None if p.grad is None else p.grad.detach().cpu().flatten().tolist()))
        gradient_observer(dict(schemaVersion=1,phase=phase,epoch=epoch,window=window,optimizerUpdate=(epoch-1)*updates_per_epoch+window,rowIndices=list(range(start,stop)),owners=owners))
    adapter=ProductExecutionAdapter(model)
    train_x,train_y,val_x,val_y=data
    for context,source in [('Training source',train_x),('Validation source',val_x)]:
        adapter.validate_input(source,device='cpu',context=context)
    criterion=create_objective(objective_recipe,actual_device) if objective_context is not None else nn.CrossEntropyLoss()
    if objective_context is not None:
        if continuation is not None and resume_checkpoint:criterion.load_state_dict(continuation['objectiveState'])
        key='restoredState' if resume_checkpoint else 'resetState'
        lineage[key].extend(name for name in ('objective','objectiveBuffers') if name not in lineage[key])
    run_id=str(uuid.uuid4())
    capture=GradientCapture(model,gradient_capture,run_id=run_id,initial_epoch=initial_epoch,final_epoch=epochs,device=actual_device) if gradient_capture is not None else None
    operation_devices=model.operation_device_disclosures(actual_device)
    device_disclosure={'operationDevices':operation_devices} if operation_devices else {}
    attention_execution=[{'nodeID':ident,'operator':'scaled_dot_product_attention','computeDevice':'mps','policy':'deviceLocalPublicOps','phase':'training'} for ident in model.order if actual_device=='mps' and model.nodes[ident]['type']=='scaled_dot_product_attention' and model.nodes[ident]['config']['dropout']>0]
    if attention_execution:device_disclosure['attentionExecution']=attention_execution
    check_cancelled()
    emit({'event':'started',**device_disclosure,'runID':run_id,'totalEpochs':epochs,'initialEpoch':initial_epoch,'trainingMode':lineage['mode'],'device':actual_device,'trainingSamples':training_count,'validationSamples':validation_count,'batchSize':min(batch_size,training_count),'splitHash':provenance['splitHash'],**({'gradientCapture':capture.started(),'failedGradientCapture':failed_gradient_capability()} if capture else {})})
    for operation in operation_devices:
        emit({'event':'warning','schemaVersion':1,'runID':run_id,'nodeID':operation['nodeID'],
              'code':'pooling.cpuFallback','computeDevice':'cpu','returnDevice':'mps',
              'message':operation['operator']+': CPU fallback for nondivisible adaptive average bins; values and gradients return to MPS.'})
    failure_stage=None
    last_backward_batch_index=None
    try:
        for epoch in range(initial_epoch+1,epochs+1):
            failure_stage=None
            check_cancelled()
            model.train(); training_loss=0.0
            objective_numerator=0.;objective_denominator=0.;objective_counts={'activeCount':0,'ignoredCount':0,'maskedCount':0}
            if scheduler is not None:scheduler.begin_epoch()
            last_backward_batch_index=None
            if capture:capture.begin_epoch(epoch,math.ceil(training_count/batch_size),training_count)
            for begin in range(0,training_count,batch_size):
                failure_stage=None
                check_cancelled()
                failure_stage="trainingForward"
                batch_count=min(batch_size,training_count-begin)
                bx=input_slice(train_x,begin,min(begin+batch_size,training_count),actual_device)
                by={key:value[begin:begin+batch_size].to(actual_device) for key,value in train_y.items()} if objective_context is not None else train_y[begin:begin+batch_size].to(actual_device)
                window_start=(begin//(batch_size*accumulation_steps))*(batch_size*accumulation_steps)
                window_stop=min(window_start+batch_size*accumulation_steps,training_count)
                window_index=window_start//(batch_size*accumulation_steps)+1
                if begin==window_start:
                    if gradient_policy is not None or trainability is not None:model.zero_grad(set_to_none=True)
                    else:optimizer.zero_grad()
                    observe_gradient('windowStart',epoch,window_index,window_start,window_stop)
                prediction=adapter._forward_trusted(bx)
                if objective_context is not None:
                    if gradient_policy is not None:
                        part=objective_partition_contribution(criterion,prediction,by,objective_scalarization)
                        loss=gradient_policy_module.scaled_contribution(part,window_denominators[window_start],gradient_policy['config']['gradient_scale'])
                        summary={**part.metadata,'aggregationNumerator':float(part.numerator.detach().item()),'aggregationDenominator':part.denominator}
                    else:
                        measured=evaluate_objective(criterion,prediction,by,objective_scalarization);loss=measured['scalar'];summary=measured['metadata']
                    objective_numerator+=summary['aggregationNumerator'];objective_denominator+=summary['aggregationDenominator']
                    for key in objective_counts:objective_counts[key]+=summary[key]
                else:
                    loss=criterion(prediction,by)
                    if gradient_policy is not None:loss=loss*batch_count/window_denominators[window_start]
                if not torch.isfinite(loss): raise ValueError('Non-finite training loss; revise the architecture or feature scaling.')
                failure_stage="trainingBackward"
                loss.backward()
                if objective_context is not None and any(p.grad is not None and not torch.isfinite(p.grad).all().item() for p in owned_parameters):raise ValueError('Objective produced nonfinite learned gradients; inspect target geometry and magnitude.')
                last_backward_batch_index=begin//batch_size
                failure_stage=None
                if capture:capture.observe(begin//batch_size,batch_count)
                if gradient_policy is not None and any(p.grad is not None and (p.grad.layout!=torch.strided or not torch.isfinite(p.grad).all().item()) for p in model.parameters()):raise ValueError('Accumulation requires finite dense gradients; sparse updates are unsupported.')
                if gradient_policy is not None and objective_context is None:training_loss+=loss.item()*window_denominators[window_start]
                elif gradient_policy is None:training_loss+=loss.item()*batch_count
                if begin+batch_count<window_stop:continue
                observe_gradient('beforeStep',epoch,window_index,window_start,window_stop)
                failure_stage="optimizerStep"
                if optimizer_recipe is not None:
                    validate_optimizer_state(optimizer,optimizer.state_dict(),optimizer_recipe,optimizer_updates,scheduler.context() if scheduler else None)
                    for parameter,count in zip(owned_parameters,optimizer_updates):
                        grad=parameter.grad
                        if grad is not None and (grad.dtype!=torch.float32 or grad.layout!=torch.strided or grad.shape!=parameter.shape or not torch.isfinite(grad).all().item() or count>=1_000_000_000):raise ValueError('Optimizer gradient/update counter is invalid or nonfinite')
                updated_indices=[i for i,p in enumerate(owned_parameters) if p.grad is not None]
                if scheduler is not None:scheduler.before_update(updated_indices)
                optimizer.step()
                if optimizer_recipe is not None:
                    optimizer_updates=[count+int(parameter.grad is not None) for parameter,count in zip(owned_parameters,optimizer_updates)]
                    if any(not torch.isfinite(parameter).all().item() for parameter in owned_parameters):raise ValueError('Optimizer step produced nonfinite learned settings; lower its rate or revise scaling')
                    validate_optimizer_state(optimizer,optimizer.state_dict(),optimizer_recipe,optimizer_updates,scheduler.context() if scheduler else None)
                if scheduler is not None and scheduler.is_batch:
                    record=scheduler.tick(epoch,window_index,updated_indices)
                    emit({'event':'schedulerTick','schemaVersion':1,'runID':run_id,'scheduler':scheduler.recipe,'record':record})
                observe_gradient('afterStep',epoch,window_index,window_start,window_stop)
                if gradient_policy is not None:model.zero_grad(set_to_none=True)
            failure_stage=None
            check_cancelled()
            failure_stage="validationForward"
            model.eval()
            with torch.no_grad():
                batches=[];objective_parts=[]
                for i in range(0,validation_count,1024):
                    failure_stage=None
                    check_cancelled()
                    failure_stage="validationForward"
                    values=adapter._forward_trusted(input_slice(val_x,i,min(i+1024,validation_count),actual_device))
                    adapter.validate_output(values,context='Validation logits',rows=min(1024,validation_count-i))
                    if objective_context is not None:
                        roles={key:value[i:i+1024].to(actual_device) for key,value in val_y.items()}
                        objective_parts.append(objective_partition_contribution(criterion,values,roles,objective_scalarization))
                    batches.append(values.cpu())
                logits=torch.cat(batches)
                failure_stage="validationLoss"
                if objective_context is not None:
                    evaluation=finish_objective_partition(objective_parts,objective_recipe,objective_scalarization);val_loss=evaluation['scalar'].item()
                    classification=evaluation['metadata'].get('classification');accuracy=classification['accuracy'] if classification else None
                else:val_loss=criterion(logits,val_y).item(); accuracy=(logits.argmax(1)==val_y).float().mean().item()
            if not math.isfinite(val_loss):raise ValueError('Validation loss is non-finite; inspect feature scaling and model parameters.')
            if objective_context is not None:
                is_sum=objective_recipe['config']['reduction']=='sum' or objective_recipe['config']['reduction']=='none' and objective_scalarization=='sum'
                denominator=1. if is_sum else objective_denominator
                train_mean=objective_numerator/denominator
                if not math.isfinite(train_mean):raise ValueError('Training objective aggregate became nonfinite.')
                event={'event':'epoch','epoch':epoch,'trainLoss':train_mean,'validationLoss':val_loss,'objective':{'componentID':objective_recipe['componentID'],'reduction':objective_recipe['config']['reduction'],'scalarization':objective_scalarization if objective_recipe['config']['reduction']=='none' else None,'training':{'observations':training_count,'aggregationNumerator':objective_numerator,'aggregationDenominator':denominator,**objective_counts},'validation':{k:v for k,v in evaluation['metadata'].items() if k!='recipe'}}}
                if accuracy is not None:event['accuracy']=accuracy
            else:event={'event':'epoch','epoch':epoch,'trainLoss':training_loss/training_count,'validationLoss':val_loss,'accuracy':accuracy}
            failure_stage=None
            check_cancelled()
            if scheduler is not None:
                if not scheduler.is_batch:
                    selected=val_loss if scheduler.recipe['config'].get('metric')=='validation_loss' else accuracy
                    record=scheduler.tick(epoch,metric_value=selected)
                    emit({'event':'schedulerTick','schemaVersion':1,'runID':run_id,'scheduler':scheduler.recipe,'record':record})
                event['scheduler']={'recipe':scheduler.recipe,'ticks':scheduler.ticks,'usedRate':scheduler.epoch_used,'nextRate':optimizer.param_groups[0]['lr'],'nextRateUnused':bool(scheduler.is_batch and scheduler.ticks==scheduler.recipe['config']['planned_total_batches'])}
            history.append(event); emit(event)
            if capture:emit(capture.end_epoch())
    except (ValueError, RuntimeError):
        # Best-effort observation transport must never replace the original failure.
        # Capture, transport, cancellation and publication errors are not computation evidence.
        if capture and failure_stage is not None:
            try:
                partial=capture.interrupted(failure_stage,last_backward_batch_index)
                if partial is not None:emit(partial)
            except Exception:
                pass
        raise
    check_cancelled()
    # One representative validation capture forward, after training. These events
    # describe actual Python operation boundaries, not every training batch or GPU timing.
    operation_sequence=0
    def capture_event(name,**detail):
        nonlocal operation_sequence
        check_cancelled()
        operation_sequence+=1
        emit({'event':name,'schemaVersion':1,'runID':run_id,'stage':'validationCapture',
              'sequence':operation_sequence,**detail})
    def observe_operation(state,ident):
        capture_event('modelNodeStarted' if state=='started' else 'modelNodeCompleted',nodeID=ident)
    capture_event('modelCaptureStarted',operationCount=len(model.order))
    with torch.no_grad():
        _,values=adapter.forward(input_slice(val_x,0,min(8,validation_count),actual_device),inspect=True,operation_observer=observe_operation)
        capture_event('modelCaptureCompleted')
        previews={i:summarize_stage(v,include_values=sample_capture['activationValues']) for i,v in values.items()}
        if objective_context is not None:
            objective_summary={k:v for k,v in evaluation['metadata'].items() if k!='recipe'}
            objective_summary['rawPreview']={**summarize_stage(evaluation['raw'],include_values=sample_capture['activationValues']),'captureStatus':'captured' if sample_capture['activationValues'] else 'omittedByPolicy'}
            prediction_header={'schemaVersion':2 if explicit_sample_capture else 1,'runID':run_id,'selection':'firstRowsOfValidationPartition','predictionInput':adapter.input_contract,'predictionRows':[],'evaluationCount':validation_count,**({'captureStatus':'captured'} if explicit_sample_capture else {})}
            class_axis=objective_recipe['config'].get('class_axis',-1)
            if bag_source:
                prediction_header['schemaVersion']=3
                prediction_rows=bag_prediction_rows(val_x,logits,val_y,header=prediction_header,objective=True,raw=evaluation['raw'] if objective_recipe['config']['reduction']=='none' else None,decode_axis=(class_axis-1 if class_axis>0 else class_axis) if accuracy is not None else None) if sample_capture['predictionRows'] else []
            else:
                prediction_rows=_objective_prediction_rows(val_x,logits,val_y,evaluation['raw'],include_raw=objective_recipe['config']['reduction']=='none',decode_axis=(class_axis-1 if class_axis>0 else class_axis) if accuracy is not None else None,header=prediction_header) if sample_capture['predictionRows'] else []
        elif bag_source:
            prediction_header={'schemaVersion':3,'runID':run_id,'selection':'firstRowsOfValidationPartition','predictionInput':adapter.input_contract,'predictionRows':[],'evaluationCount':validation_count,**({'captureStatus':'captured'} if explicit_sample_capture else {})}
            prediction_rows=bag_prediction_rows(val_x,logits,val_y,header=prediction_header) if sample_capture['predictionRows'] else []
        else:prediction_rows=[{'input':x.tolist(),'label':int(y),'predicted':int(p)} for x,y,p in zip(val_x[:200],val_y[:200],logits.argmax(1)[:200])] if sample_capture['predictionRows'] else []
        predictions=[{'x':float(row['input'][0]),'y':float(row['input'][1]),'label':row['label'],'predicted':row['predicted']} for row in prediction_rows] if objective_context is None and adapter.input_dtype==torch.float32 and tuple(adapter.input_shape)==(2,) else []
    snapshot=json.loads(json.dumps(graph)); canonical=json.dumps(snapshot,sort_keys=True,separators=(',',':'))
    manifest={'schemaVersion':1,'recipeVersion':RECIPE_VERSION,'status':'completed','runID':run_id,'graph':snapshot,'graphHash':hashlib.sha256(canonical.encode()).hexdigest(),'seed':seed,'runtime':{'python':platform.python_version(),'torch':torch.__version__,'platform':platform.platform(),'device':actual_device,**device_disclosure},'settings':{'epochs':epochs,'optimizer':'Adam','learningRate':learning_rate,'requestedBatchSize':batch_size,'batchSize':min(batch_size,training_count),'lastBatchSize':training_count%batch_size or min(batch_size,training_count),'batchesPerEpoch':math.ceil(training_count/batch_size),'sampleLimit':sample_limit,'splitSeed':split_seed,'validationPercent':validation_percent,'deterministicAlgorithms':actual_device=='cpu','requestedDevice':device},'evaluation':{'accuracy':accuracy,'validationLoss':val_loss,'count':validation_count},'provenance':provenance}
    if objective_context is not None:
        manifest['settings'].update(objectiveRecipe=objective_recipe,targetMapping=target_mapping,objectiveScalarization=objective_scalarization)
        manifest['evaluation']={'validationLoss':val_loss,'count':validation_count,'objective':objective_summary}
        if accuracy is not None:manifest['evaluation']['accuracy']=accuracy
    if optimizer_recipe is not None:
        manifest['settings']['optimizer']=OPTIMIZER_NAMES[optimizer_recipe['componentID']]
        manifest['settings']['optimizerRecipe']=optimizer_recipe
        manifest['settings']['optimizerRateSource']='explicitRecipe'
    if scheduler is not None:manifest['settings']['schedulerRecipe']=scheduler_recipe
    if initialization_recipe is not None:manifest['settings']['initializationRecipe']=initialization_recipe
    if gradient_policy is not None:
        manifest['settings'].update(gradientPolicyRecipe=gradient_policy,microbatchesPerEpoch=microbatches,optimizerUpdatesPerEpoch=updates_per_epoch)
        manifest['runtime']['updateCadence']=update_cadence
        manifest['derivativeCapture']=derivative_capture
        if capture:manifest['runtime']['gradientSummaryContext']=dict(schemaVersion=1,phase='afterBackwardBeforeStep',values='accumulatedWindowSoFar')
    if trainability is not None:manifest['settings']['trainabilityRecipe']=trainability

    if explicit_sample_capture:
        manifest['sampleCapture']=sample_capture
        for preview in previews.values():preview['captureStatus']='captured' if sample_capture['activationValues'] else 'omittedByPolicy'
    manifest['lineage']=lineage
    manifest['settings']['initialEpoch']=initial_epoch;manifest['settings']['epochsExecuted']=epochs-initial_epoch
    manifest['settings']['partitionMode']='prepared' if prepared is not None else 'generatedSplit'
    with ArtifactStorage(output,TRAINING_STORAGE_BYTES,check_cancelled=check_cancelled) as storage:
        if prepared is not None:
            for field in ('sampleLimit','splitSeed','validationPercent'):manifest['settings'][field]=None
            preparation=output/'preparation';preparation.mkdir()
            if sum(len(content) for content in prepared[2].values())>202000000:raise ValueError('Prepared artifact copy exceeds size bound')
            for name,content in prepared[2].items():
                check_cancelled();storage.atomic_bytes('preparation/'+name,content,2000000 if name=='manifest.json' else 50000000)
        if composite_source is not None:
            composite_directory=output/'composite';composite_directory.mkdir()
            storage.atomic_bytes('composite/source.json',composite_source[0],TRAINING_ARTIFACT_LIMITS['composite/source.json'])
            manifest['provenance']['composite']=composite_provenance
        check_cancelled()
        # Manifest is the last committed artifact; absence means incomplete, never a successful run.
        storage.atomic_json('graph.json',snapshot,TRAINING_ARTIFACT_LIMITS['graph.json'])
        storage.atomic_json('metrics.json',history,TRAINING_ARTIFACT_LIMITS['metrics.json'])
        storage.atomic_bytes('model.py',export_source(model,initialization_recipe,trainability,gradient_policy).encode(),TRAINING_ARTIFACT_LIMITS['model.py'])
        if initialization_recipe is not None:
            storage.atomic_bytes('initialization_recipe.py',initialization.initialization_recipe_code(initialization_recipe,composition=hasattr(model,'composition_root')).encode(),TRAINING_ARTIFACT_LIMITS['initialization_recipe.py'])
        if objective_context is not None:
            if __package__:
                from .loss_export import objective_recipe_code
            else:
                from loss_export import objective_recipe_code
            storage.atomic_bytes('objective_recipe.py',objective_recipe_code(objective_recipe,objective_scalarization).encode(),TRAINING_ARTIFACT_LIMITS['objective_recipe.py'])
        if optimizer_recipe is not None:storage.atomic_bytes('training_recipe.py',training_recipe_code(optimizer_recipe,scheduler_recipe).encode(),TRAINING_ARTIFACT_LIMITS['training_recipe.py'])
        if gradient_policy is not None or trainability is not None:
            if __package__:from .gradient_portable import gradient_policy_code,trainability_recipe_code
            else:from gradient_portable import gradient_policy_code,trainability_recipe_code
            if gradient_policy is not None:storage.atomic_bytes('gradient_policy.py',gradient_policy_code(gradient_policy,objective_recipe,objective_scalarization,composition=hasattr(model,'composition_root'),**({'bag_input_contract':adapter.input_contract} if bag_source else {})).encode(),TRAINING_ARTIFACT_LIMITS['gradient_policy.py'])
            if trainability is not None:storage.atomic_bytes('trainability_recipe.py',trainability_recipe_code(trainability,composition=hasattr(model,'composition_root')).encode(),TRAINING_ARTIFACT_LIMITS['trainability_recipe.py'])

        if scheduler is not None:
            scheduler_trace=storage.atomic_json('rate-trace.json',scheduler.trace(),4000000)
            trace_bytes=(output/'rate-trace.json').read_bytes();manifest['rateTrace']={'schemaVersion':1,'path':'rate-trace.json','bytes':len(trace_bytes),'sha256':hashlib.sha256(trace_bytes).hexdigest(),'records':scheduler.ticks,'timing':scheduler.trace()['timing']}
        check_cancelled()
        storage.atomic_write('checkpoint.pt',TRAINING_ARTIFACT_LIMITS['checkpoint.pt'],lambda stream:torch.save({'checkpointSchemaVersion':lifecycle.SCHEMA_VERSION,'contract':checkpoint_contract,'modelState':cpu_state(model.state_dict()),'optimizerState':cpu_state(optimizer.state_dict()),'epoch':epochs,'seed':seed,'rngState':torch.get_rng_state(),**({'optimizerUpdates':optimizer_updates} if optimizer_recipe is not None else {}),**({'scheduler':scheduler.checkpoint()} if scheduler is not None else {}),**({'objectiveState':cpu_state(criterion.state_dict())} if objective_context is not None else {}),**({'updateSchedule':dict(schemaVersion=1,completedEpochs=epochs,completedMicrobatches=epochs*microbatches,completedOptimizerUpdates=epochs*updates_per_epoch,pendingMicrobatches=0)} if gradient_policy is not None else {})},stream))
        checkpoint_bytes=(output/'checkpoint.pt').read_bytes()
        manifest['checkpoint']={'schemaVersion':lifecycle.SCHEMA_VERSION,'sha256':hashlib.sha256(checkpoint_bytes).hexdigest(),'bytes':len(checkpoint_bytes),'completedEpoch':epochs,'contract':checkpoint_contract,'metricsSHA256':hashlib.sha256((output/'metrics.json').read_bytes()).hexdigest()}
        if gradient_policy is not None and gradient_policy['config']['requests']:
            if __package__:from .requested_gradients import analyze_requested_gradients
            else:from requested_gradients import analyze_requested_gradients
            source_identity=dict(runID=run_id,graphSHA256=hashlib.sha256((output/'graph.json').read_bytes()).hexdigest(),checkpointSHA256=manifest['checkpoint']['sha256'],preparationManifestSHA256=provenance['preparation']['manifestSHA256'],preparationGraphDigest=provenance['preparation']['graphDigest'])
            derivatives=analyze_requested_gradients(model,{'training':move_to_device(train_x,actual_device),'validation':move_to_device(val_x,actual_device)},gradient_policy,partition_row_ids={'training':provenance['trainingRowIDs'],'validation':provenance['validationRowIDs']},source_identity=source_identity,device=actual_device,check_cancelled=check_cancelled,derivative_capture=derivative_capture)
            raw=gradient_policy_module.encoded(derivatives);storage.atomic_bytes('derivative-results.json',raw,8388608)
            manifest['derivativeResults']=dict(schemaVersion=derivatives['schemaVersion'],path='derivative-results.json',bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),requests=len(derivatives['requests']),elements=sum(len(t.get('values',[])) for q in derivatives['requests'] for t in q['targets']),phase='finalSnapshotEvaluation')
        result={'event':'completed',**device_disclosure,'runID':run_id,'accuracy':accuracy,'validationLoss':val_loss,'outputDir':str(output),'device':actual_device,'evaluationCount':validation_count,'predictions':predictions,'predictionRows':prediction_rows,'predictionInput':adapter.input_contract,'previews':previews}
        prediction_artifact={'schemaVersion':1,'runID':run_id,'selection':'firstRowsOfValidationPartition',
                             **{key:result[key] for key in ('predictionInput','predictionRows','evaluationCount')}}
        if explicit_sample_capture:
            result['sampleCapture']=sample_capture
            prediction_artifact['schemaVersion']=2
            prediction_artifact['captureStatus']='captured' if sample_capture['predictionRows'] else 'omittedByPolicy'
        if bag_source:prediction_artifact['schemaVersion']=3
        prediction_bytes=json.dumps(prediction_artifact,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        if len(prediction_bytes)>1_000_000:raise ValueError('Prediction preview exceeds its 1 MB artifact budget.')
        storage.atomic_bytes('predictions.json',prediction_bytes,TRAINING_ARTIFACT_LIMITS['predictions.json'])
        manifest['predictionPreview']={'schemaVersion':prediction_artifact['schemaVersion'],'file':'predictions.json','sha256':hashlib.sha256(prediction_bytes).hexdigest(),'bytes':len(prediction_bytes)}
        activation_artifact={'schemaVersion':1,'runID':run_id,'graphHash':manifest['graphHash'],
                             'selection':'firstValidationRows','previews':previews}
        if explicit_sample_capture:activation_artifact['sampleCapture']=sample_capture
        activation_bytes=json.dumps(activation_artifact,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        storage.atomic_bytes('activations.json',activation_bytes,TRAINING_ARTIFACT_LIMITS['activations.json'])
        manifest['activationPreview']={'schemaVersion':1,'file':'activations.json','sha256':hashlib.sha256(activation_bytes).hexdigest(),'bytes':len(activation_bytes)}
        if objective_context is not None:
            if accuracy is None:result.pop('accuracy',None)
            result['objective']=objective_summary
        if capture:
            capture_bytes=json.dumps(capture.artifact(),sort_keys=True,separators=(',',':'),allow_nan=False).encode()
            if len(capture_bytes)>8_000_000:raise ValueError('Gradient summary artifact exceeds 8 MB capture limit.')
            storage.atomic_bytes('gradient-summary.json',capture_bytes,TRAINING_ARTIFACT_LIMITS['gradient-summary.json'])
            metadata={'schemaVersion':1,'file':'gradient-summary.json','sha256':hashlib.sha256(capture_bytes).hexdigest(),'bytes':len(capture_bytes),'observedEpochCount':len(capture.summaries),'capturedEpochCount':sum(e['observedBatchCount']>0 for e in capture.summaries),'observedBatchCount':sum(e['observedBatchCount'] for e in capture.summaries),'initialEpoch':initial_epoch,'finalEpoch':epochs}
            manifest['gradientCapture']=metadata;result['gradientCapture']=metadata
        manifest_path=output/'manifest.json'
        owned=storage.atomic_json('manifest.json',manifest,TRAINING_ARTIFACT_LIMITS['manifest.json'])
        try:
            check_cancelled()
            emit(result)
            return result
        except BaseException:
            # Only invalidate the manifest published by this operation, never a
            # concurrent replacement owned by another writer.
            try:
                current=manifest_path.lstat()
                if (current.st_dev,current.st_ino)==(owned.st_dev,owned.st_ino):manifest_path.unlink()
            except FileNotFoundError:pass
            raise
