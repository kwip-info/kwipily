"""One expanded-node resolver for actual registered native composition state."""
import torch

def native_owners(model):
    if not hasattr(model,'composition_root'):
        if __package__:from .trainability import _legacy_native_owners as legacy
        else:from trainability import _legacy_native_owners as legacy
        return legacy(model)
    return _inventory(model,False)

def state_owners(model):
    """Buffers retain their actual native dtype and relative operator name."""
    if not hasattr(model,'composition_root'):raise ValueError('Composition state ownership requires an admitted native hierarchy.')
    return _inventory(model,True)

def _inventory(model,buffers):
    if type(model.layers) is not dict or type(model.layer_keys) is not dict:raise ValueError('Composition needs non-owning exact native layer lookup.')
    reverse={path:node for node,path in model.layer_keys.items()}
    if len(reverse)!=len(model.layer_keys) or set(reverse)!=set(model.layers):raise ValueError('Composition native operation ownership is ambiguous.')
    for path,module in model.layers.items():
        if model.get_submodule(path) is not module:raise ValueError('Composition lookup differs from actual native registration.')
    items=list(model.named_buffers(remove_duplicate=False) if buffers else model.named_parameters(remove_duplicate=False))
    if len({id(p) for _,p in items})!=len(items):raise ValueError('Aliased native state is refused.')
    if not buffers and any(isinstance(p,torch.nn.parameter.UninitializedParameter) for _,p in items):raise ValueError('Materialize lazy model parameters before resolving owners.')
    if not buffers and sum(p.numel() for _,p in items)>200000:raise ValueError('Model exceeds the200000-parameter limit.')
    result=[]
    for name,p in items:
        if p.layout!=torch.strided or p.device.type not in ('cpu','mps') or not p.numel() or (not buffers and p.dtype!=torch.float32) or not torch.isfinite(p.detach()).all().item():raise ValueError('Native state requires finite dense supported values.')
        matches=[(node,name[len(path)+1:]) for path,node in reverse.items() if name.startswith(path+'.')]
        if len(matches)!=1:raise ValueError('Native state has no unique expanded-node owner.')
        node,relative=matches[0]
        metadata=dict(node=node,parameter=relative,name=name,shape=list(p.shape),dtype=str(p.dtype).replace('torch.',''))
        if not buffers:metadata['requiresGrad']=bool(p.requires_grad)
        result.append((metadata,p))
    return result


def composition_identity(model):
    """Actual registered persistent state in state_dict order; flags are separate."""
    import copy
    parameters=native_owners(model);buffers=state_owners(model)
    p=[{k:v for k,v in metadata.items() if k!='requiresGrad'} for metadata,_ in parameters]
    inventory={m['name']:dict(m,kind='parameter') for m in p}
    inventory.update({m['name']:dict(m,kind='buffer') for m,_ in buffers})
    persistent=[]
    for name,value in model.state_dict().items():
        if name not in inventory:raise ValueError('Composition checkpoint state has no native owner.')
        item=inventory[name]
        if item['shape']!=list(value.shape) or item['dtype']!=str(value.dtype).replace('torch.',''):raise ValueError('Composition persistent state differs from native inventory.')
        persistent.append(item)
    return dict(schemaVersion=1,plan=copy.deepcopy(model._composition_plan),parameterOwnership=p,stateOwnership=persistent)

def portable_source():
    import inspect
    return inspect.getsource(_inventory)+'\ndef native_owners(model):\n    return _inventory(model,False)\n'
