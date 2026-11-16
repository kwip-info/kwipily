"""Explicit output addresses shared by model validation, execution and code generation."""
from dataclasses import dataclass


@dataclass(frozen=True)
class OutputAddress:
    node: str
    port: str


def select_output(outputs, address):
    """Single tensors represent only 'out'; named outputs require explicit mappings.

    Return the original object: no copies, dtype conversion or autograd detachment.
    This also selects declared layouts/types and generated local identifiers.
    """
    if not isinstance(address, OutputAddress):
        raise ValueError('An explicit node and output port are required.')
    if address.node not in outputs:
        raise ValueError(f'Missing output node {address.node!r}.')
    value = outputs[address.node]
    if isinstance(value, dict):
        if address.port not in value:
            raise ValueError(f'Missing output {address.node!r}.{address.port!r}.')
        return value[address.port]
    if address.port != 'out':
        raise ValueError(f'Unresolved output {address.node!r}.{address.port!r}.')
    return value

def map_stage(value, transform):
    """One declared output set, not arbitrary recursive object serialization."""
    return {port:transform(tensor) for port,tensor in value.items()} if isinstance(value,dict) else transform(value)


def summarize_stage(value, *, include_values=True, heatmap_budget=None, check_cancelled=lambda:None):
    def summary(tensor):
        result={'shape':list(tensor.shape),'dtype':str(tensor.dtype).removeprefix('torch.'),'values':tensor.flatten()[:8].detach().cpu().tolist() if include_values else []}
        if heatmap_budget is not None:result['heatmap']=heatmap_budget.capture(tensor,check_cancelled)
        return result
    return {'outputs':map_stage(value,summary)} if isinstance(value,dict) else summary(value)
