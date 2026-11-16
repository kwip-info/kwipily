"""Explicit observation operations for tensors and typed bag packets.

Packet checks here establish bounded source geometry, not a particular lookup
vocabulary, model allocation, objective or device-operation capability. Connected
operators retain their stricter admission. No packet pretends to be a Tensor.
"""
import hashlib
import json
import torch
if __package__:
    from .embedding_bag_packet import EmbeddingBagPacket,validate_packet,collate_bags
else:
    from embedding_bag_packet import EmbeddingBagPacket,validate_packet,collate_bags

_GEOMETRY = dict(num_embeddings=100_000_000,embedding_dim=1,mode='sum')


def _tensor(value):
    if (not isinstance(value,torch.Tensor) or value.is_nested or value.layout!=torch.strided
            or value.device.type not in ('cpu','mps')):
        raise ValueError('Model input requires a dense tensor on CPU/MPS or a typed EmbeddingBag packet.')
    return value


def _packet(value):
    return validate_packet(value,**_GEOMETRY)


def _named(value):
    if type(value) is not dict or not 1<=len(value)<=8 or any(type(k) is not str or not k or len(k)>64 for k in value):
        raise ValueError('Named model inputs require 1–8 named tensors.')
    if any(type(v) is dict for v in value.values()):raise ValueError('Named inputs cannot contain nested dictionaries.')
    counts={observation_count(v) for v in value.values()}
    if len(counts)!=1:raise ValueError('Named inputs must share the same aligned observation axis.')
    return value


def observation_count(value):
    if type(value) is dict:
        _named(value)
        return observation_count(next(iter(value.values())))
    if type(value) is EmbeddingBagPacket:
        _packet(value)
        return value.indices.shape[0] if value.layout=='fixed' else value.offsets.numel()-int(value.include_last_offset)
    tensor=_tensor(value)
    if tensor.ndim==0 or tensor.shape[0]<=0:
        raise ValueError('A model source must contain a nonempty observation axis.')
    return tensor.shape[0]


def select_observations(value,positions):
    count=observation_count(value)
    if type(value) is dict:return {k:select_observations(v,positions) for k,v in value.items()}
    if (type(positions) not in (list,tuple) or not positions
            or any(type(index) is not int or not 0<=index<count for index in positions)
            or len(set(positions))!=len(positions)):
        raise ValueError('Select unique, in-range whole-number observation positions.')
    if type(value) is EmbeddingBagPacket:
        result,_=collate_bags(value,positions,**_GEOMETRY)
        return result
    selection=torch.tensor(positions,dtype=torch.int64,device=value.device)
    return value.index_select(0,selection)


def move_to_device(value,device):
    observation_count(value)
    try:placement=torch.device(device)
    except (TypeError,RuntimeError,ValueError) as error:
        raise ValueError('Choose an available CPU or MPS device for model input.') from error
    if placement.type not in ('cpu','mps') or (placement.type=='mps' and not torch.backends.mps.is_available()):
        raise ValueError('Choose an available CPU or MPS device; no input fallback is enabled.')
    if type(value) is EmbeddingBagPacket:
        result=EmbeddingBagPacket(value.layout,value.indices.to(placement),
            value.offsets.to(placement) if value.offsets is not None else None,
            value.per_sample_weights.to(placement) if value.per_sample_weights is not None else None,
            value.include_last_offset,value.schema_version)
        _packet(result)
        return result
    if type(value) is dict:return {k:move_to_device(v,placement) for k,v in value.items()}
    return value.to(placement)


def canonical_input_tensors(value):
    if type(value) is dict:
        _named(value)
        if any(type(v) is dict for v in value.values()):raise ValueError('Named input tensors cannot contain nested dictionaries.')
        return tuple((k,_tensor(value[k])) for k in sorted(value))
    if type(value) is EmbeddingBagPacket:
        _packet(value)
        return (('indices',value.indices),)+(
            (('offsets',value.offsets),) if value.offsets is not None else ())+(
            (('per_sample_weights',value.per_sample_weights),) if value.per_sample_weights is not None else ())
    return (('tensor',_tensor(value)),)


def _update_tensor(hasher,tensor):
    value=tensor.detach().cpu().contiguous()
    hasher.update(str(value.dtype).encode())
    hasher.update(str(tuple(value.shape)).encode())
    hasher.update(value.numpy().tobytes())


def model_inputs_digest(values):
    values=tuple(values)
    hasher=hashlib.sha256()
    # Deliberately identical to the existing tensor_digest byte protocol. New
    # packet framing cannot change continuation identity for an ordinary flow.
    if all(isinstance(value,torch.Tensor) for value in values):
        for value in values:_update_tensor(hasher,value)
        return hasher.hexdigest()
    if not 1<=len(values)<=64:
        raise ValueError('Packet identity requires a bounded input collection.')
    hasher.update(b'kwip.model-input.packet.v1\0')
    for position,value in enumerate(values):
        tensors=canonical_input_tensors(value)
        metadata={'position':position,'kind':'namedTensors' if type(value) is dict else 'tensor'}
        if type(value) is EmbeddingBagPacket:
            metadata.update(kind='embeddingBagPacket',schemaVersion=value.schema_version,
                layout=value.layout,includeLastOffset=value.include_last_offset,
                perSampleWeights=value.per_sample_weights is not None)
        for role,tensor in tensors:
            header=json.dumps({**metadata,'role':role},sort_keys=True,separators=(',',':')).encode()
            hasher.update(len(header).to_bytes(4,'big'));hasher.update(header)
            _update_tensor(hasher,tensor)
    return hasher.hexdigest()
