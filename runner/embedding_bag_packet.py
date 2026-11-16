"""Typed bag data and whole-observation collation; no model allocation authority.

schema_version is the Python identity field for serialized schemaVersion=1.
This module does not serialize tensors, detach floating sources or grant engine
parameter-budget/device-operation admission. A frozen packet does not make its
tensor storage immutable: validate again at each consumption boundary.
"""
from dataclasses import dataclass

import torch

MAXIMUM_BAGS = 4096
MAXIMUM_ITEMS = 1_000_000
MAXIMUM_OUTPUT_ELEMENTS = 2_000_000
MAXIMUM_TABLE_SHAPE_ELEMENTS = 100_000_000


@dataclass(frozen=True)
class EmbeddingBagPacket:
    layout: str
    indices: torch.Tensor
    offsets: torch.Tensor | None = None
    per_sample_weights: torch.Tensor | None = None
    include_last_offset: bool = False
    schema_version: int = 1


def validate_packet(packet, *, num_embeddings, embedding_dim, mode,
                    maximum_bags=4096, maximum_items=1_000_000,
                    maximum_output_elements=2_000_000):
    """Validate real source tensors without truncation, copying or allocation.

    Fixed input has nonempty [bags,items-per-bag] rows. Packed input permits
    zero items and empty bags, but always at least one bag. Item weights are
    finite float32 sources on the index device, admitted only for sum.
    """
    _positive_integer(num_embeddings, MAXIMUM_TABLE_SHAPE_ELEMENTS, 'num_embeddings')
    _positive_integer(embedding_dim, MAXIMUM_TABLE_SHAPE_ELEMENTS, 'embedding_dim')
    if num_embeddings > MAXIMUM_TABLE_SHAPE_ELEMENTS // embedding_dim:
        raise ValueError('EmbeddingBag table shape exceeds the static shape bound; engine parameter admission is separate.')
    _positive_integer(maximum_bags, MAXIMUM_BAGS, 'maximum_bags')
    _positive_integer(maximum_items, MAXIMUM_ITEMS, 'maximum_items')
    _positive_integer(maximum_output_elements, MAXIMUM_OUTPUT_ELEMENTS, 'maximum_output_elements')
    if type(mode) is not str or mode not in ('sum', 'mean', 'max'):
        raise ValueError('EmbeddingBag mode must be sum, mean or max.')
    if type(packet) is not EmbeddingBagPacket:
        raise ValueError('EmbeddingBag requires its typed source packet.')
    if type(packet.schema_version) is not int or packet.schema_version != 1:
        raise ValueError('EmbeddingBag packet schemaVersion must be 1.')
    if type(packet.layout) is not str or packet.layout not in ('fixed', 'packed'):
        raise ValueError('EmbeddingBag layout must be fixed or packed.')
    if type(packet.include_last_offset) is not bool:
        raise ValueError('EmbeddingBag include_last_offset must be boolean.')
    _tensor(packet.indices, (torch.int64,), 'indices')
    ids = packet.indices
    if ids.numel() > maximum_items:
        raise ValueError('EmbeddingBag indices exceed the item budget.')
    if packet.layout == 'fixed':
        if ids.ndim != 2 or ids.shape[0] <= 0 or ids.shape[1] <= 0 or packet.offsets is not None:
            raise ValueError('Fixed EmbeddingBag requires nonempty [bags,items] indices and no offsets.')
        bags = ids.shape[0]
    else:
        if ids.ndim != 1:
            raise ValueError('Packed EmbeddingBag indices require rank one.')
        _tensor(packet.offsets, (torch.int64,), 'offsets', device=ids.device)
        offsets = packet.offsets
        if offsets.ndim != 1:
            raise ValueError('Packed offsets require rank one.')
        bags = offsets.numel() - int(packet.include_last_offset)
        if not 0 < bags <= maximum_bags:
            raise ValueError('Packed offsets must describe at least one bag within the bag budget.')
        if offsets[0].item() != 0 or offsets.min().item() < 0 or offsets.max().item() > ids.numel():
            raise ValueError('Packed offsets must start at zero and remain within the item array.')
        if offsets.numel() > 1 and (offsets[1:] < offsets[:-1]).any().item():
            raise ValueError('Packed offsets must be nondecreasing; equal offsets represent empty bags.')
        if packet.include_last_offset and offsets[-1].item() != ids.numel():
            raise ValueError('Terminal offset must equal the number of packed items.')
    if not 0 < bags <= maximum_bags or bags > maximum_output_elements // embedding_dim:
        raise ValueError('EmbeddingBag bags or result exceed the declared data budget.')
    if ids.numel() and (ids.min().item() < 0 or ids.max().item() >= num_embeddings):
        raise ValueError('EmbeddingBag indices must select existing table rows.')
    if packet.per_sample_weights is not None:
        if mode != 'sum':
            raise ValueError('EmbeddingBag per-sample weights require sum reduction.')
        _tensor(packet.per_sample_weights, (torch.float32,), 'per_sample_weights', device=ids.device)
        if packet.per_sample_weights.shape != ids.shape:
            raise ValueError('EmbeddingBag per-sample weights must have exactly the index shape.')
        if not torch.isfinite(packet.per_sample_weights).all().item():
            raise ValueError('EmbeddingBag per-sample weights must be finite.')
    return packet


def collate_bags(packet, bag_indices, *, targets=None, num_embeddings,
                 embedding_dim, mode, maximum_bags=4096,
                 maximum_items=1_000_000, maximum_output_elements=2_000_000):
    """Reorder/subselect complete bags, synchronizing weights and target rows.

    Unique selections implement shuffling/subsets without implicitly enabling
    replacement sampling. Targets are optional numeric dense tensors whose first
    axis exactly matches bag count. Floating targets and item weights retain their
    autograd path. Input tensor storage is not changed.
    """
    contract = dict(num_embeddings=num_embeddings, embedding_dim=embedding_dim, mode=mode,
                    maximum_bags=maximum_bags, maximum_items=maximum_items,
                    maximum_output_elements=maximum_output_elements)
    validate_packet(packet, **contract)
    bags = (packet.indices.shape[0] if packet.layout == 'fixed'
            else packet.offsets.numel() - int(packet.include_last_offset))
    if (type(bag_indices) not in (list, tuple) or not 0 < len(bag_indices) <= maximum_bags
            or any(type(index) is not int or not 0 <= index < bags for index in bag_indices)
            or len(set(bag_indices)) != len(bag_indices)):
        raise ValueError('Select unique, in-range whole-number bag positions within the bag budget.')
    if targets is not None:
        _tensor(targets, (torch.int32, torch.int64, torch.float32, torch.float64), 'targets',
                device=packet.indices.device)
        if targets.ndim == 0 or targets.shape[0] != bags or targets.numel() > maximum_output_elements:
            raise ValueError('Targets require exact bag-row alignment within the data budget.')
        if any(size <= 0 for size in targets.shape):
            raise ValueError('Target sample dimensions must be nonempty.')
        if targets.is_floating_point() and not torch.isfinite(targets).all().item():
            raise ValueError('Targets must contain finite numeric values.')
    selection = torch.tensor(bag_indices, dtype=torch.int64, device=packet.indices.device)
    selected_targets = targets.index_select(0, selection) if targets is not None else None
    if packet.layout == 'fixed':
        result = EmbeddingBagPacket('fixed', packet.indices.index_select(0, selection),
                    per_sample_weights=(packet.per_sample_weights.index_select(0, selection)
                                        if packet.per_sample_weights is not None else None),
                    include_last_offset=packet.include_last_offset)
    else:
        starts = packet.offsets.tolist()
        if not packet.include_last_offset:
            starts.append(packet.indices.numel())
        positions = []
        cursor = 0
        offsets = [0]
        for index in bag_indices:
            positions.append((starts[index], starts[index + 1]))
            cursor += starts[index + 1] - starts[index]
            offsets.append(cursor)
        # Reconstruct from complete slices, including zero-length slices; never
        # confuse item coordinates with observation coordinates.
        ids = torch.cat([packet.indices[start:end] for start, end in positions])
        weights = (torch.cat([packet.per_sample_weights[start:end] for start, end in positions])
                   if packet.per_sample_weights is not None else None)
        if not packet.include_last_offset:
            offsets.pop()
        result = EmbeddingBagPacket('packed', ids,
                    torch.tensor(offsets, dtype=torch.int64, device=ids.device), weights,
                    packet.include_last_offset)
    validate_packet(result, **contract)
    return result, selected_targets


def _positive_integer(value, maximum, name):
    if type(value) is not int or not 0 < value <= maximum:
        raise ValueError(f'{name} must be a positive whole number no greater than {maximum}.')


def _tensor(value, dtypes, name, *, device=None):
    if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
            or value.dtype not in dtypes or value.device.type not in ('cpu', 'mps')):
        raise ValueError(f'{name} requires a dense supported numeric tensor on CPU or MPS with its declared dtype.')
    if device is not None and value.device != device:
        raise ValueError(f'{name} must use the same device as indices.')
