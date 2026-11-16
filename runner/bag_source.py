"""Typed bag-coordinate source, separate from observation-first tensor layouts.

The source validates generic packet geometry against the static100M index bound.
The connected EmbeddingBag independently enforces its actual vocabulary/output
geometry. No source packet is serialized, copied, detached or allocated here.
"""
from dataclasses import dataclass
if __package__:
    from .embedding_bag_packet import validate_packet
else:
    from embedding_bag_packet import validate_packet

FIELDS = frozenset(('layout','maximum_items_per_bag','per_sample_weights','include_last_offset'))

@dataclass(frozen=True)
class BagCoordinateLayout:
    source_id: str
    role: str
    layout: str
    maximum_items_per_bag: int
    per_sample_weights: bool
    include_last_offset: bool


def normalize_bag_source(config):
    if type(config) is not dict or set(config) != FIELDS:
        raise ValueError('Bag input requires exactly its four configuration1 fields.')
    if type(config['layout']) is not str or config['layout'] not in ('fixed','packed'):
        raise ValueError('Bag input layout must be fixed or packed.')
    if type(config['maximum_items_per_bag']) is not int or not 1 <= config['maximum_items_per_bag'] <= 128:
        raise ValueError('Maximum items per bag requires an integer from1 to128.')
    if any(type(config[key]) is not bool for key in ('per_sample_weights','include_last_offset')):
        raise ValueError('Bag input weight and terminal-offset choices require Boolean values.')
    return dict(config)


def source_layouts(source_id,config):
    config = normalize_bag_source(config)
    if type(source_id) is not str or not 1 <= len(source_id) <= 128:
        raise ValueError('Bag source requires its actual bounded node identity.')
    roles = ('indices',) + (('offsets',) if config['layout']=='packed' else ()) + (
        ('per_sample_weights',) if config['per_sample_weights'] else ())
    return {role:BagCoordinateLayout(source_id,role,**config) for role in roles}


def execute_bag_source(packet,config):
    config = normalize_bag_source(config)
    validate_packet(packet,num_embeddings=100_000_000,embedding_dim=1,mode='sum')
    if (packet.layout != config['layout'] or packet.include_last_offset != config['include_last_offset']
            or (packet.per_sample_weights is not None) != config['per_sample_weights']):
        raise ValueError('Bag input packet must match the selected layout, terminal-offset and floating-weight source roles.')
    if packet.layout=='fixed':
        maximum = packet.indices.shape[1]
    else:
        offsets = packet.offsets
        maximum = int((offsets[1:]-offsets[:-1]).max().item()) if offsets.numel()>1 else 0
        if not packet.include_last_offset:
            maximum = max(maximum,packet.indices.numel()-int(offsets[-1].item()))
    if maximum > config['maximum_items_per_bag']:
        raise ValueError('Bag input exceeds maximum items per bag; revise the source mapping or declared limit.')
    result = {'indices':packet.indices}
    if config['layout']=='packed':result['offsets']=packet.offsets
    if config['per_sample_weights']:result['per_sample_weights']=packet.per_sample_weights
    return result


def capability():
    """Independent reviewed coordinate contract; no ordinary Embedding fallback grant."""
    return {
        'version': {'major': 1, 'minor': 0, 'patch': 0},
        'adapterID': 'kwip.runner.graph-v2.bag_input',
        'graphSchemaVersions': [2], 'packetSchemaVersion': 1,
        'layouts': ['fixed', 'packed'],
        'coordinateDTypes': {'indices': 'int64', 'offsets': 'int64', 'per_sample_weights': 'float32'},
        'maximumItemsPerBag': 128, 'maximumBags': 4096, 'maximumItems': 1_000_000,
        'maximumOutputElements': 2_000_000,
        'integerDerivatives': 'unavailable', 'sourceWeights': 'realPairedInput',
        'batching': 'wholeBags', 'emptyPackedBags': 'preserve'
    }
