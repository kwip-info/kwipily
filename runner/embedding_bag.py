"""Checked real native EmbeddingBag; no engine/device/optimizer capability grant."""
import torch
if __package__:
    from .embedding_bag_configuration import normalize_embedding_bag
    from .embedding_bag_packet import validate_packet
else:
    from embedding_bag_configuration import normalize_embedding_bag
    from embedding_bag_packet import validate_packet

MAXIMUM_PARAMETERS = 2_000_000


class CheckedEmbeddingBag(torch.nn.EmbeddingBag):
    """Direct native weight owner; composition adds its sole .operation prefix."""
    def __init__(self, config, *, device='cpu'):
        canonical = normalize_embedding_bag(config)
        if canonical['num_embeddings'] > MAXIMUM_PARAMETERS // canonical['embedding_dim']:
            raise ValueError('EmbeddingBag table exceeds the general model parameter limit; reduce its rows or width.')
        try:
            placement = torch.device(device)
        except (TypeError, RuntimeError, ValueError) as error:
            raise ValueError('EmbeddingBag requires an explicit CPU or MPS device.') from error
        if placement.type not in ('cpu', 'mps') or (placement.type == 'mps' and not torch.backends.mps.is_available()):
            raise ValueError('EmbeddingBag requires an available CPU or MPS device; no fallback is enabled.')
        _device_controls(canonical, placement, not canonical['freeze'])
        super().__init__(canonical['num_embeddings'], canonical['embedding_dim'],
            max_norm=canonical['max_norm'], norm_type=canonical['norm_type'],
            scale_grad_by_freq=canonical['scale_grad_by_freq'], mode=canonical['mode'],
            sparse=canonical['sparse'], include_last_offset=canonical['include_last_offset'],
            padding_idx=canonical['padding_idx'], device=placement, dtype=torch.float32)
        self.config = canonical
        self.weight.requires_grad_(not canonical['freeze'])

    def forward(self, packet):
        canonical = normalize_embedding_bag(self.config)
        _device_controls(canonical, self.weight.device, self.weight.requires_grad)
        validate_packet(packet, num_embeddings=canonical['num_embeddings'],
                        embedding_dim=canonical['embedding_dim'], mode=canonical['mode'])
        if packet.layout != canonical['layout'] or packet.include_last_offset != canonical['include_last_offset']:
            raise ValueError('EmbeddingBag packet layout and terminal-offset policy must match its configuration.')
        if (packet.per_sample_weights is not None) != canonical['per_sample_weights']:
            raise ValueError('EmbeddingBag per-item weight input presence must match its configuration.')
        if any(getattr(self, name) != canonical[name] for name in (
                'num_embeddings', 'embedding_dim', 'max_norm', 'norm_type',
                'scale_grad_by_freq', 'mode', 'sparse', 'include_last_offset', 'padding_idx')):
            raise ValueError('EmbeddingBag native controls differ from its admitted configuration.')
        if (self.weight.layout != torch.strided or self.weight.dtype != torch.float32
                or self.weight.device.type not in ('cpu', 'mps')
                or self.weight.device != packet.indices.device
                or tuple(self.weight.shape) != (canonical['num_embeddings'], canonical['embedding_dim'])
                or not torch.isfinite(self.weight).all().item()):
            raise ValueError('EmbeddingBag needs matching finite dense float32 table ownership and device.')
        result = super().forward(packet.indices, packet.offsets, per_sample_weights=packet.per_sample_weights)
        if not torch.isfinite(result).all().item() or not torch.isfinite(self.weight).all().item():
            raise ValueError('EmbeddingBag produced nonfinite output or table values; revise inputs and norm controls.')
        return result


def _device_controls(config, device, trainable):
    if device.type == 'mps' and config['max_norm'] is not None:
        raise ValueError('EmbeddingBag maximum norm requires CPU; select CPU to avoid native MPS CPU fallback.')
    if device.type == 'mps' and config['scale_grad_by_freq'] and trainable:
        raise ValueError('EmbeddingBag frequency-scaled table gradients require CPU; select CPU or keep the table frozen.')


def capability():
    """Independent reviewed coordinate contract; no ordinary Embedding fallback grant."""
    return {
        'version': {'major': 1, 'minor': 0, 'patch': 0},
        'adapterID': 'kwip.runner.graph-v2.embedding_bag',
        'graphSchemaVersions': [2], 'packetSchemaVersion': 1,
        'layouts': ['fixed', 'packed'],
        'coordinateDTypes': {'indices': 'int64', 'offsets': 'int64', 'per_sample_weights': 'float32'},
        'maximumItemsPerBag': 128, 'maximumBags': 4096, 'maximumItems': 1_000_000,
        'maximumOutputElements': 2_000_000,
        'integerDerivatives': 'unavailable', 'sourceWeights': 'realPairedInput',
        'batching': 'wholeBags', 'emptyPackedBags': 'preserve',
        'modes': ['mean', 'sum', 'max'], 'weightedModes': ['sum'], 'outputDType': 'float32',
        'maximumParameters': 2_000_000, 'maximumStaticTableElements': 100_000_000,
        'paddingGradient': 'suppressed', 'frozenWeightsMayRenormalize': True,
        'trainableSparseAdam': False, 'mpsMaxNormExecution': 'reject',
        'mpsTrainableFrequencyScaling': 'reject', 'sourceWeightsDerivatives': 'sumOnly'
    }
