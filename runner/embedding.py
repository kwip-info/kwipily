"""Checked lookup execution; not yet a registered graph operator."""
import torch
from torch import nn
if __package__:
    from .embedding_configuration import normalize_embedding, MAX_ELEMENTS
else:
    from embedding_configuration import normalize_embedding, MAX_ELEMENTS


class CheckedEmbedding(nn.Embedding):
    def __init__(self,config):
        config=normalize_embedding(config)
        if config['sparse'] and config['scale_grad_by_freq']:
            raise ValueError('Embedding sparse gradients do not support frequency scaling; disable one option.')
        frozen=config['freeze']
        super().__init__(**{key:value for key,value in config.items() if key!='freeze'})
        self.weight.requires_grad_(not frozen)

    def forward(self,indices):
        if not isinstance(indices,torch.Tensor) or indices.dtype not in (torch.int32,torch.int64):
            raise ValueError('Embedding indices require int32 or int64; floating values are not implicitly truncated.')
        if not 2<=indices.ndim<=7 or any(size<=0 for size in indices.shape):
            raise ValueError('Embedding indices require a batch and 1..6 nonempty sample axes.')
        if indices.numel()>MAX_ELEMENTS//self.embedding_dim:
            raise ValueError('Embedding indices would exceed the bounded output allocation.')
        if indices.min().item()<0 or indices.max().item()>=self.num_embeddings:
            raise ValueError(f'Embedding indices must be in [0, {self.num_embeddings-1}]; correct the source mapping.')
        return super().forward(indices)


def capability():
    """Product lookup extension; not a claim of general embedding/API coverage."""
    return {
        'version':{'major':1,'minor':0,'patch':0},
        'adapterID':'kwip.runner.graph-v2.embedding','graphSchemaVersions':[2],
        'inputDType':'int64','weightDType':'float32',
        'maximumInputSampleRank':6,'maximumOutputRank':8,
        'maximumPreparedColumns':128,'maximumParameters':2000000,
        'preservesBatch':True,'outOfRangePolicy':'reject',
        'paddingGradient':'suppressed','frequencyScaling':True,
        'frozenWeightsMayRenormalize':True,
        'trainingOptimizer':'Adam','trainableSparseGradients':False,
        'previewCacheSchemaForMaxNorm':2,'mpsMaxNormExecution':'cpuFallback',
    }
