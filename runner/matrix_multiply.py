"""Differentiable matrix products independently within each observation.

This helper does not register an executable graph component. Leading sample axes
broadcast, while the leading observation axis must match exactly.
"""
import torch
import math
try:
    from .tensor_layout import BatchedLayout
except ImportError:
    from tensor_layout import BatchedLayout


def matrix_multiply(left, right):
    for operand in (left, right):
        if (not isinstance(operand, torch.Tensor) or operand.layout != torch.strided
                or operand.ndim < 3 or operand.shape[0] <= 0):
            raise ValueError('Matrix multiplication requires dense tensors with a nonempty batch and two matrix axes.')
    if left.shape[0] != right.shape[0]:
        raise ValueError('Observation counts must match; the batch never broadcasts.')
    if left.dtype != right.dtype or left.dtype not in (torch.float32, torch.float64):
        raise ValueError('Matrix multiplication requires matching float32 or float64 tensors.')
    if left.device != right.device:
        raise ValueError('Matrix multiplication operands must share a device.')
    a = BatchedLayout(tuple(left.shape[1:]))
    b = BatchedLayout(tuple(right.shape[1:]))
    output = a.matmul(b)  # Validate contraction and broadcasting before allocation.
    if left.shape[0] > 100_000_000 // math.prod(output.sample):
        raise ValueError('Matrix multiplication output exceeds the 100M-element allocation budget.')
    sample_rank = max(len(a.sample), len(b.sample))
    # Insert after N, not before it: ordinary right alignment can confuse N with heads.
    left = left.reshape((left.shape[0],) + (1,) * (sample_rank - len(a.sample)) + a.sample)
    right = right.reshape((right.shape[0],) + (1,) * (sample_rank - len(b.sample)) + b.sample)
    return torch.matmul(left, right)


def capability():
    return {
        'version': {'major': 1, 'minor': 0, 'patch': 0},
        'adapterID': 'kwip.runner.graph-v2.matrixMultiply',
        'graphSchemaVersions': [2], 'maximumRank': 8, 'preservesBatch': True,
        'minimumSampleRank': 2, 'sampleBroadcasting': True,
        'maximumOutputElements': 100_000_000, 'graphDTypes': ['float32'],
    }
