"""Shared local PyTorch device selection, also shipped with portable evaluators."""
import torch


def resolve_device(requested):
    if requested not in ('auto', 'cpu', 'mps'):
        raise ValueError('Device must be auto, cpu or mps.')
    available = torch.backends.mps.is_available()
    resolved = ('mps' if available else 'cpu') if requested == 'auto' else requested
    if resolved == 'mps' and not available:
        raise ValueError('Apple GPU (MPS) is unavailable on this machine. Choose CPU explicitly.')
    return resolved
