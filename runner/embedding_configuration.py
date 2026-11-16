"""Pure embedding configuration shared by identity and execution."""
import math

MAX_ELEMENTS = 100_000_000
FIELDS = {'num_embeddings','embedding_dim','padding_idx','max_norm','norm_type',
          'scale_grad_by_freq','sparse','freeze'}


def _integer(value,low,high,name):
    if type(value) not in (int,float) or not low<=value<=high or value!=int(value):
        raise ValueError(f'{name} must be an integer from {low} to {high}.')
    return int(value)


def _positive(value,name):
    try:
        valid=type(value) in (int,float) and value>0 and math.isfinite(value)
    except OverflowError:
        valid=False
    if not valid:raise ValueError(f'{name} must be positive and finite.')
    return float(value)


def normalize_embedding(config):
    if type(config) is not dict or set(config)!=FIELDS:
        raise ValueError('Embedding requires exactly its declared configuration fields.')
    count=_integer(config['num_embeddings'],1,MAX_ELEMENTS,'num_embeddings')
    width=_integer(config['embedding_dim'],1,MAX_ELEMENTS//count,'embedding_dim')
    padding=config['padding_idx']
    if padding is not None:
        padding=_integer(padding,-count,count-1,'padding_idx') % count
    maximum=config['max_norm']
    if maximum is not None:maximum=_positive(maximum,'max_norm')
    norm=_positive(config['norm_type'],'norm_type')
    for name in ('scale_grad_by_freq','sparse','freeze'):
        if type(config[name]) is not bool:raise ValueError(f'{name} must be boolean.')
    return {**config,'num_embeddings':count,'embedding_dim':width,'padding_idx':padding,
            'max_norm':maximum,'norm_type':norm}


