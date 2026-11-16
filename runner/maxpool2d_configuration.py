"""Dimension-specific entry points over shared spatial pooling geometry."""
if __package__:
    from .spatial_maxpool_configuration import normalize_spatial_maxpool,resolve_spatial_maxpool,spatial_capability
else:
    from spatial_maxpool_configuration import normalize_spatial_maxpool,resolve_spatial_maxpool,spatial_capability

_AXES=('height','width')


def normalize_maxpool2d(config):
    return normalize_spatial_maxpool(config,_AXES)


def resolve_maxpool2d(config,layout):
    return resolve_spatial_maxpool(config,layout,_AXES)



def capability():
    return spatial_capability(_AXES)
