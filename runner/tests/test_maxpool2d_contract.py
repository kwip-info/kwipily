import itertools
import pytest
import torch
from runner.maxpool2d_configuration import normalize_maxpool2d, resolve_maxpool2d
from runner.tensor_layout import BatchedLayout


def config(**changes):
    return dict(kernel_size=[2,3],stride=None,padding=[0,0],dilation=[1,1],ceil_mode=False,return_indices=False)|changes


def test_canonical_pairs_nullable_stride_and_indices():
    c=config(return_indices=True)
    assert normalize_maxpool2d(c)==c
    result=resolve_maxpool2d(c,BatchedLayout((3,5,7)))
    assert result['layout']==BatchedLayout((3,2,2))
    assert result['indicesLayout']==result['layout']


@pytest.mark.parametrize('key,value',[
    ('kernel_size',2),('kernel_size',[2]),('kernel_size',[2,3,4]),('kernel_size',[True,3]),
    ('stride',[1,0]),('stride',[1,1.5]),('padding',[2,0]),('padding',[-1,0]),
    ('dilation',[1,4097]),('ceil_mode',1),('return_indices',0),('unknown',0)])
def test_strict_settings(key,value):
    with pytest.raises(ValueError):normalize_maxpool2d(config(**{key:value}))


def test_small_asymmetric_geometry_against_torch():
    checked=0
    for h,w,k,s,d,ceil in itertools.product([1,2,5],[1,3,7],[[2,3],[3,2]], [None,[2,1]], [[1,1],[2,1]], [False,True]):
        c=config(kernel_size=k,stride=s,padding=[1,1],dilation=d,ceil_mode=ceil)
        try: actual=torch.nn.functional.max_pool2d(torch.ones(1,2,h,w),**c)
        except RuntimeError:
            with pytest.raises(ValueError):resolve_maxpool2d(c,BatchedLayout((2,h,w)))
        else:
            assert resolve_maxpool2d(c,BatchedLayout((2,h,w)))['layout'].sample==tuple(actual.shape[1:])
        checked+=1
    assert checked==144


@pytest.mark.parametrize('shape',[(2,4),(2,3,4,5),(513,2,2),(2,4097,2),(2,2,4097)])
def test_shape_bounds(shape):
    with pytest.raises(ValueError):resolve_maxpool2d(config(),BatchedLayout(shape))
