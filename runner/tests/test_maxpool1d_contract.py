import itertools
import pytest
import torch
from runner.maxpool1d_configuration import normalize_maxpool1d, resolve_maxpool1d
from runner.tensor_layout import BatchedLayout


def config(**changes):
    return dict(kernel_size=3,stride=None,padding=0,dilation=1,ceil_mode=False,return_indices=False) | changes


@pytest.mark.parametrize('key,value',[
    ('kernel_size',True),('kernel_size',0),('kernel_size',4097),('stride',0),('stride',1.5),
    ('padding',2),('padding',-1),('dilation',0),('ceil_mode',1),('return_indices',1),
    ('kernel_size',float('nan')),('stride',float('inf')),('dilation',[]),
])
def test_rejects_malformed_settings(key,value):
    with pytest.raises(ValueError):normalize_maxpool1d(config(**{key:value}))


def test_exact_keys_and_canonical_null_stride():
    assert normalize_maxpool1d(config(kernel_size=3.0))==config()
    assert resolve_maxpool1d(config(),BatchedLayout((2,7)))['layout']==BatchedLayout((2,2))
    assert resolve_maxpool1d(config(return_indices=True),BatchedLayout((2,7)))['indicesLayout']==BatchedLayout((2,2))
    assert resolve_maxpool1d(config(),BatchedLayout((2,7)))['indicesLayout'] is None
    for c in [None,{},config(unknown=1),{k:v for k,v in config().items() if k!='stride'}]:
        with pytest.raises(ValueError):normalize_maxpool1d(c)


def test_exhaustive_small_geometry_matches_actual_torch():
    # Includes negative numerators, right-padding correction and dilation gaps.
    for length,kernel,stride,dilation,ceil in itertools.product(range(1,9),range(1,6),(None,1,2,3),range(1,4),(False,True)):
        for padding in range(kernel//2+1):
            c=config(kernel_size=kernel,stride=stride,padding=padding,dilation=dilation,ceil_mode=ceil,return_indices=True)
            layer=torch.nn.MaxPool1d(**c)
            try:values,indices=layer(torch.arange(length,dtype=torch.float64).reshape(1,1,length))
            except RuntimeError:
                with pytest.raises(ValueError):resolve_maxpool1d(c,BatchedLayout((1,length)))
            else:
                result=resolve_maxpool1d(c,BatchedLayout((1,length)))
                assert values.shape[1:]==result['layout'].sample,(c,length)
                assert indices.shape[1:]==result['indicesLayout'].sample
                assert indices.dtype==torch.int64


@pytest.mark.parametrize('shape',[(2,),(1,2,3),(513,2),(2,4097)])
def test_rejects_app_shape_bounds(shape):
    with pytest.raises(ValueError):resolve_maxpool1d(config(),BatchedLayout(shape))
