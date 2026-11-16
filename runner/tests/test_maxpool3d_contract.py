import itertools
import pytest
import torch
from runner.maxpool3d_configuration import normalize_maxpool3d,resolve_maxpool3d
from runner.tensor_layout import BatchedLayout


def config(**changes):
    return dict(kernel_size=[2,2,3],stride=None,padding=[0,0,0],dilation=[1,1,1],ceil_mode=False,return_indices=False)|changes


def test_triples_null_stride_and_indices():
    c=config(return_indices=True)
    assert normalize_maxpool3d(c)==c
    result=resolve_maxpool3d(c,BatchedLayout((2,5,7,8)))
    assert result['layout']==BatchedLayout((2,2,3,2))
    assert result['indicesLayout']==result['layout']


@pytest.mark.parametrize('key,value',[
    ('kernel_size',2),('kernel_size',[2,2]),('kernel_size',[2,2,3,4]),('kernel_size',[True,2,3]),
    ('stride',[1,0,1]),('stride',[1,1,1.5]),('padding',[2,0,0]),('padding',[0,-1,0]),
    ('dilation',[1,1,4097]),('ceil_mode',1),('return_indices',0),('unknown',0)])
def test_strict_settings(key,value):
    with pytest.raises(ValueError):normalize_maxpool3d(config(**{key:value}))


def test_asymmetric_geometry_matches_actual_torch():
    count=0
    for depth,height,width,k,s,d,ceil in itertools.product([1,4],[1,3],[2,7],[[2,2,3],[3,2,2]], [None,[2,1,2]], [[1,1,1],[2,1,2]], [False,True]):
        c=config(kernel_size=k,stride=s,padding=[1,1,1],dilation=d,ceil_mode=ceil)
        try:actual=torch.nn.functional.max_pool3d(torch.ones(1,2,depth,height,width),**c)
        except RuntimeError:
            with pytest.raises(ValueError):resolve_maxpool3d(c,BatchedLayout((2,depth,height,width)))
        else:assert resolve_maxpool3d(c,BatchedLayout((2,depth,height,width)))['layout'].sample==tuple(actual.shape[1:])
        count+=1
    assert count==128


@pytest.mark.parametrize('shape',[(2,4),(2,3,4),(2,3,4,5,6),(513,2,2,3),(2,4097,2,3),(2,2,4097,3),(2,2,2,4097)])
def test_shape_bounds(shape):
    with pytest.raises(ValueError):resolve_maxpool3d(config(),BatchedLayout(shape))


@pytest.mark.parametrize('indices',[False,True])
def test_actual_mps_volume_pooling_support(indices):
    if not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
    x=torch.randn(2,2,4,5,6,requires_grad=True);xm=x.detach().to('mps').requires_grad_()
    settings=dict(kernel_size=[2,3,2],stride=[1,2,1],padding=[0,1,0],dilation=[1,1,2],ceil_mode=True,return_indices=indices)
    a=torch.nn.functional.max_pool3d(x,**settings);b=torch.nn.functional.max_pool3d(xm,**settings)
    av=a[0] if indices else a;bv=b[0] if indices else b
    torch.testing.assert_close(av,bv.cpu(),rtol=0,atol=0)
    if indices:assert torch.equal(a[1],b[1].cpu())
    torch.testing.assert_close(torch.autograd.grad(av.sum(),x)[0],torch.autograd.grad(bv.sum(),xm)[0].cpu(),rtol=0,atol=0)
