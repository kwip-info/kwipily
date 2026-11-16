import itertools
import pytest
import torch
from runner.avgpool1d_configuration import normalize_avgpool1d, resolve_avgpool1d
from runner.tensor_layout import BatchedLayout


def config(**changes):
    return dict(kernel_size=3, stride=None, padding=0, ceil_mode=False, count_include_pad=True) | changes


@pytest.mark.parametrize('key,value', [('kernel_size',True),('kernel_size',0),('kernel_size',4097),('stride',0),('stride',1.5),('padding',2),('padding',-1),('ceil_mode',1),('count_include_pad',1),('stride',float('nan'))])
def test_rejects_invalid(key,value):
    with pytest.raises(ValueError): normalize_avgpool1d(config(**{key:value}))


def test_exact_settings_and_shapes():
    assert normalize_avgpool1d(config(kernel_size=3.0)) == config()
    for value in [None,{},config(dilation=1)]:
        with pytest.raises(ValueError): normalize_avgpool1d(value)
    for shape in [(2,), (1,2,3), (513,2), (2,4097)]:
        with pytest.raises(ValueError): resolve_avgpool1d(config(),BatchedLayout(shape))


def reference(x,c):
    # Independent window enumeration: only configured padding contributes to divisor;
    # the extra region created by ceiling never does.
    k=c['kernel_size']; s=c['stride'] or k; p=c['padding']; length=x.shape[-1]
    rows=[]
    start=-p
    while start < length:
        end=start+k
        if end > length+p+(s-1 if c['ceil_mode'] else 0): break
        real=x[..., max(start,0):min(end,length)]
        divisor=min(end,length+p)-start if c['count_include_pad'] else real.shape[-1]
        rows.append(real.sum(-1)/divisor)
        start+=s
    if not rows: raise ValueError('empty')
    return torch.stack(rows,-1)


def test_geometry_values_gradients_against_independent_windows():
    for length,k,s,ceil,padcount in itertools.product(range(1,8),range(1,5),(None,1,3),(False,True),(False,True)):
        for p in range(k//2+1):
            c=config(kernel_size=k,stride=s,padding=p,ceil_mode=ceil,count_include_pad=padcount)
            x=torch.linspace(-2,3,length,dtype=torch.float64).reshape(1,1,length).requires_grad_()
            try: expected=reference(x,c)
            except ValueError:
                with pytest.raises(ValueError):resolve_avgpool1d(c,BatchedLayout((1,length)))
                continue
            actual=torch.nn.functional.avg_pool1d(x,**c)
            assert actual.shape[1:]==resolve_avgpool1d(c,BatchedLayout((1,length)))['layout'].sample
            torch.testing.assert_close(actual,expected)
            weights=torch.arange(1,actual.numel()+1,dtype=x.dtype).reshape(actual.shape)
            torch.testing.assert_close(torch.autograd.grad(actual,x,weights)[0],torch.autograd.grad(expected,x,weights)[0])


def test_worked_learning_example_divisors_and_gradient():
    x=torch.tensor([[[2.,4.,6.,8.]]],dtype=torch.float64,requires_grad=True)
    for include,values,gradient in [(True,[2.,6.,4.],[1/3,2/3,1/3,5/6]),(False,[3.,6.,8.],[1/2,5/6,1/3,4/3])]:
        c=config(stride=2,padding=1,ceil_mode=True,count_include_pad=include)
        out=torch.nn.functional.avg_pool1d(x,**c)
        torch.testing.assert_close(out,torch.tensor([[values]],dtype=x.dtype))
        torch.testing.assert_close(torch.autograd.grad(out.sum(),x)[0],torch.tensor([[gradient]],dtype=x.dtype))
