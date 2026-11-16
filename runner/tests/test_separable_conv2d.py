"""Separable block semantics: contraction into a dense kernel, without a new runner operation."""
import copy
import itertools

import pytest
import torch
import torch.nn.functional as F

from runner.engine import GraphModel, export_source
from runner.tests.test_depthwise_conv2d_graph import graph as depthwise_graph


def graph(multiplier=2,depthwise_bias=True,pointwise_bias=True,padding_mode='zeros'):
    authored=depthwise_graph({'channel_multiplier':multiplier,'bias':depthwise_bias,
                              'padding_policy':'same','padding_mode':padding_mode})
    pointwise=dict(id='pointwise',type='conv2d',name='Channel mixing',config=dict(
        in_channels=2*multiplier,out_channels=3,kernel_size=[1,1],stride=[1,1],
        dilation=[1,1],groups=1,bias=pointwise_bias,padding_policy='explicit',padding=[0,0],padding_mode='zeros'))
    authored['nodes'].insert(4,pointwise)
    authored['nodes'][2]['name']='tuple'
    next(n for n in authored['nodes'] if n['id']=='head')['config']['in_features']=3*4*5
    authored['edges']=[dict(id=str(i),source=a['id'],sourcePort='out',target=b['id'],targetPort='in')
                       for i,(a,b) in enumerate(zip(authored['nodes'],authored['nodes'][1:]))]
    return authored


def dense_parameters(depthwise,pointwise,multiplier):
    # Explicit contraction over multiplier branches keeps each original channel distinct.
    kernel=torch.stack([torch.stack([
        sum((pointwise.weight[o,c*multiplier+m,0,0]*depthwise.weight[c*multiplier+m,0]
             for m in range(multiplier)))
        for c in range(2)]) for o in range(3)])
    zero=pointwise.weight.sum()*0
    values=[]
    for o in range(3):
        value=pointwise.bias[o] if pointwise.bias is not None else zero
        if depthwise.bias is not None:
            value=value+sum((pointwise.weight[o,c*multiplier+m,0,0]*depthwise.bias[c*multiplier+m]
                             for c in range(2) for m in range(multiplier)),start=zero)
        values.append(value)
    bias=torch.stack(values)
    return kernel,bias


@pytest.mark.parametrize('multiplier',[1,2,3])
@pytest.mark.parametrize('depthwise_bias,pointwise_bias',list(itertools.product([False,True],repeat=2)))
@pytest.mark.parametrize('padding_mode',['zeros','reflect'])
def test_dense_contraction_output_all_gradients_and_export(multiplier,depthwise_bias,pointwise_bias,padding_mode):
    authored=graph(multiplier,depthwise_bias,pointwise_bias,padding_mode);original=copy.deepcopy(authored)
    torch.manual_seed(281);model=GraphModel(authored).double()
    x=torch.randn(2,3,dtype=torch.float64).t().requires_grad_();assert not x.is_contiguous()
    actual,values=model(x,inspect=True)
    depthwise=model.layers['convolution'];pointwise=model.layers['pointwise']
    assert depthwise.groups==2 and pointwise.groups==1
    kernel,bias=dense_parameters(depthwise,pointwise,multiplier)
    # Kernel [2,3] has same padding left/right=1, top=0,bottom=1.
    padded=F.pad(values['shaped'],(1,1,0,1),mode='constant' if padding_mode=='zeros' else padding_mode)
    dense=F.conv2d(padded,kernel,bias)
    torch.testing.assert_close(values['pointwise'],dense,atol=1e-12,rtol=1e-10)
    expected=F.linear(dense.flatten(1),model.layers['head'].weight,model.layers['head'].bias)
    observed_gradients=torch.autograd.grad(actual.square().sum(),(x,*model.parameters()),retain_graph=True)
    dense_gradients=torch.autograd.grad(expected.square().sum(),(x,*model.parameters()))
    for observed,reference in zip(observed_gradients,dense_gradients):
        torch.testing.assert_close(observed,reference,atol=1e-11,rtol=1e-9)
    scope={};source=export_source(model);exec(source,scope)
    exported=scope['Model']().double();exported.load_state_dict(model.state_dict(),strict=True)
    torch.testing.assert_close(exported(x),actual,atol=0,rtol=0)
    exported_gradients=torch.autograd.grad(exported(x).square().sum(),(x,*exported.parameters()))
    for observed,portable in zip(observed_gradients,exported_gradients):
        torch.testing.assert_close(observed,portable,atol=0,rtol=0)
    assert authored==original
    assert 'runner' not in source and 'nn.Conv2d(' in source


@pytest.mark.parametrize('multiplier',[1,3])
@pytest.mark.parametrize('depthwise_bias,pointwise_bias',list(itertools.product([False,True],repeat=2)))
@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='Apple GPU unavailable')
def test_actual_mps_composed_values_and_all_gradients(multiplier,depthwise_bias,pointwise_bias):
    authored=graph(multiplier,depthwise_bias,pointwise_bias)
    torch.manual_seed(412);cpu=GraphModel(authored);gpu=GraphModel(authored).to('mps')
    gpu.load_state_dict(cpu.state_dict())
    x=torch.randn(2,2,requires_grad=True);y=x.detach().to('mps').requires_grad_()
    a=cpu(x);b=gpu(y)
    torch.testing.assert_close(a,b.cpu(),atol=1e-5,rtol=1e-4)
    da=torch.autograd.grad(a.square().sum(),(x,*cpu.parameters()))
    db=torch.autograd.grad(b.square().sum(),(y,*gpu.parameters()))
    for left,right in zip(da,db):torch.testing.assert_close(left,right.cpu(),atol=1e-5,rtol=1e-4)
