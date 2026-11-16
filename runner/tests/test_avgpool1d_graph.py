import pytest
import torch
from runner.engine import GraphModel,REGISTRY,export_source,capabilities
from runner.preview import preview
from runner.tests.test_avgpool1d_contract import config,reference
from runner.tests.test_conv1d_graph import graph as conv_graph


def graph(include=True):
    g=conv_graph();n=next(n for n in g['nodes'] if n['id']=='convolution')
    n.update(type='avg_pool1d',config=config(stride=2,padding=1,ceil_mode=True,count_include_pad=include))
    next(n for n in g['nodes'] if n['id']=='head')['config']['in_features']=8
    return g


@pytest.mark.parametrize('include',[False,True])
def test_actual_graph_independent_values_gradient_and_export(include):
    model=GraphModel(graph(include),REGISTRY).double()
    x=torch.randn(3,2,14,dtype=torch.float64)[...,::2].requires_grad_()
    actual=model.execute_node('convolution',{'shaped':x},None)
    expected=reference(x,graph(include)['nodes'][3]['config'])
    torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(),x)[0],torch.autograd.grad(expected.sum(),x)[0])
    namespace={};exec(export_source(model),namespace)
    exported=namespace['Model']().double();exported.load_state_dict(model.state_dict())
    value=torch.randn(5,2,dtype=torch.float64,requires_grad=True)
    torch.testing.assert_close(model(value),exported(value),rtol=0,atol=0)
    torch.testing.assert_close(torch.autograd.grad(model(value).sum(),value)[0],torch.autograd.grad(exported(value).sum(),value)[0],rtol=0,atol=0)
    with pytest.raises(ValueError,match='AvgPool1d'):exported(torch.empty(0,2,dtype=torch.float64))


def test_cache_and_exact_training_resume(tmp_path):
    from runner.engine import run
    from runner.tests.test_conv1d_lifecycle import assert_nested_equal
    g=graph();a=preview(g,tmp_path/'a',registry=REGISTRY,device='cpu')
    b=preview(g,tmp_path/'b',registry=REGISTRY,device='cpu',resume_from='flat',cache=a['cachePath'])
    assert a['previews']==b['previews']
    options=dict(registry=REGISTRY,device='cpu',sample_limit=64,batch_size=32)
    run(g,tmp_path/'full',epochs=3,**options)
    run(g,tmp_path/'first',epochs=1,**options)
    run(g,tmp_path/'resumed',epochs=3,resume_checkpoint=tmp_path/'first'/'checkpoint.pt',**options)
    a=torch.load(tmp_path/'full'/'checkpoint.pt',weights_only=True)
    b=torch.load(tmp_path/'resumed'/'checkpoint.pt',weights_only=True)
    for key in ('modelState','optimizerState','rngState'):assert_nested_equal(a[key],b[key])


@pytest.mark.parametrize('include',[False,True])
def test_mps_forward_backward(include):
    if not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
    a=GraphModel(graph(include),REGISTRY);b=GraphModel(graph(include),REGISTRY).to('mps');b.load_state_dict(a.state_dict())
    x=torch.randn(4,2,requires_grad=True);y=x.detach().to('mps').requires_grad_()
    torch.testing.assert_close(a(x),b(y).cpu(),rtol=2e-5,atol=2e-6)
    torch.testing.assert_close(torch.autograd.grad(a(x).sum(),x)[0],torch.autograd.grad(b(y).sum(),y)[0].cpu(),rtol=2e-5,atol=2e-6)
