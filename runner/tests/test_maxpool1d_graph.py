import json
import pytest
import torch
from runner.engine import GraphModel,REGISTRY,export_source,validate
from runner.preview import preview
from runner.tests.test_maxpool1d_contract import config
from runner.tests.test_conv1d_graph import graph as conv_graph

@pytest.fixture
def registry(tmp_path):
    return REGISTRY

def graph(indices=True):
    g=conv_graph();n=next(n for n in g['nodes'] if n['id']=='convolution')
    n['type']='max_pool1d';n['config']=config(kernel_size=3,stride=2,padding=1,ceil_mode=True,return_indices=indices)
    next(n for n in g['nodes'] if n['id']=='head')['config']['in_features']=8
    return g

@pytest.mark.parametrize('indices',[False,True])
def test_pool_forward_gradients_and_export(registry,indices):
    g=graph(indices);m=GraphModel(g,registry).double()
    x=torch.tensor([[[-3.,2.,1.,4.,4.,-1.,3.],[-7.,-2.,-4.,-1.,-5.,-6.,-3.]]],requires_grad=True,dtype=torch.float64)
    result=m.execute_node('convolution',{'shaped':x},None)
    out=result['out'] if indices else result
    # Explicit valid positions of each overlapping window, first winner on ties.
    windows=[[0,1],[1,2,3],[3,4,5],[5,6]]
    expected=torch.stack([x[...,positions].max(dim=-1).values for positions in windows],-1)
    torch.testing.assert_close(out,expected,rtol=0,atol=0)
    torch.testing.assert_close(torch.autograd.grad(out.sum(),x,retain_graph=True)[0],torch.autograd.grad(expected.sum(),x)[0],rtol=0,atol=0)
    if indices:assert torch.equal(result['indices'],torch.tensor([[[1,3,3,6],[1,3,3,6]]]))
    ns={};exec(export_source(m),ns);exported=ns['Model']().double();exported.load_state_dict(m.state_dict())
    inp=torch.randn(4,2,dtype=torch.float64,requires_grad=True)
    torch.testing.assert_close(m(inp),exported(inp),rtol=0,atol=0)
    torch.testing.assert_close(torch.autograd.grad(m(inp).sum(),inp)[0],torch.autograd.grad(exported(inp).sum(),inp)[0],rtol=0,atol=0)


def test_indices_cannot_feed_float_layer_and_disabled_indices_refuse(registry):
    for enabled in (False,True):
        g=graph(enabled)
        next(e for e in g['edges'] if e['source']=='convolution')['sourcePort']='indices'
        with pytest.raises(ValueError,match='output|floating-point'):validate(g,registry)


def test_real_pooling_preview_resume_retains_indices(tmp_path,registry):
    g=graph();a=preview(g,tmp_path/'first',registry=registry,device='cpu')
    b=preview(g,tmp_path/'second',registry=registry,device='cpu',resume_from='flat',cache=a['cachePath'])
    assert a['previews']==b['previews']
    assert b['previews']['convolution']['outputs']['indices']['dtype']=='int64'


def test_train_resume_and_port_capture(tmp_path,registry):
    from runner.engine import run
    from runner.tests.test_conv1d_lifecycle import assert_nested_equal
    g=graph();options=dict(registry=registry,device='cpu',batch_size=32,sample_limit=64)
    full=tmp_path/'full';first=tmp_path/'first';resumed=tmp_path/'resumed'
    result=run(g,full,epochs=3,**options)
    run(g,first,epochs=1,**options)
    run(g,resumed,epochs=3,resume_checkpoint=first/'checkpoint.pt',**options)
    a=torch.load(full/'checkpoint.pt',weights_only=True);b=torch.load(resumed/'checkpoint.pt',weights_only=True)
    for key in ('modelState','optimizerState','rngState'):assert_nested_equal(a[key],b[key])
    assert result['previews']['convolution']['outputs']['indices']['dtype']=='int64'
    import hashlib
    saved=(full/'activations.json').read_bytes()
    artifact=json.loads(saved);manifest=json.loads((full/'manifest.json').read_bytes())
    assert artifact['previews']==result['previews']
    assert artifact['runID']==result['runID'] and artifact['graphHash']==manifest['graphHash']
    assert manifest['activationPreview']==dict(schemaVersion=1,file='activations.json',sha256=hashlib.sha256(saved).hexdigest(),bytes=len(saved))



def test_capture_policy_omits_both_values_and_indices(tmp_path,registry):
    from runner.engine import run
    result=run(graph(),tmp_path/'private',registry=registry,device='cpu',epochs=1,sample_limit=64,
               sample_capture={'schemaVersion':1,'activationValues':False,'predictionRows':False})
    captured=result['previews']['convolution']
    assert captured['captureStatus']=='omittedByPolicy'
    assert all(item['values']==[] for item in captured['outputs'].values())
    assert result['predictionRows']==[]
    artifact=json.loads((tmp_path/'private'/'activations.json').read_bytes())
    assert artifact['previews']==result['previews']
    assert artifact['sampleCapture']=={'schemaVersion':1,'activationValues':False,'predictionRows':False}


@pytest.mark.parametrize('indices',[False,True])
def test_actual_mps_matches_cpu_forward_and_gradient(registry,indices):
    if not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
    g=graph(indices);cpu=GraphModel(g,registry);mps=GraphModel(g,registry).to('mps')
    mps.load_state_dict(cpu.state_dict())
    x=torch.randn(5,2,requires_grad=True);xm=x.detach().to('mps').requires_grad_()
    y,stages=cpu(x,inspect=True);ym,ms=mps(xm,inspect=True)
    torch.testing.assert_close(y,ym.cpu(),rtol=2e-5,atol=2e-6)
    torch.testing.assert_close(torch.autograd.grad(y.sum(),x)[0],torch.autograd.grad(ym.sum(),xm)[0].cpu(),rtol=2e-5,atol=2e-6)
    if indices:assert torch.equal(stages['convolution']['indices'],ms['convolution']['indices'].cpu())


def test_export_refuses_empty_authored_batch_and_keeps_label_safe(registry):
    authored=graph();authored['nodes'][0]['name']='tuple'
    model=GraphModel(authored,registry)
    namespace={};exec(export_source(model),namespace)
    exported=namespace['Model']();exported.load_state_dict(model.state_dict())
    with pytest.raises(ValueError,match='MaxPool1d.*authored sample shape'):
        exported(torch.empty(0,2))
    value=torch.randn(3,2)
    torch.testing.assert_close(exported(value),model(value))


@pytest.mark.parametrize('indices',[False,True])
def test_noncontiguous_pool_input_preserves_values_indices_and_gradients(registry,indices):
    model=GraphModel(graph(indices),registry)
    x=torch.randn(3,2,14)[...,::2].requires_grad_()
    assert not x.is_contiguous()
    contiguous=x.detach().contiguous().requires_grad_()
    actual=model.execute_node('convolution',{'shaped':x},None)
    expected=model.execute_node('convolution',{'shaped':contiguous},None)
    a=actual['out'] if indices else actual
    b=expected['out'] if indices else expected
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    if indices:assert torch.equal(actual['indices'],expected['indices'])
    torch.testing.assert_close(torch.autograd.grad(a.sum(),x)[0],
                               torch.autograd.grad(b.sum(),contiguous)[0],rtol=0,atol=0)


@pytest.mark.parametrize('indices',[False,True])
def test_all_padding_dilated_window_has_valid_geometry_but_capture_refuses_infinity(registry,indices):
    from runner.product_execution import ProductExecutionAdapter
    authored=graph(indices)
    nodes={n['id']:n for n in authored['nodes']}
    nodes['expand']['config']['out_features']=2
    nodes['shaped']['config']['shape']=[2,1]
    nodes['convolution']['config']=config(kernel_size=2,stride=1,padding=1,dilation=2,
                                          return_indices=indices)
    nodes['head']['config']['in_features']=2
    model=GraphModel(authored,registry)
    result=model.execute_node('convolution',{'shaped':torch.ones(3,2,1)},None)
    out=result['out'] if indices else result
    assert tuple(out.shape)==(3,2,1) and torch.isneginf(out).all()
    with pytest.raises(ValueError,match='Captured component convolution.*finite'):
        ProductExecutionAdapter(model).validate_stage(result,'convolution',rows=3)
