import json
import pytest
import torch
from runner.engine import GraphModel,REGISTRY,export_source,validate,run
from runner.preview import preview
from runner.tests.test_maxpool2d_contract import config
from runner.tests.test_maxpool1d_graph import graph as graph1d

@pytest.fixture
def registry(tmp_path):
    return REGISTRY


def graph(indices=True):
    g=graph1d(indices);nodes={n['id']:n for n in g['nodes']}
    nodes['expand']['config']['out_features']=40
    nodes['shaped']['config']['shape']=[2,4,5]
    nodes['convolution'].update(type='max_pool2d',config=config(stride=[1,2],return_indices=indices))
    nodes['head']['config']['in_features']=12
    return g

@pytest.mark.parametrize('indices',[False,True])
def test_window_oracle_gradient_and_export(registry,indices):
    m=GraphModel(graph(indices),registry).double()
    # Ties and noncontiguous storage; overlapping windows accumulate selected gradients.
    x=(torch.arange(80,dtype=torch.float64).reshape(1,2,4,10)%7)[...,::2].requires_grad_()
    assert not x.is_contiguous()
    expected=[];locations=[]
    for h in range(3):
        for w in (0,2):
            positions=[r*5+c for r in range(h,h+2) for c in range(w,w+3)]
            values,winners=x.flatten(2)[...,positions].max(-1)
            expected.append(values);locations.append(torch.tensor(positions)[winners])
    expected=torch.stack(expected,-1).reshape(1,2,3,2)
    result=m.execute_node('convolution',{'shaped':x},None)
    actual=result['out'] if indices else result
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(),x,retain_graph=True)[0],torch.autograd.grad(expected.sum(),x)[0],rtol=0,atol=0)
    if indices:assert torch.equal(result['indices'],torch.stack(locations,-1).reshape(1,2,3,2))
    ns={};exec(export_source(m),ns);exported=ns['Model']().double();exported.load_state_dict(m.state_dict())
    inp=torch.randn(3,2,dtype=torch.float64,requires_grad=True)
    torch.testing.assert_close(exported(inp),m(inp),rtol=0,atol=0)
    torch.testing.assert_close(torch.autograd.grad(exported(inp).sum(),inp)[0],torch.autograd.grad(m(inp).sum(),inp)[0],rtol=0,atol=0)
    with pytest.raises(ValueError,match='MaxPool2d'):exported(torch.empty(0,2,dtype=torch.float64))


def test_nullable_stride_and_typed_port_refusal(registry):
    g=graph();nodes={n['id']:n for n in g['nodes']}
    nodes['convolution']['config']['stride']=None;nodes['head']['config']['in_features']=4
    GraphModel(g,registry)(torch.ones(3,2))
    for enabled in (False,True):
        g=graph(enabled);next(e for e in g['edges'] if e['source']=='convolution')['sourcePort']='indices'
        with pytest.raises(ValueError,match='output|floating-point'):validate(g,registry)


def test_checkpoint_cache_and_saved_samples(tmp_path,registry):
    from runner.tests.test_conv1d_lifecycle import assert_nested_equal
    g=graph();options=dict(registry=registry,device='cpu',batch_size=32,sample_limit=64)
    full=tmp_path/'full';first=tmp_path/'first';resumed=tmp_path/'resumed'
    result=run(g,full,epochs=3,**options)
    run(g,first,epochs=1,**options)
    run(g,resumed,epochs=3,resume_checkpoint=first/'checkpoint.pt',**options)
    a=torch.load(full/'checkpoint.pt',weights_only=True);b=torch.load(resumed/'checkpoint.pt',weights_only=True)
    for key in ('modelState','optimizerState','rngState'):assert_nested_equal(a[key],b[key])
    assert json.loads((full/'activations.json').read_text())['previews']==result['previews']
    a=preview(g,tmp_path/'preview',registry=registry,device='cpu')
    b=preview(g,tmp_path/'reuse',registry=registry,device='cpu',resume_from='flat',cache=a['cachePath'])
    assert a['previews']==b['previews']
    assert b['previews']['convolution']['outputs']['indices']['dtype']=='int64'


@pytest.mark.parametrize('indices',[False,True])
def test_actual_mps_forward_backward_and_indices(registry,indices):
    if not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
    cpu=GraphModel(graph(indices),registry);mps=GraphModel(graph(indices),registry).to('mps')
    mps.load_state_dict(cpu.state_dict())
    x=torch.randn(5,2,requires_grad=True);xm=x.detach().to('mps').requires_grad_()
    y,stages=cpu(x,inspect=True);ym,ms=mps(xm,inspect=True)
    torch.testing.assert_close(y,ym.cpu(),rtol=2e-5,atol=2e-6)
    torch.testing.assert_close(torch.autograd.grad(y.sum(),x)[0],torch.autograd.grad(ym.sum(),xm)[0].cpu(),rtol=2e-5,atol=2e-6)
    if indices:assert torch.equal(stages['convolution']['indices'],ms['convolution']['indices'].cpu())


@pytest.mark.parametrize('indices',[False,True])
def test_all_padding_window_refuses_nonfinite_capture(registry,indices):
    from runner.product_execution import ProductExecutionAdapter
    g=graph(indices);nodes={n['id']:n for n in g['nodes']}
    nodes['expand']['config']['out_features']=2
    nodes['shaped']['config']['shape']=[2,1,1]
    nodes['convolution']['config']=config(kernel_size=[2,2],stride=[1,1],padding=[1,1],dilation=[2,2],return_indices=indices)
    nodes['head']['config']['in_features']=2
    m=GraphModel(g,registry)
    result=m.execute_node('convolution',{'shaped':torch.ones(3,2,1,1)},None)
    assert torch.isneginf(result['out'] if indices else result).all()
    with pytest.raises(ValueError,match='Captured component convolution.*finite'):
        ProductExecutionAdapter(m).validate_stage(result,'convolution',rows=3)
