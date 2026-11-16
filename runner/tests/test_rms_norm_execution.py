"""Independent RMS formula, state, derivative and generated-source checks."""
import copy
import pytest
import torch
from runner.engine import GraphModel, REGISTRY, export_source
from runner.tests.test_layer_norm_execution import graph as layer_graph


def graph(shape=(3,), affine=True, eps=1e-5):
    g=layer_graph(list(shape),True,False)
    g['nodes'][3].update(type='rms_norm',config=dict(normalized_shape=list(shape),eps=eps,elementwise_affine=affine))
    return g


@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('shape',[(3,),(2,3)])
@pytest.mark.parametrize('affine',[False,True])
def test_formula_gradients_no_centering_state_and_export(device,shape,affine):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
    torch.manual_seed(53)
    model=GraphModel(graph(shape,affine),REGISTRY).to(device)
    layer=model.layers[model.layer_keys['probabilities']]
    if affine:
        with torch.no_grad():layer.weight.copy_(torch.linspace(.5,1.5,layer.weight.numel(),device=device).reshape(shape))
    x=torch.tensor([[.2,-.4],[3.,1.],[-2.,.1]],device=device,requires_grad=True)
    result,stages=model(x,inspect=True);hidden=stages['shape']
    axes=tuple(range(hidden.ndim-len(shape),hidden.ndim))
    expected=hidden/torch.sqrt(hidden.square().mean(axes,keepdim=True)+1e-5)
    if affine:expected=expected*layer.weight
    torch.testing.assert_close(stages['probabilities'],expected,atol=3e-5,rtol=3e-5)
    upstream=torch.linspace(-2,3,hidden.numel(),device=device).reshape(hidden.shape)
    targets=[hidden]+list(layer.parameters())
    actual=torch.autograd.grad(stages['probabilities'],targets,upstream,retain_graph=True)
    oracle=torch.autograd.grad(expected,targets,upstream)
    for a,b in zip(actual,oracle):torch.testing.assert_close(a,b,atol=5e-5,rtol=5e-5)
    assert len(list(layer.parameters()))==int(affine)
    assert not list(layer.buffers()) and not hasattr(layer,'bias')
    assert (stages['probabilities'].mean(axes).abs()>1e-3).any(), 'RMS must not subtract mean'
    model.eval();torch.testing.assert_close(model(x),result,atol=0,rtol=0)
    torch.testing.assert_close(model(x[:1]),result[:1],atol=3e-5,rtol=3e-5)
    scope={};exec(export_source(model),scope);replay=scope['Model']().to(device)
    replay.load_state_dict(model.state_dict(),strict=True)
    torch.testing.assert_close(replay(x),result,atol=0,rtol=0)


@pytest.mark.parametrize('mutation',[
    {'bias':True},{'eps':0},{'eps':float('nan')},{'eps':True},
    {'elementwise_affine':1},{'normalized_shape':[True]},
    {'normalized_shape':[6]},{'normalized_shape':[1,2,3]},
])
def test_refuses_invalid_configuration_and_batch_reduction(mutation):
    g=graph();g['nodes'][3]['config'].update(mutation)
    with pytest.raises(ValueError):GraphModel(g,REGISTRY)


def test_zero_constant_epsilon_and_independent_finite_difference():
    m=GraphModel(graph((2,3)),REGISTRY).double();layer=m.layers[m.layer_keys['probabilities']]
    z=torch.zeros(1,2,3,dtype=torch.float64,requires_grad=True)
    y=layer(z);assert torch.isfinite(y).all()
    assert torch.isfinite(torch.autograd.grad(y.sum(),z)[0]).all()
    constant=layer(torch.full((1,2,3),7.,dtype=torch.float64))
    assert (constant>.99).all(), 'Constant nonzero input is not centered to zero'
    x=torch.tensor([[[.2,-.4,1.1],[2.,-.3,.7]]],dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda a,w: torch.func.functional_call(layer,{'weight':w},(a,)),(x,layer.weight),eps=1e-6,atol=1e-5)
    checkpoint=copy.deepcopy(m.state_dict())
    opt=torch.optim.Adam(m.parameters(),lr=.01);opt.zero_grad();m(torch.ones(2,2,dtype=torch.float64)).sum().backward();opt.step()
    assert not torch.equal(checkpoint['layers.'+m.layer_keys['probabilities']+'.weight'],m.state_dict()['layers.'+m.layer_keys['probabilities']+'.weight'])
    m.load_state_dict(checkpoint,strict=True)
    assert torch.equal(checkpoint['layers.'+m.layer_keys['probabilities']+'.weight'],m.state_dict()['layers.'+m.layer_keys['probabilities']+'.weight'])

@pytest.mark.parametrize('affine',[False,True])
def test_real_training_exact_continuation_cache_and_export(tmp_path,affine):
    import json
    from runner.engine import run
    from runner.preview import preview
    from runner.tests.test_checkpoint_lifecycle import state,assert_nested_equal
    value=graph((2,3),affine);full,first,resumed=(tmp_path/name for name in ('full','first','resumed'))
    options=dict(registry=REGISTRY,device='cpu',batch_size=64,sample_limit=128)
    run(value,full,epochs=4,gradient_capture={'schemaVersion':1,'everyEpochs':1,'maxBatchesPerEpoch':1,'maxCaptures':4},**options)
    run(value,first,epochs=2,**options);parent=(first/'checkpoint.pt').read_bytes()
    run(value,resumed,epochs=4,resume_checkpoint=first/'checkpoint.pt',**options)
    for key in ('modelState','optimizerState','rngState'):assert_nested_equal(state(full)[key],state(resumed)[key])
    assert json.loads((full/'metrics.json').read_text())==json.loads((resumed/'metrics.json').read_text())
    assert (first/'checkpoint.pt').read_bytes()==parent
    inventory=json.loads((full/'gradient-summary.json').read_text())['inventory']['parameters']
    norm=[p for p in inventory if p['nodeID']=='probabilities']
    assert len(norm)==int(affine) and all(p['shape']==[2,3] for p in norm)
    scope={};exec((full/'model.py').read_text(),scope);exported=scope['Model']();exported.load_state_dict(state(full)['modelState'],strict=True)
    model=GraphModel(value,REGISTRY);model.load_state_dict(state(full)['modelState'],strict=True)
    x=torch.tensor([[.2,.3],[-1.,2.]])
    torch.testing.assert_close(exported(x),model(x),atol=0,rtol=0)
    captured=preview(value,tmp_path/'preview',registry=REGISTRY,device='cpu',checkpoint=full/'checkpoint.pt')
    reused=preview(value,tmp_path/'reused',registry=REGISTRY,device='cpu',checkpoint=full/'checkpoint.pt',resume_from='probabilities',cache=captured['cachePath'])
    assert reused['previews']==captured['previews']
    changed=graph((2,3),affine,eps=.1)
    with pytest.raises(ValueError,match='semantic architecture'):
        run(changed,tmp_path/'bad-resume',epochs=5,resume_checkpoint=first/'checkpoint.pt',**options)
    with pytest.raises(ValueError):
        preview(changed,tmp_path/'bad-cache',registry=REGISTRY,device='cpu',checkpoint=full/'checkpoint.pt',resume_from='probabilities',cache=captured['cachePath'])
    assert (first/'checkpoint.pt').read_bytes()==parent
