import copy,json
import pytest
import torch
from runner.engine import GraphModel,validate,export_source,run
from runner.tests.test_runner import graph
from runner.checkpoint_lifecycle import semantic_graph
from runner.preview import preview

def v2(bias=True):
    g=graph();g['schemaVersion']=2
    for n in g['nodes']:
        if n['type']=='linear':n['config']['bias']=bias
    return g

def test_semantic_legacy_true_equivalent_false_distinct():
    assert semantic_graph(graph())==semantic_graph(v2())
    assert semantic_graph(graph())!=semantic_graph(v2(False))

@pytest.mark.parametrize('value',[None,0,1,'true',[],{}])
def test_bias_requires_boolean(value):
    g=v2();g['nodes'][1]['config']['bias']=value
    with pytest.raises(ValueError):validate(g)

def test_v1_remains_strict_and_v2_requires_bias():
    g=graph();g['nodes'][1]['config']['bias']=True
    with pytest.raises(ValueError):validate(g)
    g=v2();del g['nodes'][1]['config']['bias']
    with pytest.raises(ValueError):validate(g)

def test_biasfalse_output_gradient_export_and_count():
    model=GraphModel(v2(False));x=torch.randn(7,2,requires_grad=True)
    assert all(layer.bias is None for layer in model.layers.values())
    assert sum(p.numel() for p in model.parameters())==sum(layer.in_features*layer.out_features for layer in model.layers.values())
    namespace={};exec(export_source(model),namespace);exported=namespace['Model']();exported.load_state_dict(model.state_dict(),strict=True)
    assert torch.equal(model(x),exported(x))
    actual=torch.autograd.grad(model(x).sum(),(x,*model.parameters()))
    expected=torch.autograd.grad(exported(x).sum(),(x,*exported.parameters()))
    assert all(torch.equal(a,b) for a,b in zip(actual,expected))

def test_legacy_resume_and_cache_migration(tmp_path):
    run(graph(),tmp_path/'parent',epochs=2,device='cpu')
    run(v2(),tmp_path/'resumed',epochs=4,device='cpu',resume_checkpoint=tmp_path/'parent/checkpoint.pt')
    run(graph(),tmp_path/'full',epochs=4,device='cpu')
    a=torch.load(tmp_path/'resumed/checkpoint.pt',weights_only=True);b=torch.load(tmp_path/'full/checkpoint.pt',weights_only=True)
    assert all(torch.equal(a['modelState'][k],v) for k,v in b['modelState'].items())
    old=preview(graph(),tmp_path/'old',device='cpu')
    result=preview(v2(),tmp_path/'new',device='cpu',resume_from='head',cache=old['cachePath'])
    assert result['reusedNodeIDs']
    with pytest.raises(ValueError,match='stale'):
        preview(v2(False),tmp_path/'badcache',device='cpu',resume_from='head',cache=old['cachePath'])
    with pytest.raises(ValueError,match='architecture'):
        run(v2(False),tmp_path/'badresume',epochs=4,device='cpu',resume_checkpoint=tmp_path/'parent/checkpoint.pt')

@pytest.mark.parametrize('device',['cpu','mps'])
def test_biasfalse_independent_formula_and_gradients(device):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
    model=GraphModel(v2(False)).to(device);layer=model.layers['left']
    x=torch.randn(7,2,device=device,requires_grad=True);g=torch.randn(7,layer.out_features,device=device)
    actual=layer(x);expected=x @ layer.weight.T
    torch.testing.assert_close(actual,expected)
    dx,dw=torch.autograd.grad(actual,(x,layer.weight),g)
    torch.testing.assert_close(dx,g @ layer.weight)
    torch.testing.assert_close(dw,g.T @ x)

def test_composite_legacy_true_comparison_and_false_rejection(tmp_path):
    from runner.tests.test_composite_source import fixture
    from runner.composite_source import load_source
    g,path,sha,_=fixture(tmp_path);g['schemaVersion']=2
    for n in g['nodes']:
        if n['type']=='linear':n['config']['bias']=True
    assert load_source(path,sha,g)
    g['nodes'][1]['config']['bias']=False
    with pytest.raises(ValueError,match='architecture'):load_source(path,sha,g)

def test_affine_false_rejected_with_actionable_reason():
    from runner.tests.test_learning_evidence import affine_request,affine_submission
    from runner.learning_evidence import generate_evidence
    submission=affine_submission();submission['graph']['schemaVersion']=2
    submission['graph']['nodes'][1]['config']['bias']=False
    evidence=generate_evidence(affine_request(),submission,submission_digest='b'*64)
    assert not evidence['fixtures']
    assert 'Enable bias' in evidence['backendError']

def test_registry2_runs_legacy_and_new_graphs(tmp_path):
    from runner.engine import REGISTRY
    registry=json.loads(REGISTRY.read_text());registry['schemaVersion']=2
    for component in registry['components']:
        for field in component['fields']:field.setdefault('kind','integer')
        if component['id']=='linear' and not any(f['key']=='bias' for f in component['fields']):
            component['fields'].append({'key':'bias','title':'Bias','kind':'boolean','defaultValue':True})
    path=tmp_path/'registry.json';path.write_text(json.dumps(registry))
    assert GraphModel(graph(),path).layers['left'].bias is not None
    assert GraphModel(v2(False),path).layers['left'].bias is None

def test_legacy_float_width_checkpoint_resumes_after_integer_migration(tmp_path):
    from runner.checkpoint_lifecycle import digest
    old=graph()
    for node in old['nodes']:
        for key,value in node['config'].items():node['config'][key]=float(value)
    run(old,tmp_path/'oldfloat',epochs=1,device='cpu')
    # Emulate the old implementation's valid representation-sensitive digest.
    path=tmp_path/'oldfloat/checkpoint.pt';checkpoint=torch.load(path,weights_only=True)
    legacy=semantic_graph(old)
    for node in legacy['nodes']:
        for key,value in node['config'].items():node['config'][key]=float(value)
    old_digest=digest(legacy);checkpoint['contract']['semanticGraphDigest']=old_digest
    torch.save(checkpoint,path)
    import hashlib
    manifest_path=path.with_name('manifest.json');manifest=json.loads(manifest_path.read_text())
    manifest['checkpoint']['contract']['semanticGraphDigest']=old_digest
    manifest['checkpoint']['sha256']=hashlib.sha256(path.read_bytes()).hexdigest();manifest['checkpoint']['bytes']=path.stat().st_size
    manifest_path.write_text(json.dumps(manifest))
    run(v2(),tmp_path/'resumedfloat',epochs=2,device='cpu',resume_checkpoint=path)
    run(v2(),tmp_path/'fullint',epochs=2,device='cpu')
    a=torch.load(tmp_path/'resumedfloat/checkpoint.pt',weights_only=True);b=torch.load(tmp_path/'fullint/checkpoint.pt',weights_only=True)
    assert all(torch.equal(a['modelState'][k],v) for k,v in b['modelState'].items())
    # Historic digest must be verified, not merely ignored during migration.
    manifest['graph']['nodes'][1]['config']['in_features']=3.0
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='architecture'):
        run(v2(),tmp_path/'tampered',epochs=2,device='cpu',resume_checkpoint=path)

def test_whole_number_identity_rejects_fractional_configuration():
    old=graph()
    for node in old['nodes']:
        for key,value in node['config'].items():node['config'][key]=float(value)
    from runner.checkpoint_lifecycle import digest
    assert digest(semantic_graph(old))==digest(semantic_graph(v2()))
    old['nodes'][1]['config']['in_features']=2.5
    with pytest.raises(ValueError):semantic_graph(old)
