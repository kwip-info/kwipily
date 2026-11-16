"""Softmax backend contract; custom registry until native integration is complete."""
from runner.tests.cli_discovery import capabilities_cli
import copy
import json
import pytest
import torch
from runner.engine import GraphModel, REGISTRY, export_source, validate
from runner.graph_semantics import semantic_graph

@pytest.fixture
def registry(tmp_path):
    data=json.loads(REGISTRY.read_text())
    if not any(c['id']=='softmax' for c in data['components']):data['components'].append({'id':'softmax','title':'Softmax','category':'Activations','inputs':['in'],'outputs':['out'],'fields':[{'key':'dim','title':'Axis','kind':'integer','defaultValue':-1,'min':-8,'max':7}]})
    path=tmp_path/'registry.json';path.write_text(json.dumps(data));return path

def graph(dim=-1):
    specs=[('input','input',{'features':2}),('project','linear',{'in_features':2,'out_features':6,'bias':True}),('shape','reshape',{'shape':[2,3]}),('probabilities','softmax',{'dim':dim}),('flat','flatten',{'start_dim':1,'end_dim':-1}),('head','linear',{'in_features':6,'out_features':2,'bias':False}),('output','output',{})]
    return {'schemaVersion':2,'name':'Softmax hidden normalization','nodes':[{'id':i,'type':t,'name':i,'x':0,'y':0,'config':c} for i,t,c in specs],'edges':[{'id':str(i),'source':specs[i][0],'sourcePort':'out','target':specs[i+1][0],'targetPort':'in'} for i in range(len(specs)-1)]}

@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('dim',[1,2,-1,-2])
def test_forward_gradient_export_and_batch_isolation(registry,device,dim):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('Actual MPS unavailable')
    torch.manual_seed(97)
    model=GraphModel(graph(dim),registry).to(device)
    x=torch.tensor([[1000.,-1000.],[.5,-2.],[1.,3.]],device=device,requires_grad=True)
    result,stages=model(x,inspect=True)
    hidden=stages['shape'];actual=stages['probabilities']
    exp=(hidden-hidden.amax(dim=dim,keepdim=True)).exp()
    expected=exp/exp.sum(dim=dim,keepdim=True)
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(actual.sum(dim=dim),torch.ones_like(actual.sum(dim=dim)))
    upstream=torch.linspace(-2,3,actual.numel(),device=device).reshape(actual.shape)
    gradient=torch.autograd.grad(actual,hidden,upstream,retain_graph=True)[0]
    oracle=actual*(upstream-(actual*upstream).sum(dim=dim,keepdim=True))
    torch.testing.assert_close(gradient,oracle,atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(model(x[:1]),result[:1],atol=1e-5,rtol=1e-5)
    scope={};source=export_source(model);exec(source,scope)
    exported=scope['Model']().to(device);exported.load_state_dict(model.state_dict(),strict=True)
    assert f'dim={dim}' in source
    torch.testing.assert_close(exported(x),result,atol=0,rtol=0)
    assert all('probabilities' not in key for key in model.state_dict())

@pytest.mark.parametrize('config',[{}, {'dim':True},{'dim':1.5},{'dim':'1'},{'dim':None},{'dim':0},{'dim':8},{'dim':-9},{'dim':float('inf')},{'dim':10**400},{'dim':1,'other':1}])
def test_malformed_config_rejected(registry,config):
    value=graph();value['nodes'][3]['config']=config
    with pytest.raises(ValueError):validate(value,registry)
    with pytest.raises(ValueError):semantic_graph(value)

@pytest.mark.parametrize('dim',[3,-3,-4])
def test_rank_and_batch_axis_rejected_before_allocation(registry,dim):
    with pytest.raises(ValueError,match='[Aa]xis|batch'):GraphModel(graph(dim),registry)

def test_schema_and_configuration_identity(registry):
    value=graph();old=copy.deepcopy(value);old['schemaVersion']=1
    with pytest.raises(ValueError):validate(old,registry)
    with pytest.raises(ValueError):semantic_graph(old)
    assert semantic_graph(graph(1))!=semantic_graph(graph(2))
    assert semantic_graph(graph(1.0))==semantic_graph(graph(1))
    model=GraphModel(graph(1.0),registry)
    assert type(model.nodes['probabilities']['config']['dim']) is int

def test_cli_discovery_advertises_reviewed_softmax_contract():
    import subprocess,sys
    result=capabilities_cli()
    assert json.loads(result.stdout)['componentCapabilities']['softmax']=={
        'version':{'major':1,'minor':0,'patch':0},
        'adapterID':'kwip.runner.graph-v2.softmax',
        'graphSchemaVersions':[2],'maximumRank':8,'preservesBatch':True}

@pytest.mark.parametrize('dim',[1,2])
def test_training_resume_capture_export_and_cache_identity(tmp_path,registry,dim):
    from runner.engine import run
    from runner.preview import preview
    from runner.tests.test_checkpoint_lifecycle import state,assert_nested_equal
    value=graph(dim);full=tmp_path/'full';first=tmp_path/'first';resumed=tmp_path/'resumed'
    options={'registry':registry,'device':'cpu','batch_size':64,'sample_limit':128}
    run(value,full,epochs=4,gradient_capture={'schemaVersion':1,'everyEpochs':1,'maxBatchesPerEpoch':1,'maxCaptures':4},**options)
    run(value,first,epochs=2,**options)
    parent_bytes=(first/'checkpoint.pt').read_bytes()
    run(value,resumed,epochs=4,resume_checkpoint=first/'checkpoint.pt',**options)
    for key in ('modelState','optimizerState','rngState'):
        assert_nested_equal(state(full)[key],state(resumed)[key])
    assert json.loads((full/'metrics.json').read_text())==json.loads((resumed/'metrics.json').read_text())
    assert (first/'checkpoint.pt').read_bytes()==parent_bytes
    report=json.loads((full/'gradient-summary.json').read_text())
    assert all(p['nodeID']!='probabilities' for p in report['inventory']['parameters'])
    model=GraphModel(value,registry);model.load_state_dict(state(full)['modelState'],strict=True)
    scope={};exec((full/'model.py').read_text(),scope)
    exported=scope['Model']();exported.load_state_dict(state(full)['modelState'],strict=True)
    x=torch.tensor([[.2,.3],[-1.,2.]])
    torch.testing.assert_close(exported(x),model(x),atol=0,rtol=0)
    a=preview(value,tmp_path/'preview',registry=registry,device='cpu',checkpoint=full/'checkpoint.pt')
    b=preview(value,tmp_path/'cached',registry=registry,device='cpu',checkpoint=full/'checkpoint.pt',resume_from='probabilities',cache=a['cachePath'])
    assert a['previews']==b['previews']
    changed=graph(2 if dim==1 else 1)
    with pytest.raises(ValueError,match='semantic architecture'):
        run(changed,tmp_path/'bad-resume',epochs=5,resume_checkpoint=first/'checkpoint.pt',**options)
    fresh=preview(value,tmp_path/'fresh',registry=registry,device='cpu')
    with pytest.raises(ValueError,match='stale|match'):
        preview(changed,tmp_path/'bad-cache',registry=registry,device='cpu',resume_from='probabilities',cache=fresh['cachePath'])
    assert not (tmp_path/'bad-resume/manifest.json').exists()

@pytest.mark.parametrize('device',['cpu','mps'])
def test_finetune_uses_saved_weights_with_fresh_recipe(tmp_path,registry,device):
    from runner.engine import run
    from runner.tests.test_checkpoint_lifecycle import state
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('Actual MPS unavailable')
    parent=tmp_path/'parent';child=tmp_path/'child';value=graph(1)
    run(value,parent,registry=registry,device=device,epochs=2,batch_size=64,sample_limit=128)
    original=(parent/'checkpoint.pt').read_bytes()
    events=[]
    run(value,child,registry=registry,device=device,epochs=1,batch_size=128,sample_limit=128,seed=19,split_seed=29,learning_rate=.01,finetune_checkpoint=parent/'checkpoint.pt',emit=events.append)
    result=state(child);manifest=json.loads((child/'manifest.json').read_text())
    assert result['epoch']==1 and result['seed']==19
    assert manifest['lineage']['mode']=='weightsOnlyFineTune'
    assert manifest['provenance']['splitSeed']==29
    assert events[0]['initialEpoch']==0
    assert all(value['step'].item()==1 for value in result['optimizerState']['state'].values())
    assert (parent/'checkpoint.pt').read_bytes()==original
    model=GraphModel(value,registry).to(device)
    model.load_state_dict(result['modelState'],strict=True)
    output=model(torch.tensor([[.1,.2]],device=device))
    assert output.shape==(1,2) and torch.isfinite(output).all()
