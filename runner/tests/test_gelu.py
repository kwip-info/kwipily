from runner.tests.cli_discovery import capabilities_cli
import copy
import json
import math
import pytest
import torch
from runner.engine import GraphModel,REGISTRY,export_source,run,validate
from runner.graph_semantics import semantic_graph
from runner.preview import preview
from runner.tests.test_checkpoint_lifecycle import state,assert_nested_equal
from runner.tests.test_runner import graph as legacy_graph

@pytest.fixture
def registry(tmp_path):
    data=json.loads(REGISTRY.read_text())
    if not any(c['id']=='gelu' for c in data['components']):data['components'].append({'id':'gelu','title':'GELU','category':'Activations','inputs':['in'],'outputs':['out'],'fields':[{'key':'approximate_tanh','title':'Tanh approximation','kind':'boolean','defaultValue':False}]})
    path=tmp_path/'registry.json';path.write_text(json.dumps(data));return path

def graph(mode=False,width=2):
    specs=[('input','input',{'features':2}),('first','linear',{'in_features':2,'out_features':width,'bias':True}),('activation','gelu',{'approximate_tanh':mode}),('last','linear',{'in_features':width,'out_features':2,'bias':False}),('output','output',{})]
    return {'schemaVersion':2,'name':'GELU graph','nodes':[{'id':i,'type':t,'name':i,'x':0,'y':0,'config':c} for i,t,c in specs],'edges':[{'id':str(n),'source':specs[n][0],'sourcePort':'out','target':specs[n+1][0],'targetPort':'in'} for n in range(len(specs)-1)]}

def oracle(value,mode):
    return .5*value*(1+torch.tanh(math.sqrt(2/math.pi)*(value+.044715*value**3))) if mode else .5*value*(1+torch.erf(value/math.sqrt(2)))

@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('mode',[False,True])
@pytest.mark.parametrize('batch,width',[(1,2),(7,512)])
def test_forward_backward_formula_and_export(device,mode,batch,width,registry):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('Actual MPS unavailable')
    torch.manual_seed(82);model=GraphModel(graph(mode,width),registry).to(device)
    storage=torch.linspace(-4,4,batch*4,device=device).reshape(batch,4)
    x=storage[:,::2].detach().requires_grad_();assert not x.is_contiguous()
    actual=model(x)
    hidden=model.layers['first'](x);expected=model.layers['last'](oracle(hidden,mode))
    torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-5)
    variables=(x,*model.parameters())
    a=torch.autograd.grad(actual.sum(),variables,retain_graph=True)
    b=torch.autograd.grad(expected.sum(),variables)
    for left,right in zip(a,b):torch.testing.assert_close(left,right,atol=5e-5,rtol=5e-5)
    scope={};source=export_source(model);exec(source,scope)
    exported=scope['Model']().to(device);exported.load_state_dict(model.state_dict(),strict=True)
    assert set(exported.state_dict())==set(model.state_dict())
    assert "approximate='tanh'" in source if mode else "approximate='none'" in source
    torch.testing.assert_close(exported(x),actual,atol=0,rtol=0)

@pytest.mark.parametrize('config',[{}, {'approximate_tanh':0},{'approximate_tanh':'tanh'},{'approximate_tanh':None},{'approximate_tanh':False,'extra':1}])
def test_malformed_config_rejected(config,registry):
    g=graph();g['nodes'][2]['config']=config
    with pytest.raises(ValueError):validate(g,registry)
    with pytest.raises(ValueError):semantic_graph(g)

def test_schema_identity_and_old_graphs_unchanged(registry):
    g=graph();old=copy.deepcopy(g);old['schemaVersion']=1
    for n in old['nodes']:
        if n['type']=='linear':n['config'].pop('bias')
    with pytest.raises(ValueError,match='schema'):validate(old,registry)
    assert semantic_graph(graph(False))!=semantic_graph(graph(True))
    assert semantic_graph(graph(False))['nodes'][0]['config']=={'approximate_tanh':False}
    legacy=legacy_graph();normalized=semantic_graph(legacy)
    migrated=copy.deepcopy(legacy);migrated['schemaVersion']=2
    for n in migrated['nodes']:
        if n['type']=='linear':n['config']['bias']=True
    assert semantic_graph(migrated)==normalized

@pytest.mark.parametrize('mode',[False,True])
def test_training_resume_preview_and_gradient_capture_lifecycle(tmp_path,registry,mode):
    g=graph(mode);full=tmp_path/'full';first=tmp_path/'first';resumed=tmp_path/'resumed'
    capture={'schemaVersion':1,'everyEpochs':1,'maxBatchesPerEpoch':1,'maxCaptures':4}
    run(g,full,epochs=4,device='cpu',registry=registry,gradient_capture=capture)
    run(g,first,epochs=2,device='cpu',registry=registry)
    run(g,resumed,epochs=4,device='cpu',registry=registry,resume_checkpoint=first/'checkpoint.pt')
    assert_nested_equal(state(full),state(resumed))
    report=json.loads((full/'gradient-summary.json').read_text())
    assert all(p['nodeID']!='activation' for p in report['inventory']['parameters'])
    a=preview(g,tmp_path/'a',device='cpu',registry=registry,checkpoint=full/'checkpoint.pt')
    b=preview(g,tmp_path/'b',device='cpu',registry=registry,checkpoint=full/'checkpoint.pt',resume_from='activation',cache=a['cachePath'])
    assert a['previews']==b['previews']
    changed=graph(not mode)
    with pytest.raises(ValueError,match='semantic architecture'):run(changed,tmp_path/'bad-resume',epochs=5,device='cpu',registry=registry,resume_checkpoint=full/'checkpoint.pt')
    fresh=preview(g,tmp_path/'fresh',device='cpu',registry=registry)
    with pytest.raises(ValueError,match='stale|match'):preview(changed,tmp_path/'bad-cache',device='cpu',registry=registry,resume_from='activation',cache=fresh['cachePath'])

def test_discovery_cli_and_composite_configuration_versions(tmp_path,registry):
    import subprocess,sys
    from runner.composite_source import validate_bodies
    result=capabilities_cli()
    capability=json.loads(result.stdout)['componentCapabilities']['gelu']
    assert capability=={'version':{'major':1,'minor':0,'patch':0},'adapterID':'kwip.runner.graph-v2.gelu','graphSchemaVersions':[2],'approximateTanhModes':[False,True]}
    body={'configurationVersion':2,'nodes':[graph()['nodes'][2]],'instances':[]}
    document={'root':body,'definitions':[]};validate_bodies(document)
    body['configurationVersion']=1
    with pytest.raises(ValueError,match='schema'):validate_bodies(document)
    source=tmp_path/'graph.json';source.write_text(json.dumps(graph()))
    result=subprocess.run([sys.executable,'runner/main.py','--graph',str(source),'--registry',str(registry),'--export-code'],capture_output=True,text=True,check=True)
    assert "approximate='none'" in json.loads(result.stdout)['code']

def test_raw_double_formula_and_parameterfree_gradient_identity(registry):
    model=GraphModel(graph(True),registry).double()
    with torch.no_grad():
        model.layers['first'].weight.copy_(torch.eye(2,dtype=torch.float64));model.layers['first'].bias.zero_();model.layers['last'].weight.copy_(torch.eye(2,dtype=torch.float64))
    x=torch.tensor([[-8,-3],[-1,0],[.5,1],[3,8]],dtype=torch.float64,requires_grad=True)
    for mode in (False,True):
        model.nodes['activation']['config']['approximate_tanh']=mode
        output=model(x);target=oracle(x,mode)
        torch.testing.assert_close(output,target,atol=1e-12,rtol=1e-12)
        actual=torch.autograd.grad(output.sum(),x,retain_graph=True)[0]
        expected=torch.autograd.grad(target.sum(),x)[0]
        torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
    assert len(list(model.parameters()))==3
