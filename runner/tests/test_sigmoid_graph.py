"""Independent Sigmoid values/derivatives plus actual graph/export/lifecycle contracts."""
from runner.tests.cli_discovery import capabilities_cli
import copy
import json
import math
import subprocess
import sys

import pytest
import torch

from runner.engine import GraphModel, REGISTRY, capabilities, export_source, run, validate
from runner.graph_semantics import effective_config, semantic_graph
from runner.preview import preview
from runner.tests.test_checkpoint_lifecycle import state, assert_nested_equal

@pytest.fixture
def registry(tmp_path):
    data = json.loads(REGISTRY.read_text())
    if not any(c['id']=='sigmoid' for c in data['components']):
        data['components'].append({'id':'sigmoid','title':'Sigmoid','category':'Activations','inputs':['in'],'outputs':['out'],'fields':[]})
    path=tmp_path/'registry.json'; path.write_text(json.dumps(data)); return path

def graph(sample=None):
    specs=[('input','input',{'features':2}),('first','linear',{'in_features':2,'out_features':2,'bias':True})]
    if sample is not None: specs.append(('shape','reshape',{'shape':sample}))
    specs.append(('activation','sigmoid',{}))
    if sample is not None: specs.append(('flat','flatten',{'start_dim':1,'end_dim':-1}))
    specs += [('last','linear',{'in_features':2,'out_features':2,'bias':False}),('output','output',{})]
    return {'schemaVersion':2,'name':'Sigmoid graph','nodes':[{'id':i,'type':t,'name':i,'x':0,'y':0,'config':c} for i,t,c in specs],
            'edges':[{'id':str(n),'source':specs[n][0],'sourcePort':'out','target':specs[n+1][0],'targetPort':'in'} for n in range(len(specs)-1)]}

def oracle(value):
    # Scalar stdlib math, independent of worker tensor implementation.
    if value >= 0: return 1/(1+math.exp(-value))
    e=math.exp(value); return e/(1+e)

def identity(model):
    with torch.no_grad():
        model.layers['first'].weight.copy_(torch.eye(2,device=model.layers['first'].weight.device))
        model.layers['first'].bias.zero_()
        model.layers['last'].weight.copy_(torch.eye(2,device=model.layers['last'].weight.device))

@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('sample',[None,[1,2],[1,1,1,1,1,1,2]])
def test_values_derivatives_shape_export_and_no_state(device,sample,registry):
    if device=='mps' and not torch.backends.mps.is_available(): pytest.skip('Actual MPS unavailable')
    model=GraphModel(graph(sample),registry).to(device); identity(model)
    raw=[-1000,-20,-8,-2,-1,0,1,2,8,20,1000,.2]
    storage=torch.empty((6,4),device=device);storage[:,::2]=torch.tensor(raw,device=device).reshape(6,2);storage[:,1::2]=999
    x=storage[:,::2].detach().requires_grad_(); assert not x.is_contiguous()
    expected=torch.tensor([oracle(v) for v in raw],device=device).reshape(6,2)
    events=[]; model.train(); output,values=model(x,inspect=True,operation_observer=lambda phase,node:events.append((phase,node)))
    torch.testing.assert_close(output,expected,atol=2e-6,rtol=2e-5)
    derivative=torch.tensor([oracle(v)*(1-oracle(v)) for v in raw],device=device).reshape(6,2)
    torch.testing.assert_close(torch.autograd.grad(output.sum(),x)[0],derivative,atol=2e-6,rtol=2e-5)
    assert values['activation'].shape == (6,*(sample or [2]))
    assert values['activation'].dtype == torch.float32
    torch.testing.assert_close(model(x[:1]),expected[:1],atol=2e-6,rtol=2e-5)
    assert not torch.allclose(output.sum(dim=-1),torch.ones(6,device=device))  # independent element transform, not Softmax
    assert torch.isfinite(output).all() and torch.all((output>=0)&(output<=1))
    assert ('started','activation') in events and ('completed','activation') in events
    assert events == [(phase,node) for node in model.order for phase in ('started','completed')]
    assert output.flatten()[5].item()==.5
    assert len(list(model.parameters()))==3 and not any('activation' in name for name in model.state_dict())
    model.eval(); torch.testing.assert_close(model(x),output,atol=0,rtol=0)
    source=export_source(model); namespace={};exec(source,namespace)
    exported=namespace['Model']().to(device);exported.load_state_dict(model.state_dict(),strict=True)
    assert 'torch.sigmoid(' in source and set(exported.state_dict())==set(model.state_dict())
    torch.testing.assert_close(exported(x),output,atol=0,rtol=0)
    torch.testing.assert_close(torch.autograd.grad(exported(x).sum(),x)[0],derivative,atol=2e-6,rtol=2e-5)

@pytest.mark.parametrize('device',['cpu','mps'])
def test_hand_computed_chain_rule_input_and_linear_parameter_gradients(device,registry):
    if device=='mps' and not torch.backends.mps.is_available(): pytest.skip('Actual MPS unavailable')
    model=GraphModel(graph(),registry).to(device)
    w=[[.7,-.3],[.2,.9]];b=[.1,-.2];v=[[1.1,-.4],[.3,.8]];rows=[[-1,.5],[.7,2.0]];coeff=[.6,-.2]
    with torch.no_grad():
        model.layers['first'].weight.copy_(torch.tensor(w,device=device));model.layers['first'].bias.copy_(torch.tensor(b,device=device));model.layers['last'].weight.copy_(torch.tensor(v,device=device))
    x=torch.tensor(rows,device=device,requires_grad=True);out=model(x)
    objective=(out*torch.tensor(coeff,device=device)).sum();objective.backward()
    gx=[];gw=[[0.,0.],[0.,0.]];gb=[0.,0.];gv=[[0.,0.],[0.,0.]];expected=[]
    for row in rows:
        s=[oracle(sum(w[j][k]*row[k] for k in range(2))+b[j]) for j in range(2)]
        expected.append([sum(v[q][j]*s[j] for j in range(2)) for q in range(2)])
        dz=[sum(coeff[q]*v[q][j] for q in range(2))*s[j]*(1-s[j]) for j in range(2)]
        gx.append([sum(dz[j]*w[j][k] for j in range(2)) for k in range(2)])
        for j in range(2):
            gb[j]+=dz[j]
            for k in range(2): gw[j][k]+=dz[j]*row[k]
        for q in range(2):
            for j in range(2): gv[q][j]+=coeff[q]*s[j]
    for actual,ref in [(out,expected),(x.grad,gx),(model.layers['first'].weight.grad,gw),(model.layers['first'].bias.grad,gb),(model.layers['last'].weight.grad,gv)]:
        torch.testing.assert_close(actual,torch.tensor(ref,device=device),atol=2e-6,rtol=2e-5)

@pytest.mark.parametrize('config',[None,[],{'axis':-1},{'temperature':1},{'inplace':True}])
def test_strict_empty_configuration(config,registry):
    g=graph();g['nodes'][2]['config']=config
    with pytest.raises(ValueError): validate(g,registry)
    with pytest.raises(ValueError): semantic_graph(g)

def test_schema_integer_and_registry_boundaries(registry,tmp_path):
    g=graph();g['schemaVersion']=1
    for node in g['nodes']:
        if node['type']=='linear': node['config'].pop('bias')
    with pytest.raises(ValueError,match='schema'):validate(g,registry)
    with pytest.raises(ValueError,match='schema'):semantic_graph(g)
    integer=graph();integer['nodes'][0].update(type='index_input',config={'shape':[2]})
    integer['nodes']= [integer['nodes'][0],integer['nodes'][2],integer['nodes'][3],integer['nodes'][4]]
    integer['edges'][0].update(target='activation');integer['edges']= [integer['edges'][0],integer['edges'][2],integer['edges'][3]]
    with pytest.raises(ValueError,match='floating'):validate(integer,registry)
    data=json.loads(registry.read_text());data['components']=[c for c in data['components'] if c['id']!='sigmoid']
    absent=tmp_path/'absent.json';absent.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='Unsupported component'):validate(graph(),absent)
    assert effective_config({'id':'s','type':'sigmoid','config':{}},2)=={}
    parameterless=graph();parameterless['nodes']=[parameterless['nodes'][0],parameterless['nodes'][2],parameterless['nodes'][4]]
    parameterless['edges']=[{'id':'a','source':'input','sourcePort':'out','target':'activation','targetPort':'in'},
                            {'id':'b','source':'activation','sourcePort':'out','target':'output','targetPort':'in'}]
    with pytest.raises(ValueError,match='trainable parameters'):validate(parameterless,registry)
    from runner.composite_source import validate_bodies
    body={'configurationVersion':2,'nodes':[graph()['nodes'][2]],'instances':[]}
    document={'root':body,'definitions':[]};validate_bodies(document)
    body['configurationVersion']=1
    with pytest.raises(ValueError,match='schema'):validate_bodies(document)


def test_discovery_allocation_free_resolver_and_actual_cli_export(registry,tmp_path):
    from runner.sigmoid_configuration import capability, normalize_sigmoid, resolve_sigmoid
    from runner.tensor_layout import BatchedLayout
    expected={'version':{'major':1,'minor':0,'patch':0},'adapterID':'kwip.runner.graph-v2.sigmoid','graphSchemaVersions':[2],
              'maximumRank':8,'preservesBatch':True,'preservesShape':True,'graphDTypes':['float32'],'configuration':{},'outputDTypes':{'out':'float32'},'parameterless':True,'deterministic':True}
    assert capability()==expected and capabilities()['componentCapabilities']['sigmoid']==expected
    assert normalize_sigmoid({})=={}
    layout=BatchedLayout((1,2));resolved=resolve_sigmoid({},layout)
    assert resolved['layout']==layout and resolved['parameterCount']==0
    with pytest.raises(ValueError):resolve_sigmoid({},[1,2])
    discovery=capabilities_cli()
    assert json.loads(discovery.stdout)['componentCapabilities']['sigmoid']==expected
    path=tmp_path/'graph.json';path.write_text(json.dumps(graph()))
    result=subprocess.run([sys.executable,'runner/main.py','--graph',str(path),'--registry',str(registry),'--export-code'],capture_output=True,text=True,check=True)
    assert 'torch.sigmoid(' in json.loads(result.stdout)['code']


def test_training_resume_capture_and_preview_cache_lifecycle(registry,tmp_path):
    g=graph();full=tmp_path/'full';first=tmp_path/'first';resumed=tmp_path/'resumed'
    run(g,full,epochs=4,device='cpu',registry=registry,gradient_capture={'schemaVersion':1,'everyEpochs':1,'maxBatchesPerEpoch':1,'maxCaptures':4})
    run(g,first,epochs=2,device='cpu',registry=registry)
    run(g,resumed,epochs=4,device='cpu',registry=registry,resume_checkpoint=first/'checkpoint.pt')
    for key in ('modelState','optimizerState','rngState'):assert_nested_equal(state(full)[key],state(resumed)[key])
    gradients=json.loads((full/'gradient-summary.json').read_text())
    assert all(p['nodeID']!='activation' for p in gradients['inventory']['parameters'])
    captured=json.loads((full/'activations.json').read_text())['previews']['activation']
    assert captured['dtype']=='float32' and captured['shape'][-1]==2 and all(0<=value<=1 for value in captured['values'])
    omitted=tmp_path/'omitted';run(g,omitted,epochs=1,device='cpu',registry=registry,sample_capture={'schemaVersion':1,'activationValues':False,'predictionRows':False})
    assert json.loads((omitted/'activations.json').read_text())['previews']['activation']['values']==[]
    a=preview(g,tmp_path/'a',device='cpu',registry=registry,checkpoint=full/'checkpoint.pt')
    b=preview(g,tmp_path/'b',device='cpu',registry=registry,checkpoint=full/'checkpoint.pt',resume_from='activation',cache=a['cachePath'])
    assert a['previews']==b['previews']
    changed=copy.deepcopy(g);changed['nodes'][2]['type']='tanh'
    with pytest.raises(ValueError,match='stale|match'):preview(changed,tmp_path/'bad-cache',device='cpu',registry=registry,checkpoint=full/'checkpoint.pt',resume_from='activation',cache=a['cachePath'])
    with pytest.raises(ValueError,match='semantic architecture'):run(changed,tmp_path/'bad-resume',epochs=5,device='cpu',registry=registry,resume_checkpoint=full/'checkpoint.pt')
