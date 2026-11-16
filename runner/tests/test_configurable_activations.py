"""Strict activation contracts and independent scalar value/derivative/chain rules."""
import copy
import json
import math
import pytest
import torch
from runner.configurable_activation_configuration import normalize_configuration, resolve_configuration, capability
from runner.engine import GraphModel, REGISTRY, capabilities, export_source, run, validate
from runner.graph_semantics import effective_config, semantic_graph
from runner.preview import preview
from runner.tensor_layout import BatchedLayout
from runner.tests.test_checkpoint_lifecycle import assert_nested_equal

LIMIT=3.4028234663852886e38
VARIANTS=[('silu',{}),('leaky_relu',{}),('leaky_relu',{'negative_slope':-.3}),('leaky_relu',{'negative_slope':0.}),('elu',{}),('elu',{'alpha':-.7}),('elu',{'alpha':0.})]

@pytest.fixture
def registry():
    # Use the actual authored production definitions; never inject or replace records.
    data=json.loads(REGISTRY.read_text())
    for kind,key,default in [('silu',None,None),('leaky_relu','negative_slope',.01),('elu','alpha',1.)]:
        matches=[row for row in data['components'] if row['id']==kind]
        assert len(matches)==1
        row=matches[0]
        assert row['inputs']==['in'] and row['outputs']==['out']
        assert row['category']=='Activations' and isinstance(row['title'],str) and row['title'].strip()
        if key is None:assert row['fields']==[]
        else:
            assert len(row['fields'])==1
            field=row['fields'][0]
            assert field['key']==key and field['kind']=='number'
            assert field['defaultValue']==default and field['min']==-LIMIT and field['max']==LIMIT
    return REGISTRY


def graph(kind,config,sample=None):
    specs=[('input','input',{'features':2}),('first','linear',{'in_features':2,'out_features':2,'bias':True})]
    if sample is not None:specs.append(('shape','reshape',{'shape':sample}))
    specs.append(('activation',kind,config))
    if sample is not None:specs.append(('flat','flatten',{'start_dim':1,'end_dim':-1}))
    specs += [('last','linear',{'in_features':2,'out_features':2,'bias':False}),('output','output',{})]
    return dict(schemaVersion=2,nodes=[dict(id=i,type=t,name=i,config=c) for i,t,c in specs],edges=[dict(id=str(n),source=a[0],sourcePort='out',target=b[0],targetPort='in') for n,(a,b) in enumerate(zip(specs,specs[1:]))])


def oracle(kind,config,x):
    if kind=='silu':
        z=math.exp(-abs(x));s=1/(1+z) if x>=0 else z/(1+z)
        return x*s,s+x*s*(1-s)
    if kind=='leaky_relu':
        slope=config.get('negative_slope',.01)
        return (x,1.) if x>0 else (slope*x,slope)
    alpha=config.get('alpha',1.)
    return (x,1.) if x>0 else (alpha*math.expm1(x),alpha*math.exp(x))


def identity(model):
    with torch.no_grad():
        model.layers['first'].weight.copy_(torch.eye(2,device=model.layers['first'].weight.device));model.layers['first'].bias.zero_()
        model.layers['last'].weight.copy_(torch.eye(2,device=model.layers['last'].weight.device))


@pytest.mark.parametrize('kind,config',VARIANTS)
@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('rank',range(2,9))
@pytest.mark.parametrize('batch',[1,3])
def test_independent_values_input_derivatives_all_ranks_and_generated_export(registry,kind,config,device,rank,batch):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('Actual MPS unavailable')
    sample=[1]*(rank-2)+[2];g=graph(kind,config,sample if rank>2 else None)
    model=GraphModel(g,registry).to(device);identity(model)
    raw=([0.,-2.] if batch==1 else [-20.,-.001,0.,.001,2.,20.])
    storage=torch.empty((batch,4),device=device);storage[:,::2]=torch.tensor(raw,device=device).reshape(batch,2);storage[:,1::2]=999
    x=storage[:,::2].detach().requires_grad_();assert not x.is_contiguous();before=x.detach().clone()
    y,stages=model(x,inspect=True)
    expected=torch.tensor([oracle(kind,config,v)[0] for v in raw],device=device).reshape(batch,2)
    derivative=torch.tensor([oracle(kind,config,v)[1] for v in raw],device=device).reshape(batch,2)
    torch.testing.assert_close(y,expected,atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(torch.autograd.grad(y.sum(),x)[0],derivative,atol=2e-6,rtol=2e-5)
    assert stages['activation'].shape==(batch,*sample)
    assert torch.equal(x,before) and not any('activation' in key for key in model.state_dict())
    model.eval();torch.testing.assert_close(model(x),y,atol=0,rtol=0)
    ns={};source=export_source(model);exec(source,ns);exported=ns['Model']().to(device);exported.load_state_dict(model.state_dict(),strict=True)
    assert 'inplace=False' in source
    torch.testing.assert_close(exported(x),expected,atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(torch.autograd.grad(exported(x).sum(),x)[0],derivative,atol=2e-6,rtol=2e-5)


@pytest.mark.parametrize('kind,config',VARIANTS)
@pytest.mark.parametrize('device',['cpu','mps'])
def test_saturation_near_zero_and_input_branch_immutability(registry,kind,config,device):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('Actual MPS unavailable')
    model=GraphModel(graph(kind,config),registry).to(device);identity(model)
    raw=[-1000.,-80.,-20.,-1.,-1e-6,0.,1e-6,.3,1.,20.,80.,1000.]
    x=torch.tensor(raw,device=device).reshape(6,2).requires_grad_();before=x.detach().clone()
    y=model.execute_node('activation',{'first':x},None)
    sibling=x*2
    torch.testing.assert_close(y,torch.tensor([oracle(kind,config,v)[0] for v in raw],device=device).reshape(6,2),atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(torch.autograd.grad(y.sum(),x)[0],torch.tensor([oracle(kind,config,v)[1] for v in raw],device=device).reshape(6,2),atol=2e-6,rtol=2e-5)
    assert torch.equal(x,before) and torch.equal(sibling,before*2) and y.data_ptr()!=x.data_ptr()


@pytest.mark.parametrize('kind,config',VARIANTS)
@pytest.mark.parametrize('device',['cpu','mps'])
def test_hand_affine_chain_rule_all_original_parameter_gradients(registry,kind,config,device):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('Actual MPS unavailable')
    model=GraphModel(graph(kind,config),registry).to(device)
    w=[[.7,-.3],[.2,.9]];b=[.1,-.2];v=[[1.1,-.4],[.3,.8]];rows=[[-1.,.5],[.7,2.]];coeff=[.6,-.2]
    with torch.no_grad():
        model.layers['first'].weight.copy_(torch.tensor(w,device=device));model.layers['first'].bias.copy_(torch.tensor(b,device=device));model.layers['last'].weight.copy_(torch.tensor(v,device=device))
    # Use actual float32 weights/input scalars in the independent arithmetic.
    w=model.layers['first'].weight.detach().cpu().tolist();b=model.layers['first'].bias.detach().cpu().tolist();v=model.layers['last'].weight.detach().cpu().tolist()
    x=torch.tensor(rows,device=device,requires_grad=True);rows=x.detach().cpu().tolist();upstream=torch.tensor(coeff,device=device);coeff=upstream.cpu().tolist()
    output=model(x);measured=torch.autograd.grad((output*upstream).sum(),(x,*model.parameters()))
    dx=[];dw=[[0.,0.],[0.,0.]];db=[0.,0.];dv=[[0.,0.],[0.,0.]];expected=[]
    for row in rows:
        before=[math.fsum(w[j][k]*row[k] for k in range(2))+b[j] for j in range(2)]
        activated=[oracle(kind,config,t)[0] for t in before]
        influence=[math.fsum(coeff[o]*v[o][j] for o in range(2))*oracle(kind,config,before[j])[1] for j in range(2)]
        dx.append([math.fsum(influence[j]*w[j][k] for j in range(2)) for k in range(2)])
        expected.append([math.fsum(v[o][j]*activated[j] for j in range(2)) for o in range(2)])
        for j in range(2):
            db[j]+=influence[j]
            for k in range(2):dw[j][k]+=influence[j]*row[k]
        for o in range(2):
            for j in range(2):dv[o][j]+=coeff[o]*activated[j]
    torch.testing.assert_close(output,torch.tensor(expected,device=device),atol=2e-6,rtol=2e-5)
    for actual,ref in zip(measured,[dx,dw,db,dv]):torch.testing.assert_close(actual,torch.tensor(ref,device=device),atol=2e-6,rtol=2e-5)


@pytest.mark.parametrize('kind',['silu','leaky_relu','elu'])
def test_exact_discovery_defaults_layout_and_schema(kind):
    expected=dict(version=dict(major=1,minor=0,patch=0),adapterID='kwip.runner.graph-v2.'+kind,graphSchemaVersions=[2],graphDTypes=['float32'],configuration={},outputDTypes=dict(out='float32'),maximumRank=8,preservesBatch=True,preservesShape=True,parameterless=True,deterministic=True,inplace=False)
    key={'leaky_relu':'negative_slope','elu':'alpha'}.get(kind)
    if key:expected['configuration']={key:dict(default=.01 if kind=='leaky_relu' else 1.,minimum=-LIMIT,maximum=LIMIT)}
    assert capability(kind)==expected and capabilities()['componentCapabilities'][kind]==expected
    assert normalize_configuration(kind,{})==({key:expected['configuration'][key]['default']} if key else {})
    for rank in range(2,9):
        layout=BatchedLayout(tuple([1]*(rank-2)+[2]));assert resolve_configuration(kind,{},layout)==dict(layout=layout,parameterCount=0)
    with pytest.raises(ValueError):resolve_configuration(kind,{},None)
    with pytest.raises(ValueError):effective_config(dict(type=kind,config={}),1)
    assert effective_config(dict(type=kind,config={}),2)==normalize_configuration(kind,{})


BAD=[('silu',{'inplace':False}),('silu',{'axis':1})]
for kind,key in [('leaky_relu','negative_slope'),('elu','alpha')]:
    BAD += [(kind,{key:value}) for value in [True,False,None,'1',float('nan'),float('inf'),-float('inf'),LIMIT*2]]
    BAD += [(kind,{key:1.,'inplace':False}),(kind,{key:1.,'unknown':0})]
@pytest.mark.parametrize('kind,config',BAD)
def test_invalid_configs_refused_before_allocation(registry,kind,config):
    with pytest.raises(ValueError):normalize_configuration(kind,config)
    with pytest.raises(ValueError):GraphModel(graph(kind,config),registry)


@pytest.mark.parametrize('kind',['silu','leaky_relu','elu'])
def test_integer_graph_and_parameterless_training_refusals(registry,kind):
    integer=graph(kind,{})
    integer['nodes']=[dict(id='input',type='index_input',config=dict(shape=[2])),dict(id='activation',type=kind,config={}),dict(id='output',type='output',config={})]
    integer['edges']=[dict(id='a',source='input',sourcePort='out',target='activation',targetPort='in'),dict(id='b',source='activation',sourcePort='out',target='output',targetPort='in')]
    with pytest.raises(ValueError,match='floating-point'):validate(integer,registry)
    integer['nodes'][0]=dict(id='input',type='input',config=dict(features=2))
    with pytest.raises(ValueError,match='trainable'):validate(integer,registry)


@pytest.mark.parametrize('kind',['silu','leaky_relu','elu'])
def test_checkpoint_cache_export_fine_tune_and_config_identity(registry,tmp_path,kind):
    g=graph(kind,{});options=dict(registry=registry,device='cpu',sample_limit=32,batch_size=32)
    run(g,tmp_path/'full',epochs=3,**options);run(g,tmp_path/'first',epochs=1,**options)
    run(g,tmp_path/'resume',epochs=3,resume_checkpoint=tmp_path/'first'/'checkpoint.pt',**options)
    full=torch.load(tmp_path/'full'/'checkpoint.pt',weights_only=True);resumed=torch.load(tmp_path/'resume'/'checkpoint.pt',weights_only=True)
    for key in ('modelState','optimizerState','rngState'):assert_nested_equal(full[key],resumed[key])
    assert json.loads((tmp_path/'full'/'metrics.json').read_text())==json.loads((tmp_path/'resume'/'metrics.json').read_text())
    a=preview(g,tmp_path/'preview',checkpoint=tmp_path/'full'/'checkpoint.pt',**dict(registry=registry,device='cpu'))
    reused=preview(g,tmp_path/'cached',checkpoint=tmp_path/'full'/'checkpoint.pt',resume_from='activation',cache=a['cachePath'],registry=registry,device='cpu')
    assert a['previews']==reused['previews'] and 'activation' not in full['modelState']
    run(g,tmp_path/'tune',epochs=1,finetune_checkpoint=tmp_path/'full'/'checkpoint.pt',seed=19,**options)
    tuned=torch.load(tmp_path/'tune'/'checkpoint.pt',weights_only=True);assert tuned['epoch']==1 and tuned['seed']==19
    assert all(v['step'].item()==1 for v in tuned['optimizerState']['state'].values())
    source=(tmp_path/'full'/'model.py').read_text();ns={};exec(source,ns);exported=ns['Model']();exported.load_state_dict(full['modelState'],strict=True)
    model=GraphModel(g,registry);model.load_state_dict(full['modelState'],strict=True)
    torch.testing.assert_close(exported(torch.ones(3,2)),model(torch.ones(3,2)),atol=0,rtol=0)
    changed=copy.deepcopy(g);node=next(n for n in changed['nodes'] if n['id']=='activation')
    if kind=='silu':node['type']='relu'
    else:node['config']={'negative_slope':.2} if kind=='leaky_relu' else {'alpha':2.}
    assert semantic_graph(changed)!=semantic_graph(g)
    with pytest.raises(ValueError):run(changed,tmp_path/'bad',epochs=4,resume_checkpoint=tmp_path/'full'/'checkpoint.pt',**options)
    with pytest.raises(ValueError):preview(changed,tmp_path/'bad-cache',checkpoint=tmp_path/'full'/'checkpoint.pt',resume_from='activation',cache=a['cachePath'],registry=registry,device='cpu')


@pytest.mark.parametrize('kind,key,default',[('leaky_relu','negative_slope',.01),('elu','alpha',1.)])
def test_default_semantics_signed_endpoints_and_extreme_output_refusal(registry,kind,key,default):
    from runner.product_execution import ProductExecutionAdapter
    assert semantic_graph(graph(kind,{}))==semantic_graph(graph(kind,{key:default}))
    assert normalize_configuration(kind,{key:LIMIT})=={key:LIMIT}
    assert normalize_configuration(kind,{key:-LIMIT})=={key:-LIMIT}
    assert normalize_configuration(kind,{key:0})=={key:0.}
    with pytest.raises(ValueError):normalize_configuration(kind,{key:10**10000})
    # A valid large coefficient does not guarantee finite output for every input.
    if kind=='leaky_relu':
        model=GraphModel(graph(kind,{key:LIMIT}),registry)
        output=model.execute_node('activation',{'first':torch.full((1,2),-10.)},None)
        assert torch.isneginf(output).all()
        with pytest.raises(ValueError,match='finite'):
            ProductExecutionAdapter(model).validate_stage(output,'activation',rows=1)


@pytest.mark.parametrize('kind',['silu','leaky_relu','elu'])
def test_rank_budget_refused_before_parameter_allocation(registry,kind,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Invalid graph must be refused before allocating Linear weights')
    monkeypatch.setattr(torch.nn,'Linear',forbidden)
    with pytest.raises(ValueError):GraphModel(graph(kind,{},[1]*7+[2]),registry)
    with pytest.raises(ValueError):resolve_configuration(kind,{},BatchedLayout((100_000_001,)))


@pytest.mark.parametrize('kind',['silu','leaky_relu','elu'])
def test_actual_mps_training_checkpoint_weights_and_preview(registry,tmp_path,kind):
    if not torch.backends.mps.is_available():pytest.skip('Actual MPS unavailable')
    g=graph(kind,{})
    run(g,tmp_path/'mps',registry=registry,device='mps',epochs=1,sample_limit=32,batch_size=16)
    checkpoint=torch.load(tmp_path/'mps'/'checkpoint.pt',weights_only=True)
    assert checkpoint['contract']['runtime']['device']=='mps' and checkpoint['contract']['guarantee']=='weightsOnly'
    captured=preview(g,tmp_path/'preview',registry=registry,device='mps',checkpoint=tmp_path/'mps'/'checkpoint.pt')
    assert captured['device']=='mps' and 'activation' in captured['previews']
