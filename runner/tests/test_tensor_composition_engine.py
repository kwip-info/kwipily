"""Actual composition graph/export behavior and independent coordinate influence."""
import copy
import hashlib
import itertools
import json
import math
import pytest
import torch
from runner.engine import GraphModel, REGISTRY, export_source, validate
from runner.graph_semantics import semantic_graph

CONFIGS={'concat':{'dim':1},'add':{'alpha':-.5,'sample_broadcast':True},'stack':{'dim':2},'permute':{'dims':[0,2,3,1]}}
LEFT={'concat':(2,3),'add':(2,3),'stack':(2,3),'permute':(2,3,4)}
RIGHT={'concat':(1,3),'add':(3,),'stack':(2,3)}


def output_sample(op,c,a,b=None):
    if op=='permute':return tuple(a[(v%(len(a)+1))-1] for v in c['dims'][1:])
    if op=='stack':axis=c['dim']%(len(a)+2)-1;return a[:axis]+(2,)+a[axis:]
    if op=='concat':axis=c.get('dim',-1)%(len(a)+1)-1;out=list(a);out[axis]+=b[axis];return tuple(out)
    if not c.get('sample_broadcast',False):return a
    count=max(len(a),len(b));pa=(1,)*(count-len(a))+a;pb=(1,)*(count-len(b))+b
    return tuple(max(x,y) for x,y in zip(pa,pb))


def graph(op,c=None,a=None,b=None):
    c=copy.deepcopy(CONFIGS[op] if c is None else c);a=LEFT[op] if a is None else a;b=RIGHT.get(op) if b is None else b;out=output_sample(op,c,a,b)
    rows=[('input','input',{'features':2}),('projectA','linear',{'in_features':2,'out_features':math.prod(a),'bias':True}),('a','reshape',{'shape':list(a)})]
    wires=[('input','projectA','in'),('projectA','a','in')]
    if op!='permute':rows.extend([('projectB','linear',{'in_features':2,'out_features':math.prod(b),'bias':True}),('b','reshape',{'shape':list(b)})]);wires.extend([('input','projectB','in'),('projectB','b','in'),('a','operation','a'),('b','operation','b')])
    else:wires.append(('a','operation','in'))
    rows.extend([('operation',op,c),('flat','flatten',{'start_dim':1,'end_dim':-1}),('head','linear',{'in_features':math.prod(out),'out_features':2,'bias':True}),('output','output',{})]);wires.extend([('operation','flat','in'),('flat','head','in'),('head','output','in')])
    return {'schemaVersion':2,'name':'Separate observations','nodes':[{'id':i,'type':kind,'name':i,'config':config} for i,kind,config in rows],'edges':[{'id':str(i),'source':source,'sourcePort':'out','target':target,'targetPort':port} for i,(source,target,port) in enumerate(wires)]}


def offset(coords,shape):
    result=0
    for p,n in zip(coords,shape):result=result*n+p
    return result


def oracle(op,c,a,b,va,vb,g,batch):
    out=output_sample(op,c,a,b);values=[];da=[0.]*len(va);db=[0.]*len(vb) if b else None;rank=len(out)+1
    for n,*p in itertools.product(range(batch),*(range(v) for v in out)):
        i=len(values)
        if op=='concat':
            axis=c.get('dim',-1)%(len(a)+1)-1;q=list(p)
            if p[axis]<a[axis]:j=offset((n,*q),(batch,*a));values.append(va[j]);da[j]+=g[i]
            else:q[axis]-=a[axis];j=offset((n,*q),(batch,*b));values.append(vb[j]);db[j]+=g[i]
        elif op=='stack':
            axis=c['dim']%rank-1;which=p[axis];q=p[:axis]+p[axis+1:];j=offset((n,*q),(batch,*a));values.append((va if which==0 else vb)[j]);(da if which==0 else db)[j]+=g[i]
        elif op=='permute':
            q=[0]*len(a)
            for source,coordinate in zip(c['dims'][1:],p):q[source%(len(a)+1)-1]=coordinate
            j=offset((n,*q),(batch,*a));values.append(va[j]);da[j]+=g[i]
        else:
            def source(point,shape):
                selected=point[len(point)-len(shape):];return offset((n,*[0 if size==1 else v for v,size in zip(selected,shape)]),(batch,*shape))
            j=source(p,a);k=source(p,b);alpha=c.get('alpha',1);values.append(va[j]+alpha*vb[k]);da[j]+=g[i];db[k]+=alpha*g[i]
    return (batch,*out),values,da,db


@pytest.mark.parametrize('op',CONFIGS)
@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('batch',[1,3])
def test_real_coordinate_values_weighted_both_influences_and_portable_model(op,device,batch):
    assert device!='mps' or torch.backends.mps.is_available()
    model=GraphModel(graph(op),REGISTRY).to(device);a=LEFT[op];b=RIGHT.get(op);out=output_sample(op,CONFIGS[op],a,b)
    va=[(i*7%17-9)/4 for i in range(batch*math.prod(a))];vb=[(i*11%19-10)/4 for i in range(batch*math.prod(b))] if b else []
    # Noncontiguous source views preserve values and must not be mutated.
    def tensor(values,shape):
        storage=torch.zeros((*shape[:-1],shape[-1]*2),device=device);storage[...,::2]=torch.tensor(values,device=device).reshape(shape);return storage[...,::2].detach().requires_grad_()
    x=tensor(va,(batch,*a));y=tensor(vb,(batch,*b)) if b else None;sources={'a':x}|({'b':y} if b else {})
    actual=model.execute_node('operation',sources,None);g=[(i%7-3)/4 for i in range(actual.numel())];shape,values,da,db=oracle(op,CONFIGS[op],a,b,va,vb,g,batch)
    assert tuple(actual.shape)==shape and actual.device.type==device
    torch.testing.assert_close(actual.cpu(),torch.tensor(values).reshape(shape),atol=2e-6,rtol=2e-6)
    gradients=torch.autograd.grad(actual,(x,y) if b else (x,),torch.tensor(g,device=device).reshape(shape))
    torch.testing.assert_close(gradients[0].cpu(),torch.tensor(da).reshape(x.shape),atol=2e-6,rtol=2e-6)
    if b:torch.testing.assert_close(gradients[1].cpu(),torch.tensor(db).reshape(y.shape),atol=2e-6,rtol=2e-6)
    assert x.detach().cpu().flatten().tolist()==va
    if b:assert y.detach().cpu().flatten().tolist()==vb
    source=export_source(model);assert 'runner.' not in source;scope={};exec(source,scope);portable=scope['Model']().to(device);portable.load_state_dict(model.state_dict(),strict=True)
    z=torch.tensor([[1.,-2.],[-3.,.5]],device=device,requires_grad=True);w=z.detach().clone().requires_grad_()
    torch.testing.assert_close(model(z),portable(w),atol=0,rtol=0)
    torch.testing.assert_close(torch.autograd.grad(model(z).sum(),z)[0],torch.autograd.grad(portable(w).sum(),w)[0],atol=0,rtol=0)
    assert set(model.state_dict())==set(portable.state_dict()) and not any('operation' in key for key in model.state_dict())


@pytest.mark.parametrize('op,problem',[(op,problem) for op in CONFIGS for problem in ['sparse','dtype','empty','shape','nonfinite','mixed-device','batch-mismatch'] if op!='permute' or problem not in ('mixed-device','batch-mismatch')])
def test_actual_boundary_refusals(op,problem):
    model=GraphModel(graph(op),REGISTRY);x=torch.ones(1,*LEFT[op]);y=torch.ones(1,*RIGHT[op]) if op!='permute' else None
    if problem=='sparse':x=x.to_sparse()
    elif problem=='dtype':x=x.double()
    elif problem=='empty':x=x[:0]
    elif problem=='shape':x=x[...,0]
    elif problem=='nonfinite':x.flatten()[0]=float('nan')
    elif problem=='mixed-device':y=y.to('mps')
    else:y=y.expand(2,*RIGHT[op])
    with pytest.raises(ValueError):model.execute_node('operation',{'a':x}|({'b':y} if y is not None else {}),None)


@pytest.mark.parametrize('op',CONFIGS)
def test_actual_output_budget_refuses_before_function_allocation(op,monkeypatch):
    model=GraphModel(graph(op),REGISTRY);batch=100_000_001;x=torch.zeros(1,*LEFT[op]).expand(batch,*LEFT[op]);y=torch.zeros(1,*RIGHT[op]).expand(batch,*RIGHT[op]) if op!='permute' else None
    called=[]
    if op!='permute':monkeypatch.setattr(torch,{'concat':'cat','add':'add','stack':'stack'}[op],lambda *a,**k:called.append(True))
    with pytest.raises(ValueError,match='budget'):model.execute_node('operation',{'a':x}|({'b':y} if y is not None else {}),None)
    assert not called


def test_historical_concat_add_defaults_preserve_semantic_and_state_identity():
    for op,a,b in [('concat',(3,),(3,)),('add',(3,),(3,))]:
        old=graph(op,{},a,b);explicit=copy.deepcopy(old);next(n for n in explicit['nodes'] if n['id']=='operation')['config']={'dim':-1} if op=='concat' else {'alpha':1,'sample_broadcast':False}
        assert semantic_graph(old)==semantic_graph(explicit)
        model=GraphModel(old,REGISTRY);other=GraphModel(explicit,REGISTRY);other.load_state_dict(model.state_dict(),strict=True);x=torch.ones(2,2)
        torch.testing.assert_close(model(x),other(x),rtol=0,atol=0)


VARIANTS=[
 ('concat',{'dim':2},(2,3),(2,1)),('concat',{'dim':-2},(2,3),(1,3)),
 ('add',{'alpha':0,'sample_broadcast':True},(2,1,3),(4,1)),
 ('add',{'alpha':2.25,'sample_broadcast':True},(3,),(2,1,3)),
 ('add',{'alpha':1,'sample_broadcast':False},(2,3),(2,3)),
 ('stack',{'dim':1},(2,3),(2,3)),('stack',{'dim':-1},(2,3),(2,3)),
 ('stack',{'dim':7},(1,1,1,1,1,2),(1,1,1,1,1,2)),
 ('permute',{'dims':[0,1]},(3,),None),
 ('permute',{'dims':[-4,-2,-1,-3]},(2,3,4),None),
 ('permute',{'dims':[0,7,6,5,4,3,2,1]},(1,1,1,1,1,1,2),None),
]

@pytest.mark.parametrize('op,c,a,b',VARIANTS)
@pytest.mark.parametrize('device',['cpu','mps'])
def test_other_axes_signed_zero_broadcast_negative_aliases_and_max_rank(op,c,a,b,device):
    model=GraphModel(graph(op,c,a,b),REGISTRY).to(device);batch=2
    va=[(i%13-6)/4 for i in range(batch*math.prod(a))];vb=[(i%11-5)/4 for i in range(batch*math.prod(b))] if b else []
    x=torch.tensor(va,device=device).reshape(batch,*a).requires_grad_();y=torch.tensor(vb,device=device).reshape(batch,*b).requires_grad_() if b else None
    actual=model.execute_node('operation',{'a':x}|({'b':y} if b else {}),None);g=[(i%5-2)/3 for i in range(actual.numel())];shape,v,da,db=oracle(op,c,a,b,va,vb,g,batch)
    torch.testing.assert_close(actual.cpu(),torch.tensor(v).reshape(shape),rtol=2e-6,atol=2e-6)
    gradients=torch.autograd.grad(actual,(x,y) if b else (x,),torch.tensor(g,device=device).reshape(shape))
    torch.testing.assert_close(gradients[0].cpu(),torch.tensor(da).reshape(x.shape),rtol=2e-6,atol=2e-6)
    if b:torch.testing.assert_close(gradients[1].cpu(),torch.tensor(db).reshape(y.shape),rtol=2e-6,atol=2e-6)
    scope={};exec(export_source(model),scope);portable=scope['Model']().to(device);portable.load_state_dict(model.state_dict(),strict=True)
    values=torch.ones(2,2,device=device);torch.testing.assert_close(model(values),portable(values),rtol=0,atol=0)


def integer_graph(op):
    a=(2,3,4) if op=='permute' else (2,3);c={'dims':[0,2,3,1]} if op=='permute' else {'dim':-1} if op in ('concat','stack') else {};out=output_sample(op,c,a,a)
    nodes=[{'id':'input','type':'index_input','config':{'shape':list(a)}},{'id':'operation','type':op,'config':c},{'id':'flat','type':'flatten','config':{'start_dim':1,'end_dim':-1}},{'id':'lookup','type':'embedding','config':{'num_embeddings':128,'embedding_dim':2,'padding_idx':None,'max_norm':None,'norm_type':2,'scale_grad_by_freq':False,'sparse':False,'freeze':False}},{'id':'flatFloat','type':'flatten','config':{'start_dim':1,'end_dim':-1}},{'id':'head','type':'linear','config':{'in_features':math.prod(out)*2,'out_features':2,'bias':True}},{'id':'output','type':'output','config':{}}]
    wires=[('input','operation','in')] if op=='permute' else [('input','operation','a'),('input','operation','b')]
    wires += [('operation','flat','in'),('flat','lookup','in'),('lookup','flatFloat','in'),('flatFloat','head','in'),('head','output','in')]
    return {'schemaVersion':2,'name':'Typed positions preserved','nodes':nodes,'edges':[{'id':str(i),'source':s,'sourcePort':'out','target':t,'targetPort':p} for i,(s,t,p) in enumerate(wires)]},a,c


@pytest.mark.parametrize('op',CONFIGS)
@pytest.mark.parametrize('device',['cpu','mps'])
def test_full_typed_integer_graph_values_portable_and_no_integer_gradient_claim(op,device):
    g,a,c=integer_graph(op);model=GraphModel(g,REGISTRY,bundled_dataset=False).to(device);x=(torch.arange(2*math.prod(a),device=device)%7).reshape(2,*a).long()
    actual=model.execute_node('operation',{'input':x},None);outshape,v,_,_=oracle(op,c,a,None if op=='permute' else a,x.cpu().flatten().tolist(),[] if op=='permute' else x.cpu().flatten().tolist(),[0]*(2*math.prod(output_sample(op,c,a,a))),2)
    assert actual.dtype==torch.int64 and actual.device.type==device and not actual.requires_grad
    assert actual.cpu().flatten().tolist()==v
    assert model.dtypes['operation']=='int64'
    scope={};exec(export_source(model),scope);portable=scope['Model']().to(device);portable.load_state_dict(model.state_dict(),strict=True)
    torch.testing.assert_close(model(x),portable(x),rtol=0,atol=0)
    if op=='add':
        for config in [{'alpha':2,'sample_broadcast':False},{'alpha':1,'sample_broadcast':True}]:
            bad=copy.deepcopy(g);next(n for n in bad['nodes'] if n['id']=='operation')['config']=config
            with pytest.raises(ValueError,match='Integer'):GraphModel(bad,REGISTRY,bundled_dataset=False)


def test_zero_alpha_still_validates_second_source_and_nonfinite_result_refuses():
    c={'alpha':0,'sample_broadcast':True};model=GraphModel(graph('add',c),REGISTRY);x=torch.ones(1,2,3);y=torch.ones(1,3);y[0,0]=float('nan')
    with pytest.raises(ValueError,match='finite'):model.execute_node('operation',{'a':x,'b':y},None)
    model=GraphModel(graph('add',{'alpha':3.4028234663852886e38,'sample_broadcast':True}),REGISTRY)
    with pytest.raises(ValueError,match='nonfinite'):model.execute_node('operation',{'a':x,'b':torch.full((1,3),2.)},None)


def test_legacy_semantic_digests_captured_from_prechange_head_are_unchanged():
    expected={'concat':'06fd861ab19628b13f29088354e58b3d95d8fe0bee9a4c1cdc5052be2c4a1376','add':'e249cc929059b1ed11eb24f86c049b40f2a66f57c828d0a26702bc088437f02f'}
    for op,digest in expected.items():
        old=graph(op,{},(3,),(3,));assert hashlib.sha256(json.dumps(semantic_graph(old),sort_keys=True,separators=(',',':')).encode()).hexdigest()==digest
        changed=copy.deepcopy(old);next(n for n in changed['nodes'] if n['id']=='operation')['config']={'dim':1} if op=='concat' else {'alpha':2,'sample_broadcast':False}
        assert semantic_graph(old)!=semantic_graph(changed)


@pytest.mark.parametrize('op',CONFIGS)
@pytest.mark.parametrize('device',['cpu','mps'])
def test_actual_training_saved_checkpoint_captures_and_inference_export(op,device,tmp_path):
    from runner.engine import run
    directory=tmp_path/'trained';run(graph(op),directory,epochs=1,sample_limit=32,batch_size=8,device=device,registry=REGISTRY,emit=lambda event:None,sample_capture={'schemaVersion':1,'activationValues':True,'predictionRows':True})
    checkpoint=torch.load(directory/'checkpoint.pt',weights_only=True,map_location='cpu');manifest=json.loads((directory/'manifest.json').read_text());preview=json.loads((directory/'activations.json').read_text())
    assert checkpoint['epoch']==1 and manifest['runtime']['device']==device and preview['previews']['operation']['captureStatus']=='captured'
    model=GraphModel(graph(op),REGISTRY);model.load_state_dict(checkpoint['modelState'],strict=True)
    scope={};exec((directory/'model.py').read_text(),scope);portable=scope['Model']();portable.load_state_dict(checkpoint['modelState'],strict=True)
    x=torch.ones(3,2);torch.testing.assert_close(model(x),portable(x),rtol=0,atol=0)


@pytest.mark.parametrize('op',['concat','add'])
def test_legacy_default_exact_resume_and_new_nondefault_refusal(op,tmp_path):
    from runner.engine import run
    from runner.tests.test_checkpoint_lifecycle import assert_nested_equal
    old=graph(op,{},(3,),(3,));explicit=copy.deepcopy(old);next(n for n in explicit['nodes'] if n['id']=='operation')['config']={'dim':-1} if op=='concat' else {'alpha':1,'sample_broadcast':False}
    full=tmp_path/'full';first=tmp_path/'first';resumed=tmp_path/'resumed'
    run(old,full,epochs=2,sample_limit=32,batch_size=8,device='cpu',registry=REGISTRY,emit=lambda event:None)
    run(old,first,epochs=1,sample_limit=32,batch_size=8,device='cpu',registry=REGISTRY,emit=lambda event:None)
    run(explicit,resumed,epochs=2,sample_limit=32,batch_size=8,device='cpu',registry=REGISTRY,resume_checkpoint=first/'checkpoint.pt',emit=lambda event:None)
    a=torch.load(full/'checkpoint.pt',weights_only=True);b=torch.load(resumed/'checkpoint.pt',weights_only=True)
    for key in ['modelState','optimizerState','rngState']:assert_nested_equal(a[key],b[key])
    changed=copy.deepcopy(explicit);next(n for n in changed['nodes'] if n['id']=='operation')['config']={'dim':1} if op=='concat' else {'alpha':2,'sample_broadcast':False}
    with pytest.raises(ValueError):run(changed,tmp_path/'bad',epochs=3,sample_limit=32,batch_size=8,device='cpu',registry=REGISTRY,resume_checkpoint=first/'checkpoint.pt',emit=lambda event:None)


@pytest.mark.parametrize('op',['concat','add'])
def test_actual_preview_cache_survives_legacy_default_normalization_but_not_nondefault(op,tmp_path):
    from runner.preview import preview
    old=graph(op,{},(3,),(3,));explicit=copy.deepcopy(old);next(n for n in explicit['nodes'] if n['id']=='operation')['config']={'dim':-1} if op=='concat' else {'alpha':1,'sample_broadcast':False}
    first=preview(old,tmp_path/'first',device='cpu',registry=REGISTRY)
    continued=preview(explicit,tmp_path/'continued',device='cpu',registry=REGISTRY,resume_from='operation',cache=first['cachePath'])
    assert continued['previews']['output']==first['previews']['output'] and 'operation' not in continued['reusedNodeIDs']
    assert 'a' in continued['reusedNodeIDs'] and 'b' in continued['reusedNodeIDs']
    changed=copy.deepcopy(old);next(n for n in changed['nodes'] if n['id']=='operation')['config']={'dim':1} if op=='concat' else {'alpha':2,'sample_broadcast':False}
    with pytest.raises(ValueError,match='stale'):preview(changed,tmp_path/'changed',device='cpu',registry=REGISTRY,resume_from='operation',cache=first['cachePath'])


@pytest.mark.parametrize('op',CONFIGS)
def test_legacy_schema_boundaries_and_unknown_settings_refuse(op):
    g=graph(op,CONFIGS[op],(3,),(3,)) if op in ('concat','add') else graph(op)
    if op in ('concat','add'):
        g['nodes']=[node for node in g['nodes'] if node['id'] not in ('a','b','flat')]
        g['edges']=[edge for edge in g['edges'] if edge['target'] not in ('a','b') and edge['source']!='flat']
        for edge in g['edges']:
            if edge['source'] in ('a','b'):edge['source']='projectA' if edge['source']=='a' else 'projectB'
            if edge['target']=='flat':edge['target']='head'
    g['schemaVersion']=1
    for node in g['nodes']:
        if node['type']=='linear':node['config'].pop('bias')
    if op in ('concat','add'):assert GraphModel(g,REGISTRY).nodes['operation']['config']==CONFIGS[op]
    else:
        with pytest.raises(ValueError,match='schema'):GraphModel(g,REGISTRY)
    g=graph(op);next(n for n in g['nodes'] if n['id']=='operation')['config']['custom_code']='do anything'
    with pytest.raises(ValueError):GraphModel(g,REGISTRY)


def test_integer_add_preserves_historical_fixed_width_arithmetic_without_float_conversion():
    from runner.engine import _execute_tensor_composition
    left=torch.tensor([[2**63-1,-2**63]],dtype=torch.int64);right=torch.tensor([[1,-1]],dtype=torch.int64)
    actual=_execute_tensor_composition(left,right,'add',{'alpha':1,'sample_broadcast':False},(2,),(2,),(2,))
    assert actual.tolist()==[[-2**63,2**63-1]]
    assert torch.equal(actual,left+right) and actual.dtype==torch.int64


@pytest.mark.parametrize('op',['concat','add'])
@pytest.mark.parametrize('device',['cpu','mps'])
def test_actual_named_pool_indices_keep_integer_type_through_legacy_composition_and_cache(op,device,tmp_path):
    from runner.preview import preview
    nodes=[('input','input',{'features':2}),('project','linear',{'in_features':2,'out_features':40,'bias':True}),('shape','reshape',{'shape':[2,5,4]}),('pool','adaptive_max_pool2d',{'output_height':3,'output_width':2,'return_indices':True}),('operation',op,{}),('flat','flatten',{'start_dim':1,'end_dim':-1}),('lookup','embedding',{'num_embeddings':40,'embedding_dim':2,'padding_idx':None,'max_norm':None,'norm_type':2,'scale_grad_by_freq':False,'sparse':False,'freeze':False}),('flatFloat','flatten',{'start_dim':1,'end_dim':-1}),('head','linear',{'in_features':48 if op=='concat' else 24,'out_features':2,'bias':True}),('output','output',{})]
    wires=[('input','out','project','in'),('project','out','shape','in'),('shape','out','pool','in'),('pool','indices','operation','a'),('pool','indices','operation','b'),('operation','out','flat','in'),('flat','out','lookup','in'),('lookup','out','flatFloat','in'),('flatFloat','out','head','in'),('head','out','output','in')]
    g={'schemaVersion':2,'name':'Winning positions preserved','nodes':[{'id':i,'type':kind,'config':c} for i,kind,c in nodes],'edges':[{'id':str(i),'source':s,'sourcePort':sp,'target':t,'targetPort':tp} for i,(s,sp,t,tp) in enumerate(wires)]}
    model=GraphModel(g,REGISTRY).to(device);assert model.dtypes['pool']['indices']==model.dtypes['operation']=='int64'
    x=torch.tensor([[1.,-2.],[-3.,.5]],device=device);logits,stages=model(x,inspect=True)
    indices=stages['pool']['indices'];expected=torch.cat((indices,indices),dim=-1) if op=='concat' else indices+indices
    assert stages['operation'].dtype==torch.int64;torch.testing.assert_close(stages['operation'],expected,rtol=0,atol=0)
    scope={};exec(export_source(model),scope);portable=scope['Model']().to(device);portable.load_state_dict(model.state_dict(),strict=True);torch.testing.assert_close(logits,portable(x),rtol=0,atol=0)
    first=preview(g,tmp_path/'first',device=device,registry=REGISTRY)
    second=preview(g,tmp_path/'second',device=device,registry=REGISTRY,resume_from='operation',cache=first['cachePath'])
    assert first['previews']['output']==second['previews']['output']
    cache=torch.load(first['cachePath'],weights_only=True,map_location='cpu');assert cache['activations']['pool']['indices'].dtype==torch.int64


@pytest.mark.parametrize('op',CONFIGS)
def test_discovered_authored_generated_descriptors_are_exact_and_legacy_layout_unchanged(op,monkeypatch):
    import importlib
    from pathlib import Path
    from runner.engine import capabilities
    from runner.tensor_composition_configuration import capability
    root=Path(__file__).resolve().parents[2]
    bundle=json.loads((root/'components/tensor-composition/bundle.json').read_text())
    authored=next(row['runtimeContract'] for row in bundle['components'] if row['definition']['id']==op)
    monkeypatch.syspath_prepend(str(root/'runner'))
    generated=importlib.import_module('runner.tensor_composition_adapters_generated')
    assert authored==capability(op)==generated.CAPABILITIES[op]()==capabilities()['componentCapabilities'][op]
    assert capabilities()['componentCapabilities']['tensorLayout']=={'version':{'major':1,'minor':0,'patch':0},'adapterID':'kwip.runner.graph-v2.tensor-layout','graphSchemaVersions':[2],'operations':['reshape','transpose','flatten'],'maximumRank':8,'preservesBatch':True}


@pytest.mark.parametrize('op',['concat','stack','add'])
def test_combined_output_budget_refuses_with_each_source_individually_in_budget(op,monkeypatch):
    from runner.engine import _execute_tensor_composition
    if op=='add':a,b,out,c,batch=(2,1,3),(4,1),(2,4,3),{'alpha':1,'sample_broadcast':True},5_000_001
    elif op=='stack':a,b,out,c,batch=(2,3),(2,3),(2,3,2),{'dim':-1},10_000_001
    else:a,b,out,c,batch=(2,3),(2,3),(2,6),{'dim':-1},10_000_001
    left=torch.zeros(1,*a).expand(batch,*a);right=torch.zeros(1,*b).expand(batch,*b)
    assert left.numel()<100_000_000 and right.numel()<100_000_000
    calls=[]
    monkeypatch.setattr(torch,'isfinite',lambda *args,**kwargs:calls.append('finite'))
    monkeypatch.setattr(torch,{'concat':'cat','stack':'stack','add':'add'}[op],lambda *args,**kwargs:calls.append('allocate'))
    with pytest.raises(ValueError,match='budget'):_execute_tensor_composition(left,right,op,c,a,b,out)
    assert calls==[]
