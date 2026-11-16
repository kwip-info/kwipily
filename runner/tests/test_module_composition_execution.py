"""Real native containers and independent dense affine/reference influence."""
import copy
import hashlib
import io
import json
import pytest
import torch
from runner.engine import GraphModel, export_source
from runner.graph_semantics import semantic_graph
from runner.module_composition_plan import validate_plan
from runner.native_ownership import native_owners

DEVICES=['cpu']+(['mps'] if torch.backends.mps.is_available() else [])

def cid(path):
    return 'root' if not path else 'c-'+hashlib.sha256(b''.join(str(len(p.encode())).encode()+b':'+p.encode() for p in path)).hexdigest()
def endpoint(node,port='out'):return dict(node=node,port=port)
def container(path,kind,children,first,last):
    return dict(id=cid(path),kind=kind,path=path,children=[dict(kind=t,id=i,key=(k if k is not None else str(n))) for n,(t,i,k) in enumerate(children)],inputs=[dict(id='input' if not path else 'values',targets=[endpoint(first,'in')])],outputs=[dict(id='output' if not path else 'result',source=endpoint(last))])
def fixture(kind='sequential',branch=False,nested=False,functional=False):
    ids=['a','b'];nodes=[dict(id='input',type='input',config={'features':2})]
    wires=[]
    if nested:
        ids=['la','lb','ra','rb'];pairs=[('input','la','in'),('la','lb','in'),('input','ra','in'),('ra','rb','in'),('lb','sum','a'),('rb','sum','b'),('sum','output','in')]
    elif branch:pairs=[('input','a','in'),('input','b','in'),('a','sum','a'),('b','sum','b'),('sum','output','in')]
    else:pairs=[('input','a','in'),('a','b','in'),('b','output','in')]
    for ident in ids:nodes.append(dict(id=ident,type='linear',config={'in_features':2,'out_features':2,'bias':True}))
    if functional and not nested and not branch:nodes[2]=dict(id='b',type='relu',config={})
    if branch or nested:nodes.append(dict(id='sum',type='add',config={}))
    nodes.append(dict(id='output',type='output',config={}))
    for n,(a,b,p) in enumerate(pairs):wires.append(dict(id=str(n),source=a,sourcePort='out',target=b,targetPort=p))
    if nested:
        root=container([],'graph',[('container',cid(['left']),None),('container',cid(['right']),None),('node','sum',None)],'la','sum')
        root['inputs'][0]['targets'].append(endpoint('ra','in'))
        containers=[root,container(['left'],'sequential',[('node','la',None),('node','lb',None)],'la','lb'),container(['right'],'sequential',[('node','ra',None),('node','rb',None)],'ra','rb')]
    else:
        children=[('node',i,('work_'+i if kind=='namedMap' else None)) for i in ids+(['sum'] if branch else [])]
        root=container([],kind,children,'a','sum' if branch else 'b')
        if branch:root['inputs'][0]['targets'].append(endpoint('b','in'))
        containers=[root]
    if not nested:
        selected=containers[0];selected['id']=cid(['block']);selected['path']=['block']
        root=copy.deepcopy(selected);root.update(id='root',kind='graph',path=[],children=[dict(kind='container',id=selected['id'],key='0')])
        selected['inputs'][0]['id']='values';selected['outputs'][0]['id']='result'
        containers=[root,selected]
    graph=dict(schemaVersion=2,name='Actual containers',nodes=nodes,edges=wires,compositionPlan=dict(schemaVersion=1,root='root',containers=containers))
    return graph

def ports(graph):
    return {n['id']:{'inputs':[] if n['type']=='input' else ['a','b'] if n['type']=='add' else ['in'],'outputs':[] if n['type']=='output' else ['out']} for n in graph['nodes']}

def assign(model):
    values={}
    with torch.no_grad():
        for index,(meta,p) in enumerate(native_owners(model)):
            value=[((index*3+i)%11-5)/8 for i in range(p.numel())]
            p.copy_(torch.tensor(value,device=p.device).reshape(p.shape));values[(meta['node'],meta['parameter'])]=(list(p.shape),value)
    return values

def scalar(graph,parameters,rows,upstream):
    values={'input':rows};gradients={};incoming={n['id']:{} for n in graph['nodes']}
    for e in graph['edges']:incoming[e['target']][e['targetPort']]=e['source']
    todo=list(graph['nodes']);order=[]
    while todo:
        ready=[n for n in todo if all(s in values for s in incoming[n['id']].values())]
        assert ready
        for n in ready:
            i=n['id'];kind=n['type'];ins=incoming[i]
            if kind=='linear':
                w=parameters[(i,'weight')][1];b=parameters[(i,'bias')][1];x=values[ins['in']]
                values[i]=[[sum(x[r][j]*w[o*2+j] for j in range(2))+b[o] for o in range(2)] for r in range(len(rows))]
            elif kind=='add':values[i]=[[a+b for a,b in zip(x,y)] for x,y in zip(values[ins['a']],values[ins['b']])]
            elif kind=='relu':values[i]=[[max(0,v) for v in x] for x in values[ins['in']]]
            elif kind=='output':values[i]=values[ins['in']]
            order.append(n);todo.remove(n)
    gradients['output']=[list(r) for r in upstream];dp={k:[0.]*len(v[1]) for k,v in parameters.items()}
    def add(i,g):
        if i not in gradients:gradients[i]=[[0.,0.] for _ in rows]
        gradients[i]=[[a+b for a,b in zip(x,y)] for x,y in zip(gradients[i],g)]
    for n in reversed(order):
        i=n['id'];kind=n['type'];ins=incoming[i];g=gradients.get(i,[[0.,0.] for _ in rows])
        if kind=='linear':
            x=values[ins['in']];w=parameters[(i,'weight')][1]
            add(ins['in'],[[sum(g[r][o]*w[o*2+j] for o in range(2)) for j in range(2)] for r in range(len(rows))])
            dp[(i,'weight')]=[sum(g[r][o]*x[r][j] for r in range(len(rows))) for o in range(2) for j in range(2)]
            dp[(i,'bias')]=[sum(g[r][o] for r in range(len(rows))) for o in range(2)]
        elif kind=='add':add(ins['a'],g);add(ins['b'],g)
        elif kind=='relu':add(ins['in'],[[v if x>0 else 0. for v,x in zip(gr,row)] for gr,row in zip(g,values[ins['in']])])
        elif kind=='output':add(ins['in'],g)
    return values['output'],gradients['input'],dp

@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('kind,branch,nested,functional',[('graph',True,False,False),('sequential',False,False,False),('orderedList',True,False,False),('namedMap',True,False,False),('graph',False,True,False),('sequential',False,False,True)])
def test_real_native_container_math_and_all_owned_influences(device,kind,branch,nested,functional):
    graph=fixture(kind,branch,nested,functional);model=GraphModel(graph,bundled_dataset=False).to(device)
    expected_type={'graph':torch.nn.Module,'sequential':torch.nn.Sequential,'orderedList':torch.nn.ModuleList,'namedMap':torch.nn.ModuleDict}[kind]
    assert isinstance(model.composition_root,torch.nn.Module)
    assert isinstance(model.get_submodule('composition_root.blocks.0'),expected_type)
    assert isinstance(model.layers,dict) and 'layers' not in model._modules
    parameters=assign(model);rows=[[-2.,.5],[1.25,3.]];upstream=[[.25,-1.],[2.,.5]]
    expected,dx,dp=scalar(graph,parameters,rows,upstream)
    x=torch.tensor(rows,device=device,requires_grad=True);out,stages=model(x,inspect=True)
    torch.testing.assert_close(out.cpu(),torch.tensor(expected),atol=2e-5,rtol=2e-5)
    (out*torch.tensor(upstream,device=device)).sum().backward()
    torch.testing.assert_close(x.grad.cpu(),torch.tensor(dx),atol=2e-5,rtol=2e-5)
    owners=native_owners(model)
    for meta,p in owners:
        assert meta['name'].startswith('composition_root.') and '.operation.' in meta['name']
        torch.testing.assert_close(p.grad.flatten().cpu(),torch.tensor(dp[(meta['node'],meta['parameter'])]),atol=2e-5,rtol=2e-5)
        assert model.get_submodule(model.layer_keys[meta['node']]) is model.layers[model.layer_keys[meta['node']]]
    assert len(owners)==len(list(model.named_parameters(remove_duplicate=False)))
    assert set(stages)=={n['id'] for n in graph['nodes']}


def test_sequence_native_forward_called_once_and_each_leaf_once():
    model=GraphModel(fixture(),bundled_dataset=False);assign(model);calls=[];events=[]
    hooks=[module.register_forward_hook(lambda m,a,o:calls.append(type(m))) for module in model.modules() if isinstance(module,torch.nn.Sequential)]
    model(torch.ones(2,2),operation_observer=lambda phase,node:events.append((phase,node)))
    for h in hooks:h.remove()
    assert calls==[torch.nn.Sequential]
    assert sorted(events)==sorted((phase,node) for phase in ['started','completed'] for node in ['input','a','b','output'])


def test_registration_order_is_actual_storage_order_independent_of_dag():
    graph=fixture('orderedList',True);graph['compositionPlan']['containers'][1]['children'].reverse()
    for i,c in enumerate(graph['compositionPlan']['containers'][1]['children']):c['key']=str(i)
    model=GraphModel(graph,bundled_dataset=False)
    assert [m['node'] for m,p in native_owners(model)]==['b','b','a','a']
    assert model.layer_keys['b']=='composition_root.blocks.0.1.operation'


def test_private_instances_and_strict_state_roundtrip():
    graph=fixture(nested=True);model=GraphModel(graph,bundled_dataset=False);assign(model)
    before={k:v.clone() for k,v in model.state_dict().items()};owners=native_owners(model)
    la=next(p for m,p in owners if (m['node'],m['parameter'])==('la','weight'))
    ra=next(p for m,p in owners if (m['node'],m['parameter'])==('ra','weight'))
    assert la is not ra and la.data_ptr()!=ra.data_ptr()
    with torch.no_grad():la.add_(1)
    assert torch.equal(ra,before[next(m['name'] for m,p in owners if p is ra)])
    stream=io.BytesIO();torch.save(model.state_dict(),stream);stream.seek(0)
    restored=GraphModel(graph,bundled_dataset=False);restored.load_state_dict(torch.load(stream,weights_only=True))
    torch.testing.assert_close(restored(torch.ones(2,2)),model(torch.ones(2,2)),atol=0,rtol=0)


def test_nested_mode_propagates_and_frozen_flags_stay_frozen():
    graph=fixture('sequential',functional=True);graph['nodes'][2]={'id':'b','type':'dropout','config':{'p':1.}}
    model=GraphModel(graph,bundled_dataset=False);assign(model);first=next(model.parameters());first.requires_grad_(False)
    model.train();assert torch.equal(model(torch.ones(2,2)),torch.zeros(2,2))
    assert all(m.training for m in model.modules())
    model.eval();assert torch.any(model(torch.ones(2,2))!=0)
    assert all(not m.training for m in model.modules()) and not first.requires_grad


def test_plan_changes_semantic_identity_and_export_retains_plan():
    g=fixture('orderedList');other=copy.deepcopy(g);other['compositionPlan']['containers'][1]['children'].reverse()
    for i,c in enumerate(other['compositionPlan']['containers'][1]['children']):c['key']=str(i)
    assert semantic_graph(g)!=semantic_graph(other)
    source=export_source(GraphModel(g,bundled_dataset=False))
    assert 'COMPOSITION_GRAPH = ' in source and repr(g['compositionPlan']) in source

@pytest.mark.parametrize('mutation',['null','boolVersion','extra','root','rootPath','kind','missingLeaf','duplicateLeaf','foreignLeaf','missingBoundary','foreignBoundary','wrongInputPort','wrongOutputPort','wrongRootInput','wrongRootOutput','wrongNumericKey','mapCollision','duplicateKey','overContainers','overChildren','overEndpoints','overBytes','sequentialOrder','auxiliaryOutput','duplicateContainer','wrongContainerID','preorder','pathDepth','cycle','aliasContainer'])
def test_forged_plan_refuses_before_native_parameter_allocation(monkeypatch,mutation):
    graph=fixture(nested=mutation in ['duplicateContainer','wrongContainerID','preorder','pathDepth','cycle','aliasContainer']);p=graph['compositionPlan'];root=p['containers'][0];selected=p['containers'][1]
    if mutation=='null':graph['compositionPlan']=None
    elif mutation=='boolVersion':p['schemaVersion']=True
    elif mutation=='extra':p['extra']=1
    elif mutation=='root':p['root']='other'
    elif mutation=='rootPath':root['path']=['foreign']
    elif mutation=='kind':selected['kind']='fake'
    elif mutation=='missingLeaf':selected['children'].pop()
    elif mutation=='duplicateLeaf':selected['children'][1]['id']='a'
    elif mutation=='foreignLeaf':selected['children'][0]['id']='foreign'
    elif mutation=='missingBoundary':root['inputs']=[]
    elif mutation=='foreignBoundary':root['inputs'][0]['targets'][0]['node']='foreign'
    elif mutation=='wrongInputPort':root['inputs'][0]['targets'][0]['port']='out'
    elif mutation=='wrongOutputPort':root['outputs'][0]['source']['port']='in'
    elif mutation=='wrongRootInput':root['inputs'][0]['id']='foreign'
    elif mutation=='wrongRootOutput':root['outputs'][0]['id']='foreign'
    elif mutation=='wrongNumericKey':selected['children'][0]['key']='named'
    elif mutation=='mapCollision':selected['kind']='namedMap';selected['children'][0]['key']='training';selected['children'][1]['key']='safe'
    elif mutation=='duplicateKey':selected['kind']='namedMap';selected['children'][0]['key']=selected['children'][1]['key']='same'
    elif mutation=='overContainers':p['containers']=p['containers']*130
    elif mutation=='overChildren':selected['children']=selected['children']*129
    elif mutation=='overEndpoints':root['inputs'][0]['targets']*=4097
    elif mutation=='overBytes':root['inputs'][0]['id']='x'*131073
    elif mutation=='sequentialOrder':selected['children'].reverse();[c.update(key=str(i)) for i,c in enumerate(selected['children'])]
    elif mutation=='auxiliaryOutput':graph['nodes'][1]={'id':'a','type':'max_pool1d','config':{'kernel_size':1,'stride':1,'padding':0,'dilation':1,'ceil_mode':False,'return_indices':True}} # base geometry or strict sequence must refuse
    elif mutation=='duplicateContainer':p['containers'].append(copy.deepcopy(p['containers'][1]))
    elif mutation=='wrongContainerID':p['containers'][1]['id']='c-wrong'
    elif mutation=='preorder':p['containers'].reverse()
    elif mutation=='pathDepth':p['containers'][1]['path']=['x']*9
    elif mutation=='cycle':root['children'][0]['id']='root'
    elif mutation=='aliasContainer':root['children'][1]['id']=root['children'][0]['id']
    def forbidden(*a,**k):raise AssertionError('Invalid composition plan reached parameter allocation')
    monkeypatch.setattr(torch.nn.Linear,'__init__',forbidden)
    with pytest.raises(ValueError):GraphModel(graph,bundled_dataset=False)


def test_legacy_absence_keeps_registration_semantics_and_export():
    graph=fixture();del graph['compositionPlan'];model=GraphModel(graph,bundled_dataset=False)
    assert isinstance(model.layers,torch.nn.ModuleDict)
    assert [name for name,p in model.named_parameters()]==['layers.a.weight','layers.a.bias','layers.b.weight','layers.b.bias']
    assert semantic_graph(graph)=={'schemaVersion':1,'nodes':[{'id':'a','type':'linear','config':{'in_features':2,'out_features':2}},{'id':'b','type':'linear','config':{'in_features':2,'out_features':2}},{'id':'input','type':'input','config':{'features':2}},{'id':'output','type':'output','config':{}}],'edges':[{'source':'a','sourcePort':'out','target':'b','targetPort':'in'},{'source':'b','sourcePort':'out','target':'output','targetPort':'in'},{'source':'input','sourcePort':'out','target':'a','targetPort':'in'}]}
    assert 'self.layers = nn.ModuleDict()' in export_source(model)

@pytest.mark.parametrize('device',DEVICES)
def test_native_buffer_bijection_mode_and_strict_reload(device):
    from runner.native_ownership import state_owners
    graph=fixture();graph['nodes'][2]={'id':'b','type':'batch_norm1d','config':{'num_features':2}}
    model=GraphModel(graph,bundled_dataset=False).to(device)
    model(torch.tensor([[1.,-2.],[3.,4.]],device=device))
    buffers=state_owners(model)
    assert [(m['node'],m['parameter'],m['dtype']) for m,_ in buffers]==[('b','running_mean','float32'),('b','running_var','float32'),('b','num_batches_tracked','int64')]
    parameter_names=[m['name'] for m,_ in native_owners(model)]
    buffer_names=[m['name'] for m,_ in buffers]
    assert set(parameter_names+buffer_names)==set(model.state_dict())
    assert all(model.get_buffer(m['name']) is p for m,p in buffers)
    assert buffers[-1][1].item()==1
    model.eval();saved={k:v.detach().clone() for k,v in model.state_dict().items()}
    restored=GraphModel(graph,bundled_dataset=False).to(device).eval();restored.load_state_dict(saved,strict=True)
    x=torch.tensor([[.3,-.7],[2.,1.]],device=device)
    assert torch.equal(model(x),restored(x))
    with pytest.raises(RuntimeError):restored.load_state_dict({k:v for k,v in saved.items() if k!=buffer_names[0]},strict=True)


def test_operations_constructed_once_and_initial_rng_matches_legacy(monkeypatch):
    graph=fixture();legacy=copy.deepcopy(graph);legacy.pop('compositionPlan')
    original=torch.nn.Linear.__init__;calls=[]
    def count(instance,*args,**kwargs):calls.append((args,kwargs));original(instance,*args,**kwargs)
    monkeypatch.setattr(torch.nn.Linear,'__init__',count)
    torch.manual_seed(971);composed=GraphModel(graph,bundled_dataset=False);after=torch.get_rng_state().clone()
    assert len(calls)==2
    torch.manual_seed(971);flat=GraphModel(legacy,bundled_dataset=False)
    assert torch.equal(after,torch.get_rng_state())
    assert all(torch.equal(a,b) for a,b in zip(composed.parameters(),flat.parameters()))
    assert len(list(composed.named_modules(remove_duplicate=False)))==len({id(m) for _,m in composed.named_modules(remove_duplicate=False)})


def test_per_call_frame_supports_reentrant_invocation_and_keeps_input_unchanged():
    model=GraphModel(fixture(),bundled_dataset=False);assign(model)
    x=torch.tensor([[1.,2.],[-1.,3.]]);before=x.clone();inner=torch.tensor([[7.,-4.]])
    expected=model(x).clone();expected_inner=model(inner).clone();triggered=[]
    def observer(phase,node):
        if phase=='started' and node=='b' and not triggered:
            triggered.append(model(inner).clone())
    assert torch.equal(model(x,operation_observer=observer),expected)
    assert torch.equal(triggered[0],expected_inner) and torch.equal(x,before)


def test_sequence_auxiliary_configured_port_refuses_before_allocation():
    graph=fixture();child_ports=ports(graph);child_ports['a']['outputs']=['out','indices']
    with pytest.raises(ValueError,match='auxiliary'):validate_plan(graph['compositionPlan'],graph,child_ports)


def test_native_lookup_and_parameter_alias_tampering_refuse():
    model=GraphModel(fixture(),bundled_dataset=False)
    model.layers[model.layer_keys['a']]=model.layers[model.layer_keys['b']]
    with pytest.raises(ValueError,match='lookup'):native_owners(model)
    model=GraphModel(fixture(),bundled_dataset=False)
    model.layers[model.layer_keys['b']].weight=model.layers[model.layer_keys['a']].weight
    with pytest.raises(ValueError,match='Aliased'):native_owners(model)
