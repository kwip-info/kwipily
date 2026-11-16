"""Named coordinate boundaries through real native ownership; literal bag math."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import uuid
import pytest
import torch
from runner import engine
from runner.composite_source import checked_source
from runner.module_composition_source import reexpand
from runner.module_composition_plan import container_id
from runner.native_ownership import native_owners,composition_identity
from runner.embedding_bag_packet import collate_bags
from runner.tests.test_bag_source_graph import graph,packet,registry
from runner.tests.test_bag_export import TABLE,OUTPUTS,GRADS,UPSTREAM
DEVICES=['cpu']+(['mps'] if torch.backends.mps.is_available() else [])


def fixture(kind='graph',layout='packed',weighted=True):
    value=graph(layout,weighted);value['nodes'].insert(2,dict(id='head',type='linear',config={'in_features':2,'out_features':2,'bias':False}))
    value['edges'][-1]['target']='head';value['edges'].append(dict(id='headout',source='head',sourcePort='out',target='result',targetPort='in'))
    roles=['indices']+(['offsets'] if layout=='packed' else [])+(['per_sample_weights'] if weighted else [])
    target=lambda p:dict(node='bag',port=p)
    root=dict(id='root',kind='graph',path=[],children=[dict(kind='container',id=container_id(['block']),key='0')],inputs=[dict(id='source',targets=[target(p) for p in roles])],outputs=[dict(id='result',source=dict(node='head',port='out'))])
    children=[dict(kind='node',id='bag',key='lookup' if kind=='namedMap' else '0'),dict(kind='node',id='head',key='projection' if kind=='namedMap' else '1')]
    block=dict(id=container_id(['block']),kind=kind,path=['block'],children=children,inputs=[dict(id=p,targets=[target(p)]) for p in roles],outputs=[dict(id='reduced',source=dict(node='head',port='out'))])
    value['compositionPlan']=dict(schemaVersion=1,root='root',containers=[root,block]);return value


def assign(model):
    with torch.no_grad():
        model.layers[model.layer_keys['bag']].weight.copy_(torch.tensor(TABLE,device=next(model.parameters()).device))
        model.layers[model.layer_keys['head']].weight.copy_(torch.eye(2,device=next(model.parameters()).device))


def exported(tmp_path,model):
    path=tmp_path/(uuid.uuid4().hex+'.py');path.write_text(engine.export_source(model));name='bag_composition_'+uuid.uuid4().hex
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    out=m.Model();out.load_state_dict(model.state_dict(),strict=True);return out,path


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('kind,layout,weighted',[('graph','packed',True),('orderedList','packed',True),('namedMap','packed',True),('graph','fixed',True),('orderedList','packed',False),('sequential','fixed',False)])
def test_real_native_named_boundaries_state_and_all_gradients(registry,tmp_path,device,kind,layout,weighted):
    value=fixture(kind,layout,weighted);model=engine.GraphModel(value,registry,bundled_dataset=False).to(device);assign(model)
    expected_type={'graph':torch.nn.Module,'orderedList':torch.nn.ModuleList,'namedMap':torch.nn.ModuleDict,'sequential':torch.nn.Sequential}[kind]
    block=model.get_submodule('composition_root.blocks.0');assert isinstance(block,expected_type)
    calls=[];hook=block.register_forward_hook(lambda *args:calls.append(1))
    item=packet(device,layout,weighted);out,stages=model(item,inspect=True);hook.remove();assert calls==[1]
    expected=[[5.5,-3.],[0.,0.],[-6.,12.],[.75,1.]] if weighted else OUTPUTS['sum']
    torch.testing.assert_close(out,torch.tensor(expected,device=device))
    for role,tensor in stages['source'].items():assert tensor is getattr(item,role)
    (out*torch.tensor(UPSTREAM,device=device)).sum().backward()
    grad=[ [0.,0.],[-1.,.5],[4.,-2.],[-9.,15.],[1.,1.5]] if weighted else GRADS['sum']
    torch.testing.assert_close(model.layers[model.layer_keys['bag']].weight.grad,torch.tensor(grad,device=device))
    expected_head=[[sum(UPSTREAM[r][o]*expected[r][i] for r in range(4)) for i in range(2)] for o in range(2)]
    torch.testing.assert_close(model.layers[model.layer_keys['head']].weight.grad,torch.tensor(expected_head,device=device))
    if weighted:
        wanted=[0.,7.,0.,0.,26.,36.,0.] if layout=='packed' else [[0.,7.,0.],[0.,0.,0.],[0.,26.,0.],[36.,0.,0.]]
        torch.testing.assert_close(item.per_sample_weights.grad,torch.tensor(wanted,device=device))
    owners=native_owners(model);assert len(owners)==2 and all(m['parameter']=='weight' and m['name'].count('.operation.')==1 for m,_ in owners)
    assert len(composition_identity(model)['stateOwnership'])==2
    clone=engine.GraphModel(value,registry,bundled_dataset=False).to(device);clone.load_state_dict(model.state_dict(),strict=True)
    torch.testing.assert_close(clone(item),out)
    portable,path=exported(tmp_path,model);portable.to(device)
    assert list(portable.state_dict())==list(model.state_dict())
    portable_weights=item.per_sample_weights.detach().clone().requires_grad_() if weighted else None
    portable_out=portable(item.indices,item.offsets,portable_weights)
    torch.testing.assert_close(portable_out,out)
    (portable_out*torch.tensor(UPSTREAM,device=device)).sum().backward()
    torch.testing.assert_close(portable.get_submodule(model.layer_keys['bag']).weight.grad,torch.tensor(grad,device=device))
    torch.testing.assert_close(portable.get_submodule(model.layer_keys['head']).weight.grad,torch.tensor(expected_head,device=device))
    if weighted:torch.testing.assert_close(portable_weights.grad,torch.tensor(wanted,device=device))
    micro=[]
    for row in range(4):
        one,_=collate_bags(item,[row],num_embeddings=5,embedding_dim=2,mode='sum')
        micro.append(portable(one.indices,one.offsets,one.per_sample_weights))
    torch.testing.assert_close(torch.cat(micro),out)


@pytest.mark.parametrize('mutation',['missing','duplicate','disabledRole','wrongRole','foreignSource','internalSource','sequentialMulti','nestedCombine'])
def test_forged_boundaries_refuse_before_native_allocation(registry,monkeypatch,mutation):
    value=fixture();root,block=value['compositionPlan']['containers']
    if mutation=='missing':root['inputs'][0]['targets'].pop()
    elif mutation=='duplicate':root['inputs'][0]['targets'].append(copy.deepcopy(root['inputs'][0]['targets'][0]))
    elif mutation=='disabledRole':value['nodes'][0]['config']['per_sample_weights']=False
    elif mutation=='wrongRole':value['edges'][1]['sourcePort']='indices'
    elif mutation=='foreignSource':value['edges'][1]['source']='head'
    elif mutation=='internalSource':block['children'].append(dict(kind='node',id='source',key='2'))
    elif mutation=='sequentialMulti':block['kind']='sequential'
    else:block['inputs']=[dict(id='all',targets=root['inputs'][0]['targets'])]
    monkeypatch.setattr(torch.nn.EmbeddingBag,'__init__',lambda *a,**k:pytest.fail('Native allocation must not happen'))
    with pytest.raises(ValueError):engine.GraphModel(value,registry,bundled_dataset=False)


def source_document(kind='namedMap'):
    flat=fixture(kind);source=copy.deepcopy(flat['nodes'][0]);result=copy.deepcopy(flat['nodes'][-1]);body=copy.deepcopy(flat['nodes'][1:-1])
    for n in [source,result,*body]:n.update(name=n['id'],x=0,y=0)
    ref=dict(id='bag-block',version=dict(major=1,minor=0,patch=0));roles=['indices','offsets','per_sample_weights']
    definition=dict(reference=ref,title='Bag block',body=dict(configurationVersion=2,nodes=body,instances=[],edges=[dict(id='reduce',source='bag',sourcePort='out',target='head',targetPort='in')]),inputs=[dict(id=p,targets=[dict(componentID='bag',portID=p)]) for p in roles],outputs=[dict(id='out',source=dict(componentID='head',portID='out'))],parameters=[],composition=dict(schemaVersion=1,kind=kind,children=[dict(localID='bag',key='lookup'),dict(localID='head',key='projection')] if kind=='namedMap' else [dict(localID='bag'),dict(localID='head')]))
    edges=[dict(id=p,source='source',sourcePort=p,target='placed',targetPort=p) for p in roles]+[dict(id='leave',source='placed',sourcePort='out',target='result',targetPort='in')]
    return dict(schemaVersion=3,name='Bag source composition',root=dict(configurationVersion=2,nodes=[source,result],instances=[dict(id='placed',definition=ref,name='Placed',x=0,y=0,overrides={})],edges=edges),definitions=[definition])


@pytest.mark.parametrize('kind',['graph','orderedList','namedMap'])
def test_independent_source_reexpansion_and_rehashed_forgery(registry,kind):
    document=source_document(kind);expanded=reexpand(document,registry)
    source=dict(nativeExpansionContractVersion=2,sourceCompositeDocument=document,**expanded)
    def verify(value):
        raw=json.dumps(value,separators=(',',':')).encode();return checked_source(raw,hashlib.sha256(raw).hexdigest(),expanded['expandedGraph'],registry)
    assert verify(source)
    changed=copy.deepcopy(source);changed['expandedGraph']['compositionPlan']['containers'][0]['inputs'][0]['targets'].pop()
    with pytest.raises(ValueError):verify(changed)
    internal=copy.deepcopy(document);internal['definitions'][0]['body']['nodes'].append(copy.deepcopy(document['root']['nodes'][0]))
    with pytest.raises(ValueError):reexpand(internal,registry)


def test_exported_hierarchy_outside_checkout(registry,tmp_path):
    model=engine.GraphModel(fixture(),registry,bundled_dataset=False);assign(model);portable,path=exported(tmp_path,model)
    torch.save(model.state_dict(),tmp_path/'weights.pt')
    code="""import importlib.util,sys,torch
s=importlib.util.spec_from_file_location('portable',sys.argv[1]);m=importlib.util.module_from_spec(s);sys.modules['portable']=m;s.loader.exec_module(m)
x=m.Model();x.load_state_dict(torch.load(sys.argv[2],weights_only=True),strict=True)
y=x(torch.tensor([1,2,1,0,3,4,0]),torch.tensor([0,3,3,5,7]),torch.tensor([.5,2.,-1.,99.,3.,.25,88.]))
assert y.tolist()==[[5.5,-3.],[0.,0.],[-6.,12.],[.75,1.]]
assert not any(n=='runner' or n.startswith('runner.') for n in sys.modules)
print('PASS')
"""
    done=subprocess.run([sys.executable,'-I','-c',code,str(path),str(tmp_path/'weights.pt')],cwd=tmp_path,capture_output=True,text=True,timeout=30)
    assert done.returncode==0,done.stderr
    assert done.stdout.strip()=='PASS'


@pytest.mark.parametrize('device',DEVICES)
def test_shared_root_named_roles_feed_private_tables_and_downstream_sequence(registry,device):
    value=fixture('orderedList');p=value['compositionPlan'];block=p['containers'][1]
    # Two actual bags independently own tables, while all three source roles fan out.
    other=copy.deepcopy(value['nodes'][1]);other['id']='other';value['nodes'].insert(2,other)
    value['nodes'].insert(3,dict(id='join',type='add',config={}))
    value['edges']=[e for e in value['edges'] if e['source']!='bag' or e['target']!='head']
    for role in ['indices','offsets','per_sample_weights']:
        value['edges'].append(dict(id='other_'+role,source='source',sourcePort=role,target='other',targetPort=role))
        target=dict(node='other',port=role);p['containers'][0]['inputs'][0]['targets'].append(target);next(i for i in block['inputs'] if i['id']==role)['targets'].append(copy.deepcopy(target))
    value['edges'] += [dict(id='left',source='bag',sourcePort='out',target='join',targetPort='a'),dict(id='right',source='other',sourcePort='out',target='join',targetPort='b'),dict(id='joined',source='join',sourcePort='out',target='head',targetPort='in')]
    sequence=dict(id=container_id(['block','chain']),kind='sequential',path=['block','chain'],children=[dict(kind='node',id='head',key='0')],inputs=[dict(id='values',targets=[dict(node='head',port='in')])],outputs=[dict(id='reduced',source=dict(node='head',port='out'))])
    block['children']=[dict(kind='node',id=i,key=str(j)) for j,i in enumerate(['other','bag','join'])]+[dict(kind='container',id=sequence['id'],key='3')];p['containers'].append(sequence)
    model=engine.GraphModel(value,registry,bundled_dataset=False).to(device);assign(model)
    with torch.no_grad():model.layers[model.layer_keys['other']].weight.copy_(2*torch.tensor(TABLE,device=device))
    left=model.layers[model.layer_keys['bag']].weight;right=model.layers[model.layer_keys['other']].weight
    assert left is not right and left.data_ptr()!=right.data_ptr()
    calls=[];hook=model.get_submodule('composition_root.blocks.0.3').register_forward_hook(lambda *a:calls.append(1))
    item=packet(device);out=model(item);hook.remove();assert calls==[1]
    torch.testing.assert_close(out,3*torch.tensor([[5.5,-3.],[0.,0.],[-6.,12.],[.75,1.]],device=device))
    (out*torch.tensor(UPSTREAM,device=device)).sum().backward()
    expected=torch.tensor([[0.,0.],[-1.,.5],[4.,-2.],[-9.,15.],[1.,1.5]],device=device)
    torch.testing.assert_close(left.grad,expected);torch.testing.assert_close(right.grad,expected)
    torch.testing.assert_close(item.per_sample_weights.grad,3*torch.tensor([0.,7.,0.,0.,26.,36.,0.],device=device))


@pytest.mark.parametrize('device',DEVICES)
def test_frozen_composed_table_keeps_real_input_weight_influence(registry,tmp_path,device):
    value=fixture('namedMap');value['nodes'][1]['config']['freeze']=True
    model=engine.GraphModel(value,registry,bundled_dataset=False).to(device);assign(model)
    item=packet(device);(model(item)*torch.tensor(UPSTREAM,device=device)).sum().backward()
    assert model.layers[model.layer_keys['bag']].weight.grad is None
    expected=torch.tensor([0.,7.,0.,0.,26.,36.,0.],device=device)
    torch.testing.assert_close(item.per_sample_weights.grad,expected)
    portable,_=exported(tmp_path,model);portable.to(device)
    assert not portable.get_submodule(model.layer_keys['bag']).weight.requires_grad
    weights=item.per_sample_weights.detach().clone().requires_grad_()
    (portable(item.indices,item.offsets,weights)*torch.tensor(UPSTREAM,device=device)).sum().backward()
    torch.testing.assert_close(weights.grad,expected)
