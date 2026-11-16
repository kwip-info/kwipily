"""Literal source/expansion fixtures; no engine adapter supplies expected expansion."""
import copy
import hashlib
import json
from pathlib import Path
import pytest
from runner.composite_source import checked_source
from runner.module_composition_source import reexpand

REGISTRY = Path(__file__).resolve().parents[2]/'Sources/KwipMLCore/Resources/components.json'
REF = {'id': 'block', 'version': {'major': 1, 'minor': 0, 'patch': 0}}

def node(ident, kind, config=None):
    return dict(id=ident,type=kind,name='',x=0,y=0,config=config or {})

def edge(ident, source, target, source_port='out', target_port='in'):
    return dict(id=ident,source=source,sourcePort=source_port,target=target,targetPort=target_port)

def swift_id(kind, path, ident):
    raw=json.dumps([kind,*path,ident],ensure_ascii=False,separators=(',',':')).replace('/', '\\/').encode()
    return kind+'-'+hashlib.sha256(raw).hexdigest() if path else ident

def fixture(kind='sequential', ident='placed'):
    body=dict(configurationVersion=2,nodes=[node('a','linear',dict(in_features=2,out_features=2,bias=True)),node('b','relu')],instances=[],edges=[edge('ab','a','b')])
    children=[dict(localID='a'),dict(localID='b')]
    if kind=='namedMap':children=[dict(localID='a',key='first'),dict(localID='b',key='second')]
    definition=dict(reference=copy.deepcopy(REF),title='Block',body=body,inputs=[dict(id='in',targets=[dict(componentID='a',portID='in')])],outputs=[dict(id='out',source=dict(componentID='b',portID='out'))],parameters=[],composition=dict(schemaVersion=1,kind=kind,children=children))
    root=dict(configurationVersion=2,nodes=[node('input','input',dict(features=2)),node('output','output')],instances=[dict(id=ident,definition=copy.deepcopy(REF),name='',x=0,y=0,overrides={})],edges=[edge('enter','input',ident),edge('leave',ident,'output')])
    document=dict(schemaVersion=3,name='Example',root=root,definitions=[definition])
    a,b=swift_id('n',[ident],'a'),swift_id('n',[ident],'b')
    cid='c-'+hashlib.sha256(str(len(ident.encode())).encode()+b':'+ident.encode()).hexdigest()
    ep=lambda n,p:dict(node=n,port=p)
    plan=dict(schemaVersion=1,root='root',containers=[
        dict(id='root',kind='graph',path=[],children=[dict(kind='container',id=cid,key='0')],inputs=[dict(id='input',targets=[ep(a,'in')])],outputs=[dict(id='output',source=ep(b,'out'))]),
        dict(id=cid,kind=kind,path=[ident],children=[dict(kind='node',id=a,key='first' if kind=='namedMap' else '0'),dict(kind='node',id=b,key='second' if kind=='namedMap' else '1')],inputs=[dict(id='in',targets=[ep(a,'in')])],outputs=[dict(id='out',source=ep(b,'out'))])])
    graph=dict(schemaVersion=2,name='Example',nodes=[*copy.deepcopy(root['nodes']),dict(node(a,'linear',dict(in_features=2,out_features=2,bias=True))),node(b,'relu')],edges=[edge(swift_id('e',[ident],'ab'),a,b),edge('enter','input',a),edge('leave',b,'output')],compositionPlan=plan)
    origins={n['id']:dict(instancePath=[],localNodeID=n['id']) for n in root['nodes']}
    origins.update({a:dict(instancePath=[ident],localNodeID='a',definition=copy.deepcopy(REF)),b:dict(instancePath=[ident],localNodeID='b',definition=copy.deepcopy(REF))})
    snapshot=dict(nativeExpansionContractVersion=2,sourceCompositeDocument=document,expandedGraph=graph,nodeOrigins=origins,consumedDefinitions=[copy.deepcopy(REF)])
    return snapshot

def check(snapshot, graph=None, registry=REGISTRY):
    raw=json.dumps(snapshot,ensure_ascii=False,separators=(',',':')).encode()
    return checked_source(raw,hashlib.sha256(raw).hexdigest(),graph or snapshot['expandedGraph'],registry)

@pytest.mark.parametrize('kind',['graph','sequential','orderedList','namedMap'])
def test_literal_all_native_container_source_reexpands_exactly_without_mutation(kind):
    s=fixture(kind);before=copy.deepcopy(s)
    expected={k:s[k] for k in ('expandedGraph','nodeOrigins','consumedDefinitions')}
    assert reexpand(s['sourceCompositeDocument'],REGISTRY)==expected
    actual=check(s)
    assert actual['nativeExpansionContractVersion']==2 and actual['expandedSemanticDigest']!='unverified'
    assert s==before

@pytest.mark.parametrize('mutation',[
    lambda s:s['sourceCompositeDocument']['definitions'][0]['body']['nodes'][0]['config'].update(out_features=3),
    lambda s:s['sourceCompositeDocument']['definitions'][0]['composition']['children'].reverse(),
    lambda s:s['sourceCompositeDocument']['definitions'][0]['reference']['version'].update(patch=1),
    lambda s:s['nodeOrigins'][next(k for k in s['nodeOrigins'] if k.startswith('n-'))].update(localNodeID='foreign'),
    lambda s:s['consumedDefinitions'][0]['version'].update(patch=1),
    lambda s:s['expandedGraph']['compositionPlan']['containers'][1]['children'][0].update(key='other'),
    lambda s:s['expandedGraph']['nodes'][-1].update(name='source does not produce this'),
    lambda s:s['sourceCompositeDocument'].update(script='executable'),
    lambda s:s['sourceCompositeDocument']['definitions'][0]['body']['nodes'][0].update(script='executable'),
    lambda s:s['sourceCompositeDocument']['definitions'][0]['composition'].update(schemaVersion=True),
    lambda s:s['sourceCompositeDocument'].update(schemaVersion=2),
    lambda s:s['sourceCompositeDocument']['root']['instances'][0].update(overrides={'undeclared':2}),
])
def test_changed_source_plan_ownership_and_unknown_fields_refuse_even_with_rehashed_envelope(mutation):
    s=fixture();mutation(s)
    with pytest.raises(ValueError):check(s)

def test_registry_is_required_and_current_requested_plan_is_not_cosmetic():
    s=fixture()
    with pytest.raises(ValueError):check(s,registry=None)
    graph=copy.deepcopy(s['expandedGraph']);graph['compositionPlan']['containers'][1]['kind']='orderedList'
    with pytest.raises(ValueError):check(s,graph)

def test_unicode_slash_namespace_and_source_positions_are_exact():
    s=fixture(ident='α/placed');s['sourceCompositeDocument']['root']['instances'][0].update(x=2,y=-3)
    for n in s['expandedGraph']['nodes'][2:]:n.update(x=2,y=-3)
    expected={k:s[k] for k in ('expandedGraph','nodeOrigins','consumedDefinitions')}
    assert reexpand(s['sourceCompositeDocument'],REGISTRY)==expected
    assert check(s)

@pytest.mark.parametrize('bad',['cycle','depth','null','duplicate','budget'])
def test_malformed_bounded_source_is_refused_before_native_allocation(bad,monkeypatch):
    import torch
    monkeypatch.setattr(torch.nn.Linear,'__init__',lambda *a,**k:pytest.fail('native allocation before source proof'))
    s=fixture()
    d=s['sourceCompositeDocument']['definitions'][0]
    if bad=='cycle':d['body']['instances']=[dict(id='recursive',definition=copy.deepcopy(REF),name='',x=0,y=0,overrides={})]
    elif bad=='depth':s['nodeOrigins'][next(k for k in s['nodeOrigins'] if k.startswith('n-'))]['instancePath']=['x']*9
    elif bad=='null':d['composition']=None
    elif bad=='duplicate':s['sourceCompositeDocument']['definitions'].append(copy.deepcopy(d))
    else:d['title']='x'*2_000_001
    with pytest.raises(ValueError):check(s)

def test_nested_private_instances_and_forwarded_boolean_override_are_independent():
    s=fixture();inner=s['sourceCompositeDocument']['definitions'][0]
    flag=dict(key='useBias',title='Use bias',kind='boolean',defaultValue=True)
    inner['parameters']=[dict(field=flag,targets=[dict(componentID='a',field='bias')])]
    outer_ref=dict(id='outer',version=dict(major=1,minor=0,patch=0))
    instances=[dict(id=ident,definition=copy.deepcopy(REF),name='',x=0,y=0,overrides={}) for ident in ('left','right')]
    outer=dict(reference=outer_ref,title='Outer',body=dict(configurationVersion=2,nodes=[],instances=instances,edges=[edge('lr','left','right')]),inputs=[dict(id='in',targets=[dict(componentID='left',portID='in')])],outputs=[dict(id='out',source=dict(componentID='right',portID='out'))],parameters=[dict(field=dict(flag,key='leftBias'),targets=[dict(componentID='left',field='useBias')])],composition=dict(schemaVersion=1,kind='orderedList',children=[dict(localID='right'),dict(localID='left')]))
    s['sourceCompositeDocument']['definitions'].append(outer)
    s['sourceCompositeDocument']['root']['instances'][0].update(definition=outer_ref,overrides=dict(leftBias=False))
    expected_nodes=copy.deepcopy(s['expandedGraph']['nodes'][:2]);expected_edges=[];origins={k:v for k,v in s['nodeOrigins'].items() if k in ('input','output')}
    rows={}
    for ident in ('left','right'):
        path=['placed',ident];a,b=swift_id('n',path,'a'),swift_id('n',path,'b')
        expected_nodes.extend([node(a,'linear',dict(in_features=2,out_features=2,bias=ident=='right')),node(b,'relu')])
        expected_edges.append(edge(swift_id('e',path,'ab'),a,b))
        for local,flat in (('a',a),('b',b)):origins[flat]=dict(instancePath=path,localNodeID=local,definition=copy.deepcopy(REF))
        cid='c-'+hashlib.sha256(b''.join(str(len(p.encode())).encode()+b':'+p.encode() for p in path)).hexdigest()
        rows[ident]=dict(id=cid,kind='sequential',path=path,children=[dict(kind='node',id=a,key='0'),dict(kind='node',id=b,key='1')],inputs=[dict(id='in',targets=[dict(node=a,port='in')])],outputs=[dict(id='out',source=dict(node=b,port='out'))])
    la,rb=expected_nodes[2]['id'],expected_nodes[-1]['id']
    expected_edges.extend([edge(swift_id('e',['placed'],'lr'),expected_nodes[3]['id'],expected_nodes[4]['id']),edge('enter','input',la),edge('leave',rb,'output')])
    outer_id=s['expandedGraph']['compositionPlan']['containers'][1]['id']
    root=copy.deepcopy(s['expandedGraph']['compositionPlan']['containers'][0]);root['inputs'][0]['targets']=[dict(node=la,port='in')];root['outputs'][0]['source']=dict(node=rb,port='out')
    outer_row=dict(id=outer_id,kind='orderedList',path=['placed'],children=[dict(kind='container',id=rows['right']['id'],key='0'),dict(kind='container',id=rows['left']['id'],key='1')],inputs=[dict(id='in',targets=[dict(node=la,port='in')])],outputs=[dict(id='out',source=dict(node=rb,port='out'))])
    s['expandedGraph'].update(nodes=expected_nodes,edges=expected_edges,compositionPlan=dict(schemaVersion=1,root='root',containers=[root,outer_row,rows['right'],rows['left']]))
    s.update(nodeOrigins=origins,consumedDefinitions=[outer_ref,copy.deepcopy(REF)])
    assert reexpand(s['sourceCompositeDocument'],REGISTRY)=={k:s[k] for k in ('expandedGraph','nodeOrigins','consumedDefinitions')}
    assert check(s)
    changed=copy.deepcopy(s);changed['sourceCompositeDocument']['root']['instances'][0]['overrides']['leftBias']=True
    with pytest.raises(ValueError):check(changed)


def test_typed_integer_products_apply_after_overrides_and_malformed_products_refuse():
    s=fixture();d=s['sourceCompositeDocument']['definitions'][0]
    d['body']['nodes'][1]=node('b','linear',dict(in_features=2,out_features=2,bias=True))
    def field(key,default):return dict(key=key,title=key,kind='integer',defaultValue=default,min=1,max=512)
    d['parameters']=[dict(field=field('width',2),targets=[dict(componentID='a',field='in_features')]),dict(field=field('multiplier',2),targets=[dict(componentID='b',field='out_features')])]
    d['integerProducts']=[dict(lhs='width',rhs='multiplier',targets=[dict(componentID='a',field='out_features'),dict(componentID='b',field='in_features')])]
    s['sourceCompositeDocument']['root']['nodes'][0]['config']['features']=3
    s['sourceCompositeDocument']['root']['instances'][0]['overrides']={'width':3}
    s['expandedGraph']['nodes'][0]['config']['features']=3
    s['expandedGraph']['nodes'][2]['config']=dict(in_features=3,out_features=6,bias=True)
    s['expandedGraph']['nodes'][3].update(type='linear',config=dict(in_features=6,out_features=2,bias=True))
    assert reexpand(s['sourceCompositeDocument'],REGISTRY)=={k:s[k] for k in ('expandedGraph','nodeOrigins','consumedDefinitions')}
    assert check(s)
    for bad in [False,{},'',dict(lhs='width'),[dict(lhs='width',rhs='missing',targets=[])]]:
        changed=copy.deepcopy(s);changed['sourceCompositeDocument']['definitions'][0]['integerProducts']=bad
        with pytest.raises(ValueError):check(changed)


def test_root_data_fanout_keeps_first_wire_and_namespaces_extra_wire():
    s=fixture('graph');d=s['sourceCompositeDocument']['definitions'][0]
    d['body']['nodes']=[node('a','linear',dict(in_features=2,out_features=2,bias=True)),node('b','linear',dict(in_features=2,out_features=2,bias=True)),node('sum','add',dict(alpha=1,sample_broadcast=False))]
    d['body']['edges']=[edge('as','a','sum',target_port='a'),edge('bs','b','sum',target_port='b')]
    d['inputs'][0]['targets']=[dict(componentID='a',portID='in'),dict(componentID='b',portID='in')]
    d['outputs'][0]['source']=dict(componentID='sum',portID='out')
    d['composition']['children']=[dict(localID=i) for i in ('a','b','sum')]
    a,b,total=[swift_id('n',['placed'],i) for i in ('a','b','sum')]
    s['expandedGraph']['nodes']=[*s['expandedGraph']['nodes'][:2],node(a,'linear',dict(in_features=2,out_features=2,bias=True)),node(b,'linear',dict(in_features=2,out_features=2,bias=True)),node(total,'add',dict(alpha=1,sample_broadcast=False))]
    s['expandedGraph']['edges']=[edge(swift_id('e',['placed'],'as'),a,total,target_port='a'),edge(swift_id('e',['placed'],'bs'),b,total,target_port='b'),edge('enter','input',a),edge(swift_id('e',['$fanout','1'],'enter'),'input',b),edge('leave',total,'output')]
    root,inner=s['expandedGraph']['compositionPlan']['containers']
    targets=[dict(node=a,port='in'),dict(node=b,port='in')]
    root['inputs'][0]['targets']=copy.deepcopy(targets);inner['inputs'][0]['targets']=targets
    root['outputs'][0]['source']=dict(node=total,port='out');inner['outputs'][0]['source']=dict(node=total,port='out')
    inner['children']=[dict(kind='node',id=i,key=str(j)) for j,i in enumerate((a,b,total))]
    s['nodeOrigins']={n['id']:dict(instancePath=[],localNodeID=n['id']) for n in s['expandedGraph']['nodes'][:2]}
    s['nodeOrigins'].update({flat:dict(instancePath=['placed'],localNodeID=local,definition=copy.deepcopy(REF)) for local,flat in (('a',a),('b',b),('sum',total))})
    assert reexpand(s['sourceCompositeDocument'],REGISTRY)=={k:s[k] for k in ('expandedGraph','nodeOrigins','consumedDefinitions')}
    assert check(s)

@pytest.mark.parametrize('selected',['plan','nullPlan','legacyPolicy'])
def test_version1_cannot_downgrade_selected_composition(selected):
    s=fixture();s['nativeExpansionContractVersion']=1;s['sourceCompositeDocument']['schemaVersion']=2
    if selected=='nullPlan':s['expandedGraph']['compositionPlan']=None
    if selected=='legacyPolicy':del s['expandedGraph']['compositionPlan']
    with pytest.raises(ValueError):check(s)

def test_parameterless_source_is_valid_without_widening_default_training_admission():
    from runner.engine import validate
    s=fixture();s['sourceCompositeDocument']['definitions'][0]['body']['nodes'][0].update(type='relu',config={})
    s['expandedGraph']['nodes'][2].update(type='relu',config={})
    assert check(s)
    with pytest.raises(ValueError,match='trainable parameters'):validate(s['expandedGraph'],REGISTRY,bundled_dataset=False)


def test_actual_selected_objective_context_admits_nonbinary_output_and_wrong_context_refuses():
    from runner.loss_configuration import validate_recipe
    s=fixture();s['sourceCompositeDocument']['definitions'][0]['body']['nodes'][0]['config']['out_features']=3
    s['expandedGraph']['nodes'][2]['config']['out_features']=3
    context=dict(recipe=validate_recipe(dict(schemaVersion=1,configurationVersion=1,componentID='loss_mse',config={})),targetMapping=dict(schemaVersion=1,componentID='loss_mse',inputColumns=['x0','x1'],roles=[dict(role='target',columns=['t0','t1','t2'],sampleShape=[3],dtype='float32')],tripletSourceGroups=[]))
    raw=json.dumps(s).encode();sha=hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError,match='2 features'):checked_source(raw,sha,s['expandedGraph'],REGISTRY)
    assert checked_source(raw,sha,s['expandedGraph'],REGISTRY,objective_context=context)
    bad=copy.deepcopy(context);bad['targetMapping']['roles'][0]['sampleShape']=[2]
    with pytest.raises(ValueError):checked_source(raw,sha,s['expandedGraph'],REGISTRY,objective_context=bad)


@pytest.mark.parametrize('field,value',[
    (dict(kind='integer',defaultValue=2,min=1,max=4),3),
    (dict(kind='number',defaultValue=.25,min=0,max=1),.5),
    (dict(kind='boolean',defaultValue=True),False),
    (dict(kind='choice',defaultValue='a',choices=[dict(value='a',title='A'),dict(value='b',title='B')]),'b'),
    (dict(kind='integerList',defaultValue=[],min=-2,max=4,minItems=0,maxItems=4),[1,2.0]),
    (dict(kind='numberList',defaultValue=None,min=-2,max=4,minItems=0,maxItems=4,allowsNull=True),[.25,-1.5]),
    (dict(kind='text',defaultValue='é',minItems=1,maxItems=4),'αβ'),
])
def test_all_typed_exposed_field_kinds_and_no_boolean_number_coercion(field,value):
    from runner.module_composition_source import _field,_accepts
    declaration=dict(key='setting',title='Setting',**field)
    assert _field(declaration)==declaration and _accepts(declaration,value)
    bad=copy.deepcopy(declaration);bad['unexpected']='code'
    with pytest.raises(ValueError):_field(bad)
    if field['kind'] in ('integer','number'):
        assert not _accepts(declaration,True)
    if field['kind'] in ('integerList','numberList'):
        assert not _accepts(declaration,[True])


def test_captured_registry_field_drift_refuses_and_duplicate_source_keys_refuse(tmp_path):
    s=fixture();registry=json.loads(REGISTRY.read_text())
    linear=next(c for c in registry['components'] if c['id']=='linear')
    next(f for f in linear['fields'] if f['key']=='in_features')['min']=3
    captured=tmp_path/'registry.json';captured.write_text(json.dumps(registry))
    with pytest.raises(ValueError):check(s,registry=captured)
    raw=json.dumps(s).replace('"nativeExpansionContractVersion": 2','"nativeExpansionContractVersion": 2, "nativeExpansionContractVersion": 2').encode()
    with pytest.raises(ValueError,match='Duplicate'):checked_source(raw,hashlib.sha256(raw).hexdigest(),s['expandedGraph'],REGISTRY)


def unused_policy_fixture():
    s=fixture()
    root=s['sourceCompositeDocument']['root']
    root['instances']=[];root['nodes'].append(node('plain','linear',dict(in_features=2,out_features=2,bias=True)))
    root['edges']=[edge('enter','input','plain'),edge('leave','plain','output')]
    s['expandedGraph']=dict(schemaVersion=2,name='Example',nodes=copy.deepcopy(root['nodes']),edges=copy.deepcopy(root['edges']))
    s['nodeOrigins']={n['id']:dict(instancePath=[],localNodeID=n['id']) for n in root['nodes']}
    s['consumedDefinitions']=[]
    return s


def test_source3_without_reachable_policy_is_independently_verified_version2_without_plan():
    s=unused_policy_fixture()
    assert reexpand(s['sourceCompositeDocument'],REGISTRY)=={k:s[k] for k in ('expandedGraph','nodeOrigins','consumedDefinitions')}
    actual=check(s)
    assert actual['nativeExpansionContractVersion']==2 and actual['consumedDefinitions']==[]
    assert 'compositionPlan' not in s['expandedGraph']


@pytest.mark.parametrize('value',[None,fixture()['expandedGraph']['compositionPlan']])
def test_unreachable_policy_cannot_forge_explicit_null_or_native_plan(value):
    s=unused_policy_fixture();s['expandedGraph']['compositionPlan']=value
    with pytest.raises(ValueError):check(s)


def test_inline_training_control_is_validated_without_changing_container_architecture():
    s=fixture();source=s['sourceCompositeDocument']
    before=reexpand(source,REGISTRY)
    policy=dict(schemaVersion=1,modelSelection='bestValidation',monitor='validationLoss',earlyStopping=dict(enabled=True,patience=2,minimumImprovement=1))
    source['execution']=dict(trainingControl=policy)
    after=reexpand(source,REGISTRY)
    # Run policy is captured metadata, not an extra node or tensor operation.
    assert after['expandedGraph']['nodes']==before['expandedGraph']['nodes']
    assert after['expandedGraph']['compositionPlan']==before['expandedGraph']['compositionPlan']
    source['execution']['trainingControl']['earlyStopping']['patience']=0
    with pytest.raises(ValueError):reexpand(source,REGISTRY)
