"""Canonical bag graph identity shared by checkpoint/source consumers."""
import copy
import pytest
from runner import engine
from runner.graph_semantics import effective_config,semantic_graph
from runner.embedding_bag_configuration import DEFAULTS


def graph():
    return {'schemaVersion':2,'nodes':[
        {'id':'source','type':'bag_input','config':{'layout':'packed','maximum_items_per_bag':4,'per_sample_weights':False,'include_last_offset':True}},
        {'id':'bag','type':'embedding_bag','config':{**DEFAULTS,'num_embeddings':5,'embedding_dim':2,'layout':'packed','include_last_offset':True}},
        {'id':'result','type':'output','config':{}}],
        'edges':[{'id':'ids','source':'source','sourcePort':'indices','target':'bag','targetPort':'indices'},
                 {'id':'offsets','source':'source','sourcePort':'offsets','target':'bag','targetPort':'offsets'},
                 {'id':'out','source':'bag','sourcePort':'out','target':'result','targetPort':'in'}]}


def test_shared_effective_config_is_engine_authority():
    value=graph()
    for node in value['nodes']:
        assert effective_config(node,2)==engine.effective_config(node,2)
    assert engine.effective_config is effective_config


def test_bag_semantic_graph_canonicalizes_dimensions_and_padding():
    value=graph();value['nodes'][1]['config'].update(num_embeddings=5.,embedding_dim=2.,padding_idx=-1)
    canonical=semantic_graph(value)
    bag=next(n for n in canonical['nodes'] if n['id']=='bag')['config']
    assert type(bag['num_embeddings']) is int and bag['num_embeddings']==5
    assert type(bag['embedding_dim']) is int and bag['embedding_dim']==2
    assert bag['padding_idx']==4
    changed=copy.deepcopy(value);changed['nodes'][1]['config']['padding_idx']=4
    assert semantic_graph(changed)==canonical


@pytest.mark.parametrize('change',['layout','weights','terminal','capacity','mode','norm','padding','freeze','sparse','width'])
def test_real_bag_controls_change_semantic_identity(change):
    original=graph();changed=copy.deepcopy(original)
    source=changed['nodes'][0]['config'];bag=changed['nodes'][1]['config']
    if change=='layout':
        source['layout']=bag['layout']='fixed'
        changed['edges']=[e for e in changed['edges'] if e['sourcePort']!='offsets']
    elif change=='weights':
        source['per_sample_weights']=bag['per_sample_weights']=True;bag['mode']='sum'
        changed['edges'].append({'id':'weights','source':'source','sourcePort':'per_sample_weights','target':'bag','targetPort':'per_sample_weights'})
    elif change=='terminal':source['include_last_offset']=bag['include_last_offset']=False
    elif change=='capacity':source['maximum_items_per_bag']=3
    elif change=='mode':bag['mode']='sum'
    elif change=='norm':bag['max_norm']=2.;bag['norm_type']=1.
    elif change=='padding':bag['padding_idx']=0
    elif change=='freeze':bag['freeze']=True
    elif change=='sparse':bag['sparse']=True
    else:bag['embedding_dim']=3
    assert semantic_graph(changed)!=semantic_graph(original)


def test_cosmetic_names_positions_edge_ids_and_order_do_not_change_identity():
    original=graph();changed=copy.deepcopy(original)
    for node in changed['nodes']:node.update(name='User label',x=17,y=-3)
    for edge in changed['edges']:edge['id']='cosmetic-'+edge['id']
    changed['nodes'].reverse();changed['edges'].reverse()
    assert semantic_graph(changed)==semantic_graph(original)


@pytest.mark.parametrize('kind',['bag_input','embedding_bag'])
def test_shared_semantics_refuses_graph1_bag_types(kind):
    node=next(n for n in graph()['nodes'] if n['type']==kind)
    with pytest.raises(ValueError,match='schema2'):effective_config(node,1)
    value=graph();value['schemaVersion']=1
    with pytest.raises(ValueError,match='schema2'):semantic_graph(value)


@pytest.mark.parametrize('kind,change',[
    ('bag_input',{'maximum_items_per_bag':129}),('bag_input',{'per_sample_weights':1}),
    ('bag_input',{'include_last_offset':None}),('bag_input',{'foreign':1}),
    ('embedding_bag',{'num_embeddings':True}),('embedding_bag',{'mode':'other'}),
    ('embedding_bag',{'mode':'max','sparse':True}),('embedding_bag',{'per_sample_weights':True}),
])
def test_shared_semantics_does_not_bypass_frozen_config_validation(kind,change):
    value=graph();node=next(n for n in value['nodes'] if n['type']==kind);node['config'].update(change)
    with pytest.raises(ValueError):effective_config(node,2)
    with pytest.raises(ValueError):semantic_graph(value)


def test_legacy_graph1_and_graph2_semantic_identity_stays_exact():
    legacy={'schemaVersion':1,'nodes':[{'id':'input','type':'input','config':{'features':2}},{'id':'linear','type':'linear','config':{'in_features':2,'out_features':2}},{'id':'out','type':'output','config':{}}],
            'edges':[{'id':'a','source':'input','sourcePort':'out','target':'linear','targetPort':'in'},{'id':'b','source':'linear','sourcePort':'out','target':'out','targetPort':'in'}]}
    expected={'schemaVersion':1,'nodes':legacy['nodes'],'edges':[{k:v for k,v in edge.items() if k!='id'} for edge in legacy['edges']]}
    assert semantic_graph(legacy)==expected
    modern=copy.deepcopy(legacy);modern['schemaVersion']=2;modern['nodes'][1]['config']['bias']=True
    assert semantic_graph(modern)==expected
