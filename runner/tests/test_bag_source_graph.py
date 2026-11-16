"""Injected reviewed-shape registry only; no publication/engine training claim."""
import copy
import json
from dataclasses import replace
from unittest.mock import patch
import pytest
import torch
from runner import engine
from runner.bag_source import BagCoordinateLayout,normalize_bag_source,source_layouts,execute_bag_source
from runner.embedding_bag_configuration import DEFAULTS
from runner.embedding_bag_packet import EmbeddingBagPacket

SOURCE=dict(layout='packed',maximum_items_per_bag=3,per_sample_weights=True,include_last_offset=True)
DEVICES=['cpu']+(['mps'] if torch.backends.mps.is_available() else [])

@pytest.fixture
def registry(tmp_path):
    value=json.loads(engine.REGISTRY.read_text())
    value['components']=[entry for entry in value['components'] if entry['id'] not in ('bag_input','embedding_bag')]
    value['components'] += [dict(id='bag_input',inputs=[],outputs=['indices','offsets','per_sample_weights'],fields=[]),dict(id='embedding_bag',inputs=['indices','offsets','per_sample_weights'],outputs=['out'],fields=[])]
    path=tmp_path/'registry.json';path.write_text(json.dumps(value));return path


def graph(layout='packed',weighted=True,terminal=True,freeze=False):
    nodes=[dict(id='source',type='bag_input',config={**SOURCE,'layout':layout,'per_sample_weights':weighted,'include_last_offset':terminal}),dict(id='bag',type='embedding_bag',config={**DEFAULTS,'num_embeddings':5,'embedding_dim':2,'mode':'sum','padding_idx':0,'layout':layout,'per_sample_weights':weighted,'include_last_offset':terminal,'freeze':freeze}),dict(id='result',type='output',config={})]
    ports=['indices']+(['offsets'] if layout=='packed' else [])+(['per_sample_weights'] if weighted else [])
    edges=[dict(id=p,source='source',sourcePort=p,target='bag',targetPort=p) for p in ports]
    edges.append(dict(id='out',source='bag',sourcePort='out',target='result',targetPort='in'))
    return dict(schemaVersion=2,nodes=nodes,edges=edges)


def packet(device,layout='packed',weighted=True,terminal=True):
    ids=[1,2,1,0,3,4,0] if layout=='packed' else [[1,2,1],[0,0,0],[0,3,0],[4,0,0]]
    weights=[.5,2.,-1.,99.,3.,.25,88.] if layout=='packed' else [[.5,2.,-1.],[0.,0.,0.],[99.,3.,0.],[.25,88.,0.]]
    return EmbeddingBagPacket(layout,torch.tensor(ids,device=device),torch.tensor([0,3,3,5,7] if terminal else [0,3,3,5],device=device) if layout=='packed' else None,torch.tensor(weights,device=device,requires_grad=True) if weighted else None,terminal)

@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('layout',['fixed','packed'])
@pytest.mark.parametrize('weighted',[False,True])
def test_real_graph_named_sources_and_independent_output_gradients(registry,device,layout,weighted):
    value=graph(layout,weighted);model=engine.GraphModel(value,registry,bundled_dataset=False).to(device)
    with torch.no_grad():model.layers['bag'].weight.copy_(torch.tensor([[9.,9.],[1.,2.],[3.,-1.],[-2.,4.],[3.,4.]],device=device))
    source=packet(device,layout,weighted);out,observed=model(source,inspect=True)
    expected=[[5.5,-3.],[0.,0.],[-6.,12.],[.75,1.]] if weighted else [[5.,3.],[0.,0.],[-2.,4.],[3.,4.]]
    torch.testing.assert_close(out,torch.tensor(expected,device=device))
    assert observed['source']['indices'] is source.indices
    if weighted:assert observed['source']['per_sample_weights'] is source.per_sample_weights
    assert list(model.state_dict())==['layers.bag.weight']
    (out*torch.tensor([[2.,-1.],[7.,11.],[-3.,5.],[4.,6.]],device=device)).sum().backward()
    expected_table=[[0.,0.],[-1.,.5],[4.,-2.],[-9.,15.],[1.,1.5]] if weighted else [[0.,0.],[4.,-2.],[2.,-1.],[-3.,5.],[4.,6.]]
    torch.testing.assert_close(model.layers['bag'].weight.grad,torch.tensor(expected_table,device=device))
    if weighted:
        expected_weights=[0.,7.,0.,0.,26.,36.,0.] if layout=='packed' else [[0.,7.,0.],[0.,0.,0.],[0.,26.,0.],[36.,0.,0.]]
        torch.testing.assert_close(source.per_sample_weights.grad,torch.tensor(expected_weights,device=device))


def test_coordinate_domain_and_source_identity(registry):
    _,_,layouts,dtypes=engine.validate(graph(),registry,include_types=True,bundled_dataset=False)
    assert isinstance(layouts['source']['indices'],BagCoordinateLayout)
    assert not hasattr(layouts['source']['indices'],'sample')
    assert layouts['source']['indices'].source_id=='source'
    assert list(layouts['source'])==['indices','offsets','per_sample_weights']
    assert dtypes['source']==dict(indices='int64',offsets='int64',per_sample_weights='float32')
    assert layouts['bag'].sample==(2,)

@pytest.mark.parametrize('changes',[{'layout':'wrong'},{'maximum_items_per_bag':0},{'maximum_items_per_bag':129},{'maximum_items_per_bag':True},{'maximum_items_per_bag':2.},{'per_sample_weights':1},{'include_last_offset':None},{'unknown':1}])
def test_strict_source_controls(changes):
    with pytest.raises(ValueError):normalize_bag_source({**SOURCE,**changes})

@pytest.mark.parametrize('key',list(SOURCE))
def test_source_fields_required(key):
    config=dict(SOURCE);del config[key]
    with pytest.raises(ValueError):normalize_bag_source(config)


def test_source_forward_enforces_per_bag_bound_and_identity():
    source=packet('cpu');values=execute_bag_source(source,SOURCE)
    assert values['per_sample_weights'] is source.per_sample_weights
    with pytest.raises(ValueError):execute_bag_source(source,{**SOURCE,'maximum_items_per_bag':2})
    with pytest.raises(ValueError):execute_bag_source(source,{**SOURCE,'layout':'fixed'})
    with pytest.raises(ValueError):execute_bag_source(source,{**SOURCE,'include_last_offset':False})
    with pytest.raises(ValueError):execute_bag_source(source,{**SOURCE,'per_sample_weights':False})

@pytest.mark.parametrize('bad',['role','flags','ordinary','two_sources','extra_input'])
def test_invalid_coordinate_connections_and_source_count_refuse(registry,bad):
    value=graph()
    if bad=='role':value['edges'][0]['sourcePort']='offsets'
    elif bad=='flags':value['nodes'][1]['config']['include_last_offset']=False
    elif bad=='ordinary':value['nodes'][1]=dict(id='bag',type='relu',config={});value['edges']= [dict(id='bad',source='source',sourcePort='indices',target='bag',targetPort='in'),value['edges'][-1]]
    elif bad=='two_sources':value['nodes'].insert(0,dict(id='other',type='bag_input',config=SOURCE));value['edges'][1]['source']='other'
    else:value['edges'].append(dict(id='extra',source='source',sourcePort='indices',target='bag',targetPort='extra'))
    with pytest.raises(ValueError):engine.validate(value,registry,bundled_dataset=False)


def test_frozen_table_count_requires_other_trainable_layer(registry):
    with pytest.raises(ValueError,match='trainable'):engine.validate(graph(freeze=True),registry,bundled_dataset=False)
    engine.validate(graph(freeze=True),registry,bundled_dataset=False,training_required=False)


def test_parameters_checked_before_native_allocation(registry):
    value=graph();value['nodes'][1]['config'].update(num_embeddings=1_000_001,embedding_dim=2)
    with patch('torch.nn.EmbeddingBag.__init__',side_effect=AssertionError('allocation forbidden')):
        with pytest.raises(ValueError,match='million'):engine.GraphModel(value,registry,bundled_dataset=False)


def test_bag_composition_refuses_and_completed_export_executes(registry):
    value=graph();value['compositionPlan']={}
    with pytest.raises(ValueError,match=r'^Composition plan has missing or unknown fields\.$'):
        engine.GraphModel(value,registry,bundled_dataset=False)
    model=engine.GraphModel(graph(),registry,bundled_dataset=False)
    namespace={};exec(compile(engine.export_source(model),'bag-model.py','exec'),namespace)
    exported=namespace['Model']();exported.load_state_dict(model.state_dict(),strict=True)
    item=packet('cpu')
    torch.testing.assert_close(exported(item.indices,item.offsets,item.per_sample_weights),model(item))


def test_new_bag_types_require_graph2_before_parameter_allocation(registry):
    value=graph();value['schemaVersion']=1
    with patch('torch.nn.EmbeddingBag.__init__',side_effect=AssertionError('allocation forbidden')):
        with pytest.raises(ValueError,match='schema2'):
            engine.GraphModel(value,registry,bundled_dataset=False)
    engine.GraphModel(graph(),registry,bundled_dataset=False)


@pytest.mark.parametrize('component,port_field,ports',[
    ('bag_input','inputs',['in']),('bag_input','outputs',['indices']),
    ('bag_input','outputs',['indices','offsets','per_sample_weights','unchecked']),
    ('embedding_bag','inputs',['indices']),('embedding_bag','outputs',['out','unchecked']),
])
def test_malformed_registry_cannot_bypass_bag_role_contract(registry,component,port_field,ports):
    value=json.loads(registry.read_text())
    definition=next(item for item in value['components'] if item['id']==component)
    definition[port_field]=ports;registry.write_text(json.dumps(value))
    with patch('torch.nn.EmbeddingBag.__init__',side_effect=AssertionError('allocation forbidden')):
        with pytest.raises(ValueError,match='registry|declared'):
            engine.GraphModel(graph(),registry,bundled_dataset=False)


@pytest.mark.parametrize('bad',['length','weights_absent','terminal','layout','index','schema'])
def test_source_packet_refuses_before_native_forward(registry,bad):
    value=graph()
    if bad=='length':value['nodes'][0]['config']['maximum_items_per_bag']=2
    model=engine.GraphModel(value,registry,bundled_dataset=False)
    source=packet('cpu')
    if bad=='weights_absent':source=replace(source,per_sample_weights=None)
    elif bad=='terminal':source=replace(source,include_last_offset=False)
    elif bad=='layout':source=replace(source,layout='fixed')
    elif bad=='index':source.indices[0]=5
    elif bad=='schema':source=replace(source,schema_version=2)
    before=model.layers['bag'].weight.detach().clone()
    with patch('torch.nn.EmbeddingBag.forward',side_effect=AssertionError('native execution forbidden')):
        with pytest.raises(ValueError):model(source)
    assert torch.equal(before,model.layers['bag'].weight)
    assert model.layers['bag'].weight.grad is None
