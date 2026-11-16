import copy
import json
import pytest
import torch
from runner.engine import GraphModel,REGISTRY,export_source
from runner.tests.test_embedding_contract import config

@pytest.fixture
def registry(tmp_path):
    value=json.loads(REGISTRY.read_text())
    path=tmp_path/'registry.json';path.write_text(json.dumps(value));return path

def graph():
    nodes=[dict(id='source',type='index_input',config={'shape':[2]}),
           dict(id='lookup',type='embedding',config=config()),
           dict(id='flat',type='flatten',config={'start_dim':1,'end_dim':-1}),
           dict(id='head',type='linear',config={'in_features':6,'out_features':2,'bias':True}),
           dict(id='result',type='output',config={})]
    edges=[dict(id=str(i),source=a['id'],sourcePort='out',target=b['id'],targetPort='in') for i,(a,b) in enumerate(zip(nodes,nodes[1:]))]
    return dict(schemaVersion=2,nodes=nodes,edges=edges)

@pytest.mark.parametrize('device',['cpu','mps'])
def test_graph_lookup_export_state_and_gradient_parity(registry,device):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
    model=GraphModel(graph(),registry).to(device)
    x=torch.tensor([[1,2],[2,4]],device=device)
    actual=model(x)
    expected=model.layers['head'](model.layers['lookup'].weight[x].flatten(1))
    torch.testing.assert_close(actual,expected)
    scope={};source=export_source(model);assert 'runner' not in source
    exec(source,scope);exported=scope['Model']().to(device)
    exported.load_state_dict(model.state_dict(),strict=True)
    torch.testing.assert_close(exported(x),actual)
    actual.sum().backward();exported(x).sum().backward()
    for a,b in zip(model.parameters(),exported.parameters()):torch.testing.assert_close(a.grad,b.grad)
    for invalid in [x.float(),torch.tensor([[1,5]],device=device),torch.tensor([[1,2,3]],device=device)]:
        with pytest.raises(ValueError):model(invalid)
        with pytest.raises(ValueError):exported(invalid)


def test_integer_features_cannot_flow_into_linear(registry):
    value=graph();value['nodes'][1]['type']='relu';value['nodes'][1]['config']={}
    with pytest.raises(ValueError,match='float'):GraphModel(value,registry)


def test_registry_optional_numeric_fields_match_embedding_semantics(registry):
    value=json.loads(registry.read_text())
    definition=next(item for item in value['components'] if item['id']=='embedding')
    definition['fields']=[dict(key='padding_idx',title='Padding row',kind='integer',defaultValue=None,min=-100_000_000,max=99_999_999,allowsNull=True),
                          dict(key='max_norm',title='Maximum norm',kind='number',defaultValue=None,min=1e-38,max=1e38,allowsNull=True)]
    registry.write_text(json.dumps(value))
    GraphModel(graph(),registry)
    configured=graph();configured['nodes'][1]['config'].update(padding_idx=0,max_norm=1.0)
    GraphModel(configured,registry)
    definition['fields'][0]['allowsNull']=False
    registry.write_text(json.dumps(value))
    with pytest.raises(ValueError):GraphModel(graph(),registry)
