"""Private-registry LazyLinear graph contract; no shared catalog exposure."""
import copy
import json
import pytest
import torch
import torch.nn.functional as F
from torch.nn.parameter import UninitializedParameter
from runner.engine import GraphModel, REGISTRY, validate, export_source


@pytest.fixture
def registry(tmp_path):
    value = json.loads(REGISTRY.read_text())
    value['components'] = [c for c in value['components'] if c['id'] != 'lazy_linear']
    value['components'].append(dict(id='lazy_linear', title='Lazy Linear', category='Layers',
        inputs=['in'], outputs=['out'], fields=[
            dict(key='out_features', title='Output features', kind='integer', defaultValue=2, min=1, max=512),
            dict(key='bias', title='Bias', kind='boolean', defaultValue=True)]))
    path = tmp_path / 'registry.json'
    path.write_text(json.dumps(value))
    return path


def graph(bias=True):
    return make_graph([
        ('input', 'input', {'features': 2}),
        ('lazy', 'lazy_linear', {'out_features': 2, 'bias': bias}),
        ('output', 'output', {})])


def make_graph(specifications):
    return dict(schemaVersion=2,
        nodes=[dict(id=i, type=t, config=c) for i,t,c in specifications],
        edges=[dict(id=f'{a[0]}-{b[0]}', source=a[0], sourcePort='out', target=b[0], targetPort='in')
               for a,b in zip(specifications, specifications[1:])])


@pytest.mark.parametrize('bias', [False, True])
def test_lazy_only_classifier_validates_and_defers_real_parameters(registry, bias):
    authored = graph(bias); original = copy.deepcopy(authored)
    order,widths = validate(authored, registry)
    assert order == ['input','lazy','output'] and widths['lazy'] == 2
    model = GraphModel(authored, registry)
    layer = model.layers[model.layer_keys['lazy']]
    assert isinstance(layer, torch.nn.LazyLinear)
    assert isinstance(layer.weight, UninitializedParameter)
    assert isinstance(layer.bias, UninitializedParameter) if bias else layer.bias is None
    assert authored == original and 'in_features' not in authored['nodes'][1]['config']


@pytest.mark.parametrize('config', [
    {'out_features':2}, {'out_features':2,'bias':1}, {'out_features':2,'bias':None},
    {'out_features':2,'bias':'true'}, {'out_features':True,'bias':True},
    {'out_features':2.5,'bias':True}, {'out_features':0,'bias':True},
    {'out_features':513,'bias':True}, {'out_features':2,'bias':True,'in_features':2},
])
def test_strict_schema2_configuration(registry, config):
    authored = graph(); authored['nodes'][1]['config'] = config
    with pytest.raises(ValueError): validate(authored, registry)


def test_schema1_does_not_gain_lazy_support(registry):
    authored = graph(); authored['schemaVersion'] = 1
    with pytest.raises(ValueError): validate(authored, registry)


@pytest.mark.parametrize('bias', [False, True])
def test_explicit_materialization_on_current_dtype_once_without_executing_graph(registry, bias, monkeypatch):
    authored = make_graph([
        ('input','input',{'features':2}),
        ('expand','linear',{'in_features':2,'out_features':6,'bias':True}),
        ('activation','relu',{}),
        ('shaped','reshape',{'shape':[2,3]}),
        ('lazy','lazy_linear',{'out_features':4,'bias':bias}),
        ('flat','flatten',{'start_dim':1,'end_dim':-1}),
        ('head','lazy_linear',{'out_features':2,'bias':True}),
        ('output','output',{})])
    original = copy.deepcopy(authored)
    model = GraphModel(authored, registry).to(dtype=torch.float64)
    def forbidden(*args, **kwargs): raise AssertionError('materialization executed authored graph')
    with monkeypatch.context() as guard:
        guard.setattr(model, 'forward', forbidden)
        guard.setattr(model.layers[model.layer_keys['expand']], 'forward', forbidden)
        guard.setattr(torch, 'relu', forbidden)
        model.materialize_lazy_parameters()
    layer = model.layers[model.layer_keys['lazy']]
    head = model.layers[model.layer_keys['head']]
    assert layer.in_features == 3 and tuple(layer.weight.shape) == (4,3)
    assert head.in_features == 8 and tuple(head.weight.shape) == (2,8)
    assert layer.weight.dtype == torch.float64 and layer.weight.device.type == 'cpu'
    assert (layer.bias is not None) == bias
    saved = {k:v.detach().clone() for k,v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    model.materialize_lazy_parameters()
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(value, model.state_dict()[key]) for key,value in saved.items())
    assert authored == original
    x = torch.randn(5,2,dtype=torch.float64,requires_grad=True)
    output,captured = model(x,inspect=True)
    assert captured['lazy'].shape == (5,2,4)
    reference = F.linear(captured['shaped'],layer.weight,layer.bias)
    torch.testing.assert_close(captured['lazy'],reference)
    expected = F.linear(reference.flatten(1),head.weight,head.bias)
    torch.testing.assert_close(output,expected)
    measured = torch.autograd.grad(output.square().sum(),(x,*model.parameters()),retain_graph=True)
    independent = torch.autograd.grad(expected.square().sum(),(x,*model.parameters()))
    for a,b in zip(measured,independent): torch.testing.assert_close(a,b)


def test_int64_operand_refused_before_materialization(registry):
    authored = graph(); authored['nodes'][0] = dict(id='input',type='index_input',config={'shape':[2]})
    with pytest.raises(ValueError, match='(?i)(float|dtype|integer|int64)'):
        validate(authored, registry)


def test_aggregate_inferred_parameter_cap_precedes_allocation(registry, monkeypatch):
    specifications = [('input','input',{'features':2})]
    specifications += [(f'lazy{i}','lazy_linear',{'out_features':512,'bias':True}) for i in range(9)]
    specifications += [('head','lazy_linear',{'out_features':2,'bias':True}),('output','output',{})]
    def forbidden(*args, **kwargs): raise AssertionError('allocated before parameter validation')
    monkeypatch.setattr(torch.nn,'LazyLinear',forbidden)
    with pytest.raises(ValueError, match='(?i)(2 million|parameter)'):
        GraphModel(make_graph(specifications), registry)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='Apple GPU unavailable')
def test_materializes_on_selected_mps_device(registry):
    model = GraphModel(graph(False), registry).to('mps')
    model.materialize_lazy_parameters()
    layer = model.layers[model.layer_keys['lazy']]
    assert layer.weight.device.type == 'mps' and layer.weight.dtype == torch.float32
    x = torch.randn(3,2,device='mps',requires_grad=True)
    measured = model(x); expected = F.linear(x,layer.weight,None)
    torch.testing.assert_close(measured,expected)
    actual = torch.autograd.grad(measured.sum(),(x,layer.weight),retain_graph=True)
    reference = torch.autograd.grad(expected.sum(),(x,layer.weight))
    for a,b in zip(actual,reference): torch.testing.assert_close(a,b)


@pytest.mark.parametrize('materialize_before_export', [False, True])
def test_safe_module_key_and_standalone_export_preserve_lazy_authorship(registry, materialize_before_export):
    authored = graph(); authored['nodes'][1]['id'] = 'forward'
    for edge in authored['edges']:
        for key in ('source','target'):
            if edge[key] == 'lazy': edge[key] = 'forward'
    original = copy.deepcopy(authored)
    model = GraphModel(authored,registry).double()
    assert model.layer_keys['forward'] != 'forward'
    if materialize_before_export: model.materialize_lazy_parameters()
    source = export_source(model)
    assert 'nn.LazyLinear(' in source and 'runner' not in source
    assert authored == original
    namespace = {}; exec(source,namespace)
    portable = namespace['Model']().double()
    assert any(isinstance(layer,torch.nn.LazyLinear) for layer in portable.modules())
    model.materialize_lazy_parameters()
    portable.load_state_dict(model.state_dict(),strict=True)
    x = torch.tensor([[1.,2.],[-2.,3.]],dtype=torch.float64)
    rng = torch.get_rng_state().clone()
    measured = portable(x)
    assert torch.equal(torch.get_rng_state(),rng), 'Loading initialized lazy state must not reinitialize parameters'
    torch.testing.assert_close(measured,model(x))
    assert all(torch.equal(value,portable.state_dict()[key]) for key,value in model.state_dict().items())

@pytest.mark.parametrize('portable', [False, True])
def test_uninitialized_forward_refuses_width_outside_authored_graph(registry, portable):
    model = GraphModel(graph(),registry)
    if portable:
        namespace = {}; exec(export_source(model),namespace)
        model = namespace['Model']()
    layer = next(layer for layer in model.modules() if isinstance(layer,torch.nn.LazyLinear))
    with pytest.raises(ValueError,match='(?i)(width|features|shape)'):
        model(torch.ones(1,3))
    assert isinstance(layer.weight,UninitializedParameter)
