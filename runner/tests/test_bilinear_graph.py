"""Bilinear runtime integration; private registry until native delivery is accepted."""
import json
import pytest
import torch
from runner.engine import GraphModel, REGISTRY, export_source


@pytest.fixture
def registry(tmp_path):
    value = json.loads(REGISTRY.read_text())
    value['components'] = [c for c in value['components'] if c['id'] != 'bilinear']
    value['components'].append(dict(id='bilinear', title='Bilinear', category='Layers',
                                   inputs=['a', 'b'], outputs=['out'], fields=[]))
    path = tmp_path / 'registry.json'; path.write_text(json.dumps(value))
    return path


def graph(bias=True, shared=False):
    specs = [('input', 'input', {'features': 2})]
    links = []
    if not shared:
        specs += [('left', 'linear', {'in_features': 2, 'out_features': 3, 'bias': True}),
                  ('right', 'linear', {'in_features': 2, 'out_features': 5, 'bias': True})]
        links += [('input', 'left', 'in'), ('input', 'right', 'in')]
    specs += [('interaction', 'bilinear', {'in1_features': 2 if shared else 3,
                'in2_features': 2 if shared else 5, 'out_features': 2, 'bias': bias}),
              ('output', 'output', {})]
    links += [('input' if shared else 'left', 'interaction', 'a'),
              ('input' if shared else 'right', 'interaction', 'b'), ('interaction', 'output', 'in')]
    return dict(schemaVersion=2, nodes=[dict(id=i, type=t, config=c) for i,t,c in specs],
                edges=[dict(id=str(i), source=a, target=b, sourcePort='out', targetPort=p)
                       for i,(a,b,p) in enumerate(links)])


@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize('shared', [True, False])
def test_values_gradients_and_portable_state_match_explicit_contraction(registry, bias, shared):
    torch.manual_seed(736)
    model = GraphModel(graph(bias, shared), registry).double()
    x = torch.randn(2, 4, dtype=torch.float64).t().requires_grad_()
    assert not x.is_contiguous()
    actual, values = model(x, inspect=True)
    left = x if shared else model.layers['left'](x)
    right = x if shared else model.layers['right'](x)
    layer = model.layers['interaction']
    expected = torch.stack([torch.stack([
        sum(left[n,i] * layer.weight[o,i,j] * right[n,j]
            for i in range(left.shape[-1]) for j in range(right.shape[-1]))
        + (layer.bias[o] if bias else 0) for o in range(2)]) for n in range(len(x))])
    torch.testing.assert_close(actual, expected)
    native = torch.autograd.grad(actual.square().sum(), (x,*model.parameters()), retain_graph=True)
    reference = torch.autograd.grad(expected.square().sum(), (x,*model.parameters()))
    for a,b in zip(native,reference): torch.testing.assert_close(a,b)
    source = export_source(model); assert 'runner' not in source
    namespace = {}; exec(source, namespace)
    exported = namespace['Model']().double(); exported.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(exported(x), expected)
    portable = torch.autograd.grad(exported(x).square().sum(), (x,*exported.parameters()))
    for a,b in zip(portable,reference): torch.testing.assert_close(a,b)
    isolated = torch.autograd.grad(model(x)[0].sum(), x)[0]
    assert torch.count_nonzero(isolated[1:]) == 0


def test_missing_and_duplicate_ports_are_rejected(registry):
    value = graph(shared=True); value['edges'].pop(0)
    with pytest.raises(ValueError, match='required input'): GraphModel(value, registry)
    value = graph(); value['edges'][3]['targetPort'] = 'a'
    with pytest.raises(ValueError, match='already connected'): GraphModel(value, registry)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='Apple GPU unavailable')
def test_mps_forward_backward_match_cpu(registry):
    torch.manual_seed(489)
    cpu = GraphModel(graph(), registry)
    gpu = GraphModel(graph(), registry).to('mps'); gpu.load_state_dict(cpu.state_dict())
    x = torch.randn(6,2,requires_grad=True); y = x.detach().to('mps').requires_grad_()
    expected = cpu(x); actual = gpu(y)
    torch.testing.assert_close(actual.cpu(),expected,rtol=1e-4,atol=1e-5)
    a = torch.autograd.grad(expected.square().sum(),(x,*cpu.parameters()))
    b = torch.autograd.grad(actual.square().sum(),(y,*gpu.parameters()))
    for first,second in zip(a,b): torch.testing.assert_close(first,second.cpu(),rtol=1e-4,atol=1e-5)


def test_higher_rank_sample_prefix_and_export(registry):
    value = graph()
    value['nodes'][1]['config']['out_features'] = 6
    value['nodes'][2]['config']['out_features'] = 10
    value['nodes'] += [dict(id='a',type='reshape',config={'shape':[2,3]}),
                       dict(id='b',type='reshape',config={'shape':[2,5]}),
                       dict(id='flat',type='flatten',config={'start_dim':1,'end_dim':-1}),
                       dict(id='head',type='linear',config={'in_features':4,'out_features':2,'bias':True})]
    value['edges'][2]['source'] = 'a'; value['edges'][3]['source'] = 'b'
    value['edges'][4]['source'] = 'head'
    for source,target in [('left','a'),('right','b'),('interaction','flat'),('flat','head')]:
        value['edges'].append(dict(id=f'{source}-{target}',source=source,target=target,sourcePort='out',targetPort='in'))
    model = GraphModel(value,registry).double()
    x = torch.randn(4,2,dtype=torch.float64,requires_grad=True)
    actual,values = model(x,inspect=True)
    layer = model.layers['interaction']
    expected = torch.einsum('nki,oij,nkj->nko',values['a'],layer.weight,values['b']) + layer.bias
    torch.testing.assert_close(values['interaction'],expected)
    namespace = {}; exec(export_source(model),namespace)
    exported = namespace['Model']().double(); exported.load_state_dict(model.state_dict())
    torch.testing.assert_close(exported(x),actual)
    assert tuple(values['interaction'].shape) == (4,2,2)


def test_safe_module_key_preserves_exported_state(registry):
    value = graph(shared=True)
    value['nodes'][1]['id'] = 'forward'
    for edge in value['edges']:
        for key in ('source','target'):
            if edge[key] == 'interaction': edge[key] = 'forward'
    model = GraphModel(value,registry)
    assert model.layer_keys['forward'] != 'forward'
    namespace = {}; exec(export_source(model),namespace)
    exported = namespace['Model'](); exported.load_state_dict(model.state_dict())
    x = torch.randn(3,2)
    torch.testing.assert_close(exported(x),model(x))


@pytest.mark.parametrize('freeze', [True, False])
def test_trainability_distinguishes_frozen_parameter_state(registry, freeze):
    value = dict(schemaVersion=2, nodes=[
        dict(id='input',type='index_input',config={'shape':[1]}),
        dict(id='lookup',type='embedding',config=dict(num_embeddings=4,embedding_dim=2,padding_idx=None,
             max_norm=None,norm_type=2,scale_grad_by_freq=False,sparse=False,freeze=freeze)),
        dict(id='flat',type='flatten',config={'start_dim':1,'end_dim':-1}),
        dict(id='output',type='output',config={})], edges=[])
    for a,b in [('input','lookup'),('lookup','flat'),('flat','output')]:
        value['edges'].append(dict(id=a,source=a,target=b,sourcePort='out',targetPort='in'))
    if freeze:
        with pytest.raises(ValueError,match='trainable'): GraphModel(value,registry)
    else:
        model = GraphModel(value,registry)
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 8
