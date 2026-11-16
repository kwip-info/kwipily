"""Bundled transform graph integration and standalone parity."""
import json
import pytest
import torch
from runner.engine import GraphModel, REGISTRY, export_source


@pytest.fixture
def transform_registry():
    return REGISTRY


def graph():
    specs=[('input','input',{'features':2}),('project','linear',{'in_features':2,'out_features':12,'bias':True}),
           ('shape','reshape',{'shape':[3.0,4.0]}),('swap','transpose',{'dim0':1,'dim1':-1}),
           ('flat','flatten',{'start_dim':1,'end_dim':-1}),
           ('head','linear',{'in_features':12,'out_features':2,'bias':True}),('output','output',{})]
    return {'schemaVersion':2,'nodes':[{'id':i,'type':t,'config':c} for i,t,c in specs],
            'edges':[{'id':f'e{k}','source':a[0],'sourcePort':'out','target':b[0],'targetPort':'in'}
                     for k,(a,b) in enumerate(zip(specs,specs[1:]))]}


def test_transform_graph_forward_gradients_and_standalone_export(transform_registry):
    model=GraphModel(graph(),transform_registry)
    x=torch.randn(5,2,requires_grad=True)
    actual,values=model(x,inspect=True)
    expected=model.layers['head'](model.layers['project'](x).reshape(5,3,4).transpose(1,2).flatten(1))
    assert values['swap'].shape==(5,4,3)
    assert 'swap' not in model.widths
    assert torch.equal(actual,expected)
    reference=torch.autograd.grad(expected.square().sum(),(x,*model.parameters()),retain_graph=True)
    observed=torch.autograd.grad(actual.square().sum(),(x,*model.parameters()))
    assert all(torch.equal(a,b) for a,b in zip(reference,observed))
    source=export_source(model)
    assert 'runner' not in source
    namespace={};exec(source,namespace)
    exported=namespace['Model']();exported.load_state_dict(model.state_dict())
    assert torch.equal(exported(x),actual)
    export_grad=torch.autograd.grad(exported(x).square().sum(),(x,*exported.parameters()))
    assert all(torch.equal(a,b) for a,b in zip(reference,export_grad))


@pytest.mark.parametrize('node,config',[(2,{'shape':[5,-1]}),(3,{'dim0':0,'dim1':1}),(4,{'start_dim':2,'end_dim':1})])
def test_shape_errors_rejected_before_execution(transform_registry,node,config):
    g=graph();g['nodes'][node]['config']=config
    with pytest.raises(ValueError):GraphModel(g,transform_registry)


def test_bundled_registry_transform_forward_backward_and_export():
    test_transform_graph_forward_gradients_and_standalone_export(REGISTRY)


def test_discovery_reports_reviewed_transform_contract():
    from runner.engine import capabilities
    assert capabilities()['componentCapabilities']['tensorLayout'] == {
        'version': {'major': 1, 'minor': 0, 'patch': 0},
        'adapterID': 'kwip.runner.graph-v2.tensor-layout',
        'graphSchemaVersions': [2],
        'operations': ['reshape', 'transpose', 'flatten'],
        'maximumRank': 8,
        'preservesBatch': True,
    }


def test_product_training_capture_and_cached_preview_support_full_sample_shapes(tmp_path):
    from runner.engine import run
    from runner.preview import preview
    from runner.product_execution import ProductExecutionAdapter
    model = GraphModel(graph())
    _, values = ProductExecutionAdapter(model).forward(torch.randn(3, 2), inspect=True)
    assert list(values['shape'].shape) == [3, 3, 4]
    events = []
    run(graph(), tmp_path/'run', epochs=1, device='cpu', sample_limit=40, emit=events.append)
    assert events[-1]['event'] == 'completed'
    assert (tmp_path/'run/manifest.json').is_file()
    first = preview(graph(), tmp_path/'first', device='cpu', stop_at='swap')
    second = preview(graph(), tmp_path/'second', device='cpu', resume_from='flat', cache=tmp_path/'first/activation-cache.pt')
    assert 'swap' in second['reusedNodeIDs']
    assert second['previews']['swap']['shape'][1:] == [4, 3]


@pytest.mark.parametrize('value', [torch.ones(3, 4, 3), torch.ones(3, 12), torch.ones(3, 3, 4, dtype=torch.float64), torch.full((3, 3, 4), float('nan'))])
def test_stage_capture_rejects_wrong_coordinates_dtype_and_nonfinite(value):
    from runner.product_execution import ProductExecutionAdapter
    adapter=ProductExecutionAdapter(GraphModel(graph()))
    with pytest.raises(ValueError, match='Captured component shape'):
        adapter.validate_stage(value, 'shape', rows=3)
