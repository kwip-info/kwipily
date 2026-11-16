"""Descriptor evidence only; shared registry activation is a separate native gate."""
from runner import lazy_linear_configuration


def test_lazy_linear_descriptor_matches_reviewed_materialization_contract():
    expected = {
        'version': {'major':1,'minor':0,'patch':0},
        'adapterID':'kwip.runner.graph-v2.lazy-linear',
        'graphSchemaVersions':[2], 'maximumRank':8, 'preservesBatch':True,
        'minimumSampleRank':1, 'maximumSampleElements':100_000_000,
        'maximumParameters':2_000_000, 'inputFeatureMinimum':1,
        'inputFeatureMaximum':4096, 'outputFeatureMinimum':1,
        'outputFeatureMaximum':512, 'biasModes':[False,True],
        'graphDTypes':['float32'], 'inputWidthInferred':True,
        'materializationPolicy':'validatedShapeBeforeOptimizer',
    }
    assert lazy_linear_configuration.capability() == expected
    # Returned JSON must not share mutable nested state with subsequent discovery.
    changed = lazy_linear_configuration.capability()
    changed['graphSchemaVersions'].append(1)
    changed['version']['minor'] = 99
    assert lazy_linear_configuration.capability() == expected


def test_worker_emits_descriptor_and_shared_registry_executes_lazy_layer():
    import torch
    from runner.engine import GraphModel, capabilities
    from runner.tests.test_lazy_linear_graph import graph
    assert capabilities()['componentCapabilities']['lazyLinear'] == lazy_linear_configuration.capability()
    model=GraphModel(graph())
    model.materialize_lazy_parameters()
    x=torch.tensor([[.1,.2],[-.3,.8]],requires_grad=True)
    output=model(x)
    assert output.shape==(2,2) and torch.isfinite(output).all()
    output.square().sum().backward()
    assert model.layers['lazy'].weight.grad.shape==(2,2)
