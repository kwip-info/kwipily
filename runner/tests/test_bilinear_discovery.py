from runner.engine import capabilities


def test_worker_advertises_exact_bilinear_contract():
    expected = {
        'version': {'major':1,'minor':0,'patch':0},
        'adapterID':'kwip.runner.graph-v2.bilinear',
        'graphSchemaVersions':[2], 'maximumRank':8,'preservesBatch':True,
        'minimumSampleRank':1,'sampleBroadcasting':False,
        'maximumSampleElements':100_000_000,'maximumParameters':2_000_000,
        'featureMinimum':1,'featureMaximum':512,
        'biasModes':[False,True],'graphDTypes':['float32'],
    }
    assert capabilities()['componentCapabilities']['bilinear'] == expected


def test_shared_registry_executes_bilinear_without_private_overrides():
    import torch
    from runner.engine import GraphModel
    from runner.tests.test_bilinear_graph import graph
    model = GraphModel(graph())
    result = model(torch.tensor([[.1,.2],[-.3,.8]]))
    assert result.shape == (2,2)
    assert torch.isfinite(result).all()
    result.square().sum().backward()
    assert model.layers['interaction'].weight.grad.shape == (2,3,5)
