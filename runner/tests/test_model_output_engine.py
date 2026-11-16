"""Actual worker and standalone Code boundary checks, including legacy graph compatibility."""
import pytest
import torch
from runner.engine import GraphModel, REGISTRY, export_source


def graph(config=None):
    return {'schemaVersion':2,'name':'Model Output contract',
            'nodes':[{'id':'input','type':'input','name':'Measurements','config':{'features':2}},
                     {'id':'head','type':'linear','name':'Identity scores','config':{'in_features':2,'out_features':2,'bias':False}},
                     {'id':'output','type':'output','name':'Model Output','config':config or {}}],
            'edges':[{'id':'values','source':'input','sourcePort':'out','target':'head','targetPort':'in'},
                     {'id':'boundary','source':'head','sourcePort':'out','target':'output','targetPort':'in'}]}


CONFIG={'checkExpectedOutput':True,'expectedDType':'float32','expectedSampleRank':'exact','expectedSampleDimensions':[2]}


def test_real_worker_and_standalone_keep_values_and_gradients_for_variable_batches():
    for configuration in ({},CONFIG):
        model=GraphModel(graph(configuration),REGISTRY)
        with torch.no_grad():model.layers[model.layer_keys['head']].weight.copy_(torch.eye(2))
        source=export_source(model);scope={};exec(source,scope);portable=scope['Model']();portable.load_state_dict(model.state_dict())
        for batch in (1,3,17):
            value=torch.arange(batch*2,dtype=torch.float32).reshape(batch,2).requires_grad_()
            for implementation in (model,portable):
                actual=implementation(value)
                torch.testing.assert_close(actual,value,rtol=0,atol=0)
                torch.testing.assert_close(torch.autograd.grad(actual.sum(),value,retain_graph=True)[0],torch.ones_like(value))


def test_runtime_checks_actual_tensor_even_when_declared_layout_matched():
    model=GraphModel(graph(CONFIG),REGISTRY)
    for value in (torch.ones(4,3),torch.ones(4,2,dtype=torch.float64),torch.ones(4,1,2)):
        with pytest.raises(ValueError,match='output.*Model Output'):
            model.execute_node('output',{'head':value},None)


def test_derived_code_enforces_boundary_and_legacy_empty_config_remains_usable():
    model=GraphModel(graph(CONFIG),REGISTRY)
    # Alter only the assertion after admission: export must encode the current contract.
    model.nodes['output']['config']=dict(CONFIG,expectedDType='float64')
    source=export_source(model);assert 'runner.' not in source
    scope={};exec(source,scope)
    with pytest.raises(ValueError,match='output.*Model Output'):
        scope['Model']()(torch.ones(3,2))
    legacy=GraphModel(graph(),REGISTRY)
    with torch.no_grad():legacy.layers[legacy.layer_keys['head']].weight.copy_(torch.eye(2))
    assert torch.equal(legacy(torch.tensor([[1.,2.]])),torch.tensor([[1.,2.]]))


def test_invalid_assertion_is_not_ignored_by_worker_admission():
    for change in ({'checkExpectedOutput':1},{'expectedDType':'float16'},{'expectedSampleDimensions':[0]}):
        with pytest.raises(ValueError,match='Model Output'):
            GraphModel(graph(dict(CONFIG,**change)),REGISTRY)

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64,torch.int32,torch.int64,torch.bool])
def test_supported_dtype_assertions_check_real_tensors_without_casts(dtype):
    from runner.model_output_configuration import check_model_output
    value=torch.ones(3,2,dtype=dtype)
    configuration=dict(CONFIG,expectedDType=str(dtype).removeprefix('torch.'))
    assert check_model_output(value,configuration,'typed-boundary') is value
    assert value.dtype==dtype and tuple(value.shape)==(3,2)
