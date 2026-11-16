import json
import math
import pytest
import torch
from runner.tensor_heatmap import summarize_tensor


def test_all_values_contribute_to_independently_checkable_tiles():
    source=torch.arange(48,dtype=torch.float32).reshape(2,4,6).requires_grad_()
    result=summarize_tensor(source,max_tiles=12,max_planes=8)
    assert result['schemaVersion']==1 and result['coverage']=='complete'
    assert result['sourceShape']==[2,4,6] and result['totalPlanes']==2
    rows,cols=result['gridShape']
    for plane_index,plane in enumerate(result['planes']):
        for r in range(rows):
            for c in range(cols):
                values=[float(source[plane_index,y,x].detach()) for y in range(4) for x in range(6) if y*rows//4==r and x*cols//6==c]
                k=r*cols+c
                assert plane['mean'][k]==pytest.approx(sum(values)/len(values))
                assert plane['minimum'][k]==min(values) and plane['maximum'][k]==max(values)
                assert plane['finiteCount'][k]==len(values)
    assert source.grad is None
    assert sum(sum(p['finiteCount']) for p in result['planes'])==48
    json.dumps(result,allow_nan=False)


def test_nonfinite_empty_constant_and_transposed_sources_are_honest():
    source=torch.tensor([[1.,float('nan')],[float('inf'),float('-inf')]])
    result=summarize_tensor(source,max_tiles=1)
    plane=result['planes'][0]
    assert plane['mean']==[1.] and plane['finiteCount']==[1]
    assert plane['nanCount']==[1] and plane['positiveInfinityCount']==[1] and plane['negativeInfinityCount']==[1]
    assert summarize_tensor(torch.empty(0,5))['coverage']=='empty'
    assert summarize_tensor(torch.full((4,5),7.))['planes'][0]['mean']==[7.]*20
    transposed=torch.arange(120.).reshape(4,5,6).transpose(0,2)
    captured=summarize_tensor(transposed)
    assert captured['sourceShape']==[6,5,4]
    assert sum(sum(p['finiteCount']) for p in captured['planes'])==120
    json.dumps(result,allow_nan=False)


def test_budgets_bound_payload_and_label_partial_plane_coverage():
    result=summarize_tensor(torch.zeros(100,32,32),max_planes=4,max_tiles=256)
    assert result['coverage']=='partialPlanes' and result['totalPlanes']==100 and len(result['planes'])==4
    assert sum(len(p['mean']) for p in result['planes'])<=256
    assert summarize_tensor(torch.empty(200,200),max_source_elements=100)['coverage']=='unavailable'
    assert summarize_tensor(torch.tensor(3.))['planeShape']==[1,1]
    assert summarize_tensor(torch.tensor([1.,2.,3.]))['planeShape']==[1,3]
    assert summarize_tensor(torch.ones(2,2,dtype=torch.complex64))['coverage']=='unavailable'


def test_cancelled_capture_does_not_swallow_cancellation():
    def cancel():raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError,match='cancelled'):summarize_tensor(torch.zeros(4,4),check_cancelled=cancel)


def test_shared_capture_budget_and_sample_policy_are_independent(tmp_path):
    from runner.tensor_heatmap import HeatmapCaptureBudget
    from runner.output_routing import summarize_stage
    from runner.capture_policy import validate_training_sample_capture_policy
    from runner.engine import run
    from runner.tests.test_runner import graph
    budget=HeatmapCaptureBudget(maximum_bytes=100,maximum_source_elements=100)
    assert budget.capture(torch.ones(8,8))['coverage']=='unavailable'
    result=summarize_stage(torch.arange(20.).reshape(4,5),include_values=False,heatmap_budget=HeatmapCaptureBudget())
    assert result['values']==[] and result['heatmap']['coverage']=='complete'
    policy=dict(schemaVersion=4,activationValues=False,predictionRows=False,performanceMeasurements=False,retainComponentOutputs=True,retainDiagnosticDetails=False,tensorHeatmaps=True,heatmapFrameLimit=0)
    assert validate_training_sample_capture_policy(policy)==policy
    captured=run(graph(),tmp_path/'run',epochs=1,device='cpu',sample_capture=policy)
    assert all(p['values']==[] and p['heatmap']['coverage']=='complete' for p in captured['previews'].values())
    disabled=dict(policy,retainComponentOutputs=False)
    with pytest.raises(ValueError,match='heatmaps'):validate_training_sample_capture_policy(disabled)


def test_scalar_and_vector_heatmaps_never_invent_source_axes():
    scalar=summarize_tensor(torch.tensor(3.))
    vector=summarize_tensor(torch.tensor([1.,2.,3.]))
    assert scalar['rowAxis'] is None and scalar['columnAxis'] is None
    assert vector['rowAxis'] is None and vector['columnAxis']==0
