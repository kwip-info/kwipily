import json
import torch
import pytest
from runner.heatmap_playback import HeatmapPlayback


def test_sparse_frames_are_bounded_and_decorated_with_actual_epochs():
    playback=HeatmapPlayback(epoch_limit=40,max_frames=6)
    for epoch in range(1,41):
        if playback.should_capture(epoch):playback.capture(epoch,{'a':torch.full((8,4),float(epoch))},source_rows=8)
    preview={'a':dict(shape=[8,4],dtype='float32',values=[1.])}
    playback.decorate(preview)
    frames=preview['a']['heatmapPlayback']['frames']
    assert len(frames)<=6 and frames[0]['epoch']==1 and frames[-1]['epoch']==40
    assert all(f['sourceRows']==8 and f['summary']['planes'][0]['mean'][0]==f['epoch'] for f in frames)
    assert len(json.dumps(preview,allow_nan=False).encode())<700000


@pytest.mark.parametrize('device',['cpu','mps'])
def test_playback_capture_reuses_forwards_without_changing_training(tmp_path,monkeypatch,device):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
    from runner.engine import run
    from runner.tests.test_runner import graph
    from runner.tests.test_checkpoint_lifecycle import assert_nested_equal
    policy=dict(schemaVersion=4,activationValues=False,predictionRows=False,performanceMeasurements=False,retainComponentOutputs=True,retainDiagnosticDetails=False,tensorHeatmaps=True,heatmapFrameLimit=4)
    from runner.product_execution import ProductExecutionAdapter
    original=ProductExecutionAdapter._forward_trusted
    calls=[]
    def measured(self,*args,**kwargs):
        calls.append(kwargs.get('inspect',False))
        return original(self,*args,**kwargs)
    monkeypatch.setattr(ProductExecutionAdapter,'_forward_trusted',measured)
    captured=run(graph(),tmp_path/'with',epochs=4,device=device,sample_capture=policy)
    with_calls=list(calls);calls.clear()
    run(graph(),tmp_path/'without',epochs=4,device=device,sample_capture=dict(policy,heatmapFrameLimit=0))
    assert len(with_calls)==len(calls) and any(with_calls) and not any(calls)
    a=torch.load(tmp_path/'with/checkpoint.pt',weights_only=True);b=torch.load(tmp_path/'without/checkpoint.pt',weights_only=True)
    for key in ('modelState','optimizerState','rngState'):assert_nested_equal(a[key],b[key])
    assert all(p['heatmapPlayback']['frames'] for p in captured['previews'].values())
    assert (tmp_path/'with/metrics.json').read_bytes()==(tmp_path/'without/metrics.json').read_bytes()


def test_optional_summaries_fit_completion_transport_without_discarding_exact_values():
    from runner.heatmap_playback import bound_completion_heatmaps
    from runner.tensor_heatmap import summarize_tensor
    summary=summarize_tensor(torch.arange(2048.).reshape(32,8,8))
    previews={str(i):dict(shape=[32,8,8],dtype='float32',values=[i],heatmap=summary,heatmapPlayback=dict(schemaVersion=1,source='validationFirstBatch',frames=[dict(epoch=1,sourceRows=32,summary=summary)])) for i in range(16)}
    event=dict(event='completed',previews=previews,predictionRows=[dict(input=[1,2],predicted=0)])
    assert len(json.dumps(event).encode())>1_000_000
    bound_completion_heatmaps(event,maximum_bytes=200_000)
    assert len(json.dumps(event,allow_nan=False).encode())<=200_000
    assert all(p['values']==[int(i)] and p['shape']==[32,8,8] for i,p in previews.items())
    assert all('heatmap' in p and 'heatmapPlayback' in p for p in previews.values())
    assert any(p['heatmap']['coverage']=='unavailable' for p in previews.values())
