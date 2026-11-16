"""Cooperative authority revocation must never publish completed training."""
import pytest
from runner import engine
from runner.tests.test_runner import graph


@pytest.mark.parametrize('boundary',['entry','batch','publication'])
def test_training_authority_cancellation_blocks_completion(tmp_path,monkeypatch,boundary):
    output=tmp_path/'run';events=[];batch_finished=False
    original=engine.torch.optim.Adam.step
    def step(self,*args,**kwargs):
        nonlocal batch_finished
        result=original(self,*args,**kwargs);batch_finished=True;return result
    monkeypatch.setattr(engine.torch.optim.Adam,'step',step)
    def check():
        if boundary=='entry' or (boundary=='batch' and batch_finished) or (boundary=='publication' and (output/'manifest.json.tmp').exists()):
            raise InterruptedError('training authority revoked')
    with pytest.raises(InterruptedError):
        engine.run(graph(),output,epochs=2,batch_size=32,device='cpu',emit=events.append,check_cancelled=check)
    assert not (output/'manifest.json').exists()
    assert not any(event['event']=='completed' for event in events)
    if boundary=='entry':assert not batch_finished and not events
    if boundary=='batch':assert batch_finished and not any(event['event']=='epoch' for event in events)
