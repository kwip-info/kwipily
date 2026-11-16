import json
import subprocess
import sys
import torch
import pytest
from runner.engine import run
from runner.tests.test_runner import graph
from runner.tests.test_checkpoint_lifecycle import state, assert_nested_equal

CONFIG={'schemaVersion':1,'everyEpochs':1,'maxBatchesPerEpoch':2,'maxCaptures':16}

def test_capture_cpu_math_and_resume_unchanged(tmp_path):
    events=[]
    off_events=[]
    run(graph(),tmp_path/'off',epochs=4,device='cpu',batch_size=32,emit=off_events.append)
    assert not any(e['event']=='gradientSummary' or 'gradientCapture' in e for e in off_events)
    run(graph(),tmp_path/'on',epochs=4,device='cpu',batch_size=32,gradient_capture=CONFIG,emit=events.append)
    assert_nested_equal(state(tmp_path/'off'),state(tmp_path/'on'))
    assert (tmp_path/'off'/'metrics.json').read_bytes()==(tmp_path/'on'/'metrics.json').read_bytes()
    assert not (tmp_path/'off'/'gradient-summary.json').exists()
    run(graph(),tmp_path/'first',epochs=2,device='cpu',batch_size=32)
    run(graph(),tmp_path/'resumed',epochs=4,device='cpu',batch_size=32,resume_checkpoint=tmp_path/'first'/'checkpoint.pt',gradient_capture=CONFIG)
    assert_nested_equal(state(tmp_path/'off'),state(tmp_path/'resumed'))
    report=json.loads((tmp_path/'resumed'/'gradient-summary.json').read_text())
    assert report['initialEpoch']==2
    assert [r['epoch'] for r in report['summaries']]==[3,4]
    assert any(e['event']=='gradientSummary' for e in events)

@pytest.mark.parametrize('device',['cpu',pytest.param('mps',marks=pytest.mark.skipif(not torch.backends.mps.is_available(),reason='Actual MPS unavailable'))])
def test_summary_analytical_missing_nonfinite_budget(device):
    from runner.gradient_capture import GradientCapture
    model=torch.nn.Linear(2,1).to(device)
    capture=GradientCapture(model,CONFIG,run_id='r',initial_epoch=0,final_epoch=4,device=device)
    capture.begin_epoch(1,5,9)
    model.weight.grad=torch.tensor([[3.,4.]],device=device)
    model.bias.grad=None
    capture.observe(0,2)
    event=capture.end_epoch()
    batch=event['batches'][0]
    assert batch['l2Norm']==5 and batch['maxAbs']==4
    assert batch['missingParameterCount']==1 and batch['missingElementCount']==1
    capture.begin_epoch(2,1,2)
    model.weight.grad=torch.tensor([[float('nan'),float('inf')]],device=device)
    model.bias.grad=torch.zeros(1,device=device)
    capture.observe(0,2)
    batch=capture.end_epoch()['batches'][0]
    assert batch['nonfiniteElementCount']==2 and batch['zeroElementCount']==1
    assert batch['l2Norm'] is None and batch['maxAbs'] is None
    json.dumps(capture.artifact(),allow_nan=False)

@pytest.mark.parametrize('config',[{**CONFIG,'everyEpochs':True},{**CONFIG,'maxCaptures':1025},{**CONFIG,'other':1}])
def test_invalid_before_allocation(tmp_path,config):
    with pytest.raises(ValueError):run(graph(),tmp_path/'out',gradient_capture=config)
    assert not (tmp_path/'out').exists()

def test_cadence_partial_batch_and_exhaustion(tmp_path):
    events=[]
    config={**CONFIG,'everyEpochs':2,'maxBatchesPerEpoch':2,'maxCaptures':3}
    run(graph(),tmp_path/'out',epochs=5,device='cpu',batch_size=300,gradient_capture=config,emit=events.append)
    rows=[e for e in events if e['event']=='gradientSummary']
    assert [e['status'] for e in rows]==['captured','notScheduled','captured','notScheduled','budgetExhausted']
    assert [e['observedBatchCount'] for e in rows]==[2,0,1,0,0]
    assert rows[2]['scheduledBatchCount']==2
    assert rows[0]['batches'][-1]['batchIndex']==rows[0]['totalBatches']-1
    assert rows[0]['batches'][-1]['trainingRows']<=300
    assert rows[0]['observedTrainingRows']==sum(b['trainingRows'] for b in rows[0]['batches'])
    import hashlib
    manifest=json.loads((tmp_path/'out'/'manifest.json').read_text())
    content=(tmp_path/'out'/'gradient-summary.json').read_bytes()
    assert manifest['gradientCapture']['sha256']==hashlib.sha256(content).hexdigest()
    assert manifest['gradientCapture']['bytes']==len(content)
    assert manifest['gradientCapture']==events[-1]['gradientCapture']

def test_cli_capture_and_reject_unrelated_commands(tmp_path):
    path=tmp_path/'graph.json';path.write_text(json.dumps(graph()))
    args=[sys.executable,'runner/main.py']
    result=subprocess.run(args+['--graph',str(path),'--output',str(tmp_path/'out'),'--epochs','1','--device','cpu','--capture-gradients'],capture_output=True,text=True)
    assert result.returncode==0,result.stdout
    assert any(json.loads(line)['event']=='gradientSummary' for line in result.stdout.splitlines())
    for extra in (['--capabilities','--capture-gradients'],['--capabilities','--gradient-max-captures','1'],['--graph',str(path),'--output',str(tmp_path/'bad'),'--gradient-max-captures','1']):
        failed=subprocess.run(args+extra,capture_output=True,text=True)
        assert failed.returncode==1 and json.loads(failed.stdout)['event']=='error'
    discovery=subprocess.run(args+['--capabilities'],capture_output=True,text=True,check=True)
    assert json.loads(discovery.stdout)['gradientCapture']['schemaVersion']==1

def test_cancellation_during_gradient_event_does_not_commit(tmp_path):
    def emit(event):
        if event['event']=='gradientSummary':raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):run(graph(),tmp_path/'cancel',epochs=2,device='cpu',gradient_capture=CONFIG,emit=emit)
    assert not (tmp_path/'cancel'/'manifest.json').exists()
    assert not (tmp_path/'cancel'/'gradient-summary.json').exists()

@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='Actual MPS unavailable')
def test_actual_mps_capture_on_off_numerical_parity(tmp_path):
    run(graph(),tmp_path/'off',epochs=3,device='mps',batch_size=32)
    run(graph(),tmp_path/'on',epochs=3,device='mps',batch_size=32,gradient_capture=CONFIG)
    a,b=state(tmp_path/'off'),state(tmp_path/'on')
    for name in a['modelState']:torch.testing.assert_close(a['modelState'][name],b['modelState'][name],rtol=1e-6,atol=1e-7)
    assert torch.equal(a['rngState'],b['rngState'])
    assert_nested_equal(a['optimizerState'],b['optimizerState'])
    assert (tmp_path/'off'/'metrics.json').read_bytes()==(tmp_path/'on'/'metrics.json').read_bytes()


def test_biasless_branch_inventory_zero_and_maximum_report_budget(tmp_path):
    from runner.gradient_capture import GradientCapture
    from runner.engine import GraphModel
    g=graph();g['schemaVersion']=2
    for node in g['nodes']:
        if node['type']=='linear':node['config']['bias']=False
    model=GraphModel(g)
    capture=GradientCapture(model,{**CONFIG,'maxBatchesPerEpoch':8,'maxCaptures':1024},run_id='r',initial_epoch=0,final_epoch=10000,device='cpu')
    assert all(not p['name'].endswith('.bias') for p in capture.inventory['parameters'])
    for parameter in model.parameters():parameter.grad=torch.zeros_like(parameter)
    for epoch in range(1,10001):
        capture.begin_epoch(epoch,8,8)
        for batch in range(8):capture.observe(batch,1)
        capture.end_epoch()
    rows=capture.summaries
    assert sum(row['observedBatchCount'] for row in rows)==1024
    assert rows[-1]['status']=='budgetExhausted'
    first=rows[0]['batches'][0]
    assert first['l2Norm']==0 and first['maxAbs']==0
    assert first['zeroElementCount']==capture.inventory['elementCount']
    assert len(json.dumps(capture.artifact(),allow_nan=False).encode())<8_000_000
    for parameter in model.parameters():parameter.grad=None
    absent=GradientCapture(model,CONFIG,run_id='r',initial_epoch=0,final_epoch=1,device='cpu')
    absent.begin_epoch(1,1,1);absent.observe(0,1)
    row=absent.end_epoch()['batches'][0]
    assert row['l2Norm'] is None and row['presentElementCount']==0
    assert row['missingElementCount']==capture.inventory['elementCount']


@pytest.mark.parametrize('device',['cpu',pytest.param('mps',marks=pytest.mark.skipif(not torch.backends.mps.is_available(),reason='Actual MPS unavailable'))])
def test_backward_summary_matches_independent_cpu_reference(device,monkeypatch):
    from runner.gradient_capture import GradientCapture, LIMITS
    # Force multiple bounded chunks; include genuine zero gradients and nonzero bias.
    monkeypatch.setitem(LIMITS,'chunkElements',2)
    model=torch.nn.Linear(3,2).to(device)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.,2.,3.],[-1.,0.,2.]],device=device))
        model.bias.copy_(torch.tensor([.5,-.5],device=device))
    x=torch.tensor([[1.,0.,2.],[-2.,0.,1.]],device=device)
    model(x).square().mean().backward()
    originals=[p.grad.detach().cpu().clone() for p in model.parameters()]
    # Python scalars are an independent reference, without a tensor dtype conversion.
    values=[float(v) for gradient in originals for v in gradient.flatten().tolist()]
    import math
    expected=math.sqrt(sum(v*v for v in values))
    assert expected>0
    capture=GradientCapture(model,CONFIG,run_id='r',initial_epoch=0,final_epoch=1,device=device)
    capture.begin_epoch(1,1,2);capture.observe(0,2)
    batch=capture.end_epoch()['batches'][0]
    assert batch['l2Norm']==pytest.approx(expected,rel=1e-12)
    assert batch['maxAbs']==max(abs(v) for v in values)
    assert batch['zeroElementCount']==sum(v==0 for v in values)
    assert batch['finiteElementCount']==len(values)
    assert batch['nonfiniteElementCount']==0
    for parameter,original in zip(model.parameters(),originals):
        assert torch.equal(parameter.grad.cpu(),original)
