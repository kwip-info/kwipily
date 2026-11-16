"""Handled numerical failures retain observations, never a completed model."""
import json
import pytest
from runner.engine import run, capabilities
from runner.gradient_capture import GradientCapture

CONFIG = {'schemaVersion': 1, 'everyEpochs': 1, 'maxBatchesPerEpoch': 1, 'maxCaptures': 4}


def lifecycle(events):
    # Completed-work chart observations are separate from handled-failure evidence.
    # Keep the original lifecycle sequence and numerical assertions exact while
    # checking that any new progress is owned and ordered.
    progress=[e for e in events if e['event']=='trainingProgress']
    if progress:
        owner=next(e['runID'] for e in events if e['event']=='started')
        assert all(e['schemaVersion']==1 and e['runID']==owner and e['sequence']==i+1 for i,e in enumerate(progress))
    return [e['event'] for e in events if e['event']!='trainingProgress']



def failure_graph():
    specs = [('input', 'input', {'features': 2}),
             ('head', 'linear', {'in_features': 2, 'out_features': 1, 'bias': True})]
    wires = [('input', 'head', 'in')]
    previous = 'head'
    for index in range(4):
        node = f'b{index}'
        specs.append((node, 'bilinear', {'in1_features': 1, 'in2_features': 1, 'out_features': 1, 'bias': True}))
        wires.extend([(previous, node, 'a'), (previous, node, 'b')])
        previous = node
    specs.extend([('bound', 'tanh', {}), ('score', 'linear', {'in_features': 1, 'out_features': 2, 'bias': True}), ('output', 'output', {})])
    wires.extend([(previous, 'bound', 'in'), ('bound', 'score', 'in'), ('score', 'output', 'in')])
    return {'schemaVersion': 2, 'name': 'Saturated gradient failure',
            'nodes': [{'id': node, 'type': kind, 'config': config} for node, kind, config in specs],
            'edges': [{'id': str(i), 'source': source, 'sourcePort': 'out', 'target': target, 'targetPort': port}
                      for i, (source, target, port) in enumerate(wires)]}


def source(tmp_path, scale=1e9):
    path = tmp_path / 'observations.csv'
    path.write_text('signal,context,label\n' + ''.join(
        f'{scale * (0.5 + i / 128)},{scale * (1 - i / 128)},{i % 2}\n' for i in range(64)))
    return path


def execute(tmp_path, events, *, capture=CONFIG, value=None, scale=1e9, **kwargs):
    return run(value or failure_graph(), tmp_path / 'output', dataset_csv=source(tmp_path, scale),
               feature_columns=['signal', 'context'], target_column='label', device='cpu',
               epochs=2, seed=42, batch_size=32, gradient_capture=capture, emit=events.append, **kwargs)


def test_real_backward_failure_retains_partial_nonfinite_counts(tmp_path):
    events = []
    with pytest.raises(ValueError, match='Non-finite training loss'):
        execute(tmp_path, events)
    assert lifecycle(events) == ['started', 'gradientInterrupted']
    started=next(e for e in events if e['event']=='started')
    partial=next(e for e in events if e['event']=='gradientInterrupted')
    assert partial['id'] == 'kwip.failed-gradient-observations.v1'
    assert partial['schemaVersion'] == 1 and partial['runID'] == started['runID']
    assert partial['epoch'] == 1 and partial['stage'] == 'trainingForward'
    assert partial['lastBackwardBatchIndex'] == 0
    assert partial['totalTrainingRows'] == started['trainingSamples']
    assert partial['observedBatchCount'] == 1 and partial['observedTrainingRows'] == 32
    assert partial['scheduledBatchCount'] == 1 and partial['captureBudgetRemaining'] == 3
    batch = partial['batches'][0]
    assert batch['nonfiniteElementCount'] == 10 and batch['presentElementCount'] == 15
    assert batch['finiteElementCount'] == 5 and batch['l2Norm'] is None and batch['maxAbs'] is None
    assert len(json.dumps(partial).encode()) <= 100_000
    assert not any((tmp_path / 'output' / name).exists() for name in ['manifest.json', 'checkpoint.pt', 'gradient-summary.json'])


def test_disabled_capture_preserves_original_failure_without_observations(tmp_path):
    events = []
    with pytest.raises(ValueError, match='Non-finite training loss'):
        execute(tmp_path, events, capture=None)
    assert lifecycle(events) == ['started']


def test_failure_before_any_backward_invents_no_observations(tmp_path):
    value = failure_graph()
    value['nodes'] = [node for node in value['nodes'] if node['id'] != 'bound']
    value['edges'] = [edge for edge in value['edges'] if edge['target'] != 'bound']
    next(edge for edge in value['edges'] if edge['source'] == 'bound')['source'] = 'b3'
    events = []
    with pytest.raises(ValueError, match='Non-finite training loss'):
        execute(tmp_path, events, value=value)
    assert lifecycle(events) == ['started']


def test_capture_failure_is_not_a_computation_observation(tmp_path, monkeypatch):
    def broken(*args):
        raise RuntimeError('capture failed')
    monkeypatch.setattr(GradientCapture, 'observe', broken)
    events = []
    with pytest.raises(RuntimeError, match='capture failed'):
        execute(tmp_path, events)
    assert lifecycle(events) == ['started']


def test_cancellation_after_backward_does_not_claim_handled_failure(tmp_path, monkeypatch):
    class Cancelled(Exception):
        pass
    observed = False
    original = GradientCapture.observe
    def observe(self, *args):
        nonlocal observed
        original(self, *args)
        observed = True
    def check():
        if observed:
            raise Cancelled('cancelled')
    monkeypatch.setattr(GradientCapture, 'observe', observe)
    events = []
    with pytest.raises(Cancelled):
        execute(tmp_path, events, check_cancelled=check)
    assert lifecycle(events) == ['started']


def test_healthy_counterpart_keeps_complete_artifact_path(tmp_path):
    events = []
    execute(tmp_path, events, scale=1)
    assert events[-1]['event'] == 'completed'
    assert not any(e['event'] == 'gradientInterrupted' for e in events)
    artifact = json.loads((tmp_path / 'output' / 'gradient-summary.json').read_text())
    assert artifact['finalEpoch'] == 2 and len(artifact['summaries']) == 2
    assert all(batch['nonfiniteElementCount'] == 0 for epoch in artifact['summaries'] for batch in epoch['batches'])


def test_failed_observation_capability_is_explicit_and_bounded():
    assert capabilities()['failedGradientCapture'] == {
        'schemaVersion':1,'id':'kwip.failed-gradient-observations.v1',
        'supportedDevices':['cpu','mps'],'phase':'afterBackwardBeforeStep',
        'limits':{'eventBytes':100_000,'artifactBytes':8_000_000,'captures':1024}}


def test_partial_epoch_does_not_invent_the_second_scheduled_batch(tmp_path):
    events = []
    with pytest.raises(ValueError, match='Non-finite training loss'):
        execute(tmp_path, events, capture={**CONFIG, 'maxBatchesPerEpoch':2})
    partial = events[-1]
    assert partial['event'] == 'gradientInterrupted'
    assert partial['scheduledBatchCount'] == 2 and partial['observedBatchCount'] == 1
    assert [batch['batchIndex'] for batch in partial['batches']] == [0]


def test_transport_failure_preserves_original_error_without_failure_snapshot(tmp_path):
    class BrokenSink(list):
        def append(self, event):
            if event['event'] == 'epoch':
                raise RuntimeError('transport failed')
            super().append(event)
    events = BrokenSink()
    with pytest.raises(RuntimeError, match='transport failed'):
        execute(tmp_path, events, scale=1)
    assert lifecycle(events) == ['started']


def test_real_cli_preserves_handled_failure_exit_and_original_error(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    graph_path = tmp_path / 'graph.json'
    graph_path.write_text(json.dumps(failure_graph()))
    result = subprocess.run([sys.executable, str(Path(__file__).parents[1] / 'main.py'),
        '--graph', str(graph_path), '--dataset-csv', str(source(tmp_path)),
        '--feature-columns-json', '["signal","context"]', '--target-column', 'label',
        '--output', str(tmp_path / 'output'), '--device', 'cpu', '--epochs', '2',
        '--batch-size', '32', '--capture-gradients', '--gradient-max-captures', '4'],
        capture_output=True, text=True, timeout=20)
    assert result.returncode == 1
    events = [json.loads(line) for line in result.stdout.splitlines()]
    assert lifecycle(events) == ['started', 'gradientInterrupted', 'error']
    assert 'Non-finite training loss' in events[-1]['message']
    assert next(e for e in events if e['event']=='gradientInterrupted')['batches'][0]['nonfiniteElementCount'] == 10


def test_exhausted_current_epoch_retains_only_observed_completed_prefix(tmp_path, monkeypatch):
    import torch
    original = torch.optim.Adam.step
    calls = 0
    def step(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError('optimizer failed after capture budget exhausted')
        return original(self, *args, **kwargs)
    monkeypatch.setattr(torch.optim.Adam, 'step', step)
    events = []
    with pytest.raises(RuntimeError, match='optimizer failed'):
        execute(tmp_path, events, scale=1, capture={**CONFIG, 'maxCaptures':1})
    assert lifecycle(events) == ['started', 'epoch', 'gradientSummary', 'gradientInterrupted']
    complete, partial = events[-2:]
    assert complete['observedBatchCount'] == 1
    assert partial['epoch'] == 2 and partial['stage'] == 'optimizerStep'
    assert partial['lastBackwardBatchIndex'] == 0
    assert partial['observedBatchCount'] == 0 and partial['captureBudgetRemaining'] == 0
    assert partial['batches'] == []
