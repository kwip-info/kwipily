"""Actual representative capture-forward events, never a simulated training sweep."""
import pytest
import torch
from runner.engine import GraphModel, export_source, run
from runner.product_execution import ProductExecutionAdapter
from runner.tests.test_runner import graph


def test_observer_covers_actual_operations_and_preserves_outputs_gradients_and_rng():
    model = GraphModel(graph())
    x = torch.randn(3, 2, requires_grad=True)
    rng = torch.get_rng_state().clone()
    expected = model(x)
    events = []
    actual = model(x, operation_observer=lambda state, node: events.append((state, node)))
    assert events == [(state, node) for node in model.order for state in ('started', 'completed')]
    assert torch.equal(actual, expected)
    assert torch.equal(torch.autograd.grad(actual.sum(), x, retain_graph=True)[0], torch.autograd.grad(expected.sum(), x)[0])
    assert torch.equal(torch.get_rng_state(), rng)
    source = export_source(model)
    assert 'operation_observer' not in source
    exported = {}; exec(source, exported)
    copied = exported['Model'](); copied.load_state_dict(model.state_dict())
    assert torch.equal(copied(x), actual)


def test_failed_operation_has_start_without_false_completion(monkeypatch):
    model = GraphModel(graph()); failed = model.order[2]
    original = model.execute_node
    def execute(node, values, x):
        if node == failed: raise ValueError('original computation error')
        return original(node, values, x)
    monkeypatch.setattr(model, 'execute_node', execute)
    events = []
    with pytest.raises(ValueError, match='original computation error'):
        ProductExecutionAdapter(model).forward(torch.ones(3, 2), inspect=True,
            operation_observer=lambda state, node: events.append((state, node)))
    assert events[-1] == ('started', failed)
    assert ('completed', failed) not in events
    assert not hasattr(model, '_capture_operation_observer')


def test_run_emits_only_one_final_validation_capture_frame(tmp_path):
    events = []
    run(graph(), tmp_path/'run', epochs=2, device='cpu', sample_limit=40, emit=events.append)
    captured = [e for e in events if e['event'].startswith('modelCapture') or e['event'].startswith('modelNode')]
    nodes = GraphModel(graph()).order
    assert len(captured) == len(nodes)*2+2
    assert [e['sequence'] for e in captured] == list(range(1, len(captured)+1))
    assert {e['stage'] for e in captured} == {'validationCapture'}
    assert len({e['runID'] for e in captured}) == 1
    assert captured[0]['event'] == 'modelCaptureStarted'
    assert captured[-1]['event'] == 'modelCaptureCompleted'
    assert [e['nodeID'] for e in captured if e['event']=='modelNodeStarted'] == nodes
    assert events.index(captured[0]) > max(i for i,e in enumerate(events) if e['event']=='epoch')
    assert all(e['schemaVersion']==1 for e in captured)
    assert captured[0]['operationCount'] == len(nodes)
    assert not any('values' in e for e in captured)


def test_observer_cancellation_stops_before_next_operation():
    model = GraphModel(graph()); events = []
    def observe(state, node):
        events.append((state, node))
        if state == 'started' and node == model.order[1]: raise KeyboardInterrupt('cancelled')
    with pytest.raises(KeyboardInterrupt, match='cancelled'):
        model(torch.ones(2, 2), operation_observer=observe)
    assert events == [('started', model.order[0]), ('completed', model.order[0]), ('started', model.order[1])]
