"""Event-bound cancellation of real joined preparation is transactional."""
import pytest
from runner.dataflow_cli import run_dataflow
from runner.tests.test_dataflow_join_integration import fixture


@pytest.mark.parametrize('policy', ['off', 'materialize'])
def test_cancel_after_join_computation_never_publishes_success(tmp_path, policy):
    _, graph, bindings = fixture(tmp_path)
    output = tmp_path / 'cancelled'
    observed = []
    def cancel_after_join(event):
        observed.append(event)
        if event['event'] == 'dataflowPreview' and event['nodeID'] == 'joined':
            assert event['outputs']['table']['preview']['rowCount'] == 2
            raise KeyboardInterrupt('Deterministic cancellation after joined output staging')
    with pytest.raises(KeyboardInterrupt):
        run_dataflow(graph, bindings, output, cancel_after_join, cache_policy=policy)
    assert any(event.get('nodeID') == 'joined' for event in observed)
    assert not any(event['event'] == 'dataflowCompleted' for event in observed)
    assert not output.exists()
    assert not list(tmp_path.glob('.dataflow-*'))
    # No lingering transaction lock or ownership prevents a subsequent complete run.
    manifest = run_dataflow(graph, bindings, output, lambda _: None, cache_policy=policy)
    assert manifest['status'] == 'completed'
