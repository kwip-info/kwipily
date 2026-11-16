"""Preparation storage limits reject bytes before publishing owned artifacts."""
import json
import pytest
from runner import dataflow_cli as adapter
from runner.tests.test_columns import config, item


def prepare(output, events=None, *, workload_policy=None):
    # Publication-phase budgets must not depend on an evolving guided practice
    # or a View index's separate up-front reservation. View has dedicated tests.
    source = output.parent/'budget-source.csv'
    source.write_text('value,label\n1,0\n2,1\n')
    graph = output.parent/'budget-graph.json'
    graph.write_text(json.dumps({'schemaVersion': 1, 'nodes': [
        {'id': 'source', 'operator': 'kwip.data.sourceCSV', 'config': {}},
        {'id': 'columns', 'operator': 'kwip.data.columns',
         'config': config([item('value', dtype='float64'), item('label', dtype='int64')])}],
        'edges': [{'from': {'node': 'source', 'port': 'table'},
                   'to': {'node': 'columns', 'port': 'table'}}]}))
    return adapter.run_dataflow(graph, {'source': str(source)},
        output, (events if events is not None else []).append, retain_partial=True,
        workload_policy=workload_policy)


def incomplete_stage(parent, output, events):
    assert not output.exists()
    assert not any(event['event'] == 'dataflowCompleted' for event in events)
    stage, = parent.glob('.dataflow-*')
    assert not (stage/'manifest.json').exists()
    assert not list(stage.glob('*.tmp'))
    return stage


def test_support_copy_budget_is_checked_before_support_bytes_are_written(tmp_path, monkeypatch):
    baseline = prepare(tmp_path/'baseline')
    prefix_bytes = sum(entry['bytes'] for entry in baseline['artifacts']
                       if entry['kind'] not in ('preparationCode', 'registry') or entry['path'] == 'prep.py')
    monkeypatch.setattr(adapter, 'MAX_ARTIFACT_BYTES', prefix_bytes + 1)
    events = []
    output = tmp_path/'rejected'
    with pytest.raises(ValueError, match='size limit|storage|budget'):
        prepare(output, events)
    stage = incomplete_stage(tmp_path, output, events)
    assert (stage/'node-000.json').is_file()
    assert all(not (stage/name).exists() for name in adapter.PORTABLE_DEPENDENCIES)


def test_generated_preparation_code_budget_is_checked_before_write(tmp_path, monkeypatch):
    baseline = prepare(tmp_path/'baseline')
    prefix_bytes = sum(entry['bytes'] for entry in baseline['artifacts']
                       if entry['kind'] not in ('preparationCode', 'registry'))
    monkeypatch.setattr(adapter, 'MAX_ARTIFACT_BYTES', prefix_bytes + 1)
    events = []
    output = tmp_path/'rejected'
    with pytest.raises(ValueError, match='size limit|storage|budget'):
        prepare(output, events)
    stage = incomplete_stage(tmp_path, output, events)
    assert not (stage/'prep.py').exists()


def test_manifest_per_file_limit_rejects_before_write(tmp_path, monkeypatch):
    monkeypatch.setattr(adapter, 'MAX_MANIFEST_BYTES', 1, raising=False)
    events = []
    output = tmp_path/'rejected'
    with pytest.raises(ValueError, match='size limit|storage|budget'):
        prepare(output, events)
    stage = incomplete_stage(tmp_path, output, events)
    assert (stage/'prep.py').is_file()


def test_total_output_budget_charges_manifest_in_addition_to_support_files(tmp_path, monkeypatch):
    baseline = prepare(tmp_path/'baseline')
    total = sum(entry['bytes'] for entry in baseline['artifacts']) + (tmp_path/'baseline/manifest.json').stat().st_size
    monkeypatch.setattr(adapter, 'MAX_OUTPUT_BYTES', total - 1, raising=False)
    events = []
    output = tmp_path/'rejected'
    with pytest.raises(ValueError, match='size limit|storage|budget'):
        prepare(output, events)
    stage = incomplete_stage(tmp_path, output, events)
    assert all((stage/entry['path']).is_file() for entry in baseline['artifacts'])


def test_exact_total_output_budget_publishes_complete_inventory(tmp_path, monkeypatch):
    baseline = prepare(tmp_path/'baseline')
    total = sum(entry['bytes'] for entry in baseline['artifacts']) + (tmp_path/'baseline/manifest.json').stat().st_size
    monkeypatch.setattr(adapter, 'MAX_OUTPUT_BYTES', total, raising=False)
    events = []
    result = prepare(tmp_path/'exact', events)
    assert result['artifacts'] == baseline['artifacts']
    assert events[-1]['event'] == 'dataflowCompleted'
    assert (tmp_path/'exact/manifest.json').is_file()


def test_captured_workload_uses_its_own_budget_instead_of_legacy_constant(tmp_path, monkeypatch):
    from runner.workload_policy import WorkloadPolicy
    policy = WorkloadPolicy('balanced')
    monkeypatch.setattr(adapter, 'MAX_OUTPUT_BYTES', 1)
    output = tmp_path/'captured'
    result = prepare(output, workload_policy=policy.to_document())
    assert result['workloadPolicy'] == policy.to_document()
    assert result['tableStorageVersion'] == 1
    assert sum(entry['bytes'] for entry in result['artifacts']) > 1
    assert (output/'manifest.json').is_file()


def test_portable_laya_code_retains_upstream_license(tmp_path):
    from pathlib import Path
    result = prepare(tmp_path/'licensed')
    source = Path(adapter.__file__).with_name('laya-LICENSE.txt')
    canonical = Path(adapter.__file__).resolve().parents[1]/'third_party/laya/LICENSE'
    assert source.read_bytes() == canonical.read_bytes()
    assert (tmp_path/'licensed/laya-LICENSE.txt').read_bytes() == canonical.read_bytes()
    assert any(entry['path'] == 'laya-LICENSE.txt' for entry in result['artifacts'])
