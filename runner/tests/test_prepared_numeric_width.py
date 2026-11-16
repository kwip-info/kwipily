"""Declared prepared inputs are independent of the legacy two-moons fixture."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from runner.dataflow_cli import run_dataflow
from runner.engine import GraphModel, export_source, validate
from runner.prepared_data import load_prepared, load_test_partition

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ['a', 'b', 'c']


def model_graph(width=3):
    return {'schemaVersion': 2, 'name': 'Prepared width contract', 'nodes': [
        {'id': 'input', 'type': 'input', 'name': '', 'x': 0, 'y': 0, 'config': {'features': width}},
        {'id': 'head', 'type': 'linear', 'name': '', 'x': 1, 'y': 0, 'config': {'in_features': width, 'out_features': 2, 'bias': True}},
        {'id': 'output', 'type': 'output', 'name': '', 'x': 2, 'y': 0, 'config': {}}], 'edges': [
        {'id': 'e1', 'source': 'input', 'sourcePort': 'out', 'target': 'head', 'targetPort': 'in'},
        {'id': 'e2', 'source': 'head', 'sourcePort': 'out', 'target': 'output', 'targetPort': 'in'}]}


def prepared(tmp_path, protected=False):
    source = tmp_path / 'source.csv'
    source.write_text(('observed_at,' if protected else '') + 'a,b,c,label\n' + ''.join((f'2026-01-{i+1:02d}T00:00:00Z,' if protected else '') + f'{i},{i+50},{i+100},{i%2}\n' for i in range(20)))
    graph = {'schemaVersion': 1, 'nodes': [
        {'id': 'source', 'operator': 'kwip.data.sourceCSV', 'config': {}},
        {'id': 'cast', 'operator': 'kwip.data.columns', 'config': {'rows': [dict(source=c,included=True,outputName=c,targetType=t) for c,t in [('observed_at','keep'),('a','float64'),('b','float64'),('c','float64'),('label','int64')] if protected or c!='observed_at'], 'castFailure':'error','trimText':False,'newColumns':'drop','missingPolicy':'error'}},
        {'id': 'split', 'operator': 'kwip.data.partition', 'config': {'strategy':'random','seed':42,'validationPercent':20,'testPercent':20,'stratifyColumn':'label'}}], 'edges': [
        {'from': {'node': 'source', 'port': 'table'}, 'to': {'node': 'cast', 'port': 'table'}},
        {'from': {'node': 'cast', 'port': 'table'}, 'to': {'node': 'split', 'port': 'table'}}]}
    if protected:
        graph['nodes'][-1] = {'id': 'split', 'operator': 'kwip.data.partition', 'config': {'strategy':'time',
            'timeColumn': 'observed_at', 'timeEncoding': 'iso8601Offset', 'trainingEnd': '2026-01-12T00:00:00Z',
            'validationEnd': '2026-01-16T00:00:00Z', 'embargoMilliseconds': 0, 'entityColumns': [], 'entityPolicy': 'allowAcrossTime'}}
    graph_file = tmp_path / 'preparation.json'; graph_file.write_text(json.dumps(graph))
    output = tmp_path / 'prepared'
    run_dataflow(graph_file, {'source': str(source)}, output, lambda _: None)
    return output, source


def load(path, features=FEATURES, **kwargs):
    return load_prepared(path / 'manifest.json', 'split', 'training', 'split', 'validation', features, 'label', **kwargs)


def test_prepared_numeric_columns_exactly_fill_declared_shape(tmp_path):
    path, _ = prepared(tmp_path)
    data, provenance, _ = load(path, sample_shape=[3])
    assert data[0].shape == (12, 3) and data[2].shape == (4, 3)
    assert data[0].dtype == torch.float32
    assert torch.equal(data[0][:, 1] - data[0][:, 0], torch.full((12,), 50.))
    assert torch.equal(data[0][:, 2] - data[0][:, 0], torch.full((12,), 100.))
    assert provenance['featureColumns'] == FEATURES
    shaped, _, _ = load(path, sample_shape=[1, 3])
    assert shaped[0].shape == (12, 1, 3)
    with pytest.raises(ValueError, match='two'):
        load(path)  # No declared shape retains the historic two-column default.


@pytest.mark.parametrize('features,shape', [(['a', 'a', 'c'], [3]), (['a', '', 'c'], [3]), ([], [3]),
    ([f'c{i}' for i in range(129)], [129]),
    (['a', 'b', 'label'], [3]), (FEATURES, [2]), (FEATURES, [0]),
    (FEATURES, [True, 3]), (FEATURES, [100_000_001]), (FEATURES, [1]*8)])
def test_bad_mapping_rejected_before_artifact_or_tensor_loading(tmp_path, monkeypatch, features, shape):
    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid mapping reached artifact loading')
    monkeypatch.setattr('runner.prepared_data.completed_inventory', forbidden)
    with pytest.raises(ValueError):
        load(tmp_path, features, sample_shape=shape)


def test_declared_model_context_keeps_legacy_and_output_guards():
    with pytest.raises(ValueError, match='exactly 2'):
        validate(model_graph())
    with pytest.raises(ValueError, match='exactly 2'):
        GraphModel(model_graph())
    _, widths = validate(model_graph(), bundled_dataset=False)
    assert widths['input'] == 3
    model = GraphModel(model_graph(), bundled_dataset=False)
    namespace = {}; exec(export_source(model), namespace)
    transferred = namespace['Model'](); transferred.load_state_dict(model.state_dict())
    x = torch.tensor([[2., 1., 0.], [9., 0., 1.]])
    torch.testing.assert_close(model(x), transferred(x))
    bad = model_graph(); bad['nodes'][1]['config']['out_features'] = 3
    with pytest.raises(ValueError):
        validate(bad, bundled_dataset=False)


def test_cli_prepared_width_three_trains_previews_and_exports_but_csv_stays_two(tmp_path):
    path, source = prepared(tmp_path)
    graph = tmp_path / 'model.json'; graph.write_text(json.dumps(model_graph()))
    raw = (path / 'manifest.json').read_bytes(); manifest = json.loads(raw)
    base = [sys.executable, str(ROOT / 'runner/main.py'), '--graph', str(graph), '--device', 'cpu']
    selection = ['--prepared-dataflow', str(path / 'manifest.json'), '--prepared-training-node', 'split',
        '--prepared-training-port', 'training', '--prepared-validation-node', 'split', '--prepared-validation-port', 'validation',
        '--prepared-manifest-sha256', hashlib.sha256(raw).hexdigest(), '--prepared-graph-digest', manifest['graphDigest'],
        '--feature-columns-json', json.dumps(FEATURES), '--target-column', 'label']
    trained = tmp_path / 'trained'
    result = subprocess.run(base + selection + ['--output', str(trained), '--epochs', '1'], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    saved = json.loads((trained / 'manifest.json').read_text())
    assert saved['provenance']['featureColumns'] == FEATURES
    assert saved['evaluation']['count'] == 4
    result = subprocess.run(base + selection + ['--preview', '--checkpoint', str(trained / 'checkpoint.pt'), '--output', str(tmp_path / 'preview')], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run(base + ['--export-code'], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    legacy = subprocess.run(base + ['--dataset-csv', str(source), '--feature-columns-json', json.dumps(FEATURES), '--target-column', 'label', '--output', str(tmp_path / 'legacy')], capture_output=True, text=True)
    assert legacy.returncode == 1 and 'two distinct' in legacy.stdout


def test_protected_test_partition_uses_the_same_declared_numeric_width(tmp_path):
    path, _ = prepared(tmp_path, protected=True)
    raw = (path / 'manifest.json').read_bytes(); pin = hashlib.sha256(raw).hexdigest()
    _, provenance, _ = load(path, sample_shape=[3], expected_manifest_sha256=pin)
    (x, y), evidence, _ = load_test_partition(path / 'manifest.json', 'split', 'test', provenance,
        expected_manifest_sha256=pin, sample_shape=[3])
    assert x.shape == (4, 3) and y.tolist() == [0, 1, 0, 1]
    assert x.tolist() == [[16., 66., 116.], [17., 67., 117.], [18., 68., 118.], [19., 69., 119.]]
    assert evidence['featureColumns'] == FEATURES
    with pytest.raises(ValueError, match='sample shape'):
        load_test_partition(path / 'manifest.json', 'split', 'test', provenance,
            expected_manifest_sha256=pin, sample_shape=[2])


def test_frozen_prepared_three_features_preserve_inference_export_and_reject_weights_mismatch(tmp_path):
    from runner.engine import run, REGISTRY
    from runner.frozen_checkpoint import load_frozen
    path, _ = prepared(tmp_path)
    loaded = load(path, sample_shape=[3])
    output = tmp_path / 'trained'
    run(model_graph(), output, epochs=1, device='cpu', prepared=loaded)
    def sha(file):
        return hashlib.sha256(file.read_bytes()).hexdigest()
    def frozen():
        return load_frozen(output / 'checkpoint.pt', expected_manifest_sha256=sha(output / 'manifest.json'),
            expected_checkpoint_sha256=sha(output / 'checkpoint.pt'), registry_path=REGISTRY,
            expected_registry_sha256=sha(REGISTRY))
    saved = torch.load(output / 'checkpoint.pt', weights_only=True, map_location='cpu')
    checkpoint = frozen()
    assert checkpoint.input_contract == {'dtype': 'float32', 'sampleShape': [3]}
    x = torch.tensor([[2., 1., 0.], [9., 0., 1.]])
    expected = x @ saved['modelState']['layers.head.weight'].T + saved['modelState']['layers.head.bias']
    torch.testing.assert_close(checkpoint.model(x), expected)
    namespace = {}; exec((output / 'model.py').read_text(), namespace)
    transferred = namespace['Model'](); transferred.load_state_dict(saved['modelState'])
    torch.testing.assert_close(transferred(x), expected)
    saved['modelState']['layers.head.weight'] = torch.zeros(2, 2)
    torch.save(saved, output / 'checkpoint.pt')
    manifest = json.loads((output / 'manifest.json').read_text())
    manifest['checkpoint'].update(sha256=sha(output / 'checkpoint.pt'), bytes=(output / 'checkpoint.pt').stat().st_size)
    (output / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='shape|incompatible'):
        frozen()
