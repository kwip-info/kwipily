import json
from unittest.mock import patch
import pytest
from runner.dataflow_cli import run_dataflow
from runner.tests.test_dataflow_cli import ROOT, invoke


def run(output, emit, **kwargs):
    return run_dataflow(ROOT/'examples/dataflow/scaler.json',
                        {'source': str(ROOT/'examples/dataflow/observations.csv')}, output, emit, **kwargs)


@pytest.mark.parametrize('failure', [KeyboardInterrupt, RuntimeError])
def test_explicit_retention_preserves_interrupted_node_bytes_without_manifest(tmp_path, failure):
    captured = {}
    def interrupt(event):
        if event['event'] == 'dataflowPreview':
            stage, = tmp_path.glob('.dataflow-*')
            captured.update({p.name: p.read_bytes() for p in stage.iterdir() if p.is_file()})
            raise failure('interrupt actual node staging')
    with pytest.raises(failure):
        run(tmp_path/'artifacts', interrupt, retain_partial=True)
    assert captured and 'node-000.json' in captured
    stage, = tmp_path.glob('.dataflow-*')
    assert {p.name: p.read_bytes() for p in stage.iterdir()} == captured
    assert not list(stage.glob('.row-scratch-*'))
    assert not (stage/'manifest.json').exists()
    assert not (tmp_path/'artifacts').exists()


def test_retention_removes_staged_success_manifest_if_publication_fails(tmp_path):
    with patch('runner.dataflow_cli.os.replace', side_effect=KeyboardInterrupt('before rename')):
        with pytest.raises(KeyboardInterrupt):
            run(tmp_path/'artifacts', lambda _: None, retain_partial=True)
    stage, = tmp_path.glob('.dataflow-*')
    assert (stage/'node-000.json').is_file()
    assert not (stage/'manifest.json').exists()
    assert not (tmp_path/'artifacts').exists()


def test_default_still_cleans_interrupted_staging(tmp_path):
    def interrupt(event):
        if event['event'] == 'dataflowPreview': raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt): run(tmp_path/'artifacts', interrupt)
    assert not list(tmp_path.glob('.dataflow-*'))


def test_retention_success_publishes_normally_without_partial_directory(tmp_path):
    manifest = run(tmp_path/'artifacts', lambda _: None, retain_partial=True)
    assert manifest['status'] == 'completed'
    assert json.loads((tmp_path/'artifacts/manifest.json').read_bytes()) == manifest
    assert not list(tmp_path.glob('.dataflow-*'))


def test_retention_invalidates_manifest_after_rename_if_completion_delivery_is_interrupted(tmp_path):
    output = tmp_path/'artifacts'
    captured = {}
    def interrupt(event):
        if event['event'] == 'dataflowCompleted':
            captured.update({p.name: p.read_bytes() for p in output.iterdir() if p.name != 'manifest.json'})
            raise KeyboardInterrupt('after rename, before completion acknowledgement')
    with pytest.raises(KeyboardInterrupt): run(output, interrupt, retain_partial=True)
    assert captured
    assert {p.name: p.read_bytes() for p in output.iterdir()} == captured
    assert not (output/'manifest.json').exists()


@pytest.mark.parametrize('args', [[], ['--capabilities'], ['--execute-dataflow', 'unused', '--capabilities']])
def test_cli_retention_requires_exclusive_dataflow_execution(args):
    result, events = invoke('--retain-partial-dataflow', *args)
    assert result.returncode != 0
    assert events[-1]['event'] == 'error'
    assert 'requires --execute-dataflow' in events[-1]['message']


def test_cli_accepts_retention_for_real_preparation(tmp_path):
    bindings = tmp_path/'bindings.json'
    bindings.write_text(json.dumps({'source': str(ROOT/'examples/dataflow/observations.csv')}))
    result, events = invoke('--execute-dataflow', ROOT/'examples/dataflow/scaler.json',
                            '--source-bindings', bindings, '--output', tmp_path/'artifacts', '--retain-partial-dataflow')
    assert result.returncode == 0, result.stdout + result.stderr
    assert events[-1]['event'] == 'dataflowCompleted'
