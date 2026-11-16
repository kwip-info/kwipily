"""Exercise nested named-input validation without a development source tree."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from runner import engine
from runner.named_prepared_data import load_named_prepared
from runner.tests.test_sequence_objective import recipe
from runner.tests.test_sequence_training import fixture


def test_named_preparation_uses_explicit_registry_when_checkout_default_is_absent(tmp_path, monkeypatch):
    _, graph, target, prepared, plan = fixture(tmp_path)
    registry = engine.REGISTRY
    monkeypatch.setattr(engine, 'REGISTRY', tmp_path / 'absent-checkout/components.json')
    result = load_named_prepared(
        prepared / 'manifest.json', 'windows', 'training', 'windows', 'validation', plan, graph,
        expected_manifest_sha256=hashlib.sha256((prepared / 'manifest.json').read_bytes()).hexdigest(),
        expected_graph_digest=json.loads((prepared / 'manifest.json').read_bytes())['graphDigest'],
        objective_recipe=recipe(), target_mapping=target, registry=registry)
    assert result[0][0]['tokens'].shape[1] == 3


def test_relocated_named_training_cli_uses_bundled_registry_without_sources_directory(tmp_path):
    _, graph, target, prepared, plan = fixture(tmp_path)
    root = Path(__file__).resolve().parents[2]
    bundle = tmp_path / 'isolated-runtime'
    shutil.copytree(root / 'runner', bundle / 'runner',
                    ignore=shutil.ignore_patterns('tests', '__pycache__', '*.pyc', '*.dylib'))
    shutil.copyfile(engine.REGISTRY, bundle / 'components.json')
    assert not (bundle / 'Sources').exists()
    graph_path = tmp_path / 'model.json'
    graph_path.write_text(json.dumps(graph))
    manifest = prepared / 'manifest.json'
    output = tmp_path / 'trained'
    command = [sys.executable, '-I', '-B', str(bundle / 'runner/main.py'),
               '--registry', str(bundle / 'components.json'), '--graph', str(graph_path),
               '--output', str(output), '--epochs', '1', '--device', 'cpu', '--batch-size', '7',
               '--prepared-dataflow', str(manifest),
               '--prepared-manifest-sha256', hashlib.sha256(manifest.read_bytes()).hexdigest(),
               '--prepared-graph-digest', json.loads(manifest.read_bytes())['graphDigest'],
               '--prepared-training-node', 'windows', '--prepared-training-port', 'training',
               '--prepared-validation-node', 'windows', '--prepared-validation-port', 'validation',
               '--named-input-plan-json', json.dumps(plan), '--objective-recipe-json', json.dumps(recipe()),
               '--objective-target-mapping-json', json.dumps(target)]
    result = subprocess.run(command, cwd=bundle, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (output / 'checkpoint.pt').is_file()
    assert json.loads((output / 'manifest.json').read_bytes())['evaluation']['objective']['activeCount'] > 0
