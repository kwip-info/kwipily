"""Preparation plans capture one validated graph, never source grants or Python."""
import ast
import hashlib
import json
import subprocess
import sys

import pytest

from runner import dataflow
from runner.tests.test_columns import config,item
from runner.preparation_plan import (
    PreparationPlan, build_plan, derive_preparation_code, execute_plan, validate_plan,
)


def graph():
    return {'schemaVersion': 1, 'nodes': [
        {'id': 'source', 'operator': 'kwip.data.sourceCSV', 'config': {'separator': ';'}},
        {'id': 'selected', 'operator': 'kwip.data.columns',
         'config': config([item('name','label')],newColumns='drop')},
    ], 'edges': [{'from': {'node': 'source', 'port': 'table'},
                 'to': {'node': 'selected', 'port': 'table'}}]}


def plan(**kwargs):
    return build_plan(graph(), engine='polars', engine_version='1.35.2', **kwargs)


def test_captured_reference_kernel_choices_do_not_change_when_new_fast_paths_arrive():
    from runner.workload_policy import WorkloadPolicy
    captured=plan(workload_policy=WorkloadPolicy('balanced')).to_document()
    captured['kernelPlan']['schemaVersion']=1
    captured['kernelPlan'].pop('fusion');captured['kernelPlan'].pop('workBoundaries') # Historical plans have no v4 physical policy.
    captured['kernelPlan']['nodes']['selected']={'kernelID':'reference.columns.v1','effect':'pure','parallel':False,'reason':'referenceSemantics'}
    assert validate_plan(captured).to_document()==captured
    captured['kernelPlan']['nodes']['selected']['parallel']=True
    with pytest.raises(ValueError):validate_plan(captured)


def test_snapshot_is_immutable_canonical_and_binds_registry_engine_and_labels():
    original = graph()
    labels = {'selected': 'Keep names'}
    captured = build_plan(original, node_labels=labels, engine='polars', engine_version='1.35.2')
    before = captured.canonical_json
    original['nodes'][0]['config']['separator'] = '|'
    labels['selected'] = 'Changed'
    document = captured.to_document()
    document['graph']['nodes'][0]['config']['separator'] = '\t'
    assert captured.canonical_json == before
    assert captured.graph_digest == dataflow.digest(graph())
    assert captured.plan_digest == hashlib.sha256(before.encode()).hexdigest()
    assert captured.plan_digest != plan().plan_digest
    assert captured.plan_digest != build_plan(graph(), engine='polars', engine_version='1.36.0').plan_digest
    assert captured.to_document()['registryDigest'] == dataflow.digest(dataflow.REGISTRY)
    assert captured.to_document()['sources'] == [{'nodeID': 'source', 'adapter': 'local.csv', 'outputKind': 'typed.table'}]
    assert validate_plan(json.loads(before)) == captured
    assert PreparationPlan.from_json(before) == captured


def test_json_key_order_does_not_change_identity():
    source = graph()
    reordered = json.loads(json.dumps(source, sort_keys=True))
    assert build_plan(source, engine='polars', engine_version='1.35.2') == build_plan(reordered, engine='polars', engine_version='1.35.2')


@pytest.mark.parametrize('mutation', [
    lambda value: value['graph']['nodes'][0]['config'].update(path='/private/data.csv'),
    lambda value: value.update(sourceBindings={'source': '/private/data.csv'}),
    lambda value: value.update(scope='wholeWorkflow'),
    lambda value: value.update(registryDigest='0' * 64),
    lambda value: value['sources'][0].update(adapter='remote.oidc'),
    lambda value: value['engine'].update(id='python'),
    lambda value: value['operatorVersions'].update({'kwip.data.sourceCSV': '99.0.0'}),
    lambda value: value['nodeLabels'].update(unknown='Not in graph'),
    lambda value: value.update(graphDigest='0' * 64),
])
def test_tampered_or_unsupported_plan_is_rejected_before_execution(mutation):
    document = plan().to_document()
    mutation(document)
    with pytest.raises(ValueError):
        validate_plan(document)


@pytest.mark.parametrize('version', ['', '1\nmalicious', 'x' * 65, None, 1])
def test_engine_version_must_be_explicit_bounded_identity(version):
    with pytest.raises(ValueError, match='version'):
        build_plan(graph(), engine='polars', engine_version=version)


def test_generated_python_safe_literals_labels_and_current_recorded_identity(tmp_path):
    malicious = 'Names\n__import__("pathlib").Path("OWNED").touch() #'
    captured = plan(node_labels={'selected': malicious})
    code = derive_preparation_code(captured)
    tree = ast.parse(code)
    assert captured.plan_digest in code and captured.graph_digest in code
    assert 'Data preparation only' in code
    assert 'Polars' in code and 'typed.table' in code
    assert 'columns' in code and 'Keep' not in code
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ('eval', 'exec') for node in ast.walk(tree))
    script = tmp_path / 'preparation.py'
    script.write_text(code)
    # A display-only script has no source grants and cannot execute on import.
    namespace = {'__name__': 'inspect_generated'}
    exec(compile(code, str(script), 'exec'), namespace)
    assert namespace['PLAN'] == captured.to_document()
    assert not (tmp_path / 'OWNED').exists()


def test_execute_forwards_authority_callbacks_and_separate_bindings(monkeypatch):
    captured = plan(node_labels={'selected': 'Names'})
    monkeypatch.setattr('runner.preparation_plan._installed_engine_version', lambda engine: '1.35.2')
    calls = []
    monkeypatch.setattr(dataflow, 'execute', lambda *args, **kwargs: calls.append((args, kwargs)) or 'result')
    source = {'source': '/explicit/source.csv'}
    destinations = {}
    check = lambda: None
    callback = lambda node, ports: None
    assert execute_plan(captured, source, destination_bindings=destinations,
                        check_cancelled=check, on_node=callback, retain_results=False) == 'result'
    args, kwargs = calls[0]
    assert args == (graph(), source)
    assert kwargs['engine'] == 'polars'
    assert kwargs['destination_bindings'] is destinations
    assert kwargs['check_cancelled'] is check and kwargs['on_node'] is callback
    assert kwargs['node_labels'] == {'selected': 'Names'} and kwargs['retain_results'] is False
    assert '/explicit/source.csv' not in captured.canonical_json


def test_engine_version_mismatch_refuses_before_source_access(monkeypatch):
    monkeypatch.setattr('runner.preparation_plan._installed_engine_version', lambda engine: '1.99.0')
    monkeypatch.setattr(dataflow, 'execute', lambda *a, **k: pytest.fail('must not read source'))
    with pytest.raises(ValueError, match='version'):
        execute_plan(plan(), {'source': '/missing/source.csv'})


def test_runtime_registry_change_refuses_recorded_plan(monkeypatch):
    captured = plan()
    monkeypatch.setattr(dataflow, 'REGISTRY', {'changed': True})
    with pytest.raises(ValueError, match='registry'):
        execute_plan(captured, {'source': '/missing/source.csv'})


def test_duplicate_json_metadata_is_rejected():
    document = plan().canonical_json
    duplicate = document.replace('"scope":"dataPreparation"', '"scope":"model","scope":"dataPreparation"')
    with pytest.raises(ValueError, match='JSON'):
        PreparationPlan.from_json(duplicate)


def test_invalid_unicode_and_nonfinite_values_are_rejected():
    with pytest.raises(ValueError, match='UTF-8'):
        PreparationPlan.from_json('\ud800')
    invalid = graph()
    invalid['nodes'][0]['config']['unknown'] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        build_plan(invalid, engine='polars', engine_version='2.0.0')


def test_cancellation_precedes_runtime_resolution(monkeypatch):
    monkeypatch.setattr('runner.preparation_plan._installed_engine_version',
                        lambda engine: pytest.fail('cancelled request resolved runtime'))
    def cancelled():
        raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError, match='cancelled'):
        execute_plan(plan(), {'source': '/missing'}, check_cancelled=cancelled)


def test_execution_cannot_replace_captured_authority(monkeypatch):
    monkeypatch.setattr('runner.preparation_plan._installed_engine_version', lambda engine: '1.35.2')
    with pytest.raises(ValueError, match='override'):
        execute_plan(plan(), {'source': '/missing'}, engine='disk')
    with pytest.raises(ValueError, match='override'):
        execute_plan(plan(), {'source': '/missing'}, node_labels={'source': 'Changed'})


@pytest.mark.parametrize('engine,version', [('disk', '1'), ('polars', '2.0.0')])
def test_derived_script_replays_real_preparation_outside_checkout(tmp_path, engine, version):
    import os
    import shutil
    from runner.dataflow_cli import PORTABLE_DEPENDENCIES
    if engine == 'polars':
        pytest.importorskip('polars')
    runtime = tmp_path / 'portable'
    runtime.mkdir()
    root = __import__('pathlib').Path(__file__).parents[1]
    for name in (*PORTABLE_DEPENDENCIES, 'preparation_plan.py'):
        shutil.copyfile(root / name, runtime / name)
    captured = build_plan(graph(), engine=engine, engine_version=version)
    script = runtime / 'prepare.py'
    script.write_text(derive_preparation_code(captured))
    source = tmp_path / 'input.csv'
    source.write_text('name;value\nAlice;1\nBob;2\n')
    bindings = tmp_path / 'sources.json'
    bindings.write_text(json.dumps({'source': str(source)}))
    unrelated = tmp_path / 'unrelated'
    unrelated.mkdir()
    environment = dict(os.environ)
    environment.pop('PYTHONPATH', None)
    environment.pop('PYTHONHOME', None)
    environment['PYTHONNOUSERSITE'] = '1'
    result = subprocess.run([sys.executable, str(script), '--source-bindings', str(bindings)],
                            cwd=unrelated, env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr + result.stdout
    events = [json.loads(line) for line in result.stdout.splitlines()]
    selected = next(event for event in events if event.get('nodeID') == 'selected')
    assert selected['outputs']['table']['preview']['rows'] == [['Alice'], ['Bob']]
    assert selected['outputs']['table']['preview']['columns'] == [{'name': 'label', 'dtype': 'string', 'nullable': False}]
    assert events[-1] == {'event': 'preparationPlanCompleted', 'planDigest': captured.plan_digest,
                          'graphDigest': captured.graph_digest}
