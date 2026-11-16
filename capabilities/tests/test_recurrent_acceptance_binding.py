"""Recurrent credit requires all four original modes, independent evidence and frozen sources."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[2]
RECEIPT = 'docs/production/TORCH_RECURRENT_ACCEPTANCE_20261008.md'
MODES = ['inference', 'training', 'input-gradients', 'state-roundtrip']
MEMBERS = {
    'RNN': ('rnn', 'rnn-sequence-memory'),
    'LSTM': ('lstm', 'lstm-separate-memory'),
    'GRU': ('gru', 'gru-reset-memory'),
    'RNNCell': ('rnn_cell', 'rnn-cell-step'),
    'LSTMCell': ('lstm_cell', 'lstm-cell-step'),
    'GRUCell': ('gru_cell', 'gru-cell-step'),
}
spec = importlib.util.spec_from_file_location('recurrent_binding_coverage', ROOT / 'scripts/coverage_report.py')
coverage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coverage)

@pytest.fixture(params=MEMBERS)
def binding(request):
    catalog = json.loads((ROOT / 'capabilities/catalogs/v1/pytorch.json').read_text())
    kind = request.param
    unit = next(u for u in catalog['capabilities'] if u['id'] == 'pytorch.torch.nn.' + kind)
    return catalog, unit, kind

def complete_binding(binding):
    catalog, unit, kind = binding
    # The real planned catalog must fail before any future receipt can imply credit.
    assert unit['implementation_status'] == 'delivered'
    assert unit['mode_scope_status'] == 'reviewed'
    assert unit['configuration_contract'] == 'verified'
    assert unit['id'] in coverage.report_catalog(catalog, ROOT)['complete_ids']
    return catalog, unit, kind

def changed_unit(catalog, unit):
    changed = copy.deepcopy(catalog)
    return changed, next(u for u in changed['capabilities'] if u['id'] == unit['id'])

def bound_rows(unit, dimension, path):
    rows = [r for r in unit['evidence'].get(dimension, []) if r['path'] == path]
    assert {unit['id'] + ':' + mode for mode in MODES} <= {r.get('scope') for r in rows}, (dimension, path)
    assert all(r['status'] == 'verified' and r['sha256'] == hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for r in rows)
    return rows

def test_original_modes_identity_backends_configuration_states_and_bundle(binding):
    catalog, unit, kind = complete_binding(binding)
    result = coverage.report_catalog(catalog, ROOT)
    assert len(catalog['capabilities']) == result['capability_count'] == 88
    assert result['complete_count'] >= 65 and result['minimum_complete_units'] == 80
    assert unit['canonical_symbol'] == 'torch.nn.' + kind and unit['family'] == 'recurrent'
    assert unit['mode_scope'] == MODES
    assert unit['backend_matrix']['status'] == 'verified'
    assert set(unit['backend_matrix']['cases']) == {'cpu-float32', 'mps-float32-unprojected'}
    assert unit['backend_matrix']['reference'] == RECEIPT
    op, task = MEMBERS[kind]
    assert unit['lesson_ids'] == ['assessment.' + task] and unit['lesson_id'] == unit['lesson_ids'][0]
    assert unit['example_id'] == 'prepared-' + op.replace('_', '-') + '-classifier'
    entry = next(e for e in json.loads((ROOT / 'components/recurrent/bundle.json').read_text())['components'] if e['definition']['id'] == op)
    assert entry['learning']['lessonIDs'] == unit['lesson_ids'] and entry['learning']['exampleID'] == unit['example_id']
    assert entry['presentation']['icon'] == op + '-v2'
    contract = entry['runtimeContract']
    assert contract['stateModes'] == ['zero', 'supplied']
    assert contract['inputPorts'] == (['in', 'hidden', 'cell'] if kind.startswith('LSTM') else ['in', 'hidden'])
    assert contract['outputPorts'] == (['out', 'hidden', 'cell'] if kind.startswith('LSTM') else ['out', 'hidden'])
    assert contract['statePersistence'] == 'explicitConnectionsOnly' and contract['stateLayout'] == 'observationFirst'
    assert contract['parameterOrder'] == 'nativeLayerDirectionGateOrder'
    assert contract['cellHiddenAlias'] is kind.endswith('Cell')
    assert contract['projectionPolicy'] == ('cpuOnly' if kind == 'LSTM' else 'notApplicable')
    assert contract['graphSchemaVersions'] == [2] and contract['graphDTypes'] == ['float32']

def test_frozen_acceptance_binds_ten_dimensions_every_original_mode(binding):
    _, unit, _ = complete_binding(binding)
    path = ROOT / RECEIPT
    assert path.is_file(), 'Root must freeze actual measured acceptance; planned evidence cannot earn credit'
    assert len(coverage.REQUIRED_EVIDENCE) == 10
    for dimension in coverage.REQUIRED_EVIDENCE:
        bound_rows(unit, dimension, RECEIPT)

def test_critical_actual_sources_receipts_and_original_icons_must_be_current(binding):
    catalog, unit, kind = complete_binding(binding)
    op = MEMBERS[kind][0]
    dependencies = [
        ('configuration', 'components/recurrent/bundle.json'),
        ('configuration', 'Sources/KwipMLCore/RecurrentConfiguration.swift'),
        ('configuration', 'runner/recurrent_configuration.py'),
        ('execution', 'runner/engine.py'),
        ('execution', 'runner/recurrent_adapters_generated.py'),
        ('execution', 'runner/tests/test_recurrent_runtime.py'),
        ('execution', 'docs/production/TORCH_RECURRENT_RUNTIME_20261008.md'),
        ('execution', 'docs/production/TORCH_RECURRENT_PRIOR_REPLAY_20261008.md'),
        ('execution', 'Sources/KwipMLApp/WorkflowWorkspaceSession.swift'),
        ('validation', 'Sources/KwipMLCore/GraphValidator.swift'),
        ('validation', 'Sources/KwipMLCore/ConfiguredComponentPorts.swift'),
        ('validation', 'Sources/KwipMLCore/GraphCommands.swift'),
        ('lesson', 'Sources/KwipMLCore/RecurrentLessonReference.swift'),
        ('lesson', 'runner/learning_recurrent.py'),
        ('lesson', 'Tests/KwipMLAppTests/RecurrentLearningIntegrationTests.swift'),
        ('example', 'Sources/KwipMLCore/RecurrentWorkflowExample.swift'),
        ('example', 'docs/production/TORCH_RECURRENT_CANONICAL_20261008.md'),
        ('diagnostics', 'Sources/KwipMLCore/GraphTensorPresentation.swift'),
        ('ui', 'Sources/KwipMLApp/Resources/ComponentIcons/' + op + '-v2.png'),
        ('ui', 'docs/design/component-icons-v2/recurrent-provenance-20261008.json'),
        ('ui', 'docs/production/TORCH_RECURRENT_NATIVE_20261008.md'),
        ('persistence', 'runner/checkpoint_lifecycle.py'),
        ('persistence', 'Sources/KwipMLApp/WorkflowLifecycleVerifier.swift'),
        ('failure_handling', 'Sources/KwipMLApp/CapabilityActions.swift'),
        ('failure_handling', 'Tests/KwipMLAppTests/RecurrentOwnershipTests.swift'),
        ('export', 'runner/engine.py'),
        ('export', 'Sources/KwipMLApp/Resources/inference.py'),
        ('export', 'docs/production/TORCH_RECURRENT_CANONICAL_20261008.md'),
        ('export', 'docs/production/TORCH_RECURRENT_NATIVE_EXPORT_20261008.md'),
    ]
    for dimension, path in dependencies:
        bound_rows(unit, dimension, path)
        changed, target = changed_unit(catalog, unit)
        rows = [r for r in target['evidence'][dimension] if r['path'] == path]
        rows[0]['sha256'] = '0' * 64
        assert unit['id'] not in coverage.report_catalog(changed, ROOT)['complete_ids'], (dimension, path)
    icon_path = 'Sources/KwipMLApp/Resources/ComponentIcons/' + op + '-v2.png'
    icon = (ROOT / icon_path).read_bytes()
    assert icon.startswith(b'\x89PNG\r\n\x1a\n')
    art = json.loads((ROOT / 'docs/design/component-icons-v2/recurrent-provenance-20261008.json').read_text())
    assert art['generator'] == 'image_gen' and {row['id'] for row in art['icons']} == {member[0] for member in MEMBERS.values()}
    row = next(row for row in art['icons'] if row['id'] == op)
    assert row['destinationPath'] == icon_path and row['sha256'] == hashlib.sha256(icon).hexdigest()

def test_missing_original_mode_dimension_or_unverified_evidence_withholds_credit(binding):
    catalog, unit, _ = complete_binding(binding)
    for dimension in coverage.REQUIRED_EVIDENCE:
        for mode in MODES:
            changed, target = changed_unit(catalog, unit)
            target['evidence'][dimension] = [r for r in target['evidence'][dimension] if r.get('scope') != unit['id'] + ':' + mode]
            assert unit['id'] not in coverage.report_catalog(changed, ROOT)['complete_ids'], (dimension, mode)
        changed, target = changed_unit(catalog, unit)
        for row in target['evidence'][dimension]:
            row['status'] = 'unverified'
        assert unit['id'] not in coverage.report_catalog(changed, ROOT)['complete_ids'], dimension
        changed, target = changed_unit(catalog, unit)
        del target['evidence'][dimension]
        assert unit['id'] not in coverage.report_catalog(changed, ROOT)['complete_ids'], dimension

def test_planned_pending_or_stale_acceptance_cannot_earn_credit(binding):
    catalog, unit, _ = complete_binding(binding)
    for field, value in [('mode_scope_status', 'reviewed-proposal-pending-acceptance'), ('configuration_contract', 'proposed-not-accepted'), ('backend_matrix', {'status': 'pending-acceptance', 'cases': ['cpu-float32', 'mps-float32-unprojected']})]:
        changed, target = changed_unit(catalog, unit)
        target[field] = value
        assert unit['id'] not in coverage.report_catalog(changed, ROOT)['complete_ids'], field
    for mode in MODES:
        changed, target = changed_unit(catalog, unit)
        rows = [r for r in target['evidence']['execution'] if r['path'] == RECEIPT and r.get('scope') == unit['id'] + ':' + mode]
        assert rows
        rows[0]['sha256'] = 'f' * 64
        assert unit['id'] not in coverage.report_catalog(changed, ROOT)['complete_ids']

def test_actual_runtime_canonical_prior_and_native_proof_bound_separately(binding):
    _, unit, _ = complete_binding(binding)
    for dimension, name in [('execution', 'RUNTIME'), ('execution', 'PRIOR_REPLAY'), ('example', 'CANONICAL'), ('ui', 'NATIVE'), ('export', 'NATIVE_EXPORT')]:
        bound_rows(unit, dimension, 'docs/production/TORCH_RECURRENT_' + name + '_20261008.md')
    runtime = (ROOT / 'docs/production/TORCH_RECURRENT_RUNTIME_20261008.md').read_text()
    canonical = (ROOT / 'docs/production/TORCH_RECURRENT_CANONICAL_20261008.md').read_text()
    assert '215 passed' in runtime and 'scalar' in runtime.lower() and 'every' in runtime.lower()
    assert 'all 12 cases passed' in canonical and 'not represented as native UI exports' in canonical
    prior = (ROOT / 'docs/production/TORCH_RECURRENT_PRIOR_REPLAY_20261008.md').read_text()
    assert '76' in prior and 'file' in prior.lower(), 'Actual 76-file bound prior replay must be recorded separately'
    # Proofs are distinct immutable dependencies; a host classifier proof cannot replace native export evidence.
    assert len({hashlib.sha256((ROOT / ('docs/production/TORCH_RECURRENT_' + name + '_20261008.md')).read_bytes()).hexdigest() for name in ['RUNTIME', 'PRIOR_REPLAY', 'CANONICAL', 'NATIVE', 'NATIVE_EXPORT']}) == 5

def test_prior_59_recursive_non_sha_semantics_and_frozen_receipts_preserved():
    baseline = json.loads(subprocess.check_output(['git', 'show', '9d6b199:capabilities/catalogs/v1/pytorch.json'], cwd=ROOT))
    current = json.loads((ROOT / 'capabilities/catalogs/v1/pytorch.json').read_text())
    prior = [u for u in baseline['capabilities'] if u.get('configuration_contract') == 'verified' and u.get('mode_scope_status') == 'reviewed' and u.get('backend_matrix', {}).get('status') == 'verified']
    assert len(prior) == 59
    def without_sha(value):
        if isinstance(value, dict):
            return {k: without_sha(v) for k, v in value.items() if k != 'sha256'}
        if isinstance(value, list):
            return [without_sha(v) for v in value]
        return value
    for old in prior:
        new = next(u for u in current['capabilities'] if u['id'] == old['id'])
        assert without_sha(new) == without_sha(old), old['id']
        for rows in old.get('evidence', {}).values():
            for row in rows:
                path = row['path']
                # Mutable art inventory is a source, not a frozen receipt; additions do not rewrite old assets.
                if path.startswith('docs/') and path != 'docs/design/component-icons-v2/catalog.json':
                    assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == row['sha256'], path
