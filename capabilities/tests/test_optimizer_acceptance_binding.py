"""Optimizer credit requires measured updates/state, qualified recipes and per-mode acceptance."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[2]
RECEIPT = 'docs/production/TORCH_STANDARD_OPTIMIZERS_ACCEPTANCE_20261008.md'
MODES = ['configured-step', 'parameter-ownership', 'state-roundtrip', 'resume']
MEMBERS = {
    'SGD': ('optimizer_sgd', ['sgd-momentum-steps'], 'prepared-sgd-classifier'),
    'Adam': ('optimizer_adam', ['adam-running-averages'], 'prepared-adam-classifier'),
    'AdamW': ('optimizer_adamw', ['adamw-separate-decay'], 'prepared-adamw-classifier'),
    'RMSprop': ('optimizer_rmsprop', ['rmsprop-squared-memory'], 'prepared-rmsprop-classifier'),
    'Adagrad': ('optimizer_adagrad', ['adagrad-accumulated-steps'], 'prepared-adagrad-classifier'),
}

spec = importlib.util.spec_from_file_location('optimizer_binding_coverage', ROOT / 'scripts/coverage_report.py')
coverage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coverage)

@pytest.fixture(params=MEMBERS)
def binding(request):
    catalog = json.loads((ROOT / 'capabilities/catalogs/v1/pytorch.json').read_text())
    kind = request.param
    unit = next(u for u in catalog['capabilities'] if u['id'] == 'pytorch.torch.optim.' + kind)
    return catalog, unit, kind

def complete_binding(binding):
    catalog, unit, kind = binding
    # Fail against actual catalog semantics before looking for any new receipt.
    assert unit['implementation_status'] == 'delivered'
    assert unit['mode_scope_status'] == 'reviewed'
    assert unit['configuration_contract'] == 'verified'
    assert unit['id'] in coverage.report_catalog(catalog, ROOT)['complete_ids']
    return catalog, unit, kind

def changed_unit(catalog, unit):
    changed = copy.deepcopy(catalog)
    target = next(u for u in changed['capabilities'] if u['id'] == unit['id'])
    return changed, target

def test_reviewed_identity_modes_devices_lessons_examples_and_denominator(binding):
    catalog, unit, kind = complete_binding(binding)
    result = coverage.report_catalog(catalog, ROOT)
    assert len(catalog['capabilities']) == result['capability_count'] == 88
    assert result['complete_count'] >= 54
    assert result['minimum_complete_units'] == 80
    assert not result['release_gate_passed']
    assert unit['family'] == 'optimizer' and unit['canonical_symbol'] == 'torch.optim.' + kind
    assert unit['mode_scope'] == MODES
    assert unit['backend_matrix']['status'] == 'verified'
    assert set(unit['backend_matrix']['cases']) == {'cpu-float32', 'mps-float32'}
    assert unit['backend_matrix']['reference'] == RECEIPT
    component_id, lesson_tasks, example_id = MEMBERS[kind]
    tasks = ['assessment.' + task for task in lesson_tasks]
    assert unit['lesson_id'] == tasks[0] and unit['lesson_ids'] == tasks
    assert unit['example_id'] == example_id
    bundle = json.loads((ROOT / 'components/optimizers/bundle.json').read_text())
    entry = next(e for e in bundle['components'] if e['definition']['id'] == component_id)
    assert entry['learning']['lessonIDs'] == tasks
    assert entry['learning']['exampleID'] == unit['example_id']
    assert entry['presentation']['icon'] == component_id + '-v2'

def test_actual_frozen_receipt_binds_ten_dimensions_and_each_reviewed_mode(binding):
    _, unit, _ = complete_binding(binding)
    path = ROOT / RECEIPT
    assert path.is_file(), 'Frozen measured acceptance required; placeholders cannot earn credit'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(coverage.REQUIRED_EVIDENCE) == 10
    for dimension in coverage.REQUIRED_EVIDENCE:
        for mode in MODES:
            rows = [r for r in unit['evidence'].get(dimension, []) if r['path'] == RECEIPT and r.get('scope') == unit['id'] + ':' + mode]
            assert rows, (dimension, mode)
            assert all(r['status'] == 'verified' and r['sha256'] == digest for r in rows)

def test_stale_configuration_ports_worker_assessment_art_and_admission_withhold_credit(binding):
    catalog, unit, kind = complete_binding(binding)
    component_id = MEMBERS[kind][0]
    # Select critical independent seams rather than repeat every source in each unit.
    dependencies = [
        ('configuration', 'components/optimizers/bundle.json'),
        ('configuration', 'Sources/KwipMLCore/OptimizerConfiguration.swift'),
        ('execution', 'runner/optimizer_configuration.py'),
        ('execution', 'runner/optimizer_adapters_generated.py'),
        ('execution', 'runner/tests/test_optimizer_runtime.py'),
        ('execution', 'Sources/KwipMLApp/WorkflowModelTrainingExecution.swift'),
        ('validation', 'Sources/KwipMLCore/OptimizerRuntimeCapabilities.swift'),
        ('lesson', 'Sources/KwipMLCore/OptimizerLessonReference.swift'),
        ('lesson', 'runner/learning_optimizers.py'),
        ('lesson', 'Sources/KwipMLApp/LessonAudienceGuide.swift'),
        ('lesson', 'Tests/KwipMLAppTests/OptimizerWindowDiscoveryTests.swift'),
        ('example', 'Sources/KwipMLCore/OptimizerWorkflowExample.swift'),
        ('ui', 'Sources/KwipMLApp/Resources/ComponentIcons/' + component_id + '-v2.png'),
        ('ui', 'docs/design/component-icons-v2/optimizer-provenance-20261008.json'),
        ('persistence', 'runner/checkpoint_lifecycle.py'),
        ('persistence', 'Sources/KwipMLApp/WorkflowArchiveContracts.swift'),
        ('persistence', 'Tests/KwipMLAppTests/OptimizerArchiveProjectionTests.swift'),
        ('diagnostics', 'Sources/KwipMLApp/OptimizerRecipeInspectorView.swift'),
        ('failure_handling', 'Tests/KwipMLAppTests/OptimizerWindowDiscoveryTests.swift'),
        ('export', 'docs/production/TORCH_STANDARD_OPTIMIZERS_CANONICAL_20261008.md'),
    ]
    for dimension, path in dependencies:
        changed, target = changed_unit(catalog, unit)
        rows = [r for r in target['evidence'].get(dimension, []) if r['path'] == path]
        assert rows, (dimension, path)
        assert {unit['id'] + ':' + mode for mode in MODES} <= {r.get('scope') for r in rows}
        rows[0]['sha256'] = '0' * 64
        assert unit['id'] not in coverage.report_catalog(changed, ROOT)['complete_ids'], (dimension, path)

def test_every_missing_dimension_or_wrong_mode_scope_withholds_credit(binding):
    catalog, unit, _ = complete_binding(binding)
    for dimension in coverage.REQUIRED_EVIDENCE:
        for mode in MODES:
            changed, target = changed_unit(catalog, unit)
            target['evidence'][dimension] = [dict(r, scope=unit['id'] + ':unreviewed-mode') if r.get('scope') == unit['id'] + ':' + mode else r for r in target['evidence'][dimension]]
            assert unit['id'] not in coverage.report_catalog(changed, ROOT)['complete_ids'], (dimension, mode)
        changed, target = changed_unit(catalog, unit)
        del target['evidence'][dimension]
        assert unit['id'] not in coverage.report_catalog(changed, ROOT)['complete_ids'], dimension

def test_pending_contract_backend_and_unverified_evidence_cannot_earn_credit(binding):
    catalog, unit, _ = complete_binding(binding)
    for field, value in [
        ('mode_scope_status', 'pending-operation-boundary-review'),
        ('configuration_contract', 'pending'),
        ('backend_matrix', {'status': 'pending', 'cases': ['cpu-float32', 'mps-float32']}),
    ]:
        changed, target = changed_unit(catalog, unit)
        target[field] = value
        assert unit['id'] not in coverage.report_catalog(changed, ROOT)['complete_ids'], field
    for dimension in coverage.REQUIRED_EVIDENCE:
        changed, target = changed_unit(catalog, unit)
        for row in target['evidence'][dimension]:
            row['status'] = 'present'
        assert unit['id'] not in coverage.report_catalog(changed, ROOT)['complete_ids'], dimension

def test_stale_receipt_and_missing_mode_execution_proof_refuse(binding):
    catalog, unit, _ = complete_binding(binding)
    for mode in MODES:
        changed, target = changed_unit(catalog, unit)
        rows = [r for r in target['evidence']['execution'] if r['path'] == RECEIPT and r.get('scope') == unit['id'] + ':' + mode]
        assert rows
        rows[0]['sha256'] = 'f' * 64
        assert unit['id'] not in coverage.report_catalog(changed, ROOT)['complete_ids']
        changed, target = changed_unit(catalog, unit)
        target['evidence']['execution'] = [r for r in target['evidence']['execution'] if r.get('scope') != unit['id'] + ':' + mode]
        assert unit['id'] not in coverage.report_catalog(changed, ROOT)['complete_ids']
