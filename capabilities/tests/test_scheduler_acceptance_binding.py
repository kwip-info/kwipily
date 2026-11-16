"""Scheduler credit requires measured rate timing/state, qualified recipes, traces and per-mode acceptance."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[2]
RECEIPT = 'docs/production/TORCH_SCHEDULERS_ACCEPTANCE_20261008.md'
MODES = ['configured-schedule', 'step-timing', 'state-roundtrip', 'resume', 'history-rate-trace']
MEMBERS = {
    'StepLR': ('scheduler_step_lr', ['step-lr-intervals'], 'prepared-step-lr-classifier'),
    'CosineAnnealingLR': ('scheduler_cosine_annealing_lr', ['cosine-lr-curve'], 'prepared-cosine-annealing-lr-classifier'),
    'ReduceLROnPlateau': ('scheduler_reduce_on_plateau', ['plateau-lr-improvement'], 'prepared-reduce-on-plateau-classifier'),
    'OneCycleLR': ('scheduler_one_cycle_lr', ['one-cycle-lr-batches'], 'prepared-one-cycle-lr-classifier'),
    'LinearLR': ('scheduler_linear_lr', ['linear-lr-warmup'], 'prepared-linear-lr-classifier'),
}
def modes(kind):
    return MODES + ['required-metric'] if kind == 'ReduceLROnPlateau' else MODES


spec = importlib.util.spec_from_file_location('scheduler_binding_coverage', ROOT / 'scripts/coverage_report.py')
coverage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coverage)

@pytest.fixture(params=MEMBERS)
def binding(request):
    catalog = json.loads((ROOT / 'capabilities/catalogs/v1/pytorch.json').read_text())
    kind = request.param
    unit = next(u for u in catalog['capabilities'] if u['id'] == 'pytorch.torch.optim.lr_scheduler.' + kind)
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
    assert result['complete_count'] >= 59
    assert result['minimum_complete_units'] == 80
    assert unit['family'] == 'scheduler' and unit['canonical_symbol'] == 'torch.optim.lr_scheduler.' + kind
    assert unit['mode_scope'] == modes(kind)
    assert unit['backend_matrix']['status'] == 'verified'
    assert set(unit['backend_matrix']['cases']) == {'cpu-float32', 'mps-float32'}
    assert unit['backend_matrix']['reference'] == RECEIPT
    component_id, lesson_tasks, example_id = MEMBERS[kind]
    tasks = ['assessment.' + task for task in lesson_tasks]
    assert unit['lesson_id'] == tasks[0] and unit['lesson_ids'] == tasks
    assert unit['example_id'] == example_id
    bundle = json.loads((ROOT / 'components/schedulers/bundle.json').read_text())
    entry = next(e for e in bundle['components'] if e['definition']['id'] == component_id)
    assert entry['learning']['lessonIDs'] == tasks
    assert entry['learning']['exampleID'] == unit['example_id']
    assert entry['presentation']['icon'] == component_id + '-v2'

def test_actual_frozen_receipt_binds_ten_dimensions_and_each_reviewed_mode(binding):
    _, unit, kind = complete_binding(binding)
    path = ROOT / RECEIPT
    assert path.is_file(), 'Frozen measured acceptance required; placeholders cannot earn credit'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(coverage.REQUIRED_EVIDENCE) == 10
    for dimension in coverage.REQUIRED_EVIDENCE:
        for mode in modes(kind):
            rows = [r for r in unit['evidence'].get(dimension, []) if r['path'] == RECEIPT and r.get('scope') == unit['id'] + ':' + mode]
            assert rows, (dimension, mode)
            assert all(r['status'] == 'verified' and r['sha256'] == digest for r in rows)

def test_stale_configuration_ports_worker_assessment_art_and_admission_withhold_credit(binding):
    catalog, unit, kind = complete_binding(binding)
    component_id = MEMBERS[kind][0]
    # Select critical independent seams rather than repeat every source in each unit.
    dependencies = [
        ('configuration', 'components/schedulers/bundle.json'),
        ('configuration', 'Sources/KwipMLCore/SchedulerConfiguration.swift'),
        ('execution', 'runner/scheduler_configuration.py'),
        ('execution', 'runner/scheduler_adapters_generated.py'),
        ('execution', 'runner/tests/test_scheduler_runtime.py'),
        ('execution', 'Sources/KwipMLApp/WorkflowModelTrainingExecution.swift'),
        ('execution', 'docs/production/TORCH_SCHEDULERS_RUNTIME_20261008.md'),
        ('validation', 'Sources/KwipMLCore/SchedulerRuntimeCapabilities.swift'),
        ('validation', 'Sources/KwipMLCore/WorkflowCompiler.swift'),
        ('lesson', 'Sources/KwipMLCore/SchedulerLessonReference.swift'),
        ('lesson', 'runner/learning_schedulers.py'),
        ('lesson', 'Sources/KwipMLApp/LessonAudienceGuide.swift'),
        ('lesson', 'Tests/KwipMLAppTests/SchedulerLearningIntegrationTests.swift'),
        ('example', 'Sources/KwipMLCore/SchedulerWorkflowExample.swift'),
        ('ui', 'Sources/KwipMLApp/Resources/ComponentIcons/' + component_id + '-v2.png'),
        ('ui', 'docs/design/component-icons-v2/scheduler-provenance-20261008.json'),
        ('ui', 'docs/production/TORCH_SCHEDULERS_NATIVE_20261008.md'),
        ('persistence', 'runner/checkpoint_lifecycle.py'),
        ('persistence', 'Sources/KwipMLApp/WorkflowArchiveContracts.swift'),
        ('persistence', 'Tests/KwipMLAppTests/SchedulerArchiveContractTests.swift'),
        ('persistence', 'Tests/KwipMLAppTests/SchedulerHistoryExportTests.swift'),
        ('diagnostics', 'Sources/KwipMLApp/OptimizerRecipeInspectorView.swift'),
        ('diagnostics', 'Sources/KwipMLApp/WorkflowLifecycleVerifier.swift'),
        ('failure_handling', 'Tests/KwipMLAppTests/SchedulerWorkflowArtifactTests.swift'),
        ('export', 'runner/tests/test_scheduler_portable_trace.py'),
        ('export', 'Sources/KwipMLApp/WorkflowHistoryReader.swift'),
        ('export', 'Sources/KwipMLApp/Resources/inference.py'),
        ('export', 'docs/production/TORCH_SCHEDULERS_CANONICAL_20261008.md'),
        ('export', 'docs/production/TORCH_SCHEDULERS_NATIVE_EXPORT_20261008.md'),
    ]
    for dimension, path in dependencies:
        changed, target = changed_unit(catalog, unit)
        rows = [r for r in target['evidence'].get(dimension, []) if r['path'] == path]
        assert rows, (dimension, path)
        assert {unit['id'] + ':' + mode for mode in modes(kind)} <= {r.get('scope') for r in rows}
        rows[0]['sha256'] = '0' * 64
        assert unit['id'] not in coverage.report_catalog(changed, ROOT)['complete_ids'], (dimension, path)

def test_every_missing_dimension_or_wrong_mode_scope_withholds_credit(binding):
    catalog, unit, kind = complete_binding(binding)
    for dimension in coverage.REQUIRED_EVIDENCE:
        for mode in modes(kind):
            changed, target = changed_unit(catalog, unit)
            target['evidence'][dimension] = [dict(r, scope=unit['id'] + ':unreviewed-mode') if r.get('scope') == unit['id'] + ':' + mode else r for r in target['evidence'][dimension]]
            assert unit['id'] not in coverage.report_catalog(changed, ROOT)['complete_ids'], (dimension, mode)
        changed, target = changed_unit(catalog, unit)
        del target['evidence'][dimension]
        assert unit['id'] not in coverage.report_catalog(changed, ROOT)['complete_ids'], dimension

def test_pending_contract_backend_and_unverified_evidence_cannot_earn_credit(binding):
    catalog, unit, kind = complete_binding(binding)
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
    catalog, unit, kind = complete_binding(binding)
    for mode in modes(kind):
        changed, target = changed_unit(catalog, unit)
        rows = [r for r in target['evidence']['execution'] if r['path'] == RECEIPT and r.get('scope') == unit['id'] + ':' + mode]
        assert rows
        rows[0]['sha256'] = 'f' * 64
        assert unit['id'] not in coverage.report_catalog(changed, ROOT)['complete_ids']
        changed, target = changed_unit(catalog, unit)
        target['evidence']['execution'] = [r for r in target['evidence']['execution'] if r.get('scope') != unit['id'] + ':' + mode]
        assert unit['id'] not in coverage.report_catalog(changed, ROOT)['complete_ids']
