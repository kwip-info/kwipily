"""Spatial patches earns credit only through measured, mode-bound bundle acceptance."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[2]
RECEIPT = 'docs/production/TORCH_SPATIAL_PATCH_ACCEPTANCE_20261008.md'
MODES = ['inference', 'training', 'input-gradients', 'state-roundtrip']
MEMBERS = {
    'unfold': ['unfold-overlapping-windows', 'unfold-padded-spaced-windows'],
    'fold': ['fold-overlap-addition', 'fold-uncovered-positions'],
}

spec = importlib.util.spec_from_file_location('spatial_patch_binding_coverage', ROOT / 'scripts/coverage_report.py')
coverage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coverage)

@pytest.fixture(params=MEMBERS)
def binding(request):
    catalog = json.loads((ROOT / 'capabilities/catalogs/v1/pytorch.json').read_text())
    kind = request.param
    unit = next(u for u in catalog['capabilities'] if u['id'] == 'pytorch.torch.nn.' + kind.title())
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
    assert result['complete_count'] >= 44
    assert result['minimum_complete_units'] == 80
    assert not result['release_gate_passed']
    assert unit['family'] == 'convolution' and unit['canonical_symbol'] == 'torch.nn.' + kind.title()
    assert unit['mode_scope'] == MODES
    assert unit['backend_matrix']['status'] == 'verified'
    assert set(unit['backend_matrix']['cases']) == {'cpu-float32', 'mps-float32'}
    assert unit['backend_matrix']['reference'] == RECEIPT
    tasks = ['assessment.' + task for task in MEMBERS[kind]]
    assert unit['lesson_id'] == tasks[0] and unit['lesson_ids'] == tasks
    assert unit['example_id'] == 'prepared-' + kind + '-classifier'
    bundle = json.loads((ROOT / 'components/spatial-patches/bundle.json').read_text())
    entry = next(e for e in bundle['components'] if e['definition']['id'] == kind)
    assert entry['learning']['lessonIDs'] == tasks
    assert entry['learning']['exampleID'] == unit['example_id']
    assert entry['presentation']['icon'] == kind + '-v2'

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
    dependencies = [
        ('configuration', 'components/spatial-patches/bundle.json'),
        ('configuration', 'scripts/component_bundles.py'),
        ('configuration', 'Sources/KwipMLCore/SpatialPatchConfiguration.swift'),
        ('execution', 'runner/spatial_patch_configuration.py'),
        ('execution', 'runner/spatial_patch_adapters_generated.py'),
        ('execution', 'runner/engine.py'),
        ('execution', 'runner/tests/test_spatial_patch_runtime.py'),
        ('validation', 'Sources/KwipMLCore/SpatialPatchCapabilityBridge.swift'),
        ('validation', 'Sources/KwipMLCore/SpatialPatchRuntimeCapabilities.swift'),
        ('validation', 'Sources/KwipMLCore/GraphValidator.swift'),
        ('validation', 'Sources/KwipMLCore/TypedConfiguration.swift'),
        ('lesson', 'Sources/KwipMLCore/SpatialPatchLearningTask.swift'),
        ('lesson', 'Sources/KwipMLCore/SpatialPatchLessonReference.swift'),
        ('lesson', 'Sources/KwipMLCore/SpatialPatchLearningSubmission.swift'),
        ('lesson', 'runner/learning_spatial_patches.py'),
        ('lesson', 'runner/learning_evidence.py'),
        ('lesson', 'runner/tests/test_learning_spatial_patches.py'),
        ('example', 'Sources/KwipMLCore/SpatialPatchWorkflowExample.swift'),
        ('example', 'Sources/KwipMLCore/PreparedWorkflowExample.swift'),
        ('ui', 'Sources/KwipMLApp/Resources/ComponentIcons/' + kind + '-v2.png'),
        ('ui', 'docs/design/component-icons-v2/spatial-patches-provenance-20261008.json'),
        ('ui', 'docs/design/component-icons-v2/catalog.json'),
        ('ui', 'Tests/KwipMLAppTests/SpatialPatchArtworkTests.swift'),
        ('ui', 'Sources/KwipMLApp/TensorLearningCanvasAdapter.swift'),
        ('ui', 'Tests/KwipMLAppTests/SpatialPatchCanvasTests.swift'),
        ('persistence', 'Tests/KwipMLAppTests/SpatialPatchLearningStoreTests.swift'),
        ('diagnostics', 'Sources/KwipMLApp/CapabilityActions.swift'),
        ('failure_handling', 'Tests/KwipMLAppTests/SpatialPatchRuntimeAdmissionTests.swift'),
        ('export', 'scripts/verify_spatial_patch_exports.py'),
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
