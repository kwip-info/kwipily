"""Credit the reviewed gradient pair only with current measured, private-safe consumer bindings."""
from capabilities.tests.artwork_inventory_compatibility import historical_artwork_entries
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[2]
BASELINE = 'ee26ff9'
CATALOG = 'capabilities/catalogs/v1/pytorch.json'
ACCEPTANCE = 'docs/production/TORCH_GRADIENT_ACCEPTANCE_20261009.md'
NATIVE = 'docs/production/TORCH_GRADIENT_NATIVE_20261009.md'
PAIR = {
    'pytorch.torch.autograd.grad': ('gradient_policy', ['requested-gradients', 'reset-gradients', 'accumulated-gradients'], ['gradient-selected-influence', 'gradient-combine-small-groups'], 'gradient-policy-v1', '944f02d2fb730ccd28052bcd57b675656524ca4bef87f9872bb697052a095661'),
    'pytorch.torch.no_grad': ('trainability', ['frozen-inference', 'explicit-trainability-transition', 'unavailable-gradient-capture'], ['trainability-keep-settings', 'trainability-next-run'], 'trainability-v1', '3e7ebe9a25da4d738a94a384e1220a0abe70d5f32158a0a17d25fd63810c6e2d'),
}
spec = importlib.util.spec_from_file_location('gradient_acceptance_coverage', ROOT / 'scripts/coverage_report.py')
coverage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coverage)


def read(path): return json.loads((ROOT / path).read_text())
def digest(path): return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
def baseline(path): return subprocess.check_output(['git', 'show', BASELINE + ':' + path], cwd=ROOT)
def without_sha(value):
    if isinstance(value, dict): return {key: without_sha(item) for key, item in value.items() if key != 'sha256'}
    if isinstance(value, list): return [without_sha(item) for item in value]
    return value


@pytest.fixture(params=PAIR)
def binding(request):
    catalog = read(CATALOG)
    unit = next(row for row in catalog['capabilities'] if row['id'] == request.param)
    return catalog, unit


def delivered(binding):
    catalog, unit = binding
    assert len(catalog['capabilities']) == 88
    assert unit['implementation_status'] == 'delivered'
    assert unit['mode_scope_status'] == 'reviewed' and unit['configuration_contract'] == 'verified'
    assert unit['backend_matrix']['status'] == 'verified'
    assert unit['backend_matrix']['reference'] == ACCEPTANCE
    assert any(case.startswith('cpu') for case in unit['backend_matrix']['cases'])
    assert any(case.startswith('mps') for case in unit['backend_matrix']['cases'])
    assert unit['id'] in coverage.report_catalog(catalog, ROOT)['complete_ids']
    return catalog, unit


def bound(unit, dimension, path):
    rows = [row for row in unit['evidence'].get(dimension, []) if row['path'] == path]
    assert {unit['id'] + ':' + mode for mode in PAIR[unit['id']][1]} <= {row.get('scope') for row in rows}, (dimension, path)
    assert all(row['status'] == 'verified' and row['sha256'] == digest(path) for row in rows), (dimension, path)


def changed(catalog, unit):
    result = dict(catalog); target = copy.deepcopy(unit)
    result['capabilities'] = [target if row['id'] == unit['id'] else row for row in catalog['capabilities']]
    return result, target


def test_exact_original_modes_members_and_genuine_assessed_publication(binding):
    _, unit = delivered(binding)
    component, modes, tasks, _, _ = PAIR[unit['id']]
    old = next(row for row in json.loads(baseline(CATALOG))['capabilities'] if row['id'] == unit['id'])
    for key in ['canonical_symbol', 'family', 'source_members', 'required_semantics', 'consolidation', 'scope_reference']:
        assert unit.get(key) == old.get(key), key
    assert unit['mode_scope'] == modes
    assert unit['lesson_ids'] == ['assessment.' + task for task in tasks] and unit['lesson_id'] in unit['lesson_ids']
    assert unit['example_ids'] == ['prepared-' + task for task in tasks] and unit['example_id'] in unit['example_ids']
    row = next(row for row in read('components/gradient_controls/bundle.json')['components'] if row['definition']['id'] == component)
    assert row['published'] is True and row['coverageID'] == unit['id'] and row['modes'] == modes
    assert row['definition']['inputs'] == [] and row['definition']['outputs'] == ['out']
    assert row['learning']['lessonIDs'] == unit['lesson_ids'] and row['learning']['exampleIDs'] == unit['example_ids']
    assert all((ROOT / path).is_file() for path in row['learning']['sources'])
    assert [entry for entry in read('Sources/KwipMLCore/Resources/components.json')['components'] if entry['id'] == component] == [row['definition']]


def test_exact_separate_runtime_descriptors_and_no_forward_execution_table():
    bundle = read('components/gradient_controls/bundle.json')
    expected = {}
    for row in bundle['components']:
        ident = row['definition']['id']
        tree = ast.parse((ROOT / ('runner/' + ident + '.py')).read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'capability')
        descriptor = ast.literal_eval(next(node.value for node in function.body if isinstance(node, ast.Return)))
        assert row['runtimeContract'] == descriptor
        expected[ident] = descriptor
    tree = ast.parse((ROOT / 'runner/gradient_controls_generated.py').read_text())
    assignments = {target.id: node.value for node in tree.body if isinstance(node, ast.Assign) for target in node.targets if isinstance(target, ast.Name)}
    assert set(assignments) == {'CAPABILITIES'}
    assert json.loads(ast.literal_eval(assignments['CAPABILITIES'].args[0])) == expected
    assert expected['gradient_policy']['version'] == expected['trainability']['version'] == '1.0.0'


def test_acceptance_and_native_current_for_all_sixty_dimension_mode_groups(binding):
    _, unit = delivered(binding)
    assert len(coverage.REQUIRED_EVIDENCE) == 10
    for dimension in coverage.REQUIRED_EVIDENCE: bound(unit, dimension, ACCEPTANCE)
    for dimension in ['execution', 'lesson', 'example', 'ui', 'persistence']:
        bound(unit, dimension, NATIVE)


CRITICAL = {
    'configuration': ['components/gradient_controls/bundle.json', 'Sources/KwipMLCore/GradientPolicyConfiguration.swift', 'Sources/KwipMLCore/TrainabilityConfiguration.swift', 'Sources/KwipMLCore/GradientControlCapabilityBridge.swift', 'scripts/component_bundles.py'],
    'execution': ['runner/gradient_policy.py', 'runner/trainability.py', 'runner/requested_gradients.py', 'runner/engine.py', 'runner/tests/test_gradient_engine.py', 'runner/tests/test_requested_gradients.py', 'runner/tests/test_trainability.py'],
    'validation': ['Sources/KwipMLCore/WorkflowCompiler.swift', 'Sources/KwipMLApp/WorkflowGradientPolicyVerifier.swift', 'Sources/KwipMLApp/WorkflowPhaseArtifacts.swift', 'Tests/KwipMLAppTests/GradientArtifactAdmissionTests.swift'],
    'diagnostics': ['Sources/KwipMLApp/GradientPolicyInspectorContext.swift', 'Sources/KwipMLApp/TrainabilityInspectorContext.swift', 'Tests/KwipMLAppTests/GradientLessonInspectorPolicyValidationTests.swift'],
    'lesson': ['Sources/KwipMLCore/GradientLessonReference.swift', 'Sources/KwipMLCore/GradientLearningSubmission.swift', 'Sources/KwipMLCore/TensorLearningDraft.swift', 'Sources/KwipMLCore/TensorLearningReceipt.swift', 'runner/learning_gradients.py', 'runner/learning_evidence.py', 'runner/tests/test_gradient_lesson_oracle.py', 'runner/tests/test_gradient_learning_dispatch.py', 'Tests/KwipMLAppTests/GradientLearningAppIntegrationTests.swift', 'Sources/KwipMLApp/GradientLearningResultsView.swift', 'Sources/KwipMLApp/LessonAudienceGuide.swift'],
    'example': ['Sources/KwipMLCore/GradientWorkflowExamples.swift', 'Sources/KwipMLCore/GradientLearningExamples.swift', 'Sources/KwipMLApp/LearnCatalogContent.swift', 'Tests/KwipMLAppTests/GradientPublicationTests.swift'],
    'ui': ['Sources/KwipMLApp/ComponentArtworkCatalog.swift', 'Sources/KwipMLApp/GradientRequestsEditorView.swift', 'Sources/KwipMLApp/TrainabilityOwnersEditorView.swift'],
    'persistence': ['Sources/KwipMLApp/GradientArchiveRecipes.swift', 'Sources/KwipMLApp/WorkflowArchiveContracts.swift', 'Sources/KwipMLApp/WorkflowHistoryReader.swift', 'runner/checkpoint_lifecycle.py', 'runner/tests/test_gradient_checkpoint.py', 'Tests/KwipMLAppTests/GradientLifecycleVerifierTests.swift'],
    'export': ['runner/gradient_portable.py', 'runner/tests/test_gradient_export.py', 'Sources/KwipMLApp/WorkflowTrainedExport.swift', 'Sources/KwipMLApp/Resources/inference.py', 'Tests/KwipMLAppTests/GradientPortableExportContractTests.swift', 'Tests/KwipMLAppTests/GradientHistoryExportTests.swift', 'docs/production/TORCH_GRADIENT_SAVED_RESULTS_20261009.md'],
    'failure_handling': ['Sources/KwipMLApp/GradientControlRuntimeCapabilities.swift', 'Sources/KwipMLApp/CapabilityActions.swift', 'Tests/KwipMLAppTests/GradientControlRuntimeAdmissionTests.swift', 'Tests/KwipMLAppTests/GradientLessonInspectorAvailabilityTests.swift'],
}


def test_critical_independent_sources_current_and_individually_stale_refuse(binding):
    catalog, unit = delivered(binding)
    for dimension, paths in CRITICAL.items():
        for path in paths:
            bound(unit, dimension, path)
            modified, target = changed(catalog, unit)
            next(row for row in target['evidence'][dimension] if row['path'] == path)['sha256'] = '0' * 64
            assert unit['id'] not in coverage.report_catalog(modified, ROOT)['complete_ids'], (dimension, path)


def test_original_olive_artwork_generation_and_registered_bytes_are_bound(binding):
    _, unit = delivered(binding)
    _, _, _, icon, pin = PAIR[unit['id']]
    provenance = 'docs/design/gradient-control-icons-v1/provenance-20261009.json'
    original = 'docs/design/gradient-control-icons-v1/' + icon + '.png'
    registered = 'Sources/KwipMLApp/Resources/ComponentIcons/' + icon + '.png'
    source = read(provenance)
    asset = next(row for row in source['assets'] if row['assetID'] == icon)
    assert source['tool'] == 'built-in image_gen' and asset['destination'] == original
    assert digest(original) == digest(registered) == asset['sha256'] == pin
    assert (ROOT / original).read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    for path in [original, registered, provenance]: bound(unit, 'ui', path)


def test_missing_mode_unverified_scope_stale_receipt_and_pending_backend_refuse(binding):
    catalog, unit = delivered(binding)
    for dimension in coverage.REQUIRED_EVIDENCE:
        for mode in PAIR[unit['id']][1]:
            modified, target = changed(catalog, unit)
            target['evidence'][dimension] = [row for row in target['evidence'][dimension] if row.get('scope') != unit['id'] + ':' + mode]
            assert unit['id'] not in coverage.report_catalog(modified, ROOT)['complete_ids'], (dimension, mode)
        modified, target = changed(catalog, unit)
        for row in target['evidence'][dimension]: row['status'] = 'unverified'
        assert unit['id'] not in coverage.report_catalog(modified, ROOT)['complete_ids'], dimension
    for kind in ['receipt', 'backend', 'configuration', 'foreign-scope', 'missing-file']:
        modified, target = changed(catalog, unit)
        if kind == 'receipt': next(row for row in target['evidence']['configuration'] if row['path'] == ACCEPTANCE)['sha256'] = '0' * 64
        elif kind == 'backend': target['backend_matrix']['status'] = 'pending'
        elif kind == 'configuration': target['configuration_contract'] = 'pending'
        elif kind == 'missing-file': target['evidence']['execution'][0]['path'] = '.local/nonexistent-gradient-acceptance-evidence'
        else:
            for row in target['evidence']['execution']: row['scope'] = unit['id'] + ':foreign-mode'
        assert unit['id'] not in coverage.report_catalog(modified, ROOT)['complete_ids'], kind


def test_prior80_recursive_non_sha_semantics_frozen_receipts_icons_and_prior84_registry():
    old = json.loads(baseline(CATALOG)); current = read(CATALOG)
    prior = [row for row in old['capabilities'] if row.get('configuration_contract') == 'verified' and row.get('mode_scope_status') == 'reviewed' and row.get('backend_matrix', {}).get('status') == 'verified']
    assert len(prior) == 80 and len(current['capabilities']) == 88
    for unit in prior:
        actual = next(row for row in current['capabilities'] if row['id'] == unit['id'])
        assert without_sha(actual) == without_sha(unit), unit['id']
        for rows in unit['evidence'].values():
            for row in rows:
                path = row['path']
                if path.startswith('docs/') and path != 'docs/design/component-icons-v2/catalog.json': assert digest(path) == row['sha256'], path
    tracked = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', BASELINE, 'Sources/KwipMLApp/Resources/ComponentIcons', 'docs/design'], cwd=ROOT, text=True).splitlines()
    for path in tracked:
        if path.endswith('.png') or 'provenance' in Path(path).name or 'RECEIPT' in Path(path).name:
            assert digest(path) == hashlib.sha256(baseline(path)).hexdigest(), path
    old_art = json.loads(baseline('docs/design/component-icons-v2/catalog.json'))['components']
    assert historical_artwork_entries(read('docs/design/component-icons-v2/catalog.json')['components'])[:len(old_art)] == old_art
    registry = read('Sources/KwipMLCore/Resources/components.json')['components']
    old_registry = json.loads(baseline('Sources/KwipMLCore/Resources/components.json'))['components']
    assert len(old_registry) == 84
    assert {'bag_input', 'embedding_bag', 'dataset_source', 'data_loader'} <= {row['id'] for row in registry}
    assert [row for row in registry if row['id'] not in {'gradient_policy', 'trainability', 'bag_input', 'embedding_bag', 'dataset_source', 'data_loader'}] == old_registry
