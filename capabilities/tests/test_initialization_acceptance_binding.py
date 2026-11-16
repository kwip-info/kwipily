"""Initialization credit is one real recipe bundle, never four aliases or a forward layer."""
from capabilities.tests.artwork_inventory_compatibility import historical_artwork_entries
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BASELINE = 'ddcd227'
CATALOG = 'capabilities/catalogs/v1/pytorch.json'
ID = 'pytorch.torch.nn.init.xavier_uniform_'
ACCEPTANCE = 'docs/production/TORCH_INITIALIZATION_ACCEPTANCE_20261009.md'
NATIVE = 'docs/production/TORCH_INITIALIZATION_NATIVE_20261009.md'
MODES = ['xavier-uniform', 'seeded-repeat', 'kaiming-uniform', 'normal', 'zeros']
LESSONS = ['assessment.initialization-starting-values', 'assessment.initialization-starting-spread']
EXAMPLES = ['prepared-initialization-starting-values', 'prepared-initialization-starting-spread']
ART_SHA = 'a9cedb8b5221ed2730b8fbdee327d7fe5dd646ed1803117ac35ad4c5ce8be897'
MEMBERS = [
    {'canonical_symbol': 'torch.nn.init.xavier_uniform_', 'required_modes': ['xavier-uniform', 'seeded-repeat'], 'original_capability_id': ID},
    {'canonical_symbol': 'torch.nn.init.kaiming_uniform_', 'required_modes': ['kaiming-uniform', 'seeded-repeat'], 'original_capability_id': 'pytorch.torch.nn.init.kaiming_uniform_'},
    {'canonical_symbol': 'torch.nn.init.normal_', 'required_modes': ['normal', 'seeded-repeat'], 'original_capability_id': 'pytorch.torch.nn.init.normal_'},
    {'canonical_symbol': 'torch.nn.init.zeros_', 'required_modes': ['zeros', 'seeded-repeat'], 'original_capability_id': 'pytorch.torch.nn.init.zeros_'},
]
spec = importlib.util.spec_from_file_location('initialization_binding_coverage', ROOT / 'scripts/coverage_report.py')
coverage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coverage)


def read(path):
    return json.loads((ROOT / path).read_text())


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def entry():
    bundle = read('components/initialization/bundle.json')
    assert bundle['kind'] == 'trainingControl' and bundle['id'] == 'kwip.initialization'
    assert len(bundle['components']) == 1
    return bundle['components'][0]


def delivered():
    catalog = read(CATALOG)
    unit = next(row for row in catalog['capabilities'] if row['id'] == ID)
    assert len(catalog['capabilities']) == 88
    # This assertion intentionally REDs until root freezes acceptance and binds evidence.
    assert unit['implementation_status'] == 'delivered'
    assert unit['configuration_contract'] == 'verified' and unit['mode_scope_status'] == 'reviewed'
    assert unit['backend_matrix']['status'] == 'verified'
    assert unit['backend_matrix']['reference'] == ACCEPTANCE
    assert any(case.startswith('cpu') for case in unit['backend_matrix']['cases'])
    assert any(case.startswith('mps') for case in unit['backend_matrix']['cases'])
    assert ID in coverage.report_catalog(catalog, ROOT)['complete_ids']
    return catalog, unit


def bound(unit, dimension, path):
    rows = [row for row in unit['evidence'].get(dimension, []) if row['path'] == path]
    assert {ID + ':' + mode for mode in MODES} <= {row.get('scope') for row in rows}, (dimension, path)
    assert all(row['status'] == 'verified' and row['sha256'] == digest(path) for row in rows), (dimension, path)


def changed_unit(catalog, unit):
    # Only the edited unit needs copying; the reporter is read-only.
    changed = dict(catalog)
    target = copy.deepcopy(unit)
    changed['capabilities'] = [target if row['id'] == ID else row for row in catalog['capabilities']]
    return changed, target


def test_consolidated_original_members_modes_and_genuine_published_control_lessons():
    _, unit = delivered()
    assert unit['canonical_symbol'] == 'torch.nn.init.xavier_uniform_' and unit['family'] == 'initialization'
    assert unit['mode_scope'] == MODES and unit['source_members'] == MEMBERS
    assert unit['lesson_ids'] == LESSONS and unit['lesson_id'] in LESSONS
    assert unit['example_ids'] == EXAMPLES and unit['example_id'] in EXAMPLES
    item = entry()
    assert item['published'] is True and item['coverageID'] == ID and item['modes'] == MODES
    assert item['definition'] == {'id': 'initialization', 'title': 'Starting settings', 'category': 'Training', 'inputs': [], 'outputs': ['out'], 'fields': []}
    assert item['editorExtension'] == 'initializationOwnersV1'
    assert item['learning']['lessonIDs'] == LESSONS and item['learning']['exampleIDs'] == EXAMPLES
    assert all((ROOT / path).is_file() for path in item['learning']['sources'])
    registry = read('Sources/KwipMLCore/Resources/components.json')['components']
    assert [row for row in registry if row['id'] == 'initialization'] == [item['definition']]
    assert not any(row['id'] in {'xavier_uniform', 'kaiming_uniform', 'normal', 'zeros'} for row in registry)


def test_descriptor_is_exact_separate_cpu_generator_policy_and_not_forward_adapter():
    item = entry()
    contract = item['runtimeContract']
    expected = {
        'version': {'major': 1, 'minor': 0, 'patch': 0}, 'adapterID': 'kwip.runner.initialization',
        'recipeSchemaVersions': [1], 'configurationVersions': [1],
        'algorithms': ['xavier_uniform', 'kaiming_uniform', 'normal', 'zeros'],
        'seedMinimum': 0, 'seedMaximum': 9223372036854775807,
        'seedDerivation': 'kwip.initialization.owner-seed.v1', 'generatorPlacement': 'cpu-float32',
        'devices': ['cpu', 'mps'], 'parameterDTypes': ['float32'], 'maximumRecipeBytes': 32768,
        'maximumRules': 128, 'maximumModelParameters': 200000,
        'ownerPolicy': 'expandedNodeRelativeNativeParameter', 'ownerOrder': 'nodeThenParameter',
        'aliasPolicy': 'rejectDuplicateParameterIdentity', 'fanPolicy': 'nativeStoredDimensions',
        'applicationTiming': 'freshAfterLazyBeforeOptimizer', 'restoredWeights': 'preserve',
        'candidateCommit': 'validateAllBeforeCopy', 'preservesParameterIdentity': True,
        'initializesBuffers': False, 'standaloneHelperMaximumBytes': 131072,
    }
    assert item['published'] is True and contract == expected
    tree = ast.parse((ROOT / 'runner/engine.py').read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'initialization_capability')
    assert ast.literal_eval(function.body[0].value) == expected
    discovery = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'capabilities')
    returned = next(node.value for node in discovery.body if isinstance(node, ast.Return))
    groups = {key.value: value for key, value in zip(returned.keys, returned.values) if isinstance(key, ast.Constant)}
    assert isinstance(groups['initializationCapabilities'], ast.Dict)
    assert [key.value for key in groups['initializationCapabilities'].keys] == ['initialization']
    assert 'initialization' not in {node.value for node in ast.walk(groups['componentCapabilities']) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    generated = ast.parse((ROOT / 'runner/initialization_adapters_generated.py').read_text())
    assignments = {target.id: node.value for node in generated.body if isinstance(node, ast.Assign) for target in node.targets if isinstance(target, ast.Name)}
    assert set(assignments) == {'CAPABILITIES'}
    assert json.loads(ast.literal_eval(assignments['CAPABILITIES'].args[0])) == {'initialization': expected}


def test_frozen_acceptance_and_native_bind_all_ten_dimensions_and_five_modes():
    _, unit = delivered()
    assert (ROOT / ACCEPTANCE).is_file()
    assert len(coverage.REQUIRED_EVIDENCE) == 10
    for dimension in coverage.REQUIRED_EVIDENCE:
        bound(unit, dimension, ACCEPTANCE)
    for dimension in ['execution', 'export', 'lesson', 'example', 'ui', 'persistence']:
        bound(unit, dimension, NATIVE)


def test_independent_runtime_transport_owner_lesson_and_export_sources_are_current():
    catalog, unit = delivered()
    required = {
        'configuration': ['components/initialization/bundle.json', 'Sources/KwipMLCore/InitializationConfiguration.swift', 'Sources/KwipMLCore/InitializationCapabilityBridge.swift', 'runner/initialization_adapters_generated.py', 'scripts/component_bundles.py'],
        'execution': ['runner/initialization.py', 'runner/engine.py', 'runner/tests/test_initialization.py', 'runner/tests/test_initialization_engine.py', 'runner/tests/test_initialization_transport.py', 'Sources/KwipMLApp/WorkflowExecutionRequest.swift', 'Sources/KwipMLApp/WorkflowRunSourceCapture.swift', 'Sources/KwipMLApp/WorkflowPhaseArguments.swift', 'Sources/KwipMLApp/WorkflowModelTrainingToolContract.swift'],
        'validation': ['Tests/KwipMLCoreTests/InitializationWorkflowCompilerTests.swift', 'Sources/KwipMLApp/WorkflowInitializationVerifier.swift', 'Tests/KwipMLAppTests/InitializationArtifactVerifierTests.swift', 'Tests/KwipMLAppTests/InitializationPhaseArtifactTests.swift'],
        'diagnostics': ['Sources/KwipMLApp/InitializationRecipeInspectorContext.swift', 'Sources/KwipMLApp/InitializationRulesEditorView.swift', 'Tests/KwipMLAppTests/InitializationRecipeInspectorTests.swift'],
        'lesson': ['Sources/KwipMLCore/InitializationLessonReference.swift', 'Sources/KwipMLCore/InitializationLearningSubmission.swift', 'Sources/KwipMLCore/TensorLearningDraft.swift', 'Sources/KwipMLCore/TensorLearningReceipt.swift', 'runner/learning_initialization.py', 'runner/learning_evidence.py', 'runner/tests/test_initialization_lesson_oracle.py', 'runner/tests/test_initialization_learning_evidence.py', 'runner/tests/test_learning_initialization.py', 'Tests/KwipMLAppTests/InitializationLearningRuntimeTests.swift', 'Sources/KwipMLApp/LessonAudienceGuide.swift', 'Tests/KwipMLAppTests/LessonAudienceGuideTests.swift'],
        'example': ['Sources/KwipMLCore/InitializationLearningExamples.swift', 'Sources/KwipMLApp/LearnCatalogContent.swift', 'Tests/KwipMLCoreTests/InitializationLearningTests.swift'],
        'persistence': ['Sources/KwipMLApp/WorkflowArchiveContracts.swift', 'Sources/KwipMLApp/WorkflowHistoryReader.swift', 'Tests/KwipMLCoreTests/InitializationLearningReceiptTests.swift', 'Tests/KwipMLAppTests/InitializationHistoryExportTests.swift'],
        'export': ['Sources/KwipMLApp/WorkflowTrainedExport.swift', 'Sources/KwipMLApp/Resources/inference.py', 'runner/tests/test_initialization_portable_real.py', 'docs/production/TORCH_INITIALIZATION_PORTABLE_REAL_20261009.md'],
        'failure_handling': ['Sources/KwipMLApp/InitializationRuntimeCapabilities.swift', 'Sources/KwipMLApp/CapabilityActions.swift', 'Tests/KwipMLAppTests/InitializationRuntimeAdmissionTests.swift', 'Tests/KwipMLAppTests/InitializationCanvasDeliveryTests.swift'],
    }
    for dimension, paths in required.items():
        for path in paths:
            bound(unit, dimension, path)
    # Independent critical dependencies must individually withhold credit when stale.
    for dimension, path in [
        ('execution', 'runner/initialization.py'), ('execution', 'runner/engine.py'),
        ('validation', 'Sources/KwipMLApp/WorkflowInitializationVerifier.swift'),
        ('lesson', 'runner/tests/test_initialization_lesson_oracle.py'),
        ('lesson', 'Sources/KwipMLCore/TensorLearningReceipt.swift'),
        ('export', 'runner/tests/test_initialization_portable_real.py'),
        ('failure_handling', 'Tests/KwipMLAppTests/InitializationRuntimeAdmissionTests.swift'),
    ]:
        changed, target = changed_unit(catalog, unit)
        next(row for row in target['evidence'][dimension] if row['path'] == path)['sha256'] = '0' * 64
        assert ID not in coverage.report_catalog(changed, ROOT)['complete_ids'], path


def test_generated_original_icon_and_registered_copy_are_identical_bound_evidence():
    _, unit = delivered()
    source = 'docs/design/initialization-icons-v1/initialization-v1.png'
    resource = 'Sources/KwipMLApp/Resources/ComponentIcons/initialization-v1.png'
    provenance = 'docs/design/initialization-icons-v1/provenance-20261009.json'
    art = read(provenance)
    assert digest(source) == digest(resource) == art['sha256'] == art['resourceSHA256'] == ART_SHA
    assert art['tool'] == 'built-in image_gen' and art['destination'] == source and art['registeredResource'] == resource
    assert (ROOT / resource).read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    for path in [source, resource, provenance, 'Sources/KwipMLApp/ComponentArtworkCatalog.swift', 'Tests/KwipMLAppTests/InitializationCanvasDeliveryTests.swift']:
        bound(unit, 'ui', path)


def test_each_missing_dimension_mode_and_unverified_scope_independently_refuses_credit():
    catalog, unit = delivered()
    for dimension in coverage.REQUIRED_EVIDENCE:
        for mode in MODES:
            changed, target = changed_unit(catalog, unit)
            target['evidence'][dimension] = [row for row in target['evidence'][dimension] if row.get('scope') != ID + ':' + mode]
            assert ID not in coverage.report_catalog(changed, ROOT)['complete_ids'], (dimension, mode)
        changed, target = changed_unit(catalog, unit)
        for row in target['evidence'][dimension]:
            row['status'] = 'unverified'
        assert ID not in coverage.report_catalog(changed, ROOT)['complete_ids'], dimension


def test_stale_receipt_missing_file_pending_device_config_and_wrong_scope_refuse_credit():
    catalog, unit = delivered()
    for kind in ['receipt', 'missing', 'backend', 'configuration', 'scope']:
        changed, target = changed_unit(catalog, unit)
        if kind == 'receipt':
            next(row for row in target['evidence']['configuration'] if row['path'] == ACCEPTANCE)['sha256'] = '0' * 64
        elif kind == 'missing':
            target['evidence']['execution'][0]['path'] = '.local/nonexistent-initialization-acceptance-evidence'
        elif kind == 'backend':
            target['backend_matrix']['status'] = 'pending'
        elif kind == 'configuration':
            target['configuration_contract'] = 'pending'
        else:
            for row in target['evidence']['lesson']:
                if row.get('scope') == ID + ':seeded-repeat':
                    row['scope'] = ID + ':foreign-repeat'
        assert ID not in coverage.report_catalog(changed, ROOT)['complete_ids'], kind


def test_prior_79_recursive_semantics_frozen_receipts_icons_and_mutable_catalog_prefix():
    baseline = json.loads(subprocess.check_output(['git', 'show', BASELINE + ':' + CATALOG], cwd=ROOT))
    current = read(CATALOG)
    prior = [unit for unit in baseline['capabilities'] if unit.get('configuration_contract') == 'verified' and unit.get('mode_scope_status') == 'reviewed' and unit.get('backend_matrix', {}).get('status') == 'verified']
    assert len(prior) == 79 and len(current['capabilities']) == 88
    def without_sha(value):
        if isinstance(value, dict):
            return {key: without_sha(item) for key, item in value.items() if key != 'sha256'}
        if isinstance(value, list):
            return [without_sha(item) for item in value]
        return value
    for old in prior:
        actual = next(unit for unit in current['capabilities'] if unit['id'] == old['id'])
        assert without_sha(actual) == without_sha(old), old['id']
        for rows in old['evidence'].values():
            for row in rows:
                path = row['path']
                if path.startswith('docs/') and path != 'docs/design/component-icons-v2/catalog.json':
                    assert digest(path) == row['sha256'], path
    tracked = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', BASELINE, 'Sources/KwipMLApp/Resources/ComponentIcons', 'docs/design'], cwd=ROOT, text=True).splitlines()
    for path in tracked:
        if path.endswith('.png') or 'provenance' in Path(path).name or 'RECEIPT' in Path(path).name:
            original = subprocess.check_output(['git', 'show', BASELINE + ':' + path], cwd=ROOT)
            assert digest(path) == hashlib.sha256(original).hexdigest(), path
    old_art = json.loads(subprocess.check_output(['git', 'show', BASELINE + ':docs/design/component-icons-v2/catalog.json'], cwd=ROOT))
    assert historical_artwork_entries(read('docs/design/component-icons-v2/catalog.json')['components'])[:len(old_art['components'])] == old_art['components']
    old_registry = json.loads(subprocess.check_output(['git', 'show', BASELINE + ':Sources/KwipMLCore/Resources/components.json'], cwd=ROOT))['components']
    actual_registry = read('Sources/KwipMLCore/Resources/components.json')['components']
    later_ids = {'initialization', 'gradient_policy', 'trainability', 'bag_input', 'embedding_bag', 'dataset_source', 'data_loader'}
    assert later_ids <= {row['id'] for row in actual_registry}
    assert [row for row in actual_registry if row['id'] not in later_ids] == old_registry
