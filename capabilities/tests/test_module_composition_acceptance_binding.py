"""Admit native composition only with all original modes, current proof, and preserved prior bundles."""
from capabilities.tests.artwork_inventory_compatibility import historical_artwork_entries
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
BASELINE = '95f21bc'
CATALOG = 'capabilities/catalogs/v1/pytorch.json'
IDENT = 'pytorch.torch.nn.Module'
ACCEPTANCE = 'docs/production/TORCH_COMPOSITION_ACCEPTANCE_20261009.md'
NATIVE = 'docs/production/TORCH_COMPOSITION_NATIVE_20261009.md'
MODES = ['branched-structure', 'nested-structure', 'parameter-ownership', 'train-eval',
         'architecture-export', 'sequential-structure', 'ordered-structure', 'named-structure']
MEMBERS = ['torch.nn.Module', 'torch.nn.Sequential', 'torch.nn.ModuleList', 'torch.nn.ModuleDict']
TASKS = ['composition-build-reuse', 'composition-order-name']
PROVENANCE = 'docs/design/component-icons-v2/module-composition-provenance-20261009.json'
ICONS = {
    'kwip.composition.sequential': ('composition-sequence-v1', 'cb41f75374830bcbe77454d9b462160c515d6a4a226be2184d537d40ec13c6e4'),
    'kwip.composition.orderedList': ('composition-ordered-v1', '559a89770b3cae4f52bbb2e06ae991cfed24d4b4c8eb1056c2b92c5ddc263f04'),
    'kwip.composition.namedMap': ('composition-named-v1', '005941894c555cfcdd729ff7b9cc25c3a0fcea3e381a7e9cfb890622376f9851'),
}
spec = importlib.util.spec_from_file_location('composition_acceptance_coverage', ROOT / 'scripts/coverage_report.py')
coverage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coverage)


def read(path): return json.loads((ROOT / path).read_text())
def digest(path): return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
def baseline(path): return subprocess.check_output(['git', 'show', BASELINE + ':' + path], cwd=ROOT)
def without_sha(value):
    if isinstance(value, dict): return {key: without_sha(item) for key, item in value.items() if key != 'sha256'}
    if isinstance(value, list): return [without_sha(item) for item in value]
    return value


def delivered(catalog=None):
    catalog = read(CATALOG) if catalog is None else catalog
    unit = next(row for row in catalog['capabilities'] if row['id'] == IDENT)
    assert len(catalog['capabilities']) == 88
    assert unit['implementation_status'] == 'delivered'
    assert unit['mode_scope_status'] == 'reviewed' and unit['configuration_contract'] == 'verified'
    assert unit['mode_scope'] == MODES
    assert unit['backend_matrix']['status'] == 'verified'
    assert unit['backend_matrix']['reference'] == ACCEPTANCE
    assert any(case.startswith('cpu') for case in unit['backend_matrix']['cases'])
    assert any(case.startswith('mps') for case in unit['backend_matrix']['cases'])
    assert IDENT in coverage.report_catalog(catalog, ROOT)['complete_ids']
    return catalog, unit


def bound(unit, dimension, path):
    rows = [row for row in unit['evidence'].get(dimension, []) if row['path'] == path]
    assert {IDENT + ':' + mode for mode in MODES} <= {row.get('scope') for row in rows}, (dimension, path)
    assert all(row['status'] == 'verified' and row['sha256'] == digest(path) for row in rows), (dimension, path)


def changed(catalog, unit):
    result = dict(catalog); target = copy.deepcopy(unit)
    result['capabilities'] = [target if row['id'] == IDENT else row for row in catalog['capabilities']]
    return result, target


def test_all_original_modes_four_members_and_two_genuine_lesson_example_routes():
    _, unit = delivered()
    old = next(row for row in json.loads(baseline(CATALOG))['capabilities'] if row['id'] == IDENT)
    for key in ['canonical_symbol', 'family', 'source_members', 'required_semantics', 'consolidation', 'scope_reference']:
        assert unit.get(key) == old.get(key), key
    assert [row['canonical_symbol'] for row in unit['source_members']] == MEMBERS
    assert unit['lesson_ids'] == ['assessment.' + task for task in TASKS]
    assert unit['lesson_id'] in unit['lesson_ids']
    assert unit['example_ids'] == ['prepared-' + task for task in TASKS]
    assert unit['example_id'] in unit['example_ids']


def test_separate_structural_runtime_descriptor_matches_swift_and_is_not_a_primitive():
    tree = ast.parse((ROOT / 'runner/module_composition_capability.py').read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'capability')
    descriptor = ast.literal_eval(next(node.value for node in function.body if isinstance(node, ast.Return)))
    assert descriptor['modes'] == MODES and descriptor['sourceMembers'] == MEMBERS
    assert descriptor['containerKinds'] == ['graph', 'sequential', 'orderedList', 'namedMap']
    assert descriptor['devices'] == ['cpu', 'mps']
    assert descriptor['exactResumeDevices'] == ['cpu'] and descriptor['mpsResume'] == 'weightsOnly'
    assert descriptor['sharedParameters'] is False
    assert descriptor['sourceVerification'] == 'independentCapturedRegistryReExpansion'
    swift = (ROOT / 'Sources/KwipMLCore/ModuleCompositionRuntimeCapability.swift').read_text()
    encoded = re.search(r'Data\(#"(\{.*\})"#\.utf8\)', swift)
    assert encoded and json.loads(encoded.group(1)) == descriptor
    primitive_ids = {row['id'] for row in read('Sources/KwipMLCore/Resources/components.json')['components']}
    assert not ({'module_composition', *ICONS} & primitive_ids)
    _, unit = delivered()
    bound(unit, 'execution', 'runner/module_composition_capability.py')
    bound(unit, 'failure_handling', 'Sources/KwipMLCore/ModuleCompositionRuntimeCapability.swift')


CRITICAL = {
    'configuration': ['Sources/KwipMLCore/ModuleCompositionEditing.swift', 'Sources/KwipMLApp/ModuleCompositionInspectorContext.swift', 'Sources/KwipMLApp/ModuleCompositionInspectorView.swift', 'Tests/KwipMLAppTests/ModuleCompositionInspectorTests.swift'],
    'execution': ['runner/module_composition_plan.py', 'runner/engine.py', 'runner/trainability.py', 'runner/tests/test_module_composition_execution.py', 'runner/tests/test_module_composition_lifecycle.py', 'Tests/KwipMLCoreTests/ModuleCompositionWorkerBridgeTests.swift'],
    'export': ['runner/engine.py', 'runner/gradient_portable.py', 'runner/tests/test_module_composition_export.py', 'Sources/KwipMLApp/WorkflowTrainedExport.swift', 'Tests/KwipMLAppTests/ModuleCompositionStandaloneTests.swift'],
    'validation': ['runner/module_composition_policy.py', 'runner/module_composition_source.py', 'Sources/KwipMLCore/WorkflowCompiler.swift', 'Tests/KwipMLCoreTests/ModuleCompositionPlanTests.swift', 'runner/tests/test_module_composition_source.py'],
    'diagnostics': ['Sources/KwipMLApp/ModuleCompositionLearningResultsView.swift', 'Sources/KwipMLCore/ModuleCompositionLearningMetadata.swift', 'Tests/KwipMLCoreTests/ModuleCompositionLearningMetadataTests.swift'],
    'lesson': ['Sources/KwipMLCore/ModuleCompositionLearningTask.swift', 'Sources/KwipMLCore/ModuleCompositionLearningSubmission.swift', 'Sources/KwipMLCore/ModuleCompositionLearningDraft.swift', 'Sources/KwipMLCore/TensorLearningReceipt.swift', 'runner/module_composition_reference.py', 'runner/module_composition_learning.py', 'runner/learning_evidence.py', 'runner/tests/test_module_composition_learning.py', 'runner/tests/test_module_composition_learning_dispatch.py', 'Tests/KwipMLAppTests/ModuleCompositionLearningRuntimeTests.swift'],
    'example': ['Sources/KwipMLCore/ModuleCompositionLearningExamples.swift', 'Sources/KwipMLApp/LearnCatalogContent.swift', 'Tests/KwipMLAppTests/ModuleCompositionLearnCatalogTests.swift'],
    'ui': ['Sources/KwipMLApp/ModuleCompositionLearningWorkbenchView.swift', 'Sources/KwipMLApp/ModuleCompositionLearningCanvasSession.swift', 'Sources/KwipMLApp/ComponentArtworkCatalog.swift', 'Sources/KwipMLApp/GraphCanvasView.swift', 'Tests/KwipMLAppTests/ModuleCompositionHistoryFocusTests.swift'],
    'persistence': ['Sources/KwipMLApp/ModuleCompositionLearningCanvasAuthority.swift', 'Sources/KwipMLApp/WorkflowArchiveCaptureWriter.swift', 'Sources/KwipMLApp/WorkflowHistoryReader.swift', 'runner/checkpoint_lifecycle.py', 'Tests/KwipMLAppTests/ModuleCompositionArchiveTests.swift', 'Tests/KwipMLAppTests/ModuleCompositionManagedRunTests.swift'],
    'failure_handling': ['Sources/KwipMLApp/CapabilityActions.swift', 'Sources/KwipMLApp/ModuleCompositionLearningCanvasAuthority.swift', 'Tests/KwipMLAppTests/ModuleCompositionRuntimeCapabilityTests.swift', 'runner/tests/test_module_composition_policy.py'],
}


def test_current_sealed_acceptance_all_eighty_dimension_mode_groups_and_native_evidence():
    _, unit = delivered()
    assert len(coverage.REQUIRED_EVIDENCE) == 10
    for dimension in coverage.REQUIRED_EVIDENCE:
        bound(unit, dimension, ACCEPTANCE)
        for path in CRITICAL[dimension]: bound(unit, dimension, path)
    # Native observations are representative journeys, not every device/mode permutation.
    for dimension in ['configuration', 'lesson', 'example', 'ui', 'persistence']:
        bound(unit, dimension, NATIVE)


def test_original_generated_three_icons_registered_without_replacing_prior_block_art():
    _, unit = delivered()
    provenance = read(PROVENANCE)
    assets = {row['id']: row for row in provenance['assets']}
    catalog = {row['id']: row for row in read('docs/design/component-icons-v2/catalog.json')['components']}
    swift = (ROOT / 'Sources/KwipMLApp/ComponentArtworkCatalog.swift').read_text()
    assert set(assets) == {asset for asset, _ in ICONS.values()}
    for component, (asset, pin) in ICONS.items():
        row = assets[asset]
        assert row['generator'] == 'image_gen.imagegen' and row['transparentBackground'] is True
        assert digest(row['path']) == digest(row['resourcePath']) == row['sha256'] == pin
        assert (ROOT / row['path']).read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
        assert catalog[component]['asset'] == asset + '.png'
        assert '"' + component + '":"' + asset + '"' in swift
        for path in [row['path'], row['resourcePath'], PROVENANCE, 'docs/design/component-icons-v2/catalog.json']:
            bound(unit, 'ui', path)
    block = 'Sources/KwipMLApp/Resources/ComponentIcons/block-v2.png'
    assert (ROOT / block).read_bytes() == baseline(block)


@pytest.mark.parametrize('dimension', coverage.REQUIRED_EVIDENCE)
def test_missing_mode_unverified_foreign_scope_and_stale_dependency_withhold_credit(dimension):
    catalog, unit = delivered()
    for mutation in ['missing-mode', 'unverified', 'foreign-scope', 'stale', 'missing-file']:
        modified, target = changed(catalog, unit)
        rows = target['evidence'][dimension]
        if mutation == 'missing-mode': target['evidence'][dimension] = [row for row in rows if row.get('scope') != IDENT + ':' + MODES[-1]]
        elif mutation == 'unverified':
            for row in rows: row['status'] = 'unverified'
        elif mutation == 'foreign-scope':
            for row in rows: row['scope'] = IDENT + ':foreign-mode'
        elif mutation == 'stale': rows[0]['sha256'] = '0' * 64
        else: rows[0]['path'] = '.local/nonexistent-composition-proof'
        assert IDENT not in coverage.report_catalog(modified, ROOT)['complete_ids'], (dimension, mutation)


@pytest.mark.parametrize('mutation', ['implementation', 'configuration', 'review', 'backend', 'no-cpu', 'no-mps'])
def test_pending_status_or_missing_required_device_refuses_admission(mutation):
    catalog, unit = delivered()
    modified, target = changed(catalog, unit)
    if mutation == 'implementation': target['implementation_status'] = 'planned'
    elif mutation == 'configuration': target['configuration_contract'] = 'pending'
    elif mutation == 'review': target['mode_scope_status'] = 'pending-operation-boundary-review'
    elif mutation == 'backend': target['backend_matrix']['status'] = 'pending'
    else:
        prefix = 'cpu' if mutation == 'no-cpu' else 'mps'
        target['backend_matrix']['cases'] = [case for case in target['backend_matrix']['cases'] if not case.startswith(prefix)]
    with pytest.raises(AssertionError): delivered(modified)


def test_prior82_recursive_semantics_frozen_receipts_art_and_original_registry_unchanged():
    old = json.loads(baseline(CATALOG)); current = read(CATALOG)
    prior = [row for row in old['capabilities'] if row.get('configuration_contract') == 'verified' and row.get('mode_scope_status') == 'reviewed' and row.get('backend_matrix', {}).get('status') == 'verified']
    assert len(prior) == 82 and len(current['capabilities']) == 88
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
            assert (ROOT / path).read_bytes() == baseline(path), path
    oldart = json.loads(baseline('docs/design/component-icons-v2/catalog.json'))
    nowart = {row['id']: row for row in historical_artwork_entries(read('docs/design/component-icons-v2/catalog.json')['components'])}
    assert all(nowart[row['id']] == row for row in oldart['components'])
    oldregistry = json.loads(baseline('Sources/KwipMLCore/Resources/components.json'))['components']
    nowregistry = {row['id']: row for row in read('Sources/KwipMLCore/Resources/components.json')['components']}
    assert len(oldregistry) == 86
    assert all(nowregistry[row['id']] == row for row in oldregistry)
