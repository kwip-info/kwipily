"""Admit the original EmbeddingBag unit only with current packet proof and unchanged prior83 semantics."""
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
BASELINE = 'bb4502bb3f46b5dadca5fe6e80a34e60b6ee64be'
CATALOG = 'capabilities/catalogs/v1/pytorch.json'
IDENT = 'pytorch.torch.nn.EmbeddingBag'
ACCEPTANCE = 'docs/production/TORCH_BAG_ACCEPTANCE_20261009.md'
NATIVE = 'docs/production/TORCH_BAG_NATIVE_20261009.md'
MODES = ['inference', 'training', 'input-gradients', 'state-roundtrip']
TASKS = ['bag-fixed-sum', 'bag-packed-mean', 'bag-packed-max', 'bag-fixed-weighted-sum', 'bag-packed-weighted-sum', 'bag-padding-norm']
EXAMPLES = ['prepared-embedding-bag-classifier']
ICONS = {
    'bag_input': ('bag-input-v1', 'bag-input-provenance-20261009.json', 'ec079864d36b0285c4ff214b87931a1acfaddd4ae22993750ee0fae00280b09f'),
    'embedding_bag': ('embedding-bag-v1', 'embedding-bag-provenance-20261009.json', 'ad5de9056c16b1c14d412d41c37f85984b52b61132e4c25b44fea35a62a74d78'),
}
spec = importlib.util.spec_from_file_location('bag_acceptance_coverage', ROOT / 'scripts/coverage_report.py')
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


CRITICAL = {
    'configuration': ['Sources/KwipMLApp/BagInputMappingInspector.swift', 'Sources/KwipMLApp/WorkflowControlNodeContext.swift', 'Sources/KwipMLApp/LossTargetMappingInspector.swift', 'Tests/KwipMLAppTests/BagInputMappingInspectorTests.swift', 'docs/production/TORCH_BAG_MAPPING_INSPECTOR_20261009.md', 'components/embeddings/bundle.json', 'Sources/KwipMLCore/EmbeddingBagConfiguration.swift', 'Sources/KwipMLCore/BagSourceConfiguration.swift', 'runner/embedding_bag_configuration.py', 'Tests/KwipMLCoreTests/EmbeddingBagConfigurationTests.swift', 'runner/tests/test_embedding_bag_configuration.py', 'docs/production/TORCH_EMBEDDING_BAG_CORE_CONFIGURATION_20261009.md'],
    'execution': ['runner/embedding_bag.py', 'runner/embedding_bag_packet.py', 'runner/bag_source.py', 'runner/engine.py', 'runner/product_execution.py', 'runner/tests/test_embedding_bag.py', 'runner/tests/test_bag_training.py', 'runner/tests/test_bag_product_execution.py', 'docs/production/TORCH_BAG_WORKER_INTEGRATION_20261009.md'],
    'export': ['runner/bag_export.py', 'Sources/KwipMLApp/Resources/inference.py', 'Sources/KwipMLApp/WorkflowHistoryReader.swift', 'Sources/KwipMLApp/TrainedModelInputContract.swift', 'runner/gradient_portable.py', 'runner/tests/test_bag_export.py', 'runner/tests/test_bag_native_inference.py', 'Tests/KwipMLAppTests/BagHistoryExportTests.swift', 'docs/production/TORCH_BAG_NATIVE_TRAINED_EXPORT_20261009.md'],
    'validation': ['Sources/KwipMLCore/GraphValidator.swift', 'Sources/KwipMLCore/BagInputMapping.swift', 'Sources/KwipMLCore/WorkflowCompiler.swift', 'runner/graph_semantics.py', 'runner/module_composition_source.py', 'runner/prepared_bag_data.py', 'Tests/KwipMLCoreTests/BagCompositionWorkerBridgeTests.swift', 'Tests/KwipMLCoreTests/BagWorkflowMappingTests.swift', 'runner/tests/test_bag_source_semantics.py', 'runner/tests/test_bag_composition.py'],
    'diagnostics': ['Sources/KwipMLApp/WorkflowBagArtifacts.swift', 'Sources/KwipMLApp/WorkflowBagActivationEvidence.swift', 'Sources/KwipMLApp/DerivativeResults.swift', 'Sources/KwipMLApp/BagResultProjection.swift', 'Sources/KwipMLApp/BagResultSectionsView.swift', 'Tests/KwipMLAppTests/BagArtifactTests.swift', 'Tests/KwipMLAppTests/BagDerivativeResultsTests.swift', 'Tests/KwipMLAppTests/BagResultPresentationTests.swift', 'docs/production/TORCH_BAG_NATIVE_DERIVATIVES_20261009.md'],
    'lesson': ['Sources/KwipMLCore/EmbeddingBagAssessment.swift', 'Sources/KwipMLCore/EmbeddingBagLessonReference.swift', 'Sources/KwipMLCore/EmbeddingBagLearningCanvas.swift', 'Sources/KwipMLCore/EmbeddingBagLearningSubmission.swift', 'runner/learning_embedding_bag.py', 'runner/learning_evidence.py', 'runner/tests/test_learning_embedding_bag.py', 'runner/tests/test_embedding_bag_learning_dispatch.py', 'Tests/KwipMLAppTests/EmbeddingBagLessonLifecycleTests.swift', 'docs/production/TORCH_BAG_LEARN_PUBLICATION_20261009.md'],
    'example': ['Sources/KwipMLCore/EmbeddingBagWorkflowExample.swift', 'Sources/KwipMLCore/EmbeddingBagExample.swift', 'Sources/KwipMLApp/Resources/bag-observations.csv', 'Tests/KwipMLAppTests/PublishedBagWorkflowTests.swift', 'docs/production/TORCH_BAG_PUBLICATION_ROOT_20261009.md'],
    'ui': ['Sources/KwipMLApp/WorkflowControlInspectorView.swift', 'Sources/KwipMLApp/BagInputMappingInspector.swift', 'Sources/KwipMLApp/EmbeddingBagRuntimeCapabilities.swift', 'Sources/KwipMLApp/ComponentArtworkCatalog.swift', 'Sources/KwipMLApp/CapabilityActions.swift', 'Sources/KwipMLApp/WorkflowCanvasAdapter.swift', 'Tests/KwipMLAppTests/EmbeddingBagRuntimeCapabilitiesTests.swift', 'Tests/KwipMLAppTests/EmbeddingBagArtworkTests.swift', 'Tests/KwipMLAppTests/EmbeddingBagRuntimeRefreshPresentationTests.swift'],
    'persistence': ['Tests/KwipMLAppTests/WorkflowArchiveBackgroundStackTests.swift', 'docs/production/TORCH_BAG_BACKGROUND_HISTORY_20261009.md', 'Tests/KwipMLAppTests/BagInputMappingInspectorTests.swift', 'Sources/KwipMLApp/WorkflowControlNodeContext.swift', 'Sources/KwipMLApp/WorkflowBagPreparedEvidence.swift', 'Sources/KwipMLApp/WorkflowArchiveContracts.swift', 'Sources/KwipMLApp/WorkflowExecutionSession.swift', 'Sources/KwipMLApp/WorkflowHistoryReader.swift', 'runner/checkpoint_lifecycle.py', 'Tests/KwipMLAppTests/BagWorkflowArchiveTests.swift', 'Tests/KwipMLAppTests/BagWorkflowSessionTests.swift', 'Tests/KwipMLAppTests/BagHistoryReaderTests.swift', 'runner/tests/test_bag_checkpoint_identity.py', 'runner/tests/test_bag_checkpoint_modes.py'],
    'failure_handling': ['Sources/KwipMLApp/WorkflowArchiveContracts.swift', 'Tests/KwipMLAppTests/WorkflowArchiveBackgroundStackTests.swift', 'docs/production/TORCH_BAG_BACKGROUND_HISTORY_20261009.md', 'capabilities/tests/test_embedding_bag_acceptance_binding.py', 'runner/frozen_checkpoint.py', 'runner/tests/test_bag_worker_authorization.py', 'runner/tests/test_bag_checkpoint_modes.py', 'runner/tests/test_bag_native_inference.py', 'Sources/KwipMLApp/EmbeddingBagRuntimeCapabilities.swift', 'Tests/KwipMLAppTests/EmbeddingBagRuntimeCapabilitiesTests.swift', 'Tests/KwipMLAppTests/BagArtifactTests.swift', 'Tests/KwipMLAppTests/BagHistoryExportTests.swift'],
}
MODE_PROOFS = {
    'inference': ['runner/tests/test_embedding_bag.py', 'runner/tests/test_bag_native_inference.py', 'docs/production/TORCH_BAG_NATIVE_INFERENCE_20261009.md'],
    'training': ['runner/tests/test_bag_training.py', 'Tests/KwipMLAppTests/PublishedBagWorkflowTests.swift', 'docs/production/TORCH_BAG_WORKER_INTEGRATION_20261009.md'],
    'input-gradients': ['runner/tests/test_bag_requested_gradients.py', 'runner/tests/test_bag_gradient_engine.py', 'Tests/KwipMLAppTests/BagDerivativeResultsTests.swift', 'docs/production/TORCH_BAG_REQUESTED_GRADIENTS_20261009.md'],
    'state-roundtrip': ['runner/tests/test_bag_checkpoint_identity.py', 'runner/tests/test_bag_checkpoint_modes.py', 'runner/tests/test_bag_composition.py', 'docs/production/TORCH_BAG_INPUT_LIFECYCLE_20261009.md'],
}


def test_original_four_modes_and_six_genuine_lessons_public_example():
    _, unit = delivered()
    old = next(row for row in json.loads(baseline(CATALOG))['capabilities'] if row['id'] == IDENT)
    for key in ['canonical_symbol', 'family', 'source_members', 'required_semantics', 'consolidation', 'scope_reference']:
        assert unit.get(key) == old.get(key), key
    assert unit['lesson_ids'] == ['assessment.' + task for task in TASKS] and unit['lesson_id'] in unit['lesson_ids']
    assert unit['example_ids'] == EXAMPLES and unit['example_id'] in EXAMPLES
    bundle = read('components/embeddings/bundle.json')
    registry = {row['id']: row for row in read('Sources/KwipMLCore/Resources/components.json')['components']}
    for component in ICONS:
        row = next(row for row in bundle['components'] if row['definition']['id'] == component)
        assert row['published'] is True and row['configurationVersion'] == 1
        assert row['learning']['lessonIDs'] == unit['lesson_ids'] and row['learning']['exampleID'] == EXAMPLES[0]
        assert row['learning']['sourceResource'] == 'bag-observations'
        assert registry[component] == row['definition']
    # The source is a supporting component, not a second unit in the reviewed denominator.
    assert not any(row['id'] == 'pytorch.bag_input' for row in read(CATALOG)['capabilities'])


def test_independent_literal_runtime_pair_matches_authored_contract():
    bundle = read('components/embeddings/bundle.json')
    for component, module in [('bag_input', 'bag_source'), ('embedding_bag', 'embedding_bag')]:
        tree = ast.parse((ROOT / ('runner/' + module + '.py')).read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'capability')
        descriptor = ast.literal_eval(next(node.value for node in function.body if isinstance(node, ast.Return)))
        row = next(row for row in bundle['components'] if row['definition']['id'] == component)
        assert descriptor == row['runtimeContract']
        assert descriptor['graphSchemaVersions'] == [2] and descriptor['packetSchemaVersion'] == 1
        assert descriptor['batching'] == 'wholeBags' and descriptor['integerDerivatives'] == 'unavailable'
    assert descriptor['weightedModes'] == ['sum'] and descriptor['modes'] == ['mean', 'sum', 'max']
    assert descriptor['mpsMaxNormExecution'] == descriptor['mpsTrainableFrequencyScaling'] == 'reject'


def test_current_sealed_evidence_all_forty_mode_dimension_groups():
    _, unit = delivered()
    assert len(coverage.REQUIRED_EVIDENCE) == 10
    for dimension in coverage.REQUIRED_EVIDENCE:
        bound(unit, dimension, ACCEPTANCE)
        for path in CRITICAL[dimension]: bound(unit, dimension, path)
    # GUI journeys are representative; this does not require every mode/device permutation.
    for dimension in ['configuration', 'lesson', 'example', 'ui', 'persistence']: bound(unit, dimension, NATIVE)
    for mode, paths in MODE_PROOFS.items():
        for path in paths:
            assert any(row['path'] == path and row.get('scope') == IDENT + ':' + mode and row['status'] == 'verified' and row['sha256'] == digest(path) for rows in unit['evidence'].values() for row in rows), (mode, path)



def test_actual_design_catalog_contains_both_shipped_packet_components():
    catalog = {row['id']: row for row in read('docs/design/component-icons-v2/catalog.json')['components']}
    for component, (asset, _, _) in ICONS.items():
        assert component in catalog, component
        assert catalog[component]['asset'] == asset + '.png'


def test_original_generated_icons_are_bound_to_real_design_catalog_and_resources():
    _, unit = delivered()
    catalog = {row['id']: row for row in read('docs/design/component-icons-v2/catalog.json')['components']}
    swift = (ROOT / 'Sources/KwipMLApp/ComponentArtworkCatalog.swift').read_text()
    for component, (asset, filename, pin) in ICONS.items():
        provenance = 'docs/design/component-icons-v2/' + filename
        row = read(provenance)
        original = 'docs/design/component-icons-v2/' + asset + '.png'
        resource = 'Sources/KwipMLApp/Resources/ComponentIcons/' + asset + '.png'
        assert row['generator'] == 'image_gen.imagegen' and row['transparentBackground'] is True
        assert row['asset'] == asset + '.png' and row['sha256'] == digest(original) == digest(resource) == pin
        assert (ROOT / original).read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
        assert catalog[component]['asset'] == asset + '.png'
        assert '"' + component + '":"' + asset + '"' in swift
        for path in [original, resource, provenance, 'docs/design/component-icons-v2/catalog.json']: bound(unit, 'ui', path)


@pytest.mark.parametrize('dimension', coverage.REQUIRED_EVIDENCE)
def test_missing_mode_unverified_foreign_scope_stale_or_missing_file_withholds_credit(dimension):
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
        else: rows[0]['path'] = '.local/nonexistent-bag-proof'
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


def test_prior83_recursive_non_sha_semantics_receipts_art_and_lookup_contracts_unchanged():
    old = json.loads(baseline(CATALOG)); current = read(CATALOG)
    prior = [row for row in old['capabilities'] if row.get('configuration_contract') == 'verified' and row.get('mode_scope_status') == 'reviewed' and row.get('backend_matrix', {}).get('status') == 'verified']
    assert len(prior) == 83 and len(current['capabilities']) == 88
    assert without_sha({key: value for key, value in current.items() if key not in ('capabilities', 'deferrals')}) == without_sha({key: value for key, value in old.items() if key not in ('capabilities', 'deferrals')})
    assert current['deferrals'] == {
        'status': 'owner-approved',
        'reference': 'docs/production/TORCH_V1_DEFERRAL_20261009.md',
        'units': [{'id': 'pytorch.torch.optim.LBFGS', 'reason': 'Owner deferred the last optimizer to a later release to prioritize LEARN and release closure on 2026-10-09.'}],
        'maximum_units': 8,
        'required': 'Every incomplete reviewed unit must retain an explicit owner-approved rationale; required families and original denominator remain unchanged.',
    }
    for unit in old['capabilities']:
        if unit['id'] not in {IDENT, 'pytorch.torch.utils.data.Dataset', 'pytorch.torch.utils.data.DataLoader', 'pytorch.torch.export.export'}:
            actual = next(row for row in current['capabilities'] if row['id'] == unit['id'])
            assert without_sha(actual) == without_sha(unit), unit['id']
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
    oldregistry = {row['id']: row for row in json.loads(baseline('Sources/KwipMLCore/Resources/components.json'))['components']}
    nowregistry = {row['id']: row for row in read('Sources/KwipMLCore/Resources/components.json')['components']}
    assert all(nowregistry[component] == oldregistry[component] for component in ['index_input', 'embedding'])
