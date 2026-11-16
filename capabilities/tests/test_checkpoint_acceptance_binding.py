"""Save/load is one artifact lifecycle unit, backed by current measured evidence."""
from capabilities.tests.artwork_inventory_compatibility import historical_artwork_entries
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BASELINE = 'e892367'
ID = 'pytorch.torch.save'
ACCEPTANCE = 'docs/production/TORCH_SAVE_ACCEPTANCE_20261009.md'
NATIVE = 'docs/production/TORCH_SAVE_NATIVE_20261009.md'
MODES = ['weights-save', 'continuation-save', 'weights-restore', 'continuation-restore', 'incompatible-artifact-refusal']
LESSONS = ['assessment.checkpoint-restore-weights', 'assessment.checkpoint-continue-training']
EXAMPLES = ['prepared-checkpoint-restore-weights', 'prepared-checkpoint-continue-training']
ART_SHA = '6fe639fbcae65fcd07ecec152855f731644725d281b93f1a01205abfeb2fbcc4'
spec = importlib.util.spec_from_file_location('checkpoint_binding_coverage', ROOT / 'scripts/coverage_report.py')
coverage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coverage)


def read(path):
    return json.loads((ROOT / path).read_text())


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def entry():
    bundle = read('components/lifecycle/bundle.json')
    assert bundle['kind'] == 'artifactLifecycle'
    return bundle['capabilities'][0]


def delivered():
    c = read('capabilities/catalogs/v1/pytorch.json')
    u = next(u for u in c['capabilities'] if u['id'] == ID)
    assert len(c['capabilities']) == 88
    assert u['implementation_status'] == 'delivered'
    assert u['configuration_contract'] == 'verified'
    assert u['mode_scope_status'] == 'reviewed'
    assert u['backend_matrix']['status'] == 'verified'
    assert u['backend_matrix']['reference'] == ACCEPTANCE
    assert any(case.startswith('cpu') for case in u['backend_matrix']['cases'])
    assert any(case.startswith('mps') for case in u['backend_matrix']['cases'])
    assert ID in coverage.report_catalog(c, ROOT)['complete_ids']
    return c, u


def bound(u, dimension, path):
    rows = [r for r in u['evidence'].get(dimension, []) if r['path'] == path]
    assert {ID + ':' + mode for mode in MODES} <= {r.get('scope') for r in rows}, (dimension, path)
    assert all(r['status'] == 'verified' and r['sha256'] == digest(path) for r in rows)


def test_consolidated_save_load_original_members_modes_and_real_lessons():
    _, u = delivered()
    assert u['canonical_symbol'] == 'torch.save' and u['family'] == 'serialization'
    assert u['mode_scope'] == MODES
    assert u['source_members'] == [
        {'canonical_symbol': 'torch.save', 'required_modes': MODES[:2], 'original_capability_id': ID},
        {'canonical_symbol': 'torch.load', 'required_modes': MODES[2:], 'original_capability_id': 'pytorch.torch.load'},
    ]
    assert u['lesson_ids'] == LESSONS and u['lesson_id'] in LESSONS
    assert u['example_ids'] == EXAMPLES and u['example_id'] in EXAMPLES
    e = entry()
    assert e['published'] is True and e['coverageID'] == ID and e['modes'] == MODES
    assert e['learning']['lessonIDs'] == LESSONS and e['learning']['exampleIDs'] == EXAMPLES
    assert all((ROOT / path).is_file() for path in e['learning']['sources'])
    # Lifecycle actions do not masquerade as forward nodes or palette operators.
    assert 'definition' not in e
    registry = read('Sources/KwipMLCore/Resources/components.json')
    definitions = registry if isinstance(registry, list) else registry['components']
    assert not any(d.get('id') == 'checkpoint_lifecycle' for d in definitions)
    assert [(b['id'], b['surface'], b['action']) for b in e['bindings']] == [
        ('weights-save', 'history', 'exportTrainedPackage'),
        ('continuation-save', 'history', 'retainCompletedCheckpoint'),
        ('weights-restore', 'train', 'selectParent'),
        ('continuation-restore', 'train', 'selectParent'),
    ]


def test_published_descriptor_retains_safe_owned_artifact_and_device_boundaries():
    e = entry()
    contract = e['runtimeContract']
    assert e['published'] and e['adapterID'] == contract['adapterID'] == 'kwip.runner.checkpoint-lifecycle'
    assert contract['version'] == {'major': 1, 'minor': 0, 'patch': 0}
    assert contract['checkpointSchemaVersions'] == [1] and contract['modes'] == MODES
    assert contract['sourceTrust'] == 'ownedCompletedHistory' and contract['loadPolicy'] == 'weightsOnlyCPUMap'
    assert contract['artifactLimits'] == {'manifest.json': 20_000_000, 'checkpoint.pt': 100_000_000, 'metrics.json': 4_000_000, 'rate-trace.json': 4_000_000}
    assert contract['requiredArtifacts'] == ['manifest.json', 'checkpoint.pt', 'metrics.json']
    assert contract['schedulerTracePolicy'] == 'requiredExactlyWhenParentHasScheduler'
    assert contract['exactResumeDevices'] == ['cpu'] and contract['fineTuneDevices'] == ['cpu', 'mps']
    assert contract['cpuResume'] == 'epochBoundaryExactCPU' and contract['mpsResume'] == 'weightsOnly'
    assert contract['fineTune'] == 'restoreModelStateResetApplicableTrainingState'
    assert contract['parentImmutable'] is True and contract['externalImport'] is False and contract['periodicSave'] is False
    assert contract['saveTiming'] == 'successfulRunCompletionAtEpochBoundary'
    tree = ast.parse((ROOT / 'runner/lifecycle_capabilities_generated.py').read_text())
    assignments = {t.id: n.value for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    call = assignments['CAPABILITIES']
    assert json.loads(ast.literal_eval(call.args[0])) == {'checkpoint_lifecycle': contract}
    assert 'SUPPORTED' not in assignments


def test_all_ten_dimensions_and_five_modes_bind_frozen_acceptance_and_native_proof():
    _, u = delivered()
    assert len(coverage.REQUIRED_EVIDENCE) == 10
    for dimension in coverage.REQUIRED_EVIDENCE:
        bound(u, dimension, ACCEPTANCE)
    for dimension in ['execution', 'export', 'ui', 'persistence', 'lesson', 'example']:
        bound(u, dimension, NATIVE)


def test_critical_actual_runtime_learning_history_and_privacy_bindings_are_current():
    _, u = delivered()
    required = {
        'configuration': ['components/lifecycle/bundle.json', 'Sources/KwipMLCore/LifecycleCapabilityBridge.swift', 'Sources/KwipMLCore/LifecycleRuntimeCapability.swift', 'runner/lifecycle_capabilities_generated.py', 'scripts/component_bundles.py'],
        'execution': ['runner/engine.py', 'runner/checkpoint_lifecycle.py', 'runner/tests/test_checkpoint_lifecycle_modes.py', 'docs/production/TORCH_SAVE_MODES_RUNTIME_20261009.md'],
        'validation': ['runner/frozen_checkpoint.py', 'Sources/KwipMLApp/WorkflowCheckpointInventory.swift', 'Tests/KwipMLAppTests/CheckpointLifecycleAdmissionTests.swift'],
        'export': ['Sources/KwipMLApp/WorkflowHistoryReader.swift', 'Sources/KwipMLApp/WorkflowTrainedExport.swift', 'Sources/KwipMLApp/Resources/inference.py'],
        'lesson': ['Sources/KwipMLCore/CheckpointLessonReference.swift', 'Sources/KwipMLCore/CheckpointLearningSubmission.swift', 'Sources/KwipMLCore/CheckpointFormativeProjection.swift', 'Sources/KwipMLCore/TensorLearningReceipt.swift', 'runner/learning_checkpoints.py', 'runner/learning_evidence.py', 'runner/tests/test_learning_checkpoints.py', 'Tests/KwipMLAppTests/CheckpointLearningRuntimeTests.swift', 'docs/production/TORCH_SAVE_LEARNING_APP_20261009.md'],
        'example': ['Sources/KwipMLApp/CheckpointLearningExamples.swift', 'Sources/KwipMLApp/LearnCatalogContent.swift'],
        'diagnostics': ['Sources/KwipMLApp/CheckpointMeasuredResultsView.swift', 'Sources/KwipMLCore/CheckpointFormativeProjection.swift'],
        'persistence': ['Sources/KwipMLApp/WorkflowLifecycleParentRecord.swift', 'Sources/KwipMLApp/WorkflowCheckpointSnapshot.swift', 'docs/production/TORCH_LOSS_SCHEDULER_PARENT_TRANSPORT_20261009.md'],
        'failure_handling': ['runner/tests/test_checkpoint_lifecycle_capability.py', 'Sources/KwipMLApp/WorkflowWorkspaceSession.swift', 'Sources/KwipMLApp/CapabilityActions.swift'],
    }
    for dimension, paths in required.items():
        for path in paths:
            bound(u, dimension, path)


def test_original_generated_artwork_and_registered_copy_are_identical_and_bound():
    _, u = delivered()
    source = 'docs/design/lifecycle-icons-v1/checkpoint-save-restore-v1.png'
    resource = 'Sources/KwipMLApp/Resources/ComponentIcons/checkpoint-save-restore-v1.png'
    provenance = 'docs/design/lifecycle-icons-v1/provenance-20261009.json'
    art = read(provenance)
    assert digest(source) == digest(resource) == art['sha256'] == art['resourceSHA256'] == ART_SHA
    assert art['tool'] == 'built-in image_gen' and art['destination'] == source and art['registeredResource'] == resource
    for path in [source, resource, provenance, 'Sources/KwipMLApp/ComponentArtworkCatalog.swift']:
        bound(u, 'ui', path)


def test_missing_modes_dimensions_and_unverified_rows_refuse_credit():
    c, _ = delivered()
    for dimension in coverage.REQUIRED_EVIDENCE:
        for mode in MODES:
            changed = copy.deepcopy(c)
            target = next(u for u in changed['capabilities'] if u['id'] == ID)
            target['evidence'][dimension] = [r for r in target['evidence'][dimension] if r.get('scope') != ID + ':' + mode]
            assert ID not in coverage.report_catalog(changed, ROOT)['complete_ids']
        changed = copy.deepcopy(c)
        target = next(u for u in changed['capabilities'] if u['id'] == ID)
        for row in target['evidence'][dimension]:
            row['status'] = 'unverified'
        assert ID not in coverage.report_catalog(changed, ROOT)['complete_ids']


def test_stale_or_missing_evidence_pending_backend_and_configuration_refuse_credit():
    c, _ = delivered()
    for kind in ['stale', 'missing', 'backend', 'configuration', 'scope']:
        changed = copy.deepcopy(c)
        u = next(u for u in changed['capabilities'] if u['id'] == ID)
        if kind == 'stale':
            u['evidence']['execution'][0]['sha256'] = '0' * 64
        elif kind == 'missing':
            u['evidence']['execution'][0]['path'] = '.local/nonexistent-checkpoint-acceptance-evidence'
        elif kind == 'backend':
            u['backend_matrix']['status'] = 'pending'
        elif kind == 'configuration':
            u['configuration_contract'] = 'pending'
        else:
            u['mode_scope_status'] = 'pending-operation-boundary-review'
        assert ID not in coverage.report_catalog(changed, ROOT)['complete_ids'], kind


def test_prior_78_recursive_non_sha_semantics_and_original_receipts_art_preserved():
    baseline = json.loads(subprocess.check_output(['git', 'show', BASELINE + ':capabilities/catalogs/v1/pytorch.json'], cwd=ROOT))
    current = read('capabilities/catalogs/v1/pytorch.json')
    prior = [u for u in baseline['capabilities'] if u.get('configuration_contract') == 'verified' and u.get('mode_scope_status') == 'reviewed' and u.get('backend_matrix', {}).get('status') == 'verified']
    assert len(prior) == 78 and len(current['capabilities']) == 88
    def without_sha(value):
        if isinstance(value, dict):
            return {k: without_sha(v) for k, v in value.items() if k != 'sha256'}
        if isinstance(value, list):
            return [without_sha(v) for v in value]
        return value
    for old in prior:
        new = next(u for u in current['capabilities'] if u['id'] == old['id'])
        assert without_sha(new) == without_sha(old), old['id']
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
    new_art = read('docs/design/component-icons-v2/catalog.json')
    assert historical_artwork_entries(new_art['components'])[:len(old_art['components'])] == old_art['components']
