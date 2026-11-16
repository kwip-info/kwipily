"""Remaining activation credit requires all reviewed modes and measured state/mask lessons."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
RECEIPT='docs/production/TORCH_REMAINING_ACTIVATION_ACCEPTANCE_20261008.md'
BASE_MODES=['forward','input-gradients','shape-contract']
MEMBERS={'PReLU':('prelu',['prelu-shared-slope','prelu-channel-slopes'],BASE_MODES+['training','state-roundtrip']), 'LogSoftmax':('log_softmax',['logsoftmax-sample-scores'],BASE_MODES), 'Dropout':('dropout',['dropout-training-pattern','dropout-evaluation-pattern'],BASE_MODES+['training','inference','seeded-repeat'])}
spec=importlib.util.spec_from_file_location('remaining_activation_binding_coverage',ROOT/'scripts/coverage_report.py')
coverage=importlib.util.module_from_spec(spec);spec.loader.exec_module(coverage)
@pytest.fixture(params=MEMBERS)
def binding(request):
    catalog=json.loads((ROOT/'capabilities/catalogs/v1/pytorch.json').read_text())
    unit=next(x for x in catalog['capabilities'] if x['id']=='pytorch.torch.nn.'+request.param)
    return catalog,unit,request.param

def test_member_review_identity_devices_and_all_assessed_identities(binding):
    catalog,unit,name=binding;kind,tasks,MODES=MEMBERS[name]
    result=coverage.report_catalog(catalog,ROOT)
    assert unit['id'] in result['complete_ids']
    assert unit['family']=='activation' and unit['mode_scope']==MODES
    assert unit['mode_scope_status']=='reviewed' and unit['configuration_contract']=='verified'
    assert unit['implementation_status']=='delivered'
    assert unit['backend_matrix']['status']=='verified' and unit['backend_matrix']['reference']==RECEIPT
    assert set(unit['backend_matrix']['cases'])=={'cpu-float32','mps-float32'}
    assert unit['lesson_id']=='assessment.'+tasks[0]
    assert unit['lesson_ids']==['assessment.'+task for task in tasks]
    assert unit['example_id']=={'PReLU':'prepared-prelu-classifier','LogSoftmax':'prepared-log-softmax-classifier','Dropout':'prepared-dropout-classifier'}[name]
    manifest=json.loads((ROOT/'components/activations/bundle.json').read_text())
    entry=next(r for r in manifest['components'] if r['definition']['id']==kind)
    assert entry['learning']['lessonIDs']==unit['lesson_ids']
    assert entry['learning']['exampleID']==unit['example_id']
    assert entry['presentation']['icon']==kind+'-v2'
    assert result['minimum_complete_units']==80 and len(catalog['capabilities'])==88
    # Reporter retains the historical minimum; owner's all-88 delivery goal is separate.
    assert not result['release_gate_passed']

def test_actual_frozen_receipt_binds_ten_dimensions_and_every_reviewed_mode(binding):
    _,unit,name= binding;MODES=MEMBERS[name][2];path=ROOT/RECEIPT
    assert path.is_file(),'Actual native/export acceptance must be frozen; no placeholder earns credit'
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    for dimension in coverage.REQUIRED_EVIDENCE:
        for mode in MODES:
            rows=[r for r in unit['evidence'].get(dimension,[]) if r['path']==RECEIPT and r.get('scope')==unit['id']+':'+mode]
            assert rows,(unit['id'],dimension,mode)
            assert all(r['status']=='verified' and r['sha256']==digest for r in rows)

def test_stale_real_config_worker_assessment_icon_or_runtime_sources_withhold_credit(binding):
    catalog,unit,name=binding;kind,_,MODES=MEMBERS[name]
    assert unit['id'] in coverage.report_catalog(catalog,ROOT)['complete_ids']
    dependencies=[
        ('configuration','components/activations/bundle.json'),
        ('configuration','scripts/component_bundles.py'),
        ('execution','runner/remaining_activation_configuration.py'),
        ('execution','runner/engine.py'),
        ('execution','runner/tests/test_remaining_activation_runtime.py'),
        ('execution','runner/remaining_activation_adapters_generated.py'),
        ('validation','Sources/KwipMLCore/RemainingActivationConfiguration.swift'),
        ('validation','Sources/KwipMLCore/RemainingActivationCapabilityBridge.swift'),
        ('validation','Sources/KwipMLCore/RemainingActivationRuntimeCapabilities.swift'),
        ('lesson','Sources/KwipMLCore/RemainingActivationLearningTask.swift'),
        ('lesson','Sources/KwipMLCore/RemainingActivationLearningSubmission.swift'),
        ('lesson','Sources/KwipMLCore/RemainingActivationLessonReference.swift'),
        ('lesson','runner/learning_remaining_activation.py'),
        ('lesson','runner/tests/test_learning_remaining_activation.py'),
        ('lesson','runner/learning_evidence.py'),
        ('ui','Sources/KwipMLApp/Resources/ComponentIcons/'+kind+'-v2.png'),
        ('ui','docs/design/component-icons-v2/remaining-activation-provenance-20261008.json'),
        ('ui','docs/design/component-icons-v2/catalog.json'),
        ('ui','Tests/KwipMLAppTests/RemainingActivationArtworkTests.swift'),
        ('persistence','Tests/KwipMLAppTests/RemainingActivationLearningStoreTests.swift'),
        ('diagnostics','Sources/KwipMLApp/CapabilityActions.swift'),
        ('failure_handling','Tests/KwipMLAppTests/RemainingActivationRuntimeAdmissionTests.swift')]
    if name in ('PReLU','Dropout'):
        dependencies.append(('persistence','runner/checkpoint_lifecycle.py'))
    if name=='Dropout':
        dependencies += [('lesson','runner/fixtures/remaining_activation_dropout_masks_v1.json'),
                         ('lesson','runner/fixtures/generate_remaining_activation_dropout_masks.py'),
                         ('lesson','Sources/KwipMLCore/Resources/remaining_activation_dropout_masks_v1.json')]

    for dimension,path in dependencies:
        changed=copy.deepcopy(catalog);target=next(c for c in changed['capabilities'] if c['id']==unit['id'])
        rows=[r for r in target['evidence'].get(dimension,[]) if r['path']==path]
        assert rows,(unit['id'],dimension,path)
        assert {unit['id']+':'+m for m in MODES}<={r.get('scope') for r in rows}
        rows[0]['sha256']='0'*64
        assert unit['id'] not in coverage.report_catalog(changed,ROOT)['complete_ids']

def test_each_missing_dimension_or_wrong_mode_scope_withholds_credit(binding):
    catalog,unit,name=binding;MODES=MEMBERS[name][2]
    assert unit['id'] in coverage.report_catalog(catalog,ROOT)['complete_ids']
    for dimension in coverage.REQUIRED_EVIDENCE:
        for mode in MODES:
            changed=copy.deepcopy(catalog);target=next(c for c in changed['capabilities'] if c['id']==unit['id'])
            # Evidence belonging to another mode must not fill the missing dimension.
            target['evidence'][dimension]=[dict(r,scope=unit['id']+':unreviewed-mode') if r.get('scope')==unit['id']+':'+mode else r for r in target['evidence'][dimension]]
            assert unit['id'] not in coverage.report_catalog(changed,ROOT)['complete_ids'],(dimension,mode)

def test_pending_scope_config_backend_or_unverified_rows_cannot_earn_credit(binding):
    catalog,unit,name=binding;MODES=MEMBERS[name][2]
    assert unit['id'] in coverage.report_catalog(catalog,ROOT)['complete_ids']
    for field,value in [('mode_scope_status','pending-operation-boundary-review'),('configuration_contract','pending'),('backend_matrix',{'status':'pending','cases':['cpu-float32','mps-float32']})]:
        changed=copy.deepcopy(catalog);target=next(c for c in changed['capabilities'] if c['id']==unit['id']);target[field]=value
        assert unit['id'] not in coverage.report_catalog(changed,ROOT)['complete_ids']
    changed=copy.deepcopy(catalog);target=next(c for c in changed['capabilities'] if c['id']==unit['id'])
    for row in target['evidence']['lesson']:row['status']='present'
    assert unit['id'] not in coverage.report_catalog(changed,ROOT)['complete_ids']

def test_stale_receipt_and_missing_forward_or_influence_proof_refuse(binding):
    catalog,unit,name=binding;MODES=MEMBERS[name][2]
    assert unit['id'] in coverage.report_catalog(catalog,ROOT)['complete_ids']
    for mode in MODES:
        changed=copy.deepcopy(catalog);target=next(c for c in changed['capabilities'] if c['id']==unit['id'])
        rows=[r for r in target['evidence']['execution'] if r['path']==RECEIPT and r.get('scope')==unit['id']+':'+mode]
        assert rows
        rows[0]['sha256']='f'*64
        assert unit['id'] not in coverage.report_catalog(changed,ROOT)['complete_ids']
        changed=copy.deepcopy(catalog);target=next(c for c in changed['capabilities'] if c['id']==unit['id'])
        target['evidence']['execution']=[r for r in target['evidence']['execution'] if r.get('scope')!=unit['id']+':'+mode]
        assert unit['id'] not in coverage.report_catalog(changed,ROOT)['complete_ids']
