"""Four pooling units need their own frozen evidence, assets and assessed lifecycle."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
RECEIPT='docs/production/TORCH_ADAPTIVE_POOLING_ACCEPTANCE_20261008.md'
MODES=['forward','input-gradients','shape-contract']
MEMBERS={'AvgPool2d':('avg_pool2d','avg-pool2d-windows'), 'AdaptiveAvgPool1d':('adaptive_avg_pool1d','adaptive-avg-pool1d-bins'), 'AdaptiveAvgPool2d':('adaptive_avg_pool2d','adaptive-avg-pool2d-bins'), 'AdaptiveMaxPool2d':('adaptive_max_pool2d','adaptive-max-pool2d-bins')}
spec=importlib.util.spec_from_file_location('adaptive_binding_coverage',ROOT/'scripts/coverage_report.py')
coverage=importlib.util.module_from_spec(spec);spec.loader.exec_module(coverage)
@pytest.fixture(params=MEMBERS)
def binding(request):
    catalog=json.loads((ROOT/'capabilities/catalogs/v1/pytorch.json').read_text())
    unit=next(x for x in catalog['capabilities'] if x['id']=='pytorch.torch.nn.'+request.param)
    return catalog,unit,request.param

def test_member_is_reviewed_with_own_assessed_identity_and_honest_device_boundary(binding):
    catalog,unit,name=binding;kind,task=MEMBERS[name]
    result=coverage.report_catalog(catalog,ROOT)
    assert unit['id'] in result['complete_ids']
    assert unit['mode_scope']==MODES and unit['mode_scope_status']=='reviewed'
    assert unit['implementation_status']=='delivered' and unit['configuration_contract']=='verified'
    assert unit['backend_matrix']['status']=='verified' and unit['backend_matrix']['reference']==RECEIPT
    assert {'cpu-float32','mps-float32'}<=set(unit['backend_matrix']['cases'])
    assert 'fallback' in unit['notes'].lower(), 'A mixed-device computation must not be presented as blanket native MPS'
    assert unit['lesson_id']=='assessment.'+task and unit['lesson_ids']==['assessment.'+task]
    assert unit['example_id']=='prepared-'+kind.replace('_','')+'-classifier'
    assert result['minimum_complete_units']==80 and len(catalog['capabilities'])==88
    assert not result['release_gate_passed']

def test_actual_frozen_receipt_binds_all_ten_dimensions_and_every_mode(binding):
    _,unit,_=binding;path=ROOT/RECEIPT
    assert path.is_file(),'Freeze real native/export acceptance before awarding any member credit'
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    for dimension in coverage.REQUIRED_EVIDENCE:
        for mode in MODES:
            rows=[r for r in unit['evidence'].get(dimension,[]) if r['path']==RECEIPT and r.get('scope')==unit['id']+':'+mode]
            assert rows,(unit['id'],dimension,mode)
            assert all(r['status']=='verified' and r['sha256']==digest for r in rows)

def test_real_component_lesson_icon_and_shared_dependency_changes_withhold_credit(binding):
    catalog,unit,name=binding;kind,_=MEMBERS[name]
    assert unit['id'] in coverage.report_catalog(catalog,ROOT)['complete_ids']
    helper='runner/avgpool2d_configuration.py' if name=='AvgPool2d' else 'runner/adaptive_pooling_configuration.py'
    native='Sources/KwipMLCore/AvgPool2dConfiguration.swift' if name=='AvgPool2d' else 'Sources/KwipMLCore/AdaptivePoolingConfiguration.swift'
    dependencies=[('execution',helper),('execution','runner/engine.py'),('execution','runner/tests/test_adaptive_pooling_engine.py'),('configuration','components/pooling/bundle.json'),('validation',native),('lesson','Sources/KwipMLCore/AdaptivePoolingLearningTask.swift'),('lesson','Sources/KwipMLCore/AdaptivePoolingLearningSubmission.swift'),('lesson','runner/learning_adaptive_pooling.py'),('lesson','runner/learning_evidence.py'),('ui','Sources/KwipMLApp/Resources/ComponentIcons/'+kind+'-v2.png'),('ui','docs/design/component-icons-v2/ADAPTIVE_POOLING_RECEIPT_20261008.md'),('persistence','Tests/KwipMLAppTests/AdaptivePoolingLearningStoreTests.swift'),('diagnostics','Sources/KwipMLApp/PoolingExecutionWarning.swift'),('ui','Sources/KwipMLApp/TensorLearningCanvasAdapter.swift'),('validation','Sources/KwipMLCore/PoolingRuntimeCapabilities.swift'),('ui','docs/design/component-icons-v2/catalog.json'),('ui','Tests/KwipMLAppTests/ComponentArtworkCoverageTests.swift')]
    for dimension,path in dependencies:
        changed=copy.deepcopy(catalog);target=next(c for c in changed['capabilities'] if c['id']==unit['id'])
        rows=[r for r in target['evidence'].get(dimension,[]) if r['path']==path]
        assert rows,(unit['id'],dimension,path)
        assert {unit['id']+':'+m for m in MODES}<={r.get('scope') for r in rows}
        rows[0]['sha256']='0'*64
        assert unit['id'] not in coverage.report_catalog(changed,ROOT)['complete_ids']

def test_missing_shape_private_lesson_or_port_evidence_withholds_member_credit(binding):
    catalog,unit,_=binding
    assert unit['id'] in coverage.report_catalog(catalog,ROOT)['complete_ids']
    for dimension,mode in [('persistence','shape-contract'),('lesson','forward'),('lesson','input-gradients'),('validation','shape-contract'),('ui','forward')]:
        changed=copy.deepcopy(catalog);target=next(c for c in changed['capabilities'] if c['id']==unit['id'])
        target['evidence'][dimension]=[r for r in target['evidence'][dimension] if r.get('scope')!=unit['id']+':'+mode]
        assert unit['id'] not in coverage.report_catalog(changed,ROOT)['complete_ids']

def test_pending_configuration_modes_or_device_matrix_cannot_earn_credit(binding):
    catalog,unit,_=binding
    assert unit['id'] in coverage.report_catalog(catalog,ROOT)['complete_ids']
    for field,value in [('mode_scope_status','pending-operation-boundary-review'),('configuration_contract','pending'),('backend_matrix',{'status':'pending','cases':['cpu-float32','mps-float32']})]:
        changed=copy.deepcopy(catalog);target=next(c for c in changed['capabilities'] if c['id']==unit['id']);target[field]=value
        assert unit['id'] not in coverage.report_catalog(changed,ROOT)['complete_ids']

def test_stale_receipt_or_unproven_mode_cannot_hide_behind_current_sources(binding):
    catalog,unit,_=binding
    assert unit['id'] in coverage.report_catalog(catalog,ROOT)['complete_ids']
    for mutation in ['stale','unproven-mode']:
        changed=copy.deepcopy(catalog);target=next(c for c in changed['capabilities'] if c['id']==unit['id'])
        rows=[r for r in target['evidence']['execution'] if r['path']==RECEIPT]
        assert rows
        if mutation=='stale':rows[0]['sha256']='f'*64
        else:
            target['evidence']['execution']=[r for r in target['evidence']['execution'] if r.get('scope')!=unit['id']+':forward']
        assert unit['id'] not in coverage.report_catalog(changed,ROOT)['complete_ids']
