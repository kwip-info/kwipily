"""Each family member needs its own mode-scoped evidence despite a shared receipt."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[2]
RECEIPT='docs/production/TORCH_CONVOLUTION_ACCEPTANCE_20261008.md'
MODES=['inference','training','input-gradients','state-roundtrip']
IDS=[f'pytorch.torch.nn.Conv{dim}d' for dim in (1,2,3)]
spec=importlib.util.spec_from_file_location('convolution_binding_coverage',ROOT/'scripts/coverage_report.py')
coverage=importlib.util.module_from_spec(spec);spec.loader.exec_module(coverage)


@pytest.fixture(params=IDS)
def binding(request):
    catalog=json.loads((ROOT/'capabilities/catalogs/v1/pytorch.json').read_text())
    unit=next(x for x in catalog['capabilities'] if x['id']==request.param)
    return catalog,unit


def test_reviewed_member_has_all_four_modes_and_own_assessed_identity(binding):
    catalog,unit=binding;kind=unit['canonical_symbol'].split('.')[-1].lower()
    result=coverage.report_catalog(catalog,ROOT)
    assert unit['id'] in result['complete_ids']
    assert unit['mode_scope']==MODES and unit['mode_scope_status']=='reviewed'
    assert unit['configuration_contract']=='verified'
    assert unit['backend_matrix']['status']=='verified'
    assert unit['backend_matrix']['reference']==RECEIPT
    assert {'cpu-float32','mps-float32'}<=set(unit['backend_matrix']['cases'])
    assert unit['lesson_id']==f'assessment.{kind}-shared-kernel'
    assert unit['lesson_ids']==[f'assessment.{kind}-shared-kernel']
    assert unit['example_id']==f'prepared-{kind}-classifier'
    assert result['minimum_complete_units']==80 and not result['release_gate_passed']


def test_shared_receipt_binds_each_member_dimension_and_mode(binding):
    _,unit=binding;path=ROOT/RECEIPT
    assert path.is_file(),'The family needs a frozen acceptance receipt before any member earns credit'
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    for dimension in coverage.REQUIRED_EVIDENCE:
        for mode in MODES:
            rows=[r for r in unit['evidence'].get(dimension,[]) if r['path']==RECEIPT and r.get('scope')==f"{unit['id']}:{mode}"]
            assert rows,(unit['id'],dimension,mode)
            assert all(r.get('status')=='verified' and r.get('sha256')==digest for r in rows)


def test_own_worker_and_shared_dependencies_cannot_hide_behind_family_receipt(binding):
    catalog,unit=binding;kind=unit['canonical_symbol'].split('.')[-1].lower()
    dependencies=[('execution',f'runner/{kind}_configuration.py'),('execution','runner/engine.py'),
                  ('execution','runner/convolution_geometry.py'),('execution','runner/tests/test_convolution_independent.py'),
                  ('configuration','components/convolution/bundle.json'),
                  ('lesson','Sources/KwipMLCore/ConvolutionLearningTask.swift'),
                  ('lesson','runner/learning_convolution.py'),
                  ('validation',f'Sources/KwipMLCore/Conv{kind[4]}dCapabilityBridge.swift')]
    for dimension,path in dependencies:
        changed=copy.deepcopy(catalog);target=next(c for c in changed['capabilities'] if c['id']==unit['id'])
        rows=[r for r in target['evidence'].get(dimension,[]) if r['path']==path]
        assert rows,(unit['id'],dimension,path,'Bind actual source, not only a receipt')
        assert {f"{unit['id']}:{m}" for m in MODES}<={r.get('scope') for r in rows}
        rows[0]['sha256']='0'*64
        assert unit['id'] not in coverage.report_catalog(changed,ROOT)['complete_ids']


def test_missing_state_or_private_lesson_evidence_withholds_member_credit(binding):
    catalog,unit=binding
    for dimension,mode in [('persistence','state-roundtrip'),('lesson','training'),('lesson','input-gradients')]:
        changed=copy.deepcopy(catalog);target=next(c for c in changed['capabilities'] if c['id']==unit['id'])
        target['evidence'][dimension]=[r for r in target['evidence'].get(dimension,[]) if r.get('scope')!=f"{unit['id']}:{mode}"]
        assert unit['id'] not in coverage.report_catalog(changed,ROOT)['complete_ids']


def test_unreviewed_mode_or_backend_boundary_withholds_member_credit(binding):
    catalog,unit=binding
    for field,value in [('mode_scope_status','pending-operation-boundary-review'),('backend_matrix',{'status':'pending','cases':['cpu-float32','mps-float32']})]:
        changed=copy.deepcopy(catalog);target=next(c for c in changed['capabilities'] if c['id']==unit['id'])
        target[field]=value
        assert unit['id'] not in coverage.report_catalog(changed,ROOT)['complete_ids']
