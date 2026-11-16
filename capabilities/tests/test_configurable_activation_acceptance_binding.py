"""Activation family credit requires each member's exact contract and three modes."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[2]
RECEIPT='docs/production/TORCH_CONFIGURABLE_ACTIVATION_ACCEPTANCE_20261008.md'
MODES=['forward','input-gradients','shape-contract']
MEMBERS={'SiLU':('silu','silu-values'), 'LeakyReLU':('leaky_relu','leaky-relu-slope'), 'ELU':('elu','elu-alpha')}
spec=importlib.util.spec_from_file_location('activation_binding_coverage',ROOT/'scripts/coverage_report.py')
coverage=importlib.util.module_from_spec(spec);spec.loader.exec_module(coverage)

@pytest.fixture(params=MEMBERS)
def binding(request):
    catalog=json.loads((ROOT/'capabilities/catalogs/v1/pytorch.json').read_text())
    unit=next(x for x in catalog['capabilities'] if x['id']=='pytorch.torch.nn.'+request.param)
    return catalog,unit,request.param

def test_member_has_reviewed_modes_backend_and_own_assessed_identity(binding):
    catalog,unit,name=binding;kind,task=MEMBERS[name]
    result=coverage.report_catalog(catalog,ROOT)
    assert unit['id'] in result['complete_ids']
    assert unit['mode_scope']==MODES and unit['mode_scope_status']=='reviewed'
    assert unit['configuration_contract']=='verified'
    assert unit['backend_matrix']['status']=='verified' and unit['backend_matrix']['reference']==RECEIPT
    assert {'cpu-float32','mps-float32'}<=set(unit['backend_matrix']['cases'])
    assert unit['lesson_id']=='assessment.'+task and unit['lesson_ids']==['assessment.'+task]
    assert unit['example_id']=='prepared-'+kind+'-classifier'
    assert result['minimum_complete_units']==80 and not result['release_gate_passed']

def test_receipt_binds_every_dimension_and_mode(binding):
    _,unit,_=binding;path=ROOT/RECEIPT
    assert path.is_file(),'Freeze acceptance before awarding any member credit'
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    for dimension in coverage.REQUIRED_EVIDENCE:
        for mode in MODES:
            rows=[r for r in unit['evidence'].get(dimension,[]) if r['path']==RECEIPT and r.get('scope')==unit['id']+':'+mode]
            assert rows,(unit['id'],dimension,mode)
            assert all(r['status']=='verified' and r['sha256']==digest for r in rows)

def test_actual_dependency_changes_withhold_member_credit(binding):
    catalog,unit,name=binding
    dependencies=[('execution','runner/configurable_activation_configuration.py'),('execution','runner/engine.py'),
                  ('execution','runner/tests/test_configurable_activations.py'),('configuration','components/activations/bundle.json'),
                  ('lesson','Sources/KwipMLCore/ConfigurableActivationLearningTask.swift'),('lesson','runner/learning_configurable_activation.py'),
                  ('validation','Sources/KwipMLCore/'+name+'Component.swift'),('ui','Sources/KwipMLApp/Resources/ComponentIcons/'+MEMBERS[name][0]+'-v2.png')]
    for dimension,path in dependencies:
        changed=copy.deepcopy(catalog);target=next(c for c in changed['capabilities'] if c['id']==unit['id'])
        rows=[r for r in target['evidence'].get(dimension,[]) if r['path']==path]
        assert rows,(unit['id'],dimension,path)
        assert {unit['id']+':'+m for m in MODES}<={r.get('scope') for r in rows}
        rows[0]['sha256']='0'*64
        assert unit['id'] not in coverage.report_catalog(changed,ROOT)['complete_ids']

def test_missing_private_lesson_or_shape_evidence_withholds_credit(binding):
    catalog,unit,_=binding
    for dimension,mode in [('persistence','shape-contract'),('lesson','forward'),('lesson','input-gradients')]:
        changed=copy.deepcopy(catalog);target=next(c for c in changed['capabilities'] if c['id']==unit['id'])
        target['evidence'][dimension]=[r for r in target['evidence'].get(dimension,[]) if r.get('scope')!=unit['id']+':'+mode]
        assert unit['id'] not in coverage.report_catalog(changed,ROOT)['complete_ids']

def test_unreviewed_modes_or_backend_boundary_withholds_credit(binding):
    catalog,unit,_=binding
    for field,value in [('mode_scope_status','pending-operation-boundary-review'),('backend_matrix',{'status':'pending','cases':['cpu-float32','mps-float32']})]:
        changed=copy.deepcopy(catalog);target=next(c for c in changed['capabilities'] if c['id']==unit['id']);target[field]=value
        assert unit['id'] not in coverage.report_catalog(changed,ROOT)['complete_ids']
