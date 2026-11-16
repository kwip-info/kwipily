"""Actual source admission and native lesson behavior, with scalar literal answers."""
import copy
import json
from pathlib import Path
import pytest
from runner import module_composition_learning as lesson

REGISTRY = Path(__file__).resolve().parents[2]/'Sources/KwipMLCore/Resources/components.json'
INPUTS = dict(coefficients=dict(shape=[4],values=[2.,1.,3.,-2.]),rows=dict(shape=[3,1],values=[-2.,.5,3.]),upstream=dict(shape=[3,1],values=[1.,-2.,.5]),targets=dict(shape=[3,1],values=[1.,-1.,.5]))

def submission(task):
    return dict(schemaVersion=1,kind='compositionLearningGraph',taskID=task,source=lesson.source_template(task))

@pytest.mark.parametrize('task',lesson.TASKS)
def test_real_complete_source_admitted_without_mutation_and_has_actual_plan(task):
    value=submission(task);before=copy.deepcopy(value)
    admitted=lesson.configuration(value,task=task,registry=REGISTRY)
    assert admitted['expandedGraph']['compositionPlan']['root']=='root'
    assert admitted['consumedDefinitions']
    assert value==before

@pytest.mark.parametrize('task',lesson.TASKS)
def test_unknown_answers_foreign_task_and_protected_width_refuse(task):
    value=submission(task)
    for key,answer in [('answers',[0]),('weights',[1]),('success',True)]:
        bad=copy.deepcopy(value);bad[key]=answer
        with pytest.raises(ValueError):lesson.configuration(bad,task=task,registry=REGISTRY)
    with pytest.raises(ValueError):lesson.configuration(value,task=next(t for t in lesson.TASKS if t!=task),registry=REGISTRY)
    bad=copy.deepcopy(value);bad['source']['definitions'][0]['body']['nodes'][0]['config']['out_features']=2
    with pytest.raises(ValueError):lesson.configuration(bad,task=task,registry=REGISTRY)

@pytest.mark.parametrize('task',lesson.TASKS)
def test_actual_native_measurement_has_five_zero_difference_criteria_and_ordered_metadata(task):
    result=lesson.measure(submission(task),INPUTS,task=task,registry=REGISTRY)
    assert len(result['tensors'])==5
    for tensor in result['tensors'].values():
        assert all(abs(v)<=2e-5 for v in tensor['values'])
    assert result['metadata']['nativeOwners'] and result['metadata']['invocations']
    assert result['metadata']['savedStateKeys']
    assert result['metadata']['independentStorage'] is True

@pytest.mark.parametrize('task',lesson.TASKS)
def test_structurally_valid_wrong_construction_is_measured_as_failed_not_transport_refusal(task):
    value=submission(task)
    source=value['source']
    if task=='composition-build-reuse':
        block=source['definitions'][0]
        # Keeping the same admitted endpoint names, route B twice and omit A's join branch.
        join=next(e for e in block['body']['edges'] if e['targetPort']=='a')
        join['source']='b'
    else:
        sequence=next(d for d in source['definitions'] if d.get('composition',{}).get('kind')=='sequential')
        sequence['composition']['children'].reverse()
        sequence['body']['edges'][0].update(source='b',target='a')
        sequence['inputs'][0]['targets'][0]['componentID']='b'
        sequence['outputs'][0]['source']['componentID']='a'
    lesson.configuration(value,task=task,registry=REGISTRY)
    measured=lesson.measure(value,INPUTS,task=task,registry=REGISTRY)
    assert any(abs(v)>2e-5 for t in measured['tensors'].values() for v in t['values'])

import torch
from runner.module_composition_reference import FIXTURES
DEVICES=['cpu']+(['mps'] if torch.backends.mps.is_available() else [])

@pytest.mark.parametrize('task',lesson.TASKS)
@pytest.mark.parametrize('device',DEVICES)
def test_all_four_independent_scalar_fixtures_actual_native_saved_full_micro_and_isolated_cli(task,device):
    import time
    start=time.monotonic()
    for fixture in FIXTURES:
        p=fixture.parameters
        inputs=copy.deepcopy(INPUTS)
        inputs['coefficients']['values']=[p.a,p.b,p.c,p.d]
        inputs['rows']['values']=list(fixture.rows)
        inputs['upstream']['values']=list(fixture.upstream)
        result=lesson.measure(submission(task),inputs,task=task,registry=REGISTRY,device=device)
        assert all(abs(v)<=2e-5 for t in result['tensors'].values() for v in t['values'])
        assert all(c['matches'] for c in result['metadata']['nativeContainers'])
        assert len(result['metadata']['nativeOwners'])==len(result['metadata']['savedStateKeys'])
    assert time.monotonic()-start<60, 'Four tiny real lifecycle fixtures must fit the existing assessment budget.'

@pytest.mark.parametrize('bad',['bool','unknown','missing','rows','nonfinite','commuting'])
def test_strict_fixture_refuses_untrusted_malformed_or_indistinguishable_tables(bad):
    inputs=copy.deepcopy(INPUTS)
    if bad=='bool':inputs['coefficients']['values'][0]=True
    elif bad=='unknown':inputs['answer']=inputs['rows']
    elif bad=='missing':del inputs['upstream']
    elif bad=='rows':inputs['rows']['values'][1]=inputs['rows']['values'][0]
    elif bad=='nonfinite':inputs['targets']['values'][0]=float('nan')
    else:inputs['coefficients']['values']=[1.,0.,1.,0.]
    with pytest.raises(ValueError):lesson.measure(submission(lesson.TASKS[0]),inputs,task=lesson.TASKS[0],registry=REGISTRY)

@pytest.mark.parametrize('task',lesson.TASKS)
def test_changed_native_container_kind_is_failed_structural_criterion(task):
    value=submission(task)
    target=value['source']['definitions'][0]
    if task==lesson.TASKS[0]:
        target['composition']['kind']='namedMap'
        for n,child in enumerate(target['composition']['children']):child['key']='child'+str(n)
    else:target['composition']['kind']='orderedList'
    result=lesson.measure(value,INPUTS,task=task,registry=REGISTRY)
    assert any(abs(v)>2e-5 for t in result['tensors'].values() for v in t['values'])


def test_four_fixture_transport_preserves_identity_cancels_and_never_copies_answers():
    fixtures=[dict(id='f'+str(i),inputs=copy.deepcopy(INPUTS),expected={'privateAnswer':[999]}) for i in range(4)]
    evidence=dict(fixtures=[],maximumTensorElements=0)
    lesson.execute(submission(lesson.TASKS[0]),fixtures,evidence,lambda:False,None,task=lesson.TASKS[0],registry=REGISTRY)
    assert [f['fixtureID'] for f in evidence['fixtures']]==['f0','f1','f2','f3']
    assert 'privateAnswer' not in json.dumps(evidence) and 'sourceDigest' not in json.dumps(evidence)
    assert all('composition' in fixture for fixture in evidence['fixtures'])
    with pytest.raises(ValueError):lesson.execute(submission(lesson.TASKS[0]),fixtures,dict(fixtures=[]),lambda:True,None,task=lesson.TASKS[0],registry=REGISTRY)


def test_isolated_packaged_loose_worker_layout_uses_only_explicit_copied_registry(tmp_path):
    import shutil,subprocess,sys
    worker=tmp_path/'worker'
    shutil.copytree(REGISTRY.parents[3]/'runner',worker,ignore=shutil.ignore_patterns('tests','__pycache__','*.pyc'))
    registry=worker/'components.json';shutil.copy2(REGISTRY,registry)
    request=tmp_path/'lesson.json';request.write_text(json.dumps(dict(submission=submission(lesson.TASKS[1]),inputs=INPUTS)))
    script="import json,sys;sys.path.insert(0,sys.argv[1]);import module_composition_learning as m;r=json.load(open(sys.argv[3]));x=m.measure(r['submission'],r['inputs'],task=m.TASKS[1],registry=sys.argv[2]);assert all(abs(v)<2e-5 for t in x['tensors'].values() for v in t['values'])"
    result=subprocess.run([sys.executable,'-I','-c',script,str(worker),str(registry),str(request)],cwd=tmp_path,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr


def test_actual_saved_state_value_tampering_fails_even_if_affine_outputs_cancel(monkeypatch):
    original=torch.save;count=0
    def altered(value,destination,*args,**kwargs):
        nonlocal count
        count+=1
        value=copy.deepcopy(value)
        if count==1:
            keys=list(value)
            value[keys[0]].mul_(2)
        return original(value,destination,*args,**kwargs)
    monkeypatch.setattr(torch,'save',altered)
    measured=lesson.measure(submission(lesson.TASKS[0]),INPUTS,task=lesson.TASKS[0],registry=REGISTRY)
    assert any(abs(v)>2e-5 for v in measured['tensors']['savedReuse']['values'])


def test_structural_failure_marker_cannot_cancel_a_negative_numeric_difference():
    value=submission(lesson.TASKS[0]);inner=value['source']['definitions'][0]
    inner['composition']['kind']='namedMap'
    for n,child in enumerate(inner['composition']['children']):child['key']='child'+str(n)
    next(e for e in inner['body']['edges'] if e['targetPort']=='a')['source']='b'
    inputs=copy.deepcopy(INPUTS);inputs['rows']['values'][0]=-.125
    measured=lesson.measure(value,inputs,task=lesson.TASKS[0],registry=REGISTRY)
    # This role's first cell is explicitly a structural-failure marker on mismatch.
    assert measured['tensors']['branchOutputs']['values'][0]>=1


def test_native_owner_metadata_reports_actual_selected_fixed_flags():
    value=submission(lesson.TASKS[0]);expanded=lesson.configuration(value,task=lesson.TASKS[0],registry=REGISTRY)
    measured=lesson.measure(value,INPUTS,task=lesson.TASKS[0],registry=REGISTRY)
    owners=measured['metadata']['nativeOwners']
    fixed=[m for m in owners if expanded['nodeOrigins'][m['node']]['localNodeID'] in ('negate','difference')]
    assert len(fixed)==4
    assert all(m['requiresGrad'] is False for m in fixed)
    first=[m for m in owners if expanded['nodeOrigins'][m['node']]['instancePath']==['block','left'] and expanded['nodeOrigins'][m['node']]['localNodeID']=='a' and m['parameter']=='weight']
    assert len(first)==1 and first[0]['requiresGrad'] is False


def revised_private_source():
    value=submission(lesson.TASKS[0]);source=value['source']
    inner=copy.deepcopy(source['definitions'][0]);inner['reference']['id']='private-inner-copy'
    outer=copy.deepcopy(source['definitions'][1]);outer['reference']['id']='private-outer-copy'
    outer['body']['instances'][0]['definition']=copy.deepcopy(inner['reference'])
    source['definitions'].extend([inner,outer])
    source['root']['instances'][0]['definition']=copy.deepcopy(outer['reference'])
    return value


def test_real_copy_on_write_references_preserve_private_paths_and_native_measurements():
    value=revised_private_source();before=copy.deepcopy(value)
    expansion=lesson.configuration(value,task=lesson.TASKS[0],registry=REGISTRY)
    assert value==before
    assert len(expansion['consumedDefinitions'])==3
    measured=lesson.measure(value,INPUTS,task=lesson.TASKS[0],registry=REGISTRY)
    assert all(abs(v)<=2e-5 for t in measured['tensors'].values() for v in t['values'])
    assert measured['metadata']['independentStorage'] is True


@pytest.mark.parametrize('change',['retired','reachable','foreign-role','dangling','cycle'])
def test_copy_on_write_never_erases_protected_retired_or_reachable_membership(change):
    value=revised_private_source();source=value['source']
    if change=='retired':source['definitions'][0]['body']['nodes'][0]['config']['bias']=False
    elif change=='reachable':source['definitions'][-2]['body']['nodes'][0]['config']['out_features']=2
    elif change=='foreign-role':source['definitions'][-1]['body']['instances'][0]['definition']=copy.deepcopy(source['definitions'][1]['reference'])
    elif change=='dangling':source['definitions'][-1]['body']['instances'][0]['definition']['id']='absent'
    else:source['definitions'][-1]['body']['instances'][0]['definition']=copy.deepcopy(source['definitions'][-1]['reference'])
    with pytest.raises(ValueError):lesson.configuration(value,task=lesson.TASKS[0],registry=REGISTRY)
