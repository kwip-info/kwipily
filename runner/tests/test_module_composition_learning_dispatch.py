"""Actual source-qualified composition lessons through shared evidence and cold CLI."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import pytest
from runner import learning_evidence as shared
from runner import module_composition_learning as lesson
from runner.module_composition_reference import FIXTURES

ROOT=Path(__file__).resolve().parents[2]
REGISTRY=ROOT/'Sources/KwipMLCore/Resources/components.json'
VERSION=dict(major=1,minor=0,patch=0)
CAPABILITIES={'moduleComposition':VERSION}

def submission(task):
    return dict(schemaVersion=1,kind='compositionLearningGraph',taskID=task,source=lesson.source_template(task))

def request(task):
    fixtures=[]
    for f in FIXTURES:
        p=f.parameters
        fixtures.append(dict(id=f.id,inputs=dict(coefficients=dict(shape=[4],values=[p.a,p.b,p.c,p.d]),rows=dict(shape=[3,1],values=list(f.rows)),upstream=dict(shape=[3,1],values=list(f.upstream)),targets=dict(shape=[3,1],values=[1.,-1.,.5]))))
    return dict(assessmentID='assessment.'+task,assessmentVersion=VERSION.copy(),fixtureDigest=hashlib.sha256(json.dumps(fixtures,sort_keys=True,separators=(',',':')).encode()).hexdigest(),runtime=dict(backend='pytorch',version=shared.runtime_versions(REGISTRY)['pytorch'],capabilities=copy.deepcopy(CAPABILITIES)),budget=dict(milliseconds=60000,maximumTensorElements=4096),fixtures=fixtures)

def measured(task,s=None,r=None,**kwargs):
    return shared.generate_evidence(r or request(task),s or submission(task),submission_digest='b'*64,registry=REGISTRY,**kwargs)

@pytest.mark.parametrize('task',lesson.TASKS)
def test_exact_fixed_policy_actual_five_tensor_criteria_and_private_safe_transport(task):
    r=request(task)
    assert shared.assessment_policy(r)==dict(backend='pytorch',capabilities=CAPABILITIES,submissionSchema=1,moduleComposition=task)
    result=measured(task)
    assert not result.get('backendError'),result
    assert result['runtime']==r['runtime'] and len(result['fixtures'])==4
    assert result['fixtureDigest']==r['fixtureDigest'] and result['submissionDigest']=='b'*64
    assert result['maximumTensorElements']<=4096
    for entry in result['fixtures']:
        assert set(entry)=={'fixtureID','tensors','partitions','composition'}
        assert set(entry['tensors'])==set(lesson.SHAPES[task])
        for name,table in entry['tensors'].items():
            assert table['shape']==lesson.SHAPES[task][name]
            assert table['values']==pytest.approx([0]*len(table['values']),abs=2e-5)
    assert all(s not in json.dumps(result) for s in ['expected','sourceDigest','coefficients','parameterValues','gradientValues'])

@pytest.mark.parametrize('task',lesson.TASKS)
def test_valid_wrong_container_kind_is_measured_failure(task):
    s=submission(task);block=s['source']['definitions'][0]
    if task==lesson.TASKS[0]:
        block['composition']['kind']='namedMap'
        for i,c in enumerate(block['composition']['children']):c['key']='child'+str(i)
    else:block['composition']['kind']='orderedList'
    result=measured(task,s)
    assert not result.get('backendError') and len(result['fixtures'])==4
    assert any(abs(v)>2e-5 for f in result['fixtures'] for t in f['tensors'].values() for v in t['values'])

@pytest.mark.parametrize('change',[
 lambda r:r['assessmentVersion'].update(minor=1),lambda r:r.update(assessmentID='assessment.composition-foreign'),
 lambda r:r.update(answers={}),lambda r:r['fixtures'][0].update(expected={'answer':[123]}),
 lambda r:r['fixtures'][0]['inputs']['rows'].update(shape=[1,3]),lambda r:r['fixtures'].pop(),
 lambda r:r['fixtures'][0]['inputs'].update(answer=dict(shape=[1],values=[999])),
 lambda r:r['budget'].update(maximumTensorElements=True),lambda r:r['fixtures'][0]['inputs'].update(rows=None),lambda r:r['budget'].update(maximumTensorElements=4097)])
def test_strict_task_fixture_geometry_budget_and_no_private_answer_input(change):
    r=request(lesson.TASKS[0]);change(r)
    with pytest.raises(ValueError):measured(lesson.TASKS[0],r=r)

@pytest.mark.parametrize('change',[
 lambda s:s.update(taskID=lesson.TASKS[1]),lambda s:s['source'].update(schemaVersion=2),
 lambda s:s.update(expected={'answer':[123]}),lambda s:s['source']['definitions'][0].update(composition=None)])
def test_source_identity_downgrade_malformed_or_answer_injection_has_no_partial_evidence(change):
    s=submission(lesson.TASKS[0]);change(s)
    result=measured(lesson.TASKS[0],s)
    assert result.get('backendError') and result['fixtures']==[]

def test_runtime_mismatch_cancellation_and_tiny_budget_never_execute_source(monkeypatch):
    def forbidden(*a,**k):raise AssertionError('No native work permitted')
    monkeypatch.setattr(lesson,'measure',forbidden)
    task=lesson.TASKS[0];r=request(task);r['runtime']['capabilities']={}
    mismatch=measured(task,r=r);assert not mismatch['fixtures'] and not mismatch['cancelled']
    cancelled=measured(task,cancel=lambda:True);assert not cancelled['fixtures'] and cancelled['cancelled']
    r=request(task);r['budget']['maximumTensorElements']=1
    with pytest.raises(ValueError):measured(task,r=r)

@pytest.mark.parametrize('task',lesson.TASKS)
def test_cold_actual_standalone_worker_request_with_captured_registry_and_digest(tmp_path,task):
    r=request(task);s=submission(task)
    plan=tmp_path/'request.json';source=tmp_path/'submission.json';registry=tmp_path/'captured-registry.json'
    registry.write_bytes(REGISTRY.read_bytes());plan.write_text(json.dumps(r,allow_nan=False))
    raw=json.dumps(s,allow_nan=False).encode();source.write_bytes(raw)
    command=[sys.executable,str(ROOT/'runner/main.py'),'--registry',str(registry),'--learning-evidence-request',str(plan),'--learning-submission',str(source),'--learning-submission-sha256',hashlib.sha256(raw).hexdigest()]
    run=subprocess.run(command,cwd=tmp_path,capture_output=True,text=True,timeout=35)
    assert run.returncode==0,run.stdout+run.stderr
    result=next(json.loads(line)['evidence'] for line in run.stdout.splitlines() if line.startswith('{') and json.loads(line).get('event')=='learningEvidence')
    assert not result.get('backendError'),result
    assert len(result['fixtures'])==4 and result['runtime']==r['runtime']
    assert result['submissionDigest']==hashlib.sha256(raw).hexdigest()
    assert all(abs(v)<=2e-5 for f in result['fixtures'] for t in f['tensors'].values() for v in t['values'])
    bad=command.copy();bad[-1]='0'*64
    rejected=subprocess.run(bad,cwd=tmp_path,capture_output=True,text=True,timeout=15)
    assert rejected.returncode!=0 and 'digest' in (rejected.stdout+rejected.stderr).lower()


def test_explicit_captured_registry_works_when_repository_registry_is_missing(tmp_path,monkeypatch):
    from runner import engine
    captured=tmp_path/'registry.json';captured.write_bytes(REGISTRY.read_bytes())
    monkeypatch.setattr(engine,'REGISTRY',tmp_path/'missing-repository-registry.json')
    task=lesson.TASKS[0]
    result=shared.generate_evidence(request(task),submission(task),submission_digest='b'*64,registry=captured)
    assert not result.get('backendError') and len(result['fixtures'])==4,result


@pytest.mark.parametrize('task',lesson.TASKS)
def test_actual_fixture_carries_exact_safe_public_native_metadata(task):
    result=measured(task)
    assert not result.get('backendError'),result
    for fixture in result['fixtures']:
        metadata=fixture.get('composition')
        assert metadata is not None, 'Every actual fixture must carry its independently validated native record.'
        assert set(metadata)=={'schemaVersion','nativeOwners','nativeContainers','invocations','savedStateKeys','independentStorage'}
        assert metadata['schemaVersion']==1 and type(metadata['schemaVersion']) is int
        assert type(metadata['independentStorage']) is bool and metadata['independentStorage']
        assert metadata['nativeOwners'] and metadata['nativeContainers'] and metadata['invocations']
        for owner in metadata['nativeOwners']:
            assert set(owner)=={'node','parameter','name','shape','dtype','requiresGrad'}
            assert owner['dtype']=='float32' and type(owner['requiresGrad']) is bool
        for container in metadata['nativeContainers']:
            assert set(container)=={'path','kind','keys','matches'}
            assert type(container['matches']) is bool
        assert all(len(pair)==2 and pair[0] in ('started','completed') for pair in metadata['invocations'])
        assert metadata['savedStateKeys']==[o['name'] for o in metadata['nativeOwners']]
        assert len(json.dumps(metadata,ensure_ascii=False).encode())<=131072
        assert all(key not in json.dumps(metadata) for key in ['sourceDigest','coefficients','upstream','parameterValues','gradientValues','expected'])


def test_public_structural_record_strips_checker_values_and_refuses_malformed_or_oversized_fields():
    task=lesson.TASKS[0]
    raw=lesson.measure(submission(task),request(task)['fixtures'][0]['inputs'],task=task,registry=REGISTRY)['metadata']
    raw['expected']=[999];raw['nativeOwners'][0]['parameterValues']=[123]
    raw['nativeContainers'][0]['privateRows']=[456]
    before=copy.deepcopy(raw)
    public=lesson._public_metadata(raw)
    assert raw==before
    assert all(key not in json.dumps(public) for key in ['sourceDigest','expected','parameterValues','privateRows'])
    for mutation in ('bool-shape','flag','array-state','phase','byte-bound'):
        bad=copy.deepcopy(raw)
        if mutation=='bool-shape':bad['nativeOwners'][0]['shape']=[True]
        elif mutation=='flag':bad['independentStorage']=1
        elif mutation=='array-state':bad['savedStateKeys'][0]=[1,2,3]
        elif mutation=='phase':bad['invocations'][0]=['backward','input']
        else:bad['nativeContainers'][0]['path']='x'*4097
        with pytest.raises(ValueError):lesson._public_metadata(bad)
