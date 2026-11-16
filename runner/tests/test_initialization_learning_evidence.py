"""Strict shared-dispatch evidence for real initialization lessons."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import pytest
from runner import learning_evidence as shared
from runner.tests.test_learning_initialization import TASKS, fixture, submission

ROOT=Path(__file__).resolve().parents[2]
VERSION={'major':1,'minor':0,'patch':0}
CAPABILITIES={'initialization':VERSION,'optimizer_sgd':VERSION}

def request(task):
    fixtures=[fixture(i,variation=.125) for i in range(4)]
    digest=hashlib.sha256(json.dumps(fixtures,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return dict(assessmentID='assessment.'+task,assessmentVersion=VERSION.copy(),fixtureDigest=digest,
                runtime=dict(backend='pytorch',version=shared.runtime_versions()['pytorch'],capabilities=copy.deepcopy(CAPABILITIES)),
                budget=dict(milliseconds=60000,maximumTensorElements=4096),fixtures=fixtures)

def measured(task,s=None,r=None,**kwargs):
    return shared.generate_evidence(r or request(task),s or submission(task),submission_digest='b'*64,**kwargs)

@pytest.mark.parametrize('task',TASKS)
def test_exact_fixed_policy_and_real_five_criteria(task):
    r=request(task)
    assert shared.assessment_policy(r)==dict(backend='pytorch',capabilities=CAPABILITIES,submissionSchema=1,initialization=task)
    result=measured(task)
    assert not result.get('backendError'),result
    assert result['runtime']==r['runtime'] and result['submissionDigest']=='b'*64
    assert result['fixtureDigest']==r['fixtureDigest'] and len(result['fixtures'])==4
    assert result['maximumTensorElements']<=4096 and not result['cancelled']
    for entry in result['fixtures']:
        assert set(entry['tensors'])=={'startingSettings','predictions','firstAdjustment','repeatIsolation','savedReuse'}
        for name,t in entry['tensors'].items():
            assert t['values']==pytest.approx([0]*len(t['values']),abs=0 if name in ('startingSettings','repeatIsolation') else 2e-6)

@pytest.mark.parametrize('task',TASKS)
def test_valid_wrong_seed_is_measured_not_rejected_or_answer_copied(task):
    s=submission(task);s['graph']['nodes'][0]['config']['seed']=43
    result=measured(task,s)
    assert not result.get('backendError') and len(result['fixtures'])==4
    assert all(any(abs(v)>1e-5 for v in e['tensors']['startingSettings']['values']) for e in result['fixtures'])

@pytest.mark.parametrize('change',[
    lambda r:r['assessmentVersion'].update(minor=1),lambda r:r.update(assessmentID='assessment.initialization-foreign'),
    lambda r:r.update(answers={}),lambda r:r['budget'].update(maximumTensorElements=True)])
def test_strict_assessment_identity_and_request_boundary(change):
    r=request(TASKS[0]);change(r)
    with pytest.raises(ValueError):measured(TASKS[0],r=r)


def test_runtime_mismatch_cancellation_and_schema_are_distinct():
    task=TASKS[0];r=request(task);r['runtime']['capabilities'].pop('optimizer_sgd')
    mismatch=measured(task,r=r);assert not mismatch['fixtures'] and not mismatch['cancelled']
    cancelled=measured(task,cancel=lambda:True);assert not cancelled['fixtures'] and cancelled['cancelled']
    s=submission(task);s['answers']={}
    invalid=measured(task,s);assert not invalid['fixtures'] and invalid.get('backendError')
    r=request(task);r['fixtures'][0]['inputs']['x']['shape']=[7,2]
    with pytest.raises(ValueError):measured(task,r=r)

@pytest.mark.parametrize('task',TASKS)
def test_cold_standalone_actual_request_json_and_submission_digest(tmp_path,task):
    r=request(task);s=submission(task)
    plan=tmp_path/'request.json';graph=tmp_path/'submission.json'
    plan.write_text(json.dumps(r,allow_nan=False));raw=json.dumps(s,allow_nan=False).encode();graph.write_bytes(raw)
    command=[sys.executable,str(ROOT/'runner/main.py'),'--learning-evidence-request',str(plan),'--learning-submission',str(graph),'--learning-submission-sha256',hashlib.sha256(raw).hexdigest()]
    done=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert done.returncode==0,done.stderr+done.stdout
    events=[json.loads(line) for line in done.stdout.splitlines() if line.startswith('{')]
    result=next(e['evidence'] for e in events if e.get('event')=='learningEvidence')
    assert not result.get('backendError'),result
    direct=measured(task)
    assert result['fixtures']==direct['fixtures'] and result['runtime']==r['runtime']
    assert result['submissionDigest']==hashlib.sha256(raw).hexdigest()
    bad=command.copy();bad[-1]='0'*64
    rejected=subprocess.run(bad,cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert rejected.returncode!=0 and 'digest' in rejected.stdout.lower()+rejected.stderr.lower()
