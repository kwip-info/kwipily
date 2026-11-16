"""Learning tools reuse retained operation lifetimes with exact captured inputs."""
import json
import os
import sys
import pytest
from runner.tests.test_operation_authorization import operation_authority,publish,claim
from runner.tests.test_evaluation_authorization import encoded,digest
from runner.tests.test_operation_bootstrap import control
from runner.tests.test_worker_bootstrap import entry,run_supervisor
from runner.tests.test_lifetime_watchdog import libraries


@pytest.fixture(params=['learningRuntime','learningAssessment'])
def learning_operation(operation_authority,request):
    f=operation_authority;value=f[1]
    for key in ('runID','directoryID','phase'):del value[key]
    value.update(schemaVersion=2,toolMode=request.param,inputFiles=[])
    if request.param=='learningRuntime':value['arguments']=['--learning-runtime']
    else:
        inputs=f[0]/'input';inputs.mkdir(mode=0o700)
        for name in ('assessment.json','submission.json'):
            raw=encoded({'kind':name});(inputs/name).write_bytes(raw)
            value['inputFiles'].append(dict(path='input/'+name,bytes=len(raw),sha256=digest(raw)))
        value['arguments']=['--learning-evidence-request',str(inputs/'assessment.json'),'--learning-submission',str(inputs/'submission.json'),'--learning-submission-sha256',value['inputFiles'][1]['sha256']]
    publish(f);return f


def test_tool_claim_matches_exact_captured_request(learning_operation):
    with claim(learning_operation) as authority:authority.check()


@pytest.mark.parametrize('change',['unknown-mode','unexpected-field','missing-budget','duplicate-option','registry','escape','wrong-inventory','bad-hash','bool-bytes'])
def test_tool_contract_failures_never_claim(learning_operation,change):
    f=learning_operation;value=f[1]
    if change=='unknown-mode':value['toolMode']='shell'
    elif change=='unexpected-field':value['runID']='unexpected'
    elif change=='missing-budget':del value['remainingSeconds']
    elif change=='duplicate-option':value['arguments']+=value['arguments'][:1]
    elif change=='registry':value['arguments']+=['--registry','/tmp/unverified.json']
    elif change=='escape':value['arguments']=['--learning-evidence-request','/tmp/request']
    elif change=='wrong-inventory':value['inputFiles']=[dict(path='../request.json',bytes=1,sha256='0'*64)]
    elif change=='bad-hash':
        if value['inputFiles']:value['inputFiles'][0]['sha256']='0'*64
        else:value['inputFiles']=[dict(path='input/submission.json',bytes=1,sha256='0'*64)]
    else:value['inputFiles']=[dict(path='input/assessment.json',bytes=True,sha256='0'*64)]
    publish(f)
    with pytest.raises((ValueError,OSError)):claim(f)
    assert not (f[0]/'claim.json').exists()


@pytest.mark.parametrize('change',['bytes','symlink','hardlink','directory-link','submission-pin'])
def test_assessment_inputs_reject_substitution(learning_operation,change):
    f=learning_operation
    if f[1]['toolMode']!='learningAssessment':pytest.skip('Assessment inputs only')
    target=f[0]/'input/submission.json'
    if change=='bytes':target.write_bytes(b'{}')
    elif change=='symlink':target.unlink();target.symlink_to('assessment.json')
    elif change=='hardlink':os.link(target,target.with_name('alias.json'))
    elif change=='directory-link':
        target.parent.rename(f[0]/'moved');(f[0]/'input').symlink_to('moved',target_is_directory=True)
    else:f[1]['arguments'][-1]='0'*64;publish(f)
    with pytest.raises((ValueError,OSError)):claim(f)
    assert not (f[0]/'claim.json').exists()


def test_assessment_input_rechecked_during_held_authority(learning_operation):
    f=learning_operation
    if f[1]['toolMode']!='learningAssessment':pytest.skip('Assessment inputs only')
    with claim(f) as authority:
        (f[0]/'input/assessment.json').write_bytes(b'{}')
        with pytest.raises((ValueError,OSError)):authority.check()


@pytest.mark.skipif(sys.platform!='darwin',reason='Native macOS lifetime guard')
def test_actual_tool_reexec_ready_then_engine(learning_operation,tmp_path,libraries):
    f=learning_operation;guard,_=libraries
    registry=tmp_path/'verified-registry.json';registry.write_text('{}')
    code,events,error=run_supervisor(entry(tmp_path,guard,registry),control(f))
    assert code==0,(events,error)
    assert events[0]==dict(event='operationReady',operationID=f[1]['operationID'],toolMode=f[1]['toolMode'],
        requestSHA256=digest(encoded(f[1])),ownershipSHA256=digest(encoded(f[2])))
    assert events[1]['arguments']==f[1]['arguments']+['--registry',str(registry)]


def test_learning_cooperative_authority_is_checked_through_evidence(tmp_path):
    from runner.learning_evidence import execute_request
    from runner.tests.test_learning_evidence import affine_request,affine_submission
    request=tmp_path/'assessment.json';submission=tmp_path/'submission.json'
    request.write_bytes(encoded(affine_request()));raw=encoded(affine_submission());submission.write_bytes(raw)
    calls=[]
    result=execute_request(request,submission,digest(raw),check_cancelled=lambda:calls.append(True))
    assert result['fixtures'] and len(calls)>3
    def revoked():raise ValueError('authority revoked')
    with pytest.raises(ValueError,match='authority revoked'):
        execute_request(request,submission,digest(raw),check_cancelled=revoked)


def test_tool_wall_allowance_cannot_exceed_capability(learning_operation):
    f=learning_operation;value=f[1]
    maximum=10 if value['toolMode']=='learningRuntime' else 65
    value.update(executionBudget=dict(schemaVersion=1,requestedSeconds=86400,maximumRunSeconds=86400,source='application'),remainingSeconds=maximum+.001)
    publish(f)
    with pytest.raises(ValueError):claim(f)
    assert not (f[0]/'claim.json').exists()
