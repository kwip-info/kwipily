"""Ordinary phases retain exact launch authority without experiment membership."""
import fcntl
import os
import uuid
import pytest
from runner.tests.test_evaluation_authorization import encoded, digest, ident


@pytest.fixture
def operation_authority(tmp_path):
    project, operation, run, directory = [str(uuid.uuid4()).upper() for _ in range(4)]
    root = tmp_path.resolve()
    folder = root / 'workflow-operations' / project / operation
    folder.mkdir(parents=True, mode=0o700)
    phase = root / 'workflow-runs' / project / directory / 'preparation'
    phase.mkdir(parents=True, mode=0o700)
    for name in ('request.json', 'bindings.json'): (phase / name).write_text('{}')
    descriptors = [os.open(folder / name, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
                   for name in ('owner.lock', 'activity.lock')]
    fcntl.flock(descriptors[0], fcntl.LOCK_EX | fcntl.LOCK_NB)
    request = dict(schemaVersion=1, kind='workflowOperation', operationID=operation,
        projectID=project, runID=run, directoryID=directory, requestIdentity='a'*64,
        phase='preparation', arguments=['--execute-dataflow', str(phase/'request.json'),
            '--source-bindings', str(phase/'bindings.json'), '--output', str(phase/'artifacts'),
            '--retain-partial-dataflow'], executionBudget=dict(schemaVersion=1,
            requestedSeconds=10,maximumRunSeconds=10,source='application'), remainingSeconds=8.5)
    metadata = dict(schemaVersion=1,kind='workflowOperation',operationID=operation,projectID=project,
        requestIdentity=request['requestIdentity'],requestSHA256=digest(encoded(request)),
        directoryIdentity=ident(folder.stat()),ownerLockIdentity=ident(os.fstat(descriptors[0])),
        activityLockIdentity=ident(os.fstat(descriptors[1])))
    fixture = folder, request, metadata, descriptors
    publish(fixture)
    yield fixture
    for descriptor in descriptors:
        if descriptor is not None: os.close(descriptor)


def publish(f):
    f[2]['requestSHA256'] = digest(encoded(f[1]))
    (f[0]/'request.json').write_bytes(encoded(f[1]))
    (f[0]/'ownership.json').write_bytes(encoded(f[2]))


def claim(f):
    from runner.operation_authorization import claim_operation
    return claim_operation(str(f[0]),digest(encoded(f[1])),digest(encoded(f[2])))


def test_operation_retains_activity_and_exact_once_claim(operation_authority):
    from runner.operation_authorization import inherit_operation
    f=operation_authority
    with claim(f) as owner:
        child=inherit_operation(str(f[0]),digest(encoded(f[1])),digest(encoded(f[2])),owner.activity_fd)
        owner.close(); child.check()
        with pytest.raises(BlockingIOError): fcntl.flock(f[3][1],fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises((ValueError,OSError)): claim(f)
        child.close()
    fcntl.flock(f[3][1],fcntl.LOCK_EX|fcntl.LOCK_NB)


@pytest.mark.parametrize('mutation', ['remaining-omitted','budget-omitted','zero','negative','bool','overflow',
    'nan','budget-bool','forged-effective','unknown-budget','null-ceiling','unknown-request','wrong-output',
    'wrong-input','unsupported-option','duplicate-option','wrong-phase','missing-input','linked-input'])
def test_invalid_operation_rejected_before_claim(operation_authority,mutation):
    f=operation_authority; value=f[1]
    if mutation=='remaining-omitted': del value['remainingSeconds']
    elif mutation=='budget-omitted': del value['executionBudget']
    elif mutation in ('zero','negative','bool','overflow','nan'):
        value['remainingSeconds']={'zero':0,'negative':-1,'bool':True,'overflow':11,'nan':float('nan')}[mutation]
    elif mutation=='budget-bool': value['executionBudget']['requestedSeconds']=True
    elif mutation=='forged-effective': value['executionBudget']['maximumRunSeconds']=9
    elif mutation=='unknown-budget': value['executionBudget']['other']=1
    elif mutation=='null-ceiling': value['executionBudget']['capabilityMaximumSeconds']=None
    elif mutation=='unknown-request': value['other']=1
    elif mutation=='wrong-output': value['arguments'][5]='/tmp/escape'
    elif mutation=='wrong-input': value['arguments'][1]=value['arguments'][3]
    elif mutation=='unsupported-option': value['arguments']+=['--evaluate-frozen','file']
    elif mutation=='duplicate-option': value['arguments']+=['--output',value['arguments'][5]]
    elif mutation=='wrong-phase': value['phase']='evaluation'
    elif mutation=='missing-input': os.unlink(value['arguments'][1])
    else:
        from pathlib import Path
        path=Path(value['arguments'][1]); path.unlink(); path.symlink_to('bindings.json')
    publish(f)
    with pytest.raises((ValueError,OSError)): claim(f)
    assert not (f[0]/'claim.json').exists()


def test_legacy_no_budget_and_capability_ceiling(operation_authority):
    f=operation_authority
    del f[1]['executionBudget']; del f[1]['remainingSeconds']; publish(f)
    with claim(f) as authority: authority.check()


def test_training_phase_with_capped_project_policy(operation_authority):
    from pathlib import Path
    f=operation_authority
    phase=Path(f[1]['arguments'][1]).parent.with_name('training');phase.mkdir()
    (phase/'request.json').write_text('{}')
    f[1].update(phase='training',arguments=['--graph',str(phase/'request.json'),'--output',str(phase/'artifacts')],
        executionBudget=dict(schemaVersion=1,requestedSeconds=20,maximumRunSeconds=10,
            capabilityMaximumSeconds=10,source='project'))
    publish(f)
    with claim(f) as authority: authority.check()


def test_owner_release_and_phase_path_replacement_revoke(operation_authority):
    f=operation_authority
    with claim(f) as authority:
        os.close(f[3][0]);f[3][0]=None
        with pytest.raises(ValueError): authority.check()
