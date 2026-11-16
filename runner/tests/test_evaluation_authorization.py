"""Dedicated evaluation authority, independent of experiment phase slots."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import uuid

import pytest

from runner.evaluation_authorization import claim_evaluation, inherit_evaluation


def encoded(value): return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
def digest(raw): return hashlib.sha256(raw).hexdigest()
def ident(info): return dict(device=info.st_dev & 0xffffffff, inode=info.st_ino)


@pytest.fixture
def evaluation_authority(tmp_path):
    project = str(uuid.uuid4()).upper(); evaluation = str(uuid.uuid4()).upper()
    folder = tmp_path.resolve() / 'workflow-evaluations' / project / evaluation
    folder.mkdir(parents=True, mode=0o700)
    descriptors = []
    for name in ['owner.lock', 'activity.lock']:
        descriptors.append(os.open(folder / name, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600))
    fcntl.flock(descriptors[0], fcntl.LOCK_EX | fcntl.LOCK_NB)
    request = dict(schemaVersion=1, kind='frozenEvaluation', evaluationID=evaluation, projectID=project,
                   requestIdentity='a'*64, snapshotReceiptSHA256='b'*64,
                   settings=dict(device='cpu', batchSize=256, previewLimit=8))
    raw = encoded(request); (folder / 'request.json').write_bytes(raw)
    metadata = dict(schemaVersion=1, kind='frozenEvaluation', evaluationID=evaluation, projectID=project,
                    requestIdentity=request['requestIdentity'], requestSHA256=digest(raw), directoryIdentity=ident(folder.stat()),
                    ownerLockIdentity=ident(os.fstat(descriptors[0])), activityLockIdentity=ident(os.fstat(descriptors[1])))
    (folder / 'ownership.json').write_bytes(encoded(metadata))
    yield folder, request, metadata, descriptors
    for descriptor in descriptors:
        if descriptor is not None: os.close(descriptor)


def claim(f):
    folder, request, metadata, _ = f
    return claim_evaluation(str(folder), digest(encoded(request)), digest(encoded(metadata)))


def test_claim_inherited_lifetime_and_duplicate_are_bound_to_exact_request(evaluation_authority):
    f = evaluation_authority
    with claim(f) as parent:
        parent.check()
        inherited = inherit_evaluation(str(f[0]), digest(encoded(f[1])), digest(encoded(f[2])), parent.activity_fd)
        child_fd = inherited.activity_fd
        assert child_fd != parent.activity_fd
        assert inherited.request == f[1]
        with pytest.raises((ValueError, OSError)): claim(f)
        parent.close(); inherited.check(); os.fstat(child_fd)
        with pytest.raises(BlockingIOError): fcntl.flock(f[3][1], fcntl.LOCK_EX | fcntl.LOCK_NB)
        inherited.close(); inherited.close()
    fcntl.flock(f[3][1], fcntl.LOCK_EX | fcntl.LOCK_NB)
    saved = json.loads((f[0] / 'claim.json').read_bytes())
    assert saved == dict(schemaVersion=1, evaluationID=f[1]['evaluationID'], requestSHA256=digest(encoded(f[1])),
                         ownershipSHA256=digest(encoded(f[2])), state='claimed')
    with pytest.raises(ValueError): parent.check()


def test_unheld_owner_rejects_before_claim_creation(evaluation_authority):
    f = evaluation_authority
    os.close(f[3][0]); f[3][0] = None
    with pytest.raises((ValueError, OSError)): claim(f)
    assert not (f[0] / 'claim.json').exists()


@pytest.mark.parametrize('mutation', ['request','metadata','leaf','ancestor','owner','activity','permissions','partial-claim','extra-request'])
def test_tampering_and_partial_claim_fail_closed(evaluation_authority, mutation):
    f = evaluation_authority; folder = f[0]
    if mutation in ('request', 'metadata'):
        (folder / ('request.json' if mutation == 'request' else 'ownership.json')).write_bytes(b'{}')
    elif mutation in ('owner', 'activity'):
        name = mutation + '.lock'; (folder / name).unlink(); (folder / name).touch(mode=0o600)
    elif mutation == 'permissions': os.chmod(folder / 'activity.lock', 0o666)
    elif mutation == 'partial-claim': (folder / 'claim.json').write_bytes(b'{')
    elif mutation == 'extra-request':
        f[1]['output'] = '/tmp/escape'; (folder / 'request.json').write_bytes(encoded(f[1]))
        f[2]['requestSHA256'] = digest(encoded(f[1])); (folder / 'ownership.json').write_bytes(encoded(f[2]))
    elif mutation == 'leaf':
        target = folder / 'request.json'; target.rename(folder / 'original.json'); target.symlink_to('original.json')
    else:
        moved = folder.parent.with_name('moved'); folder.parent.rename(moved); folder.parent.symlink_to(moved, target_is_directory=True)
    with pytest.raises((ValueError, OSError)): claim(f)
    if mutation == 'partial-claim': assert (folder / 'claim.json').read_bytes() == b'{'


def test_held_authority_rechecks_bytes_owner_and_named_identity(evaluation_authority):
    f = evaluation_authority
    with claim(f) as authority:
        original = (f[0] / 'request.json').read_bytes()
        (f[0] / 'request.json').write_bytes(b'{}')
        with pytest.raises((ValueError, OSError)): authority.check()
        (f[0] / 'request.json').write_bytes(original)
        authority.check()
        os.close(f[3][0]); f[3][0] = None
        with pytest.raises((ValueError, OSError)): authority.check()


def test_staging_exclusive_activity_blocks_claim_without_consuming_it(evaluation_authority):
    f = evaluation_authority
    fcntl.flock(f[3][1], fcntl.LOCK_EX | fcntl.LOCK_NB)
    with pytest.raises((ValueError, OSError)):
        claim(f)
    assert not (f[0] / 'claim.json').exists()
    fcntl.flock(f[3][1], fcntl.LOCK_UN)
    with claim(f) as authority:
        authority.check()


@pytest.mark.parametrize('role,version,accepted',[('validation',2,True),('test',2,False),('validation',1,False),('training',2,False)])
def test_explicit_validation_authority_settings_are_versioned_and_role_bound(evaluation_authority,role,version,accepted):
    folder,request,metadata,_=evaluation_authority
    request['settings'].update(settingsVersion=version,partitionRole=role)
    raw=encoded(request);(folder/'request.json').write_bytes(raw)
    metadata['requestSHA256']=digest(raw);(folder/'ownership.json').write_bytes(encoded(metadata))
    if accepted:
        with claim(evaluation_authority) as authority:
            authority.check();assert authority.request['settings']['partitionRole']=='validation'
    else:
        with pytest.raises(ValueError):claim(evaluation_authority)
        assert not (folder/'claim.json').exists()
