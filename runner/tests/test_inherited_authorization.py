"""Child validation retains the inherited shared description without reclaiming."""
import fcntl
import json
import os
import selectors
import subprocess
import sys

import pytest
from runner.tests.test_worker_authorization import ROOT, claim, digest, encoded, launch_fixture
from runner.tests.test_worker_supervisor import descriptor_snapshot


def inherit(f, descriptor):
    from runner.worker_authorization import inherit_launch
    folder,intent,_,_=f
    return inherit_launch(str(folder),intent['nonce'],digest(encoded(intent)),
                          digest((folder/'ownership.json').read_bytes()),descriptor)


def test_inherited_close_preserves_callers_and_claim_bytes(launch_fixture):
    folder,intent,_,_=launch_fixture
    with claim(launch_fixture) as parent:
        path=folder/f"claim-{intent['nonce']}.json";before=path.read_bytes()
        descriptors=descriptor_snapshot()
        with inherit(launch_fixture,parent.activity_fd) as child:
            assert child.activity_fd!=parent.activity_fd
            assert not os.get_inheritable(child.activity_fd)
            child.check()
            assert child.intent==intent
        os.fstat(parent.activity_fd)
        parent.check()
        assert path.read_bytes()==before
        assert descriptor_snapshot()==descriptors


@pytest.mark.parametrize('mutation',['wrongInode','standardFD','closedFD','missingClaim','revokedClaim',
                                     'changedIntent','changedMetadata'])
def test_invalid_child_authority_never_rewrites_claim(launch_fixture,mutation):
    folder,intent,metadata,_=launch_fixture
    with claim(launch_fixture) as parent:
        descriptor=parent.activity_fd;extra=None
        path=folder/f"claim-{intent['nonce']}.json"
        if mutation=='wrongInode':extra=os.open(folder/'owner.lock',os.O_RDONLY);descriptor=extra
        elif mutation=='standardFD':descriptor=0
        elif mutation=='closedFD':
            descriptor=os.dup(parent.activity_fd);os.close(descriptor)
        elif mutation=='missingClaim':path.unlink()
        elif mutation=='revokedClaim':
            value=json.loads(path.read_bytes());value['state']='revoked';path.write_bytes(encoded(value))
        elif mutation=='changedIntent':
            (folder/f"launch-{intent['nonce']}.json").write_bytes(encoded({**intent,'ordinal':2}))
        else:
            (folder/'ownership.json').write_bytes(encoded({**metadata,'createdAtUnixSeconds':2}))
        before=path.read_bytes() if path.exists() else None
        descriptors=descriptor_snapshot()
        try:
            with pytest.raises((ValueError,OSError)):inherit(launch_fixture,descriptor)
            assert (path.read_bytes() if path.exists() else None)==before
            assert descriptor_snapshot()==descriptors
        finally:
            if extra is not None:os.close(extra)


def test_real_child_keeps_activity_lease_after_parent_authority_closes(launch_fixture):
    folder,intent,_,_=launch_fixture
    parent=claim(launch_fixture)
    script='''
import sys
sys.path.insert(0,sys.argv[1])
from worker_authorization import inherit_launch
with inherit_launch(*sys.argv[2:6],int(sys.argv[6])) as authority:
 authority.check();print('READY',flush=True)
 sys.stdin.buffer.read(1)
 authority.check()
'''
    process=None
    try:
        process=subprocess.Popen([sys.executable,'-I','-c',script,str(ROOT/'runner'),str(folder),intent['nonce'],
                                  digest(encoded(intent)),digest((folder/'ownership.json').read_bytes()),str(parent.activity_fd)],
                                 pass_fds=(parent.activity_fd,),stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,bufsize=0)
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout,selectors.EVENT_READ)
            assert selector.select(timeout=8),'Child authorization handshake missing'
            assert process.stdout.readline()==b'READY\n'
        parent.close()
        contender=os.open(folder/'activity.lock',os.O_RDONLY)
        try:
            with pytest.raises(BlockingIOError):fcntl.flock(contender,fcntl.LOCK_EX|fcntl.LOCK_NB)
            process.stdin.write(b'x');process.stdin.flush()
            assert process.wait(timeout=8)==0
            fcntl.flock(contender,fcntl.LOCK_EX|fcntl.LOCK_NB)
        finally:os.close(contender)
    finally:
        parent.close()
        if process is not None:
            if process.poll() is None:process.kill()
            process.communicate(timeout=8)
