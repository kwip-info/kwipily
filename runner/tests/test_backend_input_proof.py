"""Retained backend inputs remain checked without O(files) hot-path opens."""
import errno
import os
import shutil
import sys
from pathlib import Path

import pytest
from runner.backend_fit_operation_contract import verify_inputs
from runner.tests.test_backend_fit_operation_contract import fixture

MAC = sys.platform == 'darwin'


@pytest.mark.skipif(not MAC, reason='Darwin kernel input proof')
def test_warm_backend_checks_do_not_reopen_inventory_or_reparse_manifest(tmp_path, monkeypatch):
    import runner.backend_fit_operation_contract as contract
    root, value = fixture(tmp_path)
    cache = {}
    verify_inputs(value, root, cache)
    opens = []
    original = os.open
    def counted(*args, **kwargs):
        opens.append(args[0]); return original(*args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(os, 'open', counted)
        patch.setattr(contract, 'parse_contract', lambda *_: pytest.fail('Repeated immutable request parsing'))
        for _ in range(20): verify_inputs(value, root, cache)
    # The root namespace remains live; inputs use retained, kernel-monitored FDs.
    assert len(opens) <= 20
    # Normal output publication and reading inputs are not input mutations.
    (root/'artifacts').mkdir()
    (root/'artifacts/result.json').write_text('{}')
    (root/'input/preparation/dataflow.json').read_bytes()
    verify_inputs(value, root, cache)


@pytest.mark.parametrize('change', ['content', 'replace', 'symlink', 'permissions', 'parent'])
def test_cached_backend_inputs_refuse_tamper(tmp_path, change):
    root, value = fixture(tmp_path); cache = {}
    verify_inputs(value, root, cache)
    target = root/'input/preparation/dataflow.json'
    if change == 'content':
        before = target.stat(); target.write_bytes(b'[]')
        os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
    elif change == 'replace':
        other = target.with_name('replacement'); other.write_bytes(target.read_bytes()); other.replace(target)
    elif change == 'symlink':
        other = tmp_path/'same.json'; other.write_bytes(target.read_bytes()); target.unlink(); target.symlink_to(other)
    elif change == 'permissions': target.chmod(0o666)
    else:
        old = root/'input/old'; target.parent.rename(old); shutil.copytree(old, target.parent)
    with pytest.raises((ValueError, OSError)): verify_inputs(value, root, cache)


@pytest.mark.skipif(not MAC, reason='Darwin sticky vnode events')
@pytest.mark.parametrize('change', ['write-and-restore', 'rename-and-restore', 'link-and-unlink'])
def test_transient_input_mutation_stays_invalid(tmp_path, change):
    root, value = fixture(tmp_path); cache = {}
    verify_inputs(value, root, cache)
    target = root/'input/preparation/dataflow.json'; before = target.stat()
    if change == 'write-and-restore':
        raw = target.read_bytes(); target.write_bytes(b'[]'); target.write_bytes(raw)
        os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
    elif change == 'rename-and-restore':
        other = target.with_name('moved'); target.rename(other); other.rename(target)
    else:
        other = target.with_name('linked'); os.link(target, other); other.unlink()
    for _ in range(2):
        with pytest.raises((ValueError, OSError)): verify_inputs(value, root, cache)


@pytest.mark.parametrize('change', ['boolean-size', 'float-size', 'changed-pin', 'escaped-output'])
def test_cached_proof_does_not_accept_changed_request(tmp_path, change):
    root, value = fixture(tmp_path); cache = {}; verify_inputs(value, root, cache)
    if change == 'boolean-size': value['inputFiles'][0]['bytes'] = True
    elif change == 'float-size': value['inputFiles'][0]['bytes'] = float(value['inputFiles'][0]['bytes'])
    elif change == 'changed-pin': value['inputFiles'][0]['sha256'] = '0'*64
    else: value['arguments'][-1] = str(tmp_path/'elsewhere')
    with pytest.raises(ValueError): verify_inputs(value, root, cache)


def test_replaced_root_is_not_readmitted_after_restore(tmp_path):
    root, value = fixture(tmp_path); cache = {}; verify_inputs(value, root, cache)
    old = root.with_name('old-operation'); root.rename(old); shutil.copytree(old, root)
    with pytest.raises(ValueError): verify_inputs(value, root, cache)
    shutil.rmtree(root); old.rename(root)
    with pytest.raises(ValueError): verify_inputs(value, root, cache)


@pytest.mark.skipif(not MAC, reason='Darwin queue lifecycle')
def test_closed_inputs_release_descriptors_and_cannot_be_reused(tmp_path):
    from runner.backend_fit_operation_contract import close_inputs
    before = len(os.listdir('/dev/fd'))
    root, value = fixture(tmp_path); cache = {}; verify_inputs(value, root, cache)
    close_inputs(cache); close_inputs(cache)
    assert len(os.listdir('/dev/fd')) <= before + 1
    with pytest.raises(ValueError): verify_inputs(value, root, cache)


@pytest.mark.skipif(not MAC, reason='Darwin queue failure')
def test_event_poll_failure_is_sticky_and_never_falls_back(tmp_path):
    root, value = fixture(tmp_path); cache = {}; verify_inputs(value, root, cache)
    proof = cache['_backend_input_proof']; original = proof.queue
    class FailedQueue:
        def control(self, *_): raise OSError(errno.EBADF, 'synthetic poll failure')
        def close(self): original.close()
    proof.queue = FailedQueue()
    for _ in range(2):
        with pytest.raises(ValueError): verify_inputs(value, root, cache)


@pytest.mark.skipif(not MAC, reason='Darwin queue resource fallback')
def test_unavailable_kernel_queue_preserves_full_verification(tmp_path, monkeypatch):
    import select
    def unavailable(): raise OSError(errno.EMFILE, 'synthetic descriptor ceiling')
    monkeypatch.setattr(select, 'kqueue', unavailable)
    root, value = fixture(tmp_path); cache = {}; verify_inputs(value, root, cache)
    verify_inputs(value, root, cache)
    (root/'input/preparation/dataflow.json').write_text('[]')
    with pytest.raises(ValueError): verify_inputs(value, root, cache)


@pytest.mark.skipif(not MAC, reason='Darwin bounded descriptor admission')
def test_small_descriptor_allowance_keeps_readers_headroom(tmp_path, monkeypatch):
    import resource
    monkeypatch.setattr(resource,'getrlimit',lambda _: (70,70))
    root,value=fixture(tmp_path);cache={};verify_inputs(value,root,cache)
    assert not cache['_backend_input_proof'].kernel
    (root/'input/preparation/dataflow.json').write_bytes(b'[]')
    with pytest.raises(ValueError):verify_inputs(value,root,cache)
