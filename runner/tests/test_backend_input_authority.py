"""The fast input proof never replaces retained owner/control authority."""
import fcntl
import json
import os
import shutil
import sys

import pytest
from runner.tests.test_operation_authorization import operation_authority, publish, claim
from runner.tests.test_backend_fit_operation_contract import fixture, digest


def install(f, tmp_path):
    source, value = fixture(tmp_path/'backend-source')
    root = f[0]; shutil.copytree(source/'input', root/'input')
    request = (root/'input/backend-request.json').read_text().replace(str(source), str(root))
    (root/'input/backend-request.json').write_text(request)
    value['arguments'] = [item.replace(str(source), str(root)) for item in value['arguments']]
    value['arguments'][3] = digest(request.encode())
    for entry in value['inputFiles']:
        raw = (root/entry['path']).read_bytes(); entry.update(bytes=len(raw), sha256=digest(raw))
    for key in ('runID', 'directoryID', 'phase'): del f[1][key]
    f[1].update(schemaVersion=2, toolMode='backendFit', **value)
    publish(f)


@pytest.mark.skipif(sys.platform!='darwin', reason='Darwin fast input authority')
def test_warm_backend_retains_owner_and_control_checks(operation_authority, tmp_path, monkeypatch):
    import runner.operation_authorization as operations
    import runner.backend_fit_operation_contract as backend
    f = operation_authority; install(f, tmp_path)
    with claim(f) as authority:
        authority.check()
        monkeypatch.setattr(operations, 'validate_request', lambda *_: pytest.fail('Repeated static request validation'))
        monkeypatch.setattr(backend, 'parse_contract', lambda *_: pytest.fail('Repeated static backend parsing'))
        for _ in range(20): authority.check()
        fcntl.flock(f[3][0], fcntl.LOCK_UN)
        with pytest.raises(ValueError): authority.check()


@pytest.mark.parametrize('change', ['owner', 'request-bytes', 'request-typed', 'claim', 'input'])
def test_backend_hot_authority_refuses_revocation_and_tamper(operation_authority, tmp_path, change):
    f = operation_authority; install(f, tmp_path)
    with claim(f) as authority:
        authority.check()
        if change == 'owner': fcntl.flock(f[3][0], fcntl.LOCK_UN)
        elif change == 'request-bytes':
            f[1]['arguments'][-1] = str(tmp_path/'other'); publish(f)
        elif change == 'request-typed':
            authority.request['inputFiles'][0]['bytes'] = float(authority.request['inputFiles'][0]['bytes'])
        elif change == 'claim': (f[0]/'claim.json').write_text('{}')
        else:
            path=f[0]/'input/preparation/dataflow.json';before=path.stat();path.write_bytes(b'[]')
            os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
        with pytest.raises(ValueError): authority.check()
