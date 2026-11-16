"""Preparation can reuse proven control bytes, never live permission/identity."""
import os
from pathlib import Path
import pytest
from runner.tests.test_operation_authorization import operation_authority,claim
from runner.tests.test_preparation_operation_authorization import preparation_operation


def test_control_bytes_are_not_reread_after_admission(preparation_operation,monkeypatch):
    with claim(preparation_operation) as authority:
        def forbidden(*args,**kwargs):raise AssertionError('unchanged control bytes reread')
        # Inputs also reuse their verified signatures; new reads here would be
        # repeated control parsing, not a changed file needing admission.
        monkeypatch.setattr(os,'read',forbidden)
        authority.check()


@pytest.mark.parametrize('name',['request.json','ownership.json','claim.json'])
@pytest.mark.parametrize('mutation',['rewrite','replace','symlink','permissions','hardlink'])
def test_control_proof_detects_mutation_with_restored_mtime(preparation_operation,name,mutation):
    with claim(preparation_operation) as authority:
        path=preparation_operation[0]/name;before=path.stat();raw=path.read_bytes()
        if mutation=='rewrite':path.write_bytes(b'x'+raw[1:]);os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
        elif mutation=='replace':
            replacement=path.with_suffix('.replacement');replacement.write_bytes(raw);os.replace(replacement,path)
        elif mutation=='symlink':path.rename(path.with_suffix('.prior'));path.symlink_to(path.with_suffix('.prior').name)
        elif mutation=='permissions':path.chmod(0o666)
        else:os.link(path,path.with_suffix('.linked'))
        with pytest.raises((ValueError,OSError)):authority.check()


def test_closed_authority_releases_control_descriptors(preparation_operation):
    authority=claim(preparation_operation)
    proofs=list(authority._control_proofs.values());assert len(proofs)==3
    descriptors=[proof.descriptor for proof in proofs]
    authority.close();authority.close()
    for descriptor in descriptors:
        with pytest.raises(OSError):os.fstat(descriptor)
    with pytest.raises(ValueError):authority.check()
