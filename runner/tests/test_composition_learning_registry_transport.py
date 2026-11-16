"""Captured registry is an optional exact third retained assessment input only."""
import os
import pytest
from runner.tests.test_operation_authorization import operation_authority,publish,claim
from runner.tests.test_evaluation_authorization import encoded,digest
from runner.worker_bootstrap import RuntimeContext,_arguments

@pytest.fixture
def captured(operation_authority):
    f=operation_authority;value=f[1]
    for key in ('runID','directoryID','phase'):del value[key]
    folder=f[0]/'input';folder.mkdir()
    value.update(schemaVersion=2,toolMode='learningAssessment',inputFiles=[])
    for name in ('assessment.json','submission.json','model-registry.json'):
        raw=encoded({'kind':name});(folder/name).write_bytes(raw)
        value['inputFiles'].append(dict(path='input/'+name,bytes=len(raw),sha256=digest(raw)))
    value['arguments']=['--learning-evidence-request',str(folder/'assessment.json'),'--learning-submission',str(folder/'submission.json'),'--learning-submission-sha256',value['inputFiles'][1]['sha256'],'--registry',str(folder/'model-registry.json')]
    publish(f);return f

def test_captured_third_registry_is_verified_and_rechecked(captured):
    f=captured
    with claim(f) as authority:
        authority.check()
        (f[0]/'input/model-registry.json').write_bytes(b'changed')
        with pytest.raises((ValueError,OSError)):authority.check()

def test_bootstrap_preserves_only_exact_captured_registry(captured,tmp_path):
    f=captured;runtime=RuntimeContext(tmp_path/'entry.py',tmp_path/'guard.dylib',tmp_path/'runtime-registry.json')
    with claim(f):assert _arguments(f[1],runtime)==f[1]['arguments']

@pytest.mark.parametrize('change',['missing-file','missing-flag','foreign-path','duplicate-flag','foreign-extra','runtime-mode','oversize','changed','symlink','hardlink'])
def test_unpinned_foreign_tampered_or_unrequested_registry_never_claims(captured,change):
    f=captured;value=f[1];target=f[0]/'input/model-registry.json'
    if change=='missing-file':value['inputFiles'].pop()
    elif change=='missing-flag':value['arguments']=value['arguments'][:-2]
    elif change=='foreign-path':value['arguments'][-1]=str(target.with_name('foreign.json'))
    elif change=='duplicate-flag':value['arguments']+=value['arguments'][-2:]
    elif change=='foreign-extra':value['inputFiles'][-1]['path']='input/foreign.json'
    elif change=='runtime-mode':value['toolMode']='learningRuntime'
    elif change=='oversize':value['inputFiles'][-1]['bytes']=2000001
    elif change=='changed':target.write_bytes(b'changed')
    elif change=='symlink':target.unlink();target.symlink_to('assessment.json')
    else:os.link(target,target.with_name('alias.json'))
    publish(f)
    with pytest.raises((ValueError,OSError)):claim(f)
    assert not (f[0]/'claim.json').exists()
