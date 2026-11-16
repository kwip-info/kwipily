"""Portable export uses its own one-shot authority and shared native supervision."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import pytest

from runner.tests.test_evaluation_authorization import encoded,digest,ident
from runner.tests.test_worker_bootstrap import entry,run_supervisor
from runner.tests.test_lifetime_watchdog import libraries
from runner.tests.test_portable_evaluation import completed, completed_snapshot
from runner.tests.test_frozen_evaluation_cli import evaluation_fixture
from runner.portable_export_authorization import claim_export,inherit_export
from runner.portable_export_cli import execute_export


@pytest.fixture
def export_authority(tmp_path):
    project=str(uuid.uuid4()).upper();export=str(uuid.uuid4()).upper();evaluation=str(uuid.uuid4()).upper()
    folder=tmp_path.resolve()/'workflow-evaluation-exports'/project/export;folder.mkdir(parents=True,mode=0o700)
    source=tmp_path.resolve()/'workflow-evaluations'/project/evaluation;source.mkdir(parents=True)
    (source/'claim.json').write_bytes(b'original immutable source claim')
    fds=[os.open(folder/name,os.O_RDWR|os.O_CREAT|os.O_EXCL,0o600) for name in ('owner.lock','activity.lock')]
    fcntl.flock(fds[0],fcntl.LOCK_EX|fcntl.LOCK_NB)
    request=dict(schemaVersion=1,kind='portableEvaluationExport',exportID=export,projectID=project,evaluationID=evaluation,requestIdentity='a'*64,evaluationDirectory=str(source),evaluationManifestSHA256='b'*64,includeTestRows=False)
    (folder/'request.json').write_bytes(encoded(request))
    metadata=dict(schemaVersion=1,kind='portableEvaluationExport',exportID=export,projectID=project,requestIdentity=request['requestIdentity'],requestSHA256=digest(encoded(request)),directoryIdentity=ident(folder.stat()),ownerLockIdentity=ident(os.fstat(fds[0])),activityLockIdentity=ident(os.fstat(fds[1])))
    (folder/'ownership.json').write_bytes(encoded(metadata))
    yield folder,request,metadata,fds
    for fd in fds:
        if fd is not None:os.close(fd)


def claim(f):return claim_export(str(f[0]),digest(encoded(f[1])),digest(encoded(f[2])))
def control(f,mode='--kwip-export-supervise'):return [mode,str(f[0]),digest(encoded(f[1])),digest(encoded(f[2]))]


def test_export_claim_separate_inherited_activity_and_immutable_source(export_authority):
    f=export_authority;source=Path(f[1]['evaluationDirectory']);before=(source/'claim.json').read_bytes()
    with claim(f) as parent:
        child=inherit_export(str(f[0]),digest(encoded(f[1])),digest(encoded(f[2])),parent.activity_fd)
        parent.close();child.check()
        with pytest.raises(BlockingIOError):fcntl.flock(f[3][1],fcntl.LOCK_EX|fcntl.LOCK_NB)
        child.close()
    assert (source/'claim.json').read_bytes()==before
    assert json.loads((f[0]/'claim.json').read_bytes())==dict(schemaVersion=1,exportID=f[1]['exportID'],requestSHA256=digest(encoded(f[1])),ownershipSHA256=digest(encoded(f[2])),state='claimed')
    with pytest.raises((ValueError,OSError)):claim(f)


@pytest.mark.parametrize('change',['source-project','source-id','source-namespace','relative-source','option-type','extra-output','unheld-owner','request-bytes','ownership-bytes','activity-identity'])
def test_export_authority_rejects_invalid_or_changed_inputs_before_claim(export_authority,change):
    f=export_authority;r=f[1]
    if change=='source-project':r['evaluationDirectory']=str(Path(r['evaluationDirectory']).parent.parent/str(uuid.uuid4()).upper()/r['evaluationID'])
    elif change=='source-id':r['evaluationID']=str(uuid.uuid4()).upper()
    elif change=='source-namespace':r['evaluationDirectory']=r['evaluationDirectory'].replace('workflow-evaluations','other')
    elif change=='relative-source':r['evaluationDirectory']='relative/'+r['evaluationID']
    elif change=='option-type':r['includeTestRows']=1
    elif change=='extra-output':r['output']='/tmp/escape'
    elif change=='unheld-owner':os.close(f[3][0]);f[3][0]=None
    elif change=='activity-identity':(f[0]/'activity.lock').unlink();(f[0]/'activity.lock').touch(mode=0o600)
    if change not in ('request-bytes','ownership-bytes'):
        (f[0]/'request.json').write_bytes(encoded(r));f[2]['requestSHA256']=digest(encoded(r));(f[0]/'ownership.json').write_bytes(encoded(f[2]))
    elif change=='request-bytes':(f[0]/'request.json').write_bytes(b'{}')
    else:(f[0]/'ownership.json').write_bytes(b'{}')
    with pytest.raises((ValueError,OSError)):claim(f)
    assert not (f[0]/'claim.json').exists()


@pytest.mark.skipif(sys.platform!='darwin',reason='Native macOS guard')
def test_export_reexec_guard_ready_exact_cli(export_authority,tmp_path,libraries):
    f=export_authority;guard,_=libraries
    code,events,errors=run_supervisor(entry(tmp_path,guard),control(f))
    assert code==0,(events,errors)
    assert events[0]==dict(event='exportReady',exportID=f[1]['exportID'],requestSHA256=digest(encoded(f[1])),ownershipSHA256=digest(encoded(f[2])))
    assert events[1]['arguments']==['--export-evaluation',str(f[0]/'request.json'),'--export-request-sha256',digest(encoded(f[1]))]


def test_export_closed_parent_never_claims(export_authority,tmp_path):
    f=export_authority
    p=subprocess.run([sys.executable,'-I','-B',str(entry(tmp_path,tmp_path/'missing.dylib')),*control(f)],input='',capture_output=True,text=True,timeout=10)
    assert p.returncode==130 and p.stdout==''
    assert not (f[0]/'claim.json').exists()


@pytest.mark.parametrize('include',[False,True])
def test_export_cli_verifies_real_package_before_completion(completed,tmp_path,include):
    old,pin=completed;source_request=json.loads((old/'request.json').read_bytes());project=source_request['projectID'];evaluation=source_request['evaluationID']
    source=tmp_path.resolve()/'workflow-evaluations'/project/evaluation;source.parent.mkdir(parents=True);shutil.move(old,source)
    stage=tmp_path.resolve()/'workflow-evaluation-exports'/project/str(uuid.uuid4()).upper();stage.mkdir(parents=True)
    request=dict(schemaVersion=1,kind='portableEvaluationExport',exportID=stage.name,projectID=project,evaluationID=evaluation,requestIdentity='c'*64,evaluationDirectory=str(source),evaluationManifestSHA256=pin,includeTestRows=include)
    raw=encoded(request);(stage/'request.json').write_bytes(raw);events=[]
    execute_export(stage/'request.json',digest(raw),events.append)
    package=stage/'package';manifest=(package/'manifest.json').read_bytes();value=json.loads(manifest)
    # Independent complete inventory from a real worker-produced package. Native
    # staging consumes this exact completion count before checking these files.
    expected={'model.py','weights.pt','input-contract.json','fitted-parameters.json','saved-summary.json',
              'portable_evaluation.py','portable_fitted.py','preprocess_parameters.py','dataflow_temporal.py',
              'model_output_configuration.py','final_evaluation.py','execution_device.py','requirements.txt','README.md'}
    if include:expected|={'test-input.json','saved-predictions.jsonl'}
    assert {item['path'] for item in value['files']}==expected
    assert events[-1]['fileCount']==15+(2 if include else 0)
    for dependency in ('preprocess_parameters.py','dataflow_temporal.py','model_output_configuration.py'):
        assert (package/dependency).read_bytes()==Path(__file__).resolve().parents[1].joinpath(dependency).read_bytes()
    assert events[0]==dict(event='exportStarted',exportID=stage.name,requestIdentity='c'*64)
    assert events[-1]==dict(event='exportCompleted',exportID=stage.name,requestIdentity='c'*64,manifestSHA256=digest(manifest),fileCount=len(value['files'])+1,totalBytes=sum(e['bytes'] for e in value['files'])+len(manifest),includesTestRows=include)
    before=manifest
    with pytest.raises((ValueError,OSError)):execute_export(stage/'request.json',digest(raw),events.append)
    assert (package/'manifest.json').read_bytes()==before
