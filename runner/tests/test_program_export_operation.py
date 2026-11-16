"""Program export reuses real retained one-shot claim and pinned source inventory."""
import json
import os
import subprocess
import sys
import uuid
import pytest
from runner.tests.test_operation_authorization import operation_authority, publish, claim
from runner.tests.test_evaluation_authorization import encoded,digest
from runner.tests.test_portable_inference import bundle
from runner.program_export_configuration import DEFAULTS
from runner.program_export_operation_contract import parse_contract,verify_inputs
from runner.operation_authorization import validate_request


def configure(f):
    folder,outer,_,_=f
    source,_,reference=bundle(folder);source.rename(folder/'source');source=folder/'source'
    manifest=json.loads((source/'export-manifest.json').read_bytes());manifest.update(runID=str(uuid.uuid4()).upper(),requestIdentity='b'*64,modelManifestSHA256='c'*64)
    raw=encoded(manifest);(source/'export-manifest.json').write_bytes(raw)
    inner=dict(schemaVersion=1,kind='programExport',projectID=outer['projectID'],exportID=outer['operationID'],requestIdentity=outer['requestIdentity'],sourceDirectory=str(source),sourceManifestSHA256=digest(raw),configuration=dict(DEFAULTS))
    (folder/'input').mkdir();inner_raw=encoded(inner);(folder/'input/program-export.json').write_bytes(inner_raw)
    files=[dict(path='source/'+row['path'],bytes=row['bytes'],sha256=row['sha256']) for row in manifest['files']]
    files += [dict(path='source/export-manifest.json',bytes=len(raw),sha256=digest(raw)),dict(path='input/program-export.json',bytes=len(inner_raw),sha256=digest(inner_raw))]
    common={k:outer[k] for k in ('kind','operationID','projectID','requestIdentity','executionBudget','remainingSeconds')}
    outer.clear();outer.update(common,schemaVersion=2,toolMode='programExport',arguments=['--export-program',str(folder/'input/program-export.json'),'--program-request-sha256',digest(inner_raw)],inputFiles=files)
    publish(f);return inner,reference


def test_real_operation_claim_keeps_exact_source_and_one_shot_identity(operation_authority):
    f=operation_authority;configure(f)
    assert validate_request(f[1])
    assert parse_contract(f[1],f[0])[1]
    with claim(f) as owner:
        owner.check();assert owner.request['toolMode']=='programExport'
        with pytest.raises((ValueError,OSError)):claim(f)
    assert (f[0]/'claim.json').is_file()


@pytest.mark.parametrize('mutation',['requestPin','foreignSource','wrongIdentity','sourceFile','extraInventory','packet','config','remaining','foreignArgument'])
def test_forged_program_operation_refuses_before_claim(operation_authority,mutation):
    f=operation_authority;inner,_=configure(f);folder=f[0]
    if mutation=='requestPin':f[1]['arguments'][3]='0'*64
    elif mutation in ('foreignSource','wrongIdentity','config'):
        if mutation=='foreignSource':inner['sourceDirectory']=str(folder.parent)
        elif mutation=='wrongIdentity':inner['exportID']=str(uuid.uuid4()).upper()
        else:inner['configuration']['maximum_batch_size']=9
        raw=encoded(inner);(folder/'input/program-export.json').write_bytes(raw)
        f[1]['arguments'][3]=digest(raw)
        for row in f[1]['inputFiles']:
            if row['path']=='input/program-export.json':row.update(bytes=len(raw),sha256=digest(raw))
    elif mutation=='sourceFile':(folder/'source/model.py').write_bytes(b'changed')
    elif mutation=='extraInventory':f[1]['inputFiles'].append(dict(path='source/private.csv',bytes=0,sha256='a'*64))
    elif mutation=='packet':
        # A freshly pinned inventory still may not redirect to packet input.
        raw=encoded({'schemaVersion':3,'input':{'kind':'bagPacket'}});(folder/'source/input-contract.json').write_bytes(raw)
        source_manifest=json.loads((folder/'source/export-manifest.json').read_bytes())
        for row in source_manifest['files']:
            if row['path']=='input-contract.json':row.update(bytes=len(raw),sha256=digest(raw))
        manifest_raw=encoded(source_manifest);(folder/'source/export-manifest.json').write_bytes(manifest_raw)
        inner['sourceManifestSHA256']=digest(manifest_raw);inner_raw=encoded(inner);(folder/'input/program-export.json').write_bytes(inner_raw);f[1]['arguments'][3]=digest(inner_raw)
        for row in f[1]['inputFiles']:
            p=folder/row['path'];value=p.read_bytes();row.update(bytes=len(value),sha256=digest(value))
    elif mutation=='remaining':f[1]['remainingSeconds']=True
    else:f[1]['arguments']+=['--graph','foreign']
    publish(f)
    with pytest.raises((ValueError,OSError)):claim(f)
    assert not (folder/'claim.json').exists()


def test_plain_main_cannot_execute_owned_program_without_managed_claim(operation_authority):
    f=operation_authority;configure(f)
    from pathlib import Path
    main=Path(__file__).resolve().parents[1]/'main.py'
    result=subprocess.run([sys.executable,str(main),*f[1]['arguments']],capture_output=True,text=True,timeout=30)
    assert result.returncode!=0
    assert 'managed' in result.stdout.lower() or 'authority' in result.stdout.lower()
    assert not (f[0]/'package').exists()


@pytest.mark.skipif(sys.platform!='darwin',reason='Native macOS lifetime guard')
@pytest.mark.parametrize('timeout',[False,True])
def test_actual_managed_supervisor_captures_or_times_out_and_releases_activity(operation_authority,tmp_path,libraries,timeout):
    import fcntl
    from runner.tests.test_worker_bootstrap import entry,run_supervisor
    from runner.tests.test_operation_bootstrap import control
    from runner.program_export import reload_program,_canonical
    f=operation_authority;_,reference=configure(f);guard,_=libraries;path=entry(tmp_path,guard)
    f[1]['remainingSeconds']=1 if timeout else 30
    f[1]['executionBudget'].update(requestedSeconds=30,maximumRunSeconds=30);publish(f)
    marker='    print(json.dumps({"event":"engineMarker","arguments":result.arguments}),flush=True)'
    code='    from main import main\n    sys.argv=[sys.argv[0],*result.arguments]\n    sys.exit(main(check_cancelled=result.authority.check))'
    if timeout:code='    import time\n    time.sleep(30)\n'+code
    text=path.read_text();assert marker in text;path.write_text(text.replace(marker,code))
    status,events,error=run_supervisor(path,control(f))
    assert status==(124 if timeout else 0),(events,error)
    assert events[0]['event']=='operationReady' and events[0]['toolMode']=='programExport'
    if timeout:assert not (f[0]/'package').exists()
    else:
        assert [e['event'] for e in events]==['operationReady','programExportStarted','programExportCompleted']
        manifest=json.loads((f[0]/'package/manifest.json').read_bytes())
        raw=(f[0]/'package/program.pt2').read_bytes()
        import torch
        x=torch.tensor([[1.,-2.],[.5,3.]])
        torch.testing.assert_close(reload_program(raw,manifest,expected_manifest_sha256=digest(_canonical(manifest)))(x),reference(x))
        assert events[-1]['manifestSHA256']==digest((f[0]/'package/manifest.json').read_bytes())
    fcntl.flock(f[3][1],fcntl.LOCK_EX|fcntl.LOCK_NB)

from runner.tests.test_lifetime_watchdog import libraries


def test_managed_loading_helper_name_budget_is_independent_and_exact():
    from runner.program_export_operation_contract import file_limit
    assert file_limit('source/data_loading_recipe.py')==131072
    with pytest.raises(ValueError):file_limit('source/unreviewed_loading.py')
