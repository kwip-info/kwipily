"""Saved preparation inspection is pinned, supervised and independent of sources."""
import os
from pathlib import Path
import shutil
import sys
import pytest
from runner.tests.test_operation_authorization import operation_authority,publish,claim
from runner.tests.test_evaluation_authorization import digest
from runner.tests.test_preview_operation_authorization import inventory
from runner.tests.test_dataflow_inspection import fixture
from runner.tests.test_operation_bootstrap import control
from runner.tests.test_worker_bootstrap import entry,run_supervisor
from runner.tests.test_lifetime_watchdog import libraries


def capture(f,tmp_path,policy='off'):
    options={}
    if policy=='reuse':
        previous,pin,_=fixture(tmp_path/'previous','materialize')
        options=dict(reuse_manifest=previous,reuse_manifest_sha256=pin)
    manifest,pin,events=fixture(tmp_path/'saved',policy,**options)
    destination=f[0]/'input/preparation';shutil.copytree(manifest.parent,destination)
    v=f[1]
    for key in ('runID','directoryID','phase'):v.pop(key,None)
    v.update(schemaVersion=2,toolMode='preparationInspection',arguments=['--inspect-dataflow',str(destination/'manifest.json'),'--dataflow-manifest-sha256',pin])
    inventory(f)
    return events


@pytest.fixture
def inspection(operation_authority,tmp_path):
    capture(operation_authority,tmp_path)
    return operation_authority


@pytest.mark.parametrize('mutation',['modified','replaced'])
def test_inspection_retains_full_inventory_and_policy_budget(inspection,mutation):
    f=inspection
    f[1].update(executionBudget=dict(schemaVersion=1,requestedSeconds=86400,maximumRunSeconds=86400,source='application'),remainingSeconds=86400);publish(f)
    with claim(f) as authority:
        path=f[0]/'input/preparation/dataflow.py';before=path.stat();raw=path.read_bytes()
        if mutation=='replaced':path.unlink()
        path.write_bytes((b'x'+raw[1:]) if mutation=='modified' else raw)
        os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
        with pytest.raises(ValueError):authority.check()


@pytest.mark.parametrize('mutation',['extra-flag','duplicate','wrong-path','pin','missing','extra','symlink','bool-size','oversize','registry','reordered'])
def test_invalid_inspection_cannot_claim(inspection,mutation):
    f=inspection;v=f[1]
    if mutation=='extra-flag':v['arguments']+=['--output','/tmp/output']
    elif mutation=='duplicate':v['arguments']+=v['arguments'][:2]
    elif mutation=='wrong-path':v['arguments'][1]='/tmp/manifest.json'
    elif mutation=='pin':v['arguments'][-1]='0'*64
    elif mutation=='missing':v['inputFiles'].pop()
    elif mutation=='extra':v['inputFiles'].append(dict(path='input/preparation/extra',bytes=0,sha256=digest(b'')));(f[0]/'input/preparation/extra').write_bytes(b'')
    elif mutation=='bool-size':v['inputFiles'][0]['bytes']=True
    elif mutation=='oversize':v['inputFiles'][0]['bytes']=50000001
    elif mutation=='registry':v['arguments']+=['--registry','/tmp/registry']
    elif mutation=='reordered':v['arguments']=v['arguments'][2:]+v['arguments'][:2]
    else:
        path=f[0]/'input/preparation/dataflow.py';path.rename(path.with_name('old'));path.symlink_to('old')
    publish(f)
    with pytest.raises((ValueError,OSError)):claim(f)
    assert not (f[0]/'claim.json').exists()


@pytest.mark.skipif(sys.platform!='darwin',reason='Native macOS lifetime guard')
@pytest.mark.parametrize('policy',['off','materialize','reuse'])
def test_actual_guarded_saved_inspection_matches_original(operation_authority,tmp_path,libraries,policy):
    f=operation_authority;original=capture(f,tmp_path,policy)
    # The worker receives only the private archive: all original saved directories disappear.
    shutil.rmtree(tmp_path/'saved')
    if (tmp_path/'previous').exists():shutil.rmtree(tmp_path/'previous')
    guard,_=libraries;path=entry(tmp_path,guard)
    marker='    print(json.dumps({"event":"engineMarker","arguments":result.arguments}),flush=True)'
    text=path.read_text();assert marker in text
    replacement='    from main import main\n    import dataflow\n    dataflow.execute=lambda *a,**k: (_ for _ in ()).throw(AssertionError("cannot execute sources"))\n    sys.argv=[sys.argv[0],*result.arguments]\n    sys.exit(main(check_cancelled=result.authority.check))'
    path.write_text(text.replace(marker,replacement))
    code,events,error=run_supervisor(path,control(f))
    assert code==0,(events,error)
    assert events[0]['toolMode']=='preparationInspection'
    assert [e['outputs'] for e in events if e['event']=='dataflowPreview']==[e['outputs'] for e in original if e['event']=='dataflowPreview']
    assert events[-1]['event']=='dataflowInspectionCompleted'
    assert events[-1]['cache']==original[-1]['cache']
    assert events[-1]['provenance']['sourceRevalidated'] is False
