"""Retained native trained-package snapshot, not a synthetic replacement model."""
import hashlib
import json
from pathlib import Path
import uuid
import pytest
import torch
from runner.program_export_cli import execute_export
from runner.program_export_configuration import DEFAULTS
from runner.program_export import reload_program, _canonical
from runner.tests.test_portable_inference import bundle


def sha(raw): return hashlib.sha256(raw).hexdigest()


def request(tmp_path):
    stage = tmp_path/'workflow-program-exports'/str(uuid.uuid4()).upper()/str(uuid.uuid4()).upper()
    stage.mkdir(parents=True)
    source, _, reference = bundle(stage)
    source.rename(stage/'source'); source = stage/'source'
    manifest = json.loads((source/'export-manifest.json').read_bytes())
    manifest.update(runID=str(uuid.uuid4()).upper(), requestIdentity='a'*64, modelManifestSHA256='b'*64)
    manifest_raw = _canonical(manifest); (source/'export-manifest.json').write_bytes(manifest_raw)
    value = {'schemaVersion':1,'kind':'programExport','projectID':stage.parent.name,'exportID':stage.name,'requestIdentity':'c'*64,'sourceDirectory':str(source),'sourceManifestSHA256':sha(manifest_raw),'configuration':dict(DEFAULTS)}
    raw = _canonical(value); path = stage/'request.json'; path.write_bytes(raw)
    return path, sha(raw), reference


def test_actual_native_trained_source_snapshot_to_program_and_source_unchanged(tmp_path):
    path, pin, reference = request(tmp_path)
    before = {p.name:p.read_bytes() for p in (path.parent/'source').iterdir() if p.is_file()}
    events=[]; result = execute_export(path,pin,events.append)
    assert [e['type'] for e in events] == ['programExportStarted','programExportCompleted']
    manifest = json.loads((path.parent/'package'/'manifest.json').read_bytes())
    assert result == manifest
    raw = (path.parent/'package'/'program.pt2').read_bytes()
    model = reload_program(raw,manifest,expected_manifest_sha256=sha(_canonical(manifest)))
    x = torch.tensor([[.25,-.5],[1.,2.]])
    torch.testing.assert_close(model(x),reference(x))
    assert before == {p.name:p.read_bytes() for p in (path.parent/'source').iterdir() if p.is_file()}
    assert manifest['sourceIdentity']['exportManifestSHA256'] == json.loads(path.read_bytes())['sourceManifestSHA256']


@pytest.mark.parametrize('corruption',['requestPin','sourcePin','model','helper','symlink','foreignSource','packet','existingOutput'])
def test_invalid_source_or_authority_refuses_before_program_capture(tmp_path,monkeypatch,corruption):
    path,pin,_=request(tmp_path); value=json.loads(path.read_bytes()); source=path.parent/'source'
    if corruption=='requestPin': pin='0'*64
    elif corruption=='sourcePin':value['sourceManifestSHA256']='0'*64
    elif corruption in ('model','helper'):
        with (source/('model.py' if corruption=='model' else 'inference.py')).open('ab') as file:file.write(b'changed')
    elif corruption=='symlink':
        original=(source/'checkpoint.pt').read_bytes(); outside=tmp_path/'outside.pt';outside.write_bytes(original);(source/'checkpoint.pt').unlink();(source/'checkpoint.pt').symlink_to(outside)
    elif corruption=='foreignSource':value['sourceDirectory']=str(tmp_path)
    elif corruption=='packet':
        contract={'schemaVersion':3,'input':{'kind':'bagPacket'}};raw=_canonical(contract);(source/'input-contract.json').write_bytes(raw)
        manifest=json.loads((source/'export-manifest.json').read_bytes())
        for row in manifest['files']:
            if row['path']=='input-contract.json':row.update(bytes=len(raw),sha256=sha(raw))
        raw=_canonical(manifest);(source/'export-manifest.json').write_bytes(raw);value['sourceManifestSHA256']=sha(raw)
    else:(path.parent/'package').mkdir()
    if corruption!='requestPin':raw=_canonical(value);path.write_bytes(raw);pin=sha(raw)
    import runner.program_export_cli as cli
    monkeypatch.setattr(cli,'build_package',lambda *a,**k:pytest.fail('invalid source reached capture'))
    with pytest.raises((ValueError,OSError)):execute_export(path,pin,lambda _:None)


def test_objective_trained_source_keeps_real_output_geometry(tmp_path):
    from runner.tests.test_objective_portable_inference import package
    stage=tmp_path/'workflow-program-exports'/str(uuid.uuid4()).upper()/str(uuid.uuid4()).upper();stage.mkdir(parents=True)
    source,_,reference,x,*_=package(stage,'loss_mse')
    source.rename(stage/'source');source=stage/'source'
    manifest=json.loads((source/'export-manifest.json').read_bytes());manifest.update(runID=str(uuid.uuid4()).upper(),requestIdentity='a'*64,modelManifestSHA256='b'*64)
    raw=_canonical(manifest);(source/'export-manifest.json').write_bytes(raw)
    value={'schemaVersion':2,'kind':'programExport','projectID':stage.parent.name,'exportID':stage.name,'requestIdentity':'c'*64,'sourceDirectory':str(source),'sourceManifestSHA256':sha(raw),'configuration':dict(DEFAULTS),'executionBudget':{'schemaVersion':1,'requestedSeconds':60,'maximumRunSeconds':60,'source':'application'}}
    raw=_canonical(value);path=stage/'request.json';path.write_bytes(raw)
    result=execute_export(path,sha(raw),lambda _:None)
    model=reload_program((stage/'package'/'program.pt2').read_bytes(),result,expected_manifest_sha256=sha(_canonical(result)))
    assert result['output']['sampleShape']==list(reference(x[:2]).shape[1:])
    torch.testing.assert_close(model(x[:2]),reference(x[:2]))


def test_cancel_and_source_drift_between_capture_boundaries_never_complete(tmp_path):
    path,pin,_=request(tmp_path)
    events=[]
    def cancelled(): raise InterruptedError('owner cancelled')
    with pytest.raises(InterruptedError):execute_export(path,pin,events.append,check_cancelled=cancelled)
    assert not events and not (path.parent/'package').exists()
    def mutate(event):
        events.append(event)
        if event['type']=='programExportStarted':
            with (path.parent/'source'/'model.py').open('ab') as file:file.write(b'changed')
    with pytest.raises(ValueError,match='changed|bound'):execute_export(path,pin,mutate)
    assert all(e['type']!='programExportCompleted' for e in events)


def test_source_directory_replacement_cannot_reuse_open_old_directory(tmp_path):
    path,pin,_=request(tmp_path);events=[]
    def replace(event):
        events.append(event)
        if event['type']=='programExportStarted':
            (path.parent/'source').rename(path.parent/'retired-source')
            (path.parent/'source').mkdir()
    with pytest.raises(ValueError,match='directory|identity'):execute_export(path,pin,replace)
    assert all(e['type']!='programExportCompleted' for e in events)

@pytest.mark.parametrize('name,size,accepted',[('data_loading_recipe.py',131072,True),('data_loading_recipe.py',131073,False),('unreviewed_loading.py',32,False)])
def test_reviewed_data_loading_helper_has_exact_name_and_small_bound(tmp_path,name,size,accepted):
    path,_,_=request(tmp_path)
    source=path.parent/'source';manifest=json.loads((source/'export-manifest.json').read_bytes())
    raw=b'#'+b' '*(size-1);(source/name).write_bytes(raw)
    manifest['files'].append(dict(path=name,bytes=size,sha256=sha(raw)))
    manifest_raw=_canonical(manifest);(source/'export-manifest.json').write_bytes(manifest_raw)
    inner=json.loads(path.read_bytes());inner['sourceManifestSHA256']=sha(manifest_raw)
    inner_raw=_canonical(inner);path.write_bytes(inner_raw)
    if accepted:
        execute_export(path,sha(inner_raw),lambda _:None)
        assert (source/name).read_bytes()==raw
    else:
        with pytest.raises((ValueError,OSError)):execute_export(path,sha(inner_raw),lambda _:None)
        assert not (path.parent/'package').exists()
