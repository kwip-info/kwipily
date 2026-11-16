"""Exact artifact lifecycle discovery and pre-deserialization archive refusal."""
import copy,hashlib,io,json,zipfile
import pytest,torch
from runner import engine,checkpoint_lifecycle as lifecycle,frozen_checkpoint
from runner.tests.test_runner import graph

EXPECTED={'version':{'major':1,'minor':0,'patch':0},'adapterID':'kwip.runner.checkpoint-lifecycle','checkpointSchemaVersions':[1],'modes':['weights-save','continuation-save','weights-restore','continuation-restore','incompatible-artifact-refusal'],'sourceTrust':'ownedCompletedHistory','loadPolicy':'weightsOnlyCPUMap','artifactLimits':{'manifest.json':20000000,'checkpoint.pt':100000000,'metrics.json':4000000,'rate-trace.json':4000000},'requiredArtifacts':['manifest.json','checkpoint.pt','metrics.json'],'schedulerTracePolicy':'requiredExactlyWhenParentHasScheduler','saveTiming':'successfulRunCompletionAtEpochBoundary','exactResumeDevices':['cpu'],'fineTuneDevices':['cpu','mps'],'cpuResume':'epochBoundaryExactCPU','mpsResume':'weightsOnly','fineTune':'restoreModelStateResetApplicableTrainingState','parentImmutable':True,'externalImport':False,'periodicSave':False}


def test_exact_lifecycle_discovery_is_sibling_not_forward_component():
    value=engine.capabilities()
    assert value.get('lifecycleCapabilities')=={'checkpoint_lifecycle':EXPECTED}
    assert 'checkpoint_lifecycle' not in value['componentCapabilities']


def test_descriptor_is_exact_and_returns_independent_mutable_containers():
    assert hasattr(lifecycle,'capability'),'Artifact lifecycle has no dedicated capability literal'
    actual=lifecycle.capability();assert actual==EXPECTED
    actual['requiredArtifacts'].append('foreign');actual['version']['major']=99
    assert lifecycle.capability()==EXPECTED


@pytest.fixture
def saved(tmp_path):
    engine.run(graph(),tmp_path/'parent',epochs=1,device='cpu')
    directory=tmp_path/'parent';manifest=json.loads((directory/'manifest.json').read_bytes())
    return directory,manifest


def replace_archive(directory,manifest,raw):
    (directory/'checkpoint.pt').write_bytes(raw)
    manifest['checkpoint'].update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    (directory/'manifest.json').write_text(json.dumps(manifest))


@pytest.mark.parametrize('kind',['compressed','expanded','duplicate','too-many','not-zip'])
def test_bad_storage_archive_refuses_before_any_torch_deserialization(saved,monkeypatch,kind):
    directory,manifest=saved;stream=io.BytesIO()
    if kind=='not-zip':raw=b'not a ZIP storage archive'
    else:
        with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED if kind=='compressed' else zipfile.ZIP_STORED) as archive:
            if kind=='expanded':
                archive.writestr('archive/data/0',b'0'*10001)
                monkeypatch.setattr(frozen_checkpoint,'MAX_CHECKPOINT_BYTES',10000)
            elif kind=='too-many':
                for i in range(10001):archive.writestr(f'archive/data/{i}',b'')
            elif kind=='duplicate':
                archive.writestr('archive/data/0',b'one');archive.writestr('archive/data/0',b'two')
            else:archive.writestr('archive/data/0',b'0'*1000)
        raw=stream.getvalue()
    replace_archive(directory,manifest,raw)
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('Rejected archive reached Torch deserialization'))
    with pytest.raises(ValueError,match='storage|archive|expanded|uncompressed'):
        lifecycle.load(directory/'checkpoint.pt',manifest['checkpoint']['contract'],'fineTune',2)


def test_actual_saved_archive_load_retains_safe_cpu_mapping_and_parent_bytes(saved,monkeypatch):
    directory,manifest=saved;before={p.name:p.read_bytes() for p in directory.iterdir() if p.is_file()}
    calls=[];original=torch.load
    def measured(*args,**kwargs):
        calls.append(kwargs.copy());return original(*args,**kwargs)
    monkeypatch.setattr(torch,'load',measured)
    payload,history,lineage=lifecycle.load(directory/'checkpoint.pt',manifest['checkpoint']['contract'],'exactResume',2)
    assert calls and all(c['weights_only'] is True and c['map_location']=='cpu' for c in calls)
    assert payload['epoch']==1 and len(history)==1 and lineage['mode']=='exactResume'
    assert before=={p.name:p.read_bytes() for p in directory.iterdir() if p.is_file()}


class DisallowedGlobal:
    def __reduce__(self):return eval,('40 + 2',)


def test_nonallowlisted_pickle_never_gets_an_unsafe_loader_retry(saved,monkeypatch):
    directory,manifest=saved
    original=torch.load
    payload=original(directory/'checkpoint.pt',map_location='cpu',weights_only=True)
    payload['foreign']=DisallowedGlobal()
    stream=io.BytesIO();torch.save(payload,stream)
    replace_archive(directory,manifest,stream.getvalue())
    calls=[]
    def measured(*args,**kwargs):
        calls.append(kwargs.copy());assert kwargs.get('weights_only') is True
        return original(*args,**kwargs)
    monkeypatch.setattr(torch,'load',measured)
    import pickle
    with pytest.raises(pickle.UnpicklingError,match='Weights only|Unsupported global'):
        lifecycle.load(directory/'checkpoint.pt',manifest['checkpoint']['contract'],'fineTune',2)
    assert len(calls)==1 and calls[0]['map_location']=='cpu'
