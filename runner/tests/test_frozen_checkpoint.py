import hashlib
import json
import torch
import pytest
from runner.engine import run, REGISTRY
from runner.tests.test_runner import graph
from runner.frozen_checkpoint import load_frozen


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

@pytest.fixture
def saved(tmp_path):
    run(graph(),tmp_path,epochs=2,device='cpu')
    return tmp_path


def load(path,**overrides):
    return load_frozen(path/'checkpoint.pt',expected_manifest_sha256=sha(path/'manifest.json'),
                       expected_checkpoint_sha256=sha(path/'checkpoint.pt'),registry_path=REGISTRY,
                       expected_registry_sha256=sha(REGISTRY),**overrides)


def rewrite(path,mutate):
    manifest=json.loads((path/'manifest.json').read_text())
    checkpoint=torch.load(path/'checkpoint.pt',weights_only=True,map_location='cpu')
    mutate(manifest,checkpoint)
    torch.save(checkpoint,path/'checkpoint.pt')
    manifest['checkpoint'].update(sha256=sha(path/'checkpoint.pt'),bytes=(path/'checkpoint.pt').stat().st_size)
    (path/'manifest.json').write_text(json.dumps(manifest))


def test_frozen_load_preserves_rng_bytes_and_exact_weights(saved):
    before={p.name:p.read_bytes() for p in saved.iterdir() if p.is_file()}
    rng=torch.get_rng_state().clone()
    frozen=load(saved)
    checkpoint=torch.load(saved/'checkpoint.pt',weights_only=True,map_location='cpu')
    assert not frozen.model.training and all(not p.requires_grad for p in frozen.model.parameters())
    assert torch.equal(torch.get_rng_state(),rng)
    for name,value in frozen.model.state_dict().items():assert torch.equal(value,checkpoint['modelState'][name])
    assert frozen.training_provenance==json.loads((saved/'manifest.json').read_text())['provenance']
    assert frozen.input_contract=={'dtype':'float32','sampleShape':[2]}
    assert before=={p.name:p.read_bytes() for p in saved.iterdir() if p.is_file()}

@pytest.mark.parametrize('pin',['expected_manifest_sha256','expected_checkpoint_sha256','expected_registry_sha256'])
def test_wrong_identity_pins_reject_before_deserialization(saved,monkeypatch,pin):
    kwargs=dict(expected_manifest_sha256=sha(saved/'manifest.json'),expected_checkpoint_sha256=sha(saved/'checkpoint.pt'),
                registry_path=REGISTRY,expected_registry_sha256=sha(REGISTRY))
    kwargs[pin]='0'*64
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('Unverified bytes reached torch.load'))
    with pytest.raises(ValueError):load_frozen(saved/'checkpoint.pt',**kwargs)

@pytest.mark.parametrize('change',['dtype','shape','extra','missing','nonfinite','graph','provenance','status','epoch'])
def test_internal_incompatibilities_reject_even_when_outer_pins_match(saved,change):
    def mutate(m,c):
        key=next(iter(c['modelState']))
        if change=='dtype':c['modelState'][key]=c['modelState'][key].double()
        elif change=='shape':c['modelState'][key]=c['modelState'][key].flatten()
        elif change=='extra':c['modelState']['unexpected']=torch.zeros(1)
        elif change=='missing':del c['modelState'][key]
        elif change=='nonfinite':c['modelState'][key].flatten()[0]=float('nan')
        elif change=='graph':m['graph']['nodes'][3]['type']='tanh'
        elif change=='provenance':m['provenance']['sourceHash']='0'*64
        elif change=='status':m['status']='cancelled'
        else:c['epoch']=999
    rewrite(saved,mutate)
    with pytest.raises(ValueError):load(saved)


def test_optimizer_and_rng_payload_are_not_restored(saved):
    rewrite(saved,lambda m,c:(c.update(optimizerState={'not':'an optimizer'},rngState=torch.zeros(1))))
    frozen=load(saved)
    assert not frozen.model.training


def test_terminal_symlink_is_rejected(saved,tmp_path):
    link=tmp_path/'alias.pt';link.symlink_to(saved/'checkpoint.pt')
    with pytest.raises(ValueError):
        load_frozen(link,expected_manifest_sha256=sha(saved/'manifest.json'),expected_checkpoint_sha256=sha(saved/'checkpoint.pt'),
                    registry_path=REGISTRY,expected_registry_sha256=sha(REGISTRY))


def test_compressed_checkpoint_expansion_rejected_before_torch_load(saved,monkeypatch):
    import zipfile
    import runner.frozen_checkpoint as frozen
    stream=__import__('io').BytesIO()
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('archive/data/0',b'0'*10001)
    (saved/'checkpoint.pt').write_bytes(stream.getvalue())
    manifest=json.loads((saved/'manifest.json').read_text())
    manifest['checkpoint'].update(sha256=sha(saved/'checkpoint.pt'),bytes=len(stream.getvalue()))
    (saved/'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(frozen,'MAX_CHECKPOINT_BYTES',10000)
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('Expanded storage budget must be checked before deserialization'))
    with pytest.raises(ValueError,match='storage|archive|expan'):load(saved)
