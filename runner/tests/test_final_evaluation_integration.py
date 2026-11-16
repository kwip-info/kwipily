"""Real prepared training -> frozen checkpoint -> untouched held-out evaluation."""
import hashlib
import math
import shutil
import torch
from runner.engine import run, REGISTRY
from runner.prepared_data import load_prepared, load_test_partition
from runner.frozen_checkpoint import load_frozen
from runner.final_evaluation import evaluate_classifier
from runner.tests.test_runner import graph
from runner.tests.test_prepared_test_data import fixture


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_training_frozen_holdout_and_relocated_replay(tmp_path):
    manifest,provenance,_,source=fixture(tmp_path/'source')
    prepared=load_prepared(manifest,'training','table','validation','table',['b','a'],'label')
    output=tmp_path/'trained';run(graph(),output,epochs=3,device='cpu',batch_size=7,prepared=prepared)
    pins=dict(expected_manifest_sha256=sha(output/'manifest.json'),expected_checkpoint_sha256=sha(output/'checkpoint.pt'),
              registry_path=REGISTRY,expected_registry_sha256=sha(REGISTRY))
    before=(output/'checkpoint.pt').read_bytes()
    frozen=load_frozen(output/'checkpoint.pt',**pins)
    source.unlink()  # Evaluation may not consult the live source.
    archived=output/'preparation/manifest.json'
    (x,y),holdout,_=load_test_partition(archived,'split','test',frozen.training_provenance,
                                      expected_manifest_sha256=sha(archived))
    result=evaluate_classifier(frozen.model,x,y,holdout['testRowIDs'],batch_size=5,device='cpu',preview_limit=3)
    assert result['evaluatedCount']==14 and len(result['preview'])==3 and len(result['predictions'])==14
    with torch.inference_mode():logits=frozen.model(x).double().tolist()
    losses=[math.log(sum(math.exp(v-max(pair)) for v in pair))+max(pair)-pair[int(target)] for pair,target in zip(logits,y)]
    assert math.isclose(result['lossMean'],math.fsum(losses)/len(y),rel_tol=1e-6,abs_tol=1e-7)
    assert result['perClass'][0]['support']==0 and result['perClass'][0]['recall'] is None
    assert (output/'checkpoint.pt').read_bytes()==before
    relocated=tmp_path/'relocated';shutil.copytree(output,relocated);shutil.rmtree(output)
    replay=load_frozen(relocated/'checkpoint.pt',**pins)
    (rx,ry),rp,_=load_test_partition(relocated/'preparation/manifest.json','split','test',replay.training_provenance,
                                    expected_manifest_sha256=sha(relocated/'preparation/manifest.json'))
    repeated=evaluate_classifier(replay.model,rx,ry,rp['testRowIDs'],batch_size=5,device='cpu',preview_limit=3)
    assert repeated==result
