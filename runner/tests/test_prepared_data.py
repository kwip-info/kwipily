import json
import hashlib
from pathlib import Path
import subprocess
import sys
import pytest
from runner.dataflow_cli import run_dataflow
from runner.prepared_data import load_prepared

ROOT=Path(__file__).resolve().parents[2]

def prepared(tmp_path):
    path=tmp_path/'prepared'
    run_dataflow(ROOT/'examples/dataflow/scaler.json',{'source':str(ROOT/'examples/dataflow/observations.csv')},path,lambda _:None)
    return path

def load(path,**kwargs):
    return load_prepared(path/'manifest.json','training','table','heldout','table',['feature','second'],'group',**kwargs)

def test_segmented_preparation_has_identical_model_inputs_and_provenance(tmp_path):
    from runner.workload_policy import LEGACY
    import torch
    legacy=prepared(tmp_path)
    segmented=tmp_path/'segmented'
    run_dataflow(ROOT/'examples/dataflow/scaler.json',{'source':str(ROOT/'examples/dataflow/observations.csv')},segmented,lambda _:None,workload_policy=LEGACY.to_document())
    expected,old,_=load(legacy)
    actual,new,_=load(segmented)
    assert all(torch.equal(a,b) for a,b in zip(expected,actual))
    assert old['trainingRowIDs']==new['trainingRowIDs'] and old['validationRowIDs']==new['validationRowIDs']
    assert old['preparation']['trainingDataVersion']==new['preparation']['trainingDataVersion']
    assert old['preparation']['fittedStateDigests']==new['preparation']['fittedStateDigests']

def test_exact_partitions_and_transform_lineage_without_resplit(tmp_path):
    path=prepared(tmp_path);data,provenance,files=load(path)
    assert [len(x) for x in data]==[24,24,8,8]
    assert provenance['split']=='prepared partitions; no resplitting'
    assert set(provenance['trainingRowIDs']).isdisjoint(provenance['validationRowIDs'])
    assert provenance['preparation']['fittedStateDigests']
    assert 'prep.py' in files


def test_corrupt_artifacts_and_mapping_fail_before_training(tmp_path):
    path=prepared(tmp_path);manifest=json.loads((path/'manifest.json').read_text())
    entry=next(a for a in manifest['artifacts'] if a.get('nodeID')=='training')
    (path/entry['path']).write_text('{}')
    with pytest.raises(ValueError,match='hash|digest'):load(path)
    path2=prepared(tmp_path/'other')
    with pytest.raises(ValueError,match='two'):load_prepared(path2/'manifest.json','training','table','heldout','table',['feature'],'group')


def test_real_prepared_training_preserves_ids_and_exports_prep(tmp_path):
    path=prepared(tmp_path);out=tmp_path/'trained'
    from runner.tests.test_runner import graph as make_graph
    graph=tmp_path/'model.json';graph.write_text(json.dumps(make_graph()))
    command=[sys.executable,str(ROOT/'runner/main.py'),'--graph',str(graph),'--output',str(out),'--device','cpu','--epochs','2','--prepared-dataflow',str(path/'manifest.json'),'--prepared-training-node','training','--prepared-training-port','table','--prepared-validation-node','heldout','--prepared-validation-port','table','--feature-columns','feature,second','--target-column','group']
    prepared_manifest=(path/'manifest.json').read_bytes()
    command+=['--prepared-manifest-sha256',hashlib.sha256(prepared_manifest).hexdigest(),'--prepared-graph-digest',json.loads(prepared_manifest)['graphDigest']]
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['provenance']['trainingSamples']==24 and manifest['evaluation']['count']==8
    assert manifest['settings']['partitionMode']=='prepared'
    assert (out/'preparation/prep.py').is_file()
    conflict=subprocess.run(command+['--split-seed','42'],capture_output=True,text=True)
    assert conflict.returncode==1 and 'already partitioned' in conflict.stdout


def test_different_transform_pipelines_are_rejected(tmp_path):
    path=prepared(tmp_path)
    with pytest.raises(ValueError,match='identical fitted'):
        load_prepared(path/'manifest.json','partition','training','heldout','table',['feature','second'],'group')


def test_rehashed_artifact_with_overlap_is_still_rejected(tmp_path):
    import hashlib
    from runner.dataflow import digest,membership_digest
    path=prepared(tmp_path);manifest=json.loads((path/'manifest.json').read_text())
    train_entry=next(a for a in manifest['artifacts'] if a.get('nodeID')=='training')
    val_entry=next(a for a in manifest['artifacts'] if a.get('nodeID')=='heldout')
    train=json.loads((path/train_entry['path']).read_text());val=json.loads((path/val_entry['path']).read_text())
    ids=val['lineage']['observationIDs'];ids[0]=train['lineage']['observationIDs'][0]
    val['lineage']['membershipDigest']=membership_digest(ids)
    val['lineage']['dataVersion']=digest({'columns':val['columns'],'rows':val['rows'],'ids':ids})
    raw=json.dumps(val).encode();(path/val_entry['path']).write_bytes(raw)
    val_entry.update(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
    (path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='overlap'):load(path)


def test_manifest_cannot_reference_paths_outside_its_directory(tmp_path):
    path=prepared(tmp_path);manifest=json.loads((path/'manifest.json').read_text());manifest['artifacts'][0]['path']='../escape'
    (path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='path'):load(path)


def test_model_graph_validation_precedes_prepared_tensor_loading(tmp_path):
    graph=tmp_path/'bad-model.json';graph.write_text('{"schemaVersion":999}')
    command=[sys.executable,str(ROOT/'runner/main.py'),'--graph',str(graph),'--output',str(tmp_path/'out'),'--prepared-dataflow',str(tmp_path/'missing.json'),'--prepared-training-node','training','--prepared-training-port','table','--prepared-validation-node','heldout','--prepared-validation-port','table','--feature-columns','x,y','--target-column','target']
    command+=['--prepared-manifest-sha256','0'*64,'--prepared-graph-digest','0'*64]
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode==1 and 'Unsupported graph schema' in result.stdout


def test_selected_manifest_pin_rejects_valid_replacement_before_tensor_allocation(tmp_path,monkeypatch):
    import hashlib,shutil
    path=prepared(tmp_path/'selected')
    original=(path/'manifest.json').read_bytes();selected=json.loads(original)
    # A different valid preparation replaces the path after native arguments were built.
    other_graph=json.loads((ROOT/'examples/dataflow/scaler.json').read_text())
    changed_source=tmp_path/'changed.csv'
    changed_source.write_text((ROOT/'examples/dataflow/observations.csv').read_text().replace('0.0,0,0','0.1,0,0',1))
    graph_path=tmp_path/'other.json';graph_path.write_text(json.dumps(other_graph))
    other=tmp_path/'replacement'
    run_dataflow(graph_path,{'source':str(changed_source)},other,lambda _:None)
    assert json.loads((other/'manifest.json').read_text())['graphDigest']==selected['graphDigest']
    for file in other.iterdir():shutil.copy2(file,path/file.name)
    def forbidden(*args,**kwargs):raise AssertionError('Tensor allocation preceded preparation pin verification')
    monkeypatch.setattr('runner.prepared_data.torch.tensor',forbidden)
    with pytest.raises(ValueError,match='selected manifest'):
        load(path,expected_manifest_sha256=hashlib.sha256(original).hexdigest(),expected_graph_digest=selected['graphDigest'])


def test_graph_pin_and_malformed_pin_rejected_before_tensor_allocation(tmp_path,monkeypatch):
    import hashlib
    path=prepared(tmp_path);raw=(path/'manifest.json').read_bytes()
    def forbidden(*args,**kwargs):raise AssertionError('Unexpected tensor allocation')
    monkeypatch.setattr('runner.prepared_data.torch.tensor',forbidden)
    with pytest.raises(ValueError,match='selected graph'):
        load(path,expected_manifest_sha256=hashlib.sha256(raw).hexdigest(),expected_graph_digest='0'*64)
    with pytest.raises(ValueError,match='SHA-256'):
        load(path,expected_manifest_sha256='malformed',expected_graph_digest='0'*64)


def test_pinned_regular_files_preserve_selected_identity(tmp_path):
    import hashlib
    path=prepared(tmp_path);raw=(path/'manifest.json').read_bytes();manifest=json.loads(raw)
    result=load(path,expected_manifest_sha256=hashlib.sha256(raw).hexdigest(),expected_graph_digest=manifest['graphDigest'])
    assert result[1]['preparation']['manifestSHA256']==hashlib.sha256(raw).hexdigest()
    saved=path/'original-manifest.json';(path/'manifest.json').rename(saved);(path/'manifest.json').symlink_to(saved)
    with pytest.raises(ValueError,match='regular file'):load(path)


def test_artifact_symlink_is_rejected_even_inside_directory(tmp_path):
    path=prepared(tmp_path);manifest=json.loads((path/'manifest.json').read_text());entry=manifest['artifacts'][0]
    original=path/entry['path'];saved=path/'saved.json';original.rename(saved);original.symlink_to(saved)
    with pytest.raises(ValueError,match='regular file'):load(path)


def test_nonregular_and_oversized_prepared_reads_are_bounded(tmp_path):
    import os
    from runner.prepared_data import bounded_read
    fifo=tmp_path/'fifo';os.mkfifo(fifo)
    with pytest.raises(ValueError,match='regular file'):bounded_read(fifo,20)
    oversized=tmp_path/'oversized';oversized.write_bytes(b'x'*21)
    with pytest.raises(ValueError,match='size bound'):bounded_read(oversized,20)
