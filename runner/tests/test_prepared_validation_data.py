"""Frozen Validation means the exact parent holdout, never protected Test."""
import copy
import hashlib
import pytest
import torch
from runner import prepared_data
from runner.tests.test_prepared_test_data import fixture, artifact, rewrite


def load(manifest, provenance, node='validation', port='table'):
    return prepared_data.load_evaluation_partition(manifest,node,port,provenance,partition_role='validation',
        expected_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())


@pytest.mark.parametrize("segmented",[False,True])
def test_validation_exact_frozen_mapping_population_values_without_test_or_refit(tmp_path,monkeypatch,segmented):
    manifest,provenance,_,source=fixture(tmp_path,segmented=segmented)
    original=copy.deepcopy(provenance)
    tensors,_,_=prepared_data.load_prepared(manifest,'training','table','validation','table',['b','a'],'label')
    source.unlink()
    monkeypatch.setattr(prepared_data.dataflow,'transform',lambda *a,**k:pytest.fail('Validation must not transform/refit or consume Test'))
    admitted=[];admit=prepared_data.admit_model_ports
    def track(files,inventory,ports):
        admitted.append(set(ports));assert ports=={('training','table'),('validation','table')}
        return admit(files,inventory,ports)
    monkeypatch.setattr(prepared_data,'admit_model_ports',track)
    (x,y),actual,_=load(manifest,provenance)
    assert admitted==[{('training','table'),('validation','table')}]
    torch.testing.assert_close(x,tensors[2],rtol=0,atol=0)
    torch.testing.assert_close(y,tensors[3],rtol=0,atol=0)
    assert actual['dataset']=='prepared-validation-v1'
    assert actual['validationSamples']==provenance['validationSamples']==15
    assert actual['validationRowIDs']==provenance['validationRowIDs']
    assert actual['preparation']['validationNode']=='validation' and actual['preparation']['validationPort']=='table'
    assert actual['preparation']['appliedStateDigests']==[]
    assert provenance==original and not any('test' in key.lower() for key in actual)


@pytest.mark.parametrize('node,port',[('split','test'),('training','table'),('split','validation')])
def test_validation_refuses_crosswired_or_alternate_endpoint(tmp_path,node,port):
    manifest,provenance,_,_=fixture(tmp_path)
    with pytest.raises(ValueError,match='(?i)validation.*endpoint'):load(manifest,provenance,node,port)


@pytest.mark.parametrize('mutation',['ids','count','version','role','values'])
def test_validation_refuses_resealed_population_or_parent_tampering(tmp_path,mutation):
    manifest,provenance,_,_=fixture(tmp_path)
    if mutation=='ids':provenance['validationRowIDs']=list(reversed(provenance['validationRowIDs']))
    elif mutation=='count':provenance['validationSamples']+=1
    elif mutation=='version':provenance['preparation']['validationDataVersion']='0'*64
    elif mutation=='role':rewrite(manifest,provenance,'validation','table',lambda table:table['lineage'].update(role='test'))
    else:rewrite(manifest,provenance,'validation','table',lambda table:table['rows'][0].__setitem__(0,999.))
    with pytest.raises(ValueError):load(manifest,provenance)


def test_unknown_partition_role_never_defaults_to_test(tmp_path):
    manifest,provenance,_,_=fixture(tmp_path)
    with pytest.raises(ValueError,match='(?i)role'):
        prepared_data.load_evaluation_partition(manifest,'split','test',provenance,partition_role='training',
            expected_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())


def test_validation_manifest_change_during_selection_refuses(tmp_path,monkeypatch):
    manifest,provenance,_,_=fixture(tmp_path)
    monkeypatch.setattr(prepared_data,'bounded_read',lambda *a,**k:b'{}')
    with pytest.raises(ValueError,match='changed while selecting Validation'):load(manifest,provenance)
