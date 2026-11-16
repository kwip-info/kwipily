"""Checkpoint input identity binds complete bags without altering tensor history."""
import dataclasses
import hashlib
import pytest
import torch
from runner.checkpoint_lifecycle import tensor_digest
from runner.embedding_bag_packet import EmbeddingBagPacket
from runner.model_input import model_inputs_digest,move_to_device


def packet():
    return EmbeddingBagPacket('packed',torch.tensor([1,2,3]),torch.tensor([0,2,2]),
                              torch.tensor([.5,2.,-1.]))


def old_digest(values):
    h=hashlib.sha256()
    for tensor in values:
        value=tensor.detach().cpu().contiguous()
        h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def test_actual_checkpoint_digest_accepts_typed_bags_and_binds_every_role():
    p=packet();target=torch.tensor([0,1,0]);baseline=tensor_digest([p,target])
    assert baseline==model_inputs_digest([p,target])
    mutations=[dataclasses.replace(p,indices=torch.tensor([1,2,4])),
               dataclasses.replace(p,offsets=torch.tensor([0,1,2])),
               dataclasses.replace(p,per_sample_weights=torch.tensor([.5,3.,-1.])),
               dataclasses.replace(p,per_sample_weights=None),
               dataclasses.replace(p,offsets=torch.tensor([0,2,2,3]),include_last_offset=True),
               EmbeddingBagPacket('fixed',torch.tensor([[1],[2],[3]]),
                                  per_sample_weights=torch.tensor([[.5],[2.],[-1.]]))]
    assert len(set(tensor_digest([value,target]) for value in mutations))==len(mutations)
    for value in mutations:assert tensor_digest([value,target])!=baseline
    assert tensor_digest([p,torch.tensor([1,1,0])])!=baseline
    assert tensor_digest([target,p])!=baseline


def test_legacy_checkpoint_tensor_bytes_are_exact_including_generator_and_empty():
    values=[torch.arange(12,dtype=torch.float32).reshape(3,4).T,
            torch.tensor([1,0]),torch.tensor([True,False]),torch.tensor(7)]
    assert tensor_digest(iter(values))==old_digest(values)
    assert tensor_digest([])==old_digest([])


def test_corrupt_packet_cannot_enter_checkpoint_identity():
    p=dataclasses.replace(packet(),offsets=torch.tensor([1,2,2]))
    with pytest.raises(ValueError,match='offset'):tensor_digest([p,torch.tensor([0,1,0])])


def test_actual_mps_packet_identity_matches_cpu_without_mutating_input():
    if not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
    p=packet();target=torch.tensor([0,1,0])
    assert tensor_digest([move_to_device(p,'mps'),target.to('mps')])==tensor_digest([p,target])
    assert p.indices.device.type=='cpu' and p.per_sample_weights.device.type=='cpu'


def test_checkpoint_contract_binds_verified_bag_mapping_and_observation_identity(tmp_path):
    import copy
    from runner.checkpoint_lifecycle import contract
    from runner.tests.test_prepared_bag_loader import fixture,load
    from runner.tests.test_bag_source_graph import graph
    path=fixture(tmp_path/'source');data,provenance,_=load(path)
    value=graph();value['nodes'][0]['config']['maximum_items_per_bag']=2
    def identity(p):return contract(value,data,p,42,3,.03,'cpu',1,['source','bag','result'])
    initial=identity(provenance)
    assert initial['data']['bagInput']['modelInput']==provenance['modelInput']
    assert initial['data']['bagInput']['sources']==provenance['bagSources']
    for mutation in ['mapping','observations','sourceIdentity']:
        changed=copy.deepcopy(provenance)
        if mutation=='mapping':changed['modelInput']['mapping']['indicesColumns'].reverse()
        elif mutation=='observations':changed['bagSources']['training']['observationIDs'].reverse()
        else:changed['bagSources']['training']['sourceDigest']='f'*64
        assert identity(changed)!=initial
    provenance['modelInput']['mapping']['indicesColumns'].reverse()
    assert initial['data']['bagInput']['modelInput']!=provenance['modelInput'], 'Contract must own its immutable copy'


def test_bag_checkpoint_binds_selected_parameter_registration_order():
    from runner.checkpoint_lifecycle import contract
    from runner.tests.test_bag_source_graph import graph
    value=graph();value['nodes'].insert(2,{'id':'other','type':'embedding_bag','config':dict(value['nodes'][1]['config'])})
    provenance={'sourceHash':'a'*64,'splitHash':'b'*64}
    data=[packet(),torch.tensor([0,1,0])]
    first=contract(value,data,provenance,42,3,.03,'cpu',1,['source','bag','other','result'])
    second=contract(value,data,provenance,42,3,.03,'cpu',1,['source','other','bag','result'])
    assert first['schedule']['parameterNodeOrder']==['bag','other']
    assert second['schedule']['parameterNodeOrder']==['other','bag']
    assert first!=second
