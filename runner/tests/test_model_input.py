import dataclasses
import hashlib
import pytest
import torch
from runner.embedding_bag_packet import EmbeddingBagPacket
from runner.model_input import (observation_count,select_observations,move_to_device,
                                canonical_input_tensors,model_inputs_digest)


def packet(terminal=False,weighted=True):
    return EmbeddingBagPacket('packed',torch.tensor([1,2,3],dtype=torch.int64),
        torch.tensor([0,2,2,3] if terminal else [0,2,2],dtype=torch.int64),
        torch.tensor([.5,2.,-1.],requires_grad=True) if weighted else None,terminal)


def test_count_uses_observations_not_items_or_terminal_boundary():
    for terminal in [False,True]:
        p=packet(terminal)
        assert observation_count(p)==3
        q=dataclasses.replace(p,indices=torch.tensor([1,2,3,4],dtype=torch.int64),
                              per_sample_weights=torch.ones(4))
        if terminal:q=dataclasses.replace(q,offsets=torch.tensor([0,2,2,4]))
        assert observation_count(q)==3
    fixed=EmbeddingBagPacket('fixed',torch.tensor([[1,2],[3,4]],dtype=torch.int64))
    assert observation_count(fixed)==2
    assert observation_count(torch.ones(5,3))==5


@pytest.mark.parametrize('terminal',[False,True])
def test_whole_bag_selection_rebases_preserves_empty_and_floating_gradients(terminal):
    p=packet(terminal)
    result=select_observations(p,[2,1,0])
    assert result.indices.tolist()==[3,1,2]
    assert result.offsets.tolist()==([0,1,1,3] if terminal else [0,1,1])
    assert result.per_sample_weights.tolist()==[-1.,.5,2.]
    (result.per_sample_weights*torch.tensor([4.,5.,6.])).sum().backward()
    assert p.per_sample_weights.grad.tolist()==[5.,6.,4.]
    assert p.indices.tolist()==[1,2,3]
    assert observation_count(result)==3


def test_tensor_selection_keeps_order_and_autograd():
    x=torch.tensor([[1.,2.],[3.,4.],[5.,6.]],requires_grad=True)
    y=select_observations(x,[2,0]);assert y.tolist()==[[5.,6.],[1.,2.]]
    y.sum().backward();assert x.grad.tolist()==[[1.,1.],[0.,0.],[1.,1.]]


@pytest.mark.parametrize('positions',[[],[True],[0,0],[-1],[3],[.5],'0'])
def test_invalid_observation_selections_refuse_for_both_types(positions):
    for value in [packet(),torch.ones(3,2)]:
        with pytest.raises(ValueError):select_observations(value,positions)


@pytest.mark.parametrize('device',['cpu','mps'])
def test_actual_device_movement_keeps_packet_fields_and_weight_gradients(device):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
    p=packet(True)
    result=move_to_device(p,device)
    assert result.schema_version==1 and result.layout=='packed' and result.include_last_offset
    assert all(t.device.type==device for _,t in canonical_input_tensors(result))
    assert [r for r,_ in canonical_input_tensors(result)]==['indices','offsets','per_sample_weights']
    result.per_sample_weights.sum().backward();assert p.per_sample_weights.grad.tolist()==[1.,1.,1.]
    x=torch.ones(2,3,requires_grad=True);y=move_to_device(x,device)
    assert y.device.type==device
    y.sum().backward();assert x.grad.tolist()==[[1.,1.,1.],[1.,1.,1.]]


def legacy_digest(values):
    h=hashlib.sha256()
    for tensor in values:
        value=tensor.detach().cpu().contiguous()
        h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode());h.update(value.numpy().tobytes())
    return h.hexdigest()


def test_ordinary_tensor_digest_is_byte_identical_including_noncontiguous_and_dtype():
    values=[torch.arange(12,dtype=torch.float32).reshape(3,4).T,
            torch.tensor([1,0],dtype=torch.int64),torch.tensor([True,False]),torch.tensor(7)]
    assert model_inputs_digest(values)==legacy_digest(values)
    assert model_inputs_digest([values[1],values[0],*values[2:]])!=model_inputs_digest(values)


def test_packet_digest_binds_geometry_weights_presence_and_values_not_device():
    p=packet()
    baseline=model_inputs_digest([p,torch.tensor([0,1,0])])
    assert len(baseline)==64 and baseline!=model_inputs_digest([packet(),torch.tensor([1,1,0])])
    mutations=[dataclasses.replace(p,offsets=torch.tensor([0,1,2])),
        dataclasses.replace(p,per_sample_weights=None),
        dataclasses.replace(p,per_sample_weights=torch.tensor([.5,3.,-1.])),
        packet(True),select_observations(p,[2,1,0])]
    for value in mutations:assert model_inputs_digest([value,torch.tensor([0,1,0])])!=baseline
    assert model_inputs_digest([packet(),torch.tensor([0,1,0])])==baseline
    if torch.backends.mps.is_available():
        assert model_inputs_digest([move_to_device(p,'mps'),torch.tensor([0,1,0],device='mps')])==baseline


@pytest.mark.parametrize('value',[None,{},torch.tensor(1),torch.ones(0,2),
    EmbeddingBagPacket('packed',torch.tensor([1]),torch.tensor([1]))])
def test_malformed_source_refuses_before_count_or_selection(value):
    with pytest.raises(ValueError):observation_count(value)
    with pytest.raises(ValueError):select_observations(value,[0])


def test_unsupported_move_and_corrupt_packet_refuse_without_fallback():
    for value in [packet(),torch.ones(2,3)]:
        with pytest.raises(ValueError):move_to_device(value,'meta')
    p=packet();p.offsets[0]=1
    with pytest.raises(ValueError):model_inputs_digest([p])
    with pytest.raises(ValueError):canonical_input_tensors(p)
