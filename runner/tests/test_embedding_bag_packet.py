import dataclasses

import pytest
import torch

from runner.embedding_bag_packet import EmbeddingBagPacket, collate_bags, validate_packet


CONTRACT = dict(num_embeddings=5, embedding_dim=2, mode='sum')


def packed(terminal=True, device='cpu', weights=True):
    ids = torch.tensor([1, 2, 1, 0, 3, 4, 0], dtype=torch.int64, device=device)
    offsets = torch.tensor([0, 3, 3, 5, 7] if terminal else [0, 3, 3, 5],
                           dtype=torch.int64, device=device)
    importance = torch.tensor([.5, 2, -1, 99, 3, .25, 88], device=device,
                              requires_grad=True) if weights else None
    return EmbeddingBagPacket('packed', ids, offsets, importance, terminal)


@pytest.mark.parametrize('terminal', [False, True])
def test_rebatch_keeps_whole_bags_and_rebases_offsets_weights_and_targets(terminal):
    source = packed(terminal)
    original = source.indices.clone(), source.offsets.clone(), source.per_sample_weights.clone()
    targets = torch.tensor([[10, 11], [20, 21], [30, 31], [40, 41]])
    result, selected = collate_bags(source, [3, 1, 0], targets=targets, **CONTRACT)
    assert result.layout == 'packed' and result.include_last_offset is terminal
    assert result.indices.tolist() == [4, 0, 1, 2, 1]
    assert result.offsets.tolist() == ([0, 2, 2, 5] if terminal else [0, 2, 2])
    assert result.per_sample_weights.tolist() == [.25, 88, .5, 2, -1]
    assert selected.tolist() == [[40, 41], [20, 21], [10, 11]]
    for value, expected in zip((source.indices, source.offsets, source.per_sample_weights), original):
        torch.testing.assert_close(value, expected)
    result.per_sample_weights.sum().backward()
    assert source.per_sample_weights.grad.tolist() == [1, 1, 1, 0, 0, 1, 1]


@pytest.mark.parametrize('terminal', [False, True])
def test_empty_bags_preserve_observation_count_even_when_no_items_are_selected(terminal):
    result, targets = collate_bags(packed(terminal), [1], **CONTRACT)
    assert targets is None and result.indices.shape == (0,)
    assert result.per_sample_weights.shape == (0,)
    assert result.offsets.tolist() == ([0, 0] if terminal else [0])
    assert validate_packet(result, **CONTRACT) is result


def test_fixed_rows_select_observations_and_keep_weights_differentiable():
    weights = torch.tensor([[.5, 2], [3, 4], [5, 6]], requires_grad=True)
    source = EmbeddingBagPacket('fixed', torch.tensor([[1, 2], [3, 0], [4, 1]]),
                                per_sample_weights=weights, include_last_offset=True)
    result, targets = collate_bags(source, [2, 0], targets=torch.tensor([7., 8., 9.]), **CONTRACT)
    assert result.offsets is None and result.include_last_offset is True
    assert result.indices.tolist() == [[4, 1], [1, 2]]
    assert result.per_sample_weights.tolist() == [[5, 6], [.5, 2]]
    assert targets.tolist() == [9, 7]
    result.per_sample_weights.sum().backward()
    assert weights.grad.tolist() == [[1, 1], [0, 0], [1, 1]]


@pytest.mark.parametrize('change', [
    dict(layout='other'), dict(include_last_offset=1), dict(schema_version=True),
    dict(schema_version=2), dict(indices=[1, 2]),
    dict(indices=torch.tensor([1., 2.])), dict(indices=torch.tensor([True])),
    dict(indices=torch.tensor([1, 2], dtype=torch.int32)),
    dict(indices=torch.tensor([-1, 2])), dict(indices=torch.tensor([5, 2])),
    dict(indices=torch.tensor([[1, 2]])), dict(offsets=None),
    dict(offsets=torch.tensor([1, 3, 3, 5, 7])),
    dict(offsets=torch.tensor([0, 3, 2, 5, 7])),
    dict(offsets=torch.tensor([0, 3, 3, 5, 8])),
    dict(offsets=torch.tensor([0, 3, 3, 5, 6])),
    dict(offsets=torch.tensor([0, 3, 3, 5, 7], dtype=torch.int32)),
    dict(offsets=torch.tensor([[0, 3, 3, 5, 7]])),
    dict(offsets=torch.tensor([0])),
    dict(per_sample_weights=torch.ones(7, dtype=torch.float64)),
    dict(per_sample_weights=torch.ones(7, dtype=torch.int64)),
    dict(per_sample_weights=torch.ones(1, 7)),
    dict(per_sample_weights=torch.tensor([1., 1, 1, 1, 1, 1, float('nan')])),
    dict(per_sample_weights=torch.tensor([1., 1, 1, 1, 1, 1, float('inf')])),
])
def test_malformed_packet_is_refused_before_dispatch(change):
    with pytest.raises(ValueError):
        validate_packet(dataclasses.replace(packed(), **change), **CONTRACT)


@pytest.mark.parametrize('source', [
    EmbeddingBagPacket('fixed', torch.tensor([1, 2])),
    EmbeddingBagPacket('fixed', torch.empty((0, 2), dtype=torch.int64)),
    EmbeddingBagPacket('fixed', torch.empty((2, 0), dtype=torch.int64)),
    EmbeddingBagPacket('fixed', torch.tensor([[1, 2]]), torch.tensor([0])),
])
def test_fixed_layout_requires_nonempty_rows_and_no_offsets(source):
    with pytest.raises(ValueError): validate_packet(source, **CONTRACT)


@pytest.mark.parametrize('mode', ['mean', 'max'])
def test_only_sum_accepts_per_item_weights(mode):
    with pytest.raises(ValueError, match='sum'):
        validate_packet(packed(), **{**CONTRACT, 'mode': mode})
    assert validate_packet(packed(weights=False), **{**CONTRACT, 'mode': mode}).layout == 'packed'


@pytest.mark.parametrize('changes', [
    dict(num_embeddings=True), dict(num_embeddings=0), dict(num_embeddings=5.),
    dict(embedding_dim=0), dict(embedding_dim=True), dict(mode='SUM'),
    dict(maximum_bags=True), dict(maximum_bags=0), dict(maximum_bags=3),
    dict(maximum_items=6), dict(maximum_output_elements=7),
    dict(maximum_bags=4097), dict(maximum_items=1_000_001),
    dict(maximum_output_elements=2_000_001),
])
def test_contract_types_and_table_source_output_budgets_are_checked(changes):
    with pytest.raises(ValueError): validate_packet(packed(), **{**CONTRACT, **changes})


@pytest.mark.parametrize('selection', [[], [True], [1.], [-1], [4], [1, 1], '1', torch.tensor([1])])
def test_invalid_observation_selections_cannot_duplicate_or_misalign_targets(selection):
    with pytest.raises(ValueError): collate_bags(packed(), selection, **CONTRACT)


@pytest.mark.parametrize('targets', [torch.tensor(1), torch.ones(3), [1, 2, 3, 4],
                                    torch.tensor([1., 2, float('nan'), 4])])
def test_targets_must_have_exact_finite_bag_alignment(targets):
    with pytest.raises(ValueError): collate_bags(packed(), [0], targets=targets, **CONTRACT)


def test_sparse_and_unsupported_device_packets_are_refused():
    sparse = torch.tensor([1, 2]).to_sparse()
    for change in [dict(indices=sparse), dict(offsets=torch.tensor([0, 7]).to_sparse()),
                   dict(per_sample_weights=torch.ones(7).to_sparse()),
                   dict(indices=torch.empty(7, dtype=torch.int64, device='meta'))]:
        with pytest.raises(ValueError): validate_packet(dataclasses.replace(packed(), **change), **CONTRACT)


def test_noncontiguous_dense_tensors_keep_real_values_and_gradient_path():
    values = torch.tensor([[1., 99], [2, 99], [3, 99], [4, 99], [5, 99], [6, 99], [7, 99]], requires_grad=True)
    source = dataclasses.replace(packed(), per_sample_weights=values[:, 0])
    result, _ = collate_bags(source, [2, 0], **CONTRACT)
    assert result.per_sample_weights.tolist() == [4, 5, 1, 2, 3]
    result.per_sample_weights.sum().backward()
    assert values.grad.tolist() == [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [0, 0], [0, 0]]


@pytest.mark.parametrize('terminal', [False, True])
def test_all_empty_packed_and_empty_final_bag_remain_real_observations(terminal):
    empty = EmbeddingBagPacket('packed', torch.empty(0, dtype=torch.int64),
                              torch.tensor([0, 0, 0, 0] if terminal else [0, 0, 0]),
                              torch.empty(0, requires_grad=True), terminal)
    result, targets = collate_bags(empty, [2, 0], targets=torch.tensor([10, 20, 30]), **CONTRACT)
    assert result.indices.numel() == 0
    assert result.offsets.tolist() == ([0, 0, 0] if terminal else [0, 0])
    assert targets.tolist() == [30, 10]
    final_empty = EmbeddingBagPacket('packed', torch.tensor([1, 2]),
                                    torch.tensor([0, 2, 2] if terminal else [0, 2]),
                                    include_last_offset=terminal)
    selected, _ = collate_bags(final_empty, [1, 0], **CONTRACT)
    assert selected.indices.tolist() == [1, 2]
    assert selected.offsets.tolist() == ([0, 0, 2] if terminal else [0, 0])


def test_reordered_float_targets_preserve_their_gradient_path():
    targets = torch.tensor([[1., 2], [3, 4], [5, 6], [7, 8]], requires_grad=True)
    _, selected = collate_bags(packed(), [3, 1], targets=targets, **CONTRACT)
    (selected * torch.tensor([[2., 3], [4, 5]])).sum().backward()
    assert targets.grad.tolist() == [[0, 0], [4, 5], [0, 0], [2, 3]]


@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_packet_collation_runs_on_supported_source_devices_without_index_derivatives(device):
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS unavailable; not device acceptance.')
    source = packed(device=device)
    targets = torch.tensor([1., 2, 3, 4], device=device, requires_grad=True)
    result, selected = collate_bags(source, [2, 0], targets=targets, **CONTRACT)
    assert result.indices.device.type == device
    assert result.indices.tolist() == [0, 3, 1, 2, 1]
    assert result.offsets.tolist() == [0, 2, 5]
    assert not result.indices.requires_grad and not result.offsets.requires_grad
    (result.per_sample_weights.sum() + selected.sum()).backward()
    assert source.per_sample_weights.grad.tolist() == [1, 1, 1, 1, 1, 0, 0]
    assert targets.grad.tolist() == [1, 0, 1, 0]
    if device == 'mps':
        with pytest.raises(ValueError, match='same device'):
            validate_packet(dataclasses.replace(source, offsets=source.offsets.cpu()), **CONTRACT)
        with pytest.raises(ValueError, match='same device'):
            collate_bags(source, [0], targets=targets.cpu(), **CONTRACT)


@pytest.mark.parametrize('targets', [torch.ones(4, dtype=torch.bool), torch.ones(4, dtype=torch.complex64),
                                    torch.ones(4, dtype=torch.float16), torch.ones(4).to_sparse()])
def test_target_dtype_and_storage_policy_is_explicit(targets):
    with pytest.raises(ValueError): collate_bags(packed(), [0], targets=targets, **CONTRACT)


def test_mutable_tensor_storage_is_revalidated_at_collation_boundary():
    source = packed()
    assert validate_packet(source, **CONTRACT) is source
    source.indices[0] = 99
    with pytest.raises(ValueError, match='existing table rows'):
        collate_bags(source, [3], **CONTRACT)


def test_data_budget_failure_happens_before_collation_allocations(monkeypatch):
    source = packed()
    def allocation_forbidden(*args, **kwargs):
        pytest.fail('Invalid source reached collation allocation.')
    monkeypatch.setattr(torch, 'cat', allocation_forbidden)
    monkeypatch.setattr(torch, 'tensor', allocation_forbidden)
    with pytest.raises(ValueError, match='item budget'):
        collate_bags(source, [0], maximum_items=6, **CONTRACT)
