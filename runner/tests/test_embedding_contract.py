import pytest
import torch
from runner.embedding import normalize_embedding, CheckedEmbedding


def config(**changes):
    result=dict(num_embeddings=5,embedding_dim=3,padding_idx=None,max_norm=None,norm_type=2,
                scale_grad_by_freq=False,sparse=False,freeze=False)
    result.update(changes)
    return result


def test_canonical_and_rejected_settings():
    result=normalize_embedding(config(num_embeddings=5.0,padding_idx=-1))
    assert result['num_embeddings']==5 and type(result['num_embeddings']) is int
    assert result['padding_idx']==4
    for change in [dict(num_embeddings=True),dict(embedding_dim=1.5),dict(num_embeddings=10**400),
                   dict(padding_idx=5),dict(padding_idx=-6),dict(max_norm=float('nan')),
                   dict(norm_type=0),dict(sparse=1),dict(freeze=None),dict(extra=1)]:
        with pytest.raises(ValueError):normalize_embedding(config(**change))


@pytest.mark.parametrize('frequency',[False,True])
def test_lookup_and_repeated_index_gradient(frequency):
    layer=CheckedEmbedding(config(padding_idx=0,scale_grad_by_freq=frequency)).double()
    weights=torch.arange(15,dtype=torch.float64).reshape(5,3)/4
    with torch.no_grad():layer.weight.copy_(weights)
    indices=torch.tensor([[0,2,2],[1,2,4]])
    output=layer(indices)
    torch.testing.assert_close(output,weights[indices])
    upstream=torch.arange(18,dtype=torch.float64).reshape(2,3,3)-4
    (output*upstream).sum().backward()
    expected=torch.zeros_like(weights)
    for index,grad in zip(indices.flatten().tolist(),upstream.reshape(-1,3)):
        if index:expected[index]+=grad/(3 if frequency and index==2 else 1)
    torch.testing.assert_close(layer.weight.grad,expected)
    assert torch.equal(layer.weight[0],weights[0])  # Imported padding row is not forcibly zeroed.


def test_reject_indices_before_dispatch_and_sparse_frequency_before_backward():
    layer=CheckedEmbedding(config())
    for indices in [torch.tensor([[.5]]),torch.tensor([[True]]),torch.tensor([[-1]]),torch.tensor([[5]])]:
        with pytest.raises(ValueError,match='indices'):layer(indices)
    with pytest.raises(ValueError,match='sparse.*frequency'):
        CheckedEmbedding(config(sparse=True,scale_grad_by_freq=True))


def test_frozen_max_norm_still_mutates_only_selected_rows():
    layer=CheckedEmbedding(config(freeze=True,max_norm=1.0))
    with torch.no_grad():layer.weight.fill_(2)
    before=layer.weight.clone()
    layer(torch.tensor([[1,1]]))
    assert not layer.weight.requires_grad
    assert torch.linalg.vector_norm(layer.weight[1])<=1.000001
    torch.testing.assert_close(layer.weight[[0,2,3,4]],before[[0,2,3,4]])


def test_semantic_identity_canonicalizes_padding_and_retains_options():
    from runner.graph_semantics import effective_config, semantic_graph
    node=dict(id='lookup',type='embedding',config=config(padding_idx=-1))
    assert effective_config(node,2)['padding_idx']==4
    with pytest.raises(ValueError):effective_config(node,1)
    graph=dict(schemaVersion=2,nodes=[node],edges=[])
    alias=dict(schemaVersion=2,nodes=[{**node,'config':config(padding_idx=4)}],edges=[])
    assert semantic_graph(graph)==semantic_graph(alias)
    for change in [dict(freeze=True),dict(sparse=True),dict(scale_grad_by_freq=True),dict(max_norm=1.0),dict(norm_type=1)]:
        other=dict(schemaVersion=2,nodes=[{**node,'config':config(padding_idx=4,**change)}],edges=[])
        assert semantic_graph(graph)!=semantic_graph(other)


@pytest.mark.parametrize('device',['cpu','mps'])
def test_device_lookup_backward_and_integer_widths(device):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
    layer=CheckedEmbedding(config()).to(device)
    for dtype in [torch.int32,torch.int64]:
        layer.zero_grad(set_to_none=True)
        indices=torch.tensor([[1,2,1]],dtype=dtype,device=device)
        out=layer(indices)
        torch.testing.assert_close(out,layer.weight[indices.long()])
        out.sum().backward()
        expected=torch.zeros_like(layer.weight);expected[1]=2;expected[2]=1
        torch.testing.assert_close(layer.weight.grad,expected)


def test_sparse_cpu_scatter_and_input_rank_rejection():
    layer=CheckedEmbedding(config(sparse=True))
    layer(torch.tensor([[1,2,1]])).sum().backward()
    assert layer.weight.grad.is_sparse
    expected=torch.zeros_like(layer.weight);expected[1]=2;expected[2]=1
    torch.testing.assert_close(layer.weight.grad.to_dense(),expected)
    for shape in [(3,), (1,)*8, (1,0)]:
        with pytest.raises(ValueError,match='axes'):layer(torch.zeros(shape,dtype=torch.int64))
