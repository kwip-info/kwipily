"""LayerNorm semantic and shape groundwork; not a registered executable node yet."""
import pytest
from runner.graph_semantics import effective_config, semantic_graph
from runner.tensor_layout import BatchedLayout


def node(**changes):
    config={'normalized_shape':[3], 'eps':1e-5, 'elementwise_affine':True, 'bias':True}
    config.update(changes)
    return {'id':'norm','type':'layer_norm','config':config}


def test_canonical_configuration_and_semantic_changes():
    canonical=effective_config(node(normalized_shape=[2.0,3.0]),2)
    assert canonical['normalized_shape']==[2,3]
    assert all(type(x) is int for x in canonical['normalized_shape'])
    base={'schemaVersion':2,'nodes':[node()],'edges':[]}
    for change in [{'eps':1e-4},{'elementwise_affine':False},{'bias':False},{'normalized_shape':[2,3]}]:
        assert semantic_graph(base)!=semantic_graph({**base,'nodes':[node(**change)]})
    with pytest.raises(ValueError):effective_config(node(),1)


@pytest.mark.parametrize('change',[
    {'normalized_shape':[]},{'normalized_shape':[True]},{'normalized_shape':[0]},
    {'normalized_shape':[-1]},{'normalized_shape':[1.5]},{'normalized_shape':[10**400]},
    {'normalized_shape':[1]*8},{'normalized_shape':[100000000,2]},
    {'normalized_shape':'3'},{'eps':True},{'eps':0},{'eps':-1},
    {'eps':float('inf')},{'eps':float('nan')},{'eps':10**400},
    {'elementwise_affine':1},{'bias':None},{'extra':1},
])
def test_malformed_configuration_rejected(change):
    with pytest.raises(ValueError):effective_config(node(**change),2)


def test_missing_fields_rejected():
    value=node();del value['config']['bias']
    with pytest.raises(ValueError):effective_config(value,2)


def test_trailing_axes_preserve_shape_and_exclude_batch():
    layout=BatchedLayout((2,3))
    assert layout.layer_norm([3])==layout
    assert layout.layer_norm([2,3])==layout
    for shape in ([2],[6],[1,2,3],[],[True]):
        with pytest.raises(ValueError):layout.layer_norm(shape)

@pytest.mark.parametrize('epsilon',[1e-300,1e300])
def test_epsilon_outside_float32_execution_domain_rejected(epsilon):
    with pytest.raises(ValueError,match='epsilon'):
        effective_config(node(eps=epsilon),2)
