"""Authoring rejects malformed bundles before generated registries can be shipped."""
import copy
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('component_bundles', ROOT / 'scripts/component_bundles.py')
bundles = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundles)


def source():
    return json.loads((ROOT / 'components/pooling/bundle.json').read_text())


def test_current_bundle_and_reproducible_generation():
    first = bundles.render(ROOT)
    assert first == bundles.render(ROOT)
    for path, data in first.items():
        assert (ROOT / path).read_text() == data
    registry = json.loads(first['Sources/KwipMLCore/Resources/components.json'])
    # The model-family ordering excludes separately tested control recipes.
    ids = [x['id'] for x in registry['components'] if x['id'] not in ('initialization','gradient_policy','trainability')]
    base = json.loads((ROOT / 'components/registry-base.json').read_text())['components']
    moe = [row for row in base if row['id'] == 'mixture_experts']
    assert len(moe) == 1
    assert [row for row in registry['components'] if row['id'] == 'mixture_experts'] == moe
    # Preserve the accepted pre-extension ordering separately from the two additive primitives.
    for family, ident in [('normalization', 'rms_norm'), ('attention', 'rotary_position')]:
        rows = json.loads((ROOT / 'components' / family / 'bundle.json').read_text())['components']
        expected = next(row['definition'] for row in rows if row['definition']['id'] == ident)
        assert [row for row in registry['components'] if row['id'] == ident] == [expected]
    ids = [ident for ident in ids if ident not in ('rms_norm', 'rotary_position', 'mixture_experts')]
    assert len(ids) == 87
    assert ids[-2:] == ['dataset_source', 'data_loader']
    ids = ids[:-2]
    loss_ids = ('loss_cross_entropy', 'loss_bce_with_logits', 'loss_mse', 'loss_l1', 'loss_nll', 'loss_kl_div', 'loss_ctc', 'loss_triplet_margin')
    assert tuple(ids[-8:]) == loss_ids
    loss_rows = json.loads((ROOT / 'components/losses/bundle.json').read_text())['components']
    assert [row for row in registry['components'] if row['id'] in loss_ids] == [row['definition'] for row in loss_rows]
    ids = [ident for ident in ids if ident not in ('split', 'chunk', 'gather', 'einsum', 'unfold', 'fold', 'batch_norm1d','batch_norm2d','instance_norm1d','instance_norm2d','group_norm','optimizer_sgd','optimizer_adam','optimizer_adamw','optimizer_rmsprop','optimizer_adagrad','scheduler_step_lr','scheduler_cosine_annealing_lr','scheduler_reduce_on_plateau','scheduler_one_cycle_lr','scheduler_linear_lr','rnn','lstm','gru','rnn_cell','lstm_cell','gru_cell','multihead_attention','transformer','transformer_encoder','transformer_decoder','scaled_dot_product_attention','attention_mask','loss_cross_entropy','loss_bce_with_logits','loss_mse','loss_l1','loss_nll','loss_kl_div','loss_ctc','loss_triplet_margin')]
    assert ids[3:-4] == ['max_pool1d', 'max_pool2d', 'max_pool3d', 'avg_pool1d', 'avg_pool2d', 'adaptive_avg_pool1d', 'adaptive_avg_pool2d', 'adaptive_max_pool2d', 'sigmoid', 'tanh', 'relu', 'gelu', 'softmax', 'silu', 'leaky_relu', 'elu', 'prelu', 'log_softmax', 'dropout', 'layer_norm', 'linear', 'bilinear', 'lazy_linear', 'reshape', 'transpose', 'flatten', 'matmul', 'embedding', 'index_input', 'bag_input', 'embedding_bag', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    assert ids[:3] == ['input', 'output', 'depthwise_conv2d']
    assert ids[-4:] == ['concat', 'add', 'stack', 'permute']


@pytest.mark.parametrize('mutation,match', [
    ('duplicate', 'Duplicate'), ('default', 'default'), ('nullable', 'default'),
    ('arity', 'default'), ('backend', 'backend'), ('missing', 'reference'),
    ('escape', 'reference'), ('adapter', 'adapter'), ('condition', 'condition'),
    ('theme', 'theme'), ('extra', 'Unknown'),
])
def test_reject_invalid_authoring(mutation, match):
    value = source()
    row = value['components'][1]
    if mutation == 'duplicate': value['components'].append(copy.deepcopy(row))
    elif mutation == 'default': row['definition']['fields'][0]['defaultValue'] = [0, 2]
    elif mutation == 'nullable': row['definition']['fields'][0]['defaultValue'] = None
    elif mutation == 'arity': row['definition']['fields'][0]['defaultValue'] = [2]
    elif mutation == 'backend': value['backend'] = 'unknown'
    elif mutation == 'missing': row['sources']['worker'] = 'runner/absent.py'
    elif mutation == 'escape': row['sources']['worker'] = '../outside.py'
    elif mutation == 'adapter': row['adapterID'] = 'arbitrary.eval'
    elif mutation == 'condition': row['outputConditions']['indices'] = 'unknown_field'
    elif mutation == 'theme': row['presentation']['theme'] = 'not-a-theme'
    elif mutation == 'extra': row['unexpected'] = True
    with pytest.raises(ValueError, match=match): bundles.validate(value, ROOT)


def test_digest_tracks_metadata():
    a = source(); b = copy.deepcopy(a)
    b['components'][0]['presentation']['purpose'] += ' Updated.'
    assert bundles.manifest_digest(a) != bundles.manifest_digest(b)


def test_worker_adapter_table_is_typed_and_only_advertises_published_entries():
    from runner.pooling_adapters_generated import RESOLVERS, CAPABILITIES
    from runner.avgpool1d_configuration import resolve_avgpool1d
    from runner.maxpool1d_configuration import resolve_maxpool1d
    from runner.maxpool2d_configuration import resolve_maxpool2d
    from runner.maxpool3d_configuration import resolve_maxpool3d
    assert {k:RESOLVERS[k] for k in ('avg_pool1d','max_pool1d','max_pool2d','max_pool3d')} == {'avg_pool1d':resolve_avgpool1d,'max_pool1d':resolve_maxpool1d,'max_pool2d':resolve_maxpool2d,'max_pool3d':resolve_maxpool3d}
    assert set(CAPABILITIES) == {'avg_pool1d','max_pool1d','max_pool2d','max_pool3d','avg_pool2d','adaptive_avg_pool1d','adaptive_avg_pool2d','adaptive_max_pool2d'}
    assert set(RESOLVERS) == set(CAPABILITIES)
    for row in source()['components']:
        if row['published']:
            assert CAPABILITIES[row['definition']['id']]() == row['runtimeContract']


@pytest.mark.parametrize('mutation', ['missing', 'empty', 'source', 'id', 'version'])
def test_published_learning_requires_real_content_and_example(mutation):
    value = source(); row = value['components'][0]
    if mutation == 'missing': row['learning'] = None
    elif mutation == 'empty': row['learning']['paragraphs'] = []
    elif mutation == 'source': row['learning']['exampleSource'] = 'missing.swift'
    elif mutation == 'id': row['learning']['exampleID'] = 'not-registered'
    elif mutation == 'version': row['learning']['contentVersion'] = 0
    with pytest.raises(ValueError): bundles.validate(value, ROOT)
