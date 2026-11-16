import copy

import pytest
import torch

from runner import dataflow as df
from runner.prepared_bag_data import prepare_bag_table


def fixture(layout='packed', weighted=True, terminal=True):
    config = dict(layout=layout, maximum_items_per_bag=3,
                  per_sample_weights=weighted, include_last_offset=terminal)
    mapping = dict(schemaVersion=1, indicesColumns=['i1', 'i2', 'i3'])
    if weighted: mapping['weightColumns'] = ['w1', 'w2', 'w3']
    if layout == 'packed': mapping['lengthColumn'] = 'count'
    columns = [dict(name=n, dtype='int64', nullable=False) for n in ['i1', 'i2', 'i3']]
    columns += [dict(name=n, dtype='float64', nullable=False) for n in ['w1', 'w2', 'w3']]
    columns += [dict(name='count', dtype='int64', nullable=False), dict(name='label', dtype='int64', nullable=False),
                dict(name='desired', dtype='float64', nullable=False)]
    rows = [[1, 2, 1, .5, 2, -1, 3, 0, .25], [0, 0, 0, 4, 5, 6, 0, 1, -.5],
            [0, 3, 4, 99, 3, .25, 2, 0, .75], [4, 0, 0, .25, 88, 99, 1, 1, 1.]]
    ids = ['bag-a', 'empty-bag', 'bag-c', 'bag-d']
    item = dict(schemaVersion=1, kind='table', columns=columns, rows=rows,
                lineage=dict(role='training', observationIDs=ids, membershipDigest=df.membership_digest(ids),
                             sourceDigest='a'*64, partitionDigest='b'*64,
                             dataVersion=df.digest(dict(columns=columns, rows=rows, ids=ids)),
                             fittedStateDigests=['c'*64], diagnostics=[]))
    return item, config, mapping


def reseal(item):
    lineage = item['lineage']
    lineage['membershipDigest'] = df.membership_digest(lineage['observationIDs'])
    lineage['dataVersion'] = df.digest(dict(columns=item['columns'], rows=item['rows'], ids=lineage['observationIDs']))


def prepare(item, config, mapping, **changes):
    options = dict(role='training', source_config=config, mapping=mapping,
                   num_embeddings=5, embedding_dim=2, mode='sum',
                   expected_table_digest=df.digest(item), target_column='label')
    options.update(changes)
    return prepare_bag_table(item, **options)


@pytest.mark.parametrize('terminal', [False, True])
def test_packed_rows_materialize_real_weights_offsets_empty_bags_and_exact_target_identity(terminal):
    item, config, mapping = fixture(terminal=terminal)
    before = copy.deepcopy(item)
    packet, targets, identity = prepare(item, config, mapping)
    assert packet.indices.tolist() == [1, 2, 1, 0, 3, 4]
    assert packet.offsets.tolist() == ([0, 3, 3, 5, 6] if terminal else [0, 3, 3, 5])
    assert packet.per_sample_weights.tolist() == [.5, 2, -1, 99, 3, .25]
    assert packet.indices.dtype == torch.int64 and packet.per_sample_weights.dtype == torch.float32
    assert targets.tolist() == [0, 1, 0, 1]
    assert identity['observationIDs'] == ['bag-a', 'empty-bag', 'bag-c', 'bag-d']
    assert identity['sourceDigest'] == 'a'*64 and identity['fittedStateDigests'] == ['c'*64]
    assert identity['mapping'] == mapping and identity['sourceConfiguration'] == config
    assert identity['tableDigest'] == df.digest(item)
    assert item == before


def test_fixed_rows_keep_every_item_and_mapping_order_without_offsets():
    item, config, mapping = fixture(layout='fixed')
    mapping['indicesColumns'] = ['i3', 'i1', 'i2']
    mapping['weightColumns'] = ['w3', 'w1', 'w2']
    packet, targets, identity = prepare(item, config, mapping)
    assert packet.indices.tolist() == [[1, 1, 2], [0, 0, 0], [4, 0, 3], [0, 4, 0]]
    assert packet.per_sample_weights[0].tolist() == [-1, .5, 2]
    assert packet.offsets is None and packet.include_last_offset is True
    assert targets.tolist() == [0, 1, 0, 1]
    assert identity['mapping']['indicesColumns'] == ['i3', 'i1', 'i2']


def test_unweighted_all_empty_bags_keep_all_target_rows():
    item, config, mapping = fixture(weighted=False)
    for row in item['rows']: row[6] = 0
    reseal(item)
    packet, targets, identity = prepare(item, config, mapping, mode='mean')
    assert packet.indices.tolist() == [] and packet.offsets.tolist() == [0, 0, 0, 0, 0]
    assert packet.per_sample_weights is None and len(targets) == 4
    assert identity['observationIDs'][1] == 'empty-bag'


@pytest.mark.parametrize('change', [
    lambda c,m: c.update(extra=1), lambda c,m: c.update(layout='other'),
    lambda c,m: c.update(maximum_items_per_bag=True), lambda c,m: c.update(maximum_items_per_bag=129),
    lambda c,m: c.update(per_sample_weights=1), lambda c,m: c.update(include_last_offset=1),
    lambda c,m: m.update(schemaVersion=True), lambda c,m: m.update(extra=1),
    lambda c,m: m.update(indicesColumns=['i1', 'i1', 'i3']),
    lambda c,m: m.update(indicesColumns=['i1', 'i2']),
    lambda c,m: m.update(indicesColumns=['i1', 'i2', 'missing']),
    lambda c,m: m.update(weightColumns=['w1', 'w2']),
    lambda c,m: m.update(weightColumns=['w1', 'w2', 'w2']),
    lambda c,m: m.update(lengthColumn='label'),
    lambda c,m: m.update(weightColumns=['w1', 'w2', 'desired']),
])
def test_exact_config_mapping_types_names_width_and_target_overlap(change):
    item, config, mapping = fixture()
    change(config, mapping)
    kwargs = {}
    if mapping.get('weightColumns') == ['w1', 'w2', 'desired']:
        kwargs = objective_options(mapping['indicesColumns'])
    with pytest.raises(ValueError): prepare(item, config, mapping, **kwargs)


def objective_options(indices):
    from runner.loss_configuration import validate_recipe
    return dict(target_column=None, objective_recipe=validate_recipe(dict(schemaVersion=1, componentID='loss_mse', configurationVersion=1, config={})),
                objective_mapping=dict(schemaVersion=1, componentID='loss_mse', inputColumns=list(indices),
                                       roles=[dict(role='target', columns=['desired'], sampleShape=[1], dtype='float32')], tripletSourceGroups=[]),
                prediction_shape=[1])


def test_explicit_objective_reuses_existing_target_loader_without_binary_coercion():
    item, config, mapping = fixture()
    packet, roles, identity = prepare(item, config, mapping, **objective_options(mapping['indicesColumns']))
    assert roles['target'].shape == (4, 1) and roles['target'].dtype == torch.float32
    assert roles['target'].flatten().tolist() == [.25, -.5, .75, 1.]
    assert packet.offsets.tolist() == [0, 3, 3, 5, 6]
    assert identity['targetColumns'] == ['desired']


@pytest.mark.parametrize('column,value', [(0, True), (0, -1), (0, 5), (0, 2.5), (0, 2**63),
                                         (3, 1e39), (3, 1e-50), (3, None),
                                         (6, -1), (6, 4), (6, True), (7, 2)])
def test_every_mapped_cell_is_validated_even_in_an_empty_or_unused_suffix(column, value):
    item, config, mapping = fixture()
    item['rows'][1][column] = value
    reseal(item)
    with pytest.raises(ValueError): prepare(item, config, mapping)


@pytest.mark.parametrize('mutation', [
    lambda t: t['lineage'].update(role='validation'),
    lambda t: t['lineage'].update(sourceDigest='not-a-hash'),
    lambda t: t['lineage'].update(partitionDigest=''),
    lambda t: t['lineage'].update(fittedStateDigests=['bad']),
    lambda t: t['lineage']['observationIDs'].__setitem__(1,'bag-a'),
    lambda t: t['lineage'].update(membershipDigest='0'*64),
    lambda t: t['lineage'].update(dataVersion='0'*64),
    lambda t: t['columns'][0].update(dtype='float64'),
])
def test_artifact_role_lineage_fitted_state_and_declared_integer_type_are_checked(mutation):
    item, config, mapping = fixture()
    mutation(item)
    with pytest.raises(ValueError): prepare(item, config, mapping)


def test_wrong_captured_table_pin_refuses_before_allocation(monkeypatch):
    item, config, mapping = fixture()
    def fail(*a, **k): pytest.fail('Wrong pin reached tensor allocation')
    monkeypatch.setattr(torch, 'tensor', fail)
    with pytest.raises(ValueError): prepare(item, config, mapping, expected_table_digest='0'*64)


def test_partition_does_not_silently_lose_a_binary_class():
    item, config, mapping = fixture()
    for row in item['rows']: row[7] = 0
    reseal(item)
    with pytest.raises(ValueError): prepare(item, config, mapping)
    _, targets, _ = prepare(item, config, mapping, require_both_binary_classes=False)
    assert targets.tolist() == [0, 0, 0, 0]


@pytest.mark.parametrize('change', [
    lambda c,m: m.pop('lengthColumn'), lambda c,m: m.pop('weightColumns'),
    lambda c,m: c.update(per_sample_weights=False), lambda c,m: c.update(layout='fixed'),
    lambda c,m: m.update(lengthColumn='i1'), lambda c,m: m.update(indicesColumns=['', 'i2', 'i3']),
    lambda c,m: m.update(indicesColumns=['i1', 'i2', 'label']),
])
def test_source_variants_cannot_keep_stale_or_overlapping_mapping_fields(change):
    item, config, mapping = fixture()
    change(config, mapping)
    with pytest.raises(ValueError): prepare(item, config, mapping)


@pytest.mark.parametrize('pin', ['', 'A'*64, 'a'*63, None, True])
def test_integrity_pin_is_required_exact_lowercase_sha256(pin):
    item, config, mapping = fixture()
    with pytest.raises(ValueError): prepare(item, config, mapping, expected_table_digest=pin)


@pytest.mark.parametrize('column,value', [(0, None), (3, float('inf')), (3, float('nan')), (3, True)])
def test_missing_and_nonfinite_unused_cells_never_pack_away(column, value):
    item, config, mapping = fixture()
    pin = df.digest(item)
    item['rows'][1][column] = value
    with pytest.raises(ValueError):
        prepare_bag_table(item, role='training', source_config=config, mapping=mapping,
                          num_embeddings=5, embedding_dim=2, mode='sum',
                          expected_table_digest=pin, target_column='label')


def test_output_identity_is_detached_and_maps_sources_targets_and_fitted_chain():
    item, config, mapping = fixture()
    _, _, first = prepare(item, config, mapping)
    _, _, second = prepare(item, config, mapping)
    first['mapping']['indicesColumns'][0] = 'changed'
    first['lineage']['observationIDs'][0] = 'changed'
    first['fittedStateDigests'].append('d'*64)
    assert item['lineage']['observationIDs'][0] == 'bag-a'
    assert second['mapping'] == mapping and second['fittedStateDigests'] == ['c'*64]
    assert second['lineage'] == item['lineage']
    reordered = copy.deepcopy(mapping)
    reordered['indicesColumns'] = list(reversed(reordered['indicesColumns']))
    _, _, changed = prepare(item, config, reordered)
    assert changed['mappingDigest'] != second['mappingDigest']


def test_objective_mapping_cannot_conflict_with_bag_input_order_or_duplicate_binary_target():
    item, config, mapping = fixture()
    options = objective_options(mapping['indicesColumns'])
    options['objective_mapping']['inputColumns'].reverse()
    with pytest.raises(ValueError): prepare(item, config, mapping, **options)
    options = objective_options(mapping['indicesColumns'])
    options['target_column'] = 'label'
    with pytest.raises(ValueError): prepare(item, config, mapping, **options)


def test_objective_loader_retains_its_own_target_dtype_and_shape_refusals():
    item, config, mapping = fixture()
    options = objective_options(mapping['indicesColumns'])
    options['objective_mapping']['roles'][0]['dtype'] = 'int64'
    with pytest.raises(ValueError): prepare(item, config, mapping, **options)
    options = objective_options(mapping['indicesColumns'])
    options['prediction_shape'] = [2]
    with pytest.raises(ValueError): prepare(item, config, mapping, **options)


def test_source_row_and_output_limits_refuse_before_tensor_allocation(monkeypatch):
    item, config, mapping = fixture()
    def fail(*a, **k): pytest.fail('Invalid output geometry reached tensor allocation')
    monkeypatch.setattr(torch, 'tensor', fail)
    with pytest.raises(ValueError): prepare(item, config, mapping, embedding_dim=500_001)
    item['rows'] = item['rows'] * 1025
    with pytest.raises(ValueError): prepare(item, config, mapping)
