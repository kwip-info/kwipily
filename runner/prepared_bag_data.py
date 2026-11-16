"""Pinned table conversion; callers retain completed-inventory/source authority.

The required canonical table digest establishes integrity only. A caller must
first admit the captured artifact through completed_inventory and its source
graph/partition authority. A self-consistent table is not permission to execute.
"""
import copy
import math
import re
import struct

import torch

if __package__:
    from . import dataflow as df
    from .dataflow_cache import table_from_artifact
    from .embedding_bag_packet import EmbeddingBagPacket, validate_packet
    from .loss_prepared_data import table_objective_tensors
    from .loss_target_mapping import validate_target_mapping
else:
    import dataflow as df
    from dataflow_cache import table_from_artifact
    from embedding_bag_packet import EmbeddingBagPacket, validate_packet
    from loss_prepared_data import table_objective_tensors
    from loss_target_mapping import validate_target_mapping


def prepare_bag_table(item, *, role, source_config, mapping, num_embeddings,
                      embedding_dim, mode, expected_table_digest,
                      target_column=None, objective_mapping=None,
                      objective_recipe=None, prediction_shape=None,
                      require_both_binary_classes=True):
    """Return (real packet, binary targets or objective roles, copied identity).

    One prepared row remains one bag, including empty packed bags. Every mapped
    item/weight is checked before dropping unused suffixes. Explicit objective
    roles are converted by the existing objective loader, not reinterpreted here.
    """
    config = _source_configuration(source_config)
    mapped = _mapping(mapping, config)
    if type(role) is not str or role not in ('training', 'validation', 'test'):
        raise ValueError('Choose an explicit training, validation or test partition.')
    if type(require_both_binary_classes) is not bool:
        raise ValueError('Binary class-presence policy must be boolean.')
    if (type(num_embeddings) is not int or type(embedding_dim) is not int
            or not 1 <= num_embeddings <= 100_000_000 or not 1 <= embedding_dim <= 100_000_000
            or num_embeddings > 100_000_000 // embedding_dim):
        raise ValueError('Bag table dimensions require bounded positive whole numbers.')
    if type(mode) is not str or mode not in ('sum', 'mean', 'max'):
        raise ValueError('Choose a supported bag reduction.')
    if config['per_sample_weights'] and mode != 'sum':
        raise ValueError('Per-sample weights require sum reduction.')
    if type(expected_table_digest) is not str or re.fullmatch('[0-9a-f]{64}', expected_table_digest) is None:
        raise ValueError('Prepared bags require the exact lowercase SHA-256 captured table pin.')
    if (type(item) is not dict or type(item.get('rows')) is not list
            or not 1 <= len(item['rows']) <= 4096 or type(item.get('lineage')) is not dict):
        raise ValueError('Prepared bags require a bounded typed table with 1..4096 observations.')
    try: actual_digest = df.digest(item)
    except (TypeError, OverflowError, ValueError):
        raise ValueError('Prepared bag artifact must contain bounded finite serializable values.') from None
    if actual_digest != expected_table_digest:
        raise ValueError('Prepared bag table differs from its captured integrity pin.')
    table = table_from_artifact(item)
    if table.role != role:
        raise ValueError('Prepared bag partition role differs from the requested role.')
    if any(re.fullmatch('[0-9a-f]{64}', value) is None
           for value in (table.source_digest, table.partition_digest)):
        raise ValueError('Prepared bags require exact source and partition digest identities.')
    if len(table.rows) * embedding_dim > 2_000_000:
        raise ValueError('Bag outputs exceed the data-element budget.')
    names = [column['name'] for column in table.columns]
    source_columns = list(mapped['indicesColumns']) + list(mapped.get('weightColumns', []))
    if config['layout'] == 'packed': source_columns.append(mapped['lengthColumn'])
    if len(set(source_columns)) != len(source_columns) or any(name not in names for name in source_columns):
        raise ValueError('Bag item, weight and length columns must be distinct and present in the table.')
    index_positions = [names.index(name) for name in mapped['indicesColumns']]
    if any(table.columns[names.index(name)]['dtype']=='image' for name in [*source_columns,*([target_column] if target_column in names else [])]):raise ValueError('Image columns require an explicit image adapter; they cannot supply bag indices, weights, lengths or targets.')
    weight_positions = [names.index(name) for name in mapped.get('weightColumns', [])]
    length_position = names.index(mapped['lengthColumn']) if config['layout'] == 'packed' else None
    if any(table.columns[position]['dtype'] != 'int64' for position in index_positions):
        raise ValueError('Bag item columns require explicit int64 schema; cast them before mapping.')
    if length_position is not None and table.columns[length_position]['dtype'] != 'int64':
        raise ValueError('Bag length requires an explicit int64 column.')
    if any(table.columns[position]['dtype'] not in ('int64', 'float64') for position in weight_positions):
        raise ValueError('Bag per-sample weights require real numeric columns.')
    objective = objective_mapping is not None
    if objective:
        if target_column is not None or objective_recipe is None or prediction_shape is None:
            raise ValueError('Explicit objectives require their recipe, prediction shape and no duplicate binary target.')
        targets_mapping = validate_target_mapping(objective_mapping, objective_recipe)
        if targets_mapping['inputColumns'] != mapped['indicesColumns']:
            raise ValueError('Objective input columns must exactly match the ordered bag item columns.')
        target_columns = [name for descriptor in targets_mapping['roles'] for name in descriptor['columns']]
        target_identity = {'objectiveMapping': targets_mapping, 'objectiveRecipe': copy.deepcopy(objective_recipe),
                           'predictionShape': copy.deepcopy(prediction_shape)}
    else:
        if objective_recipe is not None or prediction_shape is not None:
            raise ValueError('Do not supply incomplete objective controls with a binary target.')
        if not _name(target_column) or target_column not in names:
            raise ValueError('Choose one present, nonblank binary target column.')
        target_columns = [target_column]
        target_identity = {'binaryTargetColumn': target_column,
                           'requireBothBinaryClasses': require_both_binary_classes}
    if set(source_columns).intersection(target_columns):
        raise ValueError('Bag source columns and target-role columns must remain separate.')
    # Validate all selected cells before packing: even empty bags retain checked
    # source values, and suffix removal cannot conceal invalid coordinates.
    id_rows = []
    weight_rows = []
    lengths = []
    for row in table.rows:
        indices = [row[position] for position in index_positions]
        if any(type(value) is not int or not 0 <= value < num_embeddings or value >= 2**63 for value in indices):
            raise ValueError('Every mapped item must be an exact int64 ID selecting an existing table row.')
        weights = [_float32(row[position]) for position in weight_positions]
        length = row[length_position] if length_position is not None else config['maximum_items_per_bag']
        if type(length) is not int or not 0 <= length <= config['maximum_items_per_bag']:
            raise ValueError('Bag lengths must be exact int64 values from zero to the configured row capacity.')
        id_rows.append(indices)
        weight_rows.append(weights)
        lengths.append(length)
    if objective:
        _, targets, _, _ = table_objective_tensors(item, role, targets_mapping, objective_recipe,
                           'int64', [config['maximum_items_per_bag']], prediction_shape)
    else:
        target_position = names.index(target_column)
        if table.columns[target_position]['dtype'] != 'int64':
            raise ValueError('Binary target requires an explicit int64 column.')
        labels = [row[target_position] for row in table.rows]
        if any(type(value) is not int or value not in (0, 1) for value in labels):
            raise ValueError('Binary targets must be exact whole-number zero or one.')
        if require_both_binary_classes and set(labels) != {0, 1}:
            raise ValueError('Prepared binary partitions must retain both classes.')
        targets = torch.tensor(labels, dtype=torch.int64)
    if config['layout'] == 'fixed':
        ids = torch.tensor(id_rows, dtype=torch.int64)
        weights = torch.tensor(weight_rows, dtype=torch.float32) if config['per_sample_weights'] else None
        offsets = None
    else:
        flattened = [value for row, length in zip(id_rows, lengths) for value in row[:length]]
        importance = [value for row, length in zip(weight_rows, lengths) for value in row[:length]]
        starts = [0]
        for length in lengths: starts.append(starts[-1] + length)
        if not config['include_last_offset']: starts.pop()
        ids = torch.tensor(flattened, dtype=torch.int64)
        weights = torch.tensor(importance, dtype=torch.float32) if config['per_sample_weights'] else None
        offsets = torch.tensor(starts, dtype=torch.int64)
    packet = EmbeddingBagPacket(config['layout'], ids, offsets, weights, config['include_last_offset'])
    validate_packet(packet, num_embeddings=num_embeddings, embedding_dim=embedding_dim, mode=mode)
    identity = {'schemaVersion': 1, 'tableDigest': expected_table_digest,
                'sourceConfiguration': copy.deepcopy(config), 'mapping': copy.deepcopy(mapped),
                'targetColumns': target_columns, 'targetMapping': target_identity,
                'sourceDigest': table.source_digest, 'partitionDigest': table.partition_digest,
                'dataVersion': table.data_version, 'membershipDigest': df.membership_digest(table.row_ids),
                'observationIDs': list(table.row_ids), 'fittedStateDigests': list(table.fitted_states),
                'role': table.role, 'lineage': copy.deepcopy(item['lineage'])}
    identity['mappingDigest'] = df.digest({'source': config, 'mapping': mapped, 'target': target_identity})
    return packet, targets, identity


def _name(value):
    return type(value) is str and bool(value.strip()) and len(value) <= 200


def _source_configuration(value):
    keys = {'layout', 'maximum_items_per_bag', 'per_sample_weights', 'include_last_offset'}
    if (type(value) is not dict or set(value) != keys or type(value['layout']) is not str
            or value['layout'] not in ('fixed', 'packed')
            or type(value['maximum_items_per_bag']) is not int or not 1 <= value['maximum_items_per_bag'] <= 128
            or type(value['per_sample_weights']) is not bool or type(value['include_last_offset']) is not bool):
        raise ValueError('Bag source requires its exact layout, 1..128 row capacity and boolean weight/terminal controls.')
    return copy.deepcopy(value)


def _mapping(value, config):
    keys = {'schemaVersion', 'indicesColumns'}
    if config['per_sample_weights']: keys.add('weightColumns')
    if config['layout'] == 'packed': keys.add('lengthColumn')
    if (type(value) is not dict or set(value) != keys
            or type(value['schemaVersion']) is not int or value['schemaVersion'] != 1):
        raise ValueError('Bag mapping must match the exact version1 source-layout and weight controls.')
    for key in ('indicesColumns', 'weightColumns'):
        if key not in keys: continue
        columns = value[key]
        if (type(columns) is not list or len(columns) != config['maximum_items_per_bag']
                or any(not _name(name) for name in columns) or len(set(columns)) != len(columns)):
            raise ValueError('Map distinct ordered item and weight column names matching the configured row capacity.')
    if 'lengthColumn' in keys and not _name(value['lengthColumn']):
        raise ValueError('Packed bags require one nonblank length column name.')
    return copy.deepcopy(value)


def _float32(value):
    try: finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError: finite = False
    if not finite:
        raise ValueError('Mapped bag importance must be a finite real number.')
    try: rounded = struct.unpack('f', struct.pack('f', value))[0]
    except (OverflowError, struct.error):
        raise ValueError('Mapped bag importance overflows float32; rescale it before mapping.') from None
    if not math.isfinite(rounded) or value != 0 and rounded == 0:
        raise ValueError('Mapped bag importance underflows or overflows float32; rescale it before mapping.')
    return rounded
