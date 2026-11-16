"""Completed, pinned paired-partition admission for prepared bag inputs."""
import copy
import json
import re

if __package__:
    from . import dataflow
    from .dataflow_cache import table_from_artifact
    from .dataflow_storage import completed_inventory, admit_model_ports, prepared_table_document
    from .prepared_bag_data import prepare_bag_table
else:
    import dataflow
    from dataflow_cache import table_from_artifact
    from dataflow_storage import completed_inventory, admit_model_ports, prepared_table_document
    from prepared_bag_data import prepare_bag_table


def load_prepared_bags(manifest_path, training_node, training_port, validation_node, validation_port,
                       *, source_config, mapping, num_embeddings, embedding_dim, mode,
                       expected_manifest_sha256, expected_graph_digest, target_column=None,
                       objective_mapping=None, objective_recipe=None, prediction_shape=None):
    """Admit captured completed preparation, without rerunning/resplitting it.

    Inventory pins authenticate captured files. This does not grant authority to
    scan other directories, run archived programs or execute an unrelated source.
    """
    for pin in (expected_manifest_sha256, expected_graph_digest):
        if type(pin) is not str or re.fullmatch('[0-9a-f]{64}', pin) is None:
            raise ValueError('Prepared bag identity requires exact lowercase SHA-256 manifest and graph pins.')
    for value in (training_node, training_port, validation_node, validation_port):
        if type(value) is not str or not 1 <= len(value) <= 128:
            raise ValueError('Select exact bounded prepared node and port names.')
    manifest, files, manifest_sha = completed_inventory(manifest_path, expected_manifest_sha256,lazy=True,selected_nodes={training_node,validation_node})
    if manifest.get('graphDigest') != expected_graph_digest:
        raise ValueError('Prepared bag data differs from the selected graph; prepare again.')
    graph = json.loads(files.get('dataflow.json', b'null'))
    if dataflow.digest(graph) != expected_graph_digest:
        raise ValueError('Preparation source graph digest mismatch.')
    nodes, _, _ = dataflow.validate(graph)
    selected_ports={(training_node,training_port),(validation_node,validation_port)}
    admit_model_ports(files,manifest,selected_ports)
    values = {}
    states = {}
    for entry in manifest['artifacts']:
        kind = entry.get('kind')
        if kind not in ('table', 'state'): continue
        node, port = entry.get('nodeID'), entry.get('port')
        if (type(node) is not str or type(port) is not str or node not in nodes
                or port not in dataflow.OPS[nodes[node][0]][1]
                or (kind == 'state') != (port == 'state')):
            raise ValueError('Prepared artifact does not name a declared source-graph output of its kind.')
        key = (node, port)
        if key in values:
            raise ValueError('Duplicate prepared node/port.')
        if kind=='table' and key not in selected_ports:continue
        raw = files[entry['path']]
        value = prepared_table_document(files,entry) if kind=='table' else json.loads(raw)
        values[key] = value
        if kind == 'state':
            state = dataflow.load_state(raw.decode())
            if state['graphDigest'] != expected_graph_digest:
                raise ValueError('Fitted state graph differs from preparation.')
            previous = states.get(state['contentDigest'])
            if previous is not None and previous != state:
                raise ValueError('Conflicting fitted state content identity.')
            states[state['contentDigest']] = state
    selected = [(training_node, training_port, 'training'),
                (validation_node, validation_port, 'validation')]
    tables = []
    for node, port, role in selected:
        if node not in nodes or port not in dataflow.OPS[nodes[node][0]][1] or port == 'state':
            raise ValueError('Select a declared table output from the captured source graph.')
        value = values.get((node, port))
        table = table_from_artifact(value)
        if table.role != role:
            raise ValueError('Selected prepared table has the wrong partition role.')
        tables.append(table)
    train, val = tables
    if (train.columns != val.columns or train.source_digest != val.source_digest
            or train.partition_digest != val.partition_digest):
        raise ValueError('Prepared partitions must share schema, source and split identity.')
    training_ids = set(train.row_ids)
    if training_ids.intersection(val.row_ids):
        raise ValueError('Prepared partitions overlap; heldout leakage.')
    fitted = list(train.fitted_states)
    if fitted != list(val.fitted_states):
        raise ValueError('Prepared partitions must use identical fitted transformations.')
    for state_id in fitted:
        state = states.get(state_id)
        if (state is None or state['partitionDigest'] != train.partition_digest
                or state['fitEvidence']['sourceDigest'] != train.source_digest
                or not set(dataflow.fit_observation_ids(state)) <= training_ids):
            raise ValueError('Prepared fitted state is missing or includes nontraining observations.')
    def convert(node, port, role):
        item = values[(node, port)]
        return prepare_bag_table(item, role=role, source_config=source_config, mapping=mapping,
                num_embeddings=num_embeddings, embedding_dim=embedding_dim, mode=mode,
                expected_table_digest=dataflow.digest(item), target_column=target_column,
                objective_mapping=objective_mapping, objective_recipe=objective_recipe,
                prediction_shape=prediction_shape)
    train_packet, train_targets, train_identity = convert(*selected[0])
    val_packet, val_targets, val_identity = convert(*selected[1])
    config = train_identity['sourceConfiguration']
    ordered_mapping = train_identity['mapping']
    provenance = {
        'dataset': 'prepared-dataflow-bags-v1', 'sourceHash': train.source_digest,
        'samples': len(train.row_ids) + len(val.row_ids), 'sourceSamples': None,
        'trainingSamples': len(train.row_ids), 'validationSamples': len(val.row_ids),
        'trainingRowIDs': list(train.row_ids), 'validationRowIDs': list(val.row_ids),
        'splitHash': dataflow.digest({'training': dataflow.membership_digest(train.row_ids),
                                     'validation': dataflow.membership_digest(val.row_ids),
                                     'partition': train.partition_digest}),
        'split': 'prepared partitions; no resplitting',
        'featureColumns': list(ordered_mapping['indicesColumns']),
        'bagSources': {'training': train_identity, 'validation': val_identity},
        'modelInput': {'schemaVersion': 1, 'kind': 'embeddingBagPacket',
            'layout': config['layout'], 'maximumItemsPerBag': config['maximum_items_per_bag'],
            'perSampleWeights': config['per_sample_weights'], 'includeLastOffset': config['include_last_offset'],
            'indicesDType': 'int64', 'offsetsDType': 'int64' if config['layout'] == 'packed' else None,
            'weightDType': 'float32' if config['per_sample_weights'] else None,
            'mapping': copy.deepcopy(ordered_mapping)},
        'preparation': {'manifestSHA256': manifest_sha, 'graphDigest': expected_graph_digest,
            'registryDigest': manifest.get('registryDigest'), 'trainingNode': training_node,
            'trainingPort': training_port, 'validationNode': validation_node, 'validationPort': validation_port,
            'trainingDataVersion': train.data_version, 'validationDataVersion': val.data_version,
            'fittedStateDigests': fitted, 'artifactDirectory': 'preparation'}}
    if objective_mapping is not None:
        target = train_identity['targetMapping']
        provenance['objectiveMapping'] = copy.deepcopy(target['objectiveMapping'])
        provenance['objectiveRecipe'] = copy.deepcopy(target['objectiveRecipe'])
    else:
        provenance['targetColumn'] = target_column
    return (train_packet, train_targets, val_packet, val_targets), provenance, files
