"""Pinned typed target partitions for explicit objectives, without binary coercion."""
import copy
import json
import math
import re
import struct
import torch
if __package__:
    from . import dataflow
    from .dataflow_cache import table_from_artifact
    from .dataflow_storage import completed_inventory, admit_model_ports, prepared_table_document
    from .prepared_data import _mapped_shape
    from .loss_configuration import validate_recipe, resolve_objective, MAX_ELEMENTS
    from .loss_target_mapping import validate_target_mapping
    from .tensor_layout import BatchedLayout
else:
    import dataflow
    from dataflow_cache import table_from_artifact
    from dataflow_storage import completed_inventory, admit_model_ports, prepared_table_document
    from prepared_data import _mapped_shape
    from loss_configuration import validate_recipe, resolve_objective, MAX_ELEMENTS
    from loss_target_mapping import validate_target_mapping
    from tensor_layout import BatchedLayout


def _float32(value, *, feature=False):
    if type(value) not in (int, float) or not math.isfinite(value) or feature and abs(value) > 1e10:
        raise ValueError('Mapped values must be finite numbers; model features are bounded to ±1e10.')
    try: rounded = struct.unpack('f', struct.pack('f', value))[0]
    except (OverflowError, struct.error):
        raise ValueError('Mapped value overflows float32; rescale it before training.') from None
    if not math.isfinite(rounded) or value != 0 and rounded == 0:
        raise ValueError('Mapped value underflows or overflows float32; rescale it before training.')
    return rounded


def table_objective_tensors(item, role, mapping, recipe, input_dtype, sample_shape, prediction_shape):
    recipe = validate_recipe(recipe); mapping = validate_target_mapping(mapping, recipe)
    shape = _mapped_shape(mapping['inputColumns'], input_dtype, sample_shape)
    descriptors = {r['role']: {'dtype': r['dtype'], 'sampleShape': r['sampleShape']} for r in mapping['roles']}
    resolve_objective(recipe['componentID'], recipe['config'], BatchedLayout(prediction_shape), descriptors)
    if type(item) is not dict or type(item.get('lineage')) is not dict:
        raise ValueError('Select a typed prepared table artifact.')
    table = table_from_artifact(item)
    if table.role != role or role not in ('training', 'validation', 'test'):
        raise ValueError('Prepared objective partition has the wrong role.')
    if any(re.fullmatch('[0-9a-f]{64}', v) is None for v in (table.source_digest, table.partition_digest)):
        raise ValueError('Prepared source and partition require exact digest identities.')
    columns = item['columns']; names = [c['name'] for c in columns]
    n = len(table.rows)
    mapped_count = len(mapping['inputColumns']) + sum(len(r['columns']) for r in mapping['roles'])
    if mapped_count > MAX_ELEMENTS // n:
        raise ValueError('Mapped objective tensors exceed the 100M element budget.')
    def tensor(selected, dtype, sample, feature=False):
        if any(name not in names for name in selected):
            raise ValueError('Mapped input or target column is absent from the prepared table.')
        indices = [names.index(name) for name in selected]
        if any(columns[i]['dtype']=='image' for i in indices):raise ValueError('Image columns require an explicit image-to-model adapter before objective mapping.')
        allowed = ('int64',) if dtype == 'int64' else ('int64', 'float64')
        if any(columns[i]['dtype'] not in allowed for i in indices):
            raise ValueError('Mapped integer values require explicit int64 columns; floating values require numeric columns.')
        rows = []
        for row in table.rows:
            values = [row[i] for i in indices]
            if dtype == 'int64':
                if any(type(v) is not int or not -(2**63) <= v < 2**63 for v in values):
                    raise ValueError('Mapped integer roles require exact int64 values.')
            else:
                values = [_float32(v, feature=feature) for v in values]
            rows.append(values)
        return torch.tensor(rows, dtype=getattr(torch, dtype)).reshape(n, *sample)
    x = tensor(mapping['inputColumns'], input_dtype, shape, True)
    roles = {r['role']: tensor(r['columns'], r['dtype'], r['sampleShape']) for r in mapping['roles']}
    # CTC retains observation-owned padded rows here. Packing occurs only for the actual batch.
    return x, roles, copy.deepcopy(item['lineage']), copy.deepcopy(columns)


def load_prepared_objective(manifest_path, training_node, training_port, validation_node, validation_port,
                            mapping, recipe, *, expected_manifest_sha256=None, expected_graph_digest=None,
                            input_dtype='float32', sample_shape=None, prediction_shape=None):
    recipe = validate_recipe(recipe); mapping = validate_target_mapping(mapping, recipe)
    for pin in (expected_manifest_sha256, expected_graph_digest):
        if pin is not None and (type(pin) is not str or re.fullmatch('[0-9a-f]{64}', pin) is None):
            raise ValueError('Prepared identity pin must be a lowercase SHA-256 digest.')
    manifest, files, manifest_sha = completed_inventory(manifest_path, expected_manifest_sha256,lazy=True,selected_nodes={training_node,validation_node})
    if expected_graph_digest is not None and manifest['graphDigest'] != expected_graph_digest:
        raise ValueError('Prepared objective data differs from the selected graph; prepare again.')
    selected_ports={(training_node,training_port),(validation_node,validation_port)}
    admit_model_ports(files,manifest,selected_ports)
    values = {}; states = {}
    for entry in manifest['artifacts']:
        if entry.get('kind') not in ('table', 'state'): continue
        key = (entry.get('nodeID'), entry.get('port'))
        if entry['kind']=='table' and key not in selected_ports:continue
        if key in values: raise ValueError('Duplicate prepared node/port.')
        value = prepared_table_document(files,entry) if entry['kind']=='table' else json.loads(files[entry['path']]); values[key] = value
        if entry['kind'] == 'state':
            state = dataflow.load_state(files[entry['path']].decode())
            if state['graphDigest'] != manifest['graphDigest']:
                raise ValueError('Fitted state graph differs from preparation.')
            states[state['contentDigest']] = state
    graph = json.loads(files.get('dataflow.json', b'null'))
    if dataflow.digest(graph) != manifest['graphDigest']: raise ValueError('Preparation graph digest mismatch.')
    dataflow.validate(graph)
    def selected(node, port, role):
        return table_objective_tensors(values.get((node, port)), role, mapping, recipe,
                                       input_dtype, sample_shape, prediction_shape)
    tx, tr, train, ts = selected(training_node, training_port, 'training')
    vx, vr, val, vs = selected(validation_node, validation_port, 'validation')
    if ts != vs or train['sourceDigest'] != val['sourceDigest'] or train['partitionDigest'] != val['partitionDigest']:
        raise ValueError('Prepared partitions must share schema, source and split identity.')
    training_ids = set(train['observationIDs'])
    if training_ids.intersection(val['observationIDs']): raise ValueError('Prepared partitions overlap; heldout leakage.')
    fitted = train.get('fittedStateDigests', [])
    if fitted != val.get('fittedStateDigests', []):
        raise ValueError('Prepared partitions must use identical fitted transformations.')
    for state_id in fitted:
        state = states.get(state_id)
        if (state is None or state['partitionDigest'] != train['partitionDigest']
                or state['fitEvidence']['sourceDigest'] != train['sourceDigest']
                or not set(dataflow.fit_observation_ids(state)) <= training_ids):
            raise ValueError('Prepared fitted state is missing or includes nontraining observations.')
    if input_dtype == 'float32' and set(map(tuple, tx.reshape(len(tx), -1).tolist())).intersection(map(tuple, vx.reshape(len(vx), -1).tolist())):
        raise ValueError('Selected features overlap across prepared partitions at float32 precision.')
    provenance = {
        'dataset': 'prepared-objective-dataflow-v1', 'sourceHash': train['sourceDigest'],
        'samples': len(tx) + len(vx), 'sourceSamples': None, 'trainingSamples': len(tx), 'validationSamples': len(vx),
        'trainingRowIDs': train['observationIDs'], 'validationRowIDs': val['observationIDs'],
        'splitHash': dataflow.digest({'training': train['membershipDigest'], 'validation': val['membershipDigest'], 'partition': train['partitionDigest']}),
        'split': 'prepared partitions; no resplitting', 'featureColumns': mapping['inputColumns'],
        'objectiveMapping': mapping, 'objectiveRecipe': recipe,
        'modelInput': {'dtype': input_dtype, 'sampleShape': list(tx.shape[1:]), 'columnOrder': mapping['inputColumns']},
        'preparation': {'manifestSHA256': manifest_sha, 'graphDigest': manifest['graphDigest'],
            'registryDigest': manifest.get('registryDigest'), 'trainingNode': training_node, 'trainingPort': training_port,
            'validationNode': validation_node, 'validationPort': validation_port,
            'trainingDataVersion': train['dataVersion'], 'validationDataVersion': val['dataVersion'],
            'fittedStateDigests': fitted, 'artifactDirectory': 'preparation'}}
    return (tx, tr, vx, vr), provenance, files
