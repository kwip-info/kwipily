"""Strict CLI transport and graph-derived prepared bag admission.

This validates representation only. Completed source inventory and independent
graph authority remain mandatory at the caller before loading any table.
"""
import copy
import json

if __package__:
    from .bag_source import normalize_bag_source, BagCoordinateLayout
    from .embedding_bag_configuration import normalize_embedding_bag, configured_input_ports
    from .graph_semantics import effective_config
else:
    from bag_source import normalize_bag_source, BagCoordinateLayout
    from embedding_bag_configuration import normalize_embedding_bag, configured_input_ports
    from graph_semantics import effective_config


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError('Bag mapping JSON contains duplicate object keys.')
        result[key] = value
    return result


def validate_mapping(value, source_config, *, target_column=None, objective_mapping=None):
    config = normalize_bag_source(source_config)
    keys = {'schemaVersion', 'indicesColumns'}
    if config['per_sample_weights']: keys.add('weightColumns')
    if config['layout'] == 'packed': keys.add('lengthColumn')
    if (type(value) is not dict or set(value) != keys
            or type(value['schemaVersion']) is not int or value['schemaVersion'] != 1):
        raise ValueError('Bag mapping requires the exact version1 fields for the actual source controls.')
    def name(item): return type(item) is str and 0 < len(item) <= 200 and bool(item.strip())
    columns = []
    for key in ('indicesColumns', 'weightColumns'):
        if key not in keys: continue
        items = value[key]
        if type(items) is not list or len(items) != config['maximum_items_per_bag'] or any(not name(item) for item in items):
            raise ValueError('Bag mapping requires ordered column names matching the actual source capacity.')
        columns.extend(items)
    if 'lengthColumn' in keys:
        if not name(value['lengthColumn']): raise ValueError('Packed bag mapping requires a length column name.')
        columns.append(value['lengthColumn'])
    if len(set(columns)) != len(columns): raise ValueError('Bag source columns must be distinct across their roles.')
    targets = []
    if objective_mapping is not None:
        if target_column is not None: raise ValueError('Bag mapping cannot combine binary and objective target controls.')
        if objective_mapping.get('inputColumns') != value['indicesColumns']:
            raise ValueError('Objective input columns must match the ordered bag indices columns.')
        targets = [name for role in objective_mapping['roles'] for name in role['columns']]
    elif target_column is not None:
        if not name(target_column): raise ValueError('Bag mapping requires a nonblank binary target column.')
        targets = [target_column]
    if set(columns).intersection(targets): raise ValueError('Bag source columns and target columns must remain separate.')
    return copy.deepcopy(value)


def decode_mapping_json(raw, source_config, *, target_column=None, objective_mapping=None):
    if type(raw) is not str or len(raw.encode('utf-8')) > 32768:
        raise ValueError('Bag mapping exceeds its32768 UTF-8 byte transport limit.')
    value = json.loads(raw, object_pairs_hook=_unique_object,
                       parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Bag mapping requires finite JSON.')))
    if raw != json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False):
        raise ValueError('Bag mapping requires canonical sorted compact UTF-8 JSON.')
    return validate_mapping(value, source_config, target_column=target_column, objective_mapping=objective_mapping)


def graph_admission(graph, layouts):
    """Derive conservative packet bounds from every checked real consumer.

    Caller must first run the authoritative graph validator. Bag coordinates
    retain a source ID and role, never a misleading observation-first shape.
    """
    sources = [node for node in graph['nodes'] if node['type'] in ('input', 'index_input', 'bag_input')]
    if len(sources) != 1 or sources[0]['type'] != 'bag_input':
        raise ValueError('Bag mapping requires the graph actual single bag input source.')
    source = sources[0]
    config = normalize_bag_source(effective_config(source, graph['schemaVersion']))
    consumers = []
    for node in graph['nodes']:
        if node['type'] != 'embedding_bag': continue
        settings = normalize_embedding_bag(effective_config(node, graph['schemaVersion']))
        for role in configured_input_ports(settings):
            matches = [edge for edge in graph['edges'] if edge['target'] == node['id'] and edge['targetPort'] == role]
            if len(matches) != 1: raise ValueError('Bag consumer requires each configured coordinate role exactly once.')
            edge = matches[0]
            value = layouts[edge['source']][edge['sourcePort']]
            if not isinstance(value, BagCoordinateLayout) or value.source_id != source['id'] or value.role != role:
                raise ValueError('Bag consumer coordinates must originate from the actual bag source.')
        consumers.append(settings)
    if not consumers: raise ValueError('Bag input requires at least one real EmbeddingBag consumer.')
    return dict(source_id=source['id'], source_config=config,
                num_embeddings=min(item['num_embeddings'] for item in consumers),
                embedding_dim=max(item['embedding_dim'] for item in consumers), mode='sum')
