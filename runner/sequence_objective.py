"""Sequence training admission from the exact retained preparation closure.

Original token populations remain required evidence, independently of optional
component capture. No mutable sources or protected Test population are opened.
"""
import json
if __package__:
    from . import dataflow as df
    from .dataflow_cache import table_from_artifact
    from .sequence_windows import verify_output
    from .loss_target_mapping import validate_target_mapping
    from .portable_text_encoding import special_tokens
else:
    import dataflow as df
    from dataflow_cache import table_from_artifact
    from sequence_windows import verify_output
    from loss_target_mapping import validate_target_mapping
    from portable_text_encoding import special_tokens


def verify_sequence_preparation(manifest,files,training_endpoint,validation_endpoint,plan,outputs,recipe,mapping,states,check):
    validate_target_mapping(mapping,recipe)
    if mapping['schemaVersion']!=2:raise ValueError('Sequence objective requires its version 2 target mapping.')
    seq=plan['sequenceObjective'];x=next(e for e in plan['inputs'] if e['name']==seq['inputName']);y=plan['roles'][0]
    role=mapping['roles'][0]
    if mapping['inputColumns']!=list(dict.fromkeys(c for e in plan['inputs'] for c in e['columns'])) or role['columns']!=y['columns'] or role['sampleShape']!=y['sampleShape']:raise ValueError('Sequence target mapping differs from the named contract.')
    graph=json.loads(files['dataflow.json'])
    if df.digest(graph)!=manifest['graphDigest']:raise ValueError('Sequence preparation graph differs.')
    if training_endpoint[0]!=validation_endpoint[0] or training_endpoint[1]!='training' or validation_endpoint[1]!='validation':raise ValueError('Connect matching Training and Validation directly from one Sequence windows component.')
    node=next((n for n in graph['nodes'] if n['id']==training_endpoint[0]),None)
    if node is None or node['operator']!='kwip.data.sequenceWindows' or node['config']['mode']!='partitions':raise ValueError('Sequence training needs captured partition-aware windows.')
    c=node['config']
    if x['columns']!=[c['inputColumn']] or y['columns']!=[c['targetColumn']] or x['sampleShape']!=[c['contextLength']] or seq['lengthColumn']!=c['lengthColumn'] or seq['maskColumn']!=(c['maskColumn'] if c['incomplete']=='pad' else '') or x['paddingValue']!=c['paddingID']:raise ValueError('Sequence input/target/length/mask mapping differs from its captured window configuration.')
    original={}
    for part in ('training','validation'):
        edges=[e for e in graph['edges'] if e['to']==dict(node=node['id'],port=part)]
        if len(edges)!=1:raise ValueError('Sequence input has ambiguous parent authority.')
        endpoint=edges[0]['from'];entry=next((e for e in manifest['artifacts'] if e.get('kind')=='table' and (e.get('nodeID'),e.get('port'))==(endpoint['node'],endpoint['port'])),None)
        if entry is None:raise ValueError('Original token populations are required replay evidence; prepare this sequence workflow again.')
        check();original[part]=table_from_artifact(json.loads(files[entry['path']]))
    verify_output(original,c,manifest['graphDigest'],outputs,check)
    training_ids=set(original['training'].row_ids)
    for part,table in original.items():
        for pin in table.fitted_states:
            state=states.get(pin)
            if state is None or state['graphDigest']!=manifest['graphDigest'] or state['partitionDigest']!=table.partition_digest or state['fitEvidence']['sourceDigest']!=table.source_digest or not set(df.fit_observation_ids(state))<=training_ids:raise ValueError('Sequence vocabulary/preparation was not fitted on its original Training population.')
            if state.get('operator')=='kwip.data.textEncode' and state['configuration']['outputColumn']==c['tokenColumn']:
                specials=special_tokens(state['configuration'])
                if len(state['vocabulary'])!=x['vocabularySize'] or c['incomplete']=='pad' and ('<pad>' not in specials or specials.index('<pad>')!=x['paddingValue']):raise ValueError('Sequence vocabulary size or reserved padding ID differs from its captured tokenizer.')
    return training_ids
