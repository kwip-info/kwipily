"""Verified prepared tables → named numeric/token tensors; no mutable source reads.

Conversion buffers are batch-bounded. This first admission retains a separately
bounded tensor population for the existing exact-continuation training engine.
"""
import json
import math
import torch
if __package__:
    from . import dataflow as df
    from .dataflow_storage import completed_inventory,admit_model_ports,prepared_table_document
    from .dataflow_cache import table_from_artifact
    from .named_inputs import validate_plan,convert_rows,MAX_BATCH_ROWS
else:
    import dataflow as df
    from dataflow_storage import completed_inventory,admit_model_ports,prepared_table_document
    from dataflow_cache import table_from_artifact
    from named_inputs import validate_plan,convert_rows,MAX_BATCH_ROWS

MAX_POPULATION_BYTES=134217728


def table_tensors(table,plan,check=lambda:None,media_reader=None):
    validate_plan(plan)
    count=len(table.rows)
    if not 1<=count<=100000:raise ValueError('Named tensor admission currently supports 1–100000 prepared examples; general streaming population admission remains a later slice.')
    per_row=sum(math.prod(e['sampleShape'])*(8 if e['dtype']=='int64' else 4) for e in plan['inputs']+plan['roles'])
    if count*per_row>MAX_POPULATION_BYTES:raise ValueError('Prepared named tensors exceed the 128 MiB retained-population allowance; reduce examples or sample shape.')
    batch=min(MAX_BATCH_ROWS,plan['maximumBatchBytes']//max(1,per_row))
    if batch<1:raise ValueError('One prepared example exceeds the named batch byte allowance.')
    populations=[{e['name']:[] for e in plan[g]} for g in ('inputs','roles')]
    for begin in range(0,count,batch):
        check();groups=convert_rows(plan,list(table.columns),list(table.rows[begin:min(begin+batch,count)]),media_reader=media_reader)
        for output,group in zip(populations,groups):
            for name,value in group.items():output[name].append(value)
    return tuple({k:torch.cat(v,dim=0) for k,v in group.items()} for group in populations)


def load_named_prepared(path,training_node,training_port,validation_node,validation_port,plan,graph,*,expected_manifest_sha256,expected_graph_digest,objective_recipe=None,target_mapping=None,registry=None,check=lambda:None):
    validate_plan(plan)
    declared=graph.get('namedInputs')
    if declared is not None:
        if declared!={e['name']:e['nodeID'] for e in plan['inputs']}:raise ValueError('Prepared names differ from model-owned input declarations.')
    elif len(plan['inputs'])!=1:raise ValueError('Multiple inputs require a model with explicit named boundaries.')
    try:
        from .engine import validate
    except ImportError:
        from engine import validate
    context={'recipe':objective_recipe,'targetMapping':target_mapping} if objective_recipe else None
    _,_,layouts,dtypes=validate(graph,registry,bundled_dataset=False,include_types=True,objective_context=context)
    for entry in plan['inputs']:
        node=entry['nodeID']
        if node not in layouts or list(layouts[node].sample)!=entry['sampleShape'] or dtypes[node]!=entry['dtype'] or next(n['type'] for n in graph['nodes'] if n['id']==node) not in ('input','index_input'):raise ValueError('Mapped tensor shape/type does not match its actual model boundary.')
    if plan['schemaVersion']==2:
        output=next(n['id'] for n in graph['nodes'] if n['type']=='output')
        source=next(e for e in plan['inputs'] if e['name']==plan['sequenceObjective']['inputName'])
        if list(layouts[output].sample)!=[source['sampleShape'][0],source['vocabularySize']] or dtypes[output]!='float32':raise ValueError('Sequence model logits must match [context, vocabulary] exactly.')
    manifest,files,sha=completed_inventory(path,expected_manifest_sha256,lazy=True,selected_nodes={training_node,validation_node})
    if manifest['graphDigest']!=expected_graph_digest:raise ValueError('Prepared graph identity differs from the selected dataflow.')
    selected={(training_node,training_port),(validation_node,validation_port)};admit_model_ports(files,manifest,selected)
    values={};states={}
    for entry in manifest['artifacts']:
        key=entry.get('nodeID'),entry.get('port')
        if entry['kind']=='table' and key in selected:
            if key in values:raise ValueError('Duplicate named prepared table authority.')
            values[key]=table_from_artifact(prepared_table_document(files,entry))
        elif entry['kind']=='state':
            state=df.load_state(files[entry['path']].decode());states[state['contentDigest']]=state
    training=values.get((training_node,training_port));validation=values.get((validation_node,validation_port))
    if training is None or validation is None or training.role!='training' or validation.role!='validation' or not training.partition_digest or training.columns!=validation.columns or training.source_digest!=validation.source_digest or training.partition_digest!=validation.partition_digest or training.fitted_states!=validation.fitted_states:raise ValueError('Named training/validation require matching schema, source, partition and fitted state.')
    training_ids=set(training.row_ids)
    if training_ids&set(validation.row_ids):raise ValueError('Named prepared partitions overlap.')
    if plan['schemaVersion']==2:
        if objective_recipe is None or target_mapping is None:raise ValueError('Sequence training requires an explicit cross-entropy objective.')
        if __package__:from .sequence_objective import verify_sequence_preparation
        else:from sequence_objective import verify_sequence_preparation
        training_ids=verify_sequence_preparation(manifest,files,(training_node,training_port),(validation_node,validation_port),plan,{'training':training,'validation':validation},objective_recipe,target_mapping,states,check)
    for pin in training.fitted_states:
        state=states.get(pin)
        if state is None or state['graphDigest']!=expected_graph_digest or state['partitionDigest']!=training.partition_digest or state['fitEvidence']['sourceDigest']!=training.source_digest or not set(df.fit_observation_ids(state))<=training_ids:raise ValueError('Named model preparation state is absent, mismatched or fitted outside Training.')
    if __package__:from .media_artifacts import validate_reference,filename
    else:from media_artifacts import validate_reference,filename
    media_entries={e['sha256']:e for e in manifest['artifacts'] if e['kind']=='preparedTensor'}
    def media_reader(reference):
        check();reference=validate_reference(reference);entry=media_entries.get(reference['sha256'])
        if entry is None or entry.get('media')!=reference or entry['path']!=filename(reference):raise ValueError('Named input media is outside its admitted run inventory.')
        return files[entry['path']]
    # Child windows remain a single source entity even if an imported graph bypassed
    # the native lineage guard. Distinct window IDs alone do not prove isolation.
    def media_parents(table):
        indices=[i for i,c in enumerate(table.columns) if c['dtype']=='tensor']
        return {row[i]['metadata']['parentSHA256'] for row in table.rows for i in indices if row[i] is not None and 'sourceStart' in row[i]['metadata'] and 'parentSHA256' in row[i]['metadata']}
    if media_parents(training)&media_parents(validation):raise ValueError('Audio windows from one recording overlap Training and Validation; partition source files before windowing.')
    tx,tr=table_tensors(training,plan,check,media_reader);vx,vr=table_tensors(validation,plan,check,media_reader)
    if declared is None:tx=next(iter(tx.values()));vx=next(iter(vx.values()))
    if objective_recipe is None:
        if set(tr)!={'target'} or tr['target'].dtype!=torch.int64 or tr['target'].ndim!=1 or any(set(r['target'].tolist())!={0,1} for r in (tr,vr)):raise ValueError('Default training requires both binary target classes in each partition; configure an explicit objective for another task.')
        tr=tr['target'];vr=vr['target']
    else:
        if {e['name'] for e in plan['roles']}!={r['role'] for r in target_mapping['roles']}:raise ValueError('Named target roles differ from the configured objective mapping.')
        for entry in plan['roles']:
            role=next(r for r in target_mapping['roles'] if r['role']==entry['name'])
            if role['dtype']!=entry['dtype'] or role['sampleShape']!=entry['sampleShape'] or role['columns']!=entry['columns']:raise ValueError('Named target mapping differs from the captured objective role.')
    provenance={'dataset':'named-prepared-v1','sourceHash':training.source_digest,'samples':len(training.rows)+len(validation.rows),'sourceSamples':None,'trainingSamples':len(training.rows),'validationSamples':len(validation.rows),'trainingRowIDs':list(training.row_ids),'validationRowIDs':list(validation.row_ids),'splitHash':df.digest({'training':df.membership_digest(training.row_ids),'validation':df.membership_digest(validation.row_ids),'partition':training.partition_digest}),'split':'prepared partitions; no resplitting','featureColumns':[c for e in plan['inputs'] for c in e['columns']],'namedInputPlan':plan,'modelInput':{'kind':'namedInputs' if declared else 'tensor','schemaVersion':1,'inputs':plan['inputs']},'preparation':{'manifestSHA256':sha,'graphDigest':expected_graph_digest,'registryDigest':manifest.get('registryDigest'),'trainingNode':training_node,'trainingPort':training_port,'validationNode':validation_node,'validationPort':validation_port,'trainingDataVersion':training.data_version,'validationDataVersion':validation.data_version,'fittedStateDigests':list(training.fitted_states),'artifactDirectory':'preparation'}}
    if objective_recipe:provenance.update(objectiveRecipe=objective_recipe,objectiveMapping=target_mapping)
    else:provenance['targetColumn']=plan['roles'][0]['columns'][0]
    return (tx,tr,vx,vr),provenance,files
