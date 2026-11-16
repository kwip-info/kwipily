"""Verify exact dataflow table artifacts before converting prepared partitions to tensors."""
import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path
import torch
if __package__:
    from .tensor_layout import BatchedLayout
    from . import dataflow
    from .dataflow_storage import completed_inventory, admit_model_ports, prepared_table_document
else:
    from tensor_layout import BatchedLayout
    import dataflow
    from dataflow_storage import completed_inventory, admit_model_ports, prepared_table_document


def bounded_read(path,limit):
    # Open without following the terminal symlink, then validate the actual descriptor.
    # O_NONBLOCK avoids hanging on a FIFO before its nonregular type can be rejected.
    try:descriptor=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    except OSError as error:raise ValueError('Prepared artifact must be a readable regular file') from error
    try:
        info=os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):raise ValueError('Prepared artifact must be a regular file')
        if info.st_size>limit:raise ValueError('Prepared artifact exceeds size bound')
        with os.fdopen(descriptor,'rb') as stream:
            descriptor=None
            value=stream.read(limit+1)
    finally:
        if descriptor is not None:os.close(descriptor)
    if len(value)>limit:raise ValueError('Prepared artifact exceeds size bound')
    return value


def _table_tensors(item,role,features,target,input_dtype,shape,*,require_both=True):
    if not isinstance(item,dict) or item.get('kind')!='table':raise ValueError('Selected preparation port must contain a table artifact')
    columns=item.get('columns');rows=item.get('rows');lineage=item.get('lineage')
    if __package__:from .image_artifacts import validate_table_version
    else:from image_artifacts import validate_table_version
    if not isinstance(columns,list):raise ValueError('Invalid prepared table schema')
    validate_table_version(item.get('schemaVersion'),columns)
    if not isinstance(columns,list) or not dataflow.names([c.get('name') for c in columns]) or not isinstance(rows,list) or not 1<=len(rows)<=100000 or not isinstance(lineage,dict):raise ValueError('Invalid prepared table bounds/schema')
    if any(not isinstance(r,list) or len(r)!=len(columns) for r in rows):raise ValueError('Prepared row width differs from schema')
    ids=lineage.get('observationIDs')
    if not isinstance(ids,list) or len(ids)!=len(rows) or any(not isinstance(x,str) or not 1<=len(x)<=200 for x in ids) or len(set(ids))!=len(ids):raise ValueError('Prepared row identities invalid or duplicate')
    if lineage.get('role')!=role or lineage.get('membershipDigest')!=dataflow.membership_digest(ids):raise ValueError('Prepared partition role or membership mismatch')
    if lineage.get('dataVersion')!=dataflow.digest({'columns':columns,'rows':rows,'ids':ids}):raise ValueError('Prepared table data version mismatch')
    if not lineage.get('partitionDigest') or not lineage.get('sourceDigest'):raise ValueError('Prepared partition lineage missing')
    names=[c['name'] for c in columns]
    if any(name not in names for name in [*features,target]):raise ValueError('Mapped feature/target absent from prepared table')
    indices=[names.index(name) for name in features];target_index=names.index(target)
    if any(columns[i].get('dtype')=='image' for i in [*indices,target_index]):raise ValueError('Image columns require an explicit image-to-model adapter before training; scalar conversion in Columns cannot convert pixels.')
    if any(columns[i].get('dtype') not in ('float64','int64') for i in indices) or columns[target_index].get('dtype')!='int64':raise ValueError('Prepared features require numeric types; binary target requires explicit int64 cast')
    if input_dtype=='int64' and any(columns[i].get('dtype')!='int64' for i in indices):raise ValueError('Integer model inputs require explicit int64 columns; choose int64 in Columns before mapping.')
    x=[];y=[]
    for row in rows:
        values_x=[row[i] for i in indices];label=row[target_index]
        if input_dtype=='int64':
            if any(type(v) is not int or not -(2**63)<=v<2**63 for v in values_x):raise ValueError('Prepared indices must be exact int64 values; missing, fractional and out-of-range values are invalid.')
        elif any(type(v) not in (int,float) or not math.isfinite(v) or abs(v)>1e10 for v in values_x):raise ValueError('Prepared features must be finite numeric values within ±1e10')
        if type(label) is not int or label not in (0,1):raise ValueError('Prepared target must be integer 0 or 1')
        x.append(values_x);y.append(label)
    if require_both and set(y)!={0,1}:raise ValueError('Both prepared partitions must retain both binary classes')
    return torch.tensor(x,dtype=getattr(torch,input_dtype)).reshape(len(x),*shape),torch.tensor(y,dtype=torch.long),lineage,columns


def _mapped_shape(features,input_dtype,sample_shape):
    if input_dtype not in ('float32','int64'):raise ValueError('Unsupported prepared model input dtype.')
    if not isinstance(features,list) or not dataflow.names(features):raise ValueError('Prepared feature columns must be distinct nonempty names.')
    if sample_shape is None:
        if input_dtype=='int64':raise ValueError('Integer source requires explicit sample shape.')
        if len(features)!=2:raise ValueError('Prepared default classifier requires exactly two different numeric features; supply the declared model sample shape.')
        shape=(2,)
    else:
        shape=BatchedLayout(sample_shape).sample
    if math.prod(shape)!=len(features):raise ValueError('Mapped columns must exactly fill the declared model sample shape.')
    return shape


def load_prepared(manifest_path,training_node,training_port,validation_node,validation_port,features,target,*,expected_manifest_sha256=None,expected_graph_digest=None,input_dtype="float32",sample_shape=None):
    shape=_mapped_shape(features,input_dtype,sample_shape)

    if target in features:raise ValueError('Target cannot also be a feature')
    for pin in (expected_manifest_sha256,expected_graph_digest):
        if pin is not None and (not isinstance(pin,str) or re.fullmatch('[0-9a-f]{64}',pin) is None):raise ValueError('Prepared identity pin must be a lowercase SHA-256 digest')
    manifest,files,actual_manifest_sha256=completed_inventory(manifest_path,expected_manifest_sha256,lazy=True,selected_nodes={training_node,validation_node})
    if expected_graph_digest is not None and manifest.get('graphDigest')!=expected_graph_digest:raise ValueError('Prepared data differs from the selected graph; select preparation again')
    selected_ports={(training_node,training_port),(validation_node,validation_port)}
    admit_model_ports(files,manifest,selected_ports)
    values={};states={}
    for entry in manifest['artifacts']:
        if entry.get('kind') not in ('table','state'):continue
        if entry.get('kind')=='table' and (entry.get('nodeID'),entry.get('port')) not in selected_ports:continue
        raw=files[entry['path']]
        if entry.get('kind') in ('table','state'):
            identity=(entry.get('nodeID'),entry.get('port'))
            if identity in values:raise ValueError('Duplicate prepared node/port')
            value=prepared_table_document(files,entry) if entry['kind']=='table' else json.loads(raw);values[identity]=value
            if entry['kind']=='state':
                state=dataflow.load_state(raw.decode())
                if state['graphDigest']!=manifest.get('graphDigest'):raise ValueError('Fitted state graph differs from preparation')
                states[state['contentDigest']]=state
    graph=json.loads(files.get('dataflow.json',b'null'))
    if dataflow.digest(graph)!=manifest.get('graphDigest'):raise ValueError('Preparation graph digest mismatch')
    dataflow.validate(graph)
    def table(node,port,role):
        return _table_tensors(values.get((node,port)),role,features,target,input_dtype,shape)
    train_x,train_y,train,train_schema=table(training_node,training_port,'training')
    val_x,val_y,val,val_schema=table(validation_node,validation_port,'validation')
    if train_schema!=val_schema or train['sourceDigest']!=val['sourceDigest'] or train['partitionDigest']!=val['partitionDigest']:raise ValueError('Prepared partitions must share schema, source and split identity')
    if set(train['observationIDs'])&set(val['observationIDs']):raise ValueError('Prepared partitions overlap; heldout leakage')
    fitted=train.get('fittedStateDigests',[])
    if fitted!=val.get('fittedStateDigests',[]):raise ValueError('Prepared partitions must use identical fitted transformations')
    for state_id in fitted:
        state=states.get(state_id)
        if state is None or state['partitionDigest']!=train['partitionDigest'] or state['fitEvidence']['sourceDigest']!=train['sourceDigest'] or not set(dataflow.fit_observation_ids(state))<=set(train['observationIDs']):raise ValueError('Prepared fitted state is missing or includes nontraining observations')
    if input_dtype=='float32' and set(map(tuple,train_x.reshape(len(train_x),-1).tolist()))&set(map(tuple,val_x.reshape(len(val_x),-1).tolist())):raise ValueError('Selected features overlap across prepared partitions at float32 precision')
    split_hash=dataflow.digest({'training':train['membershipDigest'],'validation':val['membershipDigest'],'partition':train['partitionDigest']})
    provenance={'dataset':'prepared-dataflow-v1','sourceHash':train['sourceDigest'],'samples':len(train_x)+len(val_x),'sourceSamples':None,'trainingSamples':len(train_x),'validationSamples':len(val_x),'trainingRowIDs':train['observationIDs'],'validationRowIDs':val['observationIDs'],'splitHash':split_hash,'split':'prepared partitions; no resplitting','featureColumns':features,'targetColumn':target,'preparation':{'manifestSHA256':actual_manifest_sha256,'graphDigest':manifest['graphDigest'],'registryDigest':manifest.get('registryDigest'),'trainingNode':training_node,'trainingPort':training_port,'validationNode':validation_node,'validationPort':validation_port,'trainingDataVersion':train['dataVersion'],'validationDataVersion':val['dataVersion'],'fittedStateDigests':fitted,'artifactDirectory':'preparation'}}
    if input_dtype=='int64':provenance['modelInput']={'dtype':input_dtype,'sampleShape':list(shape),'columnOrder':features}
    return (train_x,train_y,val_x,val_y),provenance,files


def _reload_training_population(manifest_path,preparation,provenance,expected_manifest_sha256,input_dtype,sample_shape,named_plan,model_graph):
    recipe=provenance.get('objectiveRecipe');mapping=provenance.get('objectiveMapping')
    if (recipe is None)!=(mapping is None):raise ValueError('Archived objective mapping is incomplete')
    args=(manifest_path,preparation['trainingNode'],preparation['trainingPort'],preparation['validationNode'],preparation['validationPort'])
    pins=dict(expected_manifest_sha256=expected_manifest_sha256,expected_graph_digest=preparation['graphDigest'])
    if named_plan is not None:
        if __package__:from .named_prepared_data import load_named_prepared
        else:from named_prepared_data import load_named_prepared
        return load_named_prepared(*args,named_plan,model_graph,**pins,objective_recipe=recipe,target_mapping=mapping)
    if recipe is not None:
        if __package__:
            from .loss_prepared_data import load_prepared_objective
            from .engine import validate
        else:
            from loss_prepared_data import load_prepared_objective
            from engine import validate
        _,_,layouts,_=validate(model_graph,bundled_dataset=False,include_types=True,objective_context=dict(recipe=recipe,targetMapping=mapping))
        output=next(n['id'] for n in model_graph['nodes'] if n['type']=='output')
        return load_prepared_objective(*args,mapping,recipe,**pins,input_dtype=input_dtype,sample_shape=sample_shape,prediction_shape=list(layouts[output].sample))
    return load_prepared(*args,provenance['featureColumns'],provenance['targetColumn'],**pins,input_dtype=input_dtype,sample_shape=sample_shape)


def load_test_partition(manifest_path,test_node,test_port,training_provenance,*,expected_manifest_sha256,input_dtype='float32',sample_shape=None,named_plan=None,model_graph=None):
    """Load an existing protected port; replay only its remaining frozen state chain.

    The caller supplies provenance pinned by the completed parent model archive.
    This verifies that provenance against preparation; it does not authenticate a
    caller-created replacement model archive or execute the preparation graph.
    """
    if not isinstance(training_provenance,dict):raise ValueError('Archived training provenance is required')
    preparation=training_provenance.get('preparation')
    if not isinstance(preparation,dict):raise ValueError('Archived preparation identity is required')
    if not isinstance(expected_manifest_sha256,str) or re.fullmatch('[0-9a-f]{64}',expected_manifest_sha256) is None:raise ValueError('Test preparation requires a SHA-256 pin')
    if preparation.get('manifestSHA256')!=expected_manifest_sha256:raise ValueError('Test preparation differs from archived training manifest')
    features=training_provenance.get('featureColumns');target=training_provenance.get('targetColumn')
    try:
        _,verified_training,files=_reload_training_population(manifest_path,preparation,training_provenance,expected_manifest_sha256,input_dtype,sample_shape,named_plan,model_graph)
    except (KeyError,TypeError) as error:raise ValueError('Incomplete archived training provenance') from error
    # Train appends these model/recipe fields after data preparation. The caller's
    # pinned model archive owns them; they are not a second data population.
    # Compare every data field exactly and reject unknown additions. In particular,
    # objectiveRecipe/objectiveMapping remain data fields and must still match.
    model_fields=('composite','objective','optimizer','scheduler','initialization','gradientPolicy','trainability')
    archived_data_provenance={key:value for key,value in training_provenance.items() if key not in model_fields}
    if verified_training!=archived_data_provenance:raise ValueError('Training provenance differs from verified archived partitions or input contract')
    # Reuse the bounded, pinned inventory verification and typed artifact decoder.
    if __package__:
        from .dataflow_cache import table_from_artifact
    else:
        from dataflow_cache import table_from_artifact
    manifest_bytes=bounded_read(manifest_path,2_000_000)
    # Bind decoding to the same exact bytes whose digest is checked.
    if hashlib.sha256(manifest_bytes).hexdigest()!=expected_manifest_sha256:raise ValueError('Preparation changed while selecting test partition')
    manifest=json.loads(manifest_bytes)
    selected_ports={(test_node,test_port),(preparation['trainingNode'],preparation['trainingPort'])}
    graph=json.loads(files['dataflow.json'])
    protected_nodes={node['id']:node for node in graph['nodes']}
    protected_edges={(edge['to']['node'],edge['to']['port']):(edge['from']['node'],edge['from']['port']) for edge in graph['edges']}
    endpoint=(test_node,test_port);visited=set()
    while endpoint[0] not in visited and endpoint[0] in protected_nodes:
        visited.add(endpoint[0])
        if protected_nodes[endpoint[0]]['operator'] in (dataflow.PREFIX+'groupSplit',dataflow.PREFIX+'temporalSplit',dataflow.PREFIX+'partition'):
            selected_ports.add(endpoint);break
        endpoint=protected_edges.get((endpoint[0],'test' if (protected_nodes[endpoint[0]]['operator']==dataflow.PREFIX+'preprocess' and protected_nodes[endpoint[0]]['config']['mode']=='partitions' or protected_nodes[endpoint[0]]['operator']==dataflow.PREFIX+'textEncode' and protected_nodes[endpoint[0]]['config']['vocabularySource']=='training') else 'table'),('', ''))
    admit_model_ports(files,manifest,selected_ports)
    values={};states={};state_ports={}
    for entry in manifest['artifacts']:
        if entry.get('kind') not in ('table','state'):continue
        key=(entry.get('nodeID'),entry.get('port'))
        if entry.get('kind')=='table' and key not in selected_ports:continue
        if entry['path'] not in files:raise ValueError('Preparation inventory changed')
        value=prepared_table_document(files,entry) if entry['kind']=='table' else json.loads(files[entry['path']]);values[key]=value
        if entry['kind']=='state':
            state=dataflow.load_state(json.dumps(value))
            states[state['contentDigest']]=state;state_ports[key]=state
    if not all(isinstance(value,str) and 1<=len(value)<=128 for value in (test_node,test_port)):raise ValueError('Invalid archived test port identity')
    selected=values.get((test_node,test_port))
    if selected is None:raise ValueError('Selected protected test port was not archived; select an existing raw test port')
    try:table=table_from_artifact(selected)
    except (KeyError,TypeError) as error:raise ValueError('Invalid protected test table') from error
    if table.role!='test':raise ValueError('Final evaluation requires a protected test role')
    train_table=table_from_artifact(values[(preparation['trainingNode'],preparation['trainingPort'])])
    if table.source_digest!=train_table.source_digest or table.partition_digest!=train_table.partition_digest:raise ValueError('Test source or partition differs from archived training')
    if set(table.row_ids)&set(training_provenance['trainingRowIDs']+training_provenance['validationRowIDs']):raise ValueError('Protected test observations overlap training or validation')
    fitted=preparation['fittedStateDigests'];prefix=list(table.fitted_states)
    if fitted[:len(prefix)]!=prefix:raise ValueError('Test fitted chain is not a prefix of archived training transformations')
    # A resealed raw table cannot pretend an existing fitted state was applied.
    # Follow only graph dependencies to establish state-chain identity; never run
    # operators or read sources. Actual frozen transformations below use artifacts.
    nodes={node['id']:node for node in graph['nodes']}
    incoming={(edge['to']['node'],edge['to']['port']):(edge['from']['node'],edge['from']['port']) for edge in graph['edges']}
    def protected_origin(node,port,visiting=()):
        if node in visiting or node not in nodes:raise ValueError('Invalid protected test graph dependency')
        operator=nodes[node]['operator'].removeprefix(dataflow.PREFIX)
        if operator in ('groupSplit','temporalSplit','partition'):
            if port!='test':raise ValueError('Selected table does not originate from a protected test output')
            origin=values.get((node,port))
            if origin is None:raise ValueError('Canonical protected test population was not archived')
            return table_from_artifact(origin)
        parent=incoming.get((node,'test' if (operator=='preprocess' and nodes[node]['config']['mode']=='partitions' or operator=='textEncode' and nodes[node]['config']['vocabularySource']=='training') else 'table'))
        if parent is None:raise ValueError('Selected table has no canonical protected test population')
        return protected_origin(*parent,visiting+(node,))
    origin=protected_origin(test_node,test_port)
    if (origin.role!='test' or origin.source_digest!=table.source_digest or
            origin.partition_digest!=table.partition_digest or set(origin.row_ids)!=set(table.row_ids)):
        raise ValueError('Selected test membership must retain the complete protected population; subsets are not final evaluation')
    def graph_chain(node,port,visiting=()):
        if node in visiting or node not in nodes:raise ValueError('Invalid archived test graph dependency')
        operator=nodes[node]['operator'].removeprefix(dataflow.PREFIX)
        if operator in ('sourceCSV','sourceFiles','capturedTable'):return []
        if operator in ('join','aggregate'):return []
        parent=incoming.get((node,'test' if (operator=='preprocess' and nodes[node]['config']['mode']=='partitions' or operator=='textEncode' and nodes[node]['config']['vocabularySource']=='training') else 'table'))
        if parent is None:raise ValueError('Test table graph lineage is unavailable')
        chain=graph_chain(*parent,visiting+(node,))
        if (operator=='preprocess' and nodes[node]['config']['mode']=='partitions' or operator=='textEncode' and nodes[node]['config']['vocabularySource']=='training'):
            state=state_ports.get((node,'state'))
            if state is None:raise ValueError('Archived Preprocess learned state is missing')
            chain=chain+[state['contentDigest']]
        if operator=='transform':
            state=state_ports.get(incoming.get((node,'state')))
            if state is None:raise ValueError('Test transform state is missing')
            chain=chain+[state['contentDigest']]
        return chain
    if graph_chain(test_node,test_port)!=prefix:raise ValueError('Selected test port fitted chain differs from its archived graph')
    selected_version=table.data_version;applied=fitted[len(prefix):]
    for state_id in applied:
        state=states.get(state_id)
        if state is None:raise ValueError('Archived training-fitted state is missing')
        table=dataflow.transform(table,state,manifest['graphDigest'])
    if list(table.columns)!=list(train_table.columns) or list(table.fitted_states)!=fitted:raise ValueError('Evaluated test schema or fitted chain differs from training')
    # This tensor adapter remains bounded before materializing streamed rows/IDs.
    if not 1<=len(table.rows)<=100000:raise ValueError('Invalid prepared table bounds/schema')
    value={'schemaVersion':1,'kind':'table','columns':list(table.columns),'rows':[list(row) for row in table.rows],
           'lineage':{**table.evidence(),'observationIDs':list(table.row_ids),'dataVersion':table.data_version,'partitionDigest':table.partition_digest,'fittedStateDigests':list(table.fitted_states)}}
    if named_plan is not None:
        if __package__:from .named_prepared_data import table_tensors
        else:from named_prepared_data import table_tensors
        if __package__:from .media_artifacts import inventory_reader
        else:from media_artifacts import inventory_reader
        x,roles=table_tensors(table,named_plan,media_reader=inventory_reader(manifest,files));y=roles if training_provenance.get('objectiveRecipe') else roles['target']
        if not model_graph.get('namedInputs'):x=next(iter(x.values()))
        if not isinstance(y,dict) and (y.dtype!=torch.int64 or y.ndim!=1 or not ((y==0)|(y==1)).all().item()):raise ValueError('Frozen classifier targets must be scalar binary int64 values.')
    elif training_provenance.get('objectiveRecipe'):
        if __package__:
            from .loss_prepared_data import table_objective_tensors
            from .engine import validate
        else:
            from loss_prepared_data import table_objective_tensors
            from engine import validate
        recipe=training_provenance['objectiveRecipe'];mapping=training_provenance['objectiveMapping']
        _,_,layouts,_=validate(model_graph,bundled_dataset=False,include_types=True,objective_context=dict(recipe=recipe,targetMapping=mapping))
        output=next(n['id'] for n in model_graph['nodes'] if n['type']=='output')
        x,y,_,_=table_objective_tensors(value,'test',mapping,recipe,input_dtype,sample_shape,list(layouts[output].sample))
    else:
        shape=_mapped_shape(features,input_dtype,sample_shape)
        x,y,_,_=_table_tensors(value,'test',features,target,input_dtype,shape,require_both=False)
    provenance={'dataset':'prepared-test-v1','sourceHash':table.source_digest,'testSamples':len(table.rows),
                'testRowIDs':list(table.row_ids),'featureColumns':list(features),**({'objectiveRecipe':training_provenance['objectiveRecipe'],'objectiveMapping':training_provenance['objectiveMapping']} if 'objectiveRecipe' in training_provenance else {'targetColumn':target}),
                'preparation':{'manifestSHA256':expected_manifest_sha256,'graphDigest':manifest['graphDigest'],
                    'registryDigest':manifest.get('registryDigest'),'testNode':test_node,'testPort':test_port,
                    'selectedDataVersion':selected_version,'evaluatedDataVersion':table.data_version,
                    'partitionDigest':table.partition_digest,'membershipDigest':dataflow.membership_digest(table.row_ids),
                    'fittedStateDigests':fitted,'appliedStateDigests':applied}}
    if input_dtype=='int64' or named_plan is not None:provenance['modelInput']=verified_training['modelInput']
    return (x,y),provenance,files


def load_evaluation_partition(manifest_path,node,port,training_provenance,*,partition_role='test',expected_manifest_sha256,input_dtype='float32',sample_shape=None,named_plan=None,model_graph=None):
    """Select only Test or the exact frozen parent's Validation mapping.

    Validation reuses the already verified heldout tensors. It never admits a
    protected Test port, executes graph operators, or applies new fitted states.
    The original protected-Test loader and provenance format remain unchanged.
    """
    if partition_role=='test':
        return load_test_partition(manifest_path,node,port,training_provenance,
            expected_manifest_sha256=expected_manifest_sha256,input_dtype=input_dtype,sample_shape=sample_shape,named_plan=named_plan,model_graph=model_graph)
    if partition_role!='validation':raise ValueError('Unsupported evaluation partition role')
    if not isinstance(training_provenance,dict):raise ValueError('Archived training provenance is required')
    preparation=training_provenance.get('preparation')
    if not isinstance(preparation,dict):raise ValueError('Archived preparation identity is required')
    if (node,port)!=(preparation.get('validationNode'),preparation.get('validationPort')):
        raise ValueError('Validation endpoint must exactly match the frozen parent mapping')
    if not isinstance(expected_manifest_sha256,str) or re.fullmatch('[0-9a-f]{64}',expected_manifest_sha256) is None:
        raise ValueError('Validation preparation requires a SHA-256 pin')
    if preparation.get('manifestSHA256')!=expected_manifest_sha256:raise ValueError('Validation preparation differs from archived training manifest')
    try:
        tensors,verified,files=_reload_training_population(manifest_path,preparation,training_provenance,expected_manifest_sha256,input_dtype,sample_shape,named_plan,model_graph)
    except (KeyError,TypeError) as error:raise ValueError('Incomplete archived training provenance') from error
    if verified!={key:value for key,value in training_provenance.items() if key not in ('composite','objective')}:
        raise ValueError('Training provenance differs from verified archived Validation population or input contract')
    value={'dataset':'prepared-validation-v1','sourceHash':verified['sourceHash'],
        'validationSamples':verified['validationSamples'],'validationRowIDs':list(verified['validationRowIDs']),
        'featureColumns':list(verified['featureColumns']),**({'objectiveRecipe':verified['objectiveRecipe'],'objectiveMapping':verified['objectiveMapping']} if 'objectiveRecipe' in verified else {'targetColumn':verified['targetColumn']}),
        'preparation':{'manifestSHA256':expected_manifest_sha256,'graphDigest':preparation['graphDigest'],
            'registryDigest':preparation.get('registryDigest'),'validationNode':node,'validationPort':port,
            'selectedDataVersion':preparation['validationDataVersion'],'evaluatedDataVersion':preparation['validationDataVersion'],
            'membershipDigest':dataflow.membership_digest(verified['validationRowIDs']),
            'fittedStateDigests':list(preparation['fittedStateDigests']),'appliedStateDigests':[]}}
    # Retain the actual archived partition identity, without opening another role.
    manifest_bytes=bounded_read(manifest_path,2_000_000)
    if hashlib.sha256(manifest_bytes).hexdigest()!=expected_manifest_sha256:raise ValueError('Preparation changed while selecting Validation partition')
    manifest=json.loads(manifest_bytes)
    entry=next(a for a in manifest['artifacts'] if a.get('nodeID')==node and a.get('port')==port)
    selected=prepared_table_document(files,entry)
    value['preparation']['partitionDigest']=selected['lineage']['partitionDigest']
    if input_dtype=='int64' or named_plan is not None:value['modelInput']=verified['modelInput']
    return (tensors[2],tensors[3]),value,files
