"""Transactional artifact adapter for the isolated dataflow engine."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import re
import stat
import sys
if __package__:
    from . import dataflow
    from .artifact_storage import ArtifactStorage
    from .dataflow_cache import Materialization,table_from_artifact,cache_allowance_maximum
    from .dataflow_storage import bounded_regular_read,completed_inventory
else:
    import dataflow
    from artifact_storage import ArtifactStorage
    from dataflow_cache import Materialization,table_from_artifact,cache_allowance_maximum
    from dataflow_storage import bounded_regular_read,completed_inventory

MAX_ARTIFACT_BYTES=200_000_000
MAX_MANIFEST_BYTES=2_000_000
MAX_OUTPUT_BYTES=202_000_000
MAX_INSPECTION_PREVIEW_BYTES=32_000_000
# Keep copied runtime dependencies and their published integrity inventory identical.
if __package__:from .preparation_dependencies import PORTABLE_DEPENDENCIES
else:from preparation_dependencies import PORTABLE_DEPENDENCIES

def add_cache_arguments(parser):
    # None distinguishes the omitted default from explicitly selecting off.
    parser.add_argument('--dataflow-cache-policy', choices=('off','materialize','reuse'))
    parser.add_argument('--dataflow-cache-limit-bytes',type=int)
    parser.add_argument('--dataflow-reuse-manifest')
    parser.add_argument('--dataflow-reuse-manifest-sha256')


def cache_arguments(args, *, executing):
    limit=args.dataflow_cache_limit_bytes
    selected=args.dataflow_cache_policy
    manifest=args.dataflow_reuse_manifest
    pin=args.dataflow_reuse_manifest_sha256
    if not executing and any(value is not None for value in (selected,manifest,pin,limit)):
        raise ValueError('Dataflow cache flags require --execute-dataflow.')
    policy=selected or 'off'
    if limit is not None and (type(limit) is not int or not 1<=limit<=cache_allowance_maximum()):raise ValueError('Dataflow cache allowance must be a positive integer within the captured retained-data allowance')
    if limit is not None and policy not in ('materialize','reuse'):raise ValueError('Dataflow cache allowance requires materialize or reuse policy')
    if policy=='reuse':
        if not manifest or not pin:
            raise ValueError('Dataflow cache reuse requires an explicit manifest and SHA-256 pin.')
    elif manifest is not None or pin is not None:
        raise ValueError('Dataflow reuse manifest and pin require cache policy reuse.')
    return {'cache_policy':policy,'reuse_manifest':manifest,'reuse_manifest_sha256':pin,'cache_limit_bytes':limit}


def read_json(path,limit=2_000_000):
    with Path(path).open('rb') as stream:content=stream.read(limit+1)
    if len(content)>limit:raise ValueError('JSON input exceeds size limit')
    return json.loads(content)


def read_node_labels(path):
    try:return json.loads(bounded_regular_read(path,65536))
    except (OSError,ValueError) as error:raise ValueError('Dataflow node-label sidecar must be valid JSON in a bounded regular file (maximum 64 KB)') from error


def run_dataflow(graph_path,bindings,output,emit,*,engine=None,workload_policy=None,**kwargs):
    if __package__:from .polars_preparation import preparation_engine
    else:from polars_preparation import preparation_engine
    if __package__:from .workload_policy import WorkloadPolicy,LEGACY,current_policy,workload_scope
    else:from workload_policy import WorkloadPolicy,LEGACY,current_policy,workload_scope
    selected_policy=current_policy() if workload_policy is None else WorkloadPolicy.from_document(workload_policy)
    kwargs.get('check_cancelled',lambda:None)()
    graph=dataflow.upgrade_graph(read_json(graph_path))
    if __package__:from .ocr_tables import job_scope
    else:from ocr_tables import job_scope
    def pack(name):
        return bounded_regular_read(Path(graph_path).parent/'ocr-pack'/name,100_000_000)
    if __package__:from .speech_tables import job_scope as speech_scope
    else:from speech_tables import job_scope as speech_scope
    if __package__:from .audio_classification_tables import job_scope as audio_scope
    else:from audio_classification_tables import job_scope as audio_scope
    if __package__:from .decision_tables import job_scope as decision_scope
    else:from decision_tables import job_scope as decision_scope
    if __package__:from .prebuilt_preparation import job_scope as preparation_model_scope
    else:from prebuilt_preparation import job_scope as preparation_model_scope
    with workload_scope(selected_policy),preparation_model_scope(Path(graph_path).parent,kwargs.get('check_cancelled',lambda:None)),decision_scope(Path(graph_path).parent,kwargs.get('check_cancelled',lambda:None)),audio_scope(Path(graph_path).parent,kwargs.get('check_cancelled',lambda:None)),preparation_engine(dataflow.selected_engine(graph,engine)),job_scope(pack,kwargs.get('check_cancelled',lambda:None)),speech_scope(Path(graph_path).parent,kwargs.get('check_cancelled',lambda:None)):
        return _run_dataflow(graph_path,bindings,output,emit,_graph=graph,_workload_captured=workload_policy is not None or selected_policy is not LEGACY,**kwargs)


def _run_dataflow(graph_path,bindings,output,emit,*,cache_policy="off",reuse_manifest=None,reuse_manifest_sha256=None,node_labels=None,retain_partial=False,check_cancelled=lambda:None,cache_limit_bytes=None,destination_bindings=None,sampling=None,complete_nodes=None,retain_outputs=True,live_previews=None,required_nodes=None,model_input_nodes=None,_graph=None,defer_database_writes=False,measurements=None,collect_measurements=False,_workload_captured=False):
    if __package__:from .polars_preparation import engine_identity
    else:from polars_preparation import engine_identity
    if __package__:from .preparation_plan import build_plan,execute_plan,derive_preparation_code
    else:from preparation_plan import build_plan,execute_plan,derive_preparation_code
    if __package__:from .preparation_demand import build_demand_plan,coverage_report
    else:from preparation_demand import build_demand_plan,coverage_report
    check_cancelled()
    if type(collect_measurements) is not bool:raise ValueError('Execution measurements require an explicit boolean')
    if collect_measurements:
        if measurements is not None:raise ValueError('Execution measurements have one collector owner')
        if __package__:from .execution_measurements import ExecutionMeasurements
        else:from execution_measurements import ExecutionMeasurements
        # Filled after graph admission; no source/configuration values are retained.
    if type(retain_outputs) is not bool:raise ValueError('Output retention must be an explicit boolean')
    if live_previews is None:live_previews=retain_outputs # Legacy omission retains its original observation behavior.
    if type(live_previews) is not bool:raise ValueError('Live previews must be an explicit boolean')
    if type(retain_partial) is not bool:raise ValueError('Partial preparation retention must be an explicit boolean')
    graph=dataflow.upgrade_graph(read_json(graph_path) if _graph is None else _graph)
    nodes,_,_=dataflow.validate(graph)
    if collect_measurements:measurements=ExecutionMeasurements(nodes,on_node=lambda record:emit({'event':'dataflowNodeTiming','measurement':record}))
    if cache_policy!='off' and any(op=='preprocess' and c.get('mode')=='frozen' for op,c in nodes.values()):raise ValueError('Saved preparation currently requires cache Off so the captured model binding is verified on every run.')
    requested_required=[] if required_nodes is None else required_nodes
    if not isinstance(requested_required,list) or len(requested_required)>128 or any(not isinstance(node,str) or node not in nodes for node in requested_required) or len(requested_required)!=len(set(requested_required)):raise ValueError('Required preparation outputs must reference distinct nodes in this graph')
    requested_model_inputs=[] if model_input_nodes is None else model_input_nodes
    if (not isinstance(requested_model_inputs,list) or len(requested_model_inputs)>128
        or any(type(node) is not str or node not in requested_required for node in requested_model_inputs)
        or len(requested_model_inputs)!=len(set(requested_model_inputs))):
        raise ValueError('Model input nodes must be distinct explicitly required preparation nodes.')
    model_input_nodes=set(requested_model_inputs)
    required_nodes=set(requested_required)|(set(nodes)-{edge['from']['node'] for edge in graph['edges']})
    required_nodes.update(node for node,(operator,c) in nodes.items() if operator=='view' or operator=='outputDelimited' and c.get('destination') in ('database','api'))
    identity=engine_identity()
    demand=build_demand_plan(graph,sampling,complete_nodes)
    if __package__:from .workload_policy import current_policy
    else:from workload_policy import current_policy
    active_policy=current_policy()
    segmented_tables=_workload_captured
    artifact_limit=active_policy.limit('retainedBytes') if segmented_tables else MAX_ARTIFACT_BYTES
    file_limit=artifact_limit if segmented_tables else 50_000_000
    inventory_limit=min(MAX_MANIFEST_BYTES//32,active_policy.limit('planNodes')*32+active_policy.limit('collectionMembers')+len(PORTABLE_DEPENDENCIES)+16) if segmented_tables else 300
    plan=build_plan(graph,workload_policy=active_policy if _workload_captured else None,engine=identity['name'],engine_version=identity['version'],node_labels=node_labels,sampling=demand.sampling,complete_nodes=demand.complete_node_ids)
    # Reserve the conservative full-output inventory, including protected test ports.
    # Required-only retention may write fewer files; support files count before source reads.
    artifact_count=sum(len(dataflow.OPS[operator][1])+(operator=='view') for operator,_ in nodes.values())+7+len(PORTABLE_DEPENDENCIES)
    if segmented_tables:artifact_count+=sum(len(dataflow.OPS[operator][1]) for operator,_ in nodes.values())
    if artifact_count>inventory_limit:raise ValueError('Dataflow artifact count exceeds inventory limit of 300; reduce the executed graph.')
    dataflow.validate_node_labels({} if node_labels is None else node_labels,[n["id"] for n in graph["nodes"]])
    if not isinstance(bindings,dict) or any(not isinstance(k,str) or not isinstance(v,str) or len(v)>4096 for k,v in bindings.items()):raise ValueError('Source bindings require node IDs mapped to local path strings')
    check_cancelled()
    materialization=Materialization(cache_policy,reuse_manifest,reuse_manifest_sha256,cache_limit_bytes=cache_limit_bytes,check_cancelled=check_cancelled)
    materialization.set_demand(demand)
    check_cancelled()
    output=Path(output).absolute()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):raise ValueError('Dataflow output must be absent or empty')
    output.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.dataflow-',dir=output.parent));stage_identity=stage.stat();artifacts=[];written=0;captured_previews={};preview_bytes=0;node_row_counts={};table_views={}
    if __package__:from .dataflow_output import OutputTransaction
    else:from dataflow_output import OutputTransaction
    destination_bindings={} if destination_bindings is None else destination_bindings
    output_transaction=OutputTransaction(destination_bindings,check_cancelled)
    output_transaction.defer_database_writes=defer_database_writes
    storage=None;segments=None;segment_count=0
    graph_hash=dataflow.digest(graph)
    if __package__:from .image_artifacts import artifact_scope,table_schema_version,references
    else:from image_artifacts import artifact_scope,table_schema_version,references
    if __package__:from . import media_artifacts as media
    else:import media_artifacts as media
    media_assets={}
    image_assets={}
    def read_image(reference, *, media_asset=False):
        check_cancelled()
        pin=reference.get('sha256') if isinstance(reference,dict) else None
        if not isinstance(reference,dict) or not isinstance(pin,str) or pin not in (media_assets if media_asset else image_assets) or (media_assets if media_asset else image_assets)[pin]!=reference:raise ValueError('Image output requires captured immutable image bytes from this execution.')
        # Anchor the directory identity before opening the fixed inventory filename.
        root=os.open(stage,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        descriptor=None
        try:
            actual=os.fstat(root)
            if (actual.st_dev,actual.st_ino)!=(stage_identity.st_dev,stage_identity.st_ino):raise ValueError('Image artifact root changed.')
            descriptor=os.open(media.filename(reference) if media_asset else 'image-'+pin+'.kwipimg',os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=root)
            before=os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_size>50_000_000:raise ValueError('Captured image must be a bounded regular file.')
            chunks=[];count=0
            while True:
                check_cancelled();block=os.read(descriptor,262144)
                if not block:break
                count+=len(block)
                if count>before.st_size:raise ValueError('Captured image bytes changed.')
                chunks.append(block)
            after=os.fstat(descriptor)
            if count!=before.st_size or (before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise ValueError('Captured image bytes changed.')
            return b''.join(chunks)
        finally:
            if descriptor is not None:os.close(descriptor)
            os.close(root)
    output_transaction.image_reader=read_image
    def write_bytes(name,content,**identity):
        nonlocal written
        check_cancelled()
        if len(artifacts)>=inventory_limit:raise ValueError('Dataflow image/artifact inventory exceeds its 300-file bound.')
        if len(content)>50_000_000 or written+len(content)>artifact_limit:raise ValueError('Dataflow artifact size limit exceeded')
        if identity.get('kind') in ('table','state'):materialization.charge_materialized(len(content))
        storage.atomic_bytes(name,content,maximum_bytes=50_000_000)
        written+=len(content)
        artifacts.append({'path':name,'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content),**identity})
    def publish_image(content,reference):
        pin=reference['sha256']
        if pin in image_assets:
            if image_assets[pin]!=reference:raise ValueError('Conflicting image artifact geometry.')
            return
        write_bytes('image-'+pin+'.kwipimg',content,kind='imagePixels',image=reference)
        image_assets[pin]=reference
    def publish_media(content,reference):
        pin=reference['sha256']
        if pin in media_assets:
            if media_assets[pin]!=reference:raise ValueError('Conflicting media artifact reference.')
            return
        kind=next(name for name,ref_kind in media.KINDS.values() if ref_kind==reference['kind'])
        write_bytes(media.filename(reference),content,kind=kind,media=reference)
        media_assets[pin]=reference
    def write(name,value,**identity):
        nonlocal written
        if len(artifacts)>=inventory_limit:raise ValueError('Dataflow image/artifact inventory exceeds its 300-file bound.')
        hasher=hashlib.sha256();size=0
        def chunks(sink):
            nonlocal size
            for block in dataflow.iter_json(value):
                check_cancelled();size+=len(block)
                if written+size>artifact_limit:raise ValueError('Dataflow artifact size limit exceeded')
                sink.write(block);hasher.update(block)
        storage.atomic_write(name,maximum_bytes=50_000_000,writer=chunks)
        if identity.get('kind') in ('table','state'):materialization.charge_materialized(size)
        written+=size
        artifacts.append({'path':name,'sha256':hasher.hexdigest(),'bytes':size,**identity})
    def node_complete(node_id,ports):
        nonlocal preview_bytes,written,segment_count
        check_cancelled()
        previews={}
        node_row_counts[node_id]={port:len(value.rows) for port,value in ports.items() if isinstance(value,dataflow.Table)}
        for port,value in ports.items():
            name=f'node-{len(artifacts):03d}.json'
            if isinstance(value,dataflow.Table):
                for reference in references(value):
                    if image_assets.get(reference['sha256'])!=reference:raise ValueError('Image table references an artifact outside its owned immutable inventory.')
                def restore_media(reference):
                    pin=reference['sha256']
                    if pin in media_assets:return
                    cached=materialization.media_asset(pin)
                    if cached is None:raise ValueError('Cached media lacks its captured immutable artifact.')
                    content,expected=cached
                    if expected!=reference:raise ValueError('Cached media typed reference differs.')
                    parent=reference['metadata'].get('parentSHA256')
                    if parent and parent not in media_assets and parent not in image_assets:
                        prior=materialization.media_asset(parent)
                        if prior is None:raise ValueError('Cached prepared media lacks its parent artifact.')
                        if prior[1].get('kind')=='imagePixelsRef':publish_image(*prior)
                        else:restore_media(prior[1])
                    media.decode_asset(content,expected=reference);publish_media(content,reference)
                for reference in media.references(value):
                    if media_assets.get(reference['sha256'])!=reference:restore_media(reference)
                    if media_assets.get(reference['sha256'])!=reference:raise ValueError('Media table references an artifact outside its owned immutable inventory.')
                if retain_outputs or cache_policy!='off' or node_id in required_nodes:
                    if node_id in model_input_nodes and (len(value.rows)>100_000 or len(value.rows)*len(value.columns)>5_000_000):
                        raise ValueError('This model requires at most 100,000 rows and five million cells. Add Row Sample before model preparation.')
                    # External brokers receive one pinned, read-only table descriptor;
                    # they have no authority to open retained segment paths. Materialize
                    # only this explicit side-effect boundary in the existing bounded
                    # inline transport. Analytics intermediates stay shared/segmented.
                    external_output = nodes[node_id][0] == 'outputDelimited' and nodes[node_id][1].get('destination') in ('database', 'api')
                    if segments is not None and node_id not in model_input_nodes and not external_output:
                        if __package__:from .segmented_table_capture import capture_table,envelope
                        else:from segmented_table_capture import capture_table,envelope
                        reference=capture_table(segments,value)
                        for descriptor in segments.descriptors[segment_count:]:
                            if len(artifacts)>=inventory_limit or written+descriptor['bytes']>artifact_limit:raise ValueError('Retained table segments exceed the captured artifact allowance.')
                            artifacts.append({'path':descriptor['path'],'bytes':descriptor['bytes'],'sha256':descriptor['sha256'],'kind':'tableSegment'})
                            written+=descriptor['bytes'];materialization.charge_materialized(descriptor['bytes'])
                        segment_count=len(segments.descriptors)
                        write(name,envelope(value,reference),nodeID=node_id,port=port,kind='table',tableStorageVersion=1)
                    else:
                        write(name,{'schemaVersion':table_schema_version(value.columns),'kind':'table','columns':value.columns,'rows':value.rows,'lineage':{**value.evidence(),'dataVersion':value.data_version,'partitionDigest':value.partition_digest,'fittedStateDigests':list(value.fitted_states),'diagnostics':list(value.diagnostics)}},nodeID=node_id,port=port,kind='table')
                if nodes[node_id][0]=='view':
                    if __package__:from .table_view import build_index,MAX_INDEX_BYTES
                    else:from table_view import build_index,MAX_INDEX_BYTES
                    if not demand.is_complete(node_id):raise ValueError('Table View requires complete upstream data')
                    table_pin=artifacts[-1]['sha256'];index_name=f'view-{len(artifacts):03d}.sqlite3'
                    if segments is not None:
                        receipt=build_index(value,stage/index_name,source_sha256=table_pin,check_cancelled=check_cancelled,retention_budget=storage.budget)
                        index=receipt['index'];written+=index['bytes']
                    else:
                        if written+MAX_INDEX_BYTES>artifact_limit:raise ValueError('Table View cannot reserve its bounded index artifact allowance')
                        storage.budget.reserve(MAX_INDEX_BYTES)
                        try:
                            receipt=build_index(value,stage/index_name,source_sha256=table_pin,check_cancelled=check_cancelled)
                        except BaseException:
                            storage.budget.release(MAX_INDEX_BYTES);raise
                        index=receipt['index'];storage.budget.release(MAX_INDEX_BYTES-index['bytes']);written+=index['bytes']
                    artifact={'path':index_name,**index,'kind':'tableViewIndex','nodeID':node_id,'port':port,'sourceSHA256':table_pin}
                    artifacts.append(artifact)
                    table_views[node_id]={**receipt,'index':artifact}
                if __package__:from .table_identity import enabled as deferred_identity
                else:from table_identity import enabled as deferred_identity
                canonical_identity=not deferred_identity() or retain_outputs or cache_policy!='off' or node_id in required_nodes
                previews[port]=output_preview(value,include_values=live_previews,canonical_identity=canonical_identity)
            else:
                write(name,value,nodeID=node_id,port=port,kind='state')
                previews[port]=output_preview(value,include_values=live_previews)
        event={'event':'dataflowPreview','nodeID':node_id,'outputs':previews,'execution':materialization.nodes[-1],
               'coverage':{'completeness':'complete' if demand.is_complete(node_id) else 'sampled','rowCounts':node_row_counts[node_id]}}
        if node_id in table_views:event['tableView']=table_views[node_id]
        bound_preview(event)
        preview_bytes+=len(json.dumps(previews,allow_nan=False).encode())
        if preview_bytes>31_000_000:raise ValueError('Preparation previews exceed their 32 MB display budget')
        if retain_outputs:captured_previews[node_id]=previews
        emit(event)
    try:
        storage=ArtifactStorage(stage,maximum_bytes=artifact_limit+MAX_MANIFEST_BYTES if segmented_tables else MAX_OUTPUT_BYTES,check_cancelled=check_cancelled)
        emit({'event':'dataflowStarted','graphDigest':graph_hash,'nodeCount':len(graph['nodes']),'cachePolicy':cache_policy,'demand':demand.to_document()})
        # Scratch lives inside this operation's writable stage and is never archived.
        scratch=Path(tempfile.mkdtemp(prefix='.row-scratch-',dir=stage))
        try:
            with dataflow.disk_scope(directory=scratch),artifact_scope(publish_image),media.artifact_scope(publish_media,lambda ref:read_image(ref,media_asset=True)):
                if segmented_tables:
                    if __package__:from .segmented_table_store import SegmentWriter
                    else:from segmented_table_store import SegmentWriter
                    segments=SegmentWriter(storage,maximum_bytes=artifact_limit).__enter__()
                try:
                    execute_plan(plan,bindings,on_node=node_complete,materialization=materialization,check_cancelled=check_cancelled,retain_results=False,
                                     on_node_started=lambda node_id:emit({'event':'dataflowNodeStarted','nodeID':node_id}),destination_bindings=destination_bindings,output_transaction=output_transaction,measurements=measurements)
                finally:
                    if segments is not None:
                        segments.__exit__(*sys.exc_info());segments=None
        finally:shutil.rmtree(scratch,ignore_errors=True)
        check_cancelled()
        if __package__:from .table_source_options import reader_versions
        else:from table_source_options import reader_versions
        dependencies={'numpy':dataflow.np.__version__,**reader_versions(graph)}
        import importlib.metadata
        for node in graph['nodes']:
            if node['operator']=='kwip.data.outputDelimited' and node['config'].get('destination','file')=='file':
                if __package__:from .table_output_writers import options
                else:from table_output_writers import options
                fmt,_=options(node['config'])
                package={'parquet':'pyarrow','xlsx':'openpyxl'}.get(fmt)
                if package:dependencies[package]=importlib.metadata.version(package)
        if identity['name']=='polars':dependencies.update({'polars':identity['version'],'polars-runtime-32':identity['version']})
        requirements=''.join(name+'=='+version+'\n' for name,version in sorted(dependencies.items()))
        write_bytes('requirements-preparation.txt',requirements.encode(),kind='preparationDependencies')
        write('preparation-plan.json',plan.to_document(),kind='preparationPlan')
        write_bytes('preparation.py',derive_preparation_code(plan).encode(),kind='preparationCode')
        if retain_outputs:write('previews.json',{'schemaVersion':1,'graphDigest':graph_hash,'outputs':captured_previews},kind='previews')
        write('dataflow.json',graph,kind='graph')
        write('source-bindings.example.json',{key:'PATH_TO_CAPTURED_API_TABLE' if nodes[key][1].get('format')=='api' else 'PATH_TO_CAPTURED_DATABASE_TABLE' if nodes[key][1].get('format')=='database' else 'PATH_TO_LOCAL_'+nodes[key][1].get('format','csv').upper() for key in bindings},kind='bindingsTemplate')
        code='''#!/usr/bin/env python3
"""Reproduce preparation with Python + NumPy and explicit local source bindings."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from dataflow_cli import read_json,run_dataflow
parser=argparse.ArgumentParser()
parser.add_argument('--source-bindings',required=True)
parser.add_argument('--output',required=True)
parser.add_argument('--destination-bindings')
args=parser.parse_args()
run_dataflow(Path(__file__).with_name('dataflow.json'),read_json(args.source_bindings,1000000),args.output,lambda event:print(json.dumps(event,allow_nan=False),flush=True),destination_bindings=read_json(args.destination_bindings,1000000) if args.destination_bindings else None)
'''
        code=code.replace('destination_bindings=read_json(args.destination_bindings,1000000)',
                          ('workload_policy='+repr(active_policy.to_document())+',' if segmented_tables else '')+'required_nodes='+repr(requested_required)+',model_input_nodes='+repr(requested_model_inputs)+',retain_outputs='+repr(retain_outputs)+',live_previews='+repr(live_previews)+',engine='+repr(engine_identity()['name'])+',sampling='+repr(demand.sampling)+',complete_nodes='+repr(list(demand.complete_node_ids))+',destination_bindings=read_json(args.destination_bindings,1000000)')
        write_bytes('prep.py',code.encode(),kind='preparationCode')
        for filename in PORTABLE_DEPENDENCIES:
            check_cancelled()
            # The registry used for this execution is frozen when dataflow is
            # imported. Archive that same snapshot even if a developer edits
            # the catalog while a long operation is running.
            content=(json.dumps(dataflow.REGISTRY,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
                     if filename=='dataflow-operators.json' else bounded_regular_read(Path(__file__).with_name(filename),50_000_000))
            write_bytes(filename,content,kind='preparationCode' if filename.endswith('.py') else 'registry')
        manifest={'schemaVersion':1,'kind':'dataflowExecution','status':'completed','graphDigest':graph_hash,'registryDigest':dataflow.digest(dataflow.REGISTRY),'numpyVersion':dataflow.np.__version__,'preparationPlanDigest':plan.plan_digest,'preparationEngine':identity,'artifacts':artifacts,'preparationCode':'prep.py','sourceBindings':'external; see source-bindings.example.json','cache':materialization.report(),'limits':{'maxRows':dataflow.MAX_TABLE_CELLS,'maxTableCells':dataflow.MAX_TABLE_CELLS,'maxScratchBytes':active_policy.limit('scratchBytes'),'eagerRowMaximum':dataflow.EAGER_ROW_LIMIT,'maxSourceBytes':dataflow.MAX_BYTES,'maxRetainedCells':5_000_000,'maxArtifactBytes':MAX_ARTIFACT_BYTES}}
        if _workload_captured:manifest['workloadPolicy']=active_policy.to_document()
        if requested_model_inputs:manifest['modelInputNodes']=sorted(requested_model_inputs)
        if segmented_tables:manifest['tableStorageVersion']=1
        manifest['limits']['maxArtifactBytes']=artifact_limit
        if not retain_outputs:manifest.update(observationRetention='requiredOnly',requiredOutputNodes=sorted(requested_required))
        manifest['executionCoverage']=coverage_report(demand,node_row_counts)
        if table_views:manifest['tableViews']=table_views
        try:output_receipts=output_transaction.commit()
        except ValueError as error:
            if hasattr(error,'node'):raise dataflow.DataFlowExecutionError(error.node,'outputDelimited',{},error,(node_labels or {}).get(error.node)) from error
            raise
        if output_receipts:manifest['outputs']=output_receipts
        check_cancelled()
        storage.atomic_bytes('manifest.json',json.dumps(manifest,indent=2,allow_nan=False).encode(),maximum_bytes=MAX_MANIFEST_BYTES)
        if output.exists() and any(output.iterdir()):raise ValueError('Output changed during execution; existing content preserved')
        check_cancelled()
        os.replace(stage,output)
        check_cancelled()
        for receipt in output_receipts:emit({'event':'dataflowOutputSaved',**receipt})
        emit({'event':'dataflowCompleted','outputDir':str(output),'manifestPath':str(output/'manifest.json'),'graphDigest':graph_hash,'artifacts':[a['path'] for a in artifacts],'cache':materialization.report(),'executionCoverage':manifest['executionCoverage']})
        return manifest
    except BaseException:
        output_transaction.rollback()
        # Even a cancellation between directory rename and completion delivery must
        # invalidate success. Compare inode ownership so preexisting user output is safe.
        for candidate in (stage,output):
            try:
                identity=candidate.lstat()
                if (identity.st_dev,identity.st_ino)==(stage_identity.st_dev,stage_identity.st_ino):
                    (candidate/'manifest.json').unlink(missing_ok=True)
            except FileNotFoundError:pass
        raise
    finally:
        output_transaction.close()
        if segments is not None:segments.__exit__(*sys.exc_info())
        if storage is not None:storage.close()
        # Native run archives retain incomplete node bytes for recovery inspection.
        # They remain in the owned staging directory, never a completed manifest.
        # Portable/default callers retain the existing cleanup behavior.
        if not retain_partial and stage.exists():shutil.rmtree(stage)


def output_preview(value,*,include_values=True,canonical_identity=True):
    if isinstance(value,dataflow.Table):
        lineage={**value.evidence(),'dataVersion':value.data_version if canonical_identity else None,'partitionDigest':value.partition_digest,'fittedStateDigests':list(value.fitted_states)}
        preview=value.preview(include_version=canonical_identity) if include_values else {'columns':list(value.columns),'rows':[],'rowCount':len(value.rows),'partition':value.role,'dataVersion':value.data_version if canonical_identity else None,'diagnostics':list(value.diagnostics[-16:]),'diagnosticCount':len(value.diagnostics)}
        result={'kind':'table','preview':preview,'lineage':{k:v for k,v in lineage.items() if k not in ('observationIDs','observationIDPrefix','observationIDSuffixes','diagnostics')}}
        if not canonical_identity:result['schemaVersion']=2
        return result
    return {'kind':'state','summary':{k:value[k] for k in ('operator','columns','means','scales','fillValues','observedCounts','emptyColumns','vocabularies','rules','contentDigest','fitDataVersion') if k in value and (include_values or k in ('operator','columns','contentDigest','fitDataVersion'))},'lineage':{k:v for k,v in value['fitEvidence'].items() if k not in ('observationIDs','observationIDPrefix','observationIDSuffixes','diagnostics')}}


def bound_preview(event):
    previews=event['outputs']
    # Native ProcessPump limits one JSON line to 1 MB, including ASCII escapes.
    while len(json.dumps(event,allow_nan=False).encode())>900_000:
        candidates=[v['preview'] for v in previews.values() if v['kind']=='table' and v['preview']['rows']]
        if not candidates:raise ValueError('Preview metadata exceeds event bound')
        largest=max(candidates,key=lambda value:len(json.dumps(value)))
        largest['rows'].pop();largest['previewTruncated']=True


def _historical_registry_compatible(registry,nodes):
    """Read frozen values using known port contracts, never historical algorithms.

    Additive catalog entries, renamed display titles and compatible configuration
    schema growth do not change already captured table/state artifact semantics.
    Execution/cache reuse retain their stronger current-runtime identity checks.
    """
    if not isinstance(registry,dict) or type(registry.get('schemaVersion')) is not int or registry['schemaVersion']!=1 or registry.get('packageID')!=dataflow.REGISTRY['packageID']:
        raise ValueError('Historical preparation registry is unsupported by this reader')
    operators=registry.get('operators')
    if not isinstance(operators,list) or not 1<=len(operators)<=256:
        raise ValueError('Historical preparation registry has invalid operator bounds')
    archived={}
    for definition in operators:
        if not isinstance(definition,dict) or not isinstance(definition.get('id'),str) or definition['id'] in archived:
            raise ValueError('Historical preparation registry has ambiguous operators')
        archived[definition['id']]=definition
    def ports(definition,key):
        values=definition.get(key)
        if not isinstance(values,list) or len(values)>128:raise ValueError('Invalid historical port contract')
        result={}
        for port in values:
            if not isinstance(port,dict) or not isinstance(port.get('id'),str) or not isinstance(port.get('kind'),str) or type(port.get('required')) is not bool or port['id'] in result:
                raise ValueError('Invalid or repeated historical port contract')
            result[port['id']]={key:port.get(key) for key in ('kind','required','partition','stateKind')}
        return result
    for operator,_ in nodes.values():
        current=dataflow.DEFINITIONS[operator];old=archived.get(dataflow.PREFIX+operator)
        version=old.get('version') if isinstance(old,dict) else None
        if not isinstance(version,dict) or set(version)!={'major','minor','patch'} or any(type(value) is not int or not 0<=value<=2**31-1 for value in version.values()) or version['major']!=current['version']['major']:
            raise ValueError('Historical operator major version is unsupported by this reader')
        if any(ports(old,key)!=ports(current,key) for key in ('inputs','outputs')):
            raise ValueError('Historical operator ports are unsupported by this reader')


def inspect_dataflow(manifest_path,manifest_sha256,emit,*,check_cancelled=lambda:None):
    """Restore historical outputs from verified bytes, without source or code execution."""
    if not isinstance(manifest_sha256,str) or re.fullmatch('[0-9a-f]{64}',manifest_sha256) is None:raise ValueError('Inspection requires a lowercase manifest SHA-256 pin')
    check_cancelled()
    emit({'event':'dataflowInspectionStarted'})
    manifest,files,pin=completed_inventory(manifest_path,manifest_sha256,check_cancelled=check_cancelled,lazy=True,allow_sample=True)
    check_cancelled()
    graph=json.loads(files.get('dataflow.json',b'null'))
    if dataflow.digest(graph)!=manifest.get('graphDigest'):raise ValueError('Preparation graph digest mismatch')
    nodes,inputs,order=dataflow.validate(graph,historical=True)
    registry=json.loads(files.get('dataflow-operators.json',b'null'))
    if dataflow.digest(registry)!=manifest.get('registryDigest'):raise ValueError('Preparation registry digest mismatch')
    _historical_registry_compatible(registry,nodes)
    version=manifest.get('numpyVersion')
    if not isinstance(version,str) or not 1<=len(version)<=128:raise ValueError('Invalid historical runtime version')
    cache=manifest.get('cache')
    if not isinstance(cache,dict) or type(cache.get('schemaVersion')) is not int or cache.get('schemaVersion')!=1 or cache.get('policy') not in ('off','materialize','reuse'):raise ValueError('Unsupported historical cache report')
    hex64=lambda value:isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None
    if not hex64(cache.get('implementationDigest')):raise ValueError('Invalid historical implementation digest')
    source_pin=cache.get('sourceManifestSHA256')
    if source_pin is not None and not hex64(source_pin):raise ValueError('Invalid historical source manifest pin')
    records=cache.get('nodes');executed=cache.get('executedNodeIDs');reused=cache.get('reusedNodeIDs')
    if not isinstance(records,list) or len(records)>128 or not isinstance(executed,list) or not isinstance(reused,list):raise ValueError('Invalid historical execution index')
    if any(not isinstance(v,str) for v in executed+reused) or len(set(executed+reused))!=len(executed+reused) or set(executed+reused)!=set(nodes):raise ValueError('Historical execution index differs from graph')
    index={}
    for record in records:
        if not isinstance(record,dict) or set(record)!= {'nodeID','cacheKey','status','reason'}:raise ValueError('Invalid historical cache node')
        ident=record['nodeID']
        if not isinstance(ident,str) or ident not in nodes or ident in index or not hex64(record['cacheKey']) or record['status'] not in ('executed','reused') or record['reason'] not in ('verifiedHit','sourceRevalidated','disabled','identityChanged','notMaterialized'):raise ValueError('Invalid historical cache node')
        if (ident in reused)!=(record['status']=='reused'):raise ValueError('Historical cache status mismatch')
        index[ident]=record
    if cache['policy']=='off':
        if records or reused:raise ValueError('Disabled cache cannot contain reuse records')
    elif set(index)!=set(nodes):raise ValueError('Historical cache node index differs from graph')
    outputs={node:{} for node in nodes};retained=0
    for entry in manifest['artifacts']:
        check_cancelled()
        if entry.get('kind') not in ('table','state'):continue
        ident=entry.get('nodeID');port=entry.get('port')
        if not isinstance(ident,str) or ident not in nodes or not isinstance(port,str) or port in outputs[ident] or port not in dataflow.OPS[nodes[ident][0]][1]:raise ValueError('Invalid historical output index')
        if (port=='state')!=(entry['kind']=='state'):raise ValueError('Historical output type mismatch')
        if entry['kind']=='table':
            if __package__:from .dataflow_table_reader import read_table_artifact
            else:from dataflow_table_reader import read_table_artifact
            value=read_table_artifact(files.path_for(entry['path']),entry['sha256'],check_cancelled=check_cancelled)
        else:
            value=dataflow.load_state(files[entry['path']].decode())
            if value['graphDigest']!=manifest['graphDigest']:raise ValueError('Historical fitted-state graph mismatch')
            if value['operator']!=dataflow.PREFIX+nodes[ident][0]:raise ValueError('Historical fitted-state operator mismatch')
        check_cancelled()
        outputs[ident][port]=output_preview(value)
        value=None
    events=[];preview_bytes=0
    for ident in order:
        check_cancelled()
        if set(outputs[ident])!=dataflow.configured_ports(*nodes[ident],inputs[ident])[2]:
            if manifest.get('observationRetention')=='requiredOnly':continue
            raise ValueError('Historical graph output is missing')
        execution=index.get(ident,{'nodeID':ident,'cacheKey':None,'status':'executed','reason':'disabled'})
        event={'event':'dataflowPreview','nodeID':ident,'outputs':outputs[ident],'execution':execution}
        if manifest.get('executionCoverage') is not None:event['coverage']=manifest['executionCoverage']['nodes'][ident]
        if ident in manifest.get('tableViews',{}):event['tableView']=manifest['tableViews'][ident]
        bound_preview(event)
        preview_bytes+=len(json.dumps(event,allow_nan=False).encode())
        if preview_bytes>MAX_INSPECTION_PREVIEW_BYTES:raise ValueError('Historical previews exceed aggregate 32 MB bound')
        events.append(event)
    for event in events:
        check_cancelled();emit(event)
    path=Path(manifest_path).absolute()
    check_cancelled()
    emit({'event':'dataflowInspectionCompleted','outputDir':str(path.parent),'manifestPath':str(path),'manifestSHA256':pin,'graphDigest':manifest['graphDigest'],'artifacts':[a['path'] for a in manifest['artifacts']],'cache':cache,'executionCoverage':manifest.get('executionCoverage'),'provenance':{'historical':True,'sourceRevalidated':False,'cacheReuseVerified':False,'registryDigest':manifest['registryDigest'],'numpyVersion':version}})
