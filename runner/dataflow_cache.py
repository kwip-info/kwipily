"""Explicit, pinned reuse of prior successful dataflow materializations; no scanning."""
import hashlib
import json
from pathlib import Path
import platform
import re
if __package__:
    from . import dataflow as df
    from .dataflow_storage import bounded_regular_read
else:
    import dataflow as df
    from dataflow_storage import bounded_regular_read

if __package__:from .preparation_dependencies import PORTABLE_DEPENDENCIES
else:from preparation_dependencies import PORTABLE_DEPENDENCIES

MAX_BYTES=200_000_000

def cache_allowance_maximum():
    if __package__:from .workload_policy import current_policy
    else:from workload_policy import current_policy
    return current_policy().limit('retainedBytes')


def implementation_digest():
    hashes={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in PORTABLE_DEPENDENCIES}
    if __package__:from .polars_preparation import engine_identity
    else:from polars_preparation import engine_identity
    import importlib.metadata
    # All bundled reader engines participate, including absent optional wheels,
    # so installing/updating one invalidates earlier retained materializations.
    readers={}
    for package in ('polars','polars-runtime-32','pyarrow','fastexcel','openpyxl','defusedxml','xlsx2csv','markdown-it-py','lxml','Pillow','easyocr','torchvision','pypdfium2','opencv-python-headless','pypdf','python-docx','typing-extensions','striprtf'):
        try:readers[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:readers[package]=None
    return df.digest({'readerVersions':readers,'engine':engine_identity(),'files':hashes,'python':platform.python_version(),'numpy':df.np.__version__,'platform':platform.platform()})

def table_identity(table):
    return {'dataVersion':table.data_version,'sourceDigest':table.source_digest,'role':table.role,'partitionDigest':table.partition_digest,'membershipDigest':df.membership_digest(table.row_ids),'fittedStateDigests':list(table.fitted_states),'diagnosticsDigest':df.digest(table.diagnostics)}


def table_from_artifact(value):
    if not isinstance(value,dict) or set(value)!={'schemaVersion','kind','columns','rows','lineage'} or value['kind']!='table':raise ValueError('Invalid cached table artifact')
    columns=value['columns'];rows=value['rows'];lineage=value['lineage'];df._fitted_module().schema_valid(columns)
    if __package__:from .image_artifacts import validate_table_version
    else:from image_artifacts import validate_table_version
    validate_table_version(value['schemaVersion'],columns)
    if not isinstance(rows,list) or not 0<=len(rows)<=df.MAX_TABLE_CELLS//len(columns) or any(not isinstance(row,list) or len(row)!=len(columns) for row in rows):raise ValueError('Invalid cached row bounds')
    ids=lineage.get('observationIDs')
    if not isinstance(ids,list) or len(ids)!=len(rows) or any(not isinstance(i,str) or not 1<=len(i)<=200 for i in ids) or len(set(ids))!=len(ids):raise ValueError('Invalid cached observation identities')
    if lineage.get('membershipDigest')!=df.membership_digest(ids) or lineage.get('role') not in ('unassigned','training','validation','test'):raise ValueError('Invalid cached partition membership')
    if lineage.get('dataVersion')!=df.digest({'columns':columns,'rows':rows,'ids':ids}):raise ValueError('Cached table data version mismatch')
    fitted=lineage.get('fittedStateDigests',[]);diagnostics=lineage.get('diagnostics',[])
    if not isinstance(fitted,list) or len(fitted)>128 or any(not isinstance(v,str) or re.fullmatch('[0-9a-f]{64}',v) is None for v in fitted):raise ValueError('Invalid cached fitted-state chain')
    if not isinstance(diagnostics,list) or len(diagnostics)>16384 or any(not isinstance(d,dict) for d in diagnostics):raise ValueError('Invalid cached diagnostics')
    for row in rows:
        for value,column in zip(row,columns):
            if value is None:
                if not column['nullable']:raise ValueError('Cached null violates column schema')
            elif not df._fitted_module().typed(value,column['dtype']):raise ValueError('Cached value violates column dtype')
    if not isinstance(lineage.get('sourceDigest'),str) or not isinstance(lineage.get('partitionDigest'),str):raise ValueError('Invalid cached source identity')
    return df.Table(tuple(columns),tuple(map(tuple,rows)),tuple(ids),lineage['sourceDigest'],lineage['role'],lineage['dataVersion'],lineage['partitionDigest'],tuple(fitted),tuple(diagnostics))


class Materialization:
    def __init__(self,policy='off',manifest_path=None,manifest_sha256=None,*,cache_limit_bytes=None,check_cancelled=lambda:None):
        if policy not in ('off','materialize','reuse'):raise ValueError('Cache policy must be off, materialize or reuse')
        if cache_limit_bytes is not None:
            if type(cache_limit_bytes) is not int or not 1<=cache_limit_bytes<=cache_allowance_maximum():raise ValueError('Dataflow cache allowance must be a positive integer within the captured retained-data allowance')
            if policy not in ('materialize','reuse'):raise ValueError('Dataflow cache allowance requires materialize or reuse policy')
        self.check_cancelled=check_cancelled
        self.materialized_limit=min(MAX_BYTES,cache_allowance_maximum()) if cache_limit_bytes is None else cache_limit_bytes
        self.materialized_bytes=0
        if policy!='reuse' and (manifest_path is not None or manifest_sha256 is not None):raise ValueError('Cache references require explicit reuse policy')
        self.policy=policy;self.implementation=implementation_digest();self.source_pin=None;self.entries={};self.prior={};self.nodes=[];self.root=None;self.read_bytes=0;self.verified_files={};self.demand_plan=None;self.current_coverage={};self.prior_coverage=None;self.asset_entries={}
        if policy=='reuse':
            if manifest_path is None or not isinstance(manifest_sha256,str) or re.fullmatch('[0-9a-f]{64}',manifest_sha256) is None:raise ValueError('Cache reuse requires an explicit manifest and SHA-256 pin')
            path=Path(manifest_path).absolute();self.root=path.parent.resolve();raw=bounded_regular_read(self.root/path.name,2_000_000)
            if hashlib.sha256(raw).hexdigest()!=manifest_sha256:raise ValueError('Cache selected-manifest integrity mismatch')
            self.source_pin=manifest_sha256;manifest=json.loads(raw);cache=manifest.get('cache',{})
            if manifest.get('status')!='completed' or manifest.get('kind')!='dataflowExecution' or type(manifest.get('schemaVersion')) is not int or manifest['schemaVersion']!=1:raise ValueError('Cache requires a completed dataflow manifest')
            if cache.get('schemaVersion')!=1 or cache.get('policy') not in ('materialize','reuse') or cache.get('implementationDigest')!=self.implementation:raise ValueError('Cache version is stale or has no reusable outputs. Choose Recompute & retain and prepare again (CLI: materialize policy).')
            if __package__:from .preparation_artifact_limits import ArtifactLimits
            else:from preparation_artifact_limits import ArtifactLimits
            bounds=ArtifactLimits(manifest)
            allowance=cache.get('materializedBytesLimit');measured=cache.get('materializedBytes')
            if 'materializedBytesLimit' in cache or 'materializedBytes' in cache:
                if type(allowance) is not int or not 1<=allowance<=bounds.total or type(measured) is not int or not 0<=measured<=allowance:
                    raise ValueError('Invalid retained cache allowance evidence')
            nodes=cache.get('nodes');artifacts=manifest.get('artifacts')
            if not isinstance(nodes,list) or not 1<=len(nodes)<=bounds.policy.limit('planNodes') or not isinstance(artifacts,list) or not 1<=len(artifacts)<=bounds.count:raise ValueError('Cache manifest count bounds exceeded')
            for node in nodes:
                if not isinstance(node,dict) or set(node)!={'nodeID','cacheKey','status','reason'} or not isinstance(node['nodeID'],str) or len(node['nodeID'])>128 or node['nodeID'] in self.prior or not isinstance(node['cacheKey'],str) or re.fullmatch('[0-9a-f]{64}',node['cacheKey']) is None:raise ValueError('Invalid cache node index')
                self.prior[node['nodeID']]=node
            seen=set();declared_bytes=0
            for entry in artifacts:
                if not isinstance(entry,dict):raise ValueError('Invalid cache artifact inventory')
                name=entry.get('path')
                if not isinstance(name,str) or Path(name).name!=name or name in ('','.','..') or name in seen:raise ValueError('Invalid cache artifact path')
                seen.add(name);size=entry.get('bytes')
                if entry.get('kind') in ('audioSamples','preparedTensor','imagePixels'):self.asset_entries[entry.get('sha256')]=entry
                if type(size) is not int or not 0<=size<=bounds.file(entry):raise ValueError('Cache artifact size bound exceeded')
                declared_bytes+=size
                if declared_bytes>bounds.total:raise ValueError('Cache total artifact bytes exceeded')
                if entry.get('kind') in ('table','state'):
                    pair=(entry.get('nodeID'),entry.get('port'))
                    if not all(isinstance(v,str) for v in pair) or pair in self.entries:raise ValueError('Invalid cache port index')
                    self.entries[pair]=entry
            if measured is not None and measured!=sum(entry['bytes'] for entry in artifacts if entry.get('kind') in ('table','state','tableSegment')):raise ValueError('Retained cache allowance evidence differs from artifact inventory')
            if manifest.get('tableStorageVersion')==1:
                if __package__:from .dataflow_storage import verify_regular_file
                else:from dataflow_storage import verify_regular_file
                for entry in artifacts:
                    if entry.get('kind')=='tableSegment':verify_regular_file(self.root/entry['path'],entry['bytes'],entry['sha256'],check_cancelled)
            self._read(next((e for e in artifacts if e.get('path')=='dataflow.json'),{}),limit=2_000_000)
            oldgraph=json.loads(self.verified_files['dataflow.json'])
            if df.digest(oldgraph)!=manifest.get('graphDigest'):raise ValueError('Cache graph integrity mismatch')
            oldnodes,oldinputs,_=df.validate(oldgraph)
            if __package__:from .image_artifacts import validate_inventory
            else:from image_artifacts import validate_inventory
            try:
                validate_inventory(manifest,self.root,check_cancelled)
                if __package__:from .media_artifacts import validate_inventory as validate_media_inventory
                else:from media_artifacts import validate_inventory as validate_media_inventory
                validate_media_inventory(manifest,self.root,check_cancelled)
            except (ValueError,OSError) as error:raise ValueError('Cache image/artifact integrity check failed: '+str(error)) from error
            if __package__:from .dataflow_storage import validated_execution_coverage
            else:from dataflow_storage import validated_execution_coverage
            plan_entry=next((entry for entry in artifacts if entry.get('path')=='preparation-plan.json'),None)
            if plan_entry is not None:self._read(plan_entry,limit=2_000_000)
            self.prior_coverage=validated_execution_coverage(manifest,self.verified_files)
            if set(oldnodes)!=set(self.prior):raise ValueError('Cache graph/node index mismatch')
            for ident,(operator,_) in oldnodes.items():
                if {port for node,port in self.entries if node==ident}!=df.configured_ports(*oldnodes[ident],oldinputs[ident])[2]:raise ValueError('Cache graph/output index mismatch')

    def media_asset(self,pin):
        if self.root is None:return None
        entry=self.asset_entries.get(pin)
        if entry is None:return None
        return self._read(entry,retain=False),entry.get('media',entry.get('image'))

    def _read(self,entry,limit=50_000_000,*,retain=True):
        name=entry.get('path')
        if not isinstance(name,str) or Path(name).name!=name:raise ValueError('Cache graph artifact is missing')
        if name in self.verified_files:return self.verified_files[name]
        raw=bounded_regular_read(self.root/name,limit);self.read_bytes+=len(raw)
        if self.read_bytes>MAX_BYTES:raise ValueError('Cache read bytes exceeded')
        if len(raw)!=entry.get('bytes') or hashlib.sha256(raw).hexdigest()!=entry.get('sha256'):raise ValueError('Cache artifact integrity check failed')
        if retain:self.verified_files[name]=raw
        return raw

    def set_demand(self, demand_plan):
        if __package__:from .preparation_demand import DemandPlan
        else:from preparation_demand import DemandPlan
        if not isinstance(demand_plan,DemandPlan):raise ValueError('Cache demand must be an immutable preparation demand plan')
        self.demand_plan=demand_plan

    def _coverage(self,node,coverage=None):
        if __package__:from .preparation_demand import normalize_sampling,is_complete_sampling
        else:from preparation_demand import normalize_sampling,is_complete_sampling
        if coverage is None:
            coverage={'completeness':'complete','sampling':{'schemaVersion':1,'mode':'all'}} if self.demand_plan is None else {
                'completeness':'complete' if self.demand_plan.is_complete(node) else 'sampled','sampling':self.demand_plan.policy_for(node)}
        if not isinstance(coverage,dict) or set(coverage)!={'completeness','sampling'} or coverage['completeness'] not in ('complete','sampled'):
            raise ValueError('Invalid cache execution coverage identity')
        sampling=normalize_sampling(coverage['sampling'])
        if (coverage['completeness']=='complete')!=is_complete_sampling(sampling):
            raise ValueError('Cache completeness differs from effective sampling policy')
        return {'schemaVersion':1,'selection':'sourceOrderPrefix','completeness':coverage['completeness'],'sampling':sampling}

    def key(self,node,operator,config,args,predecessors,graph_digest,source=None,coverage=None):
        effective=self._coverage(node,coverage)
        self.current_coverage[node]=effective
        if __package__:from .table_identity import enabled as deferred_identity
        else:from table_identity import enabled as deferred_identity
        if self.policy=='off' and deferred_identity():
            # Domain-separated non-reusable bookkeeping, never a content cache key.
            return df.digest(['kwip.cache.disabled.v1',self.implementation,graph_digest,node,operator,config,predecessors,effective])
        inputs={port:table_identity(value) if isinstance(value,df.Table) else {'stateDigest':value['contentDigest']} for port,value in args.items()}
        identity={'schemaVersion':1,'implementationDigest':self.implementation,'nodeID':node,'operator':df.PREFIX+operator,'config':config,'inputs':inputs,'predecessors':predecessors,'executionCoverage':effective}
        if source is not None:identity['source']=table_identity(source)
        if operator.endswith('Fit') or operator in ('transform','preprocess','textEncode'):identity['graphDigest']=graph_digest
        return df.digest(identity)

    def lookup(self,node,key,operator,args,graph_digest,config=None):
        if self.policy!='reuse' or operator in ('sourceCSV','outputDelimited','extractText','transcribeAudio','classifyAudio','decide') or (operator=='prepareMedia' and (config.get('outputMode')=='image' or config.get('representation','samples')!='samples')):return None
        if operator=='prepareMedia' and any(col['name']==config.get('inputColumn') and col['dtype']=='tensor' for col in args['table'].columns):return None
        previous=self.prior.get(node)
        if previous is None or previous['cacheKey']!=key:return None
        effective=self.current_coverage.get(node,self._coverage(node))
        if self.prior_coverage is not None:
            prior=self.prior_coverage['nodes'][node]
            prior_policy={'schemaVersion':1,'mode':'all'} if prior['completeness']=='complete' else self.prior_coverage['sampling']
            if effective!=self._coverage(node,{'completeness':prior['completeness'],'sampling':prior_policy}):return None
        elif effective['completeness']!='complete':return None
        outputs={}
        # Retain the verified prior artifact order. Sorting multi-output ports here
        # changes node-N filenames on cache reuse (e.g. test before training),
        # despite identical captured tables and observation memberships.
        declared=df.configured_ports(operator,config,args)[2]
        for ident,port in self.entries:
            if ident!=node or port not in declared:continue
            entry=self.entries[(node,port)]
            if entry['kind']=='table' and (entry['bytes']>1_000_000 or entry.get('tableStorageVersion')==1):
                if __package__:from .dataflow_table_reader import read_table_artifact
                else:from dataflow_table_reader import read_table_artifact
                value=read_table_artifact(self.root/entry['path'],entry['sha256'],check_cancelled=self.check_cancelled)
                self.read_bytes+=entry['bytes']
                if self.read_bytes>MAX_BYTES:raise ValueError('Cache read bytes exceeded')
            else:
                raw=self._read(entry,retain=False)
                if entry['kind']=='table':value=table_from_artifact(json.loads(raw))
                else:value=df.load_state(raw.decode())
            if (port=='state')!=(entry['kind']=='state'):raise ValueError('Cache port type mismatch')
            outputs[port]=value
        if operator=='sequenceWindows':
            if __package__:from .sequence_windows import verify_output
            else:from sequence_windows import verify_output
            verify_output(args,config,graph_digest,outputs,self.check_cancelled)
            return outputs
        if operator=='textEncode':
            if __package__:from .text_encoding import verify_output
            else:from text_encoding import verify_output
            verify_output(args,config,graph_digest,outputs,self.check_cancelled)
            return outputs
        if operator=='preprocess':
            if __package__:from .dataflow_preprocess import verify_output
            else:from dataflow_preprocess import verify_output
            verify_output(args,config,graph_digest,outputs,self.check_cancelled)
            return outputs
        if operator=='formula':
            if __package__:from .dataflow_formula import compile_plan
            else:from dataflow_formula import compile_plan
            expected_columns,_=compile_plan(args['table'].columns,config)
            result=outputs.get('table');source=args['table']
            if not isinstance(result,df.Table) or list(result.columns)!=expected_columns or len(result.rows)!=len(source.rows) or len(result.row_ids)!=len(source.row_ids):raise ValueError('Cached Formula schema/count differs')
            for actual,expected in zip(result.row_ids,source.row_ids):
                self.check_cancelled()
                if actual!=expected:raise ValueError('Cached Formula observation order differs')
        if operator=='partition':
            if __package__:from .dataflow_partition_tool import verify_output
            else:from dataflow_partition_tool import verify_output
            verify_output(args['table'],config,outputs,self.check_cancelled)
            return outputs
        if operator in ('groupSplit','temporalSplit'):
            df._partition_module().verify_partition_output(args['table'],operator,config,outputs)
            return outputs
        if operator in ('sort','unique','reshapeTable','splitText'):
            module=df._table_tool_module(operator)
            module.verify_output(args['table'],config,outputs if operator=='unique' else outputs['table'],self.check_cancelled)
            return outputs
        if operator=='prepareMedia':
            if __package__:from .prepare_media import verify_output
            else:from prepare_media import verify_output
            verify_output(args['table'],config,outputs['table'],self.check_cancelled)
            return outputs
        if operator=='union':
            if __package__:from .dataflow_union import verify_output
            else:from dataflow_union import verify_output
            verify_output(args,config,outputs['table'],self.check_cancelled)
            return outputs
        if operator=='aggregate':
            df._aggregate_module().verify_aggregate_output(args['table'],config,outputs['table'])
            return outputs
        if operator=='join':
            df._join_module().verify_join_output(args['left'],args['right'],config,outputs['table'])
            return outputs
        tables=[value for value in args.values() if isinstance(value,df.Table)]
        for value in outputs.values():
            if isinstance(value,df.Table):
                if len(tables)!=1 or value.source_digest!=tables[0].source_digest:raise ValueError('Cached table source/membership lineage mismatch')
                if isinstance(tables[0].row_ids,df.StoredIDs):
                    source_ids=tables[0].row_ids
                    if any(not source_ids.contains(identity) for identity in value.row_ids):raise ValueError('Cached table source/membership lineage mismatch')
                elif not set(value.row_ids)<=set(tables[0].row_ids):raise ValueError('Cached table source/membership lineage mismatch')
                if operator!='split' and (value.role!=tables[0].role or value.partition_digest!=tables[0].partition_digest):raise ValueError('Cached table partition lineage mismatch')
                expected=tables[0].fitted_states+((args['state']['contentDigest'],) if operator=='transform' else ())
                if value.fitted_states!=expected:raise ValueError('Cached table fitted-state lineage mismatch')
            elif value['operator']!=df.PREFIX+operator or value['graphDigest']!=graph_digest or value['fitDataVersion']!=args['training'].data_version or value['fitEvidence']!=args['training'].evidence() or value.get('inputFittedStateDigests',[])!=list(args['training'].fitted_states):raise ValueError('Cached fit-state lineage mismatch')
        if operator=='split':
            a=outputs['training'];b=outputs['validation'];source=tables[0]
            if a.role!='training' or b.role!='validation' or a.partition_digest!=b.partition_digest or set(a.row_ids)&set(b.row_ids) or set(a.row_ids)|set(b.row_ids)!=set(source.row_ids):raise ValueError('Cached split is not a disjoint complete partition')
        return outputs

    def charge_materialized(self,size):
        if self.policy=='off':return
        if self.materialized_bytes+size>self.materialized_limit:raise ValueError('Dataflow cache materialization exceeds its byte allowance; increase the allowance or reduce prepared outputs')
        self.materialized_bytes+=size

    def record(self,node,key,reused,source=False):
        status='reused' if reused else 'executed'
        reason='verifiedHit' if reused else 'sourceRevalidated' if source else 'disabled' if self.policy=='off' else 'identityChanged' if node in self.prior else 'notMaterialized'
        self.nodes.append({'nodeID':node,'cacheKey':key,'status':status,'reason':reason})

    def report(self):
        value={'schemaVersion':1,'policy':self.policy,'implementationDigest':self.implementation,'sourceManifestSHA256':self.source_pin,'nodes':self.nodes if self.policy!='off' else [],'executedNodeIDs':[n['nodeID'] for n in self.nodes if n['status']=='executed'],'reusedNodeIDs':[n['nodeID'] for n in self.nodes if n['status']=='reused']}
        if self.policy!='off':value.update(materializedBytesLimit=self.materialized_limit,materializedBytes=self.materialized_bytes)
        return value
