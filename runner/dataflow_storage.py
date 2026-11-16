"""Bounded regular-file reads for portable dataflow artifacts (no framework imports)."""
import os
import stat
from pathlib import Path


def bounded_regular_read(path,limit):
    try:descriptor=os.open(Path(path),os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    except OSError as error:raise ValueError('Dataflow artifact must be a readable regular file') from error
    try:
        info=os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):raise ValueError('Dataflow artifact must be a regular file')
        if info.st_size>limit:raise ValueError('Dataflow artifact exceeds size bound')
        with os.fdopen(descriptor,'rb') as stream:
            descriptor=None;content=stream.read(limit+1)
    finally:
        if descriptor is not None:os.close(descriptor)
    if len(content)>limit:raise ValueError('Dataflow artifact exceeds size bound')
    return content


def completed_inventory(path, expected_sha256=None, *, check_cancelled=lambda:None, lazy=False, allow_sample=False, selected_nodes=None):
    """Read a complete pinned inventory as bytes; never import archived programs."""
    import hashlib
    import json
    import re
    if expected_sha256 is not None and (not isinstance(expected_sha256,str) or re.fullmatch('[0-9a-f]{64}',expected_sha256) is None):raise ValueError('Manifest pin must be a lowercase SHA-256 digest')
    check_cancelled()
    path=Path(path).absolute()
    raw=bounded_regular_read(path,2_000_000)
    check_cancelled()
    pin=hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and pin!=expected_sha256:raise ValueError('Prepared data differs from the selected manifest; select preparation again')
    manifest=json.loads(raw)
    if not isinstance(manifest,dict) or type(manifest.get('schemaVersion')) is not int or manifest.get('schemaVersion')!=1 or manifest.get('kind')!='dataflowExecution' or manifest.get('status')!='completed':raise ValueError('Manifest must describe completed dataflow execution')
    if __package__:from .preparation_artifact_limits import ArtifactLimits
    else:from preparation_artifact_limits import ArtifactLimits
    bounds=ArtifactLimits(manifest)
    entries=manifest.get('artifacts')
    if not isinstance(entries,list) or not 1<=len(entries)<=bounds.count:raise ValueError('Invalid prepared artifact inventory')
    files={'manifest.json':raw};total=0
    for entry in entries:
        check_cancelled()
        if not isinstance(entry,dict):raise ValueError('Invalid prepared artifact inventory entry')
        name=entry.get('path');size=entry.get('bytes')
        if not isinstance(name,str) or Path(name).name!=name or name in ('','.','..','manifest.json') or name in files:raise ValueError('Invalid or duplicate prepared artifact path')
        if type(size) is not int or not 0<=size<=bounds.file(entry):raise ValueError('Invalid artifact size bound')
        total+=size
        if total>bounds.total:raise ValueError('Prepared artifacts exceed total size bound')
        if lazy:
            verify_regular_file(path.parent/name,size,entry.get('sha256'),check_cancelled)
            files[name]=None
            continue
        content=bounded_regular_read(path.parent/name,size)
        check_cancelled()
        if len(content)!=size or hashlib.sha256(content).hexdigest()!=entry.get('sha256'):raise ValueError('Prepared artifact hash/digest or size mismatch')
        check_cancelled()
        files[name]=content
    check_cancelled()
    result_files=VerifiedArtifactFiles(path.parent,entries,raw,check_cancelled) if lazy else files
    if __package__:from .image_artifacts import validate_inventory
    else:from image_artifacts import validate_inventory
    validate_inventory(manifest,path.parent,check_cancelled)
    if __package__:from .media_artifacts import validate_inventory as validate_media_inventory
    else:from media_artifacts import validate_inventory as validate_media_inventory
    validate_media_inventory(manifest,path.parent,check_cancelled)
    if manifest.get('observationRetention') is not None:
        if manifest['observationRetention']!='requiredOnly':raise ValueError('Invalid preparation observation retention')
        if __package__:from . import dataflow
        else:import dataflow
        graph=json.loads(result_files['dataflow.json']);nodes,inputs,_=dataflow.validate(graph,historical=True)
        if dataflow.digest(graph)!=manifest['graphDigest']:raise ValueError('Preparation graph identity differs')
        selected=manifest.get('requiredOutputNodes')
        if not isinstance(selected,list) or len(selected)>128 or any(not isinstance(node,str) or node not in nodes for node in selected) or len(selected)!=len(set(selected)):raise ValueError('Invalid required preparation node selection')
        required=set(selected)|(set(nodes)-{edge['from']['node'] for edge in graph['edges']})
        cache=manifest.get('cache',{}).get('policy','off')!='off'
        expected={(node,port) for node,(operator,config) in nodes.items() for port in dataflow.configured_ports(operator,config,inputs[node])[2] if port=='state' or cache or node in required or operator=='view'}
        actual=[(entry.get('nodeID'),entry.get('port')) for entry in entries if entry.get('kind') in ('table','state')]
        if len(actual)!=len(set(actual)) or set(actual)!=expected or any(entry.get('kind')=='previews' or entry.get('path')=='previews.json' for entry in entries):raise ValueError('Required preparation outputs are missing or optional observations retained')
    if allow_sample:
        validated_execution_coverage(manifest,result_files)
    else:
        require_complete_coverage(manifest,result_files,nodes=selected_nodes)
    return manifest,result_files,pin


def verify_regular_file(path, size, expected_sha256, check_cancelled=lambda:None):
    """Hash a pinned regular file without retaining its bytes."""
    import hashlib
    try:descriptor=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    except OSError as error:raise ValueError('Dataflow artifact must be a readable regular file') from error
    try:
        before=os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size!=size:raise ValueError('Prepared artifact hash/digest or size mismatch')
        hasher=hashlib.sha256();count=0
        while True:
            check_cancelled();block=os.read(descriptor,min(262144,size+1-count))
            if not block:break
            count+=len(block);hasher.update(block)
            if count>size:raise ValueError('Prepared artifact exceeds size bound')
        after=os.fstat(descriptor)
        if (count!=size or hasher.hexdigest()!=expected_sha256 or before.st_size!=after.st_size
            or before.st_mtime_ns!=after.st_mtime_ns or before.st_ctime_ns!=after.st_ctime_ns):raise ValueError('Prepared artifact hash/digest or size mismatch')
    finally:os.close(descriptor)


from collections.abc import Mapping
class VerifiedArtifactFiles(Mapping):
    """Verified inventory; bytes are fetched one file at a time without retention."""
    def __init__(self, root, entries, manifest, check):
        self.root=root;self.entries={entry['path']:entry for entry in entries};self.manifest=manifest;self.check=check
    def __iter__(self):return iter(['manifest.json',*self.entries])
    def __len__(self):return len(self.entries)+1
    def __contains__(self,name):return name=='manifest.json' or name in self.entries
    def __getitem__(self, name):
        import hashlib
        if name=='manifest.json':return self.manifest
        entry=self.entries[name];self.check()
        content=bounded_regular_read(self.root/name,entry['bytes'])
        if len(content)!=entry['bytes'] or hashlib.sha256(content).hexdigest()!=entry['sha256']:raise ValueError('Prepared artifact hash/digest or size mismatch')
        return content
    def path_for(self,name):return self.root/name


def admit_model_ports(files, manifest, ports):
    """Refuse oversized selected inputs before any full JSON/tensor allocation."""
    if __package__:from .dataflow_table_reader import read_table_artifact
    else:from dataflow_table_reader import read_table_artifact
    require_complete_coverage(manifest,files,nodes={node for node,_ in ports})
    seen=set()
    for entry in manifest['artifacts']:
        if entry.get('kind') not in ('table','state'):continue
        identity=(entry.get('nodeID'),entry.get('port'))
        if any(not isinstance(value,str) for value in identity) or identity in seen:raise ValueError('Duplicate or invalid prepared node/port')
        seen.add(identity)
    for entry in manifest['artifacts']:
        if entry.get('kind')!='table' or (entry.get('nodeID'),entry.get('port')) not in ports:continue
        if entry['bytes']<=1_000_000 and entry.get('tableStorageVersion')!=1:continue
        table=read_table_artifact(files.path_for(entry['path']),entry['sha256'])
        try:
            if len(table.rows)>100_000 or len(table.rows)*len(table.columns)>5_000_000:
                raise ValueError('This model currently requires an in-memory table within 100,000 rows and five million cells. Add Row Sample before splitting and training. Streaming model input is not available yet.')
        finally:
            table.rows.store.close();table.row_ids.store.close()


def prepared_table_document(files, entry):
    """Selected model inputs only: adapt pinned segments within existing model bounds.

    Analytics and live viewers keep their shared, paged representations. Legacy
    tensor loaders still own their typed/lineage checks; this changes physical
    decoding, never partition membership, fitted state or historical identity.
    """
    import json
    value=json.loads(files[entry['path']])
    if type(value) is not dict or value.get('kind')!='segmentedTable':return value
    if not isinstance(files,VerifiedArtifactFiles):raise ValueError('Segmented model input requires its pinned inventory context.')
    if __package__:
        from .dataflow_table_reader import read_table_artifact
        from .image_artifacts import table_schema_version
    else:
        from dataflow_table_reader import read_table_artifact
        from image_artifacts import table_schema_version
    table=read_table_artifact(files.path_for(entry['path']),entry['sha256'],check_cancelled=files.check)
    try:
        if len(table.rows)>100_000 or len(table.rows)*len(table.columns)>5_000_000:
            raise ValueError('This model currently requires an in-memory table within 100,000 rows and five million cells. Add Row Sample before splitting and training. Streaming model input is not available yet.')
        return {'schemaVersion':table_schema_version(table.columns),'kind':'table','columns':list(table.columns),
                'rows':[list(row) for row in table.rows],
                'lineage':{**table.evidence(),'observationIDs':list(table.row_ids),'dataVersion':table.data_version,
                           'partitionDigest':table.partition_digest,'fittedStateDigests':list(table.fitted_states),'diagnostics':list(table.diagnostics)}}
    finally:
        table.rows.store.close()
        if table.row_ids.store is not table.rows.store:table.row_ids.store.close()


def validated_execution_coverage(manifest, files=None):
    """Bind completeness claims to the verified immutable preparation snapshot.

    Historical v1 full preparations have no coverage field. A new demand-bearing
    plan must retain its coverage receipt, even when all branches are complete.
    """
    import hashlib
    import json
    coverage=manifest.get('executionCoverage')
    plan=None
    if files is not None and 'preparation-plan.json' in files:
        plan=json.loads(files['preparation-plan.json'])
        canonical=json.dumps(plan,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')
        if hashlib.sha256(canonical).hexdigest()!=manifest.get('preparationPlanDigest'):
            raise ValueError('Preparation plan identity differs from pinned manifest')
        if type(plan) is not dict or plan.get('schemaVersion') not in (1,2,3):
            raise ValueError('Unsupported preparation plan coverage version')
    if plan is not None and plan.get('schemaVersion')==3:
        if __package__:from .workload_policy import WorkloadPolicy
        else:from workload_policy import WorkloadPolicy
        captured=WorkloadPolicy.from_document(plan.get('workloadPolicy')).to_document()
        if manifest.get('workloadPolicy')!=captured:raise ValueError('Preparation workload differs from its pinned plan')
    elif manifest.get('workloadPolicy') is not None or plan is not None and 'workloadPolicy' in plan:
        raise ValueError('Historical preparation cannot claim a new workload policy')
    if coverage is None:
        if plan is not None and (plan.get('schemaVersion') in (2,3) or 'demand' in plan):
            raise ValueError('Preparation demand requires an explicit complete or sampled execution coverage receipt')
        return None
    if type(coverage) is not dict or coverage.get('schemaVersion')!=1 or type(coverage.get('schemaVersion')) is not int:
        raise ValueError('Invalid execution coverage schema')
    if coverage.get('status') not in ('complete','sampled'):
        raise ValueError('Execution coverage must disclose complete or sampled results')
    if files is None:
        complete=coverage.get('completeNodeIDs');sampled=coverage.get('sampledNodeIDs');entries=coverage.get('nodes')
        if (not isinstance(complete,list) or not isinstance(sampled,list) or not isinstance(entries,dict)
            or any(not isinstance(node,str) for node in complete+sampled)
            or len(set(complete+sampled))!=len(complete)+len(sampled)
            or set(entries)!=set(complete+sampled)
            or coverage['status']!=('sampled' if sampled else 'complete')
            or any(not isinstance(entry,dict) or entry.get('completeness')!=('complete' if node in complete else 'sampled') for node,entry in entries.items())):
            raise ValueError('Invalid complete or sampled execution coverage evidence')
        return coverage
    if plan is None:
        raise ValueError('Execution coverage requires its pinned preparation plan')
    if __package__:
        from . import dataflow
        from .preparation_demand import DemandPlan,build_demand_plan,validate_demand_plan,validate_coverage
    else:
        import dataflow
        from preparation_demand import DemandPlan,build_demand_plan,validate_demand_plan,validate_coverage
    graph=json.loads(files['dataflow.json'])
    if dataflow.digest(graph)!=manifest.get('graphDigest') or plan.get('graphDigest')!=manifest.get('graphDigest') or plan.get('graph')!=graph:
        raise ValueError('Execution coverage graph differs from its pinned preparation')
    if plan.get('schemaVersion') in (2,3):
        demand=DemandPlan(json.dumps(plan.get('demand'),sort_keys=True,separators=(',',':'),allow_nan=False))
        validate_demand_plan(demand,graph)
    else:
        if 'demand' in plan:
            raise ValueError('Historical preparation plans cannot introduce sampling demand')
        demand=build_demand_plan(graph)
    return validate_coverage(coverage,graph,demand_plan=demand)


def require_complete_coverage(manifest, files=None, *, nodes=None):
    """Admit complete selected ancestry; unrelated sampled branches remain inspectable."""
    coverage=validated_execution_coverage(manifest,files)
    if coverage is None:
        return
    if nodes is None:
        if coverage.get('status')!='complete' or coverage.get('sampledNodeIDs'):
            raise ValueError('Sampled preparation cannot satisfy a complete-data consumer; prepare its required branches completely')
        return
    if files is None:
        raise ValueError('Selected complete-data admission requires pinned graph and plan evidence')
    import json
    if __package__:from . import dataflow
    else:import dataflow
    graph=json.loads(files['dataflow.json'])
    known,inputs,_=dataflow.validate(graph)
    requested=set(nodes)
    if not requested or requested-set(known):
        raise ValueError('Complete-data admission requires known selected preparation nodes')
    pending=list(requested)
    while pending:
        for parent,_ in inputs[pending.pop()].values():
            if parent not in requested:
                requested.add(parent);pending.append(parent)
    if any(coverage['nodes'][node]['completeness']!='complete' for node in requested):
        raise ValueError('Sampled preparation or fitted state cannot satisfy this complete-data consumer; prepare the selected ancestry completely')
