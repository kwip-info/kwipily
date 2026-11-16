"""Shared standard-library verification of immutable owned tool inputs."""
import hashlib
import json
import os
from pathlib import Path
import stat
if __package__:
    from .worker_authorization import require, directory, read_regular, signature, object_json, hash_value
else:
    from worker_authorization import require, directory, read_regular, signature, object_json, hash_value


def relative_parts(name):
    require(isinstance(name,str) and '\0' not in name, 'Invalid tool input path')
    parts=name.split('/')
    require(all(part and part not in ('.','..') and len(part.encode())<=255 for part in parts), 'Invalid tool path components')
    return parts


def same_json(left,right):
    """Numeric formatting may differ; booleans never impersonate numbers."""
    if type(left) in (int,float) and type(right) in (int,float):return left==right
    if type(left) is not type(right):return False
    if type(left) is dict:return set(left)==set(right) and all(same_json(left[key],right[key])for key in left)
    if type(left) is list:return len(left)==len(right) and all(same_json(a,b)for a,b in zip(left,right))
    return left==right


def verify_pinned_file(root,name,entry,cache):
    parts=relative_parts(name);fd=directory(str(Path(root).joinpath(*parts[:-1])))
    try:
        descriptor=os.open(parts[-1],os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK|os.O_CLOEXEC,dir_fd=fd)
        try:
            before=os.fstat(descriptor)
            require(stat.S_ISREG(before.st_mode) and before.st_uid==os.geteuid() and before.st_nlink==1
                and before.st_mode & 0o022==0 and before.st_size==entry['bytes'],'Unsafe tool input file')
            named=os.stat(parts[-1],dir_fd=fd,follow_symlinks=False)
            require(signature(before)==signature(named),'Preview named input differs from descriptor')
            prior=cache.get(name)
            if prior is not None:
                require(prior[:2]==signature(before)[:2],'Captured tool input file was replaced')
                if prior==signature(before):return
            digest=hashlib.sha256();count=0
            while True:
                chunk=os.read(descriptor,min(65536,entry['bytes']+1-count))
                if not chunk:break
                count+=len(chunk);require(count<=entry['bytes'],'Preview input grew');digest.update(chunk)
            after=os.fstat(descriptor);named=os.stat(parts[-1],dir_fd=fd,follow_symlinks=False)
            require(signature(before)==signature(after)==signature(named) and count==entry['bytes']
                and digest.hexdigest()==entry['sha256'],'Captured tool input changed')
            cache[name]=signature(after)
        finally:os.close(descriptor)
    finally:os.close(fd)


def verify_completed_inventory(root,prefix,inventory,manifest_pin,graph_digest=None,*,admission_policy=None):
    fd=directory(str(Path(root)/prefix))
    try:raw=read_regular(fd,'manifest.json',2000000)
    finally:os.close(fd)
    require(hashlib.sha256(raw).hexdigest()==manifest_pin,'Captured manifest changed')
    manifest=object_json(raw)
    require(type(manifest.get('schemaVersion')) is int and manifest['schemaVersion']==1
        and manifest.get('kind')=='dataflowExecution' and manifest.get('status')=='completed','Invalid completed preparation manifest')
    hash_value(manifest.get('graphDigest'))
    if graph_digest is not None:require(manifest['graphDigest']==graph_digest,'Prepared graph digest differs')
    if __package__:from .preparation_artifact_limits import ArtifactLimits
    else:from preparation_artifact_limits import ArtifactLimits
    limits=ArtifactLimits(manifest,admission_policy=admission_policy)
    entries=manifest.get('artifacts');require(isinstance(entries,list) and 1<=len(entries)<=limits.count,'Invalid preparation artifact count')
    expected={prefix+'/manifest.json'};total=0
    for entry in entries:
        require(isinstance(entry,dict),'Invalid preparation artifact entry')
        name=entry.get('path');require(isinstance(name,str) and len(relative_parts(name))==1 and name!='manifest.json','Invalid prepared artifact path')
        relative=prefix+'/'+name
        require(relative not in expected and relative in inventory,'Missing or repeated prepared artifact')
        require(type(entry.get('bytes')) is int and 0<=entry['bytes']<=limits.file(entry)
            and entry['bytes']==inventory[relative]['bytes'] and entry.get('sha256')==inventory[relative]['sha256'],
            'Prepared artifact pin differs from captured inventory')
        total+=entry['bytes'];expected.add(relative)
    require(total<=limits.total,'Prepared artifact total exceeds limit')
    require(expected=={name for name in inventory if name.startswith(prefix+'/')},'Extra prepared input outside pinned manifest')
    return manifest


def validate_bag_arguments(options, source_config=None):
    """Framework-free preclaim transport checks; never expands or runs a model.

    The post-guard bag_input_mapping helper enforces the same mapping contract.
    Preclaim code must not import it: its typed coordinate imports require Torch.
    """
    raw=options.get('--bag-input-mapping-json')
    require(type(raw) is str and len(raw.encode('utf-8'))<=32768,'Invalid bounded bag mapping transport')
    mapping=object_json(raw)
    require(raw==json.dumps(mapping,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False),'Bag mapping must be canonical compact UTF-8 JSON')
    keys=set(mapping)
    require({'schemaVersion','indicesColumns'}<=keys<= {'schemaVersion','indicesColumns','weightColumns','lengthColumn'}
        and type(mapping['schemaVersion']) is int and mapping['schemaVersion']==1,'Invalid bag mapping schema')
    def name(value):return type(value) is str and bool(value.strip()) and len(value)<=200
    names=[]
    for key in ('indicesColumns','weightColumns'):
        if key not in mapping:continue
        values=mapping[key]
        require(type(values) is list and 1<=len(values)<=128 and all(name(value) for value in values),'Invalid ordered bag mapping columns')
        require(len(values)==len(mapping['indicesColumns']),'Bag mapping capacities differ')
        names+=values
    if 'lengthColumn' in mapping:
        require(name(mapping['lengthColumn']),'Invalid bag length column');names.append(mapping['lengthColumn'])
    require(len(names)==len(set(names)),'Bag mapping roles overlap')
    prepared={'--prepared-dataflow','--prepared-manifest-sha256','--prepared-graph-digest','--prepared-training-node','--prepared-training-port','--prepared-validation-node','--prepared-validation-port'}
    require(prepared<=set(options) and not {'--feature-columns-json','--dataset-csv','--sample-limit','--split-seed','--validation-percent'}&set(options),'Bag mapping requires prepared partitions without ordinary data controls')
    for key in ('--prepared-manifest-sha256','--prepared-graph-digest'):hash_value(options[key])
    if '--objective-target-mapping-json' in options:
        require('--target-column' not in options,'Bag objective cannot carry a binary target')
        targets=object_json(options['--objective-target-mapping-json'])
        require(targets.get('inputColumns')==mapping['indicesColumns'],'Objective input columns differ from bag indices')
        target_names=[column for role in targets['roles'] for column in role['columns']]
    else:
        require(name(options.get('--target-column')),'Bag mapping requires its binary target column')
        target_names=[options['--target-column']]
    require(not set(names)&set(target_names),'Bag source and target roles overlap')
    if source_config is not None:
        config=source_config
        require(type(config) is dict and set(config)=={'layout','maximum_items_per_bag','per_sample_weights','include_last_offset'},'Invalid captured bag source controls')
        require(type(config['layout']) is str and config['layout'] in ('fixed','packed')
            and type(config['maximum_items_per_bag']) is int and 1<=config['maximum_items_per_bag']<=128
            and all(type(config[key]) is bool for key in ('per_sample_weights','include_last_offset')),'Invalid captured bag source configuration')
        expected={'schemaVersion','indicesColumns'}|({'weightColumns'}if config['per_sample_weights']else set())|({'lengthColumn'}if config['layout']=='packed'else set())
        require(keys==expected and len(mapping['indicesColumns'])==config['maximum_items_per_bag'],'Bag mapping differs from actual source roles/capacity')
    return mapping


def verify_bag_graph_inputs(graph,registry,source,options):
    """Check captured roles/configuration only; full semantics run after guard."""
    require(type(graph) is dict and type(graph.get('schemaVersion')) is int and graph['schemaVersion']==2,'Bag graph requires schema2')
    nodes=graph.get('nodes');edges=graph.get('edges')
    require(type(nodes) is list and 1<=len(nodes)<=128 and type(edges) is list and len(edges)<=256,'Invalid bounded captured bag graph')
    require(all(type(node) is dict and type(node.get('id')) is str for node in nodes),'Invalid bag node identities')
    by_id={node['id']:node for node in nodes};require(len(by_id)==len(nodes),'Repeated bag graph node identity')
    sources=[node for node in nodes if node.get('type') in ('input','index_input','bag_input')]
    require(len(sources)==1 and sources[0]['type']=='bag_input','Bag mapping requires the actual single bag source')
    actual=sources[0];mapping=validate_bag_arguments(options,actual.get('config'))
    config=actual['config'];roles=['indices']+(['offsets']if config['layout']=='packed'else[])+(['per_sample_weights']if config['per_sample_weights']else[])
    require(type(registry) is dict and type(registry.get('schemaVersion')) is int and registry['schemaVersion']==2 and type(registry.get('components')) is list,'Invalid captured bag registry')
    components=registry['components'];require(all(type(item) is dict and type(item.get('id')) is str for item in components),'Invalid captured component identities')
    declared={item['id']:item for item in components};require(len(declared)==len(components),'Duplicate captured registry component')
    require('bag_input' in declared and 'embedding_bag' in declared and set(roles)<=set(declared['bag_input'].get('outputs',[])),'Captured registry lacks enabled bag roles')
    if __package__:from .embedding_bag_configuration import normalize_embedding_bag,configured_input_ports
    else:from embedding_bag_configuration import normalize_embedding_bag,configured_input_ports
    consumers=[node for node in nodes if node.get('type')=='embedding_bag'];require(consumers,'Bag source requires a real consumer')
    for node in consumers:
        settings=normalize_embedding_bag(node.get('config'))
        require((settings['layout'],settings['per_sample_weights'],settings['include_last_offset'])==(config['layout'],config['per_sample_weights'],config['include_last_offset']),'Bag consumer differs from source controls')
        enabled=configured_input_ports(settings)
        require(set(enabled)<=set(declared['embedding_bag'].get('inputs',[])),'Captured registry lacks bag consumer roles')
        incoming=[edge for edge in edges if type(edge) is dict and edge.get('target')==node['id']]
        require(len(incoming)==len(enabled),'Missing or extra bag consumer coordinate role')
        for role in enabled:
            selected=[edge for edge in incoming if edge.get('targetPort')==role]
            require(len(selected)==1 and selected[0].get('source')==actual['id'] and selected[0].get('sourcePort')==role,'Bag role must connect its matching actual source role')
    require(all(type(edge) is dict and (edge.get('source')!=actual['id'] or edge.get('target') in {node['id']for node in consumers} and edge.get('sourcePort') in roles)for edge in edges),'Bag coordinates cannot target ordinary components')
    if source is not None:
        require(type(source) is dict and same_json(source.get('expandedGraph'),graph),'Captured composite expanded graph differs from bag graph')
    elif 'compositionPlan' in graph:raise ValueError('Bag composition requires its captured composite source')
    return mapping


def _owned_json(path,limit):
    fd=directory(str(Path(path).parent))
    try:raw=read_regular(fd,Path(path).name,limit)
    finally:os.close(fd)
    return raw,object_json(raw)


def requires_bag_admission(options,graph_path):
    if '--bag-input-mapping-json' in options:return True
    # Ordinary historical routes retain their existing late graph diagnostics.
    try:_,graph=_owned_json(graph_path,2000000)
    except (ValueError,TypeError):return False
    nodes=graph.get('nodes')
    return type(nodes) is list and any(type(node) is dict and node.get('type')=='bag_input'for node in nodes)


def _runtime_registry_json(path):
    """Pinned runtime files may be root-owned in a store installation."""
    path=Path(path);parent=directory(str(path.parent))
    try:
        fd=os.open(path.name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK|os.O_CLOEXEC,dir_fd=parent)
        try:
            before=os.fstat(fd)
            require(stat.S_ISREG(before.st_mode) and before.st_size<=2000000 and before.st_mode&0o022==0,'Invalid bounded runtime registry')
            raw=b''
            while len(raw)<=2000000:
                chunk=os.read(fd,min(65536,2000001-len(raw)))
                if not chunk:break
                raw+=chunk
            after=os.fstat(fd);named=os.stat(path.name,dir_fd=parent,follow_symlinks=False)
            require(len(raw)<=2000000 and signature(before)==signature(after)==signature(named),'Runtime registry changed during admission')
        finally:os.close(fd)
    finally:os.close(parent)
    return raw,object_json(raw)


def verify_phase_bag_inputs(phase,options,identity,cache):
    """Bind workflow-phase paths to exact supported bag capture contracts."""
    phase=Path(phase);root=phase.parent
    require('--bag-input-mapping-json' in options,'Actual bag graph requires its captured bag mapping option')
    raw,capture=_owned_json(root/'capture.json',8000000)
    pin=hashlib.sha256(raw).hexdigest()
    require(cache.setdefault('$bagCaptureSHA256',pin)==pin,'Bag workflow capture changed after admission')
    # Match WorkflowArchiveCodec's exact supported triples, not an open-ended
    # version range. JSON booleans/floats must not pass integer equality.
    header=tuple(capture.get(key) for key in ('schemaVersion','compilerContractVersion','captureFormatVersion'))
    require(all(type(value) is int for value in header)
        and header in ((12,11,7),(13,12,8),(14,12,8),(15,13,8),(16,14,8),(17,15,8),(18,15,8),(13,16,8),(14,16,8),(15,16,8),(16,16,8),(17,16,8),(18,16,8),(19,16,8)),
        'Bag workflow requires a supported exact capture/compiler/format contract')
    for key in ('projectID','directoryID','runID','requestIdentity'):require(capture.get(key)==identity[key],'Bag capture differs from retained phase identity')
    plan=capture.get('plan');require(type(plan) is dict,'Missing captured bag plan')
    _,graph=_owned_json(phase/'request.json',2000000)
    require(same_json(plan.get('modelGraph'),graph),'Bag phase graph differs from captured plan')
    for key,filename,limit in [('modelRegistry','model-registry.json',2000000),('runtimeEvidence','runtime-evidence.json',256000)]:
        entry=capture.get(key);require(type(entry) is dict and set(entry)=={'path','bytes','sha256'} and entry['path']==filename,'Invalid captured bag input pin')
        require(type(entry['bytes']) is int and 0<=entry['bytes']<=limit,'Invalid captured bag input size')
        hash_value(entry['sha256'])
        verify_pinned_file(root,filename,entry,cache)
    _,registry=_owned_json(root/'model-registry.json',2000000)
    _,evidence=_owned_json(root/'runtime-evidence.json',256000)
    registry_path=options.get('--registry');require(type(registry_path) is str,'Bag phase requires its captured runtime registry')
    registry_raw,runtime_registry=_runtime_registry_json(registry_path)
    require(evidence.get('workerFiles',{}).get('components.json')==hashlib.sha256(registry_raw).hexdigest() and same_json(registry,runtime_registry),'Bag runtime registry differs from captured evidence')
    source=None
    if '--composite-source' in options:
        require(options['--composite-source']==str(phase/'composite-source.json'),'Bag phase source path differs')
        source_raw,source=_owned_json(phase/'composite-source.json',2000000)
        require(hashlib.sha256(source_raw).hexdigest()==options.get('--composite-source-sha256') and same_json(source,plan.get('modelProvenance')),'Bag source differs from captured plan/pin')
    else:require(plan.get('modelProvenance') is None,'Captured bag provenance was omitted from launch')
    mapping=verify_bag_graph_inputs(graph,registry,source,options)
    require(plan.get('bagInput')==mapping,'Bag launch mapping differs from captured plan')
    for role in ('training','validation'):
        require(plan.get(role)==dict(node=options['--prepared-'+role+'-node'],port=options['--prepared-'+role+'-port']),'Bag selected preparation endpoint differs from captured plan')
    if '--objective-target-mapping-json' in options:
        recipe=plan.get('lossRecipe');require(type(recipe) is dict and recipe.get('targetMapping')==object_json(options['--objective-target-mapping-json']) and recipe.get('configuration')==object_json(options['--objective-recipe-json']) and plan.get('targetColumn') is None and plan.get('objectiveScalarization','mean')==options.get('--objective-scalarization','mean'),'Bag objective mapping/recipe/scalarization differs from captured plan')
    else:require(plan.get('targetColumn')==options.get('--target-column') and plan.get('lossRecipe') is None,'Bag binary target differs from captured plan')
    preparation=root/'preparation/artifacts'
    require(options['--prepared-dataflow']==str(preparation/'manifest.json'),'Bag preparation path differs from workflow phase')
    manifest_raw,manifest=_owned_json(preparation/'manifest.json',2000000)
    inventory={'preparation/artifacts/manifest.json':dict(bytes=len(manifest_raw),sha256=hashlib.sha256(manifest_raw).hexdigest())}
    if __package__:from .preparation_artifact_limits import request_policy,ArtifactLimits
    else:from preparation_artifact_limits import request_policy,ArtifactLimits
    policy=request_policy(identity)
    limits=ArtifactLimits(manifest,admission_policy=policy)
    entries=manifest.get('artifacts');require(type(entries) is list and 1<=len(entries)<=limits.count,'Invalid bag preparation inventory')
    for entry in entries:
        require(type(entry) is dict and len(relative_parts(entry.get('path')))==1,'Invalid bag preparation artifact')
        name='preparation/artifacts/'+entry['path'];require(name not in inventory,'Repeated bag preparation artifact')
        inventory[name]=entry
    verify_completed_inventory(root,'preparation/artifacts',inventory,options['--prepared-manifest-sha256'],options['--prepared-graph-digest'],admission_policy=policy)
    for name,entry in inventory.items():verify_pinned_file(root,name,entry,cache)
