"""Owned standalone preparation inputs; no data engines imported at admission."""
import os
from pathlib import Path

if __package__:
    from .worker_authorization import require, hash_value, directory, read_regular, object_json, sha, parse_bounded_options, PREPARATION, validate_preparation_cache_allowance,validate_model_input_nodes
    from .worker_input_contract import relative_parts, verify_pinned_file, verify_completed_inventory
    from .preparation_demand_configuration import parse_sampling_json,parse_complete_nodes_json
else:
    from worker_authorization import require, hash_value, directory, read_regular, object_json, sha, parse_bounded_options, PREPARATION, validate_preparation_cache_allowance,validate_model_input_nodes
    from worker_input_contract import relative_parts, verify_pinned_file, verify_completed_inventory
    from preparation_demand_configuration import parse_sampling_json,parse_complete_nodes_json

CONTROL_LIMITS={'input/graph.json':2000000,'input/bindings.json':1000000,'input/node-labels.json':65536}
OPTIONAL_CONTROL_LIMITS={'input/decision-models.json':1000000,'input/audio-models.json':1000000,'input/speech-models.json':1000000,'input/destinations.json':1000000,'input/preparation-models.json':1000000}
ALLOWED=(PREPARATION-{'--retain-partial-dataflow'})|{'--dataflow-node-labels','--dataflow-sampling','--dataflow-complete-nodes'}
REQUIRED={'--execute-dataflow','--source-bindings','--dataflow-node-labels','--output','--dataflow-cache-policy'}
REUSE={'--dataflow-reuse-manifest','--dataflow-reuse-manifest-sha256'}


if __package__:from .preparation_artifact_limits import request_policy,prepared_file_limit,inventory_limit
else:from preparation_artifact_limits import request_policy,prepared_file_limit,inventory_limit


def file_limit(name):
    parts=relative_parts(name)
    if len(parts)==4 and parts[:2]==['input','preparation-models']:
        hash_value(parts[2]);require(parts[3] in ('state.json','preparation.json'),'Unreviewed learned preparation model member')
        return 32000000 if parts[3]=='state.json' else 20000000
    if len(parts)==3 and parts[:2]==['input','ocr-pack']:
        if __package__:from .ocr_tables import ARCHIVES
        else:from ocr_tables import ARCHIVES
        require(parts[2] in ARCHIVES,'Unreviewed OCR pack member');return ARCHIVES[parts[2]][1]
    if 4<=len(parts)<=5 and parts[:2]==['input','decision-models']:
        hash_value(parts[2])
        if __package__:from .model_asset_operation_contract import allowed
        else:from model_asset_operation_contract import allowed
        require(allowed('/'.join(parts[3:])),'Unreviewed decision model member');return 2000000000 if name.endswith('.safetensors') else 8000000
    if len(parts)==4 and parts[:2]==['input','audio-models']:
        hash_value(parts[2])
        if __package__:from .model_asset_operation_contract import allowed
        else:from model_asset_operation_contract import allowed
        require(allowed(parts[3]),'Unreviewed audio classification model member');return 1400000000 if parts[3].endswith('.safetensors') else 8000000
    if len(parts)==4 and parts[:2]==['input','speech-models']:
        hash_value(parts[2])
        if __package__:from .model_asset_operation_contract import allowed
        else:from model_asset_operation_contract import allowed
        require(allowed(parts[3]),'Unreviewed speech model member');return 1400000000 if parts[3].endswith('.safetensors') else 8000000
    if name in CONTROL_LIMITS:return CONTROL_LIMITS[name]
    if name in OPTIONAL_CONTROL_LIMITS:return OPTIONAL_CONTROL_LIMITS[name]
    if len(parts)==3 and parts[:2]==['input','captured-tables'] and parts[2].endswith('.json'):
        hash_value(parts[2][:-5]);return 50000000
    if len(parts)==4 and parts[:2]==['input','sources']:
        hash_value(parts[2]);return 10000000
    if len(parts)==4 and parts[:2]==['input','file-collections'] and parts[3]=='manifest.json':
        hash_value(parts[2]);return 2000000
    if len(parts)==5 and parts[:2]==['input','file-collections'] and parts[3]=='members':
        hash_value(parts[2]);return 67108864
    if name=='input/reuse/manifest.json':return 2000000
    if len(parts)==3 and parts[:2]==['input','reuse']:return 50000000
    raise ValueError('Unsupported preparation input path')


def parse_contract(value,base=None):
    workload=request_policy(value);retained=workload.limit('retainedBytes') if workload else 200000000
    options=parse_bounded_options(value['arguments'],ALLOWED,maximum_argument_bytes=4096)
    require(REQUIRED<=set(options),'Missing preparation options')
    if '--dataflow-sampling' in options:parse_sampling_json(options['--dataflow-sampling'])
    if '--dataflow-complete-nodes' in options:parse_complete_nodes_json(options['--dataflow-complete-nodes'])
    require(options.get('--defer-database-writes','true')=='true','Database writes must use the native broker')
    require(options.get('--dataflow-capture-outputs','true') in ('true','false'),'Invalid preparation output retention')
    require(options.get('--dataflow-live-previews','false') in ('true','false'),'Invalid preparation live-preview choice')
    require(options.get('--dataflow-measurements','false') in ('true','false'),'Invalid preparation measurements')
    if '--dataflow-required-nodes' in options:parse_complete_nodes_json(options['--dataflow-required-nodes'])
    validate_model_input_nodes(options)
    graph=options['--execute-dataflow']
    require(graph.startswith('/') and str(Path(graph))==graph and all(p not in ('','.','..') for p in graph.split('/')[1:]),'Noncanonical preparation graph path')
    root=Path(base) if base is not None else Path(graph).parent.parent
    for flag,name in [('--execute-dataflow','input/graph.json'),('--source-bindings','input/bindings.json'),
                      ('--dataflow-node-labels','input/node-labels.json'),('--output','artifacts')]:
        require(options[flag]==str(root/name),'Preparation path escapes retained operation')
    if '--destination-bindings' in options:
        require(options['--destination-bindings']==str(root/'input/destinations.json'),'Output destination control escapes retained operation')
    validate_preparation_cache_allowance(options)
    policy=options['--dataflow-cache-policy'];require(policy in ('off','materialize','reuse'),'Invalid preparation cache policy')
    if policy=='reuse':
        require(REUSE<=set(options) and options['--dataflow-reuse-manifest']==str(root/'input/reuse/manifest.json'),'Invalid selected reuse manifest')
        hash_value(options['--dataflow-reuse-manifest-sha256'])
    else:require(not REUSE&set(options),'Reuse inputs require reuse policy')
    files=value['inputFiles'];require(isinstance(files,list) and 3<=len(files)<=(min(62500,inventory_limit(workload)+workload.limit('collectionMembers')+128) if workload else 1536),'Invalid preparation inventory count')
    inventory={};total=0;decision_bytes=0;decision_weights=0;sources=0;reuse_artifacts=0;reuse_bytes=0
    for entry in files:
        require(isinstance(entry,dict) and set(entry)=={'path','bytes','sha256'},'Invalid preparation inventory entry')
        name=entry['path'];limit=file_limit(name)
        if name.startswith('input/reuse/'):limit=prepared_file_limit(name.split('/')[-1],workload)
        require(name not in inventory and type(entry['bytes']) is int and 0<=entry['bytes']<=limit,'Invalid preparation input size or duplicate')
        hash_value(entry['sha256']);inventory[name]=entry;total+=entry['bytes']
        if name.startswith('input/decision-models/'):
            decision_bytes+=entry['bytes']
            if name.endswith('/model.safetensors'):decision_weights+=1
        if name.startswith('input/sources/'):
            require(name.split('/')[2]==entry['sha256'],'Source digest path differs from captured bytes');sources+=1
        elif name.startswith('input/file-collections/'):
            if name.endswith('/manifest.json'):
                require(name.split('/')[2]==entry['sha256'],'Collection manifest digest path differs');sources+=1
        elif name.startswith('input/reuse/'):
            require(policy=='reuse','Unselected reuse inputs')
            if name!='input/reuse/manifest.json':reuse_artifacts+=1;reuse_bytes+=entry['bytes']
    require(set(CONTROL_LIMITS)<=set(inventory) and sources<=128 and total-decision_bytes<=retained+1300000000 and decision_bytes<=2100000000 and decision_weights<=1,'Preparation input bounds exceeded')
    require(('input/destinations.json' in inventory)==('--destination-bindings' in options),'Destination input must be explicitly selected and pinned')
    require(reuse_artifacts<=(inventory_limit(workload) if workload else 300) and reuse_bytes<=retained,'Reuse inventory exceeds bounds')
    if policy=='reuse':
        require('input/reuse/manifest.json' in inventory and inventory['input/reuse/manifest.json']['sha256']==options['--dataflow-reuse-manifest-sha256'], 'Selected reuse pin differs from inventory')
    return options,inventory


def _json(root,name,inventory,cache):
    # verify_inputs freshly checks every retained file's identity, ownership,
    # mode and signature before reaching this function. Its pinned-file verifier
    # rehashes changed bytes and rejects inode replacement. Reuse only the
    # decoded semantic control for that exact verified file signature and pin;
    # this is not a cached owner/permission decision.
    signature=cache.get(name)
    require(type(signature) is tuple and len(signature)==8,'Preparation control lacks current file verification')
    key=('decodedPreparationControl',name);prior=cache.get(key)
    pin=inventory[name]['sha256']
    if prior is not None and prior[0]==signature and prior[1]==pin:return prior[2]
    parts=relative_parts(name);fd=directory(str(Path(root).joinpath(*parts[:-1])))
    try:raw=read_regular(fd,parts[-1],(CONTROL_LIMITS|OPTIONAL_CONTROL_LIMITS)[name])
    finally:os.close(fd)
    require(len(raw)==inventory[name]['bytes'] and sha(raw)==pin,'Preparation control changed')
    value=object_json(raw);cache[key]=(signature,pin,value)
    return value


def verify_inputs(value,root,cache):
    options,inventory=parse_contract(value,root)
    for name,entry in inventory.items():verify_pinned_file(root,name,entry,cache)
    # Scientific relationships depend only on this exact parsed request and
    # freshly verified immutable input set. Rechecking them per bounded work
    # checkpoint does not strengthen the live owner/file/destination checks.
    fingerprint=(str(root),tuple(sorted(options.items())),tuple((name,entry['bytes'],entry['sha256'],cache[name]) for name,entry in sorted(inventory.items())))
    semantic_key=('verifiedPreparationSemantics',)
    prior=cache.get(semantic_key)
    if prior is not None and prior[0]==fingerprint:
        for path in prior[1]:
            fd=directory(path);os.close(fd)
        try:fd=directory(str(Path(root)/'artifacts'))
        except FileNotFoundError:pass
        else:os.close(fd)
        return inventory
    graph=_json(root,'input/graph.json',inventory,cache)
    require(set(graph)=={'schemaVersion','nodes','edges'} and type(graph['schemaVersion']) is int and graph['schemaVersion']==1,'Invalid preparation graph schema')
    require(isinstance(graph['nodes'],list) and 1<=len(graph['nodes'])<=128
        and isinstance(graph['edges'],list) and len(graph['edges'])<=256,'Invalid preparation graph bounds')
    nodes=set();sources=set()
    for node in graph['nodes']:
        require(isinstance(node,dict) and set(node)=={'id','operator','config'},'Invalid preparation graph node')
        ident=node['id'];require(isinstance(ident,str) and 1<=len(ident)<=128 and ident not in nodes,'Invalid preparation node identity')
        require(isinstance(node['operator'],str) and isinstance(node['config'],dict),'Invalid preparation node fields')
        nodes.add(ident)
        if node['operator'] in ('kwip.data.sourceCSV','kwip.data.sourceFiles','kwip.data.capturedTable'):sources.add(ident)
    if '--dataflow-required-nodes' in options:require(set(parse_complete_nodes_json(options['--dataflow-required-nodes']))<=nodes,'Required retention references unknown nodes')
    if '--dataflow-complete-nodes' in options:
        require(set(parse_complete_nodes_json(options['--dataflow-complete-nodes']))<=nodes,'Complete demand references unknown preparation nodes')
    bindings=_json(root,'input/bindings.json',inventory,cache)
    require(set(bindings)==sources,'Captured bindings differ from source nodes')
    _verify_frozen_preparation(graph,root,inventory,cache)
    pack={name:entry for name,entry in inventory.items() if name.startswith('input/ocr-pack/')}
    if any(node['operator']=='kwip.data.extractText' for node in graph['nodes']):
        if __package__:from .ocr_tables import ARCHIVES
        else:from ocr_tables import ARCHIVES
        require(set(pack)=={'input/ocr-pack/'+name for name in ARCHIVES},'OCR requires its complete captured pack')
        for name,(_,size,pin) in ARCHIVES.items():
            entry=pack['input/ocr-pack/'+name];require(entry['bytes']==size and entry['sha256']==pin,'OCR captured asset differs from reviewed pin')
    else:require(not pack,'Unselected OCR pack input')
    speech={n['id'] for n in graph['nodes'] if n['operator']=='kwip.data.transcribeAudio'}
    captured={name:entry for name,entry in inventory.items() if name.startswith('input/speech-models/')}
    if speech:
        import hashlib,json
        require('input/speech-models.json' in inventory,'Speech model resource manifest is missing')
        pins=_json(root,'input/speech-models.json',inventory,cache);require(set(pins)==speech,'Speech model resources differ from graph')
        expected={}
        for pin in pins.values():
            require(type(pin) is dict and set(pin)=={'identity','files'},'Invalid speech model authority');hash_value(pin['identity'])
            require(type(pin['files']) is list and 3<=len(pin['files'])<=128,'Speech model file allowance exceeded')
            names=[f.get('path') for f in pin['files'] if type(f) is dict]
            require(len(names)==len(pin['files']) and len(set(names))==len(names) and {'config.json','tokenizer.json','preprocessor_config.json'}<=set(names),'Speech requires distinct model, tokenizer and processor pins')
            require(hashlib.sha256(json.dumps(pin['files'],sort_keys=True,separators=(',',':')).encode()).hexdigest()==pin['identity'],'Speech inventory identity differs')
            for f in pin['files']:
                require(type(f) is dict and set(f)=={'path','bytes','sha256'} and '/' not in f['path'],'Invalid speech file pin')
                path='input/speech-models/'+pin['identity']+'/'+f['path'];file_limit(path)
                expected[path]=dict(f,path=path)
        require(expected==captured,'Speech resources differ from pinned model files')
    else:require(not captured and 'input/speech-models.json' not in inventory,'Unselected speech model capture')
    decisions={n['id'] for n in graph['nodes'] if n['operator']=='kwip.data.decide'}
    captured={name:entry for name,entry in inventory.items() if name.startswith('input/decision-models/')}
    if decisions:
        import hashlib,json
        require('input/decision-models.json' in inventory,'Typed decisions model resource manifest is missing')
        pins=_json(root,'input/decision-models.json',inventory,cache);require(set(pins)==decisions,'Typed decisions model resources differ from graph')
        expected={}
        for pin in pins.values():
            require(type(pin) is dict and set(pin)=={'identity','files'},'Invalid decision model authority');hash_value(pin['identity'])
            require(type(pin['files']) is list and 3<=len(pin['files'])<=128,'Typed decisions model file allowance exceeded')
            names=[f.get('path') for f in pin['files'] if type(f) is dict]
            require(len(names)==len(pin['files']) and len(set(names))==len(names) and {'rl_agent_config.json','encoder/config.json','tokenizer/tokenizer.json','tokenizer/tokenizer_config.json'}<=set(names),'Typed decisions requires distinct model and processor pins')
            require(hashlib.sha256(json.dumps(pin['files'],sort_keys=True,separators=(',',':')).encode()).hexdigest()==pin['identity'],'Typed decisions inventory identity differs')
            for f in pin['files']:
                require(type(f) is dict and set(f)=={'path','bytes','sha256'},'Invalid decision file pin')
                path='input/decision-models/'+pin['identity']+'/'+f['path'];file_limit(path)
                expected[path]=dict(f,path=path)
        require(expected==captured,'Typed decisions resources differ from pinned model files')
    else:require(not captured and 'input/decision-models.json' not in inventory,'Unselected decision model capture')
    audio={n['id'] for n in graph['nodes'] if n['operator']=='kwip.data.classifyAudio'}
    captured={name:entry for name,entry in inventory.items() if name.startswith('input/audio-models/')}
    if audio:
        import hashlib,json
        require('input/audio-models.json' in inventory,'Audio classification model resource manifest is missing')
        pins=_json(root,'input/audio-models.json',inventory,cache);require(set(pins)==audio,'Audio classification model resources differ from graph')
        expected={}
        for pin in pins.values():
            require(type(pin) is dict and set(pin)=={'identity','files'},'Invalid audio model authority');hash_value(pin['identity'])
            require(type(pin['files']) is list and 3<=len(pin['files'])<=128,'Audio classification model file allowance exceeded')
            names=[f.get('path') for f in pin['files'] if type(f) is dict]
            require(len(names)==len(pin['files']) and len(set(names))==len(names) and {'config.json','preprocessor_config.json'}<=set(names),'Audio classification requires distinct model and processor pins')
            require(hashlib.sha256(json.dumps(pin['files'],sort_keys=True,separators=(',',':')).encode()).hexdigest()==pin['identity'],'Audio classification inventory identity differs')
            for f in pin['files']:
                require(type(f) is dict and set(f)=={'path','bytes','sha256'} and '/' not in f['path'],'Invalid audio file pin')
                path='input/audio-models/'+pin['identity']+'/'+f['path'];file_limit(path)
                expected[path]=dict(f,path=path)
        require(expected==captured,'Audio classification resources differ from pinned model files')
    else:require(not captured and 'input/audio-models.json' not in inventory,'Unselected audio model capture')
    expected_sources={name for name in inventory if name.startswith(('input/sources/','input/file-collections/','input/captured-tables/'))};used=set();collection_members=0
    operators={node['id']:node['operator'] for node in graph['nodes']}
    if __package__:from .file_collection_contract import parse_manifest,owned_member_paths
    else:from file_collection_contract import parse_manifest,owned_member_paths
    for ident,path in bindings.items():
        require(isinstance(path,str) and len(path)<=4096,'Invalid preparation source path')
        matches=[name for name in expected_sources if path==str(Path(root)/name)]
        require(len(matches)==1,'Source binding escapes captured inventory');name=matches[0]
        if operators[ident]=='kwip.data.sourceFiles':
            require(name.startswith('input/file-collections/') and name.endswith('/manifest.json'),'File Input requires its captured collection manifest')
            parts=relative_parts(name);fd=directory(str(Path(root).joinpath(*parts[:-1])))
            try:raw=read_regular(fd,parts[-1],2000000)
            finally:os.close(fd)
            require(len(raw)==inventory[name]['bytes'] and sha(raw)==inventory[name]['sha256'],'Collection manifest changed')
            document=parse_manifest(raw)
            if __package__:from .file_input_options import validate_configuration
            else:from file_input_options import validate_configuration
            selected=validate_configuration(next(node['config'] for node in graph['nodes'] if node['id']==ident))
            require(document['selection']=={key:selected[key] for key in document['selection']},'Collection selection differs from authorized File Input configuration')
            collection_members+=len(document['members']);require(collection_members<=1000,'Preparation collection member count exceeded')
            used.update(owned_member_paths(document,name,inventory))
        elif operators[ident]=='kwip.data.capturedTable':
            pin=next(node['config'].get('sha256') for node in graph['nodes'] if node['id']==ident)
            require(name=='input/captured-tables/'+str(pin)+'.json' and inventory[name]['sha256']==pin,'Captured predictions differ from their exact artifact pin');used.add(name)
        else:
            require(name.startswith('input/sources/'),'Table Input cannot consume a file collection');used.add(name)
    require(used==expected_sources,'Unbound source or collection member in preparation inventory')
    sinks={node['id'] for node in graph['nodes'] if node['operator']=='kwip.data.outputDelimited' and node['config'].get('destination') not in ('database','api')}
    destinations=_json(root,'input/destinations.json',inventory,cache) if '--destination-bindings' in options else {}
    require(set(destinations)==sinks,'Destination grants differ from output components')
    for path in destinations.values():
        require(isinstance(path,str) and 1<=len(path.encode('utf-8'))<=4096 and '\0' not in path and path.startswith('/') and str(Path(path))==path and all(part not in ('','.','..') for part in path.split('/')[1:]),'Invalid private output destination path')
        fd=directory(path);os.close(fd)
    labels=_json(root,'input/node-labels.json',inventory,cache)
    require(len(labels)<=128 and set(labels)<=nodes and all(isinstance(label,str) and label.strip() and len(label)<=200 for label in labels.values()),'Invalid preparation diagnostic labels')
    if options['--dataflow-cache-policy']=='reuse':
        verify_completed_inventory(root,'input/reuse',inventory,options['--dataflow-reuse-manifest-sha256'],admission_policy=request_policy(value))
    try:fd=directory(str(Path(root)/'artifacts'))
    except FileNotFoundError:pass
    else:os.close(fd)
    cache[semantic_key]=(fingerprint,tuple(destinations.values()))
    return inventory


def _verify_frozen_preparation(graph,root,inventory,cache):
    """Admit exact inert resource pairs, without importing NumPy/data engines.

    Full model execution validation remains in the frozen worker. Admission binds
    the selected graph, complete files and portable rules to that same model.
    """
    import json
    selected={n['id'] for n in graph['nodes'] if n['operator']=='kwip.data.preprocess' and n['config'].get('mode')=='frozen'}
    captured={name for name in inventory if name.startswith('input/preparation-models/')}
    if not selected:
        require(not captured and 'input/preparation-models.json' not in inventory,'Unselected learned preparation resource');return
    require('input/preparation-models.json' in inventory,'Learned preparation model manifest is missing')
    mapping=_json(root,'input/preparation-models.json',inventory,cache)
    require(set(mapping)==selected,'Learned preparation resources differ from graph')
    def read(name):
        parts=relative_parts(name);fd=directory(str(Path(root).joinpath(*parts[:-1])))
        try:raw=read_regular(fd,parts[-1],file_limit(name))
        finally:os.close(fd)
        require(len(raw)==inventory[name]['bytes'] and sha(raw)==inventory[name]['sha256'],'Learned preparation resource changed')
        return raw,object_json(raw)
    used=set()
    if __package__:from . import preprocess_parameters as portable
    else:import preprocess_parameters as portable
    for pin in mapping.values():
        require(type(pin) is dict and set(pin)=={'path','sha256','preparationPath','preparationSHA256'},'Invalid learned preparation resource pin')
        parts=relative_parts(pin['path'])
        require(len(parts)==3 and parts[0]=='preparation-models' and parts[2]=='state.json','Unsafe learned preparation model path');hash_value(parts[1])
        require(pin['preparationPath']=='preparation-models/'+parts[1]+'/preparation.json','Learned rules must share their model directory')
        model_name='input/'+pin['path'];rules_name='input/'+pin['preparationPath']
        for name,key in [(model_name,'sha256'),(rules_name,'preparationSHA256')]:
            hash_value(pin[key]);require(name in captured and inventory[name]['sha256']==pin[key],'Learned preparation inventory differs from resource pin')
        _,model=read(model_name);_,envelope=read(rules_name)
        require(set(envelope)=={'schemaVersion','kind','modelSHA256','featureColumns','inputSchema','outputSchema','parameters','contentDigest'} and type(envelope['schemaVersion']) is int and envelope['schemaVersion']==1 and envelope['kind']=='kwipml.prebuilt-preparation.v1','Invalid learned preparation envelope')
        canonical=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        require(envelope['modelSHA256']==pin['sha256'] and envelope['contentDigest']==sha(canonical({k:v for k,v in envelope.items() if k!='contentDigest'})),'Learned preparation model binding or content digest differs')
        require(model.get('stateFormat')=='kwipml.prebuilt-model.v1' and model.get('featureColumns')==envelope['featureColumns'],'Learned preparation feature order differs from model')
        parameters=envelope['parameters'];require(type(parameters) is list and 1<=len(parameters)<=16,'Invalid learned preparation chain')
        previous=envelope['inputSchema'];digests=set()
        for p in parameters:
            portable.validate(p)
            require(p['configuration']['mode']=='partitions' and p['inputSchema']==previous and p['sourceContentDigest'] not in digests,'Invalid ordered learned preparation chain')
            digests.add(p['sourceContentDigest']);previous=p['outputSchema']
        require(previous==envelope['outputSchema'],'Learned preparation output boundary differs')
        used.update([model_name,rules_name])
    require(used==captured,'Unbound learned preparation input')
