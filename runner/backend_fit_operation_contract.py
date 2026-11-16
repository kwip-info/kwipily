"""Exact retained input authority for allowlisted backend fitting."""
import json
import os
from pathlib import Path
if __package__:
    from .worker_authorization import require,hash_value,parse_bounded_options,directory,read_regular,object_json
    from .worker_input_contract import relative_parts,verify_pinned_file,verify_completed_inventory
else:
    from worker_authorization import require,hash_value,parse_bounded_options,directory,read_regular,object_json
    from worker_input_contract import relative_parts,verify_pinned_file,verify_completed_inventory


if __package__:from .preparation_artifact_limits import request_policy,prepared_file_limit,inventory_limit
else:from preparation_artifact_limits import request_policy,prepared_file_limit,inventory_limit


def file_limit(name):
    if name=='input/model-state.json':return 32_000_000
    if name=='input/model-preparation.json':return 20_000_000
    if name=='input/backend-request.json':return 100_000
    if name=='input/preparation/manifest.json':return 2_000_000
    parts=relative_parts(name)
    require(len(parts)==3 and parts[:2]==['input','preparation'],'Unsupported backend input path')
    return 50_000_000


def parse_contract(value,base=None):
    workload=request_policy(value);retained=workload.limit('retainedBytes') if workload else 200000000
    options=parse_bounded_options(value['arguments'],{'--fit-backend','--backend-request-sha256','--output'},set(),maximum_argument_bytes=4096,maximum_total_bytes=16384)
    require(set(options)=={'--fit-backend','--backend-request-sha256','--output'},'Missing backend request or output')
    request=options['--fit-backend'];root=Path(base) if base is not None else Path(request).parent.parent
    require(root.is_absolute() and request==str(root/'input/backend-request.json') and options['--output']==str(root/'artifacts'),'Backend request paths escape retained operation')
    hash_value(options['--backend-request-sha256'])
    files=value['inputFiles'];require(isinstance(files,list) and 2<=len(files)<=(inventory_limit(workload)+3 if workload else 303),'Invalid backend input inventory')
    inventory={};total=0
    for entry in files:
        require(isinstance(entry,dict) and set(entry)=={'path','bytes','sha256'},'Invalid backend file entry')
        name=entry['path'];limit=file_limit(name)
        if name.startswith('input/preparation/'):limit=prepared_file_limit(name.split('/')[-1],workload)
        require(name not in inventory and type(entry['bytes']) is int and 0<=entry['bytes']<=limit,'Invalid backend input size or duplicate')
        hash_value(entry['sha256']);inventory[name]=entry;total+=entry['bytes']
    require(total<=retained+2_100_000 and {'input/backend-request.json','input/preparation/manifest.json'}<=set(inventory),'Missing backend captured inputs')
    require(inventory['input/backend-request.json']['sha256']==options['--backend-request-sha256'],'Backend request pin differs')
    return options,inventory


def verify_inputs(value,root,cache):
    proof=cache.get('_backend_input_proof')
    if proof is not None:
        try:
            if proof.check(value,root):return
            return _verify_inputs_full(value,root,cache)
        except BaseException:
            proof.invalidate()
            raise
    _verify_inputs_full(value,root,cache)
    if __package__:from .backend_input_proof import BackendInputProof
    else:from backend_input_proof import BackendInputProof
    _,inventory=parse_contract(value,root)
    cache['_backend_input_proof']=BackendInputProof(value,root,inventory)


def close_inputs(cache):
    proof=cache.get('_backend_input_proof')
    if proof is not None:proof.close()


def _verify_inputs_full(value,root,cache):
    options,inventory=parse_contract(value,root)
    for name,entry in inventory.items():verify_pinned_file(root,name,entry,cache)
    fd=directory(str(Path(root)/'input'))
    try:raw=read_regular(fd,'backend-request.json',100_000)
    finally:os.close(fd)
    request=object_json(raw)
    require(isinstance(request,dict) and request.get('backendID') in ('sklearn','tensorflow'),'Unsupported backend identity')
    if request.get('model') is not None:
        model=request['model'];base={'path','sha256'};with_rules=base|{'preparationPath','preparationSHA256'}
        require(type(model) is dict and set(model) in (base,with_rules) and model['path']==str(Path(root)/'input/model-state.json') and inventory.get('input/model-state.json',{}).get('sha256')==model['sha256'],'Classifier state differs from captured input.')
        hash_value(model['sha256'])
        if set(model)==with_rules:
            config=request.get('configuration',{})
            require(request['backendID']=='sklearn' and config.get('task')=='prebuilt' and config.get('operation') in ('predict','evaluate'),'Learned preparation is only a frozen Prebuilt input')
            hash_value(model['preparationSHA256'])
            require(model['preparationPath']==str(Path(root)/'input/model-preparation.json') and inventory.get('input/model-preparation.json',{}).get('sha256')==model['preparationSHA256'],'Learned preparation differs from captured input')
        else:require('input/model-preparation.json' not in inventory,'Unselected learned preparation input')
    else:require(not {'input/model-state.json','input/model-preparation.json'}&set(inventory),'Unselected model resource')
    prepared=request.get('prepared');require(isinstance(prepared,dict),'Missing prepared backend input')
    require(prepared.get('manifest')==str(Path(root)/'input/preparation/manifest.json'),'Backend preparation escapes retained input')
    require(prepared.get('manifestSHA256')==inventory['input/preparation/manifest.json']['sha256'],'Backend preparation pin differs')
    verify_completed_inventory(root,'input/preparation',inventory,prepared['manifestSHA256'],prepared.get('graphDigest'),admission_policy=request_policy(value))
