"""Standard-library admission for immutable saved preparation inspection."""
from pathlib import Path
if __package__:
    from .worker_authorization import require,hash_value
    from .worker_input_contract import relative_parts,verify_pinned_file,verify_completed_inventory
else:
    from worker_authorization import require,hash_value
    from worker_input_contract import relative_parts,verify_pinned_file,verify_completed_inventory

PREFIX='input/preparation'
MANIFEST=PREFIX+'/manifest.json'


if __package__:from .preparation_artifact_limits import request_policy,prepared_file_limit,inventory_limit
else:from preparation_artifact_limits import request_policy,prepared_file_limit,inventory_limit


def parse_contract(value,base=None):
    if __package__:from .preparation_artifact_limits import request_policy,prepared_file_limit,inventory_limit
    else:from preparation_artifact_limits import request_policy,prepared_file_limit,inventory_limit
    workload=request_policy(value);maximum=workload.limit('retainedBytes') if workload else 200000000
    args=value['arguments'];files=value['inputFiles']
    require(isinstance(args,list) and len(args)==4
        and all(isinstance(arg,str) and arg and '\0' not in arg and len(arg.encode())<=4096 for arg in args)
        and args[0]=='--inspect-dataflow' and args[2]=='--dataflow-manifest-sha256','Invalid preparation inspection arguments')
    hash_value(args[3])
    path=args[1]
    require(path.startswith('/') and str(Path(path))==path
        and all(part not in ('','.','..') for part in path.split('/')[1:]),'Noncanonical inspection manifest path')
    root=Path(base) if base is not None else Path(path).parent.parent.parent
    require(path==str(root/MANIFEST),'Inspection manifest escapes retained operation')
    require(isinstance(files,list) and 2<=len(files)<=(inventory_limit(workload)+1 if workload else 301),'Invalid saved preparation inventory count')
    inventory={};artifact_bytes=0;total=0
    for entry in files:
        require(isinstance(entry,dict) and set(entry)=={'path','bytes','sha256'},'Invalid saved preparation inventory entry')
        name=entry['path'];parts=relative_parts(name)
        require(len(parts)==3 and parts[:2]==['input','preparation'] and name not in inventory,'Invalid saved preparation input path')
        limit=prepared_file_limit(parts[-1],workload)
        require(type(entry['bytes']) is int and 0<=entry['bytes']<=limit,'Saved preparation input exceeds size bound')
        hash_value(entry['sha256']);inventory[name]=entry;total+=entry['bytes']
        if name!=MANIFEST:artifact_bytes+=entry['bytes']
    require(MANIFEST in inventory and inventory[MANIFEST]['sha256']==args[3],'Inspection manifest pin differs from inventory')
    require(artifact_bytes<=maximum and total<=maximum+2000000,'Saved preparation exceeds total size bound')
    return inventory


def verify_inputs(value,root,cache):
    inventory=parse_contract(value,root)
    for name,entry in inventory.items():verify_pinned_file(root,name,entry,cache)
    verify_completed_inventory(root,PREFIX,inventory,value['arguments'][3],admission_policy=request_policy(value))
