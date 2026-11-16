"""Strict captured inventory for a frozen-policy game operation; no arbitrary source paths."""
import os
from pathlib import Path
if __package__:
    from .worker_authorization import require,hash_value,directory,read_regular,object_json
    from .worker_input_contract import verify_pinned_file
else:
    from worker_authorization import require,hash_value,directory,read_regular,object_json
    from worker_input_contract import verify_pinned_file

def file_limit(name):
    if name=='input/session.json':return 2_000_000
    if name in ('input/A.pt','input/B.pt'):return 100_000_000
    raise ValueError('Unreviewed Game session input.')

def parse_contract(value,base=None):
    a=value['arguments'];require(type(a) is list and len(a)==2 and a[0]=='--game-session','Game session requires exact managed arguments.')
    root=Path(base) if base is not None else Path(a[1]).parent.parent
    require(a[1]==str(root/'input/session.json') and root.name==value['operationID'] and root.parent.name==value['projectID'] and root.parent.parent.name=='workflow-operations','Game session path escapes owned operation.')
    files=value['inputFiles'];require(type(files) is list and 1<=len(files)<=3,'Invalid Game session inventory.')
    inventory={};total=0
    for f in files:
        require(type(f) is dict and set(f)=={'path','bytes','sha256'},'Invalid game artifact.')
        name=f['path'];limit=file_limit(name)
        require(name not in inventory and type(f['bytes']) is int and 0<f['bytes']<=limit,'Invalid game artifact size or duplicate.')
        hash_value(f['sha256']);inventory[name]=f;total+=f['bytes']
    require('input/session.json' in inventory and total<=202_000_000,'Missing game request or excess bytes.')
    return {},inventory

def verify_inputs(value,root,cache):
    _,inventory=parse_contract(value,root)
    for name,f in inventory.items():verify_pinned_file(root,name,f,cache)
    fd=directory(str(Path(root)/'input'))
    try:inner=object_json(read_regular(fd,'session.json',2_000_000))
    finally:os.close(fd)
    require(set(inner)=={'schemaVersion','definition','configuration','policies','validateOnly'} and inner['schemaVersion']==1 and type(inner['validateOnly']) is bool and type(inner['policies']) is dict and not set(inner['policies'])-{'A','B'},'Invalid session envelope.')
    expected={'input/session.json'}
    for role,p in inner['policies'].items():
        require(type(p) is dict and set(p)<= {'graph','initialization','checkpointSHA256','sourceIdentity'} and 'graph' in p and 'sourceIdentity' in p,'Invalid game policy snapshot.')
        hash_value(p['sourceIdentity'])
        if 'checkpointSHA256' in p:
            name='input/'+role+'.pt';require(name in inventory and inventory[name]['sha256']==p['checkpointSHA256'],'Saved policy checkpoint pin differs.');expected.add(name)
    require(set(inventory)==expected,'Unexpected Game session inputs.')
    return inner
