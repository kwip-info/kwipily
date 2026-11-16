"""Exact captured Self-play authority, including an optional verified owned resume inventory."""
import os,re
from pathlib import Path
if __package__:
    from .worker_authorization import require,hash_value,directory,read_regular,object_json
    from .worker_input_contract import verify_pinned_file
else:
    from worker_authorization import require,hash_value,directory,read_regular,object_json
    from worker_input_contract import verify_pinned_file

def replay_limit(name):
    if name=='latest.json':return 128
    if re.fullmatch(r'chunks/[a-f0-9]{64}\.jsonl',name):return 4_000_000
    if re.fullmatch(r'checkpoints/iteration-[0-9]{2}/(model|optimizer|rng)\.pt',name):return 100_000_000
    if re.fullmatch(r'checkpoints/iteration-[0-9]{2}/manifest\.json',name):return 2_000_000
    if re.fullmatch(r'checkpoints/iteration-[0-9]{2}/PINNED',name):return 64
    raise ValueError('Unreviewed Self-play artifact path.')

def file_limit(name):
    if name in ('input/self-play.json','input/document.json','input/campaign-grant.json'):return 2_000_000
    if re.fullmatch(r'input/opponents/[A-F0-9-]{36}/(model.pt|model.json|identity.json)',name):return 100_000_000 if name.endswith('model.pt') else 2_000_000
    if name.startswith('input/resume/'):return replay_limit(name.removeprefix('input/resume/'))
    raise ValueError('Unreviewed Self-play input.')

def parse_contract(value,base=None):
    a=value['arguments'];require(type(a) is list and len(a)==2 and a[0]=='--self-play','Self-play requires exact managed arguments.')
    root=Path(base) if base is not None else Path(a[1]).parent.parent
    require(a[1]==str(root/'input/self-play.json') and root.name==value['operationID'] and root.parent.name==value['projectID'] and root.parent.parent.name=='workflow-operations','Self-play path escapes owned operation.')
    files=value['inputFiles'];require(type(files) is list and 1<=len(files)<=600,'Invalid Self-play inventory.')
    inventory={};total=0
    for f in files:
        require(type(f) is dict and set(f)=={'path','bytes','sha256'},'Invalid Self-play artifact.')
        name=f['path'];limit=file_limit(name)
        require(name not in inventory and type(f['bytes']) is int and 0<f['bytes']<=limit,'Invalid Self-play artifact size or duplicate.')
        hash_value(f['sha256']);inventory[name]=f;total+=f['bytes']
    require({'input/self-play.json','input/document.json'}<=set(inventory) and total<=1_000_000_000,'Missing Self-play request or excess bytes.')
    return {},inventory

def verify_inputs(value,root,cache):
    _,inventory=parse_contract(value,root)
    for name,f in inventory.items():verify_pinned_file(root,name,f,cache)
    fd=directory(str(Path(root)/'input'))
    try:inner=object_json(read_regular(fd,'self-play.json',2_000_000))
    finally:os.close(fd)
    require(set(inner)=={'schemaVersion','definition','graph','configuration','initialization','trainability','documentSHA256','trainAddress','parentOperationID','resume','validateOnly'} and inner['schemaVersion']==1 and type(inner['resume']) is bool and type(inner['validateOnly']) is bool,'Invalid Self-play envelope.')
    require(type(inner['trainAddress']) is dict and set(inner['trainAddress'])=={'domain','localID'} and inner['trainAddress']['domain']=='control' and type(inner['trainAddress']['localID']) is str,'Invalid captured Train address.')
    hash_value(inner['documentSHA256']);require(inventory['input/document.json']['sha256']==inner['documentSHA256'],'Captured document pin differs.')
    parent=inner['parentOperationID'];require((parent is None and not inner['resume']) or (inner['resume'] and type(parent) is str and str(__import__('uuid').UUID(parent)).upper()==parent),'Resume lineage differs.')
    require(inner['resume']==('input/resume/latest.json' in inventory),'Resume inventory differs.')
    if not inner['resume']:require(all(n in {'input/self-play.json','input/document.json','input/campaign-grant.json'} or n.startswith('input/opponents/') for n in inventory),'Unexpected Self-play inputs.')
    if 'input/campaign-grant.json' in inventory:
        if __package__:from .self_play_campaign_grant import campaign_directory
        else:from self_play_campaign_grant import campaign_directory
        grant=object_json((Path(root)/'input/campaign-grant.json').read_bytes());campaign_directory(root,grant)
        require(inner['configuration'].get('campaign',{}).get('enabled') is True and not inner['resume'],'Campaign grant requires campaign mode without copied resume inputs.')
    return inner
