"""Pure admission for immutable host-captured File Input collections."""
import json
import re
from pathlib import PurePosixPath

MAX_MEMBER_BYTES=64*1024*1024
MAX_TOTAL_BYTES=256*1024*1024
MAX_MEMBERS=1000
MAX_MANIFEST_BYTES=2_000_000
_HASH=re.compile(r'^[0-9a-f]{64}$')


def require(ok,message):
    if not ok:raise ValueError(message)


def relative(value):
    require(type(value) is str and 0<len(value.encode('utf-8'))<=4096 and '\0' not in value and '\\' not in value,'Invalid collection relative path')
    parts=value.split('/')
    require(len(parts)<=33 and all(x not in ('','.','..') and len(x.encode())<=255 for x in parts),'Unsafe collection path component')
    return parts


def validate_manifest(document):
    require(type(document) is dict and set(document)=={'schemaVersion','kind','sourceMode','readMode','complete','members','selection'},'Invalid collection manifest fields')
    require(type(document['schemaVersion']) is int and document['schemaVersion']==1 and document['kind']=='kwipFileCollection','Unsupported file collection')
    require(document['sourceMode'] in ('file','directory') and document['readMode'] in ('metadata','content') and type(document['complete']) is bool,'Invalid collection mode')
    selection=document['selection']
    keys={'sourceMode','extensions','includeSubfolders','includeHidden','includePatterns','excludePatterns','symlinks','readMode','unknownFiles','maxFiles','maxDepth','maxFileBytes','maxTotalBytes','errorPolicy'}
    require(type(selection) is dict and set(selection)==keys,'Invalid collection selection fields')
    if __package__:from .file_input_options import validate_configuration
    else:from file_input_options import validate_configuration
    normalized=validate_configuration(selection)
    require(selection=={key:normalized[key] for key in keys} and selection['sourceMode']==document['sourceMode'] and selection['readMode']==document['readMode'],'Collection selection differs from normalized modes')
    members=document['members'];require(type(members) is list and len(members)<=MAX_MEMBERS,'Collection member count exceeded')
    require(document['sourceMode']!='file' or len(members)<=1,'Single-file collection contains multiple members')
    names=set();captures=set();total=0
    for member in members:
        require(type(member) is dict and {'relativePath','capturePath','bytes','sha256','extension','status'}<=set(member) and not set(member)-{'relativePath','capturePath','bytes','sha256','extension','status','modifiedAt','createdAt','error'},'Invalid collection member fields')
        relative(member['relativePath']);require(member['relativePath'] not in names,'Duplicate collection member');names.add(member['relativePath'])
        require(type(member['bytes']) is int and 0<=member['bytes']<=2**63-1,'Invalid collection member size')
        require(type(member['extension']) is str and len(member['extension'])<=32 and (not member['extension'] or re.fullmatch('[a-z0-9_-]+',member['extension'])),'Invalid collection extension')
        status=member['status'];require(status in ('content','metadata','error'),'Invalid collection member status')
        require(document['readMode']!='metadata' or status!='content','Metadata collection contains body capture')
        if status=='content':
            require(member['bytes']<=MAX_MEMBER_BYTES,'Collection member byte limit exceeded');total+=member['bytes']
            require(total<=MAX_TOTAL_BYTES,'Collection total byte limit exceeded')
            require(type(member['sha256']) is str and _HASH.fullmatch(member['sha256']),'Invalid collection content pin')
            parts=relative(member['capturePath']);require(len(parts)==2 and parts[0]=='members' and member['capturePath'] not in captures,'Invalid collection capture path');captures.add(member['capturePath'])
            require('error' not in member,'Captured content contains error status')
        else:
            require(member['capturePath'] is None and member['sha256'] is None,'Uncaptured collection member claims content authority')
            if status=='error':require(type(member.get('error')) is str and 0<len(member['error'])<=1000 and '\0' not in member['error'],'Invalid collection error')
            else:require('error' not in member,'Metadata member contains error')
        for key in ('modifiedAt','createdAt'):
            if key in member:require(type(member[key]) is str and 0<len(member[key])<=64 and '\0' not in member[key],'Invalid collection timestamp')
    require([m['relativePath'] for m in members]==sorted(names),'Collection members must be in deterministic relative-path order')
    return document


def parse_manifest(raw):
    require(type(raw) is bytes and len(raw)<=MAX_MANIFEST_BYTES,'Collection manifest exceeds byte limit')
    def unique(pairs):
        result={}
        for key,value in pairs:
            require(key not in result,'Duplicate collection JSON field');result[key]=value
        return result
    def bad(_):raise ValueError('Nonfinite collection value')
    try:document=json.loads(raw,object_pairs_hook=unique,parse_constant=bad)
    except (json.JSONDecodeError,UnicodeDecodeError,RecursionError) as error:raise ValueError('Invalid collection manifest JSON') from error
    return validate_manifest(document)


def owned_member_paths(document,manifest_relative,inventory):
    validate_manifest(document);parts=relative(manifest_relative)
    require(len(parts)==4 and parts[:2]==['input','file-collections'] and _HASH.fullmatch(parts[2]) and parts[3]=='manifest.json','Invalid collection manifest operation path')
    require(manifest_relative in inventory and inventory[manifest_relative]['sha256']==parts[2],'Collection manifest path differs from pin')
    parent='/'.join(parts[:-1]);owned={manifest_relative}
    for member in document['members']:
        if member['status']!='content':continue
        name=parent+'/'+member['capturePath'];require(name in inventory,'Collection member missing from admitted inventory')
        entry=inventory[name]
        require(entry['bytes']==member['bytes'] and entry['sha256']==member['sha256'],'Collection member differs from declared size/pin');owned.add(name)
    return owned
