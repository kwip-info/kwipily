"""Strict, standard-library-only File Input options from the shared reader catalog."""
import copy
import json
import math
import re
from pathlib import Path

SPEC_PATH=Path(__file__).with_name('file-input-options.json')
SPEC=json.loads(SPEC_PATH.read_text(encoding='utf-8'))
if SPEC.get('schemaVersion')!=1:raise ValueError('Unsupported File Input reader catalog')
READER_ORDER=('text','pdf','docx','rtf','odt','image','audio')
ALL_EXTENSIONS=[extension for name in READER_ORDER for extension in SPEC['extensions'][name]]
READER_DEFINITIONS=SPEC['readers']
if set(READER_DEFINITIONS)!=set(('text','pdf','docx','rtf','odt','image','audio')) or set(SPEC['extensions'])!=set(READER_DEFINITIONS):raise ValueError('Invalid File Input reader catalog')
for definition in READER_DEFINITIONS.values():
    fields=definition['fields']
    if not 1<=len(fields)<=64 or len({field['key'] for field in fields})!=len(fields):raise ValueError('Duplicate or unbounded File Input fields')
    for field in fields:
        if not field['key'] or not field['title'] or field['kind'] not in ('choice','boolean','integer','number','string','integerList'):raise ValueError('Unsupported File Input field')
        if field['kind']=='choice' and (not field.get('choices') or len({x['value'] for x in field['choices']})!=len(field['choices'])):raise ValueError('Invalid File Input choices')
        if field['kind'] in ('integer','number') and not (math.isfinite(field['min']) and math.isfinite(field['max']) and field['min']<=field['max']):raise ValueError('Invalid File Input bounds')
DEFAULTS={
 'sourceMode':'file','extensions':[extension for name in READER_ORDER if name!='audio' for extension in SPEC['extensions'][name]],'includeSubfolders':False,'includeHidden':False,
 'includePatterns':['**/*'],'excludePatterns':[],'symlinks':'skip','readMode':'content',
 'errorPolicy':'fail','unknownFiles':'metadata','maxFiles':1000,'maxDepth':32,
 'maxFileBytes':67108864,'maxTotalBytes':268435456,
 'metadataFields':['file_id','name','relative_path','extension','detected_type','byte_size'],
 'readers':{name:{field['key']:field['defaultValue'] for field in definition['fields']} for name,definition in READER_DEFINITIONS.items()},
}


def defaults():return copy.deepcopy(DEFAULTS)


def get_reader(name):
    if type(name) is not str or name not in READER_DEFINITIONS:raise ValueError('Unknown File Input reader')
    return copy.deepcopy(READER_DEFINITIONS[name])


def _require(valid,message):
    if not valid:raise ValueError(message)


def _text(value,maximum=4096):return type(value) is str and len(value)<=maximum and '\0' not in value


def _list(value,maximum=128):
    _require(type(value) is list and len(value)<=maximum,'File Input list exceeds its supported bound')
    return value


def validate_reader_options(name,options=None):
    definition=get_reader(name)
    if options is None:options={}
    _require(type(options) is dict,'Reader options must be an object')
    fields={field['key']:field for field in definition['fields']}
    _require(not set(options)-set(fields),'Unknown '+name+' reader option')
    result={key:copy.deepcopy(options.get(key,field['defaultValue'])) for key,field in fields.items()}
    for key,field in fields.items():
        value=result[key];kind=field['kind']
        if kind=='choice':_require(type(value) is str and value in [choice['value'] for choice in field['choices']],'Choose a supported '+name+' '+key)
        elif kind=='boolean':_require(type(value) is bool,'Reader '+key+' requires a Boolean')
        elif kind=='integer':_require(type(value) is int and field['min']<=value<=field['max'],'Reader '+key+' requires a bounded integer')
        elif kind=='number':_require(type(value) in (int,float) and field['min']<=value<=field['max'] and math.isfinite(value),'Reader '+key+' requires a finite bounded number')
        elif kind=='string':_require(_text(value),'Reader '+key+' requires bounded text')
        elif kind=='integerList':
            _list(value,1000)
            _require(all(type(item) is int and 1<=item<=1000000 for item in value) if key=='pages' else all(type(item) is int and item in (0,90,180,270) for item in value),'Invalid PDF '+key)
            _require(len(set(value))==len(value) and (key!='orientations' or bool(value)),'PDF selections must be unique; orientations cannot be empty')
        else:raise ValueError('Unknown File Input reader field kind')
    if name=='image' and result['frames']=='range':_require(result['frameStart']<=result['frameEnd'],'The last image frame must not precede the first')
    return result


def validate_configuration(config):
    _require(type(config) is dict and not set(config)-set(DEFAULTS),'Unknown File Input setting; file paths belong in private bindings')
    result=defaults();result.update(copy.deepcopy(config))
    for key,choices in {'sourceMode':('file','directory'),'symlinks':('skip','error'),'readMode':('content','metadata'),'errorPolicy':('fail','record','skip'),'unknownFiles':('metadata','error','skip')}.items():
        _require(type(result[key]) is str and result[key] in choices,'Choose a supported File Input '+key)
    for key in ('includeSubfolders','includeHidden'):_require(type(result[key]) is bool,key+' requires a Boolean')
    for key,upper in {'maxFiles':1000,'maxDepth':32,'maxFileBytes':67108864,'maxTotalBytes':268435456}.items():
        _require(type(result[key]) is int and 1<=result[key]<=upper,key+' must be a positive integer within its File Input resource bound')
    _require(result['maxFileBytes']<=result['maxTotalBytes'],'Per-file byte limit must not exceed the total byte limit')
    extensions=_list(result['extensions'])
    _require(all(type(item) is str and re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,31}',item) for item in extensions) and len(set(extensions))==len(extensions),'Extensions must be distinct lowercase names without dots or paths; empty selects all files')
    metadata=_list(result['metadataFields'])
    _require(all(type(item) is str and item in SPEC['metadataFields'] for item in metadata) and len(set(metadata))==len(metadata) and 'file_id' in metadata,'Choose distinct supported metadata fields')
    _require(result['readMode']!='metadata' or bool(result['metadataFields']),'Metadata-only reading needs at least one metadata field')
    for key in ('includePatterns','excludePatterns'):
        patterns=_list(result[key])
        for pattern in patterns:
            _require(_text(pattern,512) and bool(pattern) and not pattern.startswith('/') and not re.match(r'^[A-Za-z]:',pattern) and '\\' not in pattern and '://' not in pattern and '..' not in pattern.split('/'),'File filters must be bounded relative glob patterns without parent traversal')
        _require(len(set(patterns))==len(patterns),'File filter patterns must be distinct')
    readers=result['readers'];_require(type(readers) is dict and not set(readers)-set(READER_DEFINITIONS),'Unknown File Input reader')
    result['readers']={name:validate_reader_options(name,readers.get(name,{})) for name in READER_DEFINITIONS}
    _require(len(json.dumps(result,ensure_ascii=False,separators=(',',':')).encode('utf-8'))<=32768,'File Input settings exceed their 32 KiB transport bound')
    return result
