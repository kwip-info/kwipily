"""Admission for inspecting one captured source; no source path or credential options."""
from pathlib import Path
import json
if __package__:
    from .csv_parser import parse_options_json
    from .worker_authorization import require,hash_value
    from .worker_input_contract import relative_parts,verify_pinned_file
else:
    from csv_parser import parse_options_json
    from worker_authorization import require,hash_value
    from worker_input_contract import relative_parts,verify_pinned_file


def normalize_table_options(configuration):
    if type(configuration) is dict and configuration.get('parserProfile')=='polars':
        if __package__:from .polars_csv_configuration import normalize_options
        else:from polars_csv_configuration import normalize_options
        return normalize_options(configuration)
    require(type(configuration) is dict and set(configuration)=={'format','options'},'Table inspection requires format/options only')
    if configuration['format']=='api':
        if __package__:from .api_tables import options
        else:from api_tables import options
        return {'format':'api','options':options(configuration['options'])}
    if configuration['format']=='database':
        if __package__:from .database_source import options
        else:from database_source import options
        # The worker only consumes the broker's local typed snapshot. These
        # portable options describe its origin; they never open a connection.
        return {'format':'database','options':options(configuration['options'])}
    if __package__:from .table_source_options import normalize_options
    else:from table_source_options import normalize_options
    fmt,options=normalize_options(configuration)
    return {'format':fmt,'options':options}


def parse_table_options_json(value):
    require(type(value) is str and len(value.encode('utf-8'))<=32768 and '\0' not in value,'Table inspection settings exceed the transport limit')
    def unique(pairs):
        result={}
        for key,item in pairs:
            require(key not in result,'Duplicate table inspection setting')
            result[key]=item
        return result
    def invalid_number(_):raise ValueError('Invalid table inspection number')
    try:
        configuration=json.loads(value,object_pairs_hook=unique,
            parse_constant=invalid_number)
    except (json.JSONDecodeError,TypeError,RecursionError) as error:
        raise ValueError('Invalid table inspection settings') from error
    normalize_table_options(configuration)
    # Preserve the approved active-field shape for host round-trip verification.
    # Reader normalization can add internal/inactive defaults (e.g. PyArrow
    # settings for a native Parquet request); those are not authored settings.
    return configuration


def parse_file_options_json(value):
    require(type(value) is str and len(value.encode())<=32768 and '\0' not in value,'File inspection settings exceed transport limit')
    def unique(pairs):
        result={}
        for key,item in pairs:
            require(key not in result,'Duplicate File Input setting');result[key]=item
        return result
    try:configuration=json.loads(value,object_pairs_hook=unique,parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Invalid File Input number')))
    except (json.JSONDecodeError,RecursionError) as error:raise ValueError('Invalid File Input settings') from error
    if __package__:from .file_input_options import validate_configuration
    else:from file_input_options import validate_configuration
    validate_configuration(configuration)
    return configuration


def parse_contract(value,base=None):
    args=value['arguments'];files=value['inputFiles']
    require(isinstance(args,list) and len(args)==4
        and all(isinstance(arg,str) and arg and '\0' not in arg
            and len(arg.encode())<=(32768 if i==3 and args[2] in ('--table-options','--file-options') else 4096) for i,arg in enumerate(args))
        and args[0]=='--inspect-data' and args[2] in ('--csv-separator','--csv-options','--table-options','--file-options'),'Invalid source inspection arguments')
    if args[2]=='--csv-separator':
        require(len(args[3])==1 and args[3] not in ('\r','\n','\0'),'Invalid source inspection separator')
    elif args[2]=='--csv-options':
        parse_options_json(args[3])
    elif args[2]=='--file-options':parse_file_options_json(args[3])
    else:parse_table_options_json(args[3])
    if args[2]=='--file-options':
        require(isinstance(files,list) and 1<=len(files)<=9,'File inspection requires one manifest and at most eight captured members')
        if __package__:from .preparation_operation_contract import file_limit
        else:from preparation_operation_contract import file_limit
        inventory={}
        for entry in files:
            require(type(entry) is dict and set(entry)=={'path','bytes','sha256'},'Invalid File Input inspection inventory')
            parts=relative_parts(entry['path']);require(parts[:2]==['input','file-collections'],'File Input inspection inventory escapes collection')
            require(entry['path'] not in inventory and type(entry['bytes']) is int and 0<=entry['bytes']<=file_limit(entry['path']),'Invalid collection inspection size')
            hash_value(entry['sha256']);inventory[entry['path']]=entry
        manifests=[name for name in inventory if name.endswith('/manifest.json')]
        require(len(manifests)==1,'File inspection requires one collection manifest')
        name=manifests[0];parts=relative_parts(name)
        root=Path(base) if base is not None else Path(args[1]).parents[len(parts)-1]
        require(args[1]==str(root/name),'File inspection binding escapes captured collection')
        require(parts[2]==inventory[name]['sha256'],'File inspection manifest pin differs from path')
        return inventory
    require(isinstance(files,list) and len(files)==1,'Inspection requires exactly one source')
    entry=files[0]
    require(isinstance(entry,dict) and set(entry)=={'path','bytes','sha256'},'Invalid inspection source entry')
    parts=relative_parts(entry['path'])
    require(len(parts)==3 and parts[:2]==['input','data'],'Invalid inspection source path')
    require(type(entry['bytes']) is int and 0<=entry['bytes']<=10000000,'Inspection source exceeds size bound')
    hash_value(entry['sha256'])
    path=args[1]
    require(path.startswith('/') and str(Path(path))==path
        and all(part not in ('','.','..') for part in path.split('/')[1:]),'Noncanonical inspection source path')
    root=Path(base) if base is not None else Path(path).parent.parent.parent
    require(path==str(root/entry['path']),'Inspection source escapes retained operation')
    return {entry['path']:entry}


def verify_inputs(value,root,cache):
    inventory=parse_contract(value,root)
    for name,entry in inventory.items():verify_pinned_file(root,name,entry,cache)
    if value['arguments'][2]=='--file-options':
        if __package__:
            from .file_collection_contract import parse_manifest,owned_member_paths
            from .worker_authorization import directory,read_regular,sha
        else:
            from file_collection_contract import parse_manifest,owned_member_paths
            from worker_authorization import directory,read_regular,sha
        import os
        name=next(name for name in inventory if name.endswith('/manifest.json'));parts=relative_parts(name)
        fd=directory(str(Path(root).joinpath(*parts[:-1])))
        try:raw=read_regular(fd,parts[-1],2000000)
        finally:os.close(fd)
        require(len(raw)==inventory[name]['bytes'] and sha(raw)==inventory[name]['sha256'],'File inspection manifest changed')
        document=parse_manifest(raw)
        if __package__:from .file_input_options import validate_configuration
        else:from file_input_options import validate_configuration
        selected=validate_configuration(parse_file_options_json(value['arguments'][3]))
        require(document['selection']=={key:selected[key] for key in document['selection']},'Collection selection differs from inspection configuration')
        require(len(document['members'])<=8,'File inspection exceeds eight selected files')
        require(owned_member_paths(document,name,inventory)==set(inventory),'Unbound File Input inspection member')
