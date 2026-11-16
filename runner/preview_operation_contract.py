"""Standard-library preview launch contract; engine owns tensor/data semantics."""
import json
import os
from pathlib import Path

if __package__:
    from .worker_input_contract import relative_parts as _parts, verify_pinned_file, verify_completed_inventory
    from .worker_authorization import require, hash_value, directory, read_regular, signature, object_json, parse_bounded_options
else:
    from worker_input_contract import relative_parts as _parts, verify_pinned_file, verify_completed_inventory
    from worker_authorization import require, hash_value, directory, read_regular, signature, object_json, parse_bounded_options

BASE={'--graph','--output','--device','--seed'}
SPLIT={'--sample-limit','--split-seed','--validation-percent'}
MAPPING={'--feature-columns-json','--target-column'}
PREPARED={'--prepared-dataflow','--prepared-manifest-sha256','--prepared-graph-digest',
          '--prepared-training-node','--prepared-training-port','--prepared-validation-node','--prepared-validation-port'}
ALLOWED=BASE|SPLIT|MAPPING|PREPARED|{'--preview','--stop-at','--resume-from','--cache','--checkpoint','--dataset-csv'}
FIXED_LIMITS={'input/graph.json':2000000,'input/weights/checkpoint.pt':100000000,'input/weights/graph.json':2000000,
              'input/cache/activation-cache.pt':32000000,'input/cache/activation-cache.json':100000,
              'input/preparation/manifest.json':2000000}


if __package__:from .preparation_artifact_limits import request_policy,prepared_file_limit,inventory_limit
else:from preparation_artifact_limits import request_policy,prepared_file_limit,inventory_limit


def file_limit(name):
    parts=_parts(name)
    if name in FIXED_LIMITS:return FIXED_LIMITS[name]
    if len(parts)==3 and parts[:2]==['input','data']:return 10000000
    if len(parts)==3 and parts[:2]==['input','preparation']:return 50000000
    raise ValueError('Unsupported preview input file')


def _integer(value,minimum,maximum):
    require(value.isascii() and value.isdecimal() and str(int(value))==value and minimum<=int(value)<=maximum, 'Invalid preview integer setting')
    return int(value)


def _name(value):
    require(isinstance(value,str) and bool(value.strip()) and len(value)<=200 and '\0' not in value,'Invalid preview name')


def validate_prepared_partition_selection(options,root,required):
    require(PREPARED<=set(options) and not (SPLIT|{'--dataset-csv'})&set(options),'Invalid prepared partition options')
    require(options['--prepared-dataflow']==str(root/'input/preparation/manifest.json'),'Prepared preview path differs')
    for key in ('--prepared-manifest-sha256','--prepared-graph-digest'):hash_value(options[key])
    for key in ('--prepared-training-node','--prepared-validation-node'):
        require(1<=len(options[key])<=128,'Invalid prepared node identity')
    for key in ('--prepared-training-port','--prepared-validation-port'):
        require(1<=len(options[key])<=200,'Invalid prepared port identity')
    required.add('input/preparation/manifest.json')


def validate_data_selection(options,root,required):
    prepared='--prepared-dataflow' in options
    if prepared:
        require(PREPARED|MAPPING<=set(options) and not (SPLIT|{'--dataset-csv'})&set(options),'Invalid prepared preview options')
        validate_prepared_partition_selection(options,root,required)
    else:
        require(SPLIT<=set(options) and not PREPARED&set(options),'Missing split settings or orphan prepared options')
        sample=_integer(options['--sample-limit'],0,100000);require(sample==0 or sample>=20,'Invalid sample limit')
        _integer(options['--split-seed'],0,2**32-1);_integer(options['--validation-percent'],5,50)
        if '--dataset-csv' in options:
            require(MAPPING<=set(options),'Missing CSV column mapping')
            path=options['--dataset-csv'];name=Path(path).name;relative='input/data/'+name
            _parts(relative)
            require(path==str(root/relative),'CSV input escapes retained operation');required.add(relative)
        else:require(not MAPPING&set(options),'Column mapping requires selected data')
    if MAPPING<=set(options):
        try:columns=json.loads(options['--feature-columns-json'])
        except (TypeError,ValueError):raise ValueError('Invalid preview feature columns') from None
        require(isinstance(columns,list) and 1<=len(columns)<=128,'Invalid preview feature columns')
        for name in columns:
            require(isinstance(name,str) and bool(name.strip()) and '\0' not in name, 'Invalid preview feature column')
            if prepared:_name(name)
        require(len(set(columns))==len(columns),'Repeated preview feature column')
        if not prepared:require(len(columns)==2,'CSV preview requires two feature columns')
        target=options['--target-column'];require(bool(target.strip()),'Invalid preview target column')
        if prepared:_name(target)
        require(target not in columns,'Target cannot also be a feature')
    return prepared


def parse_contract(value,base=None):
    workload=request_policy(value);retained=workload.limit('retainedBytes') if workload else 200000000
    options=parse_bounded_options(value['arguments'],ALLOWED,{'--preview'},maximum_count=64,maximum_argument_bytes=4096,maximum_total_bytes=262144)
    require(BASE<=set(options),'Missing required preview options')
    graph=options['--graph']
    require(graph.startswith('/') and str(Path(graph))==graph and all(p not in ('','.','..') for p in graph.split('/')[1:]), 'Noncanonical preview graph path')
    inferred=Path(graph).parent.parent
    root=Path(base) if base is not None else inferred
    require(graph==str(root/'input/graph.json') and options['--output']==str(root/'artifacts'), 'Preview paths escape retained operation')
    require(options['--device'] in ('auto','cpu','mps'),'Invalid preview device')
    _integer(options['--seed'],0,2**32-1)
    required={'input/graph.json'}
    if '--preview' in options:
        require('--stop-at' in options and not {'--resume-from','--cache'}&set(options),'Invalid preview selection')
    else:
        require({'--resume-from','--cache'}<=set(options) and '--stop-at' not in options,'Invalid preview resume selection')
        require(options['--cache']==str(root/'input/cache/activation-cache.pt'),'Preview cache path differs')
        required|={'input/cache/activation-cache.pt','input/cache/activation-cache.json'}
    if '--checkpoint' in options:
        require(options['--checkpoint']==str(root/'input/weights/checkpoint.pt'),'Preview checkpoint path differs')
        required|={'input/weights/checkpoint.pt','input/weights/graph.json'}
    prepared=validate_data_selection(options,root,required)
    files=value['inputFiles'];require(isinstance(files,list) and 1<=len(files)<=(inventory_limit(workload)+6 if workload else 306),'Invalid preview inventory count')
    inventory={};total=0;preparation_total=0
    for entry in files:
        require(isinstance(entry,dict) and set(entry)=={'path','bytes','sha256'},'Invalid preview inventory entry')
        name=entry['path'];limit=file_limit(name)
        if name.startswith('input/preparation/'):limit=prepared_file_limit(name.split('/')[-1],workload)
        require(name not in inventory and type(entry['bytes']) is int and 0<=entry['bytes']<=limit,'Invalid preview input size or duplicate')
        hash_value(entry['sha256']);inventory[name]=entry;total+=entry['bytes']
        if prepared and name.startswith('input/preparation/'):
            required.add(name)
            if name!='input/preparation/manifest.json':preparation_total+=entry['bytes']
    require(total<=retained+140000000 and preparation_total<=retained,'Preview captured inputs exceed total size bound')
    require(set(inventory)==required,'Preview input inventory differs from selected options')
    if prepared:require(inventory['input/preparation/manifest.json']['sha256']==options['--prepared-manifest-sha256'],'Prepared manifest pin differs')
    return options,inventory


def verify_inputs(value,root,cache):
    options,inventory=parse_contract(value,root)
    for name,entry in inventory.items():verify_pinned_file(root,name,entry,cache)
    if '--prepared-dataflow' in options:
        verify_completed_inventory(root,'input/preparation',inventory,options['--prepared-manifest-sha256'],options['--prepared-graph-digest'],admission_policy=request_policy(value))
    output=Path(root)/'artifacts'
    try:fd=directory(str(output))
    except FileNotFoundError:pass
    else:os.close(fd)
