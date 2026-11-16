"""Bounded standalone training inputs; tensor semantics remain in the engine."""
import hashlib
import json
import math
import os
from pathlib import Path
if __package__:
    from .capture_policy import validate_training_sample_capture_policy
    from .worker_authorization import require,hash_value,parse_bounded_options,directory,read_regular,object_json,validate_gradient_arguments,validate_loading_arguments
    from .worker_input_contract import relative_parts,verify_pinned_file,verify_completed_inventory,validate_bag_arguments,verify_bag_graph_inputs,requires_bag_admission
    from .preview_operation_contract import BASE,SPLIT,MAPPING,PREPARED,validate_data_selection,validate_prepared_partition_selection,_integer
else:
    from capture_policy import validate_training_sample_capture_policy
    from worker_authorization import require,hash_value,parse_bounded_options,directory,read_regular,object_json,validate_gradient_arguments,validate_loading_arguments
    from worker_input_contract import relative_parts,verify_pinned_file,verify_completed_inventory,validate_bag_arguments,verify_bag_graph_inputs,requires_bag_admission
    from preview_operation_contract import BASE,SPLIT,MAPPING,PREPARED,validate_data_selection,validate_prepared_partition_selection,_integer

OBJECTIVE={'--objective-recipe-json','--objective-target-mapping-json','--objective-scalarization'}
SETTINGS={'--epochs','--batch-size','--learning-rate'}
GRADIENT={'--gradient-every-epochs','--gradient-max-batches-per-epoch','--gradient-max-captures'}
PARENTS={'--resume-checkpoint','--finetune-checkpoint'}
COMPOSITE={'--composite-source','--composite-source-sha256'}
ALLOWED=OBJECTIVE|BASE|SETTINGS|SPLIT|MAPPING|PREPARED|GRADIENT|PARENTS|COMPOSITE|{'--dataset-recipe-json','--loader-recipe-json','--bag-input-mapping-json','--capture-gradients','--dataset-csv','--training-control-json','--training-progress-json','--sample-capture-json','--gradient-policy-json','--trainability-json','--derivative-capture-json','--initialization-recipe-json','--optimizer-recipe-json','--scheduler-recipe-json'}
LIMITS={'input/graph.json':2000000,'input/preparation/manifest.json':2000000,'input/model-registry.json':2000000,
        'input/continuation-parent/checkpoint.pt':100000000,'input/continuation-parent/manifest.json':20000000,
        'input/continuation-parent/metrics.json':4000000,'input/continuation-parent/rate-trace.json':4000000,'input/composite/source.json':2000000}


if __package__:from .preparation_artifact_limits import request_policy,prepared_file_limit,inventory_limit
else:from preparation_artifact_limits import request_policy,prepared_file_limit,inventory_limit


def parse_contract(value,base=None):
    workload=request_policy(value);retained=workload.limit('retainedBytes') if workload else 200000000
    has_objective=any(arg.split('=',1)[0] in OBJECTIVE for arg in value['arguments'] if isinstance(arg,str))
    has_initialization=any(arg in ('--dataset-recipe-json','--loader-recipe-json','--initialization-recipe-json','--gradient-policy-json','--trainability-json') for arg in value['arguments'] if isinstance(arg,str))
    has_bag='--bag-input-mapping-json' in value['arguments']
    options=parse_bounded_options(value['arguments'],ALLOWED,{'--capture-gradients'},maximum_argument_bytes=524288 if has_objective else 32768 if has_initialization or has_bag else 8192,maximum_total_bytes=786432 if has_objective else 262144)
    for key,text in options.items():
        if key not in OBJECTIVE|{'--dataset-recipe-json','--loader-recipe-json','--bag-input-mapping-json','--initialization-recipe-json','--gradient-policy-json','--trainability-json'} and isinstance(text,str):require(len(text.encode())<=8192,'Training argument exceeds its size bound')
    validate_gradient_arguments(options)
    validate_loading_arguments(options)
    if has_bag:validate_bag_arguments(options)
    objective_mapping=None
    if OBJECTIVE&set(options):
        require({'--objective-recipe-json','--objective-target-mapping-json'}<=set(options),'Objective requires recipe and typed mapping')
        require(len(options['--objective-recipe-json'].encode())<=32768 and len(options['--objective-target-mapping-json'].encode())<=524288,'Objective controls exceed their size bounds')
        require(options.get('--objective-scalarization','mean') in ('mean','sum'),'Invalid objective scalarization')
        if __package__:
            from .loss_configuration import validate_recipe as validate_objective_recipe
            from .loss_target_mapping import validate_target_mapping
        else:
            from loss_configuration import validate_recipe as validate_objective_recipe
            from loss_target_mapping import validate_target_mapping
        objective_recipe=validate_objective_recipe(json.loads(options['--objective-recipe-json']))
        objective_mapping=validate_target_mapping(json.loads(options['--objective-target-mapping-json']),objective_recipe)
        require('--prepared-dataflow' in options and not (MAPPING|SPLIT|{'--dataset-csv'})&set(options),'Objective requires prepared typed mapping without legacy or split controls')
    require(BASE|SETTINGS<=set(options),'Missing training settings')
    if '--training-control-json' in options:
        if __package__:from .training_control import validate_policy as validate_training_control
        else:from training_control import validate_policy as validate_training_control
        require(len(options['--training-control-json'].encode())<=1024,'Training control policy is oversized')
        validate_training_control(json.loads(options['--training-control-json']))
    if '--training-progress-json' in options:
        if __package__:from .training_progress import validate_policy
        else:from training_progress import validate_policy
        require(len(options['--training-progress-json'].encode())<=256,'Training reporting policy is oversized')
        validate_policy(json.loads(options['--training-progress-json']))
    if '--sample-capture-json' in options:
        require(len(options['--sample-capture-json'].encode())<=1024,'Sample capture policy exceeds its size bound')
        validate_training_sample_capture_policy(json.loads(options['--sample-capture-json']))
    if '--optimizer-recipe-json' in options:
        require(len(options['--optimizer-recipe-json'].encode())<=8192,'Optimizer recipe exceeds its size bound')
        if __package__:from .optimizer_configuration import validate_recipe
        else:from optimizer_configuration import validate_recipe
        validate_recipe(json.loads(options['--optimizer-recipe-json']))
    if '--scheduler-recipe-json' in options:
        require('--optimizer-recipe-json' in options,'Schedulers require an explicit optimizer recipe')
        require(len(options['--scheduler-recipe-json'].encode())<=8192,'Scheduler recipe exceeds its size bound')
        if __package__:from .scheduler_configuration import validate_recipe as validate_scheduler
        else:from scheduler_configuration import validate_recipe as validate_scheduler
        validate_scheduler(json.loads(options['--scheduler-recipe-json']))
    if '--initialization-recipe-json' in options:
        require(len(options['--initialization-recipe-json'].encode('utf-8'))<=32768,'Initialization recipe exceeds its size bound')
        if __package__:from .initialization import validate_recipe as validate_initialization_recipe
        else:from initialization import validate_recipe as validate_initialization_recipe
        validate_initialization_recipe(json.loads(options['--initialization-recipe-json']))
    graph=options['--graph']
    require(graph.startswith('/') and str(Path(graph))==graph and all(p not in ('','.','..') for p in graph.split('/')[1:]),'Noncanonical training graph path')
    root=Path(base) if base is not None else Path(graph).parent.parent
    require(graph==str(root/'input/graph.json') and options['--output']==str(root/'artifacts'),'Training paths escape retained operation')
    require(options['--device'] in ('auto','cpu','mps'),'Invalid training device')
    _integer(options['--seed'],0,2**32-1);_integer(options['--epochs'],1,10000);_integer(options['--batch-size'],1,100000)
    try:rate=float(options['--learning-rate'])
    except ValueError:raise ValueError('Invalid learning rate') from None
    require(math.isfinite(rate) and (0<=rate<=3.4028234663852886e38 if '--optimizer-recipe-json' in options else 0<rate<=1),'Invalid learning rate')
    required={'input/graph.json'}
    if has_bag:
        validate_prepared_partition_selection(options,root,required);prepared=True
        required.add('input/model-registry.json')
    elif objective_mapping is not None:
        validate_prepared_partition_selection(options,root,required);prepared=True
    else:prepared=validate_data_selection(options,root,required)
    if MAPPING<=set(options):
        columns=json.loads(options['--feature-columns-json'])
        require(len(columns)==2 and all(len(name)<=200 for name in columns),'Training requires two bounded feature columns')
    if '--capture-gradients' in options:
        require(GRADIENT<=set(options),'Missing gradient capture settings')
        for key,limit in [('--gradient-every-epochs',10000),('--gradient-max-batches-per-epoch',8),('--gradient-max-captures',1024)]:_integer(options[key],1,limit)
    else:require(not GRADIENT&set(options),'Orphan gradient capture settings')
    parent=PARENTS&set(options);require(len(parent)<=1,'Choose one continuation mode')
    if parent:
        require(options[next(iter(parent))]==str(root/'input/continuation-parent/checkpoint.pt'),'Continuation escapes retained operation')
        required|={name for name in LIMITS if name.startswith('input/continuation-parent/') and not name.endswith('/rate-trace.json')}
        if any(isinstance(item,dict) and item.get('path')=='input/continuation-parent/rate-trace.json' for item in value['inputFiles']):required.add('input/continuation-parent/rate-trace.json')
    if COMPOSITE&set(options):
        require(COMPOSITE<=set(options) and options['--composite-source']==str(root/'input/composite/source.json'),'Invalid composite source selection')
        hash_value(options['--composite-source-sha256']);required.add('input/composite/source.json')
    files=value['inputFiles'];require(isinstance(files,list) and 1<=len(files)<=(inventory_limit(workload)+7 if workload else 307),'Invalid training inventory count')
    inventory={};total=0;prepared_bytes=0
    for entry in files:
        require(isinstance(entry,dict) and set(entry)=={'path','bytes','sha256'},'Invalid training input entry')
        name=entry['path'];parts=relative_parts(name);limit=LIMITS.get(name)
        if limit is None:
            require(len(parts)==3 and parts[:2] in (['input','data'],['input','preparation']),'Unsupported training input path')
            limit=10000000 if parts[1]=='data' else prepared_file_limit(parts[-1],workload)
        require(name not in inventory and type(entry['bytes']) is int and 0<=entry['bytes']<=limit,'Invalid training input size or duplicate')
        hash_value(entry['sha256']);inventory[name]=entry;total+=entry['bytes']
        if prepared and name.startswith('input/preparation/'):
            required.add(name)
            if name!='input/preparation/manifest.json':prepared_bytes+=entry['bytes']
    require(total<=retained+130000000 and prepared_bytes<=retained and set(inventory)==required,'Training inventory differs from selection or exceeds bounds')
    if prepared:require(inventory['input/preparation/manifest.json']['sha256']==options['--prepared-manifest-sha256'],'Prepared manifest pin differs')
    if COMPOSITE<=set(options):require(inventory['input/composite/source.json']['sha256']==options['--composite-source-sha256'],'Composite source pin differs')
    return options,inventory


def verify_inputs(value,root,cache):
    options,inventory=parse_contract(value,root)
    for name,entry in inventory.items():verify_pinned_file(root,name,entry,cache)
    if requires_bag_admission(options,Path(root)/'input/graph.json'):
        require('--bag-input-mapping-json' in options,'Actual bag graph requires its captured bag mapping option')
        fd=directory(str(Path(root)/'input'))
        try:
            graph=object_json(read_regular(fd,'graph.json',2000000))
            registry=object_json(read_regular(fd,'model-registry.json',2000000))
        finally:os.close(fd)
        source=None
        if '--composite-source' in options:
            fd=directory(str(Path(root)/'input/composite'))
            try:source=object_json(read_regular(fd,'source.json',2000000))
            finally:os.close(fd)
        verify_bag_graph_inputs(graph,registry,source,options)
    if PARENTS&set(options):verify_parent_trace(root,inventory)
    if '--prepared-dataflow' in options:verify_completed_inventory(root,'input/preparation',inventory,options['--prepared-manifest-sha256'],options['--prepared-graph-digest'],admission_policy=request_policy(value))
    try:fd=directory(str(Path(root)/'artifacts'))
    except FileNotFoundError:pass
    else:os.close(fd)


def verify_parent_trace(root,inventory):
    """Validate parent-owned schedule dependencies before a worker loads Torch state."""
    prefix='input/continuation-parent/'
    fd=directory(str(Path(root)/'input/continuation-parent'))
    try:
        manifest=object_json(read_regular(fd,'manifest.json',20000000))
        checkpoint=manifest.get('checkpoint',{});contract=checkpoint.get('contract',{})
        require(isinstance(contract,dict),'Parent checkpoint contract is invalid')
        scheduler=contract.get('scheduler');pin=manifest.get('rateTrace')
        present=prefix+'rate-trace.json' in inventory
        if scheduler is None:
            require(not present and pin is None,'Parent without scheduler cannot carry a trace')
            return
        require(isinstance(scheduler,dict) and set(scheduler)=={'schemaVersion','recipe'} and type(scheduler.get('schemaVersion')) is int and scheduler['schemaVersion']==1,'Invalid parent scheduler contract')
        if __package__:
            from .scheduler_configuration import validate_recipe
        else:
            from scheduler_configuration import validate_recipe
        recipe=validate_recipe(scheduler['recipe'])
        require(recipe==scheduler['recipe'] and manifest.get('settings',{}).get('schedulerRecipe')==recipe,'Parent scheduler recipe differs')
        timing='afterOptimizerBatch' if recipe['componentID']=='scheduler_one_cycle_lr' else 'afterCompletedEpochValidation'
        require(present and isinstance(pin,dict) and set(pin)=={'schemaVersion','path','bytes','sha256','records','timing'} and type(pin.get('schemaVersion')) is int and pin['schemaVersion']==1 and pin.get('path')=='rate-trace.json' and type(pin.get('bytes')) is int and 1<=pin['bytes']<=4000000 and type(pin.get('records')) is int and 0<=pin['records']<=10000 and pin.get('timing')==timing,'Scheduled parent requires a bounded pinned trace')
        hash_value(pin['sha256'])
        entry=inventory[prefix+'rate-trace.json']
        require(entry['bytes']==pin['bytes'] and entry['sha256']==pin['sha256'],'Parent trace inventory differs from manifest')
        raw=read_regular(fd,'rate-trace.json',4000000)
        require(len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['sha256'],'Parent trace bytes differ')
        trace=object_json(raw)
        require(set(trace)=={'schemaVersion','recipe','timing','initialization','records'} and type(trace.get('schemaVersion')) is int and trace['schemaVersion']==1 and trace.get('recipe')==recipe and trace.get('timing')==timing and isinstance(trace.get('records'),list) and len(trace['records'])==pin['records'],'Parent trace identity/count differs')
    finally:os.close(fd)
