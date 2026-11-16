"""Measured source selection and grouping through real native production helpers."""
import copy
import math
import torch
from torch.utils.data import TensorDataset, Subset, IterableDataset
if __package__:
    from .dataset_sources import make_source
    from .dataset_loader_configuration import validate_recipe
    from .batch_loading import open_epoch
else:
    from dataset_sources import make_source
    from dataset_loader_configuration import validate_recipe
    from batch_loading import open_epoch

TASKS=('dataset-indexed-observations','dataset-aligned-tensors','dataset-selected-view','dataset-finite-stream','loader-complete-and-short-groups','loader-repeatable-mixing','loader-aligned-sequences','loader-continue-grouping')

def compile_graph(submission):
    if type(submission) is not dict or set(submission)!={'schemaVersion','kind','taskID','graph'} or type(submission['schemaVersion']) is not int or submission['schemaVersion']!=1 or submission['kind']!='dataLoadingLearningGraph' or submission['taskID'] not in TASKS:
        raise ValueError('Use this versioned Dataset/DataLoader lesson construction, never answers or code.')
    graph=submission['graph']
    if type(graph) is not dict or set(graph)!={'schemaVersion','name','nodes','edges'} or type(graph['schemaVersion']) is not int or graph['schemaVersion']!=2 or type(graph['name']) is not str or type(graph['nodes']) is not list or len(graph['nodes'])!=3 or type(graph['edges']) is not list or len(graph['edges'])!=2:
        raise ValueError('Use one Dataset, one DataLoader and the supplied Train.')
    nodes={}
    for node in graph['nodes']:
        if type(node) is not dict or set(node)!={'id','type','name','x','y','config'} or type(node['id']) is not str or not 1<=len(node['id'].encode())<=128 or node['id'] in nodes or type(node['name']) is not str or len(node['name'])>256 or any(type(node[k]) not in (int,float) or not math.isfinite(node[k]) for k in ('x','y')):
            raise ValueError('Use uniquely identified components with finite positions.')
        nodes[node['id']]=node
    if sorted(n['type'] for n in nodes.values())!=['data_loader','dataset_source','train']:
        raise ValueError('Ordinary Input cannot replace the Dataset recipe.')
    by_type={n['type']:n for n in nodes.values()}
    if by_type['train']['config']!={}:raise ValueError('The checker supplies Train settings and source authority.')
    required={(by_type[t]['id'],'out',by_type['train']['id'],p) for t,p in (('dataset_source','dataset'),('data_loader','loader'))}
    actual=set();identities=set()
    for edge in graph['edges']:
        if type(edge) is not dict or set(edge)!={'id','source','sourcePort','target','targetPort'} or any(type(v) is not str or not v for v in edge.values()) or edge['id'] in identities:
            raise ValueError('Use unique named recipe connections.')
        identities.add(edge['id']);actual.add(tuple(edge[k] for k in ('source','sourcePort','target','targetPort')))
    if actual!=required:raise ValueError('Connect both recipe outputs to their matching Train inputs.')
    recipes=[]
    for component in ('dataset_source','data_loader'):
        recipes.append(validate_recipe(dict(schemaVersion=1,componentID=component,configurationVersion=1,config=by_type[component]['config'])))
    return submission['taskID'],*recipes

def _parse(task,inputs,parse_tensor):
    roles={'sourceRows','desiredRows','rowIDs','seed'}
    if task in ('loader-aligned-sequences','loader-continue-grouping'):roles.add('sourceLengths')
    if task=='loader-continue-grouping':roles.add('sourceWeights')
    if type(inputs) is not dict or set(inputs)!=roles:raise ValueError('Supply the exact source, target and observation identity fields.')
    parsed={}
    for key,raw in inputs.items():
        shape,values=parse_tensor(raw)
        if type(shape) not in (list,tuple) or not shape or len(shape)>3 or any(type(n) is not int or not 1<=n<=32 for n in shape) or len(values)!=math.prod(shape) or any(type(v) not in (int,float) or not math.isfinite(v) or abs(v)>1_000_000 for v in values):
            # Private seed is allowed its explicit UInt32 range below.
            if key!='seed':raise ValueError('Source arrays must contain bounded finite numeric cells.')
        parsed[key]=(list(shape),values)
    count=parsed['sourceRows'][0][0]
    expected=[count,3,1] if task=='loader-aligned-sequences' else [count,3] if task=='loader-continue-grouping' else [count,2]
    if not 5<=count<=32 or parsed['sourceRows'][0]!=expected or parsed['desiredRows'][0]!=[count,1] or parsed['rowIDs'][0]!=[count]:raise ValueError('Keep source fields and desired outcomes aligned by observation.')
    ids=parsed['rowIDs'][1]
    if len(set(ids))!=count or any(v!=int(v) or v<0 for v in ids):raise ValueError('Observation IDs must be distinct bounded whole numbers.')
    seed_shape,seed=parsed['seed']
    if seed_shape!=[1] or len(seed)!=1 or type(seed[0]) not in (int,float) or not math.isfinite(seed[0]) or seed[0]!=int(seed[0]) or not 0<=seed[0]<2**32:raise ValueError('The private fixture seed must be one UInt32 whole number.')
    if 'sourceLengths' in parsed and (parsed['sourceLengths'][0]!=[count] or any(v!=int(v) or not 0<=v<=3 for v in parsed['sourceLengths'][1])):raise ValueError('True sequence/item lengths must align with whole observations.')
    if 'sourceWeights' in parsed and (parsed['sourceWeights'][0]!=[count,3] or any(v!=int(v) or not 0<=v<16 for v in parsed['sourceRows'][1])):raise ValueError('Whole-bag IDs and real item weights must have aligned source cells.')
    return parsed

def _tensor(value):return {'shape':list(value.shape),'values':value.detach().cpu().reshape(-1).tolist()}
def _measure(source,recipe,check_cancelled):
    with open_epoch(source,recipe,0,check_cancelled=check_cancelled) as batches:return list(batches)
def _groups(batches,config,sequence=False):
    width=config['batch_size'];groups=len(batches)
    ids=torch.full((groups,width),-1,dtype=torch.int64);targets=torch.zeros(groups,width);lengths=[]
    rows=torch.zeros(groups,width,3,1) if sequence else None
    true_lengths=torch.zeros(groups,width) if sequence else None
    for i,batch in enumerate(batches):
        n=len(batch.row_ids);lengths.append(n);ids[i,:n]=torch.tensor(batch.row_ids);targets[i,:n]=batch.roles['targets'].reshape(-1)
        if sequence:rows[i,:n]=batch.input;true_lengths[i,:n]=batch.roles['lengths'].reshape(-1)
    result={'batchRowIDs':_tensor(ids),'batchTargets':_tensor(targets)}
    if sequence:result.update(batchedInputs=_tensor(rows),sequenceLengths=_tensor(true_lengths))
    else:result['batchLengths']={'shape':[groups],'values':lengths}
    return result

def execute(submission,fixtures,evidence,stopped,parse_tensor):
    task,dataset,loader=compile_graph(submission)
    for fixture in fixtures:
        if stopped():return
        parsed=_parse(task,fixture['inputs'],parse_tensor)
        evidence['maximumTensorElements']=max(evidence['maximumTensorElements'],16*sum(len(v) for _,v in parsed.values()))
        if stopped():return
        def check_cancelled():
            if stopped():raise InterruptedError('Data loading lesson cancelled.')
        if task=='loader-continue-grouping':
            measured=_continuation(parsed,dataset,loader,check_cancelled)
            evidence['fixtures'].append({'fixtureID':fixture['id'],'tensors':measured,'partitions':{}})
            continue
        rows=torch.tensor(parsed['sourceRows'][1],dtype=torch.float32).reshape(parsed['sourceRows'][0])
        targets=torch.tensor(parsed['desiredRows'][1],dtype=torch.float32).reshape(parsed['desiredRows'][0])
        ids=[int(v) for v in parsed['rowIDs'][1]];roles={'targets':targets}
        if 'sourceLengths' in parsed:roles['lengths']=torch.tensor(parsed['sourceLengths'][1],dtype=torch.int64)
        source=make_source(rows,roles,ids,dataset)
        if task in TASKS[:4]:
            records=list(source.native) if isinstance(source.native,IterableDataset) else [source.native[i] for i in range(len(source.native))]
            kind=1 if isinstance(source.native,TensorDataset) else 2 if isinstance(source.native,Subset) else 3 if isinstance(source.native,IterableDataset) else 0
            measured={'selectedRowIDs':{'shape':[len(records)],'values':[r.row_id for r in records]},'selectedInputs':_tensor(torch.cat([r.input for r in records])),'selectedTargets':_tensor(torch.cat([r.roles['targets'] for r in records])),'sourceKind':{'shape':[1],'values':[kind]}}
        else:
            batches=_measure(source,loader,check_cancelled)
            if task=='loader-repeatable-mixing':
                again=_measure(source,loader,check_cancelled)
                changed=copy.deepcopy(loader);changed['config']['seed']=(changed['config']['seed']+1)%(2**63)
                different=_measure(source,changed,check_cancelled)
                order=[i for b in batches for i in b.row_ids];repeat=[i for b in again for i in b.row_ids];other=[i for b in different for i in b.row_ids]
                alignment=[]
                observed={identity:(b.input[j],b.roles['targets'][j]) for b in batches for j,identity in enumerate(b.row_ids)}
                for j,identity in enumerate(ids):
                    if identity not in observed:alignment.extend([1,1,1]);continue
                    actual,target=observed[identity];alignment.extend((actual-rows[j]).reshape(-1).tolist()+(target-targets[j]).reshape(-1).tolist())
                membership=[order.count(i)-1 for i in ids]
                checks=[int(order==repeat),int(order!=other),int(sorted(order)==sorted(ids)),int(not any(alignment))]
                measured={'shuffleChecks':{'shape':[4],'values':checks},'selectionDifference':{'shape':[len(ids)],'values':membership},'alignmentDifference':{'shape':[len(ids),3],'values':alignment}}
            else:
                measured=_groups(batches,loader['config'],task=='loader-aligned-sequences')
                if task=='loader-complete-and-short-groups':
                    dropped=copy.deepcopy(loader);dropped['config']['drop_last']=True
                    measured['dropLastRowIDs']=_groups(_measure(source,dropped,check_cancelled),dropped['config'])['batchRowIDs']
        check_cancelled()
        evidence['fixtures'].append({'fixtureID':fixture['id'],'tensors':measured,'partitions':{}})


def _same(first,second):
    if isinstance(first,torch.Tensor):return isinstance(second,torch.Tensor) and first.dtype==second.dtype and first.shape==second.shape and torch.equal(first,second)
    if type(first) is dict:return type(second) is dict and first.keys()==second.keys() and all(_same(first[k],second[k]) for k in first)
    if type(first) in (list,tuple):return type(second) is type(first) and len(first)==len(second) and all(_same(a,b) for a,b in zip(first,second))
    return type(first) is type(second) and first==second


def _continuation(parsed,dataset,loader,check_cancelled):
    # The original rows become a genuine completed preparation with held-out
    # data. No in-memory provenance or lesson-only packet impersonates it.
    import csv
    import hashlib
    import json
    import multiprocessing
    from pathlib import Path
    import tempfile
    if __package__:
        from . import engine,dataflow
        from .dataflow_cli import run_dataflow
        from .prepared_bag_loader import load_prepared_bags
        from .embedding_bag_configuration import DEFAULTS as BAG_DEFAULTS
        from .loss_configuration import canonical_configuration as loss_config
        from .optimizer_configuration import canonical_configuration as optimizer_config
    else:
        import engine,dataflow
        from dataflow_cli import run_dataflow
        from prepared_bag_loader import load_prepared_bags
        from embedding_bag_configuration import DEFAULTS as BAG_DEFAULTS
        from loss_configuration import canonical_configuration as loss_config
        from optimizer_configuration import canonical_configuration as optimizer_config
    count=parsed['sourceRows'][0][0];seed=int(parsed['seed'][1][0])
    owned_before={p.pid for p in multiprocessing.active_children()}
    previous_deterministic=torch.are_deterministic_algorithms_enabled();previous_warn=torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        with tempfile.TemporaryDirectory(prefix='kwipml-loading-lesson-') as directory:
            root=Path(directory);source=root/'source.csv'
            names=['i0','i1','i2','w0','w1','w2','length','desired','observation']
            with source.open('w',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(names)
                for row in range(count):
                    writer.writerow([int(v) for v in parsed['sourceRows'][1][row*3:row*3+3]]+list(parsed['sourceWeights'][1][row*3:row*3+3])+[int(parsed['sourceLengths'][1][row]),parsed['desiredRows'][1][row],int(parsed['rowIDs'][1][row])])
            # This current lesson explicitly reserves all three populations from
            # supplied whole bags (nine rows: 5 Training, 2 Validation, 2 Test).
            # It does not claim equivalence with retired two-way Split membership;
            # the independent oracle compares fresh and continued execution on
            # the same pinned current preparation. Reserved Test is never loaded.
            flow={'schemaVersion':1,'nodes':[{'id':'s','operator':'kwip.data.sourceCSV','config':{'separator':','}},{'id':'c','operator':'kwip.data.columns','config':{'rows':[{'source':n,'included':True,'outputName':n,'targetType':'int64' if n in ('i0','i1','i2','length','observation') else 'float64'} for n in names],'newColumns':'drop','castFailure':'error','trimText':False,'missingPolicy':'error'}},{'id':'p','operator':'kwip.data.partition','config':{'strategy':'random','seed':seed,'validationPercent':25,'testPercent':25}}],'edges':[{'from':{'node':'s','port':'table'},'to':{'node':'c','port':'table'}},{'from':{'node':'c','port':'table'},'to':{'node':'p','port':'table'}}]}
            flow_path=root/'dataflow.json';flow_path.write_text(json.dumps(flow))
            preparation=root/'prepared';run_dataflow(flow_path,{'s':str(source)},preparation,lambda _:check_cancelled())
            source_config={'layout':'packed','maximum_items_per_bag':3,'per_sample_weights':True,'include_last_offset':True}
            objective={'schemaVersion':1,'componentID':'loss_mse','configurationVersion':1,'config':loss_config('loss_mse',{})}
            mapping={'schemaVersion':1,'componentID':'loss_mse','inputColumns':['i0','i1','i2'],'roles':[{'role':'target','columns':['desired'],'sampleShape':[1],'dtype':'float32'}],'tripletSourceGroups':[]}
            vocabulary=max(int(v) for v in parsed['sourceRows'][1])+1
            prepared=load_prepared_bags(preparation/'manifest.json','p','training','p','validation',source_config=source_config,mapping={'schemaVersion':1,'indicesColumns':['i0','i1','i2'],'weightColumns':['w0','w1','w2'],'lengthColumn':'length'},num_embeddings=vocabulary,embedding_dim=2,mode='sum',expected_manifest_sha256=hashlib.sha256((preparation/'manifest.json').read_bytes()).hexdigest(),expected_graph_digest=dataflow.digest(flow),objective_mapping=mapping,objective_recipe=objective,prediction_shape=[1])
            bag=dict(BAG_DEFAULTS,num_embeddings=vocabulary,embedding_dim=2,mode='sum',layout='packed',per_sample_weights=True,include_last_offset=True)
            graph={'schemaVersion':2,'nodes':[{'id':'source','type':'bag_input','config':source_config},{'id':'bag','type':'embedding_bag','config':bag},{'id':'learn','type':'linear','config':{'in_features':2,'out_features':1,'bias':True}},{'id':'result','type':'output','config':{}}],'edges':[{'id':role,'source':'source','sourcePort':role,'target':'bag','targetPort':role} for role in ('indices','offsets','per_sample_weights')]+[{'id':'learned','source':'bag','sourcePort':'out','target':'learn','targetPort':'in'},{'id':'result','source':'learn','sourcePort':'out','target':'result','targetPort':'in'}]}
            optimizer={'schemaVersion':1,'componentID':'optimizer_sgd','configurationVersion':1,'config':optimizer_config('optimizer_sgd',{'lr':.02,'momentum':.5})}
            options=dict(seed=seed,batch_size=loader['config']['batch_size'],device='cpu',prepared=prepared,dataset_recipe=dataset,loader_recipe=loader,objective_recipe=objective,target_mapping=mapping,optimizer_recipe=optimizer,check_cancelled=check_cancelled)
            full_trace=[];resume_trace=[]
            engine.run(graph,root/'full',epochs=2,gradient_observer=full_trace.append,**options)
            engine.run(graph,root/'parent',epochs=1,**options)
            parent=root/'parent/checkpoint.pt';parent_bytes=parent.read_bytes()
            engine.run(graph,root/'child',epochs=2,resume_checkpoint=parent,gradient_observer=resume_trace.append,**options)
            full=torch.load(root/'full/checkpoint.pt',weights_only=True);child=torch.load(root/'child/checkpoint.pt',weights_only=True)
            full_order=[i for event in full_trace if event['phase']=='windowStart' and event['epoch']==2 for i in event['rowIndices']]
            child_order=[i for event in resume_trace if event['phase']=='windowStart' and event['epoch']==2 for i in event['rowIndices']]
            # Map actual training positions back to original supplied identities;
            # held-out rows have rank -1 in both executions, a real difference 0.
            manifest=json.loads((preparation/'manifest.json').read_bytes())
            artifact=next(a for a in manifest['artifacts'] if a.get('nodeID')=='p' and a.get('port')=='training')
            table=json.loads((preparation/artifact['path']).read_bytes());column=next(i for i,c in enumerate(table['columns']) if c['name']=='observation')
            original=[row[column] for row in table['rows']]
            full_ranks={original[position]:rank for rank,position in enumerate(full_order)};child_ranks={original[position]:rank for rank,position in enumerate(child_order)}
            differences=[full_ranks.get(identity,-1)-child_ranks.get(identity,-1) for identity in parsed['rowIDs'][1]]
            loading=json.loads((root/'child/manifest.json').read_bytes())['dataLoading']
            config=loading['loaderRecipe']['config']
            joined=not ({p.pid for p in multiprocessing.active_children()}-owned_before)
            # These public lesson controls are part of the requested outcome.
            # Comparing two runs with the same wrong recipe is insufficient.
            requested_source={'kind':'indexed','indices':[],'max_rows':100000}
            requested_loader={'batch_size':4,'shuffle':True,'drop_last':False,'seed':42,'num_workers':2,'prefetch_factor':1,'timeout':5.0,'collation':'whole_bags'}
            requested_policy=loading['datasetRecipe']['config']==requested_source and config==requested_loader
            checks=[int(full_order==child_order and bool(full_order)),int(requested_policy and full['contract']==child['contract'] and full['dataLoadingState']==child['dataLoadingState']),int(_same(full['modelState'],child['modelState']) and _same(full['optimizerState'],child['optimizerState']) and _same(full['rngState'],child['rngState'])),int(joined)]
            after=parent.read_bytes();byte_difference=sum(a!=b for a,b in zip(parent_bytes,after))+abs(len(parent_bytes)-len(after))
            return {'continuationChecks':{'shape':[4],'values':checks},'readerPolicy':{'shape':[2],'values':[config['num_workers'],config['prefetch_factor'] or 0]},'parentByteDifference':{'shape':[1],'values':[byte_difference]},'epochOrderDifference':{'shape':[count],'values':differences}}
    finally:
        torch.use_deterministic_algorithms(previous_deterministic,warn_only=previous_warn)
