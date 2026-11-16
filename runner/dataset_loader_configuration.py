"""Strict standalone source/loading recipes; never source preparation authority."""
import copy
import json
import math

MAX_RECIPE_BYTES = 32768
MAX_ROWS = 100000
DATASET_DEFAULTS = dict(kind='indexed', indices=[], max_rows=100000)
LOADER_DEFAULTS = dict(batch_size=32, shuffle=False, drop_last=False, seed=42,
                      num_workers=0, prefetch_factor=None, timeout=0.0, collation='auto')


def _int(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} requires an exact integer in {low}..{high}.')


def validate_recipe(value):
    if type(value) is not dict or set(value) != {'schemaVersion','configurationVersion','componentID','config'}:
        raise ValueError('Choose a complete versioned Dataset or DataLoader recipe.')
    if any(type(value[k]) is not int or value[k] != 1 for k in ('schemaVersion','configurationVersion')):
        raise ValueError('Dataset/DataLoader recipes require schema and configuration version 1.')
    component=value['componentID'];config=value['config']
    if type(component) is not str or component not in ('dataset_source','data_loader'):
        raise ValueError('Unknown Dataset/DataLoader recipe component.')
    defaults=DATASET_DEFAULTS if component=='dataset_source' else LOADER_DEFAULTS
    if type(config) is not dict or set(config)!=set(defaults):
        raise ValueError('Supply exactly the complete selected recipe controls.')
    if component=='dataset_source':
        kind=config['kind'];indices=config['indices']
        if type(kind) is not str or kind not in ('indexed','boundedIterable','tensor','subset'):
            raise ValueError('Choose indexed, boundedIterable, tensor or subset source.')
        _int(config['max_rows'],1,MAX_ROWS,'max_rows')
        if kind!='boundedIterable' and config['max_rows']!=MAX_ROWS:
            raise ValueError('max_rows only changes boundedIterable; retain 100000 for other sources.')
        if type(indices) is not list or len(indices)>MAX_ROWS:
            raise ValueError('Subset indices require a bounded ordered list.')
        if kind=='subset':
            if not indices:raise ValueError('Subset must select at least one original training row.')
            for index in indices:_int(index,0,MAX_ROWS-1,'Subset index')
            if len(set(indices))!=len(indices):raise ValueError('Subset indices must be unique.')
        elif indices:raise ValueError('Only subset sources select indices.')
    else:
        _int(config['batch_size'],1,1024,'batch_size')
        _int(config['seed'],0,2**63-1,'seed')
        _int(config['num_workers'],0,2,'num_workers')
        for key in ('shuffle','drop_last'):
            if type(config[key]) is not bool:raise ValueError(f'{key} requires a Boolean.')
        if config['collation'] not in ('auto','fixed_stack','padded_sequence','whole_bags') or type(config['collation']) is not str:
            raise ValueError('Choose a named fixed, padded sequence or whole-bag collation.')
        timeout=config['timeout']
        if type(timeout) not in (int,float) or not math.isfinite(timeout) or not 0<=timeout<=30:
            raise ValueError('timeout requires a finite number in 0..30 seconds.')
        if config['num_workers']==0:
            if config['prefetch_factor'] is not None or timeout!=0:
                raise ValueError('Zero workers require null prefetch_factor and zero timeout.')
        else:
            _int(config['prefetch_factor'],1,2,'prefetch_factor')
            if timeout<=0:raise ValueError('Spawned workers require a positive bounded timeout.')
    result=copy.deepcopy(value)
    if component=='data_loader':result['config']['timeout']=float(config['timeout'])
    if len(json.dumps(result,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode())>MAX_RECIPE_BYTES:
        raise ValueError('Dataset/DataLoader recipe exceeds 32768 UTF-8 bytes; reduce selected indices.')
    return result


def capability(component):
    """Independent reviewed control descriptors, never forward operators."""
    common=dict(version='1.0.0',recipeSchemaVersion=1,configurationVersion=1,
        workflowSchemaVersion=13,compilerContractVersion=12,recipeByteMaximum=32768,
        devices=['cpu','mps'])
    if component=='dataset_source':
        return dict(common,adapterID='kwip.runner.dataset_source',
            modes=['indexed-source','bounded-iterable-source','tensor-source','selected-index-view'],
            sourceKinds=['indexed','boundedIterable','tensor','subset'],
            nativeTypes=['Dataset','IterableDataset','TensorDataset','Subset'],
            partition='trainingOnly',rowMaximum=100000,
            subset='uniqueOrderedOriginalTrainingIndices',iterable='restartableFinitePrefix',
            tensorSourceDomains=['tensor'],inputDomains=['tensor','bagPacket'])
    if component=='data_loader':
        return dict(common,adapterID='kwip.runner.data_loader',
            modes=['batch-order','shuffle-seed','short-final-batch','drop-last','collation','bounded-workers','continuation-order'],
            batchMinimum=1,batchMaximum=1024,workerMinimum=0,workerMaximum=2,
            prefetchMinimum=1,prefetchMaximum=2,timeoutMinimum=0.0,timeoutMaximum=30.0,
            workerConfiguration='zeroNullPrefetchZeroTimeoutOrPositivePrefetchPositiveTimeout',
            workingByteMaximum=268435456,collations=['auto','fixed_stack','padded_sequence','whole_bags'],
            multiprocessingContext='spawn',persistentWorkers=False,inOrder=True,pinMemory=False,
            orderVersion='kwip.data-loader.epoch-order.v1',gradientTransport='zeroWorkerOnly',
            partialBatch='retainOrDropGlobally',validation='orderedComplete',
            continuation='cpuEpochBoundaryExactMPSWeightsOnly',portableHelper='data_loading_recipe.py',
            portableHelperByteMaximum=131072)
    raise ValueError('Unknown Dataset/DataLoader control capability.')
