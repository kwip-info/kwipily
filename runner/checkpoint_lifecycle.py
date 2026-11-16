"""Versioned epoch-boundary CPU resume and explicitly weights-only fine-tuning."""
import hashlib
import copy
import io
import json
import math
from pathlib import Path
import platform
import stat
import torch

SCHEMA_VERSION=1
MAX_CHECKPOINT_BYTES=100_000_000

def validate_objective_checkpoint(stored,manifest,checkpoint):
    if 'objective' in stored:
        if __package__:
            from .loss_configuration import validate_recipe as validate_objective_recipe
            from .loss_prepared_data import validate_target_mapping
            from .loss_objectives import create_objective
        else:
            from loss_configuration import validate_recipe as validate_objective_recipe
            from loss_prepared_data import validate_target_mapping
            from loss_objectives import create_objective
        objective=stored['objective']
        if type(objective) is not dict or set(objective)!={'schemaVersion','recipe','targetMapping','scalarization'} or type(objective['schemaVersion']) is not int or objective['schemaVersion']!=1 or objective['scalarization'] not in ('mean','sum'):raise ValueError('Objective checkpoint contract is malformed.')
        recipe=validate_objective_recipe(objective['recipe']);mapping=validate_target_mapping(objective['targetMapping'],recipe)
        if objective['recipe']!=recipe or objective['targetMapping']!=mapping:raise ValueError('Objective checkpoint identity is not canonical.')
        settings=manifest.get('settings',{});provenance=manifest.get('provenance',{}).get('objective')
        if settings.get('objectiveRecipe')!=recipe or settings.get('targetMapping')!=mapping or settings.get('objectiveScalarization')!=objective['scalarization'] or type(provenance) is not dict or set(provenance)!={'recipe','targetMapping','scalarization','torchVersion'} or provenance['recipe']!=recipe or provenance['targetMapping']!=mapping or provenance['scalarization']!=objective['scalarization'] or provenance['torchVersion']!=manifest.get('runtime',{}).get('torch'):raise ValueError('Objective parent metadata differs from the pinned recipe/target mapping.')
        parent_objective=create_objective(recipe,'cpu');parent_objective.load_state_dict(checkpoint.get('objectiveState'))
    elif 'objectiveState' in checkpoint or any(key in manifest.get('settings',{}) for key in ('objectiveRecipe','targetMapping','objectiveScalarization')) or 'objective' in manifest.get('provenance',{}):raise ValueError('An undeclared objective checkpoint or metadata cannot be restored.')
    return stored.get('objective')



def capability():
    return {'version': {'major': 1, 'minor': 0, 'patch': 0}, 'adapterID': 'kwip.runner.checkpoint-lifecycle', 'checkpointSchemaVersions': [1], 'modes': ['weights-save', 'continuation-save', 'weights-restore', 'continuation-restore', 'incompatible-artifact-refusal'], 'sourceTrust': 'ownedCompletedHistory', 'loadPolicy': 'weightsOnlyCPUMap', 'artifactLimits': {'manifest.json': 20000000, 'checkpoint.pt': 100000000, 'metrics.json': 4000000, 'rate-trace.json': 4000000}, 'requiredArtifacts': ['manifest.json', 'checkpoint.pt', 'metrics.json'], 'schedulerTracePolicy': 'requiredExactlyWhenParentHasScheduler', 'saveTiming': 'successfulRunCompletionAtEpochBoundary', 'exactResumeDevices': ['cpu'], 'fineTuneDevices': ['cpu', 'mps'], 'cpuResume': 'epochBoundaryExactCPU', 'mpsResume': 'weightsOnly', 'fineTune': 'restoreModelStateResetApplicableTrainingState', 'parentImmutable': True, 'externalImport': False, 'periodicSave': False}

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

if __package__:
    from .graph_semantics import semantic_graph
else:
    from graph_semantics import semantic_graph


def tensor_digest(tensors):
    # Keep ordinary continuation bytes unchanged; typed sources bind their
    # coordinate roles and geometry as well as every actual tensor value.
    tensors=tuple(tensors)
    if not all(isinstance(value,torch.Tensor) for value in tensors):
        if __package__:
            from .model_input import model_inputs_digest
        else:
            from model_input import model_inputs_digest
        return model_inputs_digest(tensors)
    h=hashlib.sha256()
    for tensor in tensors:
        value=tensor.detach().cpu().contiguous()
        h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode());h.update(value.numpy().tobytes())
    return h.hexdigest()

def contract(graph,data,provenance,seed,batch_size,learning_rate,device,recipe_version,execution_order,optimizer_recipe=None,parameter_ownership=None):
    result = {'schemaVersion':SCHEMA_VERSION,'semanticGraphDigest':digest(semantic_graph(graph)),'recipeVersion':recipe_version,'runtime':{'python':platform.python_version(),'torch':str(torch.__version__),'platform':platform.platform(),'machine':platform.machine(),'device':device},'optimizer':{'name':'Adam','learningRate':learning_rate,'betas':[.9,.999],'eps':1e-8,'weightDecay':0,'amsgrad':False},'schedule':{'batchSize':batch_size,'shuffle':False,'dropLast':False,'numThreads':torch.get_num_threads(),'linearNodeOrder':[ident for ident in execution_order if next(n for n in graph['nodes'] if n['id']==ident)['type']=='linear']},'modelSeed':seed,'data':{'tensorDigest':tensor_digest(data),'sourceDigest':provenance['sourceHash'],'splitDigest':provenance['splitHash'],'preparation':provenance.get('preparation')},'guarantee':'epochBoundaryExactCPU' if device=='cpu' else 'weightsOnly'}
    if optimizer_recipe is not None:
        if __package__:
            from .optimizer_configuration import validate_recipe
        else:
            from optimizer_configuration import validate_recipe
        if not isinstance(parameter_ownership,list) or not parameter_ownership:raise ValueError('Explicit optimizer needs ordered parameter ownership')
        result['optimizer']={'schemaVersion':1,'recipe':validate_recipe(optimizer_recipe),'parameterOwnership':parameter_ownership}
    if provenance.get('namedInputPlan') is not None:result['data']['namedInputPlan']=copy.deepcopy(provenance['namedInputPlan'])
    kinds={node['id']:node['type'] for node in graph['nodes']}
    if 'bag_input' in kinds.values():
        # Prepared-source admission owns authorization; continuation separately
        # binds its selected mapping and ordered observation provenance.
        result['data']['bagInput']=copy.deepcopy({'schemaVersion':1,
            'modelInput':provenance.get('modelInput'),'sources':provenance.get('bagSources')})
    lazy_order=[ident for ident in execution_order if kinds[ident]=='lazy_linear']
    if lazy_order:
        # Preserve historical signatures; only lazy graphs bind deferred initialization.
        result['schedule']['lazyInitializationOrder']=lazy_order
    if lazy_order or {'rms_norm','prelu','conv1d','conv2d','conv3d','conv_transpose1d','conv_transpose2d','conv_transpose3d','depthwise_conv2d'} & set(kinds.values()):
        # New parameter families bind optimizer registration independently of the
        # sorted semantic graph; preexisting non-convolution signatures stay unchanged.
        result['schedule']['parameterNodeOrder']=[ident for ident in execution_order if kinds[ident] in {'prelu','linear','lazy_linear','bilinear','layer_norm','rms_norm','embedding','conv1d','conv2d','conv3d','conv_transpose1d','conv_transpose2d','conv_transpose3d','depthwise_conv2d'}]
    normalization={'batch_norm1d','batch_norm2d','instance_norm1d','instance_norm2d','group_norm'}
    if normalization & set(kinds.values()):
        result['schedule']['normalizationNodeOrder']=[ident for ident in execution_order if kinds[ident] in normalization]
        result['schedule']['parameterNodeOrder']=[ident for ident in execution_order if kinds[ident] in normalization | {'prelu','linear','lazy_linear','bilinear','layer_norm','rms_norm','embedding','conv1d','conv2d','conv3d','conv_transpose1d','conv_transpose2d','conv_transpose3d','depthwise_conv2d'}]
        from_config={node['id']:node['config'] for node in semantic_graph(graph)['nodes']}
        buffers=[ident for ident in execution_order if kinds[ident] in normalization and from_config[ident].get('track_running_stats',False)]
        if buffers:result['schedule']['bufferNodeOrder']=buffers
    recurrent={'rnn','lstm','gru','rnn_cell','lstm_cell','gru_cell'}
    if recurrent & set(kinds.values()):
        from_config={node['id']:node['config'] for node in semantic_graph(graph)['nodes']}
        result['schedule']['recurrentNodeOrder']=[ident for ident in execution_order if kinds[ident] in recurrent]
        result['schedule']['parameterNodeOrder']=[ident for ident in execution_order if kinds[ident] in recurrent | normalization | {'prelu','linear','lazy_linear','bilinear','layer_norm','rms_norm','embedding','conv1d','conv2d','conv3d','conv_transpose1d','conv_transpose2d','conv_transpose3d','depthwise_conv2d'}]
        stochastic=[ident for ident in execution_order if kinds[ident]=='dropout' or (kinds[ident] in recurrent and from_config[ident].get('dropout',0)>0)]
        if stochastic:result['schedule']['stochasticNodeOrder']=stochastic
    if 'dropout' in kinds.values() and not recurrent & set(kinds.values()):
        result['schedule']['stochasticNodeOrder']=[ident for ident in execution_order if kinds[ident]=='dropout']
    attention={'multihead_attention','transformer','transformer_encoder','transformer_decoder','scaled_dot_product_attention'}
    if attention & set(kinds.values()):
        from_config={node['id']:node['config'] for node in semantic_graph(graph)['nodes']}
        result['schedule']['attentionNodeOrder']=[ident for ident in execution_order if kinds[ident] in attention]
        learned_attention=attention-{'scaled_dot_product_attention'}
        result['schedule']['parameterNodeOrder']=[ident for ident in execution_order if kinds[ident] in learned_attention | recurrent | normalization | {'prelu','linear','lazy_linear','bilinear','layer_norm','rms_norm','embedding','conv1d','conv2d','conv3d','conv_transpose1d','conv_transpose2d','conv_transpose3d','depthwise_conv2d'}]
        stochastic=[ident for ident in execution_order if kinds[ident]=='dropout' or (kinds[ident] in recurrent | attention and from_config[ident].get('dropout',0)>0)]
        if stochastic:result['schedule']['stochasticNodeOrder']=stochastic
    if 'embedding_bag' in kinds.values():
        learned=attention-{'scaled_dot_product_attention'} | recurrent | normalization | {'prelu','linear','lazy_linear','bilinear','layer_norm','rms_norm','embedding','embedding_bag','conv1d','conv2d','conv3d','conv_transpose1d','conv_transpose2d','conv_transpose3d','depthwise_conv2d'}
        result['schedule']['parameterNodeOrder']=[ident for ident in execution_order if kinds[ident] in learned]
    return result


def regular_read(path,maximum):
    path=Path(path)
    info=path.lstat()
    if not stat.S_ISREG(info.st_mode):raise ValueError('Checkpoint artifacts must be regular files, not symbolic links')
    if info.st_size>maximum:raise ValueError('Checkpoint artifact exceeds size bound')
    with path.open('rb') as stream:value=stream.read(maximum+1)
    if len(value)>maximum:raise ValueError('Checkpoint artifact exceeds size bound')
    return value

def _initialization_identity(manifest, stored):
    settings=manifest.get('settings',{});provenance=manifest.get('provenance',{})
    if type(settings) is not dict or type(provenance) is not dict:raise ValueError('Checkpoint settings/provenance must be objects.')
    if 'initialization' not in stored:
        if 'initializationRecipe' in settings or 'initialization' in provenance:raise ValueError('Undeclared initialization metadata cannot be restored.')
        return
    if __package__:
        from .initialization import validate_recipe,SEED_VERSION
    else:
        from initialization import validate_recipe,SEED_VERSION
    identity=stored['initialization']
    required={'schemaVersion','recipe','resolvedOwners','generatorPlacement','seedDerivation'}
    if type(identity) is not dict or set(identity)!=required or type(identity['schemaVersion']) is not int or identity['schemaVersion']!=1 or identity['generatorPlacement']!='cpu-float32' or identity['seedDerivation']!=SEED_VERSION:
        raise ValueError('Checkpoint initialization identity is malformed.')
    recipe=validate_recipe(identity['recipe'])
    if recipe!=identity['recipe'] or settings.get('initializationRecipe')!=recipe:raise ValueError('Checkpoint initialization recipe differs from selected settings.')
    owners=identity['resolvedOwners'];rules=recipe['config']['rules']
    if type(owners) is not list or len(owners)!=len(rules):raise ValueError('Checkpoint initialization ownership is malformed.')
    names=set();elements=0
    for owner,rule in zip(owners,rules):
        if type(owner) is not dict or set(owner)!={'node','parameter','name','shape','dtype'} or owner['node']!=rule['owner']['node'] or owner['parameter']!=rule['owner']['parameter'] or type(owner['name']) is not str or not (owner in stored['composition']['parameterOwnership'] if 'composition' in stored else owner['name'].startswith('layers.')) or owner['name'] in names or owner['dtype']!='float32':raise ValueError('Checkpoint initialization owner identity differs from selected rules.')
        shape=owner['shape']
        if type(shape) is not list or any(type(v) is not int or not 1<=v<=200000 for v in shape):raise ValueError('Checkpoint initialization owner shape is malformed.')
        elements+=math.prod(shape);names.add(owner['name'])
    if elements>200000:raise ValueError('Checkpoint initialization ownership exceeds model parameter budget.')
    actual=provenance.get('initialization');application='applied' if manifest.get('lineage',{}).get('mode')=='newTraining' else 'notAppliedOnRestore'
    expected={**identity,'torchVersion':stored.get('runtime',{}).get('torch'),'application':application}
    if actual!=expected:raise ValueError('Checkpoint initialization provenance differs from its immutable contract.')


def _gradient_policy_identity(manifest,stored):
    if __package__:
        from .gradient_policy import validate_recipe as policy_recipe,validate_capture,validate_compatibility
        from .trainability import validate_recipe as flags_recipe
    else:
        from gradient_policy import validate_recipe as policy_recipe,validate_capture,validate_compatibility
        from trainability import validate_recipe as flags_recipe
    settings=manifest.get('settings',{});provenance=manifest.get('provenance',{});runtime=manifest.get('runtime',{})
    if 'gradientPolicy' in stored:
        item=stored['gradientPolicy']
        if type(item) is not dict or set(item)!={'schemaVersion','recipe','schedule','checkpointBoundary'} or type(item['schemaVersion']) is not int or item['schemaVersion']!=1 or item['checkpointBoundary']!='completedEpochNoPendingGradients':raise ValueError('Gradient-policy checkpoint identity is malformed.')
        recipe=policy_recipe(item['recipe']);k=recipe['config']['accumulation_steps'];m=settings.get('batchesPerEpoch');u=settings.get('optimizerUpdatesPerEpoch')
        if type(m) is not int or m<1 or type(u) is not int or u!=math.ceil(m/k) or type(settings.get('microbatchesPerEpoch')) is not int or settings.get('microbatchesPerEpoch')!=m:raise ValueError('Gradient-policy update schedule differs from actual microbatch counts.')
        expected=dict(microbatchesPerEpoch=m,optimizerUpdatesPerEpoch=u,accumulationSteps=k,gradientScale=recipe['config']['gradient_scale'])
        if type(item['schedule']) is not dict or any(type(item['schedule'].get(k)) is not int for k in ('microbatchesPerEpoch','optimizerUpdatesPerEpoch','accumulationSteps')):raise ValueError('Gradient-policy schedule requires exact integer counts.')
        if type(provenance.get('gradientPolicy',{}).get('schemaVersion')) is not int:raise ValueError('Gradient-policy provenance requires an exact schema version.')
        if item['schedule']!=expected or settings.get('gradientPolicyRecipe')!=recipe or provenance.get('gradientPolicy')!=dict(schemaVersion=1,recipe=recipe,implementation='kwip.runner.gradient_policy',torchVersion=stored['runtime']['torch'],schedule=expected):raise ValueError('Gradient-policy metadata differs from its immutable recipe/schedule.')
        if type(runtime.get('updateCadence')) is not dict or any(type(runtime['updateCadence'].get(k)) is not int for k in ('schemaVersion','microbatchesPerEpoch','optimizerUpdatesPerEpoch','accumulationSteps')):raise ValueError('Update cadence requires exact integer counts.')
        if runtime.get('updateCadence')!=dict(schemaVersion=1,unit='optimizerUpdate',microbatchesPerEpoch=m,optimizerUpdatesPerEpoch=u,accumulationSteps=k):raise ValueError('Gradient-policy runtime cadence differs.')
        capture=validate_capture(manifest.get('derivativeCapture'))
        if recipe['config']['requests'] and not capture['values']:raise ValueError('Parent derivative requests lack value permission.')
        objective=stored.get('objective',{})
        validate_compatibility(recipe,objective.get('recipe'),objective.get('scalarization','mean'))
    elif any(k in settings for k in ('gradientPolicyRecipe','microbatchesPerEpoch','optimizerUpdatesPerEpoch')) or 'gradientPolicy' in provenance or 'updateCadence' in runtime or 'derivativeCapture' in manifest:raise ValueError('Undeclared gradient policy metadata cannot be restored.')
    if 'trainability' in stored:
        item=stored['trainability']
        if type(item) is not dict or set(item)!={'schemaVersion','recipe','parameterOwnership','optimizerParameterNames'} or type(item['schemaVersion']) is not int or item['schemaVersion']!=1:raise ValueError('Trainability checkpoint identity is malformed.')
        recipe=flags_recipe(item['recipe']);inventory=item['parameterOwnership'];names=item['optimizerParameterNames']
        if type(inventory) is not list or not inventory or type(names) is not list:raise ValueError('Trainability ownership is missing.')
        seen=set();elements=0
        for owner in inventory:
            if type(owner) is not dict or set(owner)!={'node','parameter','name','shape','dtype','requiresGrad'} or any(type(owner[k]) is not str or not owner[k] or '\0' in owner[k] for k in ('node','parameter','name')) or owner['name'] in seen or type(owner['requiresGrad']) is not bool or owner['dtype']!='float32' or type(owner['shape']) is not list or any(type(d) is not int or not 1<=d<=200000 for d in owner['shape']):raise ValueError('Trainability native inventory is malformed.')
            seen.add(owner['name']);elements+=math.prod(owner['shape'])
        if elements>200000 or names!=[o['name'] for o in inventory if o['requiresGrad']] or not names:raise ValueError('Trainability optimizer ownership is empty or differs from native flags.')
        if 'recipe' in stored['optimizer'] and names!=[o['name'] for o in stored['optimizer']['parameterOwnership']]:raise ValueError('Optimizer and trainability ordered ownership differ.')
        if type(provenance.get('trainability',{}).get('schemaVersion')) is not int:raise ValueError('Trainability provenance requires an exact schema version.')
        if settings.get('trainabilityRecipe')!=recipe or provenance.get('trainability')!=dict(schemaVersion=1,recipe=recipe,implementation='kwip.runner.trainability',torchVersion=stored['runtime']['torch'],parameterOwnership=inventory,optimizerParameterNames=names):raise ValueError('Trainability metadata differs from immutable policy/native ownership.')
    elif 'trainabilityRecipe' in settings or 'trainability' in provenance:raise ValueError('Undeclared trainability metadata cannot be restored.')



def _composition_identity(manifest,stored,current):
    selected=current.get('composition');recorded=stored.get('composition')
    graph=manifest.get('graph',{})
    if selected is None:
        if recorded is not None or (type(graph) is dict and 'compositionPlan' in graph):raise ValueError('Unselected composition checkpoint metadata is refused.')
        return
    if type(recorded) is not dict or set(recorded)!={'schemaVersion','plan','parameterOwnership','stateOwnership'} or type(recorded.get('schemaVersion')) is not int or recorded['schemaVersion']!=1 or digest(recorded)!=digest(selected):
        raise ValueError('Checkpoint composition plan and actual native ownership must match.')
    if type(graph) is not dict or graph.get('compositionPlan')!=recorded['plan']:raise ValueError('Checkpoint composition plan differs from its saved graph.')

def _composition_state(checkpoint,stored):
    if 'composition' not in stored:return
    state=checkpoint.get('modelState');owners=stored['composition']['stateOwnership']
    if type(state) is not dict or list(state)!=[owner['name'] for owner in owners]:raise ValueError('Checkpoint composition persistent state order differs from native ownership.')
    for owner in owners:
        value=state[owner['name']]
        if not isinstance(value,torch.Tensor) or value.layout!=torch.strided or str(value.dtype).replace('torch.','')!=owner['dtype'] or list(value.shape)!=owner['shape'] or not torch.isfinite(value).all().item():raise ValueError('Checkpoint composition state differs from finite native ownership.')


def _data_loading_identity(manifest,stored):
    settings=manifest.get('settings',{});provenance=manifest.get('provenance',{})
    if 'dataLoading' not in stored:
        if any(k in settings for k in ('datasetRecipe','loaderRecipe')) or 'dataLoading' in manifest or 'dataLoading' in provenance:
            raise ValueError('Undeclared data-loading metadata cannot be restored.')
        return
    if __package__:
        from .dataset_loader_configuration import validate_recipe
        from .batch_loading import ORDER_VERSION
    else:
        from dataset_loader_configuration import validate_recipe
        from batch_loading import ORDER_VERSION
    item=stored['dataLoading']
    keys={'schemaVersion','datasetRecipe','loaderRecipe','orderVersion','originalTrainingRows','selectedRows','retainedRows','droppedRows','batchSize','batchesPerEpoch','selectionSHA256'}
    if type(item) is not dict or set(item)!=keys or type(item['schemaVersion']) is not int or item['schemaVersion']!=1 or item['orderVersion']!=ORDER_VERSION:
        raise ValueError('Data-loading checkpoint identity is malformed.')
    recipes={}
    for field,component in [('datasetRecipe','dataset_source'),('loaderRecipe','data_loader')]:
        value=item[field]
        if value is not None:
            normalized=validate_recipe(value)
            if normalized!=value or normalized['componentID']!=component or validate_recipe(settings.get(field))!=value:
                raise ValueError('Data-loading checkpoint recipe differs from selected settings.')
            recipes[field]=value['config']
        elif field in settings:raise ValueError('Unselected data-loading settings cannot be restored.')
    if not recipes:raise ValueError('Data-loading identity requires an actually selected recipe.')
    ids=provenance.get('trainingRowIDs');count=item['originalTrainingRows']
    if type(count) is not int or not 1<=count<=100000 or type(ids) is not list or len(ids)!=count or len(set(ids))!=count:
        raise ValueError('Data-loading parent requires complete original training identities.')
    config=recipes.get('datasetRecipe',{});kind=config.get('kind','indexed')
    positions=config['indices'] if kind=='subset' else list(range(min(count,config['max_rows']) if kind=='boundedIterable' else count))
    if any(i>=count for i in positions):raise ValueError('Data-loading parent subset exceeds original training partition.')
    batch=item['batchSize']
    if type(batch) is not int or not 1<=batch<=100000 or batch!=settings.get('requestedBatchSize'):
        raise ValueError('Data-loading parent batch authority differs.')
    loader=recipes.get('loaderRecipe',{})
    if loader and loader['batch_size']!=batch:raise ValueError('Data-loading recipe batch size differs.')
    retained=len(positions)//batch*batch if loader.get('drop_last',False) else len(positions)
    expected={'selectedRows':len(positions),'retainedRows':retained,'droppedRows':len(positions)-retained,'batchesPerEpoch':math.ceil(retained/batch)}
    if retained==0 or any(type(item[k]) is not int or item[k]!=v for k,v in expected.items()):raise ValueError('Data-loading parent population/counts differ from its recipes.')
    selection=[list(positions),[ids[i] for i in positions]]
    sha=hashlib.sha256(json.dumps(selection,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    same=lambda a,b:json.dumps(a,sort_keys=True,separators=(',',':'),allow_nan=False)==json.dumps(b,sort_keys=True,separators=(',',':'),allow_nan=False)
    if item['selectionSHA256']!=sha or not same(manifest.get('dataLoading'),item) or not same(provenance.get('dataLoading'),item):
        raise ValueError('Data-loading parent selected row identity differs from prepared provenance.')
    if settings.get('batchesPerEpoch')!=expected['batchesPerEpoch'] or settings.get('lastBatchSize')!=(retained%batch or min(batch,retained)):
        raise ValueError('Data-loading batch counts differ from actual retained rows.')
    if stored['schedule'].get('shuffle')!=loader.get('shuffle',False) or stored['schedule'].get('dropLast')!=loader.get('drop_last',False):
        raise ValueError('Data-loading parent schedule differs from selected policy.')


def _prepared_resume_comparison(stored,current,manifest):
    """Keep transport pins as evidence, compare independently verified data semantics.

    Completed preparation inventories and their segment/range pins are verified by
    the prepared loader before this contract is built. A fresh immutable layout
    changes that inventory's hash, without changing tensors, ordered membership,
    graph, mappings or fitted state. All of those remain exact comparisons below.
    Never normalize an unknown preparation shape or a parent/provenance mismatch.
    """
    preparation=stored.get('data',{}).get('preparation')
    current_preparation=current.get('data',{}).get('preparation')
    if preparation is None:return stored
    if preparation!=manifest.get('provenance',{}).get('preparation'):
        raise ValueError('Checkpoint preparation contract differs from parent provenance')
    fields={'manifestSHA256','graphDigest','registryDigest','trainingNode','trainingPort','validationNode','validationPort','trainingDataVersion','validationDataVersion','fittedStateDigests','artifactDirectory'}
    def pin(value):return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)
    if (type(preparation) is dict and type(current_preparation) is dict
        and set(preparation)==fields and set(current_preparation)==fields
        and preparation['artifactDirectory']==current_preparation['artifactDirectory']=='preparation'
        and pin(preparation['manifestSHA256']) and pin(current_preparation['manifestSHA256'])):
        normalized=copy.deepcopy(stored)
        normalized['data']['preparation']['manifestSHA256']=current_preparation['manifestSHA256']
        return normalized
    return stored


def load(path,current,mode,target_epoch):
    path=Path(path);manifest_raw=regular_read(path.with_name('manifest.json'),20_000_000);manifest=json.loads(manifest_raw)
    metadata=manifest.get('checkpoint') if isinstance(manifest,dict) else None
    if not isinstance(metadata,dict) or type(metadata.get('schemaVersion')) is not int or metadata['schemaVersion']!=SCHEMA_VERSION:raise ValueError('A versioned lifecycle checkpoint is required; legacy checkpoints are not supported by resume/fine-tune')
    if manifest.get('status')!='completed' or not isinstance(manifest.get('runID'),str):raise ValueError('Checkpoint requires its completed parent run manifest')
    raw=regular_read(path,MAX_CHECKPOINT_BYTES);sha=hashlib.sha256(raw).hexdigest()
    if sha!=metadata.get('sha256') or len(raw)!=metadata.get('bytes'):raise ValueError('Checkpoint integrity check failed')
    stored=metadata.get('contract')
    if not isinstance(stored,dict) or stored.get('schemaVersion')!=SCHEMA_VERSION:raise ValueError('Checkpoint semantic architecture is incompatible')
    _composition_identity(manifest,stored,current)
    _initialization_identity(manifest,stored)
    _gradient_policy_identity(manifest,stored)
    _data_loading_identity(manifest,stored)
    comparison=stored
    if stored.get('semanticGraphDigest')!=current['semanticGraphDigest']:
        # Verify the historic digest against its exact recorded graph before applying
        # the sole legacy numeric-representation equivalence (2.0 versus 2).
        saved_graph=manifest.get('graph')
        try:
            if not isinstance(saved_graph,dict) or type(saved_graph.get('schemaVersion')) is not int or saved_graph['schemaVersion']!=1:raise ValueError('Not a legacy graph')
            normalized=semantic_graph(saved_graph)
            legacy={'schemaVersion':1,'nodes':sorted([{'id':n['id'],'type':n['type'],'config':n['config']} for n in saved_graph['nodes']],key=lambda n:n['id']),'edges':normalized['edges']}
            if digest(legacy)!=stored.get('semanticGraphDigest') or digest(normalized)!=current['semanticGraphDigest']:raise ValueError('Different architecture')
        except (KeyError,TypeError,ValueError) as error:raise ValueError('Checkpoint semantic architecture is incompatible') from error
        comparison={**stored,'semanticGraphDigest':current['semanticGraphDigest']}

    if mode=='exactResume':
        if current['runtime']['device']!='cpu' or stored.get('guarantee')!='epochBoundaryExactCPU':raise ValueError('Exact resume is verified for CPU checkpoints on CPU only; use explicit weights-only fine-tuning otherwise')
        comparison=_prepared_resume_comparison(comparison,current,manifest)
        if comparison!=current:raise ValueError('Exact resume requires matching runtime, optimizer, seed, batch schedule and data partitions')
    # Share the reviewed ZIP storage preflight with frozen inference. Import here
    # because frozen_checkpoint itself imports this module and the graph engine.
    if __package__:
        from .frozen_checkpoint import _check_storage_archive
    else:
        from frozen_checkpoint import _check_storage_archive
    _check_storage_archive(raw)
    checkpoint=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    if not isinstance(checkpoint,dict) or checkpoint.get('checkpointSchemaVersion')!=SCHEMA_VERSION or checkpoint.get('contract')!=stored:raise ValueError('Checkpoint payload contract mismatch')
    _composition_state(checkpoint,stored)
    if 'initialization' in stored:
        model_state=checkpoint.get('modelState')
        if type(model_state) is not dict:raise ValueError('Checkpoint model state is malformed.')
        for owner in stored['initialization']['resolvedOwners']:
            value=model_state.get(owner['name'])
            if not isinstance(value,torch.Tensor) or value.dtype!=torch.float32 or list(value.shape)!=owner['shape'] or value.layout!=torch.strided or not torch.isfinite(value).all().item():raise ValueError('Checkpoint initialization owner differs from actual learned state.')
    validate_objective_checkpoint(stored,manifest,checkpoint)
    epoch=checkpoint.get('epoch')
    if type(epoch) is not int or not 1<=epoch<=10000 or epoch!=metadata.get('completedEpoch'):raise ValueError('Checkpoint epoch is invalid')
    if 'dataLoading' in stored:
        expected_loading=dict(schemaVersion=1,completedEpochs=epoch,nextEpoch=epoch,pendingBatches=0)
        if checkpoint.get('dataLoadingState')!=expected_loading or any(type(v) is not int for v in checkpoint.get('dataLoadingState',{}).values()):raise ValueError('Data loading checkpoints require a complete epoch with no pending batches.')
    elif 'dataLoadingState' in checkpoint:raise ValueError('Undeclared loader epoch state cannot be restored.')
    if 'gradientPolicy' in stored:
        schedule=stored['gradientPolicy']['schedule'];expected_updates=dict(schemaVersion=1,completedEpochs=epoch,completedMicrobatches=epoch*schedule['microbatchesPerEpoch'],completedOptimizerUpdates=epoch*schedule['optimizerUpdatesPerEpoch'],pendingMicrobatches=0)
        if checkpoint.get('updateSchedule')!=expected_updates or any(type(v) is not int for v in checkpoint.get('updateSchedule',{}).values()):raise ValueError('Checkpoint must end at a completed epoch without pending accumulated gradients.')
    elif 'updateSchedule' in checkpoint:raise ValueError('Undeclared accumulated update schedule cannot be restored.')
    if 'trainability' in stored:
        for owner in stored['trainability']['parameterOwnership']:
            value=checkpoint.get('modelState',{}).get(owner['name'])
            if not isinstance(value,torch.Tensor) or value.dtype!=torch.float32 or value.layout!=torch.strided or list(value.shape)!=owner['shape'] or not torch.isfinite(value).all().item():raise ValueError('Trainability native ownership differs from actual saved learned state.')
    if 'recipe' in stored.get('optimizer',{}):
        counts=checkpoint.get('optimizerUpdates');ownership=stored['optimizer'].get('parameterOwnership');batches=stored['gradientPolicy']['schedule']['optimizerUpdatesPerEpoch'] if 'gradientPolicy' in stored else manifest.get('settings',{}).get('batchesPerEpoch')
        if type(ownership) is not list or type(counts) is not list or len(counts)!=len(ownership) or type(batches) is not int or batches<1 or any(type(count) is not int or not 0<=count<=min(1_000_000_000,epoch*batches) for count in counts):raise ValueError('Explicit optimizer update counts differ from completed parent batches/ownership')
    if 'conditionalOptimizerUpdates' in checkpoint:
        if not any(n.get('type')=='mixture_experts' for n in manifest.get('graph',{}).get('nodes',[])) or 'recipe' in stored.get('optimizer',{}):raise ValueError('Undeclared conditional optimizer ledger.')
    if 'scheduler' in stored:
        envelope=checkpoint.get('scheduler');trace_meta=manifest.get('rateTrace')
        if type(envelope) is not dict or type(trace_meta) is not dict or set(trace_meta)!={'schemaVersion','path','bytes','sha256','records','timing'} or type(trace_meta.get('schemaVersion')) is not int or trace_meta.get('schemaVersion')!=1 or type(trace_meta.get('bytes')) is not int or not 1<=trace_meta['bytes']<=4_000_000 or type(trace_meta.get('records')) is not int or not 0<=trace_meta['records']<=10000 or trace_meta.get('path')!='rate-trace.json':raise ValueError('Scheduler checkpoint requires its pinned rate trace artifact')
        trace_raw=regular_read(path.with_name('rate-trace.json'),4_000_000)
        if len(trace_raw)!=trace_meta['bytes'] or hashlib.sha256(trace_raw).hexdigest()!=trace_meta['sha256']:raise ValueError('Scheduler rate trace integrity mismatch')
        trace=json.loads(trace_raw)
        if trace!=envelope.get('rateTrace') or envelope.get('recipe')!=stored['scheduler'].get('recipe') or envelope.get('ticks')!=trace_meta['records'] or trace_meta['timing']!=envelope.get('timing'):raise ValueError('Scheduler trace/checkpoint recipe or count differs')
        batches=stored['gradientPolicy']['schedule']['optimizerUpdatesPerEpoch'] if 'gradientPolicy' in stored else manifest.get('settings',{}).get('batchesPerEpoch');is_batch=stored['scheduler']['recipe']['componentID']=='scheduler_one_cycle_lr'
        if type(batches) is not int or batches<1 or envelope['ticks']!=(epoch*batches if is_batch else epoch):raise ValueError('Scheduler ticks differ from actual completed epochs/batches')
        for index,record in enumerate(trace['records']):
            expected_epoch=index//batches+1 if is_batch else index+1
            if record.get('epoch')!=expected_epoch or (is_batch and record.get('batch')!=index%batches+1):raise ValueError('Scheduler trace timing differs from actual completed batches')
    elif 'scheduler' in checkpoint:raise ValueError('An undeclared scheduler checkpoint cannot be restored')
    weights=checkpoint.get('modelState')
    if not isinstance(weights,dict) or not weights or any(not isinstance(k,str) or not isinstance(v,torch.Tensor) or not torch.isfinite(v).all() for k,v in weights.items()):raise ValueError('Checkpoint model weights are invalid/nonfinite')
    if 'trainingControl' in stored:
        if __package__:from .training_control import TrainingControl,validate_policy as validate_training_control
        else:from training_control import TrainingControl,validate_policy as validate_training_control
        identity=stored['trainingControl']
        if type(identity) is not dict or set(identity)!={'schemaVersion','policy'} or identity['schemaVersion']!=1:raise ValueError('Invalid training control checkpoint identity.')
        policy=validate_training_control(identity['policy'])
        selected=TrainingControl(policy,state=checkpoint.get('trainingControlState'),best_model=weights if policy['modelSelection']=='bestValidation' else None)
        if selected.last_epoch!=epoch or manifest.get('trainingControl')!=selected.report():raise ValueError('Selected model state differs from its checkpoint provenance.')
        final=checkpoint.get('continuationModelState')
        if policy['modelSelection']=='bestValidation':
            if type(final) is not dict or set(final)!=set(weights) or any(not isinstance(v,torch.Tensor) or v.shape!=weights[k].shape or v.dtype!=weights[k].dtype or v.layout!=torch.strided or not torch.isfinite(v).all() for k,v in final.items()):raise ValueError('Terminal continuation model state is invalid.')
        elif final is not None:raise ValueError('Final-epoch selection cannot invent separate model weights.')
        metrics_raw=regular_read(path.with_name('metrics.json'),4_000_000)
        if hashlib.sha256(metrics_raw).hexdigest()!=metadata.get('metricsSHA256'):raise ValueError('Selected model metric history integrity mismatch.')
        rows=json.loads(metrics_raw)
        if type(rows) is not list or len(rows)!=epoch:raise ValueError('Selected model history differs from completed epochs.')
        replay=TrainingControl(policy)
        first_stop=None
        for index,row in enumerate(rows,1):
            if type(row) is not dict or row.get('epoch')!=index:raise ValueError('Selected model metric epochs are not ordered.')
            stopped=replay.observe(index,row.get('validationLoss'))
            if stopped and first_stop is None:first_stop=index
        if replay.state()!=selected.state() or first_stop not in (None,epoch):raise ValueError('Selected model/stopping state contradicts validation history.')
        if manifest.get('evaluation',{}).get('validationLoss')!=selected.selected_value:raise ValueError('Selected model evaluation differs from measured validation loss.')
    elif any(key in checkpoint for key in ('trainingControlState','continuationModelState')) or 'trainingControl' in manifest:
        raise ValueError('Undeclared model-selection/continuation state.')
    history=[]
    if mode=='exactResume':
        if target_epoch<=epoch:raise ValueError('Resume target epoch must exceed checkpoint completed epoch')
        rng=checkpoint.get('rngState')
        if not isinstance(rng,torch.Tensor) or rng.dtype!=torch.uint8 or rng.shape!=torch.get_rng_state().shape:raise ValueError('Checkpoint RNG state invalid')
        if not isinstance(checkpoint.get('optimizerState'),dict):raise ValueError('Checkpoint optimizer state missing')
        history_raw=regular_read(path.with_name('metrics.json'),4_000_000)
        if hashlib.sha256(history_raw).hexdigest()!=metadata.get('metricsSHA256'):raise ValueError('Checkpoint metric history integrity mismatch')
        history=json.loads(history_raw)
        if not isinstance(history,list) or len(history)!=epoch or any(not isinstance(e,dict) or e.get('epoch')!=i or e.get('event')!='epoch' for i,e in enumerate(history,1)):raise ValueError('Checkpoint metric history does not match completed epochs')
        for event in history:
            required=('trainLoss','validationLoss') if 'objective' in stored else ('trainLoss','validationLoss','accuracy')
            if 'objective' in stored:
                op=stored['objective']['recipe']['componentID'];cfg=stored['objective']['recipe']['config'];eligible=op=='loss_nll' or op=='loss_cross_entropy' and cfg['target_mode']=='class_indices'
                if 'accuracy' in event and (not eligible or type(event['accuracy']) not in (int,float) or not 0<=event['accuracy']<=1):raise ValueError('Objective metric history invents unavailable classification accuracy.')
                summary=event.get('objective')
                if type(summary) is not dict or set(summary)!={'componentID','reduction','scalarization','training','validation'} or summary['componentID']!=op or summary['reduction']!=cfg['reduction'] or summary['scalarization']!=(stored['objective']['scalarization'] if cfg['reduction']=='none' else None):raise ValueError('Objective metric history differs from the selected reduction/identity.')
                def finite_metadata(value):
                    if type(value) is dict:return all(finite_metadata(v) for v in value.values())
                    if type(value) is list:return all(finite_metadata(v) for v in value)
                    if type(value) in (float,int):return torch.isfinite(torch.tensor(value,dtype=torch.float64)).item()
                    return value is None or type(value) is str
                if not finite_metadata(summary):raise ValueError('Objective metric history has nonfinite or foreign values.')
            for key in required:
                if type(event.get(key)) not in (int,float) or not torch.isfinite(torch.tensor(event[key],dtype=torch.float64)):raise ValueError('Checkpoint metric history contains invalid values')
    lineage={'mode':mode,'parentRunID':manifest['runID'],'parentCheckpointSHA256':sha,'parentCompletedEpoch':epoch,'restoredState':['parameters','optimizer','randomGenerator','epoch','metrics'] if mode=='exactResume' else ['parameters'],'resetState':[] if mode=='exactResume' else ['optimizer','randomGenerator','epoch','metrics'],'guarantee':'epochBoundaryExactCPU' if mode=='exactResume' else 'weightsOnly'}
    if current.get('schedule',{}).get('bufferNodeOrder'):lineage['restoredState'].insert(1,'buffers')
    if 'objective' in stored:lineage['restoredState' if mode=='exactResume' else 'resetState'].extend(['objective','objectiveBuffers'])
    if 'scheduler' in stored:lineage['restoredState' if mode=='exactResume' else 'resetState'].extend(['scheduler','rateTrace'])
    for key in ('gradientPolicy','trainability','dataLoading'):
        if key in stored:lineage['restoredState' if mode=='exactResume' else 'resetState'].append(key)
    return checkpoint,history,lineage


def restore_optimizer(optimizer,state,expected_steps,parameter_steps=None):
    expected=optimizer.state_dict()['param_groups'];actual=state.get('param_groups')
    if not isinstance(actual,list) or len(actual)!=len(expected):raise ValueError('Adam checkpoint parameter groups incompatible')
    for a,b in zip(actual,expected):
        if {k:v for k,v in a.items() if k!='params'}!={k:v for k,v in b.items() if k!='params'} or len(a.get('params',[]))!=len(b['params']):raise ValueError('Adam checkpoint hyperparameters disagree with lifecycle contract')
    count=sum(len(group['params']) for group in optimizer.param_groups)
    if parameter_steps is not None and (type(parameter_steps) is not list or len(parameter_steps)!=count or any(type(v) is not int or not 0<=v<=expected_steps for v in parameter_steps)):raise ValueError('Conditional Adam update ledger is invalid.')
    optimizer.load_state_dict(state)
    index=0
    for group in optimizer.param_groups:
        for parameter in group['params']:
            expected=expected_steps if parameter_steps is None else parameter_steps[index];index+=1
            item=optimizer.state.get(parameter)
            if parameter_steps is not None and expected==0 and item is None:continue
            if not isinstance(item,dict) or set(item)!={'step','exp_avg','exp_avg_sq'}:raise ValueError('Adam checkpoint state is incomplete')
            for name in ('exp_avg','exp_avg_sq'):
                value=item[name]
                if not isinstance(value,torch.Tensor) or value.shape!=parameter.shape or value.dtype!=parameter.dtype or not torch.isfinite(value).all():raise ValueError('Adam checkpoint moment tensor is incompatible')
            step=item['step']
            if not isinstance(step,torch.Tensor) or step.numel()!=1 or not torch.isfinite(step).all() or step.item()!=expected:raise ValueError('Adam checkpoint step is invalid')
