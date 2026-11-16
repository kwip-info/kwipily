"""Pure derived workflow source; replay delegates to existing execution adapters.

Generation never opens project sources or imports model/backend runtimes. The
native exporter supplies the trusted bundled runner beside the generated files.
"""
import hashlib
import importlib.metadata
import json
import math
import pprint
import re

MAX_PLAN_BYTES=2_000_000
SCOPES={'workflow','preparation','model','training'}
TOP={'schemaVersion','scope','document','compilerContractVersion','modelRegistry','dataRegistry','selectedTargets','preparation','models','trainings'}
RECIPE_FLAGS={
 'named_input_plan':'--named-input-plan-json',
 'optimizer_recipe':'--optimizer-recipe-json','scheduler_recipe':'--scheduler-recipe-json',
 'objective_recipe':'--objective-recipe-json','target_mapping':'--objective-target-mapping-json',
 'objective_scalarization':'--objective-scalarization','initialization_recipe':'--initialization-recipe-json',
 'gradient_policy':'--gradient-policy-json','trainability':'--trainability-json',
 'dataset_recipe':'--dataset-recipe-json','loader_recipe':'--loader-recipe-json',
 'sample_capture':'--sample-capture-json','derivative_capture':'--derivative-capture-json',
 'gradient_capture':None}
SETTINGS={'device','epochs','featureColumns','targetColumn','batchSize','sampleLimit','seed','splitSeed','validationPercent','learningRate','gradientCapture','trainingControl'}


def _require(ok,message):
    if not ok:raise ValueError(message)


def _json(value):
    try:return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
    except (TypeError,ValueError,RecursionError) as error:raise ValueError('Workflow Code requires finite JSON data') from error


def _keys(value,required,optional=()):
    _require(isinstance(value,dict) and set(required)<=set(value)<=set(required)|set(optional),'Unknown or missing Workflow Code fields')


def _id(value):
    _require(type(value) is str and 1<=len(value)<=256 and '\0' not in value,'Invalid Workflow Code identity')
    return value


def _graph(value):
    _keys(value,{'schemaVersion','name','nodes','edges'},{'execution','compositionPlan','namedInputs'})
    _require(type(value['schemaVersion']) is int and isinstance(value['nodes'],list) and isinstance(value['edges'],list),'Invalid model graph')
    execution=value.get('execution',{})
    _require(isinstance(execution,dict) and not set(execution)-SETTINGS,'Model graph contains private or unknown execution fields')


def _registry_identity(value):
    """Swift omits nil port optionals; JSON Schema numeric bounds retain meaning."""
    value=json.loads(_json(value))
    for operator in value.get('operators',[]):
        # Native registry encoding omits prose-only descriptions. They cannot
        # alter operator settings, ports, versions or execution identity.
        operator.pop('description',None)
        for port in operator.get('inputs',[])+operator.get('outputs',[]):
            for key in ('partition','stateKind','maximumConnections'):
                if port.get(key) is None:port.pop(key,None)
    return value


def _is_model_boundary(node):
    payload = node.get('payload', {})
    return ((node['address']['domain'] == 'control' and payload.get('kind') == 'modelDefinition')
            or (node['address']['domain'] == 'model' and payload.get('kind') == 'modelPrimitive'
                and payload.get('type') == 'output'))


def validate_snapshot(snapshot):
    _keys(snapshot,TOP-{'preparation'},{'preparation'})
    _require(type(snapshot['schemaVersion']) is int and snapshot['schemaVersion']==1 and snapshot['scope'] in SCOPES,'Unsupported Workflow Code schema/scope')
    _require(type(snapshot['compilerContractVersion']) is int and snapshot['compilerContractVersion'] in (12,13,14,15),'Unsupported workflow compiler contract')
    raw=_json(snapshot);_require(len(raw.encode())<=MAX_PLAN_BYTES,'Workflow Code snapshot exceeds 2 MB')
    snapshot=json.loads(raw)
    _require(isinstance(snapshot['document'],dict) and isinstance(snapshot['modelRegistry'],dict) and isinstance(snapshot['dataRegistry'],dict),'Missing captured workflow registries')
    document=snapshot['document'];_keys(document,{'schemaVersion','kind','projectID','revision','name','nodes','edges','definitions','drafts'},{'guidedBuild','modelInputs','modelSourceVersions','namedModelInputs'})
    _require(document['kind']=='workflowProject' and isinstance(document['nodes'],list),'Invalid captured workflow document')
    _require(type(document['schemaVersion']) is int and 1<=document['schemaVersion']<=23,'Unsupported captured workflow version')
    _require(snapshot['compilerContractVersion']==(15 if document['schemaVersion']>=17 else 14 if document['schemaVersion']>=16 else 13 if document['schemaVersion']>=15 else 12),'Workflow version differs from compiler contract')
    versions=document.get('modelSourceVersions')
    if versions is not None:
        _require(document['schemaVersion']>=14 and isinstance(versions,dict) and len(versions)<=128,'Invalid retained model source schema')
        models={node['address']['localID'] for node in document['nodes'] if _is_model_boundary(node)}
        for model_id,version in versions.items():
            _id(model_id)
            _require(model_id in models and type(version) is int and version in (2,3),'Invalid retained model source version')
    ownership=document.get('modelInputs')
    if ownership is not None:
        _require(document['schemaVersion']>=14 and isinstance(ownership,dict) and len(ownership)<=128,'Invalid model input ownership schema')
        by_address={(node['address']['domain'],node['address']['localID']):node for node in document['nodes']}
        for model_id,input_id in ownership.items():
            _id(model_id);_id(input_id)
            model=by_address.get(('control',model_id))
            if model is None or not _is_model_boundary(model):model=by_address.get(('model',model_id))
            boundary=by_address.get(('model',input_id))
            _require(model is not None and _is_model_boundary(model),'Model input owner is not a model definition')
            _require(boundary is not None and boundary['payload']['kind']=='modelPrimitive' and boundary['payload']['type'] in ('input','index_input'),'Model input ownership references an invalid tensor boundary')
    named_models=document.get('namedModelInputs')
    if named_models is not None:
        _require(document['schemaVersion']>=19 and type(named_models) is dict and len(named_models)<=128,'Invalid named model ownership')
        nodes={(node['address']['domain'],node['address']['localID']):node for node in document['nodes']}
        for model_id,names in named_models.items():
            _require(any(key[1]==model_id and _is_model_boundary(value) for key,value in nodes.items()),'Named owner is not a model boundary')
            _require(type(names) is dict and 1<=len(names)<=8 and len(set(names.values()))==len(names),'Named boundaries must be distinct')
            for name,node_id in names.items():
                _require(type(name) is str and re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,63}',name) is not None,'Invalid input name')
                node=nodes.get(('model',node_id))
                _require(node is not None and node['payload']['kind']=='modelPrimitive' and node['payload']['type'] in ('input','index_input'),'Named boundary does not exist')
    selected=snapshot['selectedTargets'];_require(isinstance(selected,list),'Invalid selected targets')
    for target in selected:_keys(target,{'domain','localID'});_id(target['localID'])
    addresses={(node['address']['domain'],node['address']['localID']) for node in document['nodes']}
    selected_addresses=[(target['domain'],target['localID']) for target in selected]
    _require(len(set(selected_addresses))==len(selected_addresses) and set(selected_addresses)<=addresses,'Selected targets differ from captured document')
    _require(isinstance(snapshot['models'],list) and isinstance(snapshot['trainings'],list),'Invalid workflow terminal inventory')
    seen=set()
    for model in snapshot['models']:
        _keys(model,{'id','graph','semanticSHA256','origins'},{'provenance'})
        ident=_id(model['id']);_require(('model',ident) not in seen,'Duplicate model identity');seen.add(('model',ident))
        _require(isinstance(model['semanticSHA256'],str) and re.fullmatch('[0-9a-f]{64}',model['semanticSHA256']) is not None,'Invalid model semantic pin')
        _graph(model['graph'])
    prep=snapshot.get('preparation')
    if prep is not None:
        _keys(prep,{'graph','nodeLabels','engine'},{'sampling','completeNodes'})
        _require(prep['engine'] in ('disk','polars'),'Invalid preparation engine')
        if __package__:from . import dataflow
        else:import dataflow
        nodes,_,_=dataflow.validate(prep['graph']);dataflow.validate_node_labels(prep['nodeLabels'],nodes)
        if __package__:from .worker_input_contract import same_json
        else:from worker_input_contract import same_json
        _require(same_json(_registry_identity(snapshot['dataRegistry']),_registry_identity(dataflow.REGISTRY)),'Captured data registry differs from this runtime')
        document_data=[node for node in document['nodes'] if node['address']['domain']=='data']
        chosen={node['id'] for node in prep['graph']['nodes']}
        if snapshot['scope'] in ('workflow','preparation'):
            _require(chosen=={node['address']['localID'] for node in document_data},'Preparation coverage differs from captured document')
        expected_nodes=[{'id':node['address']['localID'],'operator':node['payload']['operatorID'],'config':node['payload']['config']} for node in document_data if node['address']['localID'] in chosen]
        expected_edges=[{'from':{'node':edge['from']['address']['localID'],'port':edge['from']['port']},'to':{'node':edge['to']['address']['localID'],'port':edge['to']['port']}} for edge in document['edges'] if edge['from']['address']['domain']=='data' and edge['to']['address']['domain']=='data' and edge['from']['address']['localID'] in chosen and edge['to']['address']['localID'] in chosen]
        _require(same_json(prep['graph'],{'schemaVersion':1,'nodes':expected_nodes,'edges':expected_edges}),'Preparation graph differs from captured document')
    for training in snapshot['trainings']:
        _keys(training,{'id','backendID','prepared','recipes','recipeSources','trainingMode','preparationCachePolicy'},{'graph','configuration','settings'})
        ident=_id(training['id']);_require(('training',ident) not in seen,'Duplicate training identity');seen.add(('training',ident))
        backend=training['backendID'];_require(backend in ('pytorch','sklearn','tensorflow','pretrainedCausal'),'Unsupported training backend')
        _require(training['trainingMode'] in ('fresh','exactResume','fineTune'),'Unsupported training mode')
        _require(training['preparationCachePolicy'] in ('off','materialize','reuse'),'Unsupported preparation cache policy')
        mapping=training['prepared'];_keys(mapping,{'trainingNode','trainingPort','validationNode','validationPort','featureColumns'},{'targetColumn','bagInput','inputDtype','sampleShape'})
        for key in ('trainingNode','trainingPort','validationNode','validationPort'):_id(mapping[key])
        _require(isinstance(mapping['featureColumns'],list) and all(type(x) is str for x in mapping['featureColumns']),'Invalid feature mapping')
        recipes=training['recipes'];_require(isinstance(recipes,dict) and not set(recipes)-set(RECIPE_FLAGS),'Unsupported captured training recipe')
        _require(isinstance(training['recipeSources'],dict),'Invalid recipe source metadata')
        if backend=='pytorch':
            _require('graph' in training and 'settings' in training and 'configuration' not in training,'PyTorch training requires captured graph and settings')
            _graph(training['graph']);settings=training['settings'];_keys(settings,{'device','epochs','batchSize','seed','learningRate'},SETTINGS)
            if settings.get('trainingControl') is not None:
                if __package__:from .training_control import validate_policy as validate_training_control
                else:from training_control import validate_policy as validate_training_control
                validate_training_control(settings['trainingControl'])
            _require(settings['device'] in ('auto','cpu','mps'),'Unsupported training device')
            for key,low,high in [('epochs',1,10000),('batchSize',1,100000),('seed',0,2**32-1)]:_require(type(settings[key]) is int and low<=settings[key]<=high,'Invalid captured training setting')
            _require(type(settings['learningRate']) in (int,float) and math.isfinite(settings['learningRate']),'Invalid learning rate')
        elif backend=='pretrainedCausal':
            _require('graph' not in training and 'settings' not in training and not recipes and not training['recipeSources'] and training['trainingMode']=='fresh' and training['preparationCachePolicy']=='off','Causal Code uses the captured Train task with explicit external model bindings')
            config=training.get('configuration');_keys(config,{'model','train'})
            node=next((n for n in document['nodes'] if n['address']=={'domain':'control','localID':ident}),None)
            _require(node is not None and node['payload']['kind']=='train' and _json(node['payload']['config'])==_json(config['train']) and 'fineTuning' in config['train'],'Causal Train settings differ from captured canvas')
            edges=[e for e in document['edges'] if e['to']['address']==node['address']]
            _require(len(edges)==3 and {e['to']['port'] for e in edges}=={'model','training','validation'},'Causal training requires its three declared inputs')
            for role in ('model','training','validation'):
                edge=next(e for e in edges if e['to']['port']==role)
                _require(edge['from']['address']=={'domain':'control' if role=='model' else 'data','localID':config['model'] if role=='model' else mapping[role+'Node']} and edge['from']['port']==('model' if role=='model' else mapping[role+'Port']),'Causal task bindings differ from canvas')
            source=next((n for n in document['nodes'] if n['address']=={'domain':'control','localID':config['model']}),None)
            _require(source is not None and source['payload']['kind']=='frozenModel','Causal task needs Load model')
            _require(not mapping['featureColumns'] and not any(k in mapping for k in ('targetColumn','bagInput','inputDtype','sampleShape')),'Causal task uses its explicit text mapping')
        else:
            _require('configuration' in training and isinstance(training['configuration'],dict) and not recipes,'Optional backend configuration must not silently drop PyTorch recipes')
            _require(training['trainingMode']=='fresh','Optional backend continuation is unsupported')
            if backend=='tensorflow':_graph(training['graph'])
    scope=snapshot['scope']
    _require(len({training['preparationCachePolicy'] for training in snapshot['trainings']})<=1,'Choose one preparation cache policy for this Code capture; mixed policies would repeat output writes')
    selected_ids={target['localID'] for target in selected}
    carriers={(node['address']['domain'],node['address']['localID']) for node in document['nodes'] if node['address']['domain']=='control' and node['payload']['kind']=='modelDefinition'}
    def visible_boundary(node):
        if not _is_model_boundary(node):return False
        if node['payload']['kind']=='modelDefinition':return True
        return not any(edge['from']['address']==node['address'] and edge['to']['port']=='source'
            and (edge['to']['address']['domain'],edge['to']['address']['localID']) in carriers
            for edge in document['edges'])
    document_models={node['address']['localID'] for node in document['nodes'] if visible_boundary(node)}
    document_training={node['address']['localID'] for node in document['nodes'] if node['payload']['kind'] in ('train','estimator')}
    expected_models=(document_models & selected_ids if selected_ids else document_models) if scope in ('workflow','model') else set()
    expected_training=(document_training & selected_ids if selected_ids else document_training) if scope in ('workflow','training') else set()
    _require({model['id'] for model in snapshot['models']}==expected_models and {training['id'] for training in snapshot['trainings']}==expected_training,'Workflow Code terminal coverage differs from captured scope')
    _require(scope not in ('workflow','preparation') or not selected,'Whole workflow/preparation scope cannot hide selected terminals')
    _require(scope!='workflow' or not any(node['payload']['kind']=='evaluateTest' or node['payload']['kind']=='frozenModel' and node['address']['localID'] not in {t['configuration']['model'] for t in snapshot['trainings'] if t['backendID']=='pretrainedCausal'} for node in document['nodes']),'Whole workflow contains Load model or archived evaluation terminals without a portable workflow adapter. Use Save model for a supported local model export.')
    _require(scope not in ('workflow','preparation') or prep is not None or (scope=='workflow' and not snapshot['trainings']),'Preparation snapshot missing for selected scope')
    _require(scope!='preparation' or (not snapshot['models'] and not snapshot['trainings']),'Preparation scope contains model/training terminals')
    _require(scope!='model' or (snapshot['models'] and not snapshot['trainings'] and prep is None),'Model scope contains unrelated execution')
    _require(scope!='training' or bool(snapshot['trainings']),'Training scope has no Train/estimator')
    if snapshot.get('preparation') is not None:_preparation_demand(snapshot)
    return snapshot


def _preparation_demand(snapshot):
    """Training endpoints require full ancestors even when other branches sample."""
    if __package__:from .preparation_demand import build_demand_plan
    else:from preparation_demand import build_demand_plan
    prep=snapshot['preparation'];complete=prep.get('completeNodes',[])
    _require(isinstance(complete,list),'Complete preparation nodes must be an array')
    complete=list(complete)
    for training in snapshot['trainings']:
        complete.extend(training['prepared'][key] for key in ('trainingNode','validationNode'))
    return build_demand_plan(prep['graph'],prep.get('sampling'),list(dict.fromkeys(complete)))


def _versions(snapshot):
    names={'numpy'}
    if (snapshot.get('preparation') or {}).get('engine')=='polars':names|={'polars','polars-runtime-32'}
    if snapshot['models'] or snapshot['trainings']:names.add('torch')
    for training in snapshot['trainings']:
        names|={'scikit-learn'} if training['backendID']=='sklearn' else {'tensorflow'} if training['backendID']=='tensorflow' else {'transformers','peft','accelerate','safetensors','psutil','tokenizers'} if training['backendID']=='pretrainedCausal' else set()
    result={}
    for name in sorted(names):
        try:result[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:result[name]=None
    if snapshot.get('preparation'):
        if __package__:from .table_source_options import reader_versions
        else:from table_source_options import reader_versions
        result.update(reader_versions(snapshot['preparation']['graph']))
    return result


def generate_package(snapshot,check_cancelled=lambda:None):
    check_cancelled();snapshot=validate_snapshot(snapshot);encoded=_json(snapshot)
    pin=hashlib.sha256(encoded.encode()).hexdigest();versions=_versions(snapshot)
    lines=['# Kwip ML — '+snapshot['scope'].replace('preparation','data preparation').title(),
           '# Derived from the captured canvas. Source and destination paths stay external.',
           '# Uses the accompanying runner; node algorithms, recipes and provenance are shared.',
           '# Snapshot SHA-256: '+pin,'import argparse','import json','from pathlib import Path',
           'from runner.workflow_code import execute_snapshot',
           'from runner.dataflow_storage import bounded_regular_read','',
           '# Registry/compiler metadata is retained in the adjacent immutable snapshot.',
           "SNAPSHOT = json.loads(bounded_regular_read(Path(__file__).with_name('workflow-plan.json'), 2_000_000))",
           '# Captured preparation nodes, model definitions and Train/estimator settings:',
           'PREPARATION = '+pprint.pformat(snapshot.get('preparation'),width=100,sort_dicts=False),
           'MODELS = '+pprint.pformat(snapshot['models'],width=100,sort_dicts=False),
           'TRAININGS = '+pprint.pformat(snapshot['trainings'],width=100,sort_dicts=False),
           "if (PREPARATION, MODELS, TRAININGS) != (SNAPSHOT.get('preparation'), SNAPSHOT['models'], SNAPSHOT['trainings']):",
           "    raise ValueError('Derived source differs from the captured workflow plan')",'',
           'def main():',
           "    parser = argparse.ArgumentParser(description='Replay the captured Kwip ML workflow.')",
           "    parser.add_argument('--bindings', help='External JSON sources/destinations/checkpoints/prepared manifest')",
           "    parser.add_argument('--output', required=True, help='New output directory; never overwrite a prior run')",
           '    args = parser.parse_args()',
           '    bindings = json.loads(bounded_regular_read(Path(args.bindings), 2_000_000)) if args.bindings else {}',
           '    execute_snapshot(SNAPSHOT, bindings, args.output)',
           '',"if __name__ == '__main__':",'    main()','']
    code='\n'.join(lines)
    files={'workflow.py':code,'workflow-plan.json':encoded,
           'requirements.txt':'\n'.join(name+('=='+version if version else '') for name,version in versions.items())+'\n',
           'bindings.example.json':json.dumps({'sources':{},'destinations':{},'checkpoints':{},'models':{},'resources':{'maximumResidentBytes':2_000_000_000},'prepared':None,'reuse':None},indent=2)+'\n',
           'README.md':'# Kwip ML derived workflow\n\nRun `python workflow.py --bindings bindings.json --output NEW_DIRECTORY`. Install requirements in an isolated environment first. The exporter must include its trusted `runner/` sources and registries; this script is not an independent replacement engine.\n\nFile Input source bindings are explicit selected file/directory paths, or an explicit {\"manifest\": \"path/to/manifest.json\"} collection. Captured collection selection settings must match; changing filters requires recapture. Reader options and metadata columns may reprocess the same verified captured bodies. File Input preparation.py requires --output NEW_DIRECTORY to retain its table and image-pixel artifact closure.\n\nSource/destination grants and continuation checkpoints are external. Imported causal Train also requires an explicit models binding keyed by its Load model ID, and a resources.maximumResidentBytes allowance. CPU exact resume uses an explicit checkpoints binding keyed by Train ID; no private app path is exported. Generate, Save model and Evaluate terminal replay remain precise unsupported whole-workflow scopes; select Training to export that task.  Training scope replays its captured preparation when present; otherwise it requires an external manifest with exact SHA-256 and graph digest pins. Reuse policy requires an external pinned reuse manifest. Fresh workflow execution produces and pins its own preparation before training. Existing outputs are never overwritten.\n\nDependency versions are recorded where installed; an unpinned optional backend means it was not installed during generation and requires separate runtime compatibility verification. This package does not include model weights, project data, credentials, or private app permissions. It executes outside the app sandbox under your Python process.\n\nRecorded History source must be exported from its retained capture; generating a current snapshot does not recreate historical code.\n'}
    if snapshot.get('preparation') is not None:
        if __package__:from .preparation_plan import build_plan,derive_preparation_code,_installed_engine_version
        else:from preparation_plan import build_plan,derive_preparation_code,_installed_engine_version
        preparation=snapshot['preparation'];engine=preparation['engine'];demand=_preparation_demand(snapshot)
        files['preparation.py']=derive_preparation_code(build_plan(preparation['graph'],engine=engine,engine_version=_installed_engine_version(engine),node_labels=preparation['nodeLabels'],sampling=demand.sampling,complete_nodes=demand.complete_node_ids))
    check_cancelled();return {'code':code,'files':files,'planDigest':pin,'scope':snapshot['scope']}


def pytorch_arguments(training,graph_path,registry_path,prepared,output,bindings):
    """Controlled translation to the existing CLI, preserving every selected recipe."""
    settings=training['settings'];mapping=training['prepared']
    args=['--graph',str(graph_path),'--registry',str(registry_path),'--output',str(output)]
    for key,flag in [('device','--device'),('epochs','--epochs'),('seed','--seed'),('batchSize','--batch-size'),('learningRate','--learning-rate')]:args += [flag,str(settings[key])]
    for key,flag in [('manifest','--prepared-dataflow'),('manifestSHA256','--prepared-manifest-sha256'),('graphDigest','--prepared-graph-digest')]:args += [flag,prepared[key]]
    for key,flag in [('trainingNode','--prepared-training-node'),('trainingPort','--prepared-training-port'),('validationNode','--prepared-validation-node'),('validationPort','--prepared-validation-port')]:args += [flag,mapping[key]]
    if settings.get('trainingControl') is not None:args += ['--training-control-json',_json(settings['trainingControl'])]
    recipes=training['recipes']
    if 'bagInput' in mapping:args += ['--bag-input-mapping-json',_json(mapping['bagInput'])]
    elif 'objective_recipe' not in recipes:args += ['--feature-columns-json',_json(mapping['featureColumns'])]
    if 'objective_recipe' not in recipes:args += ['--target-column',mapping['targetColumn']]
    for key,value in recipes.items():
        if key=='gradient_capture':continue
        args += [RECIPE_FLAGS[key],str(value) if key=='objective_scalarization' else _json(value)]
    capture=recipes.get('gradient_capture',settings.get('gradientCapture'))
    if capture:
        args += ['--capture-gradients']
        for key,flag in [('everyEpochs','--gradient-every-epochs'),('maxBatchesPerEpoch','--gradient-max-batches-per-epoch'),('maxCaptures','--gradient-max-captures')]:args += [flag,str(capture[key])]
    if training['trainingMode']!='fresh':
        path=bindings.get('checkpoints',{}).get(training['id'])
        _require(type(path) is str and path,'Selected continuation requires an external checkpoint')
        args += ['--resume-checkpoint' if training['trainingMode']=='exactResume' else '--finetune-checkpoint',path]
    return args


def execute_snapshot(snapshot,bindings,output,emit=lambda event:print(json.dumps(event,allow_nan=False)),check_cancelled=lambda:None):
    """Portable replay. User execution, never called merely to display Code."""
    from pathlib import Path
    import subprocess
    import sys
    snapshot=validate_snapshot(snapshot);_keys(bindings,set(),{'sources','destinations','checkpoints','models','resources','prepared','reuse'})
    output=Path(output);_require(not output.exists() and not output.is_symlink(),'Workflow output must be a new directory')
    output.mkdir(parents=True,mode=0o700)
    registry=output/'model-registry.json';registry.write_text(_json(snapshot['modelRegistry']))
    prep=snapshot.get('preparation');prepared=bindings.get('prepared');scope=snapshot['scope']
    preparations={}
    if scope in ('workflow','preparation','training') and prep is not None:
        if __package__:from .dataflow_cli import run_dataflow
        else:from dataflow_cli import run_dataflow
        graph=output/'preparation-graph.json';graph.write_text(_json(prep['graph']));demand=_preparation_demand(snapshot)
        policies={training['preparationCachePolicy'] for training in snapshot['trainings']}
        for index,policy in enumerate(sorted(policies or {'off'})):
            cache={'cache_policy':policy}
            if policy=='reuse':
                reuse=bindings.get('reuse');_keys(reuse,{'manifest','manifestSHA256'})
                cache.update(reuse_manifest=reuse['manifest'],reuse_manifest_sha256=reuse['manifestSHA256'])
            location=output/('preparation' if len(policies)<=1 else f'preparation-{index:03d}')
            result=run_dataflow(graph,bindings.get('sources',{}),location,emit,engine=prep['engine'],node_labels=prep['nodeLabels'],destination_bindings=bindings.get('destinations',{}),check_cancelled=check_cancelled,sampling=demand.sampling,complete_nodes=demand.complete_node_ids,**cache)
            manifest=location/'manifest.json'
            preparations[policy]={'manifest':str(manifest),'manifestSHA256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'graphDigest':result['graphDigest']}
    if scope in ('workflow','model') and snapshot['models']:
        if __package__:from .engine import GraphModel,export_source
        else:from engine import GraphModel,export_source
        for index,model in enumerate(snapshot['models']):
            check_cancelled();source=export_source(GraphModel(model['graph'],str(registry),bundled_dataset=False,architecture_only=True))
            (output/f'model-{index:03d}.py').write_text(source)
    if scope in ('workflow','training'):
        for index,training in enumerate(snapshot['trainings']):
            prepared=preparations.get(training['preparationCachePolicy'],prepared)
            _keys(prepared,{'manifest','manifestSHA256','graphDigest'})
            check_cancelled();destination=output/f'training-{index:03d}';backend=training['backendID']
            if backend=='pytorch':
                graph=output/f'model-{index:03d}.json';graph.write_text(_json(training['graph']))
                args=pytorch_arguments(training,graph,registry,prepared,destination,bindings)
                subprocess.run([sys.executable,str(Path(__file__).with_name('main.py')),*args],check=True)
            elif backend=='pretrainedCausal':
                if __package__:from .pretrained_workflow import run_prepared
                else:from pretrained_workflow import run_prepared
                run_prepared(training,prepared,bindings,destination,emit,check_cancelled)
            else:
                import importlib
                module=importlib.import_module(('.' if __package__ else '')+('sklearn_workflow' if backend=='sklearn' else 'tensorflow_workflow'),package=__package__)
                config=training['configuration'] if backend=='sklearn' else {'graph':training['graph'],'settings':training['configuration']}
                mapping={key:value for key,value in training['prepared'].items() if key in ('trainingNode','trainingPort','validationNode','validationPort','featureColumns','targetColumn')}
                module.run_prepared({'schemaVersion':1,'backendID':backend,'configuration':config,'prepared':{**mapping,**prepared}},destination,emit,check_cancelled=check_cancelled)
    check_cancelled();emit({'event':'workflowCodeCompleted','scope':scope})
