import builtins
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import pytest
from runner import dataflow
from runner.workflow_code import generate_package,validate_snapshot,pytorch_arguments


def snapshot():
    graph={'schemaVersion':1,'nodes':[{'id':'source','operator':'kwip.data.sourceCSV','config':{'separator':','}}],'edges':[]}
    return {'schemaVersion':1,'scope':'workflow','compilerContractVersion':12,
      'document':{'schemaVersion':14,'kind':'workflowProject','projectID':'p','revision':1,'name':'Demo','nodes':[{'address':{'domain':'data','localID':'source'},'name':'Observations','x':0,'y':0,'payload':{'kind':'dataOperator','operatorID':'kwip.data.sourceCSV','config':{'separator':','}}}], 'edges':[],'definitions':[],'drafts':[]},
      'modelRegistry':{'schemaVersion':1,'components':[]},'dataRegistry':copy.deepcopy(dataflow.REGISTRY),'selectedTargets':[],
      'preparation':{'graph':graph,'nodeLabels':{'source':'Observations'},'engine':'disk'},'models':[],'trainings':[]}


def test_missing_terminal_cannot_claim_whole_workflow():
    value=snapshot();value['document']['nodes'] += [{'address':{'domain':'control','localID':'train'},'name':'Train','x':0,'y':0,'payload':{'kind':'train','config':{}}}]
    with pytest.raises(ValueError,match='coverage'):generate_package(value)


@pytest.mark.parametrize('change',['field','scope','version','privategraph','recipe'])
def test_unknown_fields_altered_scope_and_private_paths_refused(change):
    value=snapshot()
    if change=='field':value['extra']=1
    elif change=='scope':value['scope']='anything'
    elif change=='version':value['compilerContractVersion']=12.0
    elif change=='privategraph':value['preparation']['graph']['privateSource']='/private/data.csv'
    else:value['scope']='training'
    with pytest.raises(ValueError):generate_package(value)


def test_generation_reads_no_user_sources_or_optional_models(monkeypatch):
    original=builtins.__import__
    def blocked(name,*args,**kwargs):
        if name.split('.')[0] in {'torch','sklearn','tensorflow'}:raise AssertionError('optional model imported for display')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',blocked)
    value=snapshot();package=generate_package(value)
    assert package['scope']=='workflow' and '/private/' not in package['code']
    assert set(package['files'])=={'workflow.py','workflow-plan.json','requirements.txt','bindings.example.json','README.md','preparation.py'}
    value['preparation']['graph']['nodes'].clear()
    assert 'source' in package['code']


@pytest.mark.parametrize('owned',[False,True])
def test_data_only_package_replays_outside_repository_with_external_source(tmp_path,owned):
    package=generate_package(model_owned_input_snapshot() if owned else snapshot());destination=tmp_path/'package';destination.mkdir()
    for name,content in package['files'].items():(destination/name).write_text(content)
    runtime=destination/'runner';runtime.mkdir()
    # Match the native export's trusted own-runtime policy; no test/cache/private files.
    for source in Path(__file__).parents[1].iterdir():
        if source.is_file() and (source.suffix in ('.py','.json') or source.name=='laya-LICENSE.txt'):shutil.copyfile(source,runtime/source.name)
    source=tmp_path/'private-observations.csv';source.write_text('a,b\n1,2\n3,4\n')
    bindings=tmp_path/'bindings.json';bindings.write_text(json.dumps({'sources':{'source':str(source)}}))
    output=tmp_path/'result'
    result=subprocess.run([sys.executable,str(destination/'workflow.py'),'--bindings',str(bindings),'--output',str(output)],cwd=tmp_path,capture_output=True,text=True,check=True)
    assert 'workflowCodeCompleted' in result.stdout
    manifest=json.loads((output/'preparation/manifest.json').read_text())
    table=json.loads((output/'preparation'/next(x['path'] for x in manifest['artifacts'] if x.get('kind')=='table')).read_text())
    assert table['rows']==[['1','2'],['3','4']]
    assert str(source) not in ''.join(package['files'].values())
    again=subprocess.run([sys.executable,str(destination/'workflow.py'),'--bindings',str(bindings),'--output',str(output)],cwd=tmp_path,capture_output=True,text=True)
    assert again.returncode!=0 and 'new directory' in again.stderr


def test_pytorch_driver_preserves_all_recipes_and_external_checkpoint():
    from runner.workflow_code import RECIPE_FLAGS
    recipes={key:{'sentinel':key} for key in RECIPE_FLAGS if key not in ('gradient_capture','objective_scalarization')}
    recipes['objective_scalarization']='sum'
    training={'id':'train','trainingMode':'exactResume','settings':{'device':'cpu','epochs':3,'seed':17,'batchSize':5,'learningRate':0.2,'gradientCapture':{'schemaVersion':1,'everyEpochs':2,'maxBatchesPerEpoch':3,'maxCaptures':7}},
      'prepared':{'trainingNode':'split','trainingPort':'training','validationNode':'split','validationPort':'validation','featureColumns':['a','b'],'targetColumn':'y'},'recipes':recipes}
    prepared={'manifest':'external/manifest.json','manifestSHA256':'a'*64,'graphDigest':'b'*64}
    args=pytorch_arguments(training,'graph.json','registry.json',prepared,'output',{'checkpoints':{'train':'external/checkpoint.pt'}})
    for key,flag in RECIPE_FLAGS.items():
        if flag:assert args[args.index(flag)+1]==('sum' if key=='objective_scalarization' else json.dumps(recipes[key],sort_keys=True,separators=(',',':')))
    assert args[args.index('--resume-checkpoint')+1]=='external/checkpoint.pt'
    assert '--feature-columns-json' not in args and '--target-column' not in args
    assert args[args.index('--gradient-every-epochs')+1]=='2'

@pytest.mark.parametrize('backend',['sklearn','tensorflow'])
def test_optional_backend_runtime_delegates_exact_prepared_request(tmp_path,monkeypatch,backend):
    import importlib
    from runner.workflow_code import execute_snapshot
    value=snapshot();value['scope']='training';del value['preparation']
    graph={'schemaVersion':2,'name':'Model','nodes':[],'edges':[]}
    value['document']['nodes']=[{'address':{'domain':'control','localID':'fit'},'name':'Fit','x':0,'y':0,'payload':{'kind':'estimator','config':{'backendID':backend}}}]
    value['trainings']=[{'id':'fit','backendID':backend,'prepared':{'trainingNode':'split','trainingPort':'training','validationNode':'split','validationPort':'validation','featureColumns':['a','b'],'targetColumn':'label'},'configuration':{'setting':'captured'},'recipes':{},'recipeSources':{},'trainingMode':'fresh','preparationCachePolicy':'off'}]
    if backend=='tensorflow':value['trainings'][0]['graph']=graph
    calls=[]
    class Backend:
        @staticmethod
        def run_prepared(request,output,emit,**kwargs):calls.append(request)
    original=importlib.import_module
    def module(name,package=None):
        if name in ('.sklearn_workflow','.tensorflow_workflow'):return Backend
        return original(name,package)
    monkeypatch.setattr(importlib,'import_module',module)
    prepared={'manifest':'external/manifest.json','manifestSHA256':'a'*64,'graphDigest':'b'*64}
    execute_snapshot(value,{'prepared':prepared},tmp_path/'out',lambda _:None)
    assert len(calls)==1 and calls[0]['prepared']=={**value['trainings'][0]['prepared'],**prepared}
    assert calls[0]['configuration']==({'setting':'captured'} if backend=='sklearn' else {'graph':graph,'settings':{'setting':'captured'}})


def test_unknown_selected_recipe_fails_before_replay():
    value=snapshot();value['scope']='training';del value['preparation']
    value['document']['nodes']=[{'address':{'domain':'control','localID':'fit'},'name':'Fit','x':0,'y':0,'payload':{'kind':'estimator','config':{}}}]
    value['trainings']=[{'id':'fit','backendID':'sklearn','prepared':{'trainingNode':'split','trainingPort':'training','validationNode':'split','validationPort':'validation','featureColumns':['a','b'],'targetColumn':'label'},'configuration':{},'recipes':{'new_unsupported_recipe':{}},'recipeSources':{},'trainingMode':'fresh','preparationCachePolicy':'off'}]
    with pytest.raises(ValueError,match='recipe'):generate_package(value)


def training_snapshot():
    root=Path(__file__).parents[2]
    value=snapshot();graph=json.loads((root/'examples/dataflow/scaler.json').read_text())
    value['preparation']['graph']=graph;value['preparation']['nodeLabels']={}
    value['modelRegistry']=json.loads((root/'Sources/KwipMLCore/Resources/components.json').read_text())
    value['document']['nodes']=[{'address':{'domain':'data','localID':n['id']},'name':'','x':0,'y':0,'payload':{'kind':'dataOperator','operatorID':n['operator'],'config':n['config']}} for n in graph['nodes']]
    value['document']['edges']=[{'id':str(i),'kind':'table','from':{'address':{'domain':'data','localID':e['from']['node']},'port':e['from']['port']},'to':{'address':{'domain':'data','localID':e['to']['node']},'port':e['to']['port']}} for i,e in enumerate(graph['edges'])]
    value['document']['nodes'].append({'address':{'domain':'control','localID':'train'},'name':'Train','x':0,'y':0,'payload':{'kind':'train','config':{}}})
    model={'schemaVersion':2,'name':'Tiny model','nodes':[{'id':'input','type':'input','config':{'features':2}},{'id':'linear','type':'linear','config':{'in_features':2,'out_features':2,'bias':True}},{'id':'output','type':'output','config':{}}],'edges':[{'id':'a','source':'input','sourcePort':'out','target':'linear','targetPort':'in'},{'id':'b','source':'linear','sourcePort':'out','target':'output','targetPort':'in'}]}
    optimizer={'schemaVersion':1,'configurationVersion':1,'componentID':'optimizer_sgd','config':{'lr':0.05}}
    value['trainings']=[{'id':'train','backendID':'pytorch','graph':model,'settings':{'device':'cpu','epochs':2,'seed':5,'batchSize':8,'learningRate':0.03},'prepared':{'trainingNode':'training','trainingPort':'table','validationNode':'heldout','validationPort':'table','featureColumns':['feature','second'],'targetColumn':'group'},'recipes':{'optimizer_recipe':optimizer},'recipeSources':{},'trainingMode':'fresh','preparationCachePolicy':'off'}]
    return value


def test_real_pytorch_whole_workflow_replay_honors_optimizer_and_prepared_rows(tmp_path):
    from runner.workflow_code import execute_snapshot
    from runner.optimizer_configuration import validate_recipe
    value=training_snapshot();root=Path(__file__).parents[2];events=[]
    execute_snapshot(value,{'sources':{'source':str(root/'examples/dataflow/observations.csv')}},tmp_path/'replay',events.append)
    manifest=json.loads((tmp_path/'replay/training-000/manifest.json').read_text())
    assert manifest['status']=='completed' and manifest['settings']['epochs']==2
    assert manifest['settings']['optimizerRecipe']==validate_recipe(value['trainings'][0]['recipes']['optimizer_recipe'])
    assert manifest['evaluation']['count']==8 and manifest['provenance']['trainingSamples']==24
    assert events[-1]['event']=='workflowCodeCompleted'


def test_registry_nil_optional_projection_keeps_schema_and_version_authority():
    value=snapshot()
    for op in value['dataRegistry']['operators']:
        for port in op['inputs']+op['outputs']:
            if port.get('maximumConnections') is None:port.pop('maximumConnections',None)
    generate_package(value)
    value['dataRegistry']['operators'][0]['version']['patch']+=1
    with pytest.raises(ValueError,match='registry'):generate_package(value)


def test_changed_preparation_vs_document_refused_before_generation():
    value=snapshot();value['preparation']['graph']['nodes'][0]['config']['separator']=';'
    with pytest.raises(ValueError,match='captured document'):generate_package(value)


def test_mixed_cache_policies_refused_without_output_side_effect(tmp_path):
    from runner.workflow_code import execute_snapshot
    value=training_snapshot();other=copy.deepcopy(value['trainings'][0]);other['id']='second';other['preparationCachePolicy']='materialize'
    value['trainings'].append(other);node=copy.deepcopy(value['document']['nodes'][-1]);node['address']['localID']='second';value['document']['nodes'].append(node)
    with pytest.raises(ValueError,match='mixed policies'):execute_snapshot(value,{},tmp_path/'absent')
    assert not (tmp_path/'absent').exists()


def test_real_sklearn_whole_workflow_if_optional_backend_is_installed(tmp_path):
    pytest.importorskip('sklearn',reason='Optional sklearn backend is not installed in this interpreter')
    from runner import sklearn_backend
    from runner.workflow_code import execute_snapshot
    value=training_snapshot();training=value['trainings'][0]
    training['backendID']='sklearn';training['configuration']=sklearn_backend.default_configuration();training.pop('graph');training.pop('settings');training['recipes']={}
    value['document']['nodes'][-1]['payload']={'kind':'estimator','config':{'backendID':'sklearn'}}
    root=Path(__file__).parents[2]
    execute_snapshot(value,{'sources':{'source':str(root/'examples/dataflow/observations.csv')}},tmp_path/'replay',lambda _:None)
    manifest=json.loads((tmp_path/'replay/training-000/manifest.json').read_text())
    assert manifest['backendID']=='sklearn' and manifest['evaluation']['count']==8


def test_real_tensorflow_whole_workflow_if_optional_backend_is_installed(tmp_path):
    pytest.importorskip('tensorflow',reason='Optional TensorFlow backend is not installed in this interpreter')
    from runner.workflow_code import execute_snapshot
    value=training_snapshot();root=Path(__file__).parents[2];example=json.loads((root/'runner/backend_examples/tensorflow_classifier.json').read_text())
    training=value['trainings'][0];training['backendID']='tensorflow';training['configuration']={**example['settings'],'epochs':2};training['graph']={**example['graph'],'name':'TensorFlow model'};training.pop('settings');training['recipes']={}
    value['document']['nodes'][-1]['payload']={'kind':'estimator','config':{'backendID':'tensorflow'}}
    execute_snapshot(value,{'sources':{'source':str(root/'examples/dataflow/observations.csv')}},tmp_path/'replay',lambda _:None)
    manifest=json.loads((tmp_path/'replay/training-000/manifest.json').read_text())
    assert manifest['backendID']=='tensorflow' and manifest['evaluation']['samples']==8


def test_workflow_code_captures_sampling_in_shared_preparation_plan():
    from runner.preparation_plan import PreparationPlan
    value=snapshot();value['preparation']['sampling']={'schemaVersion':1,'mode':'rowCount','rowCount':7}
    value['preparation']['completeNodes']=[]
    package=generate_package(value)
    namespace={'__name__':'inspect_generated'}
    exec(package['files']['preparation.py'],namespace)
    plan=PreparationPlan.from_json(json.dumps(namespace['PLAN']))
    assert plan.to_document()['demand']['sampling']['rowCount']==7
    assert plan.to_document()['demand']['sampledNodeIDs']==['source']


def test_older_code_snapshot_omitting_optional_all_sampling_keeps_same_preparation_program():
    from runner.workflow_code import _preparation_demand
    older=snapshot();explicit=copy.deepcopy(older)
    explicit['preparation'].update(sampling={'schemaVersion':1,'mode':'all'},completeNodes=[])
    assert _preparation_demand(validate_snapshot(older))==_preparation_demand(validate_snapshot(explicit))
    assert generate_package(older)['files']['preparation.py']==generate_package(explicit)['files']['preparation.py']


def test_workflow_code_training_demands_complete_required_branches():
    from runner.workflow_code import _preparation_demand
    value=training_snapshot()
    value['preparation']['sampling']={'mode':'rowCount','rowCount':3}
    demand=_preparation_demand(validate_snapshot(value))
    for field in ('trainingNode','validationNode'):
        assert demand.is_complete(value['trainings'][0]['prepared'][field])
    assert demand.is_complete('source')


def test_workflow_code_rejects_unknown_sampling_fields_and_complete_nodes():
    value=snapshot();value['preparation']['sampling']={'mode':'random'}
    with pytest.raises(ValueError):validate_snapshot(value)
    value=snapshot();value['preparation']['completeNodes']=['unknown']
    with pytest.raises(ValueError):validate_snapshot(value)


def model_owned_input_snapshot():
    value=snapshot();value['scope']='preparation'
    value['document']['nodes'] += [
        {'address':{'domain':'model','localID':'entry'},'name':'Retained input','x':0,'y':0,'payload':{'kind':'modelPrimitive','type':'input','configurationVersion':2,'config':{'features':2}}},
        {'address':{'domain':'control','localID':'model'},'name':'Model','x':0,'y':0,'payload':{'kind':'modelDefinition'}}]
    value['document']['modelInputs']={'model':'entry'}
    value['document']['modelSourceVersions']={'model':2}
    return value


def test_model_input_ownership_is_retained_in_portable_code_package():
    value=model_owned_input_snapshot()
    package=generate_package(value)
    captured=json.loads(package['files']['workflow-plan.json'])
    assert captured['document']['modelInputs']=={'model':'entry'}
    assert captured['document']['modelSourceVersions']=={'model':2}
    assert captured['document']['nodes'][-2]['payload']['config']=={'features':2}
    assert value['document']['modelInputs']=={'model':'entry'}


@pytest.mark.parametrize('ownership,version',[
    ({'model':'missing'},14),({'missing':'entry'},14),({'model':'source'},14),
    ({'model':'entry'},13),({'model':42},14),([],14),({'model':'entry'},15),
])
def test_model_input_ownership_rejects_wrong_refs_types_and_schema(ownership,version):
    value=model_owned_input_snapshot();value['document']['modelInputs']=ownership;value['document']['schemaVersion']=version
    with pytest.raises(ValueError):validate_snapshot(value)


@pytest.mark.parametrize('versions',[{'model':1},{'model':4},{'missing':2},{'model':True},[]])
def test_retained_model_source_versions_are_bounded_and_reference_the_owner(versions):
    value=model_owned_input_snapshot();value['document']['modelSourceVersions']=versions
    with pytest.raises(ValueError):validate_snapshot(value)


def test_layer_owned_workflow_requires_matching_compiler_pair():
    value=snapshot();value['document']['schemaVersion']=15
    with pytest.raises(ValueError,match='compiler contract'):validate_snapshot(value)
    value['compilerContractVersion']=13
    assert validate_snapshot(value)['document']['schemaVersion']==15
    value['document']['schemaVersion']=14
    with pytest.raises(ValueError,match='compiler contract'):validate_snapshot(value)


def test_inline_run_workflow_requires_its_exact_compiler_pair():
    value=snapshot();value['document']['schemaVersion']=16
    value['compilerContractVersion']=14
    assert validate_snapshot(value)['document']['schemaVersion']==16
    for compiler in (12,13,15):
        changed=copy.deepcopy(value);changed['compilerContractVersion']=compiler
        with pytest.raises(ValueError,match='compiler contract'):validate_snapshot(changed)
    legacy=snapshot();legacy['compilerContractVersion']=14
    with pytest.raises(ValueError,match='compiler contract'):validate_snapshot(legacy)


def test_trained_artifact_workflow_requires_its_exact_compiler_pair():
    value=snapshot();value['document']['schemaVersion']=17
    value['compilerContractVersion']=15
    # Schemas20/21 add reviewed model-source/fine-tuning configuration,
    # without changing compiler15 for supported canvas workflows.
    for schema in (17,18,19,20,21,22,23):
        supported=copy.deepcopy(value);supported['document']['schemaVersion']=schema
        assert validate_snapshot(supported)['document']['schemaVersion']==schema
    for schema,compiler in ((17,14),(16,15),(21,14),(24,15)):
        changed=copy.deepcopy(value);changed['document']['schemaVersion']=schema
        changed['compilerContractVersion']=compiler
        with pytest.raises(ValueError):validate_snapshot(changed)


def test_consolidated_model_output_retains_owned_input_and_source_version():
    value = model_owned_input_snapshot()
    node = value['document']['nodes'][-1]
    node['address']['domain'] = 'model'
    node['payload'] = {'kind':'modelPrimitive','type':'output','configurationVersion':2,'config':{}}
    package = generate_package(value)
    captured = json.loads(package['files']['workflow-plan.json'])
    assert captured['document']['modelInputs'] == {'model':'entry'}
    assert captured['document']['modelSourceVersions'] == {'model':2}
    node['payload']['type'] = 'linear'
    with pytest.raises(ValueError, match='retained model source'):
        validate_snapshot(value)


def test_consolidated_output_cannot_disappear_from_whole_workflow_code():
    value = snapshot()
    value['document']['nodes'].append({'address':{'domain':'model','localID':'prediction'},
        'name':'Prediction','x':0,'y':0,
        'payload':{'kind':'modelPrimitive','type':'output','configurationVersion':2,'config':{}}})
    with pytest.raises(ValueError, match='terminal coverage'):
        generate_package(value)


def test_wrong_domain_carrier_cannot_hide_a_model_output_from_code_coverage():
    value = snapshot()
    value['document']['nodes'] += [
        {'address':{'domain':'model','localID':'prediction'},'name':'Prediction','x':0,'y':0,
         'payload':{'kind':'modelPrimitive','type':'output','configurationVersion':2,'config':{}}},
        {'address':{'domain':'model','localID':'fake-carrier'},'name':'Wrong owner','x':0,'y':0,
         'payload':{'kind':'modelDefinition'}}]
    value['document']['edges'].append({'id':'fake-bridge','kind':'modelDefinition',
        'from':{'address':{'domain':'model','localID':'prediction'},'port':'definition'},
        'to':{'address':{'domain':'model','localID':'fake-carrier'},'port':'source'}})
    with pytest.raises(ValueError):
        generate_package(value)


def test_train_owned_mapping_source_schema_uses_existing_lowered_compiler():
    value=snapshot();value['document']['schemaVersion']=18;value['compilerContractVersion']=15
    assert validate_snapshot(value)['document']['schemaVersion']==18


def test_real_workflow_code_preserves_selected_model_and_optional_heatmaps(tmp_path):
    from runner.workflow_code import execute_snapshot
    value=training_snapshot();root=Path(__file__).parents[2]
    control=dict(schemaVersion=1,modelSelection='bestValidation',monitor='validationLoss',earlyStopping=dict(enabled=True,patience=2,minimumImprovement=1000000))
    capture=dict(schemaVersion=4,activationValues=False,predictionRows=False,performanceMeasurements=False,retainComponentOutputs=True,retainDiagnosticDetails=False,tensorHeatmaps=True,heatmapFrameLimit=6)
    value['trainings'][0]['settings'].update(epochs=8,trainingControl=control)
    value['trainings'][0]['recipes']['sample_capture']=capture
    package=generate_package(value)
    execute_snapshot(json.loads(package['files']['workflow-plan.json']),{'sources':{'source':str(root/'examples/dataflow/observations.csv')}},tmp_path/'replay',lambda _:None)
    manifest=json.loads((tmp_path/'replay/training-000/manifest.json').read_text())
    assert manifest['trainingControl']['completedEpoch']==3
    assert manifest['trainingControl']['policy']==control
    assert manifest['settings']['epochLimit']==8
    activations=json.loads((tmp_path/'replay/training-000/activations.json').read_text())
    assert activations['sampleCapture']==capture


def causal_snapshot():
    value=snapshot();value['document']['schemaVersion']=21;value['compilerContractVersion']=15;value['scope']='training'
    run={'device':'cpu','epochs':1,'batchSize':2,'seed':23,'learningRate':.001}
    config={'fineTuning':{'mode':'lora','targets':['attentionInput'],'rank':2,'alpha':4,'maximumTokens':8,'stride':7,'exampleFormat':'promptResponse','promptColumn':'prompt','responseColumn':'response','separator':' '},'dataMapping':{},'runSetup':run}
    partition={'id':'partition','operator':'kwip.data.partition','config':{'strategy':'random','seed':42,'validationPercent':25,'testPercent':25}}
    value['preparation']['graph']['nodes'].append(partition)
    value['preparation']['graph']['edges']=[{'from':{'node':'source','port':'table'},'to':{'node':'partition','port':'table'}}]
    for domain,ident,payload in [('data','partition',{'kind':'dataOperator','operatorID':partition['operator'],'config':partition['config']}),('control','loaded',{'kind':'frozenModel','config':{}}),('control','train',{'kind':'train','config':config})]:
        value['document']['nodes'].append({'address':{'domain':domain,'localID':ident},'name':ident,'x':0,'y':0,'payload':payload})
    def edge(a,aport,b,bport,kind):return {'from':{'address':{'domain':'data' if a in ('source','partition') else 'control','localID':a},'port':aport},'to':{'address':{'domain':'data' if b=='partition' else 'control','localID':b},'port':bport},'kind':kind}
    value['document']['edges']=[edge('source','table','partition','table','table'),edge('loaded','model','train','model','frozenModel'),edge('partition','training','train','training','table'),edge('partition','validation','train','validation','table')]
    value['preparation']['nodeLabels']['partition']='partition'
    value['selectedTargets']=[{'domain':'control','localID':'train'}]
    value['trainings']=[dict(id='train',backendID='pretrainedCausal',configuration={'model':'loaded','train':config},prepared=dict(trainingNode='partition',trainingPort='training',validationNode='partition',validationPort='validation',featureColumns=[]),recipes={},recipeSources={},trainingMode='fresh',preparationCachePolicy='off')]
    return value


def test_portable_causal_train_delegates_actual_response_loss_and_refuses_changed_mapping(tmp_path):
    # Use the actual small safetensors/tokenizer fixture, not a mocked training adapter.
    from runner.tests.test_pretrained_models import folder
    from runner.workflow_code import execute_snapshot
    from runner.pretrained_training import verify_result
    fixture=folder.__wrapped__(tmp_path)
    value=causal_snapshot();source=tmp_path/'private.csv';source.write_text('prompt,response\n'+('hello,world .\nworld,hello .\n'*8))
    before={p.name:p.read_bytes() for p in fixture.iterdir() if p.is_file()}
    package=generate_package(value)
    assert str(fixture) not in ''.join(package['files'].values())
    bindings=dict(sources={'source':str(source)},models={'loaded':str(fixture)},resources={'maximumResidentBytes':200_000_000})
    execute_snapshot(value,bindings,tmp_path/'run',emit=lambda _:None)
    report=verify_result(tmp_path/'run/training-000')
    assert report['settings']['exampleFormat']=='promptResponse' and report['settings']['mode']=='lora'
    assert before=={p.name:p.read_bytes() for p in fixture.iterdir() if p.is_file()}
    altered=copy.deepcopy(value);altered['trainings'][0]['prepared']['validationPort']='test'
    with pytest.raises(ValueError,match='bindings differ'):validate_snapshot(altered)
    with pytest.raises(ValueError,match='model folder'):execute_snapshot(value,{'sources':{'source':str(source)}},tmp_path/'missing',emit=lambda _:None)
