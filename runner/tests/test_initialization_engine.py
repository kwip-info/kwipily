"""Initialization policy in actual training, portable output and continuation."""
import copy
import json
import pytest
import torch
from runner import engine,initialization
from runner.tests.test_initialization import recipe,rule
from runner.tests.test_checkpoint_lifecycle import assert_nested_equal,state
from runner.tests.test_checkpoint_lifecycle_modes import fixture,inventory
from runner.tests.test_optimizer_operation_contract import recipe as optimizer
from runner.tests.test_runner import graph as legacy_graph

@pytest.fixture(autouse=True)
def restore_policy():
    enabled=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled()
    yield
    torch.use_deterministic_algorithms(enabled,warn_only=warn)

@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('algorithm',['xavier_uniform','kaiming_uniform','normal','zeros'])
def test_real_fresh_algorithms_and_exported_fresh_helper(device,algorithm,tmp_path,monkeypatch):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
    graph=legacy_graph();selected=recipe(rule('head','bias',algorithm='normal',mean=.3,std=0),rule('head',algorithm=algorithm))
    calls=[];original=initialization.apply_initialization
    def observe(model,value,*args,**kwargs):
        result=original(model,value,*args,**kwargs);calls.append({k:v.detach().cpu().clone() for k,v in model.state_dict().items()});return result
    monkeypatch.setattr(initialization,'apply_initialization',observe)
    engine.run(graph,tmp_path/'run',epochs=1,batch_size=16,sample_limit=32,device=device,optimizer_recipe=optimizer(),initialization_recipe=selected)
    saved=state(tmp_path/'run');manifest=json.loads((tmp_path/'run/manifest.json').read_text())
    assert len(calls)==1
    for key in calls[0]:assert torch.equal(saved['modelState'][key],calls[0][key])
    assert manifest['settings']['initializationRecipe']==selected
    info=manifest['provenance']['initialization'];contract=manifest['checkpoint']['contract']['initialization']
    assert info=={**contract,'torchVersion':str(torch.__version__),'application':'applied'}
    assert contract==dict(schemaVersion=1,recipe=selected,resolvedOwners=initialization.resolve_owners(engine.GraphModel(graph),selected),generatorPlacement='cpu-float32',seedDerivation=initialization.SEED_VERSION)
    assert len((tmp_path/'run/initialization_recipe.py').read_bytes())<=131072
    assert len((tmp_path/'run/training_recipe.py').read_bytes())<=8192
    namespace={};exec((tmp_path/'run/model.py').read_text(),namespace);portable=namespace['Model']().to(device)
    portable.load_state_dict(saved['modelState']);portable.eval()
    helper={};exec((tmp_path/'run/initialization_recipe.py').read_text(),helper)
    helper['apply_initialization'](portable,helper['INITIALIZATION_RECIPE'])
    for owner in contract['resolvedOwners']:assert torch.equal(portable.state_dict()[owner['name']].cpu(),calls[0][owner['name']])


def test_implicit_adam_lazy_and_reserved_node_export_mapping(tmp_path,monkeypatch):
    graph=dict(schemaVersion=2,nodes=[dict(id='input',type='input',config={'features':2}),dict(id='items',type='lazy_linear',config=dict(out_features=2,bias=True)),dict(id='output',type='output',config={})],edges=[dict(id='a',source='input',sourcePort='out',target='items',targetPort='in'),dict(id='b',source='items',sourcePort='out',target='output',targetPort='in')])
    selected=recipe(rule('items',algorithm='normal',mean=.2,std=0));snapshots=[];original=initialization.apply_initialization
    def observe(model,value,*args,**kwargs):
        assert model.layers[model.layer_keys['items']].weight.shape==(2,2)
        result=original(model,value,*args,**kwargs);snapshots.append(result);return result
    monkeypatch.setattr(initialization,'apply_initialization',observe)
    engine.run(graph,tmp_path/'run',epochs=1,batch_size=16,sample_limit=32,device='cpu',initialization_recipe=selected)
    assert len(snapshots)==1
    namespace={};exec((tmp_path/'run/model.py').read_text(),namespace);model=namespace['Model']()
    assert model.layer_keys['items']!='items'
    model(torch.zeros(1,2))  # Portable LazyLinear requires its documented shape-valid materialization.
    helper={};exec((tmp_path/'run/initialization_recipe.py').read_text(),helper);helper['apply_initialization'](model,selected)
    assert torch.equal(model.layers[model.layer_keys['items']].weight,torch.full((2,2),.2))


def test_cpu_exact_all_state_and_nonfresh_never_reinitializes(tmp_path,monkeypatch):
    graph,options=fixture(tmp_path/'source');selected=recipe(rule(algorithm='normal',mean=.1,std=.4))
    # The genuine prepared graph calls its affine output 'learn'.
    options['initialization_recipe']=selected
    engine.run(graph,tmp_path/'full',epochs=4,**options);engine.run(graph,tmp_path/'parent',epochs=2,**options)
    pins=inventory(tmp_path/'parent')
    def forbidden(*args,**kwargs):raise AssertionError('restored weights must not be initialized')
    monkeypatch.setattr(initialization,'apply_initialization',forbidden)
    engine.run(graph,tmp_path/'resumed',epochs=4,resume_checkpoint=tmp_path/'parent/checkpoint.pt',**options)
    a,b=state(tmp_path/'full'),state(tmp_path/'resumed')
    for key in ('modelState','optimizerState','optimizerUpdates','scheduler','objectiveState','rngState'):assert_nested_equal(a[key],b[key])
    for name in ('metrics.json','rate-trace.json'):assert json.loads((tmp_path/'full'/name).read_text())==json.loads((tmp_path/'resumed'/name).read_text())
    assert json.loads((tmp_path/'resumed/manifest.json').read_text())['provenance']['initialization']['application']=='notAppliedOnRestore'
    for changed in (None,recipe(rule(algorithm='normal',mean=.7,std=.4))):
        altered={**options,'initialization_recipe':changed}
        with pytest.raises(ValueError):engine.run(graph,tmp_path/('absent' if changed is None else 'changed'),epochs=3,resume_checkpoint=tmp_path/'parent/checkpoint.pt',**altered)
    assert inventory(tmp_path/'parent')==pins


@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('selected_child',[False,True])
def test_finetune_selected_or_absent_target_preserves_loaded_weights(tmp_path,monkeypatch,device,selected_child):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
    graph=legacy_graph();parent_recipe=recipe(rule('head',algorithm='normal',mean=2,std=0))
    options=dict(device=device,optimizer_recipe=optimizer(),sample_limit=32,batch_size=16)
    engine.run(graph,tmp_path/'parent',epochs=1,initialization_recipe=parent_recipe,**options);parent=state(tmp_path/'parent');pins=inventory(tmp_path/'parent')
    def forbidden(*args,**kwargs):raise AssertionError('fine-tuning must preserve restored weights')
    monkeypatch.setattr(initialization,'apply_initialization',forbidden)
    child_recipe=recipe(rule('head',algorithm='zeros')) if selected_child else None
    engine.run(graph,tmp_path/'child',epochs=1,finetune_checkpoint=tmp_path/'parent/checkpoint.pt',initialization_recipe=child_recipe,**options)
    assert_nested_equal(parent['modelState'],state(tmp_path/'child')['modelState'])
    manifest=json.loads((tmp_path/'child/manifest.json').read_text())
    assert ('initializationRecipe' in manifest['settings'])==selected_child
    assert ('initialization' in manifest['provenance'])==selected_child
    assert ('initialization' in manifest['checkpoint']['contract'])==selected_child
    assert (tmp_path/'child/initialization_recipe.py').exists()==selected_child
    if selected_child:
        assert manifest['provenance']['initialization']['application']=='notAppliedOnRestore'
        assert manifest['provenance']['initialization']['recipe']==child_recipe
    assert inventory(tmp_path/'parent')==pins


def test_absent_policy_and_invalid_owner_do_not_change_legacy_contract(tmp_path):
    graph=legacy_graph();engine.run(graph,tmp_path/'legacy',epochs=1,device='cpu',sample_limit=32)
    manifest=json.loads((tmp_path/'legacy/manifest.json').read_text())
    assert 'initialization' not in manifest['checkpoint']['contract'] and 'initialization' not in manifest['provenance']
    assert 'initializationRecipe' not in manifest['settings'] and not (tmp_path/'legacy/initialization_recipe.py').exists()
    assert 'self.layer_keys' not in (tmp_path/'legacy/model.py').read_text()
    with pytest.raises(ValueError):engine.run(graph,tmp_path/'refused',epochs=1,device='cpu',initialization_recipe=recipe(rule('missing')))
    assert not (tmp_path/'refused/manifest.json').exists()


def test_discovery_publishes_exact_independent_initialization_descriptor():
    expected={'version':{'major':1,'minor':0,'patch':0},'adapterID':'kwip.runner.initialization','recipeSchemaVersions':[1],'configurationVersions':[1],'algorithms':['xavier_uniform','kaiming_uniform','normal','zeros'],'seedMinimum':0,'seedMaximum':9223372036854775807,'seedDerivation':'kwip.initialization.owner-seed.v1','generatorPlacement':'cpu-float32','devices':['cpu','mps'],'parameterDTypes':['float32'],'maximumRecipeBytes':32768,'maximumRules':128,'maximumModelParameters':200000,'ownerPolicy':'expandedNodeRelativeNativeParameter','ownerOrder':'nodeThenParameter','aliasPolicy':'rejectDuplicateParameterIdentity','fanPolicy':'nativeStoredDimensions','applicationTiming':'freshAfterLazyBeforeOptimizer','restoredWeights':'preserve','candidateCommit':'validateAllBeforeCopy','preservesParameterIdentity':True,'initializesBuffers':False,'standaloneHelperMaximumBytes':131072}
    assert engine.capabilities()['initializationCapabilities']=={'initialization':expected}


@pytest.mark.parametrize('change',['provenance','settings','version','placement','extra','application','torch'])
def test_parent_initialization_metadata_is_bound_before_deserialization(tmp_path,monkeypatch,change):
    from runner.tests.test_checkpoint_lifecycle import rewrite_checkpoint
    selected=recipe(rule('head'));engine.run(legacy_graph(),tmp_path/'parent',epochs=1,device='cpu',sample_limit=32,initialization_recipe=selected)
    def mutate(checkpoint,manifest):
        if change=='provenance':manifest['provenance']['initialization']['recipe']['config']['seed']=43
        elif change=='settings':manifest['settings']['initializationRecipe']['config']['seed']=43
        elif change=='application':manifest['provenance']['initialization']['application']='inherited'
        elif change=='torch':manifest['provenance']['initialization']['torchVersion']='other'
        else:
            if change=='version':checkpoint['contract']['initialization']['schemaVersion']=True
            elif change=='placement':checkpoint['contract']['initialization']['generatorPlacement']='mps'
            else:checkpoint['contract']['initialization']['foreign']=0
            manifest['checkpoint']['contract']=copy.deepcopy(checkpoint['contract'])
    rewrite_checkpoint(tmp_path/'parent',mutate)
    def forbidden(*args,**kwargs):raise AssertionError('unbound initialization metadata reached torch.load')
    monkeypatch.setattr(torch,'load',forbidden)
    with pytest.raises(ValueError):engine.run(legacy_graph(),tmp_path/'child',epochs=1,device='cpu',sample_limit=32,finetune_checkpoint=tmp_path/'parent/checkpoint.pt')
