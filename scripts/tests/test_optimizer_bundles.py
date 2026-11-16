from scripts.tests.registry_history import historical_output_presentation
"""Optimizer recipes are bounded training controls, never model-forward adapters."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('optimizer_bundle_authoring',ROOT/'scripts/component_bundles.py')
bundles=importlib.util.module_from_spec(spec);spec.loader.exec_module(bundles)
IDS=('optimizer_sgd','optimizer_adam','optimizer_adamw','optimizer_rmsprop','optimizer_adagrad')
NAMES=('SGD','Adam','AdamW','RMSprop','Adagrad')
TASKS=('sgd-momentum-steps','adam-running-averages','adamw-separate-decay','rmsprop-squared-memory','adagrad-accumulated-steps')
def source():return json.loads((ROOT/'components/optimizers/bundle.json').read_text())
def row(v,ident):return next(r for r in v['components']if r['definition']['id']==ident)
def fixture_root(v,tmp_path):
    # Synthetic files check metadata policy only; never represent delivered artwork/code.
    for r in v['components']:
        refs=list(r['sources'].values())+['Sources/KwipMLApp/Resources/ComponentIcons/'+r['presentation']['icon']+'.png']
        refs += [r['learning']['exampleSource'],'Sources/KwipMLApp/Resources/workflow-observations.csv']+['Sources/KwipMLCore/Optimizer'+n+'.swift' for n in ['LearningTask','LessonReference','LearningSubmission','CapabilityBridge']]
        for ref in refs:
            p=tmp_path/ref;p.parent.mkdir(parents=True,exist_ok=True);p.touch()
    return tmp_path

def test_prior53_registry_definitions_preserved():
    v=json.loads((ROOT/'Sources/KwipMLCore/Resources/components.json').read_text())
    old=[r for r in v['components']if r['id']not in ('rms_norm','rotary_position','mixture_experts',*IDS,'bag_input','embedding_bag','dataset_source','data_loader','initialization','gradient_policy','trainability','scheduler_step_lr','scheduler_cosine_annealing_lr','scheduler_reduce_on_plateau','scheduler_one_cycle_lr','scheduler_linear_lr','rnn','lstm','gru','rnn_cell','lstm_cell','gru_cell','multihead_attention','transformer','transformer_encoder','transformer_decoder','scaled_dot_product_attention','attention_mask','loss_cross_entropy','loss_bce_with_logits','loss_mse','loss_l1','loss_nll','loss_kl_div','loss_ctc','loss_triplet_margin')]
    assert len(old)==53
    old=historical_output_presentation(old)
    assert hashlib.sha256(json.dumps(old,sort_keys=True,separators=(',',':')).encode()).hexdigest()=='412735b7d46a0906a5f32eeb77cb93f07429f7b1249069f15b9cff8306c3f7fe'

@pytest.mark.parametrize('ident,name,task',list(zip(IDS,NAMES,TASKS)))
def test_exact_control_recipe_identity(ident,name,task):
    v=source();assert [r['definition']['id']for r in v['components']]==list(IDS)
    r=row(v,ident);d=r['definition']
    assert d['title']==name and d['category']=='Training'
    assert d['inputs']==[]and d['outputs']==['out']
    assert r['configurationVersion']==1 and r['published']is True
    assert r['adapterID']=='kwip.runner.optimizer.'+ident
    assert r['shapeRuleID']=='kwip.recipe.'+ident+'.v1'
    assert r['outputConditions']=={}
    assert r['presentation']['theme']=='green'and r['presentation']['icon']==ident+'-v2'
    assert r['learning']['lessonIDs']==['assessment.'+task]
    assert r['learning']['exampleID']=='prepared-'+ident.removeprefix('optimizer_')+'-classifier'
    assert 'graphSchemaVersions'not in r['runtimeContract']
    assert r['runtimeContract']['recipeSchemaVersions']==[1]

@pytest.mark.parametrize('ident',IDS)
def test_strict_metadata_accepts_reviewed_recipe_only(ident,tmp_path):
    v=source();root=fixture_root(v,tmp_path);bundles.validate(v,root)
    row(v,ident)['runtimeContract']['graphSchemaVersions']=[2]
    with pytest.raises(ValueError):bundles.validate(v,root)

def test_generated_references_are_distinct_real_training_factories():
    rendered=bundles.render_swift(source(),'optimizers')
    for ident,name in zip(IDS,NAMES):
        assert 'OptimizerCapabilityBridge.'+ident.removeprefix('optimizer_')+'Declaration'in rendered
        assert 'WorkflowExamples.prepared'+name+'Classifier(device: $0)'in rendered
    assert 'BundledOptimizerComponents'in rendered

def test_generated_python_table_is_recipe_only():
    import sys
    sys.path.insert(0,str(ROOT/'runner'))
    table={};exec(bundles.render_optimizer_adapters(source()),table)
    assert set(table['IDS'])==set(IDS)
    assert 'RESOLVERS'not in table
    assert table['NORMALIZERS']['optimizer_sgd']({})['lr']==.001
    assert table['NORMALIZERS']['optimizer_adamw']({})['weight_decay']==.01

@pytest.mark.parametrize('ident',IDS)
@pytest.mark.parametrize('mutation',['field','default','publication','adapter','shape','dtype','recipeVersion','group','buffers','flags','ports','icon','theme','lesson','example','source','unknown','learning'])
def test_unreviewed_mutations_refuse(ident,mutation,tmp_path):
    v=source();root=fixture_root(v,tmp_path);bundles.validate(v,root);r=row(v,ident)
    if mutation=='field':r['definition']['fields'].append({'key':'arbitrary'})
    elif mutation=='default':r['definition']['fields'][0]['defaultValue']=True
    elif mutation=='publication':r['published']=1
    elif mutation=='adapter':r['adapterID']='kwip.runner.graph-v2.linear'
    elif mutation=='shape':r['shapeRuleID']='kwip.shape.linear.v1'
    elif mutation=='dtype':r['runtimeContract']['graphDTypes']=['int64']
    elif mutation=='recipeVersion':r['runtimeContract']['recipeSchemaVersions']=[1,2]
    elif mutation=='group':r['runtimeContract']['oneParameterGroup']=False
    elif mutation=='buffers':r['runtimeContract']['excludesBuffers']=False
    elif mutation=='flags':r['runtimeContract']['foreach']=True
    elif mutation=='ports':r['definition']['inputs']=['in']
    elif mutation=='icon':r['presentation']['icon']='linear-v2'
    elif mutation=='theme':r['presentation']['theme']='violet'
    elif mutation=='lesson':r['learning']['lessonIDs']=['assessment.linear']
    elif mutation=='example':r['learning']['exampleID']='prepared-parallel-classifier'
    elif mutation=='source':r['sources']['worker']='../escape.py'
    elif mutation=='unknown':r['runtimeContract']['forward']=True
    else:r['learning']=None
    with pytest.raises(ValueError):bundles.validate(v,root)

@pytest.mark.parametrize('ident',IDS)
def test_missing_original_art_refuses(ident,tmp_path):
    v=source();root=fixture_root(v,tmp_path)
    (root/'Sources/KwipMLApp/Resources/ComponentIcons'/f'{ident}-v2.png').unlink()
    with pytest.raises(ValueError,match='source reference'):bundles.validate(v,root)

@pytest.mark.parametrize('ident',IDS)
def test_actual_runtime_descriptor_exact_literal(ident):
    import sys
    sys.path.insert(0,str(ROOT/'runner'))
    from optimizer_configuration import capability
    assert row(source(),ident)['runtimeContract']==capability(ident)

@pytest.mark.parametrize('ident',IDS)
def test_unpublished_optimizer_absent_from_all_callable_tables(ident):
    import sys
    sys.path.insert(0,str(ROOT/'runner'))
    v=source();row(v,ident)['published']=False
    table={};exec(bundles.render_optimizer_adapters(v),table)
    for key in ('IDS','CAPABILITIES','NORMALIZERS'):assert set(table[key])==set(IDS)-{ident}

def test_all_prior_generated_family_bytes_unchanged():
    names={'pooling':'Pooling','activations':'Activation','normalization':'Normalization','layers':'Layer','tensor-layout':'TensorLayout','tensor-math':'TensorMath','embeddings':'Embedding','convolution':'Convolution','tensor-composition':'TensorComposition','tensor-routing':'TensorRouting','spatial-patches':'SpatialPatch'}
    for family,name in names.items():
        value=json.loads((ROOT/'components'/family/'bundle.json').read_text())
        assert bundles.render_swift(value,family)==(ROOT/'Sources/KwipMLCore'/f'Bundled{name}Components.generated.swift').read_text(),family

@pytest.mark.parametrize('ident,defaults',[
    ('optimizer_sgd',dict(lr=.001,momentum=0,dampening=0,weight_decay=0,nesterov=False,maximize=False)),
    ('optimizer_adam',dict(lr=.001,beta1=.9,beta2=.999,eps=1e-8,weight_decay=0,amsgrad=False,decoupled_weight_decay=False,maximize=False)),
    ('optimizer_adamw',dict(lr=.001,beta1=.9,beta2=.999,eps=1e-8,weight_decay=.01,amsgrad=False,maximize=False)),
    ('optimizer_rmsprop',dict(lr=.01,alpha=.99,eps=1e-8,weight_decay=0,momentum=0,centered=False,maximize=False)),
    ('optimizer_adagrad',dict(lr=.01,lr_decay=0,weight_decay=0,initial_accumulator_value=0,eps=1e-10,maximize=False)),
])
def test_fields_match_independently_reviewed_defaults_and_bounds(ident,defaults):
    fields=row(source(),ident)['definition']['fields']
    assert {f['key']:f['defaultValue']for f in fields}==defaults
    assert [f['key']for f in fields]==list(defaults)
    for f in fields:
        if type(defaults[f['key']])is bool:
            assert f['kind']=='boolean'and 'min'not in f and 'max'not in f
        else:
            assert f['kind']=='number'
            assert f['min']==(1.1754943508222875e-38 if f['key']=='eps'else 0)
            assert f['max']==(.9999999403953552 if f['key']in ('beta1','beta2')else 1 if f['key']in ('momentum','dampening','alpha')else 3.4028234663852886e38)

@pytest.mark.parametrize('ident',IDS)
def test_every_declared_field_default_is_frozen(ident,tmp_path):
    v=source();root=fixture_root(v,tmp_path)
    for index,field in enumerate(row(v,ident)['definition']['fields']):
        changed=copy.deepcopy(v)
        changed['components'][IDS.index(ident)]['definition']['fields'][index]['defaultValue']=not field['defaultValue']if type(field['defaultValue'])is bool else -1
        with pytest.raises(ValueError):bundles.validate(changed,root)

def test_canonical_outputs_resolve_actual_resources_and_match_generated_files():
    rendered=bundles.render(ROOT)
    for path in ['Sources/KwipMLCore/Resources/components.json','Sources/KwipMLCore/BundledOptimizerComponents.generated.swift','runner/optimizer_adapters_generated.py']:
        assert (ROOT/path).read_text()==rendered[path],path
    assert len([r for r in json.loads(rendered['Sources/KwipMLCore/Resources/components.json'])['components'] if r['id'] not in ('rms_norm','rotary_position','mixture_experts','initialization','gradient_policy','trainability','bag_input','embedding_bag','dataset_source','data_loader')])==83
