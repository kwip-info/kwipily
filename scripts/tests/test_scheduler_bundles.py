from scripts.tests.registry_history import historical_output_presentation
"""Schedulers are bounded training recipes; preserve every accepted earlier component."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('scheduler_bundle_authoring',ROOT/'scripts/component_bundles.py')
bundles=importlib.util.module_from_spec(spec);spec.loader.exec_module(bundles)
IDS=('scheduler_step_lr','scheduler_cosine_annealing_lr','scheduler_reduce_on_plateau','scheduler_one_cycle_lr','scheduler_linear_lr')
NAMES=('StepLR','CosineAnnealingLR','ReduceLROnPlateau','OneCycleLR','LinearLR')
TASKS=('step-lr-intervals','cosine-lr-curve','plateau-lr-improvement','one-cycle-lr-batches','linear-lr-warmup')
def source():return json.loads((ROOT/'components/schedulers/bundle.json').read_text())
def row(v,ident):return next(r for r in v['components'] if r['definition']['id']==ident)
def fixture_root(v,tmp_path):
    # Synthetic reference files test authoring validation, never delivered code or artwork.
    for r in v['components']:
        refs=list(r['sources'].values())+['Sources/KwipMLApp/Resources/ComponentIcons/'+r['presentation']['icon']+'.png',r['learning']['exampleSource'],'Sources/KwipMLApp/Resources/workflow-observations.csv']
        refs+=['Sources/KwipMLCore/Scheduler'+n+'.swift' for n in ['LearningTask','LessonReference','LearningSubmission','CapabilityBridge']]
        for ref in refs:
            p=tmp_path/ref;p.parent.mkdir(parents=True,exist_ok=True);p.touch()
    return tmp_path
@pytest.mark.parametrize('ident,name,task',list(zip(IDS,NAMES,TASKS)))
def test_exact_recipe_identity(ident,name,task):
    v=source();assert [r['definition']['id'] for r in v['components']]==list(IDS)
    r=row(v,ident);d=r['definition'];assert d['title']==name and d['category']=='Training'
    assert d['inputs']==[] and d['outputs']==['out']
    assert r['configurationVersion']==1 and r['published'] is True
    assert r['adapterID']=='kwip.runner.scheduler.'+ident
    assert r['shapeRuleID']=='kwip.recipe.'+ident+'.v1'
    assert r['outputConditions']=={} and r['presentation']['theme']=='green'
    assert r['presentation']['icon']==ident+'-v2'
    assert r['learning']['lessonIDs']==['assessment.'+task]
    assert r['learning']['exampleID']=='prepared-'+ident.removeprefix('scheduler_').replace('_','-')+'-classifier'
    assert 'graphSchemaVersions' not in r['runtimeContract']
    assert r['runtimeContract']['recipeSchemaVersions']==[1]
@pytest.mark.parametrize('ident',IDS)
def test_exact_descriptor_matches_actual_reviewed_helper(ident):
    import sys
    sys.path.insert(0,str(ROOT/'runner'))
    from scheduler_configuration import capability
    assert row(source(),ident)['runtimeContract']==capability(ident)
def test_training_only_generated_tables_and_factories():
    v=source();rendered=bundles.render_swift(v,'schedulers')
    assert 'BundledSchedulerComponents' in rendered
    for ident,name,declaration in zip(IDS,NAMES,['stepLRDeclaration','cosineAnnealingLRDeclaration','reduceOnPlateauDeclaration','oneCycleLRDeclaration','linearLRDeclaration']):
        assert 'SchedulerCapabilityBridge.'+declaration in rendered
        assert 'WorkflowExamples.prepared'+name+'Classifier(device: $0)' in rendered
    text=bundles.render_scheduler_adapters(v)
    assert 'RESOLVERS' not in text and 'NORMALIZERS' in text and 'CAPABILITIES' in text
@pytest.mark.parametrize('ident',IDS)
def test_reviewed_metadata_validates_with_synthetic_references(ident,tmp_path):
    v=source();bundles.validate(v,fixture_root(v,tmp_path))
    row(v,ident)['runtimeContract']['graphSchemaVersions']=[2]
    with pytest.raises(ValueError):bundles.validate(v,tmp_path)

def test_prior58_ordered_registry_definitions_exact():
    value=json.loads((ROOT/'Sources/KwipMLCore/Resources/components.json').read_text())
    old=[r for r in value['components'] if r['id'] not in ('rms_norm','rotary_position','mixture_experts',*IDS,'bag_input','embedding_bag','dataset_source','data_loader','initialization','gradient_policy','trainability','rnn','lstm','gru','rnn_cell','lstm_cell','gru_cell','multihead_attention','transformer','transformer_encoder','transformer_decoder','scaled_dot_product_attention','attention_mask','loss_cross_entropy','loss_bce_with_logits','loss_mse','loss_l1','loss_nll','loss_kl_div','loss_ctc','loss_triplet_margin')]
    assert len(old)==58
    old=historical_output_presentation(old)
    assert hashlib.sha256(json.dumps(old,sort_keys=True,separators=(',',':')).encode()).hexdigest()=='7a4c8f74e4205d74e687cc7c5a033b1420158e875a05cb828c939467e39f786d'

@pytest.mark.parametrize('ident',IDS)
@pytest.mark.parametrize('mutation',['field','default','publication','adapter','shape','descriptor','ports','icon','theme','lesson','example','source','unknown','learning'])
def test_unreviewed_recipe_mutations_refuse(ident,mutation,tmp_path):
    v=source();root=fixture_root(v,tmp_path);bundles.validate(v,root);r=row(v,ident)
    if mutation=='field':r['definition']['fields'].append({'key':'arbitrary'})
    elif mutation=='default':r['definition']['fields'][0]['defaultValue']=True
    elif mutation=='publication':r['published']=1
    elif mutation=='adapter':r['adapterID']='kwip.runner.graph-v2.linear'
    elif mutation=='shape':r['shapeRuleID']='kwip.shape.linear.v1'
    elif mutation=='descriptor':r['runtimeContract']['recipeSchemaVersions']=[1,2]
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
def test_original_generated_icon_required(ident,tmp_path):
    v=source();root=fixture_root(v,tmp_path)
    (root/'Sources/KwipMLApp/Resources/ComponentIcons'/f'{ident}-v2.png').unlink()
    with pytest.raises(ValueError,match='source reference'):bundles.validate(v,root)

def test_all_prior_generated_family_bytes_unchanged():
    names={'pooling':'Pooling','activations':'Activation','normalization':'Normalization','layers':'Layer','tensor-layout':'TensorLayout','tensor-math':'TensorMath','embeddings':'Embedding','convolution':'Convolution','tensor-composition':'TensorComposition','tensor-routing':'TensorRouting','spatial-patches':'SpatialPatch','optimizers':'Optimizer'}
    for family,name in names.items():
        value=json.loads((ROOT/'components'/family/'bundle.json').read_text())
        assert bundles.render_swift(value,family)==(ROOT/'Sources/KwipMLCore'/f'Bundled{name}Components.generated.swift').read_text(),family

@pytest.mark.parametrize('ident,defaults',[
 ('scheduler_step_lr',dict(step_size=2,gamma=.1)),
 ('scheduler_cosine_annealing_lr',dict(T_max=10,eta_min=0)),
 ('scheduler_reduce_on_plateau',dict(metric='validation_loss',mode='min',factor=.1,patience=10,threshold=.0001,threshold_mode='rel',cooldown=0,min_lr=0,eps=1e-8)),
 ('scheduler_one_cycle_lr',dict(max_lr=.1,planned_total_batches=10,pct_start=.3,anneal_strategy='cos',cycle_momentum=True,base_momentum=.85,max_momentum=.95,div_factor=25,final_div_factor=10000,three_phase=False)),
 ('scheduler_linear_lr',dict(start_factor=1/3,end_factor=1,total_iters=5)),
])
def test_independently_reviewed_control_defaults_and_bounds(ident,defaults):
    fields=row(source(),ident)['definition']['fields']
    assert [f['key'] for f in fields]==list(defaults)
    assert {f['key']:f['defaultValue'] for f in fields}==defaults
    for f in fields:
        if f['kind']=='integer':
            assert f['max']==10000
            assert f['min']==(0 if f['key'] in ('patience','cooldown') else 3 if f['key']=='planned_total_batches' else 1)
        elif f['kind']=='number':
            assert f['min']==(1.1754943508222875e-38 if f['key'] in ('start_factor','end_factor','pct_start','div_factor','final_div_factor') else 0)
            assert f['max']==(1 if f['key'] in ('start_factor','end_factor','pct_start','factor','base_momentum','max_momentum') else 3.4028234663852886e38)

@pytest.mark.parametrize('ident',IDS)
def test_unpublished_recipe_has_no_discovery_or_normalizer(ident):
    import sys
    sys.path.insert(0,str(ROOT/'runner'))
    v=source();row(v,ident)['published']=False
    table={};exec(bundles.render_scheduler_adapters(v),table)
    for key in ('IDS','NORMALIZERS','CAPABILITIES'):assert set(table[key])==set(IDS)-{ident}
    assert 'RESOLVERS' not in table
    for other in set(IDS)-{ident}:
        assert table['NORMALIZERS'][other]({})==row(v,other)['runtimeContract']['defaults']

@pytest.mark.parametrize('ident',IDS)
@pytest.mark.parametrize('field', ['requiresExplicitOptimizer','timing','tickMaximum','traceByteMaximum','horizonPolicy','statePolicy','terminalNextRatePolicy','optimizerComponentIDs'])
def test_runtime_context_contract_mutations_refuse(ident,field,tmp_path):
    v=source();root=fixture_root(v,tmp_path)
    row(v,ident)['runtimeContract'][field]=None
    with pytest.raises(ValueError):bundles.validate(v,root)

def test_actual_sources_and_canonical_outputs_exist_without_drift():
    bundles.validate(source(),ROOT)
    outputs=bundles.render(ROOT)
    assert len([r for r in json.loads(outputs['Sources/KwipMLCore/Resources/components.json'])['components'] if r['id'] not in ('rms_norm','rotary_position','mixture_experts','initialization','gradient_policy','trainability','bag_input','embedding_bag','dataset_source','data_loader')])==83
    for path,text in outputs.items():
        assert (ROOT/path).is_file(),path
        assert (ROOT/path).read_text()==text,path

@pytest.mark.parametrize('ident,expected',[
 ('scheduler_reduce_on_plateau',{'metric':[('validation_loss','Validation loss'),('validation_accuracy','Validation accuracy')],'mode':[('min','Smaller is better'),('max','Larger is better')],'threshold_mode':[('rel','Relative change'),('abs','Absolute change')]}),
 ('scheduler_one_cycle_lr',{'anneal_strategy':[('cos','Smooth cosine'),('linear','Straight lines')]}),
])
def test_choice_fields_use_real_native_component_choice_objects(ident,expected):
    fields={f['key']:f for f in row(source(),ident)['definition']['fields']}
    for key,choices in expected.items():
        assert fields[key]['choices']==[{'value':value,'title':title} for value,title in choices]
