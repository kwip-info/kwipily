from scripts.tests.registry_history import historical_output_presentation
"""Spatial patch metadata must declare real ordered ports, controls and learning."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('patch_bundles',ROOT/'scripts/component_bundles.py')
bundles=importlib.util.module_from_spec(spec);spec.loader.exec_module(bundles)
IDS=('unfold','fold')
LESSONS={'unfold':['assessment.unfold-overlapping-windows','assessment.unfold-padded-spaced-windows'],'fold':['assessment.fold-overlap-addition','assessment.fold-uncovered-positions']}
def source():return json.loads((ROOT/'components/spatial-patches/bundle.json').read_text())
def row(v,ident):return next(r for r in v['components']if r['definition']['id']==ident)
def fixture_root(v,tmp_path):
    # Isolated synthetic references test validation policy, never actual delivery.
    for r in v['components']:
        refs=list(r['sources'].values())+['Sources/KwipMLApp/Resources/ComponentIcons/'+r['presentation']['icon']+'.png']
        if r['learning']:refs += [r['learning']['exampleSource'],'Sources/KwipMLApp/Resources/'+r['learning']['sourceResource']+'.csv']+['Sources/KwipMLCore/SpatialPatch'+name+'.swift'for name in ['LearningTask','LessonReference','LearningSubmission','CapabilityBridge']]
        for ref in refs:
            p=tmp_path/ref;p.parent.mkdir(parents=True,exist_ok=True);p.touch()
    return tmp_path

def test_prior46_registry_definition_projection_is_unchanged_before_generation():
    v=json.loads((ROOT/'Sources/KwipMLCore/Resources/components.json').read_text())
    old=[d for d in v['components']if d['id']not in ('rms_norm','rotary_position','mixture_experts',*IDS,'bag_input','embedding_bag','dataset_source','data_loader','initialization','gradient_policy','trainability','batch_norm1d','batch_norm2d','instance_norm1d','instance_norm2d','group_norm','optimizer_sgd','optimizer_adam','optimizer_adamw','optimizer_rmsprop','optimizer_adagrad','scheduler_step_lr','scheduler_cosine_annealing_lr','scheduler_reduce_on_plateau','scheduler_one_cycle_lr','scheduler_linear_lr','rnn','lstm','gru','rnn_cell','lstm_cell','gru_cell','multihead_attention','transformer','transformer_encoder','transformer_decoder','scaled_dot_product_attention','attention_mask','loss_cross_entropy','loss_bce_with_logits','loss_mse','loss_l1','loss_nll','loss_kl_div','loss_ctc','loss_triplet_margin')]
    assert len(old)==46
    old=historical_output_presentation(old)
    assert hashlib.sha256(json.dumps(old,sort_keys=True,separators=(',',':')).encode()).hexdigest()=='59024c486d112131f4004ef9211a54cdfea46768ecfd2682dfba9af0b64daf12'

@pytest.mark.parametrize('ident',IDS)
def test_published_ids_declared_port_order_theme_and_assessed_identity(ident):
    v=source();assert [r['definition']['id']for r in v['components']]==list(IDS)
    r=row(v,ident);d=r['definition']
    assert r['published']is True and r['configurationVersion']==2
    assert d['category']=='Tensor layout'
    assert d['outputs']==['out']
    assert d['inputs']==['in']
    assert r['outputConditions']=={}
    assert r['presentation']['theme']=='violet'and r['presentation']['icon']==ident+'-v2'
    assert r['learning']['lessonIDs']==LESSONS[ident]
    assert r['learning']['exampleID']=='prepared-'+ident+'-classifier'

@pytest.mark.parametrize('ident',IDS)
def test_exact_field_defaults_bounds_and_presets(ident):
    fields={f['key']:f for f in row(source(),ident)['definition']['fields']}
    assert list(fields)==['kernel_size','stride','dilation','padding']+(['output_size']if ident=='fold'else [])
    for key,field in fields.items():
        assert field['kind']=='integerList'
        assert (field['minItems'],field['maxItems'])==(2,2)
        assert (field['min'],field['max'])==((0 if key=='padding'else 1),4096)
        assert field['defaultValue']==({'kernel_size':[2,2],'stride':[1,1],'dilation':[1,1],'padding':[0,0],'output_size':[2,2]}[key])

@pytest.mark.parametrize('ident',IDS)
@pytest.mark.parametrize('mutation',['field','default','schema','rank','dtype','version','batch','adapter','shape','budget','arity','output','publication','condition','icon','theme','lesson','example','source','extra','missingLearning'])
def test_unreviewed_metadata_mutations_refuse(ident,mutation,tmp_path):
    v=source();r=row(v,ident);root=fixture_root(v,tmp_path);bundles.validate(v,root)
    if mutation=='field':r['definition']['fields'].append({'key':'inplace'})
    elif mutation=='default':r['definition']['fields'][0]['defaultValue']=True
    elif mutation=='schema':r['runtimeContract']['graphSchemaVersions']=[1,2]
    elif mutation=='rank':r['runtimeContract']['maximumRank']=9
    elif mutation=='dtype':r['runtimeContract']['graphDTypes']=['int64']
    elif mutation=='version':r['runtimeContract']['version']['minor']=True
    elif mutation=='batch':r['runtimeContract']['preservesBatch']=False
    elif mutation=='adapter':r['adapterID']='arbitrary.executable'
    elif mutation=='shape':r['shapeRuleID']='other.shape'
    elif mutation=='budget':r['runtimeContract']['maximumTensorElements']=100000001
    elif mutation=='arity':r['definition']['inputs'].append('third')
    elif mutation=='output':r['definition']['outputs'].append('other')
    elif mutation=='publication':r['published']=1
    elif mutation=='condition':r['outputConditions']={'out':'custom'}
    elif mutation=='icon':r['presentation']['icon']='reshape-v2'
    elif mutation=='theme':r['presentation']['theme']='blue'
    elif mutation=='lesson':r['learning']['lessonIDs']=['assessment.reshape']
    elif mutation=='example':r['learning']['exampleID']='prepared-reshape-classifier'
    elif mutation=='source':r['sources']['worker']='../escape.py'
    elif mutation=='extra':r['runtimeContract']['inplace']=True
    else:r['learning']=None
    with pytest.raises(ValueError):bundles.validate(v,root)

@pytest.mark.parametrize('ident',IDS)
def test_missing_original_artwork_refuses(ident,tmp_path):
    v=source();root=fixture_root(v,tmp_path)
    (root/'Sources/KwipMLApp/Resources/ComponentIcons'/f'{ident}-v2.png').unlink()
    with pytest.raises(ValueError,match='source reference'):bundles.validate(v,root)

def test_distinct_real_bridge_and_example_references():
    rendered=bundles.render_swift(source(),'spatial-patches')
    for ident in IDS:
        assert 'SpatialPatchCapabilityBridge.'+ident+'Declaration'in rendered
        assert 'WorkflowExamples.prepared'+ident.title()+'Classifier(device: $0)'in rendered


@pytest.mark.parametrize('ident',IDS)
def test_actual_helper_descriptor_matches_independent_literal(ident):
    import sys
    sys.path.insert(0,str(ROOT/'runner'))
    from spatial_patch_configuration import capability
    assert row(source(),ident)['runtimeContract']==capability(ident)

@pytest.mark.parametrize('ident',IDS)
def test_unpublished_spatial_members_are_absent_from_every_callable_table(ident):
    import sys
    sys.path.insert(0,str(ROOT/'runner'))
    v=source();row(v,ident)['published']=False
    table={};exec(bundles.render_spatial_patch_adapters(v),table)
    for key in ('IDS','CAPABILITIES','NORMALIZERS','RESOLVERS'):
        assert set(table[key])==set(IDS)-{ident}

def test_generated_table_uses_real_geometry_and_preserves_paired_defaults():
    import sys
    sys.path.insert(0,str(ROOT/'runner'))
    from tensor_layout import BatchedLayout
    table={};exec(bundles.render_spatial_patch_adapters(source()),table)
    assert table['NORMALIZERS']['unfold']({})=={'kernel_size':[2,2],'stride':[1,1],'dilation':[1,1],'padding':[0,0]}
    unfolded=table['RESOLVERS']['unfold']({},BatchedLayout((2,3,4)))
    assert unfolded['layout'].sample==(8,6)and unfolded['workPerObservation']==48
    folded=table['RESOLVERS']['fold']({'output_size':[3,4],'padding':[1,1]},BatchedLayout((8,20)))
    assert folded['layout'].sample==(2,3,4)and folded['workPerObservation']==160

def test_all_ten_prior_family_swift_renderers_preserve_actual_generated_bytes():
    names={'pooling':'Pooling','activations':'Activation','normalization':'Normalization','layers':'Layer','tensor-layout':'TensorLayout','tensor-math':'TensorMath','embeddings':'Embedding','convolution':'Convolution','tensor-composition':'TensorComposition','tensor-routing':'TensorRouting'}
    for family,name in names.items():
        value=json.loads((ROOT/'components'/family/'bundle.json').read_text())
        assert bundles.render_swift(value,family)==(ROOT/'Sources/KwipMLCore'/f'Bundled{name}Components.generated.swift').read_text(),family

def test_actual48_registry_and_generated_outputs_bind_the_exact_new_bundle():
    import sys
    sys.path.insert(0,str(ROOT/'runner'))
    registry=json.loads((ROOT/'Sources/KwipMLCore/Resources/components.json').read_text())['components'];registry=[r for r in registry if r['id'] not in ('rms_norm','rotary_position','mixture_experts','initialization','gradient_policy','trainability','bag_input','embedding_bag','dataset_source','data_loader')]
    assert len(registry)==83 and len({r['id']for r in registry})==83
    for ident in IDS:assert next(d for d in registry if d['id']==ident)==row(source(),ident)['definition']
    path=ROOT/'runner/spatial_patch_adapters_generated.py'
    assert path.read_text()==bundles.render_spatial_patch_adapters(source())
    module_spec=importlib.util.spec_from_file_location('actual_spatial_adapters',path)
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    assert module.IDS==IDS
    for ident in IDS:assert module.CAPABILITIES[ident]()==row(source(),ident)['runtimeContract']
    assert (ROOT/'Sources/KwipMLCore/BundledSpatialPatchComponents.generated.swift').read_text()==bundles.render_swift(source(),'spatial-patches')
