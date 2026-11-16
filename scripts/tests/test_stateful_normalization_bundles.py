from scripts.tests.registry_history import historical_output_presentation
"""Reviewed normalization controls/state are authored, not implicit constructor defaults."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('normalization_bundles',ROOT/'scripts/component_bundles.py')
bundles=importlib.util.module_from_spec(spec);spec.loader.exec_module(bundles)
IDS=('batch_norm1d','batch_norm2d','instance_norm1d','instance_norm2d','group_norm')
def source():return json.loads((ROOT/'components/normalization/bundle.json').read_text())
def member(ident):return next((r for r in source()['components']if r['definition']['id']==ident),None)
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def test_existing_layernorm_record_remains_byte_semantically_identical():
    assert digest(source()['components'][0])=='495689441189b96c4297bd53b2c3b11354a7c4fe5a244f3ede64c0815447f92f'
def test_prior48_registry_projection_remains_identical():
    old=[r for r in json.loads((ROOT/'Sources/KwipMLCore/Resources/components.json').read_text())['components']if r['id']not in ('rms_norm','rotary_position','mixture_experts',*IDS,'rms_norm','bag_input','embedding_bag','dataset_source','data_loader','initialization','gradient_policy','trainability','optimizer_sgd','optimizer_adam','optimizer_adamw','optimizer_rmsprop','optimizer_adagrad','scheduler_step_lr','scheduler_cosine_annealing_lr','scheduler_reduce_on_plateau','scheduler_one_cycle_lr','scheduler_linear_lr','rnn','lstm','gru','rnn_cell','lstm_cell','gru_cell','multihead_attention','transformer','transformer_encoder','transformer_decoder','scaled_dot_product_attention','attention_mask','loss_cross_entropy','loss_bce_with_logits','loss_mse','loss_l1','loss_nll','loss_kl_div','loss_ctc','loss_triplet_margin')]
    assert len(old)==48
    old=historical_output_presentation(old)
    assert digest(old)=='c6b220127538b26068641e8b7358c550f8d3734b6e45ec35775b4e8f90ec8ab2'
@pytest.mark.parametrize('ident',IDS)
def test_five_published_bundles_declare_no_hidden_channels_or_state_inputs(ident):
    r=member(ident);assert r is not None,'Missing reviewed authored normalization unit'
    assert r['published']is True and r['configurationVersion']==2
    assert r['definition']['inputs']==['in']and r['definition']['outputs']==['out']
    assert r['outputConditions']=={}
    assert r['presentation']['theme']=='mauve'and r['presentation']['icon']==ident+'-v2'
    assert len(r['learning']['lessonIDs'])==2
    assert r['learning']['exampleID']=='prepared-'+ident.replace('_','-')+'-classifier'
@pytest.mark.parametrize('ident',IDS)
def test_exact_explicit_channels_bias_tracking_and_averaging_controls(ident):
    r=member(ident);assert r is not None,'Missing configurable normalization'
    f={x['key']:x for x in r['definition']['fields']}
    defaults={'eps':1e-5,'affine':not ident.startswith('instance'),'bias':True}
    if ident=='group_norm':defaults.update(num_channels=2,num_groups=1)
    else:defaults.update(num_features=2,momentum=.1,track_running_stats=ident.startswith('batch'))
    if ident.startswith('batch'):defaults['averaging']='moving'
    assert {k:v['defaultValue']for k,v in f.items()}==defaults
    for k in ['num_features','num_channels','num_groups']:
        if k in f:assert f[k]['kind']=='integer'and(f[k]['min'],f[k]['max'])==(1,512)
    assert f['eps']['kind']=='number'and(f['eps']['min'],f['eps']['max'])==(1.1754943508222875e-38,3.4028234663852886e38)
    if 'momentum'in f:assert f['momentum']['kind']=='number'and(f['momentum']['min'],f['momentum']['max'])==(0,1)and not f['momentum'].get('allowsNull',False)
    if 'averaging'in f:assert [x['value']for x in f['averaging']['choices']]==['moving','cumulative']

def synthetic_references(value,tmp_path):
    # Empty references are restricted to isolated metadata-policy tests, not delivery.
    for r in value['components']:
        refs=list(r['sources'].values())+['Sources/KwipMLApp/Resources/ComponentIcons/'+r['presentation']['icon']+'.png']
        if r['learning']:refs += [r['learning']['exampleSource'],'Sources/KwipMLApp/Resources/'+r['learning']['sourceResource']+'.csv']
        if r['definition']['id']in IDS:refs += ['Sources/KwipMLCore/StatefulNormalization'+n+'.swift'for n in ['CapabilityBridge','LearningTask','LessonReference','LearningSubmission']]
        for ref in refs:
            p=tmp_path/ref;p.parent.mkdir(parents=True,exist_ok=True);p.touch()
    return tmp_path
@pytest.mark.parametrize('ident',IDS)
@pytest.mark.parametrize('mutation',['field','default','nullableMomentum','publication','version','dtype','rank','state','adapter','shape','port','theme','icon','lesson','example','source'])
def test_independent_control_state_learning_and_asset_mutations_refuse(ident,mutation,tmp_path):
    value=source();r=next((r for r in value['components']if r['definition']['id']==ident),None)
    assert r is not None,'New metadata not authored'
    root=synthetic_references(value,tmp_path);bundles.validate(value,root)
    if mutation=='field':r['definition']['fields'].append({'key':'running_mean'})
    elif mutation=='default':r['definition']['fields'][0]['defaultValue']=True
    elif mutation=='nullableMomentum':next(f for f in r['definition']['fields']if f['key']==('eps'if ident=='group_norm'else'momentum'))['allowsNull']=True
    elif mutation=='publication':r['published']=1
    elif mutation=='version':r['configurationVersion']=True
    elif mutation=='dtype':r['runtimeContract']['graphDTypes']=['float64']
    elif mutation=='rank':r['runtimeContract']['inputSampleRanks']=[8]
    elif mutation=='state':r['runtimeContract']['counterBehavior']='trainable'
    elif mutation=='adapter':r['adapterID']='other.executable'
    elif mutation=='shape':r['shapeRuleID']='generic.unchanged'
    elif mutation=='port':r['definition']['inputs'].append('statistics')
    elif mutation=='theme':r['presentation']['theme']='blue'
    elif mutation=='icon':r['presentation']['icon']='layer_norm-v2'
    elif mutation=='lesson':r['learning']['lessonIDs']=['assessment.layer-norm-rows']
    elif mutation=='example':r['learning']['exampleID']='prepared-layer-norm-classifier'
    else:r['sources']['worker']='../escape.py'
    with pytest.raises(ValueError):bundles.validate(value,root)
@pytest.mark.parametrize('ident',IDS)
def test_missing_original_icon_refuses(ident,tmp_path):
    value=source();assert member(ident)is not None
    root=synthetic_references(value,tmp_path)
    (root/('Sources/KwipMLApp/Resources/ComponentIcons/'+ident+'-v2.png')).unlink()
    with pytest.raises(ValueError):bundles.validate(value,root)

def test_actual_helper_descriptors_match_reviewed_literals_without_torch_dependency():
    import sys
    sys.path.insert(0,str(ROOT/'runner'))
    from stateful_normalization_configuration import capability
    for ident in IDS:
        assert member(ident)is not None
        assert member(ident)['runtimeContract']==capability(ident)==bundles.STATEFUL_NORMALIZATION_CONTRACTS[ident]

def test_real_references_and_canonical_generated_outputs_have_no_placeholder_scaffolds():
    value=source();assert all(member(i)is not None for i in IDS)
    bundles.validate(value,ROOT)
    generated=bundles.render(ROOT)
    for path,contents in generated.items():assert (ROOT/path).read_text()==contents,path
    swift=generated['Sources/KwipMLCore/BundledNormalizationComponents.generated.swift']
    assert 'StatefulNormalizationCapabilityBridge.batchNorm1dDeclaration'in swift
    assert 'preparedInstanceNorm2dClassifier'in swift
    assert 'StatefulNormalizationCapabilityBridge.layerNormDeclaration'not in swift
    namespace={};exec(generated['runner/stateful_normalization_adapters_generated.py'],namespace)
    assert namespace['IDS']==IDS
    assert all(callable(namespace[name][i])for name in ('CAPABILITIES','NORMALIZERS','RESOLVERS')for i in IDS)
