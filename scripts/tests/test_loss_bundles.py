"""Objective recipes remain unpublished controls until complete evidence is attached."""
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('loss_bundle_authoring',ROOT/'scripts/component_bundles.py')
bundles=importlib.util.module_from_spec(spec);spec.loader.exec_module(bundles)
sys.path.insert(0,str(ROOT/'runner'))
from loss_configuration import capability,canonical_configuration
IDS=('loss_cross_entropy','loss_bce_with_logits','loss_mse','loss_l1','loss_nll','loss_kl_div','loss_ctc','loss_triplet_margin')
NAMES=('CrossEntropyLoss','BCEWithLogitsLoss','MSELoss','L1Loss','NLLLoss','KLDivLoss','CTCLoss','TripletMarginLoss')
def source():return json.loads((ROOT/'components/losses/bundle.json').read_text())
def unpublished_fixture():
    v=source()
    for r in v['components']:r['published']=False;r['learning']=None
    return v

def test_real_metadata_validates_without_invented_references():
    v=source();bundles.validate(v,ROOT)
    assert [r['definition']['id']for r in v['components']]==list(IDS)
    for r,name in zip(v['components'],NAMES):
        assert r['definition']['title']==name
        assert r['definition']['inputs']==[]and r['definition']['outputs']==['out']
        assert type(r['published'])is bool
        if r['published']:assert r['learning']is not None
        assert r['runtimeContract']==capability(r['definition']['id'])
        assert r['runtimeContract']['fixedWeights']=='buffers'
        assert r['runtimeContract']['version']=='1.0.0'

@pytest.mark.parametrize('mutation',['ports','field','default','boolNumber','count','runtime','adapter','source','unknown','publication','order','icon','bounds','choice'])
def test_unreviewed_recipe_mutations_refuse(mutation):
    v=unpublished_fixture();r=v['components'][0]
    if mutation=='ports':r['definition']['inputs']=['in']
    elif mutation=='field':r['definition']['fields'].append({'key':'x'})
    elif mutation=='default':r['definition']['fields'][2]['defaultValue']=0
    elif mutation=='boolNumber':r['definition']['fields'][3]['defaultValue']=[True]
    elif mutation=='count':r['definition']['fields'][3]['maxItems']=4097
    elif mutation=='runtime':r['runtimeContract']['fixedWeights']='parameters'
    elif mutation=='adapter':r['adapterID']='kwip.runner.graph-v2.linear'
    elif mutation=='source':r['sources']['native']='../outside.swift'
    elif mutation=='unknown':r['runtimeContract']['SUPPORTED']=True
    elif mutation=='publication':r['published']=True
    elif mutation=='order':v['components'].reverse()
    elif mutation=='icon':r['presentation']['icon']='linear-v2'
    elif mutation=='bounds':r['definition']['fields'][3]['min']=-1
    else:r['definition']['fields'][0]['choices'][0]['value']='batchmean'
    with pytest.raises(ValueError):bundles.validate(v,ROOT)

def test_generated_swift_has_eight_metadata_entries_no_unpublished_callables():
    rendered=bundles.render_swift(unpublished_fixture(),'losses')
    assert 'BundledLossComponents'in rendered
    assert 'components/losses/bundle.json'in rendered
    assert 'public static func metadata'in rendered
    for ident in IDS:assert ident in rendered
    assert 'WorkflowExamples.prepared'not in rendered
    assert 'LossCapabilityBridge.'not in rendered
    assert ' = [\n\n    ]'not in rendered
    assert 'static let declarations:'in rendered and ' = [:]'in rendered

def test_generated_python_is_recipe_only_and_publication_gated():
    table={};exec(bundles.render_loss_adapters(unpublished_fixture()),table)
    for key in ('IDS','CAPABILITIES','NORMALIZERS'):assert not table[key]
    assert not {'SUPPORTED','RESOLVERS','EXECUTORS'}&set(table)
    # Renderer policy only; this deliberately does not publish or validate an incomplete bundle.
    v=unpublished_fixture();v['components'][0]['published']=True
    table={};exec(bundles.render_loss_adapters(v),table)
    assert table['IDS']==('loss_cross_entropy',)
    assert table['CAPABILITIES']['loss_cross_entropy']==capability('loss_cross_entropy')
    assert table['NORMALIZERS']['loss_cross_entropy']({})==canonical_configuration('loss_cross_entropy',{})

def test_all_prior_generated_outputs_preserved_and_new_unpublished_registry_unchanged():
    outputs=bundles.render(ROOT)
    assert 'Sources/KwipMLCore/BundledLossComponents.generated.swift'in outputs
    assert 'runner/loss_adapters_generated.py'in outputs
    for path,text in outputs.items():
        if path in ('Sources/KwipMLCore/BundledLossComponents.generated.swift','runner/loss_adapters_generated.py'):continue
        assert (ROOT/path).read_text()==text,path

@pytest.mark.parametrize('ident,settings',[
    ('loss_cross_entropy',{'reduction':'none','class_axis':1,'class_weights':[.25,2.0],'label_smoothing':.2}),
    ('loss_bce_with_logits',{'weight_mode':'configured','weight_shape':[2],'weight_values':[.25,3],'pos_weight_mode':'mapped'}),
    ('loss_mse',{'reduction':'sum','target_broadcast':True}),
    ('loss_l1',{'reduction':'none','target_broadcast':False}),
    ('loss_nll',{'class_axis':1,'class_weights':[0,2],'ignore_index':-7}),
    ('loss_kl_div',{'reduction':'batchmean','log_target':True}),
    ('loss_ctc',{'blank':2,'zero_infinity':True,'target_layout':'packed'}),
    ('loss_triplet_margin',{'margin':.4,'p':1.5,'eps':0,'swap':True})])
def test_recipe_normalizers_preserve_independently_chosen_nondefault_controls(ident,settings):
    v=unpublished_fixture()
    # Isolate renderer behavior; never present this modified fixture as published product metadata.
    next(r for r in v['components']if r['definition']['id']==ident)['published']=True
    table={};exec(bundles.render_loss_adapters(v),table)
    actual=table['NORMALIZERS'][ident](settings)
    for key,value in settings.items():assert actual[key]==value
    assert table['CAPABILITIES'][ident]['targetRoles']==(['target','input_lengths','target_lengths']if ident=='loss_ctc'else []if ident=='loss_triplet_margin'else ['target','weight','pos_weight']if ident=='loss_bce_with_logits'else ['target'])

@pytest.mark.parametrize('index',range(8))
def test_missing_source_or_learning_cannot_publish_any_objective(index):
    v=unpublished_fixture();v['components'][index]['published']=True
    with pytest.raises(ValueError,match='requires learning'):bundles.validate(v,ROOT)
    v=source();v['components'][index]['sources']['tests']='missing/never-authored.py'
    with pytest.raises(ValueError,match='source reference'):bundles.validate(v,ROOT)
