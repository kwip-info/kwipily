from scripts.tests.registry_history import historical_output_presentation
"""Attention authoring keeps named weights, binary masks and real delivery references."""
import copy, hashlib, importlib.util, json, sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('attention_bundles', ROOT/'scripts/component_bundles.py')
bundles = importlib.util.module_from_spec(spec); spec.loader.exec_module(bundles)
IDS = ('multihead_attention','transformer','transformer_encoder','transformer_decoder','scaled_dot_product_attention','attention_mask')
TASKS = dict(zip(IDS[:5], ('multihead-attention-connections','transformer-source-target','transformer-encoder-context','transformer-decoder-memory','scaled-attention-similarity')))
def source(): return json.loads((ROOT/'components/attention/bundle.json').read_text())
def row(value, ident): return next(r for r in value['components'] if r['definition']['id'] == ident)
def test_prior69_ordered_definition_projection_remains_identical():
    registry = json.loads((ROOT/'Sources/KwipMLCore/Resources/components.json').read_text())['components'];registry=[r for r in registry if r['id'] not in ('initialization','gradient_policy','trainability','bag_input','embedding_bag','dataset_source','data_loader')]
    old = [r for r in registry if r['id'] not in (*IDS,'mixture_experts','rotary_position','rms_norm','bag_input','embedding_bag','dataset_source','data_loader','initialization','loss_cross_entropy','loss_bce_with_logits','loss_mse','loss_l1','loss_nll','loss_kl_div','loss_ctc','loss_triplet_margin')]
    assert len(old) == 69
    old=historical_output_presentation(old)
    assert hashlib.sha256(json.dumps(old,sort_keys=True,separators=(',',':')).encode()).hexdigest() == '26406bd5723d42487d715aca04beaa97c6979fde3cd4c5bd05a87bb08fafd391'
def test_generated_attention_table_has_reviewed_callable_emitter():
    assert callable(getattr(bundles, 'render_attention_adapters', None)), 'Missing reviewed attention callable adapter table emitter'

FLOATS = {'multihead_attention':['query','key','value'],'transformer':['source','target'],'transformer_encoder':['in'],'transformer_decoder':['target','memory'],'scaled_dot_product_attention':['query','key','value'],'attention_mask':['query','key']}
MASKS = {'multihead_attention':[('attention_mask_mode','attn_mask'),('padding_mask_mode','key_padding_mask')], 'transformer':[('source_mask_mode','src_mask'),('target_mask_mode','tgt_mask'),('memory_mask_mode','memory_mask'),('source_padding_mode','src_key_padding_mask'),('target_padding_mode','tgt_key_padding_mask'),('memory_padding_mode','memory_key_padding_mask')], 'transformer_encoder':[('mask_mode','mask'),('padding_mask_mode','src_key_padding_mask')], 'transformer_decoder':[('target_mask_mode','tgt_mask'),('memory_mask_mode','memory_mask'),('target_padding_mode','tgt_key_padding_mask'),('memory_padding_mode','memory_key_padding_mask')], 'scaled_dot_product_attention':[('mask_mode','mask')], 'attention_mask':[]}

@pytest.mark.parametrize('ident', IDS)
def test_native_port_order_binary_masks_and_actual_named_weights(ident):
    value=source(); assert [r['definition']['id'] for r in value['components']] == [*IDS,'rotary_position']
    entry=row(value,ident); definition=entry['definition']
    assert definition['category']=='Layers' and entry['presentation']['theme']=='rose'
    assert definition['inputs']==FLOATS[ident]+[port for _,port in MASKS[ident]]
    assert definition['outputs']==(['mask','padding'] if ident=='attention_mask' else ['out','weights'] if ident=='multihead_attention' else ['out'])
    assert entry['inputConditions']=={port:key+'=supplied' for key,port in MASKS[ident]}
    assert entry['outputConditions']==({'weights':'return_weights=true'} if ident=='multihead_attention' else {})
    assert entry['runtimeContract']['maskConvention']=='oneBlockedZeroAllowed'
    assert entry['runtimeContract']['statePersistence']=='none'
    assert entry['presentation']['icon']==ident+'-v2'

@pytest.mark.parametrize('ident', IDS)
def test_defaults_nullable_options_and_native_choice_objects_match_declared_recipe(ident):
    sys.path.insert(0,str(ROOT));from runner.attention_configuration import normalize_attention
    fields=row(source(),ident)['definition']['fields'];defaults={f['key']:f['defaultValue'] for f in fields}
    assert defaults==normalize_attention(ident,{})
    for field in fields:
        if field['kind']=='choice': assert all(isinstance(c,dict) and set(c)=={'value','title'} for c in field['choices'])
    bykey={f['key']:f for f in fields}
    if ident=='multihead_attention':
        for key in ('kdim','vdim'):
            f=bykey[key];assert (f['kind'],f['defaultValue'],f['min'],f['max'],f['allowsNull'],f['enabledValue'])==('integer',None,1,4096,True,4)
        assert 'kdim_mode' not in bykey and 'vdim_mode' not in bykey
        assert defaults['return_weights'] is True and defaults['average_weights'] is True
    if ident=='scaled_dot_product_attention':
        assert bykey['scale']['defaultValue'] is None and bykey['scale']['allowsNull'] is True and bykey['scale']['enabledValue']==1
        assert bykey['scale']['min']==1.401298464324817e-45 and 'scale_mode' not in bykey
    if 'layer_norm_eps' in bykey:assert (bykey['layer_norm_eps']['defaultValue'],bykey['layer_norm_eps']['min'])==(1e-5,1.401298464324817e-45)
    if ident=='attention_mask':
        for key in ('blocked_queries','blocked_keys','padding_keys'):
            f=bykey[key];assert (f['defaultValue'],f['min'],f['max'],f['minItems'],f['maxItems'])==([],0,4095,0,4096)

@pytest.mark.parametrize('ident',IDS)
def test_five_assessed_lessons_and_genuine_mask_alias_have_distinct_examples(ident):
    entry=row(source(),ident);task=TASKS.get(ident,TASKS['multihead_attention'])
    assert entry['learning']['lessonIDs']==['assessment.'+task]
    assert entry['learning']['exampleID']=='prepared-'+ident.replace('_','-')+'-classifier'
    assert entry['learning']['exampleSource']=='Sources/KwipMLCore/AttentionWorkflowExample.swift'
    assert len(set(TASKS.values()))==5 and len({r['learning']['exampleID']for r in source()['components']})==7
    assert entry['learning']['paragraphs'] and all(isinstance(p,str)and p.strip()for p in entry['learning']['paragraphs'])

def test_real_resource_references_must_exist_before_canonical_generation():
    for entry in source()['components']:
        refs=list(entry['sources'].values())+[entry['learning']['exampleSource'],'Sources/KwipMLCore/AttentionLearningTask.swift','Sources/KwipMLCore/AttentionLessonReference.swift','Sources/KwipMLCore/AttentionLearningSubmission.swift','Sources/KwipMLCore/AttentionCapabilityBridge.swift','Sources/KwipMLApp/Resources/ComponentIcons/'+entry['definition']['id']+'-v2.png']
        for reference in refs: assert (ROOT/reference).is_file(), 'Missing real delivery resource '+reference

def test_reviewed_family_renderer_emits_actual_entry_ids_and_condition_maps():
    generated=bundles.render_swift(source(),'attention')
    assert 'public enum BundledAttentionComponents' in generated
    assert 'public let inputConditions: [String: String]' in generated
    for ident in IDS:assert '"id": "'+ident+'"' in generated

def synthetic_references(value,root):
    # Isolated empty files test metadata validation policy only. They never prove
    # genuine artwork, executable sources, lesson delivery or canonical resources.
    for entry in value['components']:
        refs=list(entry['sources'].values())+[entry['learning']['exampleSource'],'Sources/KwipMLApp/Resources/'+entry['learning']['sourceResource']+'.csv','Sources/KwipMLCore/AttentionLearningTask.swift','Sources/KwipMLCore/AttentionLessonReference.swift','Sources/KwipMLCore/AttentionLearningSubmission.swift','Sources/KwipMLCore/AttentionCapabilityBridge.swift','Sources/KwipMLApp/Resources/ComponentIcons/'+entry['definition']['id']+'-v2.png']
        for reference in refs:
            p=root/reference;p.parent.mkdir(parents=True,exist_ok=True);p.touch()
    return root

@pytest.mark.parametrize('ident',IDS)
@pytest.mark.parametrize('mutation',['field','default','boolInteger','choiceObject','input','output','condition','outputCondition','publication','configVersion','adapter','shape','runtime','dtype','batch','budget','maskConvention','implicitCache','sdpaWeights','icon','theme','lesson','example','source','learning','extra'])
def test_unreviewed_attention_and_mask_contract_mutations_refuse(ident,mutation,tmp_path):
    value=source();entry=row(value,ident);root=synthetic_references(value,tmp_path)
    bundles.validate(value,root)
    if mutation=='field':entry['definition']['fields'].append({'key':'enable_gqa'})
    elif mutation=='default':entry['definition']['fields'][0]['defaultValue']=3
    elif mutation=='boolInteger':entry['definition']['fields'][0]['defaultValue']=True
    elif mutation=='choiceObject':
        f=next(f for f in entry['definition']['fields']if f['kind']=='choice');f['choices']=[c['value']for c in f['choices']]
    elif mutation=='input':entry['definition']['inputs'].append('unreviewed_cache')
    elif mutation=='output':entry['definition']['outputs'].append('unreviewed_weights')
    elif mutation=='condition':entry['inputConditions']['unreviewed_mask']='mask_mode=none'
    elif mutation=='outputCondition':entry['outputConditions']['weights']='return_weights=false'
    elif mutation=='publication':entry['published']=1
    elif mutation=='configVersion':entry['configurationVersion']=2
    elif mutation=='adapter':entry['adapterID']='arbitrary.executable'
    elif mutation=='shape':entry['shapeRuleID']='other.shape'
    elif mutation=='runtime':entry['runtimeContract']['version']['minor']=True
    elif mutation=='dtype':entry['runtimeContract']['maskDType']='float32'
    elif mutation=='batch':entry['runtimeContract']['preservesBatch']=False
    elif mutation=='budget':entry['runtimeContract']['maximumScoreElements']=100000001
    elif mutation=='maskConvention':entry['runtimeContract']['maskConvention']='oneAllowedZeroBlocked'
    elif mutation=='implicitCache':entry['runtimeContract']['statePersistence']='automaticKVCache'
    elif mutation=='sdpaWeights':entry['runtimeContract']['multiheadWeights']='fabricatedAllOperations'
    elif mutation=='icon':entry['presentation']['icon']='linear-v2'
    elif mutation=='theme':entry['presentation']['theme']='cyan'
    elif mutation=='lesson':entry['learning']['lessonIDs']=['assessment.rnn-sequence-memory']
    elif mutation=='example':entry['learning']['exampleID']='prepared-parallel-classifier'
    elif mutation=='source':entry['sources']['worker']='../escape.py'
    elif mutation=='learning':entry['learning']=None
    else:entry['unreviewed']=True
    with pytest.raises(ValueError):bundles.validate(value,root)

@pytest.mark.parametrize('ident',IDS)
def test_runtime_descriptors_are_literal_aligned_and_generated_table_is_allocation_free(ident):
    sys.path.insert(0,str(ROOT));from runner.attention_configuration import capability
    assert row(source(),ident)['runtimeContract']==capability(ident)
    text=bundles.render_attention_adapters(source())
    assert 'from attention_configuration import capability, normalize_attention, resolve_attention, configured_input_ports' in text
    for name in ('CAPABILITIES','NORMALIZERS','RESOLVERS','CONFIGURED_INPUTS'):assert name+' ='in text
    assert 'import torch'not in text and 'eval('not in text
