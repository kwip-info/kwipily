from scripts.tests.registry_history import historical_output_presentation
"""Recurrent bundles bind exact controls, named states and real delivery references."""
import copy,hashlib,importlib.util,json,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('recurrent_bundles',ROOT/'scripts/component_bundles.py');bundles=importlib.util.module_from_spec(spec);spec.loader.exec_module(bundles)
IDS=('rnn','lstm','gru','rnn_cell','lstm_cell','gru_cell')
TASKS=dict(zip(IDS,['rnn-sequence-memory','lstm-separate-memory','gru-reset-memory','rnn-cell-step','lstm-cell-step','gru-cell-step']))
SUFFIXES=dict(zip(IDS,['RNN','LSTM','GRU','RNNCell','LSTMCell','GRUCell']))
DECLARATIONS=dict(zip(IDS,['rnn','lstm','gru','rnnCell','lstmCell','gruCell']))
def source():return json.loads((ROOT/'components/recurrent/bundle.json').read_text())
def row(v,ident):return next(r for r in v['components'] if r['definition']['id']==ident)
def synthetic_references(v,tmp):
    # Empty isolated references exercise validator policy, never actual source/asset acceptance.
    for r in v['components']:
        refs=list(r['sources'].values())+[f"Sources/KwipMLApp/Resources/ComponentIcons/{r['definition']['id']}-v2.png",r['learning']['exampleSource'],'Sources/KwipMLApp/Resources/workflow-observations.csv']+['Sources/KwipMLCore/Recurrent'+n+'.swift' for n in ('LearningTask','LessonReference','LearningSubmission','CapabilityBridge')]
        for ref in refs:p=tmp/ref;p.parent.mkdir(parents=True,exist_ok=True);p.touch()
    return tmp

def test_prior63_ordered_definition_projection_remains_identical():
    registry=json.loads((ROOT/'Sources/KwipMLCore/Resources/components.json').read_text())['components'];registry=[r for r in registry if r['id'] not in ('rms_norm','rotary_position','mixture_experts','initialization','gradient_policy','trainability','bag_input','embedding_bag','dataset_source','data_loader')];old=[r for r in registry if r['id'] not in ('rms_norm','rotary_position','mixture_experts',*IDS,'bag_input','embedding_bag','dataset_source','data_loader','initialization','multihead_attention','transformer','transformer_encoder','transformer_decoder','scaled_dot_product_attention','attention_mask','loss_cross_entropy','loss_bce_with_logits','loss_mse','loss_l1','loss_nll','loss_kl_div','loss_ctc','loss_triplet_margin')]
    assert len(old)==63
    old=historical_output_presentation(old)
    assert hashlib.sha256(json.dumps(old,sort_keys=True,separators=(',',':')).encode()).hexdigest()=='b53cb1ab24414bfeaa9c5d9cc830c332663b53a872a159f13d6f89df5cee652e'

@pytest.mark.parametrize('ident',IDS)
def test_exact_named_ports_conditional_state_inputs_and_family_identity(ident):
    v=source();assert [r['definition']['id'] for r in v['components']]==list(IDS);r=row(v,ident);d=r['definition'];state=['hidden']+(['cell'] if ident.startswith('lstm') else [])
    assert r['published'] is True and r['configurationVersion']==1
    assert d['category']=='Layers' and d['inputs']==['in']+state and d['outputs']==['out']+state
    assert r['inputConditions']=={key:'state_mode=supplied' for key in state} and r['outputConditions']=={}
    assert r['presentation']['theme']=='rose' and r['presentation']['icon']==ident+'-v2'
    assert r['learning']['lessonIDs']==['assessment.'+TASKS[ident]]
    assert r['learning']['exampleID']=='prepared-'+ident.replace('_','-')+'-classifier'

@pytest.mark.parametrize('ident',IDS)
def test_controls_match_frozen_bounds_defaults_and_native_choice_objects(ident):
    fields={f['key']:f for f in row(source(),ident)['definition']['fields']};expected=['input_size','hidden_size','bias','state_mode']+([] if ident.endswith('_cell') else ['num_layers','bidirectional','dropout'])+(['nonlinearity'] if ident in ('rnn','rnn_cell') else [])+(['proj_size'] if ident=='lstm' else [])
    assert list(fields)==expected
    for key,default,lower,upper in [('input_size',2,1,4096),('hidden_size',2,1,512)]+([] if ident.endswith('_cell') else [('num_layers',1,1,8)])+([('proj_size',0,0,511)] if ident=='lstm' else []):
        f=fields[key];assert f['kind']=='integer' and type(f['defaultValue']) is int and (f['defaultValue'],f['min'],f['max'])==(default,lower,upper)
    assert fields['bias']['defaultValue'] is True
    assert fields['state_mode']['defaultValue']=='zero' and [c['value'] for c in fields['state_mode']['choices']]==['zero','supplied']
    if 'nonlinearity' in fields:assert fields['nonlinearity']['defaultValue']=='tanh' and [c['value'] for c in fields['nonlinearity']['choices']]==['tanh','relu']
    if 'dropout' in fields:assert (fields['dropout']['defaultValue'],fields['dropout']['min'],fields['dropout']['max'])==(0,0,1)

@pytest.mark.parametrize('ident',IDS)
@pytest.mark.parametrize('mutation',['field','default','boolInteger','nativeChoices','input','output','condition','outputCondition','publication','configVersion','adapter','shape','runtime','batch','budget','projection','dtype','icon','theme','lesson','example','source','missingLearning','extra'])
def test_unreviewed_contract_mutations_refuse(ident,mutation,tmp_path):
    v=source();r=row(v,ident);bundles.validate(v,synthetic_references(v,tmp_path))
    if mutation=='field':r['definition']['fields'].append({'key':'last_epoch'})
    elif mutation=='default':r['definition']['fields'][0]['defaultValue']=3
    elif mutation=='boolInteger':r['definition']['fields'][0]['defaultValue']=True
    elif mutation=='nativeChoices':r['definition']['fields'][3]['choices']=['zero','supplied']
    elif mutation=='input':r['definition']['inputs'].append('carry')
    elif mutation=='output':r['definition']['outputs'].remove('hidden')
    elif mutation=='condition':r['inputConditions']['hidden']='state_mode=zero'
    elif mutation=='outputCondition':r['outputConditions']['hidden']='state_mode=supplied'
    elif mutation=='publication':r['published']=1
    elif mutation=='configVersion':r['configurationVersion']=2
    elif mutation=='adapter':r['adapterID']='arbitrary.executable'
    elif mutation=='shape':r['shapeRuleID']='other.shape'
    elif mutation=='runtime':r['runtimeContract']['version']['minor']=True
    elif mutation=='batch':r['runtimeContract']['preservesBatch']=False
    elif mutation=='budget':r['runtimeContract']['maximumTensorElements']=100000001
    elif mutation=='projection':r['runtimeContract']['mpsProjectionPolicy']='silentlyFallback'
    elif mutation=='dtype':r['runtimeContract']['graphDTypes']=['int64']
    elif mutation=='icon':r['presentation']['icon']='linear-v2'
    elif mutation=='theme':r['presentation']['theme']='violet'
    elif mutation=='lesson':r['learning']['lessonIDs']=['assessment.reshape']
    elif mutation=='example':r['learning']['exampleID']='prepared-parallel-classifier'
    elif mutation=='source':r['sources']['worker']='../escape.py'
    elif mutation=='missingLearning':r['learning']=None
    else:r['unreviewed']=True
    with pytest.raises(ValueError):bundles.validate(v,tmp_path)

@pytest.mark.parametrize('ident',IDS)
def test_missing_original_icon_refuses(ident,tmp_path):
    v=source();root=synthetic_references(v,tmp_path);(root/f'Sources/KwipMLApp/Resources/ComponentIcons/{ident}-v2.png').unlink()
    with pytest.raises(ValueError,match='source reference'):bundles.validate(v,root)

def test_actual_helper_contract_and_unpublished_callable_tables():
    sys.path.insert(0,str(ROOT/'runner'));from recurrent_configuration import capability
    v=source()
    for ident in IDS:
        assert row(v,ident)['runtimeContract']==capability(ident)
        unpublished=copy.deepcopy(v);row(unpublished,ident)['published']=False;table={};exec(bundles.render_recurrent_adapters(unpublished),table)
        for key in ('IDS','CAPABILITIES','NORMALIZERS','RESOLVERS','CONFIGURED_INPUTS'):assert set(table[key])==set(IDS)-{ident}

def test_bridge_examples_and_new_entry_conditions_use_real_generated_declarations():
    rendered=bundles.render_swift(source(),'recurrent')
    assert 'public let inputConditions: [String: String]' in rendered
    for ident in IDS:
        assert 'RecurrentCapabilityBridge.'+DECLARATIONS[ident]+'Declaration' in rendered
        assert 'WorkflowExamples.prepared'+SUFFIXES[ident]+'Classifier(device: $0)' in rendered

def test_actual_resources_and_generated_drift_are_required():
    bundles.validate(source(),ROOT)
    expected=bundles.render(ROOT)
    for name,value in expected.items():assert (ROOT/name).read_text()==value,name
    registry=json.loads(expected['Sources/KwipMLCore/Resources/components.json'])['components'];registry=[r for r in registry if r['id'] not in ('rms_norm','rotary_position','mixture_experts','initialization','gradient_policy','trainability','bag_input','embedding_bag','dataset_source','data_loader')];assert len(registry)==83 and len({r['id'] for r in registry})==83
    assert [r['id'] for r in registry if r['id'] not in ('rms_norm','rotary_position','mixture_experts','multihead_attention','transformer','transformer_encoder','transformer_decoder','scaled_dot_product_attention','attention_mask','loss_cross_entropy','loss_bce_with_logits','loss_mse','loss_l1','loss_nll','loss_kl_div','loss_ctc','loss_triplet_margin')][-6:]==list(IDS)

def test_every_prior_family_swift_projection_is_byte_identical():
    families={'pooling':'Pooling','activations':'Activation','normalization':'Normalization','layers':'Layer','tensor-layout':'TensorLayout','tensor-math':'TensorMath','embeddings':'Embedding','convolution':'Convolution','tensor-composition':'TensorComposition','tensor-routing':'TensorRouting','spatial-patches':'SpatialPatch','optimizers':'Optimizer','schedulers':'Scheduler'}
    for family,name in families.items():
        v=json.loads((ROOT/'components'/family/'bundle.json').read_text())
        assert bundles.render_swift(v,family)==(ROOT/'Sources/KwipMLCore'/f'Bundled{name}Components.generated.swift').read_text(),family

@pytest.mark.parametrize('ident',IDS)
def test_real_callable_normalization_state_order_and_named_geometry(ident):
    sys.path.insert(0,str(ROOT/'runner'));from tensor_layout import BatchedLayout
    table={};exec(bundles.render_recurrent_adapters(source()),table)
    assert table['CONFIGURED_INPUTS'][ident]({})==('in',)
    c=table['NORMALIZERS'][ident]({'state_mode':'supplied'})
    expected=('in','hidden','cell') if ident.startswith('lstm') else ('in','hidden')
    assert table['CONFIGURED_INPUTS'][ident](c)==expected
    sample=(2,) if ident.endswith('_cell') else (3,2)
    state=(2,) if ident.endswith('_cell') else (1,2)
    resolved=table['RESOLVERS'][ident](c,[BatchedLayout(sample)]+[BatchedLayout(state) for _ in expected[1:]])
    assert list(resolved['outputs'])==['out','hidden']+(['cell'] if ident.startswith('lstm') else [])
    assert resolved['outputs']['hidden'].sample==state
    assert resolved['outputs']['out'].sample==((2,) if ident.endswith('_cell') else (3,2))
    assert resolved['parameterCount']>0
