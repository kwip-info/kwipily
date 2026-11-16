from scripts.tests.registry_history import historical_output_presentation
"""Exact new activation metadata, ownership and prior-row preservation."""
import copy,hashlib,importlib.util,json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('remaining_bundles',ROOT/'scripts/component_bundles.py');bundles=importlib.util.module_from_spec(spec);spec.loader.exec_module(bundles)
IDS=('prelu','log_softmax','dropout')
LESSONS={'prelu':['assessment.prelu-shared-slope','assessment.prelu-channel-slopes'],'log_softmax':['assessment.logsoftmax-sample-scores'],'dropout':['assessment.dropout-training-pattern','assessment.dropout-evaluation-pattern']}
OLD={'sigmoid':'3cedfef712e2a193051f5b2475be98da55b6a16f4048b822c842879fa8be33b3','tanh':'29dbd8eb35a0d87c42cacc91a302b79efdc36e476f5066d5f3ec79faa4fc9de7','relu':'cfb2d7c47b0b68f35961cd72b5e140574eb74bd7b34950af4d4339a4996b2f3a','gelu':'3a4cd3b0f7aff9dfd7ab45f1b0b61681506dbe11f4e6d8bad0c5f10c2a8d833c','softmax':'c25797f776827f0e9a98de14236716d6f7c329d8bacd841a538221f9a599f08a','silu':'809703fb932947660addd2691a7092b321c4d58859c039ee0ba5f0b54e82c38e','leaky_relu':'61aecdd99829e71d8108f3b957c9d19652b66308f6c93599d1eea1f045448567','elu':'0a105b7725f03c45188d9bf1070a1ee27e1371683c98ed24344f2f92f2a75b79'}
def source():return json.loads((ROOT/'components/activations/bundle.json').read_text())
def row(value,ident):
    matches=[r for r in value['components'] if r['definition']['id']==ident]
    assert len(matches)==1,'Exactly one authored '+ident+' row required'
    return matches[0]
def fixture_root(value,tmp_path):
    # Explicit isolated validation fixtures are not evidence of actual source or generated artwork.
    for r in value['components']:
        refs=list(r['sources'].values())+['Sources/KwipMLApp/Resources/ComponentIcons/'+r['presentation']['icon']+'.png']
        if r['learning']:refs += [r['learning']['exampleSource'],'Sources/KwipMLApp/Resources/'+r['learning']['sourceResource']+'.csv']
        for ref in refs:p=tmp_path/ref;p.parent.mkdir(parents=True,exist_ok=True);p.touch()
    return tmp_path

def test_all_eight_existing_rows_preserve_exact_semantics_and_descriptor_bytes():
    v=source()
    for ident,digest in OLD.items():assert hashlib.sha256(json.dumps(row(v,ident),sort_keys=True,separators=(',',':')).encode()).hexdigest()==digest

def test_three_distinct_published_members_have_five_assessed_lessons():
    v=source();assert [r['definition']['id'] for r in v['components']]==list(OLD)+list(IDS)
    for ident in IDS:
        r=row(v,ident)
        assert r['published'] is True and r['configurationVersion']==2
        assert r['definition']['inputs']==['in'] and r['definition']['outputs']==['out']
        assert r['presentation']['theme']=='amber' and r['presentation']['icon']==ident+'-v2'
        assert r['learning']['lessonIDs']==LESSONS[ident]
        assert r['learning']['exampleID']=='prepared-'+ident.replace('_','-')+'-classifier'

@pytest.mark.parametrize('ident',IDS)
def test_exact_configurable_control_defaults_and_boundaries(ident):
    fields={f['key']:f for f in row(source(),ident)['definition']['fields']}
    if ident=='prelu':
        assert set(fields)=={'num_parameters','init'}
        assert fields['num_parameters']['defaultValue']==1 and fields['num_parameters']['min']==1 and fields['num_parameters']['max']==4096
        assert fields['init']['defaultValue']==.25 and fields['init']['min']==-3.4028234663852886e38 and fields['init']['max']==3.4028234663852886e38
    elif ident=='log_softmax':assert fields['dim']['defaultValue']==-1 and (fields['dim']['min'],fields['dim']['max'])==(-8,7)
    else:assert fields['p']['defaultValue']==.5 and (fields['p']['min'],fields['p']['max'])==(0,1)

@pytest.mark.parametrize('ident',IDS)
@pytest.mark.parametrize('mutation',['field','default','schema','rank','dtype','version','batch','adapter','shapeRule','budget','modeContract','arity','output','publication','condition','icon','theme','lesson','example','source','extra','missingLearning'])
def test_unreviewed_metadata_and_runtime_contract_mutations_refuse(ident,mutation,tmp_path):
    v=source();r=row(v,ident);root=fixture_root(v,tmp_path);bundles.validate(v,root)
    if mutation=='field':r['definition']['fields'].append({'key':'inplace'})
    elif mutation=='default':r['definition']['fields'][0]['defaultValue']=True
    elif mutation=='schema':r['runtimeContract']['graphSchemaVersions']=[1,2]
    elif mutation=='rank':r['runtimeContract']['maximumRank']=9
    elif mutation=='dtype':r['runtimeContract']['graphDTypes']=['int64']
    elif mutation=='version':r['runtimeContract']['version']['minor']=True
    elif mutation=='batch':r['runtimeContract']['preservesBatch']=False
    elif mutation=='adapter':r['adapterID']='arbitrary.executable'
    elif mutation=='shapeRule':r['shapeRuleID']='other.shape'
    elif mutation=='budget':r['runtimeContract']['maximumTensorElements']=100000001
    elif mutation=='modeContract':r['runtimeContract'][{'prelu':'channelAxis','log_softmax':'outputKind','dropout':'modeSource'}[ident]]='unreviewed'
    elif mutation=='arity':r['definition']['inputs'].append('second')
    elif mutation=='output':r['definition']['outputs'].append('other')
    elif mutation=='publication':r['published']=1
    elif mutation=='condition':r['outputConditions']={'out':'p'}
    elif mutation=='icon':r['presentation']['icon']='relu-v2'
    elif mutation=='theme':r['presentation']['theme']='blue'
    elif mutation=='lesson':r['learning']['lessonIDs']=['assessment.relu-values']
    elif mutation=='example':r['learning']['exampleID']='prepared-relu-classifier'
    elif mutation=='source':r['sources']['worker']='../escape.py'
    elif mutation=='extra':r['runtimeContract']['inplace']=True
    else:r['learning']=None
    with pytest.raises(ValueError):bundles.validate(v,root)

def test_each_member_uses_distinct_actual_bridge_and_example_factory():
    v=source()
    for ident in IDS:row(v,ident)
    rendered=bundles.render_swift(v,'activations')
    for bridge,factory in [('prelu','PReLU'),('logSoftmax','LogSoftmax'),('dropout','Dropout')]:
        assert 'RemainingActivationCapabilityBridge.'+bridge+'Declaration' in rendered
        assert 'WorkflowExamples.prepared'+factory+'Classifier(device: $0)' in rendered

@pytest.mark.parametrize('ident',IDS)
def test_runtime_helper_matches_independent_literal_descriptor(ident):
    import sys
    sys.path.insert(0,str(ROOT/'runner'))
    from remaining_activation_configuration import capability
    assert row(source(),ident)['runtimeContract']==capability(ident)

@pytest.mark.parametrize('ident',IDS)
def test_missing_original_artwork_is_a_real_validation_failure(ident,tmp_path):
    v=source();root=fixture_root(v,tmp_path)
    (root/'Sources/KwipMLApp/Resources/ComponentIcons'/f'{ident}-v2.png').unlink()
    with pytest.raises(ValueError,match='source reference'):bundles.validate(v,root)

def test_generator_preserves_entire_previous_activation_projection():
    v=source();v['components']=[r for r in v['components'] if r['definition']['id'] in OLD]
    previous=bundles.render_swift(v,'activations')
    assert hashlib.sha256(previous.encode()).hexdigest()=='3959c0eb78c321e7336f0e20f746c95e7ce06efb51961ad2a8c1feadefcb7817'

@pytest.mark.parametrize('ident,config,parameters',[
    ('prelu',{'num_parameters':2,'init':-.5},2),
    ('log_softmax',{'dim':1},0),
    ('dropout',{'p':1},0),
])
def test_generated_tables_bind_real_normalization_resolution_and_capability(ident,config,parameters):
    import sys
    sys.path.insert(0,str(ROOT/'runner'))
    from remaining_activation_adapters_generated import CAPABILITIES,NORMALIZERS,RESOLVERS
    from tensor_layout import BatchedLayout
    assert set(CAPABILITIES)==set(NORMALIZERS)==set(RESOLVERS)==set(IDS)
    assert CAPABILITIES[ident]()==row(source(),ident)['runtimeContract']
    normalized=NORMALIZERS[ident](config)
    result=RESOLVERS[ident](normalized,BatchedLayout((2,3)))
    assert result['layout'].sample==(2,3) and result['parameterCount']==parameters
    assert result['config']==normalized


def test_actual_registry_preserves_all_previous39_definitions_after_three_additions():
    registry=json.loads((ROOT/'Sources/KwipMLCore/Resources/components.json').read_text())['components'];registry=[r for r in registry if r['id'] not in ('rms_norm','rotary_position','mixture_experts','initialization','gradient_policy','trainability','bag_input','embedding_bag','dataset_source','data_loader')]
    assert len(registry)==83 and len({d['id'] for d in registry})==83
    previous=[d for d in registry if d['id'] not in ('rms_norm','rotary_position','mixture_experts',*IDS,'bag_input','embedding_bag','dataset_source','data_loader','initialization','split','chunk','gather','einsum','unfold','fold','batch_norm1d','batch_norm2d','instance_norm1d','instance_norm2d','group_norm','optimizer_sgd','optimizer_adam','optimizer_adamw','optimizer_rmsprop','optimizer_adagrad','scheduler_step_lr','scheduler_cosine_annealing_lr','scheduler_reduce_on_plateau','scheduler_one_cycle_lr','scheduler_linear_lr','rnn','lstm','gru','rnn_cell','lstm_cell','gru_cell','multihead_attention','transformer','transformer_encoder','transformer_decoder','scaled_dot_product_attention','attention_mask','loss_cross_entropy','loss_bce_with_logits','loss_mse','loss_l1','loss_nll','loss_kl_div','loss_ctc','loss_triplet_margin')]
    assert len(previous)==39
    previous=historical_output_presentation(previous)
    assert hashlib.sha256(json.dumps(previous,sort_keys=True,separators=(',',':')).encode()).hexdigest()=='8e7d7c360e5db94e98fe6fa5e971e339c9133828c185e2650eb9dd809f717026'
    for ident in IDS:
        assert next(d for d in registry if d['id']==ident)==row(source(),ident)['definition']
    for filename in ('RemainingActivationLearningTask.swift','RemainingActivationLessonReference.swift','RemainingActivationLearningSubmission.swift'):
        assert (ROOT/'Sources/KwipMLCore'/filename).is_file()


def test_generated_worker_table_excludes_unpublished_remaining_activations(tmp_path):
    import ast
    value=source();row(value,'prelu')['published']=False
    pooling=json.loads((ROOT/'components/pooling/bundle.json').read_text())
    fixture_root(value,tmp_path);fixture_root(pooling,tmp_path)
    for family,manifest in [('pooling',pooling),('activations',value)]:
        p=tmp_path/'components'/family/'bundle.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(manifest))
    (tmp_path/'components/registry-base.json').write_text((ROOT/'components/registry-base.json').read_text())
    result=bundles.render(tmp_path)
    module=ast.parse(result['runner/remaining_activation_adapters_generated.py'])
    advertised=next(ast.literal_eval(n.value) for n in module.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='IDS' for t in n.targets))
    assert set(advertised)=={'log_softmax','dropout'}
