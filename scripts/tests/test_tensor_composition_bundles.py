from scripts.tests.registry_history import historical_output_presentation
"""Exact additive composition controls, identities and source/reference boundaries."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('composition_bundles',ROOT/'scripts/component_bundles.py')
bundles=importlib.util.module_from_spec(spec);spec.loader.exec_module(bundles)
IDS=('concat','add','stack','permute')
TASKS={'concat':'concat-sample-axis','add':'add-shared-pattern','stack':'stack-new-direction','permute':'permute-sample-order'}
def source():return json.loads((ROOT/'components/tensor-composition/bundle.json').read_text())
def row(value,ident):return next(r for r in value['components'] if r['definition']['id']==ident)
def reference_root(value,tmp_path):
    # Explicit fixture isolates authoring policy from concurrently arriving real source/art files.
    for r in value['components']:
        for ref in list(r['sources'].values())+[r['learning']['exampleSource'],'Sources/KwipMLApp/Resources/'+r['learning']['sourceResource']+'.csv','Sources/KwipMLApp/Resources/ComponentIcons/'+r['presentation']['icon']+'.png']:
            p=tmp_path/ref;p.parent.mkdir(parents=True,exist_ok=True);p.touch()
    return tmp_path

def test_four_reviewed_definitions_are_authored_once_without_duplicate_base_entries():
    value=source()
    assert [r['definition']['id'] for r in value['components']]==list(IDS)
    base=json.loads((ROOT/'components/registry-base.json').read_text())
    assert not {'concat','add','stack','permute'}&{r['id'] for r in base['components']}
    assert {r['id'] for r in base['components']}=={'input','output','depthwise_conv2d','mixture_experts'}
    for ident in IDS:
        r=row(value,ident)
        assert r['definition']['inputs']==(['in'] if ident=='permute' else ['a','b'])
        assert r['definition']['outputs']==['out'] and r['outputConditions']=={}
        assert r['learning']['lessonIDs']==['assessment.'+TASKS[ident]]
        assert r['learning']['exampleID']=='prepared-'+ident+'-classifier'
        assert r['presentation']['theme']==('violet' if ident=='permute' else 'blue')
        assert r['presentation']['icon']==ident+'-v2'

def test_controls_preserve_historical_defaults_and_expose_nondefault_configuration():
    value=source();fields={i:{f['key']:f for f in row(value,i)['definition']['fields']} for i in IDS}
    for ident in ('concat','stack'):
        assert set(fields[ident])=={'dim'}
        assert fields[ident]['dim']['defaultValue']==-1
        assert (fields[ident]['dim']['min'],fields[ident]['dim']['max'])==(-8,7)
    assert set(fields['add'])=={'alpha','sample_broadcast'}
    assert fields['add']['alpha']['kind']=='number' and fields['add']['alpha']['defaultValue']==1
    assert fields['add']['alpha']['min']==-3.4028234663852886e38 and fields['add']['alpha']['max']==3.4028234663852886e38
    assert fields['add']['sample_broadcast']['defaultValue'] is False
    assert fields['permute']['dims']['defaultValue']==[0,2,1]
    assert fields['permute']['dims']['minItems']==2 and fields['permute']['dims']['maxItems']==8

@pytest.mark.parametrize('ident',IDS)
@pytest.mark.parametrize('mutation',['field','default','rank','schema','dtype','batch','arity','version','budget','output','lesson','icon','theme','adapter','condition','source','extraContract','publication'])
def test_unreviewed_config_transport_source_and_lesson_changes_refuse(ident,mutation,tmp_path):
    value=source();root=reference_root(value,tmp_path);bundles.validate(value,root);r=row(value,ident)
    if mutation=='field':r['definition']['fields'].append({'key':'out'})
    elif mutation=='default':r['definition']['fields'][0]['defaultValue']=True
    elif mutation=='rank':r['runtimeContract']['maximumRank']=9
    elif mutation=='schema':r['runtimeContract']['graphSchemaVersions']=[3]
    elif mutation=='dtype':r['runtimeContract']['graphDTypes']=['float16']
    elif mutation=='batch':r['runtimeContract']['preservesBatch']=False
    elif mutation=='arity':r['definition']['inputs'].append('third')
    elif mutation=='version':r['runtimeContract']['version']['major']=True
    elif mutation=='budget':r['runtimeContract']['maximumTensorElements']=100000001
    elif mutation=='output':r['definition']['outputs'].append('mutating')
    elif mutation=='lesson':r['learning']['lessonIDs']=['assessment.legacy-merge']
    elif mutation=='icon':r['presentation']['icon']='add-v2' if ident!='add' else 'concat-v2'
    elif mutation=='theme':r['presentation']['theme']='amber'
    elif mutation=='adapter':r['adapterID']='dynamic.code'
    elif mutation=='condition':r['outputConditions']={'out':'sample_broadcast'}
    elif mutation=='source':r['sources']['worker']='../escape.py'
    elif mutation=='extraContract':r['runtimeContract']['out']=True
    elif mutation=='publication':r['published']=1
    with pytest.raises(ValueError):bundles.validate(value,root)

def test_each_new_component_uses_its_own_factory_and_bridge():
    rendered=bundles.render_swift(source(),'tensor-composition')
    for ident in IDS:
        assert 'WorkflowExamples.prepared'+ident.capitalize()+'Classifier(device: $0)' in rendered
        assert 'BuiltInCapabilityBridge.'+ident+'Declaration' in rendered
    assert 'public let lessonIDs: [String]' in rendered

def test_remaining_base_rows_are_semantically_byte_identical():
    hashes={'input':'d56a445d64a12ac50987b86656f53b5c600776222919a1557f1a9ea7b344c4a2','output':'7048b23162d1d333dab3b63c2efc3703bb8c9fec4394e61b915cfe1ee61716e5','depthwise_conv2d':'86b2703d57459fb68b74bd12402846552d1caf9df0f6d33c553a12a00273731f'}
    for definition in json.loads((ROOT/'components/registry-base.json').read_text())['components']:
        if definition['id']=='mixture_experts': continue  # Added after this frozen cohort; verified separately.
        assert hashlib.sha256(json.dumps(historical_output_presentation([definition])[0],sort_keys=True,separators=(',',':')).encode()).hexdigest()==hashes[definition['id']]

@pytest.mark.parametrize('ident',IDS)
def test_independently_pinned_runtime_contract_matches_actual_helper(ident):
    import sys
    sys.path.insert(0,str(ROOT/'runner'))
    from tensor_composition_configuration import capability
    assert row(source(),ident)['runtimeContract']==capability(ident)

@pytest.mark.parametrize('ident',IDS)
def test_missing_actual_asset_or_source_refuses(ident,tmp_path):
    value=source();root=reference_root(value,tmp_path)
    asset=root/'Sources/KwipMLApp/Resources/ComponentIcons'/f'{ident}-v2.png'
    asset.unlink()
    with pytest.raises(ValueError,match='source reference'):bundles.validate(value,root)

@pytest.mark.parametrize('ident,config,left,right,expected',[
    ('concat',{},(2,3),(2,1),(2,4)),
    ('add',{'alpha':-0.5,'sample_broadcast':True},(2,3),(3,),(2,3)),
    ('stack',{'dim':2},(2,3),(2,3),(2,2,3)),
    ('permute',{'dims':[0,2,3,1]},(2,3,4),None,(3,4,2)),
])
def test_generated_tables_bind_actual_typed_helpers(ident,config,left,right,expected):
    import sys
    sys.path.insert(0,str(ROOT/'runner'))
    from tensor_layout import BatchedLayout
    from tensor_composition_adapters_generated import CAPABILITIES,NORMALIZERS,RESOLVERS
    assert set(CAPABILITIES)==set(NORMALIZERS)==set(RESOLVERS)==set(IDS)
    assert CAPABILITIES[ident]()==row(source(),ident)['runtimeContract']
    normalized=NORMALIZERS[ident](config)
    result=RESOLVERS[ident](normalized,BatchedLayout(left),None if right is None else BatchedLayout(right))
    assert result.sample==expected


def test_actual_resources_generate_deterministically_without_prior_family_drift():
    outputs=bundles.render(ROOT)
    assert outputs==bundles.render(ROOT)
    for path,text in outputs.items():
        assert (ROOT/path).read_text()==text,path
    registry=json.loads(outputs['Sources/KwipMLCore/Resources/components.json'])['components'];registry=[r for r in registry if r['id'] not in ('rms_norm','rotary_position','mixture_experts','initialization','gradient_policy','trainability','bag_input','embedding_bag','dataset_source','data_loader')]
    assert len(registry)==83 and len({d['id'] for d in registry})==83
    assert [d['id'] for d in registry if d['id'] not in ('rms_norm','rotary_position','mixture_experts','split','chunk','gather','einsum','unfold','fold','batch_norm1d','batch_norm2d','instance_norm1d','instance_norm2d','group_norm','optimizer_sgd','optimizer_adam','optimizer_adamw','optimizer_rmsprop','optimizer_adagrad','scheduler_step_lr','scheduler_cosine_annealing_lr','scheduler_reduce_on_plateau','scheduler_one_cycle_lr','scheduler_linear_lr','rnn','lstm','gru','rnn_cell','lstm_cell','gru_cell','multihead_attention','transformer','transformer_encoder','transformer_decoder','scaled_dot_product_attention','attention_mask','loss_cross_entropy','loss_bce_with_logits','loss_mse','loss_l1','loss_nll','loss_kl_div','loss_ctc','loss_triplet_margin')] [-4:]==list(IDS)
    for ident in IDS:
        assert next(d for d in registry if d['id']==ident)==row(source(),ident)['definition']
