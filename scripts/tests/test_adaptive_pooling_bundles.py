"""Reviewed consumer controls and transport identities for the additive pooling batch."""
import copy
import importlib.util
import json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('adaptive_bundles',ROOT/'scripts/component_bundles.py')
bundles=importlib.util.module_from_spec(spec);spec.loader.exec_module(bundles)
NEW=('avg_pool2d','adaptive_avg_pool1d','adaptive_avg_pool2d','adaptive_max_pool2d')
def source():return json.loads((ROOT/'components/pooling/bundle.json').read_text())
def row(value,ident):return next(r for r in value['components'] if r['definition']['id']==ident)

def test_all_four_components_have_distinct_real_authoring_and_learning_identities():
    value=source()
    assert set(NEW)<=set(r['definition']['id'] for r in value['components'])
    for ident in NEW:
        r=row(value,ident)
        assert r['published'] is True
        assert r['presentation']['icon']==ident+'-v2'
        assert r['presentation']['theme']=='green'
        assert r['learning']['lessonIDs']==['assessment.'+ident.replace('_','-')+('-windows' if ident=='avg_pool2d' else '-bins')]
        assert r['learning']['exampleID']=='prepared-'+ident.replace('_','')+'-classifier'
        assert r['sources']['worker'] in ['runner/avgpool2d_configuration.py','runner/adaptive_pooling_configuration.py']

@pytest.mark.parametrize('ident',NEW)
def test_exact_reviewed_controls_and_runtime_contract_are_not_a_generic_wrapper(ident):
    r=row(source(),ident)
    keys={f['key'] for f in r['definition']['fields']}
    if ident=='avg_pool2d':
        assert keys=={'kernel_size','stride','padding','ceil_mode','count_include_pad','divisor_override'}
        divisor=next(f for f in r['definition']['fields'] if f['key']=='divisor_override')
        assert divisor['defaultValue'] is None and divisor['enabledValue']==1
        assert divisor['min']==-2147483647 and divisor['max']==2147483647
    else:
        assert keys==({'output_size'} if ident.endswith('1d') else {'output_height','output_width'})|({'return_indices'} if ident=='adaptive_max_pool2d' else set())
        assert r['runtimeContract']['nullOutputKeepsInputAxis']==(not ident.endswith('1d'))
        assert r['runtimeContract']['outputElementMaximum']==100000000
    assert r['runtimeContract']['graphSchemaVersions']==[2]
    assert r['runtimeContract']['graphDTypes']==['float32']
    assert r['outputConditions']==({'indices':'return_indices'} if ident=='adaptive_max_pool2d' else {})

@pytest.mark.parametrize('ident',NEW)
@pytest.mark.parametrize('mutation',['field','default','rank','dtype','typedVersion','schema','limit','indices','lesson','icon','source','example','condition','extraContract'])
def test_unreviewed_transport_controls_references_and_lesson_bindings_refused(ident,mutation,tmp_path):
    value=source();r=row(value,ident)
    # A temporary reference tree isolates authoring validation from root's concurrent source/art delivery.
    for record in value['components']:
        for reference in list(record['sources'].values())+[record['learning']['exampleSource'],'Sources/KwipMLApp/Resources/'+record['learning']['sourceResource']+'.csv','Sources/KwipMLApp/Resources/ComponentIcons/'+record['definition']['id']+'-v2.png']:
            p=tmp_path/reference;p.parent.mkdir(parents=True,exist_ok=True);p.touch()
    bundles.validate(value,tmp_path)
    if mutation=='field':r['definition']['fields'].append({'key':'custom_code'})
    elif mutation=='default':r['definition']['fields'][0]['defaultValue']=True
    elif mutation=='rank':r['runtimeContract']['sampleRank']=8
    elif mutation=='dtype':r['runtimeContract']['graphDTypes']=['float64']
    elif mutation=='typedVersion':r['runtimeContract']['version']['major']=True
    elif mutation=='schema':r['runtimeContract']['graphSchemaVersions']=[1,2]
    elif mutation=='limit':r['runtimeContract']['channelMaximum']=513
    elif mutation=='indices':r['definition']['outputs'].append('extra')
    elif mutation=='lesson':r['learning']['lessonIDs']=['assessment.other']
    elif mutation=='icon':r['presentation']['icon']='max_pool2d-v2'
    elif mutation=='source':r['sources']['worker']='../escape.py'
    elif mutation=='example':r['learning']['exampleID']='prepared-parallel-classifier'
    elif mutation=='condition':r['outputConditions']={'indices':'ceil_mode'}
    elif mutation=='extraContract':r['runtimeContract']['inplace']=True
    with pytest.raises(ValueError):bundles.validate(value,tmp_path)

def test_generated_factories_are_per_component_not_existing_max_pool_aliases():
    rendered=bundles.render_swift(source(),'pooling')
    for suffix,decl in [('AvgPool2d','avgPool2d'),('AdaptiveAvgPool1d','adaptiveAvgPool1d'),('AdaptiveAvgPool2d','adaptiveAvgPool2d'),('AdaptiveMaxPool2d','adaptiveMaxPool2d')]:
        assert 'WorkflowExamples.prepared'+suffix+'Classifier(device: $0)' in rendered
        assert 'BuiltInCapabilityBridge.'+decl+'Declaration' in rendered

def test_accepted_pooling_rows_remain_semantically_frozen():
    import hashlib
    expected={'max_pool1d':'ed488857809ab0f1808daed92a98745e3cfb9953472a77f7765429bdc3cccb01','max_pool2d':'087dec0d5808f151e70ae5630603d87221c29e20cc4f6fb770278ff7485515f9','max_pool3d':'7353c9bf7982a72202e60f1d44eb7b02929d9a8fd448dde2357d6e3acdfc5a12','avg_pool1d':'8503c7de1b4f1c0688b1559ca64a3c09d8da0364bb19e159a7fca605c7f3e18f'}
    for ident,digest in expected.items():
        r=row(source(),ident)
        assert hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()==digest

@pytest.mark.parametrize('ident',NEW)
def test_authored_descriptor_matches_actual_allocation_free_worker_capability(ident):
    from runner.avgpool2d_configuration import capability as avg_capability
    from runner.adaptive_pooling_configuration import capability as adaptive_capability
    actual=avg_capability() if ident=='avg_pool2d' else adaptive_capability(ident)
    assert row(source(),ident)['runtimeContract']==actual
