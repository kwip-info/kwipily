import copy
import importlib.util
import json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('pooling_bundles',ROOT/'scripts/component_bundles.py')
bundles=importlib.util.module_from_spec(spec);spec.loader.exec_module(bundles)
def source():
    v=json.loads((ROOT/'components/pooling/bundle.json').read_text())
    for r in v['components']:
        ident=r['definition']['id'];r['presentation'].update(theme='green',icon=ident+'-v2')
        if ident in {'max_pool1d','max_pool2d','max_pool3d','avg_pool1d'}:
            r['learning']['lessonIDs']=['assessment.'+ident.replace('_','-')+'-windows']
    return v

def test_pooling_metadata_binds_four_actual_lessons_and_generated_art():
    v=source();bundles.validate(v,ROOT);s=bundles.render_swift(v,'pooling')
    assert 'public let lessonIDs: [String]' in s
    assert 'public static func metadata(' in s
    for r in v['components']:
        assert r['learning']['lessonIDs'][0] in s
        assert r['presentation']['icon'] in s

@pytest.mark.parametrize('index',[0,1,2,3])
@pytest.mark.parametrize('mutation',['definition','stride','runtime','dtype','typedVersion','indices','lesson','missingLesson','icon','theme','example','state'])
def test_pooling_exact_config_runtime_output_learning_and_artwork_drift_refused(index,mutation):
    v=source();r=v['components'][index]
    if mutation=='definition':r['definition']['title']='Other pool'
    elif mutation=='stride':r['definition']['fields'][1]['defaultValue']=1
    elif mutation=='runtime':r['runtimeContract']['nullStrideFollowsKernel']=False
    elif mutation=='dtype':r['runtimeContract']['graphDTypes']=['float64']
    elif mutation=='typedVersion':r['runtimeContract']['version']['major']=True
    elif mutation=='indices':r['outputConditions']={'indices':'ceil_mode'}
    elif mutation=='lesson':r['learning']['lessonIDs']=['assessment.other']
    elif mutation=='missingLesson':r['learning'].pop('lessonIDs')
    elif mutation=='icon':r['presentation']['icon']='pooling'
    elif mutation=='theme':r['presentation']['theme']='teal'
    elif mutation=='example':r['learning']['exampleID']='prepared-parallel-classifier'
    elif mutation=='state':r['runtimeContract']['parameters']=True
    with pytest.raises(ValueError):bundles.validate(v,ROOT)
