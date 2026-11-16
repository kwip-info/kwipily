import copy
import json
from pathlib import Path
import pytest
from scripts.tests.registry_history import historical_output_presentation

ROOT=Path(__file__).resolve().parents[2]
HISTORIC={'id':'output','title':'Output','category':'Data','inputs':['in'],'outputs':[],'fields':[]}

@pytest.mark.parametrize('path',['components/registry-base.json','Sources/KwipMLCore/Resources/components.json'])
def test_model_output_renames_only_title_and_category(path):
    rows=json.loads((ROOT/path).read_text())['components']
    output=next(row for row in rows if row['id']=='output')
    assert output=={**HISTORIC,'title':'Model Output','category':'Boundaries'}
    assert historical_output_presentation([output])==[HISTORIC]
    assert output['title']=='Model Output' and output['category']=='Boundaries'

def test_historical_projection_keeps_runtime_changes_detectable():
    changed=copy.deepcopy(HISTORIC);changed['inputs']=['changed'];changed['fields']=[{'id':'unexpected'}]
    projected=historical_output_presentation([changed])[0]
    assert projected['inputs']==['changed'] and projected['fields']==[{'id':'unexpected'}]
    assert projected!=HISTORIC

@pytest.mark.parametrize('changes',[{'title':'Unexpected'},{'category':'Unexpected'},{'title':'Model Output','category':'Data'}])
def test_projection_refuses_unapproved_presentation(changes):
    with pytest.raises(AssertionError):historical_output_presentation([{**HISTORIC,**changes}])


def test_generated_pooling_learning_uses_current_model_output_arrows():
    from scripts.tests.registry_history import POOLING_HISTORY_PATH,POOLING_MODEL_OUTPUT_ARROWS,historical_pooling_output_prose
    import scripts.component_bundles as bundles
    actual=(ROOT/POOLING_HISTORY_PATH).read_text()
    assert bundles.render(ROOT)[POOLING_HISTORY_PATH]==actual
    authority=json.dumps(json.loads((ROOT/'components/pooling/bundle.json').read_text()),ensure_ascii=False)
    for arrow in POOLING_MODEL_OUTPUT_ARROWS:
        assert actual.count(arrow)==1 and authority.count(arrow)==1
    projected=historical_pooling_output_prose(actual)
    for arrow in POOLING_MODEL_OUTPUT_ARROWS:
        projected=projected.replace(arrow.replace('Model Output','Output'),arrow)
    import re
    marker=r'(public static let manifestDigest = ")[a-f0-9]{64}(")'
    # Only the computed source digest changes in addition to the exact four arrows.
    assert re.sub(marker,r'\1<DIGEST>\2',projected)==re.sub(marker,r'\1<DIGEST>\2',actual)
