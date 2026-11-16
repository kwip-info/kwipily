"""Strict source recipe bundle admission, genuine descriptors and deterministic generation."""
import copy
import json
from pathlib import Path
import pytest
from scripts import component_bundles as bundles
from runner.dataset_loader_configuration import capability

ROOT=Path(__file__).resolve().parents[2]
def authored():return json.loads((ROOT/'components/training-data/bundle.json').read_text())

def test_actual_authored_data_loading_recipes_and_worker_contracts_admit():
    value=authored();bundles.validate(value,ROOT)
    for row in value['components']:
        assert row['runtimeContract']==capability(row['definition']['id'])
        assert row['definition']['inputs']==[] and row['definition']['outputs']==['out']
    assert [len(r['modes']) for r in value['components']]==[4,7]

@pytest.mark.parametrize('mutation',[
    lambda v:v['components'][0]['definition']['inputs'].append('in'),
    lambda v:v['components'][0]['definition']['fields'][1].update(maxItems=200000),
    lambda v:v['components'][1]['runtimeContract'].update(workerMaximum=3),
    lambda v:v['components'][1]['runtimeContract'].update(persistentWorkers=0),
    lambda v:v['components'][0]['learning']['lessonIDs'].pop(),
    lambda v:v['components'][1]['learning']['sources'].append('../secret'),
    lambda v:v['components'][1]['presentation'].update(icon='docs/design/lifecycle-icons-v1/checkpoint-save-restore-v1.png'),
    lambda v:v['components'][0].update(configurationVersion=True),
])
def test_data_recipe_fake_forward_controls_privacy_artwork_drift_refused(mutation):
    value=authored();mutation(value)
    with pytest.raises(ValueError):bundles.validate(value,ROOT)

def test_recipe_generation_is_deterministic_and_publication_matches_authored_contract():
    first=bundles.render(ROOT);second=bundles.render(ROOT)
    assert first==second
    swift=first['Sources/KwipMLCore/BundledDataLoadingComponents.generated.swift']
    assert 'public enum BundledDataLoadingComponents' in swift and 'public static let entries' in swift
    published={row['definition']['id'] for row in authored()['components'] if row['published']}
    assert swift.count('DataLoadingCapabilityBridge.declaration') == len(published)
    registry=json.loads(first['Sources/KwipMLCore/Resources/components.json'])
    assert {'dataset_source','data_loader'} & {r['id'] for r in registry['components']} == published
    assert 'RESOLVERS' not in first['runner/data_loading_capabilities_generated.py']

def test_published_data_recipe_requires_registered_icon_identical_to_generated_original(monkeypatch):
    value=authored();value['components'][0]['published']=True
    real=Path.is_file
    monkeypatch.setattr(Path,'is_file',lambda p:False if p.name=='dataset-v1.png' and 'ComponentIcons' in p.parts else real(p))
    with pytest.raises(ValueError,match='artwork|registered'):bundles.validate(value,ROOT)


def test_authored_editor_limits_fit_shared_list_and_exact_number_controls():
    value=json.loads((ROOT/'components/training-data/bundle.json').read_text())
    definitions={row['definition']['id']:row['definition'] for row in value['components']}
    source={f['key']:f for f in definitions['dataset_source']['fields']}
    loader={f['key']:f for f in definitions['data_loader']['fields']}
    assert source['indices']['maxItems']==4096
    assert source['indices']['max']==99999
    assert loader['seed']['max']==9007199254740991
