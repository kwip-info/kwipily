import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('interaction_profile', Path(__file__).parents[1] / 'interaction_profile.py')
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


def receipt(value=1, **changes):
    metadata = dict(fixtureVersion=1, nodes=64, iterations=3, configuration='release', machine='arm64', os='test', cpu='fixture', toolchain='swift', mode='scripted', instrumentationVersion=1, harnessSHA256='fixture', renderEnvironment={'viewport':[1000,700], 'backingScale':2, 'appearance':'dark'})
    metadata.update(changes)
    return dict(metadata=metadata, capture=dict(schemaVersion=1, droppedSamples=0, samples=[dict(metric='selection', scenario='selection', milliseconds=value)]))


def test_percentiles_and_delta_are_observations():
    comparison = profile.compare(receipt(2), receipt(3))
    assert comparison[0]['p95DeltaMS'] == 1
    assert comparison[0]['p95DeltaPercent'] == 50
    assert comparison[0]['baselineCount'] == 1


@pytest.mark.parametrize('field,value', [('nodes',256), ('configuration','debug'), ('machine','different'), ('iterations',5), ('harnessSHA256','changed'), ('os','other')])
def test_incompatible_comparisons_are_refused(field,value):
    with pytest.raises(ValueError, match='incompatible'):
        profile.compare(receipt(), receipt(**{field:value}))


def test_truncation_and_missing_metric_are_not_improvements():
    truncated = receipt(); truncated['capture']['droppedSamples'] = 1
    with pytest.raises(ValueError, match='truncated'):
        profile.compare(receipt(), truncated)
    missing = receipt(); missing['capture']['samples'] = []
    with pytest.raises(ValueError, match='coverage'):
        profile.compare(receipt(), missing)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1, True])
def test_invalid_timings_are_refused(value):
    with pytest.raises(ValueError):
        profile.compare(receipt(), receipt(value))


def test_manual_captures_cannot_masquerade_as_scripted_baselines():
    with pytest.raises(ValueError, match='scripted'):
        profile.compare(receipt(mode='manual'), receipt(mode='manual'))


def test_changed_render_environment_is_incompatible():
    candidate = receipt(); candidate['metadata']['renderEnvironment']['backingScale'] = 1
    with pytest.raises(ValueError, match='renderEnvironment'):
        profile.compare(receipt(), candidate)
