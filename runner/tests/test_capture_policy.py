import itertools

import pytest

from runner.capture_policy import (
    resolve_training_sample_capture_policy,
    validate_training_sample_capture_policy,
)


DEFAULT = {'schemaVersion': 1, 'activationValues': True, 'predictionRows': True}


def test_legacy_default_is_explicit_and_each_resolution_is_independent():
    first = resolve_training_sample_capture_policy()
    assert first == DEFAULT
    first['activationValues'] = False
    assert resolve_training_sample_capture_policy(None) == DEFAULT


@pytest.mark.parametrize('activation,predictions', list(itertools.product([False, True], repeat=2)))
def test_independent_capture_choices_are_preserved_without_aliasing(activation, predictions):
    policy = dict(DEFAULT, activationValues=activation, predictionRows=predictions)
    validated = validate_training_sample_capture_policy(policy)
    resolved = resolve_training_sample_capture_policy(policy)
    assert validated == resolved == policy
    assert validated is not policy and resolved is not policy
    policy['predictionRows'] = not predictions
    assert validated['predictionRows'] is predictions
    assert resolved['predictionRows'] is predictions


@pytest.mark.parametrize('value', [None, [], '', 0, False, {},
    {key: value for key, value in DEFAULT.items() if key != 'activationValues'},
    dict(DEFAULT, extra=False), dict(DEFAULT, schemaVersion=True),
    dict(DEFAULT, schemaVersion=1.0), dict(DEFAULT, schemaVersion='1'),
    dict(DEFAULT, schemaVersion=2), dict(DEFAULT, schemaVersion=None)])
def test_strict_validator_rejects_missing_unknown_or_wrongly_typed_schema(value):
    with pytest.raises(ValueError):
        validate_training_sample_capture_policy(value)
    if value is not None:
        with pytest.raises(ValueError):
            resolve_training_sample_capture_policy(value)


@pytest.mark.parametrize('field', ['activationValues', 'predictionRows'])
@pytest.mark.parametrize('value', [None, 0, 1, 0.0, 1.0, '', 'true', 'false', [], {}])
def test_flags_require_actual_booleans(field, value):
    policy = dict(DEFAULT, **{field: value})
    with pytest.raises(ValueError):
        validate_training_sample_capture_policy(policy)
    with pytest.raises(ValueError):
        resolve_training_sample_capture_policy(policy)
