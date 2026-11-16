"""Approved copy changes must not mask execution or unknown presentation drift."""
import copy
import pytest
from scripts.tests.registry_history import historical_output_presentation


def current():
    return {'id': 'add', 'category': 'Merge', 'fields': [
        {'key': 'alpha', 'title': 'B multiplier', 'defaultValue': 1, 'min': -4096}]}


def test_approved_copy_projection_is_nondestructive_and_keeps_contract_values():
    row = current()
    before = copy.deepcopy(row)
    result = historical_output_presentation([row])[0]
    assert row == before
    assert result == dict(row, fields=[dict(row['fields'][0], title='Second source multiplier')])
    mutated = copy.deepcopy(row)
    mutated['fields'][0]['defaultValue'] = 2
    assert historical_output_presentation([mutated]) != historical_output_presentation([row])


def test_unknown_copy_is_not_silently_accepted():
    row = current()
    row['fields'][0]['title'] = 'Do something else'
    with pytest.raises(AssertionError):
        historical_output_presentation([row])
