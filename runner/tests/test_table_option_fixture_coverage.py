"""Every exposed native reader field must name a real dedicated fixture test.

This checks coverage membership, not test success: pytest executes those tests
separately. The paired Swift check derives the denominator from current fields.
"""
import importlib
import json
from pathlib import Path


def test_every_exposed_option_has_a_dedicated_executable_fixture_case():
    inventory = json.loads((Path(__file__).parent / 'fixtures/table_options/inventory.json').read_text())
    expected = {context + '.' + key for context, keys in inventory['contexts'].items() for key in keys}
    observed = {}
    for suffix in ('csv_parquet', 'excel', 'markup_json', 'database'):
        module = importlib.import_module('runner.tests.test_table_options_' + suffix)
        for option, names in module.OPTION_CASES.items():
            assert option not in observed, f'Duplicate ownership: {option}'
            assert isinstance(names, (list, tuple)) and names, f'No fixture for {option}'
            for name in names:
                assert name.startswith('test_'), (option, name)
                function = getattr(module, name, None)
                assert callable(function) and function.__module__ == module.__name__, (option, name)
            observed[option] = names
    assert set(observed) == expected, {'missing': sorted(expected-set(observed)), 'unexposed': sorted(set(observed)-expected)}
