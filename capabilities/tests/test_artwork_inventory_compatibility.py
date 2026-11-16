"""The approved alias cannot mask other inventory changes or mutate callers."""
from copy import deepcopy
import pytest
from capabilities.tests.artwork_inventory_compatibility import historical_artwork_entries

CURRENT = {'id':'kwip.data.sourceCSV','title':'Table Input','category':'Data','asset':'table-input-v1.png'}
HISTORICAL = {'id':'kwip.data.sourceCSV','title':'CSV source','category':'Data','asset':'csv-table-v2.png'}


def test_only_exact_approved_alias_is_normalized_without_touching_other_rows():
    other = {'id':'linear','title':'Changed title remains visible','category':'Layers','asset':'linear-v2.png'}
    entries = [other, dict(CURRENT)]
    saved = deepcopy(entries)
    assert historical_artwork_entries(entries) == [other, HISTORICAL]
    assert entries == saved


@pytest.mark.parametrize('entries', [[], [CURRENT, CURRENT], [HISTORICAL],
    [{**CURRENT,'title':'Anything'}], [{**CURRENT,'asset':'foreign.png'}],
    [{**CURRENT,'category':'Other'}], [{**CURRENT,'unknown':'extra'}]])
def test_missing_duplicate_old_or_unapproved_alias_is_refused(entries):
    with pytest.raises(AssertionError):
        historical_artwork_entries(entries)
