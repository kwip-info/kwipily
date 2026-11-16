"""Test-only bridge for the exact reviewed CSV-to-Table-Input inventory alias.

Historical comparison callers still compare every other row, order and original
proof; this does not normalize changed art bytes, new families or arbitrary labels.
"""

def historical_artwork_entries(entries):
    matches = [row for row in entries if row.get('id') == 'kwip.data.sourceCSV']
    assert len(matches) == 1, 'Exactly one current Table Input inventory row required'
    current = {'id':'kwip.data.sourceCSV','title':'Table Input','category':'Data','asset':'table-input-v1.png'}
    assert matches[0] == current, 'Unapproved Table Input alias metadata'
    historical = {'id':'kwip.data.sourceCSV','title':'CSV source','category':'Data','asset':'csv-table-v2.png'}
    return [dict(historical if row == current else row) for row in entries]
