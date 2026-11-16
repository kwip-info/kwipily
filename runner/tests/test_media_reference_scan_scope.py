"""Scalar tables must not materialize rows for an unrelated media inventory scan."""
from types import SimpleNamespace
import pytest
from runner import media_artifacts as media


class UnreadableRows:
    def __iter__(self):
        raise AssertionError('Scalar payload rows crossed into Python during media scan')


def test_scalar_table_media_inventory_does_not_visit_rows():
    table = SimpleNamespace(columns=({'name':'amount','dtype':'float64'},), rows=UnreadableRows())
    assert list(media.references(table)) == []


def test_media_table_still_checks_reference_kind():
    _, reference = media.encode_asset(bytes(2), kind='audio', shape=[1,1],
        axes=['samples','channels'], dtype='int16', metadata={'sampleRate':16000,'sourceBits':16})
    table = SimpleNamespace(columns=({'name':'clip','dtype':'audio'},), rows=((reference,), (None,)))
    assert list(media.references(table)) == [reference]
    wrong = SimpleNamespace(columns=({'name':'clip','dtype':'tensor'},), rows=((reference,),))
    with pytest.raises(ValueError, match='kind differs'):
        list(media.references(wrong))
