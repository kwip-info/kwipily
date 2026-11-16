"""Independent civil-time fixtures; canonical integers are explicit expectations."""
import unittest
from runner import dataflow_temporal as t

class TemporalTests(unittest.TestCase):
    def test_epoch_units_and_leap_date(self):
        self.assertEqual(t.parse('1970-01-01','date'),0)
        self.assertEqual(t.parse('2000-02-29','date'),11016)
        self.assertEqual(t.parse('12:30:01.000002','time'),45001000002)
        self.assertEqual(t.parse('1970-01-01T00:00:00.000001Z','datetime'),1)
        self.assertEqual(t.parse('1969-12-31T23:59:59.999999Z','datetime'),-1)
    def test_explicit_custom_pattern(self):
        self.assertEqual(t.parse('29/02/2000','date',pattern='%d/%m/%Y'),11016)
        with self.assertRaises(ValueError):t.parse('01/02/2000','date')
        with self.assertRaises(ValueError):t.parse('2023-02-29','date')
        self.assertIsNone(t.parse('','date',invalid='missing'))
        self.assertIsNone(t.parse(None,'date'))
    def test_precision_never_truncated(self):
        for text in ('2024-01-01T00:00:00.1234567Z','12:00:00.1234567'):
            with self.assertRaisesRegex(ValueError,'precision'):t.parse(text,'datetime' if 'T' in text else 'time',invalid='missing')
    def test_dst_ambiguity(self):
        args=dict(source_timezone='America/New_York')
        with self.assertRaisesRegex(ValueError,'ambiguous'):t.parse('2024-11-03T01:30:00','datetime',**args)
        self.assertEqual(t.parse('2024-11-03T01:30:00','datetime',ambiguous='earliest',**args),1730611800000000)
        self.assertEqual(t.parse('2024-11-03T01:30:00','datetime',ambiguous='latest',**args),1730615400000000)
        self.assertIsNone(t.parse('2024-11-03T01:30:00','datetime',ambiguous='missing',**args))
        with self.assertRaisesRegex(ValueError,'nonexistent'):t.parse('2024-03-10T02:30:00','datetime',**args)
        self.assertIsNone(t.parse('2024-03-10T02:30:00','datetime',nonexistent='missing',**args))
    def test_offset_format_extract_difference(self):
        self.assertEqual(t.parse('1970-01-01T01:00:00+01:00','datetime'),0)
        self.assertEqual(t.format_value(0,'datetime',timezone='America/New_York'),'1969-12-31T19:00:00.000000-05:00')
        self.assertEqual(t.extract(0,'datetime','hour',timezone='America/New_York'),19)
        self.assertEqual(t.difference(3600000001,0,'datetime',unit='microseconds',representation='integer'),3600000001)
        self.assertEqual(t.difference(5400000000,0,'datetime',unit='hours',representation='decimal'),1.5)
        with self.assertRaisesRegex(ValueError,'exact'):t.difference(1,0,'datetime',unit='seconds',representation='integer')
    def test_schema_and_physical_validation(self):
        self.assertEqual(t.metadata('datetime','UTC'),{'version':1,'unit':'microsecond','timezone':'UTC'})
        self.assertTrue(t.valid_value(0,'date'))
        self.assertFalse(t.valid_value(True,'date'))
        self.assertFalse(t.valid_value(86400000000,'time'))
        with self.assertRaises(ValueError):t.metadata('datetime','Not/AZone')
        with self.assertRaises(ValueError):t.parse('2024-01-01T00:00:00','datetime')
        with self.assertRaises(ValueError):t.parse('12:00:00Z','time')

if __name__=='__main__':unittest.main()

def test_numeric_custom_format_preserves_escaped_percent_and_year_padding():
    assert t.format_value(t.parse('0001-01-01','date'),'date',pattern='%%Y %Y/%m/%d')=='%Y 0001/01/01'
    assert t.format_value(123456,'time',pattern='%H:%M:%S.%f')=='00:00:00.123456'

def test_escaped_pattern_literals_never_supply_required_components():
    import pytest
    for value,dtype,pattern in (
        ('%Y-%m-%d','date','%%Y-%%m-%%d'),
        ('%H:%M:%S','time','%%H:%%M:%%S'),
        ('%Y-02-29T12:30:01','datetime','%%Y-%m-%dT%H:%M:%S'),
    ):
        with pytest.raises(ValueError,match='explicitly include'):
            t.parse(value,dtype,pattern=pattern,source_timezone='UTC')


def test_escaped_pattern_literals_do_not_count_as_forbidden_components():
    assert t.parse('2000-02-29 %H %M %S %f %z','date',pattern='%Y-%m-%d %%H %%M %%S %%f %%z')==11016
    assert t.parse('12:30:01 %Y %m %d %z','time',pattern='%H:%M:%S %%Y %%m %%d %%z')==45001000000
    # A literal percent followed by a genuine directive remains a real component.
    assert t.parse('%2000-02-29','date',pattern='%%%Y-%m-%d')==11016

def test_real_directives_still_refuse_discarding_other_temporal_parts():
    import pytest
    for dtype in ('date','time'):
        with pytest.raises(ValueError,match='cannot discard'):
            t.parse('2000-02-29 12:30:01',dtype,pattern='%Y-%m-%d %H:%M:%S')
