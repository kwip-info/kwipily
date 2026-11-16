import gc
import json
from pathlib import Path
import tempfile
import tracemalloc
import unittest


def test_disk_selected_pairs_batch_checks_preserve_values_and_bound_cancellation(tmp_path):
    import pytest
    cancelled=[False];checks=[0]
    def check():
        checks[0]+=1
        if cancelled[0]:raise InterruptedError('cancelled')
    store=DiskStore(directory=tmp_path,check_cancelled=check)
    try:
        expected=[((i,2**53+1 if i%2 else 0.5),f'id:{i}') for i in range(1000)]
        for row,rid in expected:store.append(row,rid)
        rows,ids=store.finish();checks[0]=0
        selected=list(range(1000))+[2,5,2]
        assert list(rows.iter_selected_pairs(selected))==[expected[i] for i in selected]
        assert checks[0]<20
        iterator=rows.iter_selected_pairs(range(1000));next(iterator);cancelled[0]=True
        with pytest.raises(InterruptedError):
            for _ in range(256):next(iterator)
    finally:store.close()


def test_disk_owner_close_precedes_late_cursor_cleanup(tmp_path):
    store=DiskStore(directory=tmp_path)
    for i in range(3):store.append((i,),str(i))
    rows,ids=store.finish();iterator=iter(rows);next(iterator);store.close();iterator.close()

from runner.dataflow_disk import DiskStore, disk_scope, iter_json


class DiskTableTests(unittest.TestCase):
    def test_sequence_and_canonical_json_match_legacy(self):
        with tempfile.TemporaryDirectory() as directory:
            store = DiskStore(directory=directory)
            expected = [(1, 'café', None), (False, '\n', 3.25), (2, '😀', -0.0)]
            for index, row in enumerate(expected):
                store.append(row, ['z', 'é', 'a'][index])
            rows, ids = store.finish()
            self.assertEqual(list(rows), expected)
            self.assertEqual(rows[-1], expected[-1])
            self.assertEqual(rows[::-1], tuple(reversed(expected)))
            self.assertEqual(list(ids.sorted_iter()), ['a', 'z', 'é'])
            actual = {'rows': rows, 'ids': ids, 'nested': {'z': [], 'a': (1, 2)}}
            ordinary = {'rows': expected, 'ids': ['z', 'é', 'a'], 'nested': {'z': [], 'a': (1, 2)}}
            chunks = list(iter_json(actual, 17))
            self.assertTrue(all(len(chunk) <= 17 for chunk in chunks))
            self.assertEqual(b''.join(chunks), json.dumps(ordinary, sort_keys=True, separators=(',', ':'), allow_nan=False, ensure_ascii=True).encode())
            store.close()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_identity_prefix_compacts_storage_without_changing_external_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = 'abc' * 20 + ':'
            store = DiskStore(directory=directory, identity_prefix=prefix)
            store.append((1,), prefix + 'z')
            store.append((2,), prefix + 'a')
            rows, ids = store.finish()
            self.assertEqual(list(ids), [prefix + 'z', prefix + 'a'])
            self.assertEqual(list(ids.sorted_iter()), [prefix + 'a', prefix + 'z'])
            self.assertTrue(ids.contains(prefix + 'z'))
            self.assertIn(prefix + 'a', ids)
            self.assertFalse(ids.contains(prefix + 'missing'))
            self.assertFalse(ids.contains('different:z'))
            self.assertFalse(ids.contains(None))
            self.assertEqual(store.connection.execute('SELECT identity FROM rows ORDER BY position').fetchall(), [('z',), ('a',)])
            store.close()

    def test_shared_budget_and_failed_append_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            with disk_scope(directory, 80_000) as budget:
                first = DiskStore()
                first.append(('x' * 20_000,), 'first')
                second = DiskStore()
                with self.assertRaisesRegex(ValueError, 'aggregate scratch'):
                    second.append(('y' * 60_000,), 'second')
                self.assertFalse(second.directory.exists())
                self.assertGreater(budget.used, 0)
                first.close()
                self.assertEqual(budget.used, 0)

    def test_unique_identity_failure_releases_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            store = DiskStore(directory=directory)
            store.append((1,), 'same')
            with self.assertRaisesRegex(ValueError, 'unique'):
                store.append((2,), 'same')
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_cancel_closes_writer_and_cancel_interrupts_reader(self):
        calls = 0
        active = False
        def check():
            nonlocal calls
            calls += 1
            if active:
                raise RuntimeError('cancelled')
        with tempfile.TemporaryDirectory() as directory:
            store = DiskStore(check, directory=directory)
            store.append((1,), 'a')
            rows, ids = store.finish()
            active = True
            with self.assertRaisesRegex(RuntimeError, 'cancelled'):
                list(rows)
            active = False
            store.close()
            store = DiskStore(check, directory=directory)
            active = True
            with self.assertRaisesRegex(RuntimeError, 'cancelled'):
                store.append((1,), 'a')
            self.assertFalse(store.directory.exists())

    def test_sequences_own_store_until_last_reference_released(self):
        with tempfile.TemporaryDirectory() as directory:
            store = DiskStore(directory=directory)
            store.append((1,), 'a')
            rows, ids = store.finish()
            location = store.directory
            del store
            gc.collect()
            self.assertTrue(location.exists())
            del rows
            gc.collect()
            self.assertEqual(ids[0], 'a')
            del ids
            gc.collect()
            self.assertFalse(location.exists())

    def test_large_stream_keeps_python_memory_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            tracemalloc.start()
            try:
                store = DiskStore(directory=directory)
                for index in range(100_005):
                    store.append((index, 'a' * 30), str(index))
                rows, ids = store.finish()
                total = sum(len(chunk) for chunk in iter_json({'rows': rows, 'ids': ids}))
                _, peak = tracemalloc.get_traced_memory()
                self.assertEqual(len(rows), 100_005)
                self.assertGreater(total, 4_000_000)
                self.assertLess(peak, 4_000_000)
                with self.assertRaisesRegex(ValueError, 'bounded materialization'):
                    rows[:]
                store.close()
            finally:
                tracemalloc.stop()

    def test_nonfinite_json_and_oversize_row_refused(self):
        with self.assertRaises(ValueError):
            b''.join(iter_json({'bad': float('nan')}))
        with tempfile.TemporaryDirectory() as directory:
            store = DiskStore(directory=directory)
            with self.assertRaisesRegex(ValueError, '1 MiB'):
                store.append(('x' * 1_048_577,), 'a')
            self.assertFalse(store.directory.exists())


if __name__ == '__main__':
    unittest.main()


def test_canonical_stream_preserves_edge_values_and_fixed_chunk_boundaries():
    from runner.dataflow_disk import iter_json
    values={'unicode':['café','😀','line\nquote"slash\\',None,True,False],
            'numbers':[-0.0,1.0,1e-200,1e200,-2**63,2**63-1],
            'wide':[list(range(128)),list(range(129))],
            'large':['😀'*70000,'tail'],'objects':[{'z':'value','a':[1,2]}]}
    expected=json.dumps(values,sort_keys=True,separators=(',',':'),allow_nan=False,ensure_ascii=True).encode()
    chunks=list(iter_json(values,4093))
    assert b''.join(chunks)==expected
    assert all(len(chunk)==4093 for chunk in chunks[:-1]) and len(chunks[-1])<=4093
    with __import__('pytest').raises(ValueError):list(iter_json({'value':float('nan')}))


def test_membership_cache_cannot_bypass_cancelled_or_closed_storage(tmp_path):
    from runner.dataflow import membership_digest
    import pytest
    cancelled=False
    def check():
        if cancelled:raise RuntimeError('owner cancelled')
    store=DiskStore(check,directory=tmp_path)
    for i,rid in enumerate(['é','a','😀']):store.append((i,),rid)
    rows,ids=store.finish()
    assert membership_digest(ids)==membership_digest(['é','a','😀'])
    cancelled=True
    with pytest.raises(RuntimeError,match='owner cancelled'):membership_digest(ids)
    cancelled=False;store.close()
    with pytest.raises(ValueError):membership_digest(ids)
