import gc
import hashlib
import json
from pathlib import Path
import tempfile
import tracemalloc
import unittest

from runner import dataflow as df
from runner.dataflow_disk import DiskStore, disk_scope, iter_json
from runner.dataflow_table_reader import read_table_artifact


class TableArtifactReaderTests(unittest.TestCase):
    def fixture(self, rows=None, ids=None):
        rows = rows if rows is not None else [[1, 'hello'], [2, 'café']]
        ids = ids if ids is not None else ['abc:1', 'xyz:2']
        columns = [{'name': 'number', 'dtype': 'int64', 'nullable': False}, {'name': 'text', 'dtype': 'string', 'nullable': True}]
        return {'schemaVersion': 1, 'kind': 'table', 'columns': columns, 'rows': rows,
                'lineage': {'sourceDigest': 'a' * 64, 'partitionDigest': '', 'role': 'unassigned',
                            'observationIDs': ids, 'membershipDigest': df.membership_digest(ids),
                            'dataVersion': df.digest({'columns': columns, 'rows': rows, 'ids': ids})}}

    def write(self, root, value, canonical=True):
        path = Path(root) / 'table.json'
        with path.open('wb') as stream:
            if canonical:
                for chunk in iter_json(value):
                    stream.write(chunk)
            else:
                stream.write(json.dumps(value, ensure_ascii=False, indent=2).encode())
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def test_reader_shares_one_budget_and_inherits_operation_directory_and_check(self):
        with tempfile.TemporaryDirectory() as root:
            path,pin=self.write(root,self.fixture())
            table=read_table_artifact(path,pin)
            self.assertIs(table.rows.store.budget,table.row_ids.store.budget)
            table.rows.store.close();table.row_ids.store.close()
            calls=[]
            with disk_scope(root,check_cancelled=lambda:calls.append(True)) as budget:
                table=read_table_artifact(path,pin)
                self.assertIs(table.rows.store.budget,budget)
                self.assertIs(table.row_ids.store.budget,budget)
                self.assertEqual(table.rows.store.directory.parent,Path(root))
                self.assertEqual(table.row_ids.store.directory.parent,Path(root))
                self.assertGreater(len(calls),0)
                table.rows.store.close();table.row_ids.store.close()
                self.assertEqual(budget.used,0)

    def test_historical_order_and_canonical_order_have_same_identity(self):
        for canonical in (True, False):
            with tempfile.TemporaryDirectory() as root, disk_scope(root):
                value = self.fixture()
                path, pin = self.write(root, value, canonical)
                table = read_table_artifact(path, pin)
                self.assertEqual(tuple(table.rows), tuple(map(tuple, value['rows'])))
                self.assertEqual(tuple(table.row_ids), tuple(value['lineage']['observationIDs']))
                self.assertEqual(table.data_version, value['lineage']['dataVersion'])
                self.assertEqual(df.membership_digest(table.row_ids), value['lineage']['membershipDigest'])
                table.rows.store.close()
                table.row_ids.store.close()

    def test_duplicate_ids_and_forged_identity_are_refused_and_cleaned(self):
        for mutation in ('duplicate', 'membership', 'dataVersion', 'dtype', 'width'):
            with tempfile.TemporaryDirectory() as root, disk_scope(root) as budget:
                value = self.fixture()
                if mutation == 'duplicate':
                    value['lineage']['observationIDs'][1] = value['lineage']['observationIDs'][0]
                elif mutation in ('membership', 'dataVersion'):
                    value['lineage']['membershipDigest' if mutation == 'membership' else mutation] = '0' * 64
                elif mutation == 'dtype':
                    value['rows'][0][0] = 'not an integer'
                    value['lineage']['dataVersion'] = df.digest({'columns': value['columns'], 'rows': value['rows'], 'ids': value['lineage']['observationIDs']})
                else:
                    value['rows'][0].append(3)
                    value['lineage']['dataVersion'] = df.digest({'columns': value['columns'], 'rows': value['rows'], 'ids': value['lineage']['observationIDs']})
                path, pin = self.write(root, value)
                with self.assertRaises(ValueError):
                    read_table_artifact(path, pin)
                self.assertEqual(budget.used, 0)
                self.assertEqual(list(Path(root).glob('.table-rows-*')), [])

    def test_pin_symlink_trailing_json_and_oversized_values_refused(self):
        with tempfile.TemporaryDirectory() as root, disk_scope(root):
            path, pin = self.write(root, self.fixture())
            with self.assertRaisesRegex(ValueError, 'integrity'):
                read_table_artifact(path, '0' * 64)
            link = Path(root) / 'link'
            link.symlink_to(path)
            with self.assertRaisesRegex(ValueError, 'regular'):
                read_table_artifact(link, pin)
            path.write_bytes(path.read_bytes() + b'null')
            with self.assertRaisesRegex(ValueError, 'Trailing'):
                read_table_artifact(path, hashlib.sha256(path.read_bytes()).hexdigest())
            value = self.fixture([['x' * 1_048_580, 'a']], ['id'])
            path, pin = self.write(root, value)
            with self.assertRaisesRegex(ValueError, 'budget|oversized'):
                read_table_artifact(path, pin)

    def test_cancel_cleans_spools(self):
        with tempfile.TemporaryDirectory() as root, disk_scope(root) as budget:
            path, pin = self.write(root, self.fixture())
            calls = 0
            def check():
                nonlocal calls
                calls += 1
                if calls == 10:
                    raise RuntimeError('cancelled')
            with self.assertRaisesRegex(RuntimeError, 'cancelled'):
                read_table_artifact(path, pin, check)
            self.assertEqual(budget.used, 0)
            self.assertEqual(list(Path(root).glob('.table-rows-*')), [])

    def test_more_than_100k_rows_are_bounded_and_preserve_identity(self):
        with tempfile.TemporaryDirectory() as root, disk_scope(root):
            store = DiskStore(identity_prefix='a' * 64 + ':')
            for index in range(100_005):
                store.append((index, 'hello'), 'a' * 64 + ':' + str(index))
            rows, ids = store.finish()
            value = self.fixture(rows, ids)
            path, pin = self.write(root, value)
            expected_version = value['lineage']['dataVersion']
            store.close()
            del value, rows, ids, store
            gc.collect()
            tracemalloc.start()
            try:
                table = read_table_artifact(path, pin)
                _, peak = tracemalloc.get_traced_memory()
                self.assertLess(peak, 5_000_000)
                self.assertEqual(len(table.rows), 100_005)
                self.assertEqual(table.rows[-1], (100_004, 'hello'))
                self.assertEqual(table.data_version, expected_version)
                table.rows.store.close()
                table.row_ids.store.close()
            finally:
                tracemalloc.stop()


if __name__ == '__main__':
    unittest.main()
