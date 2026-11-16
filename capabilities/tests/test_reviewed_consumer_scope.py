"""Reviewed consumer scope must preserve source identities and compulsory modes."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest
import tempfile

ROOT = Path(__file__).resolve().parents[2]
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value
coverage = module('coverage', ROOT / 'scripts/coverage_report.py')
base = module('base_tests', Path(__file__).with_name('test_coverage_report.py'))

class ReviewedScopeContractTests(unittest.TestCase):
    setUp = base.CoverageTests.setUp
    report = base.CoverageTests.report
    rewrite_inventory = base.CoverageTests.rewrite_inventory
    def grouped(self):
        unit = self.catalog['capabilities'][0]
        self.records[2].update(disposition='mapped', capability_ids=['op'])
        unit['source_members'] = [
            {'canonical_symbol':'lib.Op', 'required_modes':['default']},
            {'canonical_symbol':'lib.Other', 'required_modes':['restore']}]
        unit['mode_scope'].append('restore')
        self.rewrite_inventory()

    def test_approved_consumer_count_is_visible_before_mode_acceptance(self):
        self.catalog['denominator_status'] = 'approved-consumer-scope'
        self.catalog['scope_decision'] = {'approved_unit_count':1, 'status':'owner-approved-consumer-scope'}
        self.catalog['boundary_review'] = {'status':'pending'}
        self.catalog['deferrals'] = {'status':'pending-owner-review', 'units':[]}
        self.catalog['capabilities'][0]['mode_scope_status'] = 'pending'
        result = self.report()
        self.assertEqual(result['approved_consumer_unit_count'], 1)
        self.assertEqual(result['minimum_complete_units'], 1)
        self.assertFalse(result['release_gate_passed'])

    def test_approved_scope_cannot_omit_final_deferral_review(self):
        self.catalog['denominator_status'] = 'approved-consumer-scope'
        self.catalog['scope_decision'] = {'approved_unit_count':1}
        self.catalog['boundary_review'] = {'status':'reviewed'}
        self.assertFalse(self.report()['release_gate_passed'])

    def test_group_cannot_hide_unmapped_member(self):
        self.grouped()
        self.records[2]['capability_ids'] = []
        self.rewrite_inventory()
        with self.assertRaisesRegex(ValueError, 'reciprocal'): self.report()

    def test_group_cannot_hide_required_mode(self):
        self.grouped()
        self.catalog['capabilities'][0]['mode_scope'].remove('restore')
        with self.assertRaisesRegex(ValueError, 'required modes'): self.report()

    def test_group_requires_evidence_for_both_directions(self):
        self.grouped()
        self.assertEqual(self.report()['complete_count'], 0)
        for items in self.catalog['capabilities'][0]['evidence'].values():
            item = copy.deepcopy(items[0]); item['scope'] = 'op:restore'; items.append(item)
        self.assertEqual(self.report()['complete_count'], 1)
        self.assertEqual(self.report()['api']['mapped_canonical_count'], 2)
        self.assertEqual(self.report()['capability_count'], 1)

    def test_group_cannot_share_secondary_member_and_mode(self):
        self.grouped()
        extra = copy.deepcopy(self.catalog['capabilities'][0])
        extra.update(id='other', canonical_symbol='lib.Other', mode_scope=['restore'])
        extra.pop('source_members')
        self.catalog['capabilities'].append(extra)
        self.records[2]['capability_ids'].append('other')
        self.rewrite_inventory()
        with self.assertRaisesRegex(ValueError, 'inflated'): self.report()

    def test_pending_deferrals_block_high_aggregate(self):
        self.catalog['deferrals'] = {'status':'pending-owner-review', 'units':[]}
        self.assertFalse(self.report()['release_gate_passed'])

    def test_approved_deferral_cannot_mask_empty_required_family(self):
        self.catalog['deferrals'] = {'status':'owner-approved', 'reference':'owner receipt', 'units':[]}
        self.catalog['required_families'].append('recurrent')
        self.assertFalse(self.report()['release_gate_passed'])

class MigratedCatalogTests(unittest.TestCase):
    def test_exact_reviewed_88_with_original_99_source_members(self):
        catalog = json.loads((ROOT/'capabilities/catalogs/v1/pytorch.json').read_text())
        selected = json.loads((ROOT/'capabilities/selection-v1.json').read_text())['pytorch']
        originals = {symbol for symbols in selected.values() for symbol in symbols}
        members = [member['canonical_symbol'] for unit in catalog['capabilities']
                   for member in unit.get('source_members', [{'canonical_symbol':unit['canonical_symbol']}])]
        self.assertEqual(len(catalog['capabilities']), 88)
        self.assertEqual(set(members), originals)
        self.assertEqual(len(members), 99)
        result = coverage.report_catalog(catalog, ROOT)
        self.assertEqual(result['approved_consumer_unit_count'], 88)
        self.assertEqual(result['minimum_complete_units'], 80)
        self.assertEqual(result['target_complete_units'], 84)
        self.assertEqual(len(result['families']), 17)
        # Numerator advances through separately reviewed evidence bindings.
        self.assertFalse(result['release_gate_passed'])
        self.assertIsNone(result['api']['framework_wide_percent'])
        self.assertEqual(result['api']['symbol_count'], 9066)

class MigrationSafetyTests(unittest.TestCase):
    def setUp(self):
        self.migration = module('consumer_migration', ROOT/'capabilities/migrate_pytorch_consumer_scope.py')
        self.catalog = json.loads((ROOT/'capabilities/catalogs/v1/pytorch.json').read_text())
        self.inventory = json.loads((ROOT/'capabilities/inventories/v1/pytorch.json').read_text())
        self.selection = json.loads((ROOT/'capabilities/selection-v1.json').read_text())['pytorch']

    def test_migration_is_idempotent_and_preserves_all_inventory_records(self):
        catalog, inventory = self.migration.migrate(self.catalog, self.inventory, self.selection)
        self.assertEqual(catalog, self.catalog)
        self.assertEqual(inventory, self.inventory)
        self.assertEqual(len(inventory), 9066)
        self.assertTrue(all(row['reason'].strip() for row in inventory))
        self.assertEqual(sum(row['disposition']=='excluded' for row in inventory),192)
        self.assertEqual(sum(row['disposition']=='unsupported' for row in inventory),3314)

    def test_all_compulsory_grouped_semantics_have_evidence_modes(self):
        units = {unit['canonical_symbol']:unit for unit in self.catalog['capabilities']}
        expected = {
            'torch.nn.Module': {'torch.nn.Module', 'torch.nn.Sequential', 'torch.nn.ModuleList', 'torch.nn.ModuleDict'},
            'torch.utils.data.Dataset': {'torch.utils.data.Dataset','torch.utils.data.IterableDataset','torch.utils.data.TensorDataset','torch.utils.data.Subset'},
            'torch.save': {'torch.save','torch.load'},
            'torch.autograd.grad': {'torch.autograd.grad','torch.autograd.backward'},
            'torch.nn.init.xavier_uniform_': {'torch.nn.init.xavier_uniform_','torch.nn.init.kaiming_uniform_','torch.nn.init.normal_','torch.nn.init.zeros_'}}
        for symbol, sources in expected.items():
            unit = units[symbol]
            self.assertEqual({member['canonical_symbol'] for member in unit['source_members']}, sources)
            self.assertEqual(set(unit['mode_scope']),{mode for member in unit['source_members'] for mode in member['required_modes']})
            self.assertTrue(unit['required_semantics'])
        self.assertIn('bounded-iterable-source', units['torch.utils.data.Dataset']['mode_scope'])
        self.assertIn('continuation-save', units['torch.save']['mode_scope'])
        self.assertIn('continuation-restore', units['torch.save']['mode_scope'])
        self.assertIn('accumulated-gradients', units['torch.autograd.grad']['mode_scope'])
        self.assertIn('named-structure', units['torch.nn.Module']['mode_scope'])
        self.assertIn('kaiming-uniform', units['torch.nn.init.xavier_uniform_']['mode_scope'])

    def test_proposal_builder_refuses_approved_catalog_before_any_write(self):
        builder = module('proposed_builder', ROOT/'capabilities/build_proposed_catalogs.py')
        with tempfile.TemporaryDirectory() as directory:
            cap = Path(directory)
            (cap/'catalogs/v1').mkdir(parents=True)
            (cap/'selection-v1.json').write_text(json.dumps({'pytorch':self.selection}))
            target = cap/'catalogs/v1/pytorch.json'
            target.write_text(json.dumps(self.catalog))
            before = target.read_bytes()
            builder.CAP = cap
            with self.assertRaisesRegex(ValueError, 'Refusing to overwrite'): builder.main()
            self.assertEqual(target.read_bytes(),before)
            self.assertEqual(sorted(path.name for path in cap.iterdir()),['catalogs','selection-v1.json'])

if __name__ == '__main__': unittest.main()
