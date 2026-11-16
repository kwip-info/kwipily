"""Portable optional helper integrity; no execution of helper source is authorized here."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[2] / 'Sources/KwipMLApp/Resources/inference.py'

class InitializationPortableInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        spec = importlib.util.spec_from_file_location('kwip_initialization_inference_inventory', SOURCE)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.module.ROOT = self.root
        self.files = {name: b'# bounded synthetic inventory fixture\n' for name in ('model.py', 'checkpoint.pt', 'inference.py', 'input-contract.json')}
    def publish(self):
        entries = []
        for name, data in self.files.items():
            (self.root / name).write_bytes(data)
            entries.append({'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
        (self.root / 'export-manifest.json').write_text(json.dumps({'schemaVersion': 1, 'kind': 'kwipmlTrainedModel', 'files': entries}))
    def test_optional_initialization_helper_is_hashed_bounded_utf8_but_never_executed(self):
        # This source must never be imported during inventory verification.
        self.files['initialization_recipe.py'] = b'raise RuntimeError("never execute a fresh initializer during inference")\n'
        self.publish()
        actual = self.module.verified_files()
        self.assertEqual(actual['initialization_recipe.py'], self.files['initialization_recipe.py'])
    def test_changed_helper_and_repinned_invalid_utf8_oversize_or_empty_refuse(self):
        self.files['initialization_recipe.py'] = b'# helper\n'
        self.publish()
        (self.root / 'initialization_recipe.py').write_bytes(b'# changed\n')
        with self.assertRaises(ValueError):
            self.module.verified_files()
        for invalid in (b'\xff', b'A' * 131073, b''):
            self.files['initialization_recipe.py'] = invalid
            self.publish()
            with self.assertRaises(ValueError):
                self.module.verified_files()
    def test_absent_policy_inventory_stays_accepted_and_old_recipe_bound_does_not_expand(self):
        self.publish()
        self.assertEqual(set(self.module.verified_files()), set(self.files))
        self.files['training_recipe.py'] = b'A' * 8193
        self.publish()
        with self.assertRaises(ValueError):
            self.module.verified_files()

if __name__ == '__main__':
    unittest.main()
