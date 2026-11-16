import importlib.util
from pathlib import Path
import tempfile
import unittest
import zlib

SCRIPT=Path(__file__).resolve().parents[1]/'capture_inventory.py'
spec=importlib.util.spec_from_file_location('capture_inventory',SCRIPT)
capture=importlib.util.module_from_spec(spec);spec.loader.exec_module(capture)

class CaptureTests(unittest.TestCase):
    def test_sphinx_only_python_framework_objects_and_expands_anchor(self):
        payload='torch.nn.Linear py:class 1 generated/torch.nn.Linear.html#$ -\nOther title std:label -1 ignored.html -\nother.Package py:class 1 other.html -\n'
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'objects.inv'
            path.write_bytes(b'# Sphinx inventory version 2\n# Project: PyTorch\n# Version: 2.14\n# zlib\n'+zlib.compress(payload.encode()))
            version,rows=capture.sphinx(path,'https://docs.pytorch.org/docs/2.14/','torch')
        self.assertEqual(version,'2.14')
        self.assertEqual(len(rows),1)
        self.assertTrue(rows[0]['doc_url'].endswith('#torch.nn.Linear'))

    def test_exact_documentation_target_alias_is_not_new_operation(self):
        rows=capture.normalize([
            {'symbol':'lib.Op','kind':'py:class','doc_url':'https://example/api#lib.Op'},
            {'symbol':'lib.Alias','kind':'py:class','doc_url':'https://example/api#lib.Op'}])
        alias=next(row for row in rows if row['symbol']=='lib.Alias')
        self.assertEqual(alias['disposition'],'alias')
        self.assertEqual(alias['canonical_symbol'],'lib.Op')

    def test_tensorflow_links_ignore_other_hosts_scripts_and_duplicate_navigation(self):
        parser=capture.TFLinks()
        parser.feed('<a href="/api_docs/python/tf/keras/layers/Dense">Dense</a><a href="https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense">same</a><a href="https://evil.example/api_docs/python/tf/Fake">bad</a><script>instructions</script>')
        self.assertEqual(parser.links,{'/api_docs/python/tf/keras/layers/Dense'})

    def test_private_and_legacy_exclusions_remain_auditable(self):
        rows=capture.normalize([{'symbol':name,'kind':'documented-page','doc_url':'https://example/'+name} for name in ['tf.raw_ops.Add','tf.compat.v1.Session','torch._internal']])
        self.assertTrue(all(row['disposition']=='excluded' and row['reason'] for row in rows))
        self.assertEqual(len(rows),3)

if __name__=='__main__':unittest.main()
