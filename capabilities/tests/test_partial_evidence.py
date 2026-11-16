import hashlib
import importlib.util
import json
import subprocess
import tempfile
import unittest
from unittest import mock
import os
from pathlib import Path

SPEC = importlib.util.spec_from_file_location('partial', Path(__file__).resolve().parents[2] / 'scripts/verify_partial_evidence.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

class PartialEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git('init', '-q')
        self.git('config', 'user.email', 'fixture@example.invalid')
        self.git('config', 'user.name', 'Fixture')
        for name in ['source.py', 'scope.md', 'registry.json', 'descriptor.json', 'report.json']:
            (self.root / name).write_text(name)
        self.git('add', '.')
        self.git('commit', '-qm', 'fixture')
        self.revision = self.git('rev-parse', 'HEAD').strip()
        def ref(path):
            return {'path': path, 'sha256': hashlib.sha256((self.root/path).read_bytes()).hexdigest()}
        self.manifest = {
            'schemaVersion': 1, 'capabilityID': 'pytorch.torch.nn.Linear',
            'evidenceRevision': 'fixture-1', 'status': 'partial', 'catalogVersion': '0.1.0',
            'implementation': {'gitCommit': self.revision, 'sourceFiles': [ref('source.py')]},
            'acceptanceScope': ref('scope.md'),
            'contracts': {'registry': ref('registry.json'), 'executionDescriptor': ref('descriptor.json')},
            'environment': {'python': '3.13.6', 'torch': '2.14.1', 'macOS': 'fixture', 'architecture': 'arm64', 'availableDevices': ['cpu']},
            'upstream': {'documentationVersion': '2.14', 'alignmentStatus': 'unreviewed'},
            'observations': [{'id': 'fixture-case', 'kind': 'automated', 'modes': ['inference'], 'dimensions': ['execution'], 'cases': ['cpu-forward'], 'outcome': 'passed', 'commandOrProcedure': 'fixture only', 'implementationRevision': self.revision, 'evidenceFiles': [ref('report.json')]}],
            'openRequirements': ['Native acceptance remains pending'], 'review': {'status': 'pending'}
        }

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.root), *args], text=True)

    def verify(self):
        return MODULE.verify(self.manifest, self.root)

    def test_historical_source_survives_checkout_change(self):
        (self.root/'source.py').write_text('new checkout source')
        result = self.verify()
        self.assertTrue(result['integrityValid'])
        self.assertFalse(result['accepted'])
        self.assertEqual(result['coverageCredit'], 0)

    def test_tampered_historical_source_hash_and_revision_fail(self):
        self.manifest['implementation']['sourceFiles'][0]['sha256'] = '0'*64
        with self.assertRaises(ValueError): self.verify()
        self.manifest['implementation']['gitCommit'] = 'HEAD'
        with self.assertRaises(ValueError): self.verify()

    def test_missing_changed_and_escaping_artifacts_fail(self):
        for path in ['../report.json', '/report.json', 'absent.json']:
            self.manifest['observations'][0]['evidenceFiles'][0]['path'] = path
            with self.assertRaises(ValueError): self.verify()
        self.manifest['observations'][0]['evidenceFiles'][0]['path'] = 'report.json'
        (self.root/'report.json').write_text('changed')
        with self.assertRaises(ValueError): self.verify()

    def test_revision_and_native_candidate_binding_required(self):
        observation = self.manifest['observations'][0]
        observation['implementationRevision'] = '0'*40
        with self.assertRaises(ValueError): self.verify()
        observation['implementationRevision'] = self.revision
        observation['kind'] = 'nativeUAT'
        with self.assertRaises(ValueError): self.verify()

    def test_duplicates_skips_and_completed_claims(self):
        self.manifest['observations'].append(dict(self.manifest['observations'][0]))
        with self.assertRaises(ValueError): self.verify()
        self.manifest['observations'].pop()
        self.manifest['observations'][0]['outcome'] = 'skipped'
        with self.assertRaises(ValueError): self.verify()
        self.manifest['observations'][0]['reason'] = 'No matching hardware'
        self.assertEqual(self.verify()['observationCounts']['skipped'], 1)
        self.manifest['status'] = 'complete'
        with self.assertRaises(ValueError): self.verify()

    def test_invented_modes_and_dimensions_do_not_enter_evidence(self):
        self.manifest['observations'][0]['modes'] = ['everything']
        with self.assertRaises(ValueError): self.verify()
        self.manifest['observations'][0]['modes'] = ['inference']
        self.manifest['observations'][0]['dimensions'] = ['looks_complete']
        with self.assertRaises(ValueError): self.verify()

    def test_unknown_fields_schema_and_missing_environment_rejected(self):
        self.manifest['unexpected'] = True
        with self.assertRaises(ValueError): self.verify()
        del self.manifest['unexpected']
        self.manifest['schemaVersion'] = True
        with self.assertRaises(ValueError): self.verify()
        self.manifest['schemaVersion'] = 1
        del self.manifest['environment']['torch']
        with self.assertRaises(ValueError): self.verify()


    def test_git_replace_cannot_rebind_historical_commit(self):
        replacement=b'replacement source'
        (self.root/'source.py').write_bytes(replacement)
        self.git('add','source.py');self.git('commit','-qm','replacement')
        new=self.git('rev-parse','HEAD').strip()
        self.git('replace',self.revision,new)
        self.manifest['implementation']['sourceFiles'][0]['sha256']=hashlib.sha256(replacement).hexdigest()
        with self.assertRaises(ValueError):self.verify()

    def test_empty_present_commit_rejected(self):
        self.manifest['observations'][0]['evidenceFiles'][0]['gitCommit']=''
        with self.assertRaises(ValueError):self.verify()

    def test_growth_between_size_check_and_read_rejected(self):
        target=self.root/'report.json';large=b'x'*(MODULE.MAX_BYTES+1)
        self.manifest['observations'][0]['evidenceFiles'][0]['sha256']=hashlib.sha256(large).hexdigest()
        original=Path.read_bytes
        def growing(path):
            if path.resolve()==target.resolve():path.write_bytes(large)
            return original(path)
        with mock.patch.object(Path,'read_bytes',growing):
            with self.assertRaises(ValueError):self.verify()

    def test_symlink_swap_at_read_cannot_match_outside_artifact(self):
        target=self.root/'report.json';outside=self.root.parent/(self.root.name+'-outside')
        outside.write_bytes(b'outside');self.addCleanup(lambda:outside.unlink(missing_ok=True))
        self.manifest['observations'][0]['evidenceFiles'][0]['sha256']=hashlib.sha256(b'outside').hexdigest()
        original=Path.read_bytes
        def swap(path):
            if path.resolve()==target.resolve():path.unlink();path.symlink_to(outside)
            return original(path)
        with mock.patch.object(Path,'read_bytes',swap):
            with self.assertRaises(ValueError):self.verify()

    def test_malformed_outcome_cli_returns_json_exit_two(self):
        self.manifest['observations'][0]['outcome']=[]
        path=self.root/'manifest.json';path.write_text(json.dumps(self.manifest))
        import sys
        result=subprocess.run([sys.executable,str(Path(MODULE.__file__)),str(path),'--root',str(self.root)],capture_output=True,text=True)
        self.assertEqual(result.returncode,2,result.stderr)
        self.assertFalse(json.loads(result.stdout)['integrityValid'])


    def test_descriptor_read_growth_is_bounded(self):
        target=self.root/'report.json';inode=target.stat().st_ino
        original=os.read;seen=0;grown=False
        def grow(descriptor,count):
            nonlocal seen,grown
            if os.fstat(descriptor).st_ino==inode:
                if not grown:target.write_bytes(b'x'*(MODULE.MAX_BYTES+100));grown=True
                result=original(descriptor,count);seen+=len(result);return result
            return original(descriptor,count)
        with mock.patch.object(MODULE.os,'read',grow):
            with self.assertRaisesRegex(ValueError,'size bound'):self.verify()
        self.assertLessEqual(seen,MODULE.MAX_BYTES+1)
        self.assertTrue(grown)

    def test_descriptor_open_rejects_leaf_swap_and_parent_symlink(self):
        target=self.root/'report.json';outside=self.root/'outside.json';outside.write_text('outside')
        original=os.open
        def swap(path,flags,*args,**kwargs):
            if path=='report.json':target.unlink();target.symlink_to(outside)
            return original(path,flags,*args,**kwargs)
        with mock.patch.object(MODULE.os,'open',swap):
            with self.assertRaises(ValueError):self.verify()
        target.unlink();target.write_text('report.json')
        (self.root/'linked').symlink_to(self.root,target_is_directory=True)
        self.manifest['observations'][0]['evidenceFiles'][0]['path']='linked/report.json'
        with self.assertRaises(ValueError):self.verify()

    def test_fifo_artifact_rejected_without_blocking(self):
        target=self.root/'report.json';target.unlink();os.mkfifo(target)
        with self.assertRaises(ValueError):self.verify()

if __name__ == '__main__': unittest.main()
