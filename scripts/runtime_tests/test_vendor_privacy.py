import importlib.util
import json
import plistlib
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def helper():
    spec = importlib.util.spec_from_file_location('vendor_privacy', ROOT / 'scripts/vendor_privacy.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def candidate(tmp_path):
    repo = tmp_path / 'repo'
    assets = repo / 'docs/production/vendor-privacy-20261009'
    shutil.copytree(ROOT / 'docs/production/vendor-privacy-20261009', assets)
    runtime = tmp_path / 'runtime'
    site = runtime / 'python/lib/python3.13/site-packages'
    for name, version in [('protobuf', '7.36.2'), ('grpcio', '1.84.0'), ('tensorflow', '2.21.0')]:
        folder = site / f'{name}-{version}.dist-info'
        folder.mkdir(parents=True)
        (folder / 'METADATA').write_text(f'Name: {name}\nVersion: {version}\n')
    for relative in ['python/lib/libpython3.13.dylib', 'python/lib/python3.13/site-packages/google/_upb/_message.abi3.so', 'python/lib/python3.13/site-packages/grpc/_cython/cygrpc.cpython-313-darwin.so', 'python/lib/python3.13/site-packages/tensorflow/libtensorflow_framework.2.dylib']:
        path = runtime / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'fixture native owner')
    headers = site / 'tensorflow/include'
    (headers / 'absl/base').mkdir(parents=True)
    (headers / 'absl/base/options.h').write_text('#define ABSL_OPTION_INLINE_NAMESPACE_NAME lts_20250814\n')
    (headers / 'google/protobuf').mkdir(parents=True)
    (headers / 'google/protobuf/empty.pb.h').write_text('#if PROTOBUF_VERSION != 6031001\n')
    return repo, runtime, tmp_path / 'VendorPrivacy'


def test_stage_exact_bundles_and_conditional_attribution(candidate):
    repo, runtime, dest = candidate
    result = helper().stage_vendor_privacy(repo, runtime, dest)
    assert len(result['bundles']) == 6
    for row in result['bundles']:
        bundle = dest / row['bundle']
        info = plistlib.loads((bundle / 'Contents/Info.plist').read_bytes())
        assert info['CFBundlePackageType'] == 'BNDL'
        assert 'CFBundleExecutable' not in info
        resources = bundle / 'Contents/Resources'
        assert (resources / 'PrivacyInfo.xcprivacy').read_bytes() == (repo / row['source_manifest']).read_bytes()
        attribution = json.loads((resources / 'PROVENANCE.json').read_text())
        assert attribution['vendor_signature_provided'] is False
        assert attribution['native_owners']
    boring = next(r for r in result['bundles'] if 'BoringSSL' in r['bundle'])
    assert boring['binding_status'] == 'conditional_release_dependency'
    assert 'installed_version' not in boring


def test_mutated_manifest_refused_without_output(candidate):
    repo, runtime, dest = candidate
    path = repo / 'docs/production/vendor-privacy-20261009/abseil-20250814.0/PrivacyInfo.xcprivacy'
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='digest'):
        helper().stage_vendor_privacy(repo, runtime, dest)
    assert not dest.exists()


def test_malformed_or_unknown_manifest_keys_refused():
    module = helper()
    with pytest.raises(ValueError):
        module.validate_manifest(b'not plist')
    with pytest.raises(ValueError, match='keys'):
        module.validate_manifest(plistlib.dumps({'NSPrivacyTracking': False, 'invented': []}))


def test_symlink_input_refused(candidate):
    repo, runtime, dest = candidate
    path = repo / 'docs/production/vendor-privacy-20261009/abseil-20250814.0/LICENSE.upstream'
    copied = path.with_name('outside')
    path.rename(copied)
    path.symlink_to(copied)
    with pytest.raises(ValueError, match='symlink'):
        helper().stage_vendor_privacy(repo, runtime, dest)
    assert not dest.exists()


def test_existing_destination_preserved(candidate):
    repo, runtime, dest = candidate
    dest.mkdir()
    (dest / 'sentinel').write_text('keep')
    with pytest.raises(FileExistsError):
        helper().stage_vendor_privacy(repo, runtime, dest)
    assert list(dest.iterdir()) == [dest / 'sentinel']
    assert (dest / 'sentinel').read_text() == 'keep'


def test_wrong_installed_version_refused(candidate):
    repo, runtime, dest = candidate
    metadata = runtime / 'python/lib/python3.13/site-packages/grpcio-1.84.0.dist-info/METADATA'
    metadata.write_text('Name: grpcio\nVersion: 9.0\n')
    with pytest.raises(ValueError, match='version'):
        helper().stage_vendor_privacy(repo, runtime, dest)
    assert not dest.exists()


def test_provenance_cannot_repin_changed_assets(candidate):
    repo, runtime, dest = candidate
    path = repo / 'docs/production/vendor-privacy-20261009/PROVENANCE.json'
    data = json.loads(path.read_text())
    data['artifacts'][0]['commit'] = '0' * 40
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='provenance digest'):
        helper().stage_vendor_privacy(repo, runtime, dest)
    assert not dest.exists()


def test_native_owner_symlink_refused(candidate):
    repo, runtime, dest = candidate
    owner = runtime / 'python/lib/libpython3.13.dylib'
    target = owner.with_suffix('.copy')
    owner.rename(target)
    owner.symlink_to(target)
    with pytest.raises(ValueError, match='symlink'):
        helper().stage_vendor_privacy(repo, runtime, dest)
    assert not dest.exists()


def test_staging_failure_cleans_partial_output(candidate, monkeypatch):
    repo, runtime, dest = candidate
    original = Path.write_bytes
    def fail_bundle(self, data):
        if self.name == 'PrivacyInfo.xcprivacy' and '.vendor-privacy-' in str(self):
            raise OSError('simulated disk failure')
        return original(self, data)
    monkeypatch.setattr(Path, 'write_bytes', fail_bundle)
    with pytest.raises(OSError, match='disk failure'):
        helper().stage_vendor_privacy(repo, runtime, dest)
    assert not dest.exists()
    assert not list(dest.parent.glob('.vendor-privacy-*'))


def test_atomic_publication_refuses_empty_destination(tmp_path):
    staging, destination = tmp_path / 'stage', tmp_path / 'existing'
    staging.mkdir()
    (staging / 'new').write_text('staged')
    destination.mkdir()
    with pytest.raises(FileExistsError):
        helper().publish_exclusive(staging, destination)
    assert destination.is_dir() and not list(destination.iterdir())
    assert (staging / 'new').read_text() == 'staged'


def test_verify_existing_preserves_exact_candidate(candidate):
    repo, runtime, dest = candidate
    module = helper()
    module.stage_vendor_privacy(repo, runtime, dest)
    before = {str(p.relative_to(dest)): (p.read_bytes(), p.stat().st_mtime_ns) for p in dest.rglob('*') if p.is_file()}
    result = module.verify_vendor_privacy(repo, runtime, dest)
    assert len(result['bundles']) == 6
    after = {str(p.relative_to(dest)): (p.read_bytes(), p.stat().st_mtime_ns) for p in dest.rglob('*') if p.is_file()}
    assert before == after


@pytest.mark.parametrize('damage', ['missing', 'stale', 'extra', 'symlink', 'extra-directory'])
def test_verify_existing_refuses_damage_without_mutation(candidate, damage):
    repo, runtime, dest = candidate
    module = helper()
    module.stage_vendor_privacy(repo, runtime, dest)
    manifest = dest / 'AbseilPrivacy.bundle/Contents/Resources/PrivacyInfo.xcprivacy'
    if damage == 'missing':
        manifest.unlink()
    elif damage == 'stale':
        manifest.write_bytes(b'stale')
    elif damage == 'extra':
        (dest / 'extra').write_text('unexpected')
    elif damage == 'extra-directory':
        (dest / 'extra').mkdir()
    else:
        manifest.unlink()
        manifest.symlink_to(repo / 'docs/production/vendor-privacy-20261009/abseil-20250814.0/PrivacyInfo.xcprivacy')
    before = [(str(p.relative_to(dest)), p.lstat().st_mtime_ns) for p in dest.rglob('*')]
    with pytest.raises(ValueError):
        module.verify_vendor_privacy(repo, runtime, dest)
    assert before == [(str(p.relative_to(dest)), p.lstat().st_mtime_ns) for p in dest.rglob('*')]


def test_verify_existing_missing_destination_does_not_create(candidate):
    repo, runtime, dest = candidate
    with pytest.raises(ValueError):
        helper().verify_vendor_privacy(repo, runtime, dest)
    assert not dest.exists()
