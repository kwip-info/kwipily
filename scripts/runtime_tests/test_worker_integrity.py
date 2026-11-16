import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import pytest

SCRIPTS=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('package_integrity',SCRIPTS/'runtime_package.py')
package=importlib.util.module_from_spec(spec);spec.loader.exec_module(package)

def bundle(tmp_path):
    root=tmp_path/'runtime';(root/'runner').mkdir(parents=True)
    shutil.copy2(SCRIPTS/'runtime_launch.py',root/'launch.py')
    (root/'components.json').write_text('{}')
    (root/'runner/main.py').write_text('print(\'{"event":"workerExecuted"}\')\n')
    (root/'runner/example.csv').write_text('x\n1\n')
    files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
    (root/'runtime-manifest.json').write_text(json.dumps({'schemaVersion':1,'workerIntegrityVersion':1,'workerSourceHashes':files}))
    return root

def launch(root):
    return subprocess.run([sys.executable,'-I','-B',str(root/'launch.py')],capture_output=True,text=True,timeout=10)

def test_packager_inventories_worker_resources_and_bootstrap(tmp_path):
    root=bundle(tmp_path)
    actual=package.worker_source_hashes(root)
    assert set(actual)=={'launch.py','components.json','runner/main.py','runner/example.csv'}
    assert actual['runner/main.py']==hashlib.sha256((root/'runner/main.py').read_bytes()).hexdigest()

@pytest.mark.parametrize('mutation',['changed','missing','extra','symlink','manifestMissing','manifestLegacy','pathEscape','badDigest','oversized'])
def test_invalid_bundle_never_imports_worker(tmp_path,mutation):
    root=bundle(tmp_path);manifest=root/'runtime-manifest.json'
    if mutation=='changed':(root/'runner/main.py').write_text('print("UNTRUSTED_EXECUTION")')
    elif mutation=='missing':(root/'runner/example.csv').unlink()
    elif mutation=='extra':(root/'runner/new.py').write_text('')
    elif mutation=='symlink':
        original=root/'runner/example.csv';original.unlink();original.symlink_to(root/'components.json')
    elif mutation=='manifestMissing':manifest.unlink()
    elif mutation=='manifestLegacy':manifest.write_text('{"schemaVersion":1}')
    elif mutation=='oversized':manifest.write_bytes(b' '*65537)
    else:
        value=json.loads(manifest.read_text())
        if mutation=='pathEscape':value['workerSourceHashes']['../escape']='a'*64
        else:value['workerSourceHashes']['runner/main.py']='invalid'
        manifest.write_text(json.dumps(value))
    p=launch(root)
    assert p.returncode==1,(mutation,p.stdout,p.stderr)
    event=json.loads(p.stdout);assert event['event']=='error' and 'integrity' in event['message'].lower()
    assert 'workerExecuted' not in p.stdout and 'UNTRUSTED_EXECUTION' not in p.stdout

def test_valid_bundle_runs_after_relocation_without_manifest_writes(tmp_path):
    root=bundle(tmp_path);before=(root/'runtime-manifest.json').read_bytes()
    moved=tmp_path/'other place';root.rename(moved)
    p=launch(moved);assert p.returncode==0,p.stderr+p.stdout
    assert json.loads(p.stdout)['event']=='workerExecuted'
    assert (moved/'runtime-manifest.json').read_bytes()==before


def test_build_refresh_updates_changed_files_and_consistent_metadata(tmp_path):
    root=bundle(tmp_path)
    (root/'runner/example.csv').write_text('x\n2\n')
    assert launch(root).returncode==1
    manifest=package.refresh_worker_integrity(root)
    assert manifest['workerIntegrityVersion']==1
    assert manifest['entryHash']==manifest['workerSourceHashes']['launch.py']
    assert manifest['registryHash']==manifest['workerSourceHashes']['components.json']
    assert launch(root).returncode==0


def test_packaged_registry_cannot_be_overridden_outside_bundle(tmp_path):
    root=bundle(tmp_path);external=tmp_path/'external.json';external.write_text('{}')
    p=subprocess.run([sys.executable,'-I','-B',str(root/'launch.py'),'--registry',str(external)],capture_output=True,text=True)
    assert p.returncode==1 and 'registry' in json.loads(p.stdout)['message'].lower()


def test_refresh_cannot_attest_an_old_unenforcing_bootstrap(tmp_path):
    root=bundle(tmp_path);(root/'launch.py').write_text('print("old bootstrap")')
    with pytest.raises(ValueError,match='bootstrap'):package.refresh_worker_integrity(root)


def source_for(root,tmp_path,monkeypatch):
    if sys.platform!='darwin' or __import__('platform').machine()!='arm64':
        pytest.skip('Canonical native packaging requires arm64 macOS')
    source=tmp_path/'source'
    (source/'scripts').mkdir(parents=True)
    (source/'Sources/KwipMLCore/Resources').mkdir(parents=True)
    shutil.copytree(root/'runner',source/'runner')
    shutil.copy2(root/'launch.py',source/'scripts/runtime_launch.py')
    shutil.copy2(root/'components.json',source/'Sources/KwipMLCore/Resources/components.json')
    for base in (source,root):
        (base/'runner/native').mkdir()
        shutil.copy2(SCRIPTS.parent/'runner/native/lifetime_watchdog.c',base/'runner/native/lifetime_watchdog.c')
    monkeypatch.setattr(package,'ROOT',source)
    package.build_native_watchdog(root)
    package.refresh_worker_integrity(root)
    return source


def test_build_verifier_matches_current_sources_without_mutating_payload(tmp_path,monkeypatch):
    root=bundle(tmp_path);source=source_for(root,tmp_path,monkeypatch)
    (source/'runner/tests').mkdir();(source/'runner/tests/test_example.py').write_text('test only')
    (source/'runner/__pycache__').mkdir();(source/'runner/__pycache__/ignored.pyc').write_bytes(b'cache')
    before={p.relative_to(root):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    result=package.verify_current_worker(root)
    assert result['status']=='matches-current-source'
    assert result['files']==6
    assert before=={p.relative_to(root):p.read_bytes() for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('mutation',['oldWorker','oldRegistry','extraWorker','missingWorker','unattested'])
def test_build_verifier_rejects_stale_even_when_rehashed(tmp_path,monkeypatch,mutation):
    root=bundle(tmp_path);source_for(root,tmp_path,monkeypatch)
    if mutation in ('oldWorker','unattested'):(root/'runner/main.py').write_text('old worker')
    elif mutation=='oldRegistry':(root/'components.json').write_text('{"old":true}')
    elif mutation=='extraWorker':(root/'runner/obsolete.py').write_text('unused')
    else:(root/'runner/example.csv').unlink()
    if mutation!='unattested':package.refresh_worker_integrity(root)
    before=(root/'runtime-manifest.json').read_bytes()
    with pytest.raises(ValueError):package.verify_current_worker(root)
    assert (root/'runtime-manifest.json').read_bytes()==before

@pytest.mark.parametrize('stale',[False,True])
def test_packaged_build_checks_worker_before_native_build(tmp_path,monkeypatch,stale):
    import os
    zsh=shutil.which('zsh')
    if not zsh:pytest.skip('Native packaging entrypoint requires zsh')
    root=bundle(tmp_path);source=source_for(root,tmp_path,monkeypatch)
    shutil.copy2(SCRIPTS/'runtime_package.py',source/'scripts/runtime_package.py')
    shutil.copy2(SCRIPTS/'build-packaged-app.sh',source/'scripts/build-packaged-app.sh')
    marker=source/'native-build-started'
    native=source/'scripts/build-app.sh'
    native.write_text('#!/bin/zsh\ntouch "${0:A:h:h}/native-build-started"\nexit 73\n')
    native.chmod(0o755)
    if stale:
        (root/'runner/main.py').write_text('old worker\n')
        package.refresh_worker_integrity(root)
    before=(root/'runtime-manifest.json').read_bytes()
    result=subprocess.run([zsh,str(source/'scripts/build-packaged-app.sh'),str(root)],
        env={**os.environ,'KWIPML_BUILD_PYTHON':sys.executable},capture_output=True,text=True,timeout=15)
    assert (root/'runtime-manifest.json').read_bytes()==before
    if stale:
        assert not marker.exists(),'A stale worker reached native build before rejection'
        assert result.returncode not in (0,73)
    else:
        assert marker.exists()
        assert result.returncode==73
        assert 'matches-current-source' in result.stdout


def test_packaged_build_rechecks_copy_after_native_build(tmp_path,monkeypatch):
    import os
    zsh=shutil.which('zsh')
    if not zsh or not Path('/usr/bin/ditto').exists():pytest.skip('macOS packaging tools required')
    root=bundle(tmp_path);source=source_for(root,tmp_path,monkeypatch)
    for name in ['runtime_package.py','build-packaged-app.sh','vendor_privacy.py']:
        shutil.copy2(SCRIPTS/name,source/'scripts'/name)
    # Exercise the real newly required privacy helper, not a bypass stub. These
    # synthetic native-owner/header bindings are resources only; no SDK runs.
    shutil.copytree(SCRIPTS.parent/'docs/production/vendor-privacy-20261009',
                    source/'docs/production/vendor-privacy-20261009')
    site=root/'python/lib/python3.13/site-packages'
    for name,version in [('protobuf','7.36.2'),('grpcio','1.84.0'),('tensorflow','2.21.0')]:
        metadata=site/f'{name}-{version}.dist-info/METADATA'
        metadata.parent.mkdir(parents=True,exist_ok=True)
        metadata.write_text(f'Name: {name}\nVersion: {version}\n')
    for relative in ['python/lib/libpython3.13.dylib',
                     'python/lib/python3.13/site-packages/google/_upb/_message.abi3.so',
                     'python/lib/python3.13/site-packages/grpc/_cython/cygrpc.cpython-313-darwin.so',
                     'python/lib/python3.13/site-packages/tensorflow/libtensorflow_framework.2.dylib']:
        owner=root/relative;owner.parent.mkdir(parents=True,exist_ok=True)
        owner.write_bytes(b'synthetic privacy owner, never executed')
    for relative,value in [('absl/base/options.h','lts_20250814'),
                           ('google/protobuf/empty.pb.h','PROTOBUF_VERSION != 6031001')]:
        header=site/'tensorflow/include'/relative;header.parent.mkdir(parents=True,exist_ok=True)
        header.write_text(value)
    candidate=source/'.local/KwipML Candidate.app'
    candidate.mkdir(parents=True)
    (candidate/'retained').write_text('previous candidate')
    native=source/'scripts/build-app.sh'
    native.write_text('#!/bin/zsh\nmkdir -p "${0:A:h:h}/.local/KwipML POC.app/Contents/Resources"\nprintf changed > "$KWIP_TEST_RUNTIME/runner/main.py"\n')
    native.chmod(0o755)
    result=subprocess.run([zsh,str(source/'scripts/build-packaged-app.sh'),str(root)],
        env={**os.environ,'KWIPML_BUILD_PYTHON':sys.executable,'KWIP_TEST_RUNTIME':str(root)},capture_output=True,text=True,timeout=15)
    assert result.returncode!=0
    assert 'runtime_package.py' in result.stderr and 'ValueError' in result.stderr
    assert 'worker files are missing, changed or unlisted' in result.stderr
    assert 'Staged 6 attributed vendor privacy resource bundles' in result.stdout
    assert (candidate/'retained').read_text()=='previous candidate'
    assert not list((source/'.local').glob('packaged-stage-*'))
