"""Production signing boundaries, using mocked platform tools only."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[2]
def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

audit = load('runtime_audit')
package = load('runtime_package')
TEAM = 'ABCDEFGHIJ'
IDENTITY = 'Apple Distribution: KWIP LLC (ABCDEFGHIJ)'

@pytest.mark.parametrize('identity,team', [
    ('Apple Development: KWIP LLC (ABCDEFGHIJ)', TEAM),
    ('Developer ID Application: KWIP LLC (ABCDEFGHIJ)', TEAM),
    (IDENTITY, 'ZZZZZZZZZZ'), ('-', TEAM), (IDENTITY, None)])
def test_distribution_selection_rejects_non_mas_and_wrong_team(identity, team):
    with pytest.raises(ValueError): audit.validate_distribution_selection(identity, team)


def test_signature_requires_exact_authority_and_team(tmp_path, monkeypatch):
    path = tmp_path / 'binary'; path.write_bytes(b'native')
    result = SimpleNamespace(returncode=0, stdout='', stderr=f'Authority={IDENTITY}\nTeamIdentifier={TEAM}\n')
    monkeypatch.setattr(audit.subprocess, 'run', lambda *a, **k: result)
    assert audit.verify_distribution_signature(path, IDENTITY, TEAM)['teamID'] == TEAM
    result.stderr = f'Authority=Apple Development: KWIP LLC ({TEAM})\nTeamIdentifier={TEAM}\n'
    with pytest.raises(ValueError, match='identity'): audit.verify_distribution_signature(path, IDENTITY, TEAM)
    result.stderr = f'Authority={IDENTITY}\nTeamIdentifier=ZZZZZZZZZZ\n'
    with pytest.raises(ValueError, match='team'): audit.verify_distribution_signature(path, IDENTITY, TEAM)


def test_distribution_audit_signs_dependencies_but_preserves_sealed_watchdog(tmp_path, monkeypatch):
    paths = [tmp_path / 'python/lib/dependency.dylib', tmp_path / 'runner/native/lifetime_watchdog.dylib']
    for path in paths: path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'\xcf\xfa\xed\xfeoriginal')
    calls = []
    monkeypatch.setattr(audit.subprocess, 'check_output', lambda args, **kw: str(args[-1]) + ':\n' if args[1] in ['-D', '-L'] else ' minos 14.0\n')
    def run(args, **kw):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout='', stderr=f'Authority={IDENTITY}\nTeamIdentifier={TEAM}\n')
    monkeypatch.setattr(audit.subprocess, 'run', run)
    result = audit.audit(tmp_path, signing_identity=IDENTITY, team_id=TEAM)
    signing = [args for args in calls if '--force' in args]
    assert len(signing) == 1 and signing[0][-1] == str(paths[0])
    assert signing[0][signing[0].index('--sign') + 1] == IDENTITY
    assert paths[1].read_bytes().endswith(b'original')
    assert all(row['distributionSignature']['teamID'] == TEAM for row in result['machOFiles'])


def test_watchdog_signs_selected_identity_before_inventory(tmp_path, monkeypatch):
    source = tmp_path / 'runner/native/lifetime_watchdog.c'; source.parent.mkdir(parents=True); source.write_text('source')
    monkeypatch.setattr(package.sys, 'platform', 'darwin')
    monkeypatch.setattr(package.os, 'uname', lambda: SimpleNamespace(machine='arm64'))
    monkeypatch.setattr(package, 'worker_source_hashes', lambda root: {})
    calls = []
    def run(args, **kw):
        calls.append(args)
        if args[0].endswith('xcrun'): Path(args[-1]).write_bytes(b'compiled')
        return SimpleNamespace(returncode=0, stdout='', stderr=f'Authority={IDENTITY}\nTeamIdentifier={TEAM}\n')
    monkeypatch.setattr(package.subprocess, 'run', run)
    package.build_native_watchdog(tmp_path, signing_identity=IDENTITY, team_id=TEAM)
    signing = next(args for args in calls if '--force' in args)
    assert signing[signing.index('--sign') + 1] == IDENTITY
    assert calls[0][0].endswith('xcrun')


@pytest.mark.parametrize('changed', ['none', 'source', 'native'])
def test_production_worker_compares_static_and_unsigned_generated_code(tmp_path, monkeypatch, changed):
    actual = tmp_path / 'runtime'; actual.mkdir()
    native = 'runner/native/lifetime_watchdog.dylib'
    (actual / 'runner/native').mkdir(parents=True)
    (actual / native).write_bytes(b'CODE-PRODUCTION-SIGNATURE' if changed != 'native' else b'FOREIGN-PRODUCTION-SIGNATURE')
    (actual / 'launch.py').write_bytes(b'changed' if changed == 'source' else b'source')
    original = (actual / native).read_bytes()
    monkeypatch.setattr(package.integrity, 'validate_worker', lambda root: None)
    monkeypatch.setattr(package, 'worker_source_hashes', lambda root: {name: package.digest(Path(root)/name) for name in [native, 'launch.py']})
    def copy(root, **kw):
        path = Path(root); (path / 'runner/native').mkdir(parents=True)
        (path / native).write_bytes(b'CODE-ADHOC-SIGNATURE'); (path / 'launch.py').write_bytes(b'source')
    monkeypatch.setattr(package, 'copy_worker_sources', copy)
    monkeypatch.setattr(package.distribution_audit, 'verify_distribution_signature', lambda *a: {'teamID':TEAM})
    def run(args, **kw):
        assert '--remove-signature' in args
        path = Path(args[-1]); path.write_bytes(path.read_bytes().split(b'-')[0])
        assert not path.is_relative_to(actual)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(package.subprocess, 'run', run)
    if changed == 'none': assert package.verify_production_worker(actual, IDENTITY, TEAM)['status'] == 'matches-current-source'
    else:
        with pytest.raises(ValueError, match='differs'): package.verify_production_worker(actual, IDENTITY, TEAM)
    assert (actual / native).read_bytes() == original


def test_fingerprint_verification_extracts_actual_leaf_and_rejects_mismatch(tmp_path, monkeypatch):
    import hashlib
    leaf = b'exact certificate DER'
    identity = hashlib.sha1(leaf).hexdigest().upper()
    calls = []
    def run(args, **kw):
        calls.append(args)
        extraction = [arg for arg in args if arg.startswith('--extract-certificates=')]
        if '--extract-certificates' in args: raise ValueError('codesign optional prefix must use equals syntax')
        if extraction:
            Path(extraction[0].split('=',1)[1]+'0').write_bytes(leaf)
        return SimpleNamespace(returncode=0,stdout='',stderr=f'Authority={IDENTITY}\nTeamIdentifier={TEAM}\n')
    monkeypatch.setattr(audit.subprocess,'run',run)
    assert audit.verify_distribution_signature(tmp_path/'binary',identity,TEAM)['identity'] == identity
    assert any(any(arg.startswith('--extract-certificates=') for arg in args) for args in calls)
    with pytest.raises(ValueError,match='fingerprint'):
        audit.verify_distribution_signature(tmp_path/'binary','F'*40,TEAM)


def test_required_identity_audit_never_signs_or_mutates_manifest(tmp_path,monkeypatch):
    path=tmp_path/'runner/native/library.dylib';path.parent.mkdir(parents=True);path.write_bytes(b'\xcf\xfa\xed\xfeoriginal')
    manifest=tmp_path/'runtime-manifest.json';manifest.write_text('{"signing":"preserve"}')
    calls=[]
    monkeypatch.setattr(audit.subprocess,'check_output',lambda args,**kw:str(path)+':\n')
    def run(args,**kw):
        calls.append(args)
        return SimpleNamespace(returncode=0,stdout='',stderr=f'Authority={IDENTITY}\nTeamIdentifier={TEAM}\n')
    monkeypatch.setattr(audit.subprocess,'run',run)
    audit.audit(tmp_path,require_identity=IDENTITY,require_team=TEAM,require_signatures=True)
    assert not any('--force' in args or args[0].endswith('install_name_tool') for args in calls)
    assert manifest.read_text() == '{"signing":"preserve"}'
    assert path.read_bytes().endswith(b'original')


@pytest.mark.parametrize('selected',['python/bin/python3','python/bin/python'])
def test_worker_entitlements_sign_manifest_executable_including_internal_symlink(tmp_path,monkeypatch,selected):
    import json,plistlib
    path=tmp_path/'python/bin/python3';path.parent.mkdir(parents=True);path.write_bytes(b'\xcf\xfa\xed\xfeoriginal')
    (path.parent/'python').symlink_to('python3')
    (tmp_path/'runtime-manifest.json').write_text(json.dumps({'pythonExecutable':selected}))
    entitlements=tmp_path/'Worker.entitlements';entitlements.write_bytes(plistlib.dumps({'com.apple.security.app-sandbox':True,'com.apple.security.inherit':True}))
    calls=[]
    monkeypatch.setattr(audit.subprocess,'check_output',lambda args,**kw:str(args[-1])+':\n')
    def run(args,**kw):
        calls.append(args)
        return SimpleNamespace(returncode=0,stdout='',stderr=f'Authority={IDENTITY}\nTeamIdentifier={TEAM}\n')
    monkeypatch.setattr(audit.subprocess,'run',run)
    audit.audit(tmp_path,signing_identity=IDENTITY,team_id=TEAM,worker_entitlements=entitlements)
    signed=[args for args in calls if '--force' in args]
    assert len(signed)==1 and signed[0][-1]==str(path.resolve()) and '--entitlements' in signed[0]


def test_all_nested_executable_helpers_receive_inherited_sandbox(tmp_path, monkeypatch):
    import json, plistlib
    names = ['python/bin/python3.13', 'python/bin/python3', 'python/bin/python',
             'python/lib/site-packages/torch/bin/protoc',
             'python/lib/site-packages/torch/bin/torch_shm_manager', 'python/lib/library.dylib']
    for name in names:
        path = tmp_path / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'\xcf\xfa\xed\xfeoriginal')
    (tmp_path / 'runtime-manifest.json').write_text(json.dumps({'pythonExecutable': names[0]}))
    entitlements = tmp_path / 'Worker.entitlements'
    entitlements.write_bytes(plistlib.dumps({'com.apple.security.app-sandbox': True, 'com.apple.security.inherit': True}))
    calls = []
    def output(args, **kw):
        if args[1] == '-hv': return 'Mach header\n MH_MAGIC_64 ARM64 ALL 0 EXECUTE 19 1000 NOUNDEFS\n' if not str(args[-1]).endswith('.dylib') else 'Mach header\n MH_MAGIC_64 ARM64 ALL 0 DYLIB 19 1000 NOUNDEFS\n'
        return str(args[-1]) + ':\n'
    monkeypatch.setattr(audit.subprocess, 'check_output', output)
    def run(args, **kw):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout='', stderr=f'Authority={IDENTITY}\nTeamIdentifier={TEAM}\n')
    monkeypatch.setattr(audit.subprocess, 'run', run)
    audit.audit(tmp_path, signing_identity=IDENTITY, team_id=TEAM, worker_entitlements=entitlements)
    signing = [args for args in calls if '--force' in args]
    assert len(signing) == 6
    for args in signing:
        assert ('--entitlements' in args) == (not args[-1].endswith('.dylib'))


@pytest.mark.parametrize('entitlements', [{}, {'com.apple.security.app-sandbox': True}, {'com.apple.security.app-sandbox': True, 'com.apple.security.inherit': True}])
def test_read_only_distribution_audit_checks_every_executable_sandbox(tmp_path, monkeypatch, entitlements):
    import plistlib
    path = tmp_path / 'python/helper'; path.parent.mkdir(); path.write_bytes(b'\xcf\xfa\xed\xfeoriginal')
    monkeypatch.setattr(audit.subprocess, 'check_output', lambda args, **kw: 'MH_MAGIC_64 ARM64 ALL 0 EXECUTE 19 1000' if args[1] == '-hv' else str(path) + ':\n')
    monkeypatch.setattr(audit, 'verify_distribution_signature', lambda *args: {'teamID': TEAM})
    calls = []
    def run(args, **kw):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout=plistlib.dumps(entitlements), stderr='')
    monkeypatch.setattr(audit.subprocess, 'run', run)
    if entitlements.get('com.apple.security.inherit'):
        audit.audit(tmp_path, require_identity=IDENTITY, require_team=TEAM, require_signatures=True)
    else:
        with pytest.raises(ValueError, match='inherit the app sandbox'):
            audit.audit(tmp_path, require_identity=IDENTITY, require_team=TEAM, require_signatures=True)
    assert not any('--force' in args for args in calls)
    assert path.read_bytes().endswith(b'original')
