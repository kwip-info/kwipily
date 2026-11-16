"""Mocked packaging controls: no local binary mutation, signing or build."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('packaged_audit', ROOT / 'scripts/runtime_audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def native(root, relative):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'\xcf\xfa\xed\xfeoriginal')
    return path


def commands(monkeypatch, identities, external=None, invalid=None):
    calls = []
    def output(args, **kwargs):
        path = args[-1]
        if args[1] == '-D':
            return path + ':\n' + (identities.get(path, '') + '\n' if identities.get(path) else '')
        if args[1] == '-L':
            links = [identities[path]] if identities.get(path) else []
            links += ['/usr/lib/libSystem.B.dylib'] + ([external] if external else [])
            return path + ':\n' + ''.join('\t' + item + ' (compatibility version 1.0.0)\n' for item in links)
        return ' minos 14.0\n'
    def run(args, **kwargs):
        calls.append(args)
        if args[0].endswith('install_name_tool'):
            identities[args[-1]] = args[2]
        return SimpleNamespace(returncode=1 if args[1] == '--verify' and args[-1] == invalid else 0)
    monkeypatch.setattr(audit.subprocess, 'check_output', output)
    monkeypatch.setattr(audit.subprocess, 'run', run)
    return calls


def test_normalizes_dependency_identity_without_resigning_sealed_worker(tmp_path, monkeypatch):
    library = native(tmp_path, 'python/lib/libomp.dylib')
    worker = native(tmp_path, 'runner/native/lifetime_watchdog.dylib')
    original = worker.read_bytes()
    calls = commands(monkeypatch, {str(library): '/opt/llvm/libomp.dylib', str(worker): '@rpath/lifetime_watchdog.dylib'})
    result = audit.audit(tmp_path, sign=True)
    assert result['fileCount'] == 2
    assert all(row['signatureVerified'] for row in result['machOFiles'])
    assert any(args[0].endswith('install_name_tool') and args[-1] == str(library) for args in calls)
    assert not any('--force' in args and args[-1] == str(worker) for args in calls)
    assert any('--verify' in args and args[-1] == str(worker) for args in calls)
    assert worker.read_bytes() == original


def test_actual_external_load_fails_before_native_signing(tmp_path, monkeypatch):
    library = native(tmp_path, 'python/lib/example.dylib')
    calls = commands(monkeypatch, {str(library): '@rpath/example.dylib'}, external='/opt/unavailable/dependency.dylib')
    with pytest.raises(ValueError, match='External runtime dependency'):
        audit.audit(tmp_path, sign=True, require_signatures=True)
    assert not any('--force' in args for args in calls)


def test_invalid_signature_fails_closed_in_packaging_mode(tmp_path, monkeypatch):
    worker = native(tmp_path, 'runner/native/lifetime_watchdog.dylib')
    commands(monkeypatch, {str(worker): '@rpath/lifetime_watchdog.dylib'}, invalid=str(worker))
    with pytest.raises(ValueError, match='signature'):
        audit.audit(tmp_path, sign=True, require_signatures=True)


def test_sealed_worker_identity_cannot_be_repaired(tmp_path, monkeypatch):
    worker = native(tmp_path, 'runner/native/lifetime_watchdog.dylib')
    calls = commands(monkeypatch, {str(worker): '/opt/wrong/guard.dylib'})
    with pytest.raises(ValueError, match='sealed worker'):
        audit.audit(tmp_path, sign=True, require_signatures=True)
    assert not any(args[0].endswith('install_name_tool') or '--force' in args for args in calls)


def test_packaged_script_audits_copy_before_entitlements_and_rechecks_worker():
    script = (ROOT / 'scripts/build-packaged-app.sh').read_text()
    audit_line = script.index('scripts/runtime_audit.py')
    assert script.index('/usr/bin/ditto "$RUNTIME_SOURCE"') < audit_line
    assert audit_line < script.index('--entitlements')
    assert '--sign-ad-hoc --require-signatures' in script
    assert script[audit_line:script.index('--entitlements')].count('--verify-current-worker') == 1
