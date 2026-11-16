"""Direct signatures must prove hardened runtime and preserve the MAS boundary."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('audit', ROOT/'scripts/runtime_audit.py')
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
TEAM = 'ABCDEFGHIJ'
DIRECT = 'Developer ID Application: KWIP LLC (ABCDEFGHIJ)'
MAS = 'Apple Distribution: KWIP LLC (ABCDEFGHIJ)'

@pytest.mark.parametrize('authority,flags,timestamp,valid', [
    (DIRECT, '0x10000(runtime)', 'Oct 10, 2026', True),
    (DIRECT, '0x0(none)', 'Oct 10, 2026', False),
    (DIRECT, '0x10000(runtime)', None, False),
    (MAS, '0x10000(runtime)', 'Oct 10, 2026', False),
])
def test_direct_signature_requires_developer_id_runtime_and_secure_timestamp(monkeypatch, authority, flags, timestamp, valid):
    details = f'Authority={authority}\nTeamIdentifier={TEAM}\nCodeDirectory v=20500 size=123 flags={flags} hashes=1+0\n'
    if timestamp: details += f'Timestamp={timestamp}\n'
    monkeypatch.setattr(audit.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout='', stderr=details, returncode=0))
    if valid:
        assert audit.verify_distribution_signature(Path('binary'), DIRECT, TEAM, channel='direct')['channel'] == 'direct'
    else:
        with pytest.raises(ValueError): audit.verify_distribution_signature(Path('binary'), DIRECT, TEAM, channel='direct')


def test_default_mas_still_rejects_developer_id():
    with pytest.raises(ValueError): audit.validate_distribution_selection(DIRECT, TEAM)
    with pytest.raises(ValueError): audit.validate_distribution_selection(MAS, TEAM, channel='direct')
    with pytest.raises(ValueError): audit.validate_distribution_selection(DIRECT, TEAM, channel='typo')


def test_direct_dependency_signing_uses_runtime_and_timestamp(tmp_path, monkeypatch):
    path = tmp_path/'python/lib/dependency.dylib'; path.parent.mkdir(parents=True); path.write_bytes(b'\xcf\xfa\xed\xfeoriginal')
    calls = []
    monkeypatch.setattr(audit.subprocess, 'check_output', lambda *a, **kw: 'file:\n')
    def run(args, **kw):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout=b'' if '--entitlements' in args else '', stderr=b'' if '--entitlements' in args else f'Authority={DIRECT}\nTeamIdentifier={TEAM}\nCodeDirectory flags=0x10000(runtime)\nTimestamp=Oct 10, 2026\n')
    monkeypatch.setattr(audit.subprocess, 'run', run)
    audit.audit(tmp_path, signing_identity=DIRECT, team_id=TEAM, channel='direct')
    signed = next(args for args in calls if '--force' in args)
    assert '--timestamp' in signed and '--timestamp=none' not in signed
    assert signed[signed.index('--options')+1] == 'runtime'
    assert '--entitlements' not in signed


def test_direct_cannot_fall_back_to_ad_hoc(tmp_path):
    with pytest.raises(ValueError): audit.audit(tmp_path, sign=True, channel='direct')


@pytest.mark.parametrize('extra', [{}, {'com.apple.security.cs.disable-library-validation':True}])
def test_direct_dependency_entitlements_cannot_weaken_runtime(tmp_path, monkeypatch, extra):
    import plistlib
    path=tmp_path/'python/lib/dependency.dylib';path.parent.mkdir(parents=True);path.write_bytes(b'\xcf\xfa\xed\xfeoriginal')
    monkeypatch.setattr(audit.subprocess,'check_output',lambda *a,**kw:'file:\n')
    def run(args,**kw):
        return SimpleNamespace(returncode=0,stdout=plistlib.dumps(extra) if '--entitlements' in args else b'',stderr=b'')
    monkeypatch.setattr(audit.subprocess,'run',run)
    monkeypatch.setattr(audit,'verify_distribution_signature',lambda *a,**kw:{'teamID':TEAM})
    if extra:
        with pytest.raises(ValueError,match='entitlements'): audit.audit(tmp_path,require_identity=DIRECT,require_team=TEAM,channel='direct')
    else: audit.audit(tmp_path,require_identity=DIRECT,require_team=TEAM,channel='direct')


def test_source_provenance_accepts_only_pinned_unsigned_code(tmp_path, monkeypatch):
    spec=importlib.util.spec_from_file_location('source', ROOT/'scripts/runtime_source_disclosure.py')
    source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
    path=tmp_path/'library.dylib';path.write_bytes(b'CODE-NEW-SIGNATURE')
    monkeypatch.setitem(source.PINS,'gccNativeUnsigned',{'library.dylib':source.sha(b'CODE')})
    def remove(args,**kw):
        target=Path(args[-1]);assert target != path
        target.write_bytes(target.read_bytes().split(b'-')[0])
    monkeypatch.setattr(source.subprocess,'run',remove)
    assert source.pinned_native(tmp_path,'library.dylib','library.dylib',source.sha(b'CODE-OLD-SIGNATURE')) == b'CODE-NEW-SIGNATURE'
    path.write_bytes(b'FOREIGN-NEW-SIGNATURE')
    with pytest.raises(ValueError,match='identity mismatch'):
        source.pinned_native(tmp_path,'library.dylib','library.dylib',source.sha(b'CODE-OLD-SIGNATURE'))


def test_worker_producer_rejects_direct_without_identity_before_copy(tmp_path):
    spec=importlib.util.spec_from_file_location('package', ROOT/'scripts/runtime_package.py')
    package=importlib.util.module_from_spec(spec);spec.loader.exec_module(package)
    destination=tmp_path/'worker'
    with pytest.raises(ValueError):package.copy_worker_sources(destination,channel='direct')
    assert not destination.exists()
