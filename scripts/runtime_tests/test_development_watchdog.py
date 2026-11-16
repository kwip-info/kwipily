"""Development guard binaries are built explicitly, never by the application."""
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('development_guard_package', ROOT/'scripts/runtime_package.py')
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)
pytestmark = pytest.mark.skipif(platform.system() != 'Darwin' or platform.machine() != 'arm64',
                               reason='Native arm64 macOS build tool')


def test_explicit_build_stages_only_library_and_exact_digest_manifest(tmp_path):
    destination = tmp_path/'native'
    result = package.prepare_development_watchdog(destination)
    library = destination/'lifetime_watchdog.dylib'
    manifest = json.loads((destination/'lifetime_watchdog.json').read_bytes())
    assert manifest == result == {
        'schemaVersion': 1,
        'sourceSHA256': hashlib.sha256((ROOT/'runner/native/lifetime_watchdog.c').read_bytes()).hexdigest(),
        'librarySHA256': hashlib.sha256(library.read_bytes()).hexdigest(),
    }
    assert {p.name for p in destination.iterdir()} == {'lifetime_watchdog.dylib','lifetime_watchdog.json'}
    assert package.prepare_development_watchdog(destination) == manifest


def test_failed_rebuild_preserves_previous_pair(tmp_path, monkeypatch):
    destination = tmp_path/'native'
    package.prepare_development_watchdog(destination)
    before = {p.name:p.read_bytes() for p in destination.iterdir()}
    source = tmp_path/'missing-source'; source.mkdir()
    monkeypatch.setattr(package,'ROOT',source)
    with pytest.raises((OSError,ValueError)):
        package.prepare_development_watchdog(destination)
    assert {p.name:p.read_bytes() for p in destination.iterdir()} == before


def test_symlink_destination_is_not_used(tmp_path):
    real = tmp_path/'real'; real.mkdir()
    alias = tmp_path/'alias'; alias.symlink_to(real,target_is_directory=True)
    with pytest.raises(ValueError):package.prepare_development_watchdog(alias)
    assert list(real.iterdir()) == []
