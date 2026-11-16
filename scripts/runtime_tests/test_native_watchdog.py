"""The packaged watchdog is generated, inventory-pinned, and independently usable."""
import hashlib
import importlib.util
import json
import os
import platform
from pathlib import Path
import selectors
import shutil
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform != 'darwin' or platform.machine() != 'arm64',
                               reason='arm64 macOS native runtime packaging')
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('native_watchdog_package', ROOT / 'scripts/runtime_package.py')
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)


@pytest.fixture
def canonical(tmp_path, monkeypatch):
    source = tmp_path / 'canonical-source'
    (source / 'runner/native').mkdir(parents=True)
    (source / 'scripts').mkdir()
    (source / 'Sources/KwipMLCore/Resources').mkdir(parents=True)
    (source / 'runner/main.py').write_text('print("fixture worker")\n')
    (source / 'Sources/KwipMLCore/Resources/components.json').write_text('{}')
    shutil.copy2(ROOT / 'scripts/runtime_launch.py', source / 'scripts/runtime_launch.py')
    shutil.copy2(ROOT / 'runner/native/lifetime_watchdog.c', source / 'runner/native/lifetime_watchdog.c')
    monkeypatch.setattr(package, 'ROOT', source)
    return source


def payload(path):
    path.mkdir()
    package.copy_worker_sources(path)
    (path / 'runtime-manifest.json').write_text(json.dumps({'schemaVersion': 1}))
    return package.refresh_worker_integrity(path)


def test_generated_library_is_reproducible_and_inventory_pinned(canonical, tmp_path):
    first = tmp_path / 'first build'
    second = tmp_path / 'different destination' / 'second'
    second.parent.mkdir()
    one = payload(first)
    two = payload(second)
    name = 'runner/native/lifetime_watchdog.dylib'
    assert name in one['workerSourceHashes'], 'Packaged native watchdog was not generated'
    first_bytes = (first / name).read_bytes()
    assert first_bytes == (second / name).read_bytes()
    assert hashlib.sha256(first_bytes).hexdigest() == one['workerSourceHashes'][name]
    assert one['workerSourceHashes'] == two['workerSourceHashes']
    assert package.verify_current_worker(first)['status'] == 'matches-current-source'
    signed = subprocess.run(['codesign', '--verify', '--strict', str(first / name)],
                            capture_output=True, text=True, timeout=15)
    assert signed.returncode == 0, signed.stderr


def test_generated_library_enforces_parent_pipe_eof(canonical, tmp_path):
    root = tmp_path / 'runtime'
    payload(root)
    library = root / 'runner/native/lifetime_watchdog.dylib'
    assert library.is_file(), 'Packaged native watchdog was not generated'
    reader, writer = os.pipe()
    script = '''
import ctypes,os,signal,sys
lib=ctypes.PyDLL(sys.argv[1]);arm=lib.kwip_lifetime_arm
arm.argtypes=[ctypes.c_int];arm.restype=ctypes.c_int
assert arm(int(sys.argv[2]))==0
os.close(int(sys.argv[2]))
print('READY',flush=True)
signal.pause()
'''
    process = subprocess.Popen([sys.executable, '-I', '-c', script, str(library), str(reader)],
                               pass_fds=(reader,), stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    os.close(reader)
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            assert selector.select(timeout=10), 'Generated watchdog did not arm'
            line = process.stdout.readline()
            assert line == b'READY\n', process.stderr.read() if line == b'' else line
        assert process.poll() is None
        os.close(writer); writer = None
        assert process.wait(timeout=10) == 125
    finally:
        if writer is not None: os.close(writer)
        if process.poll() is None: process.kill()
        process.communicate(timeout=10)


def test_canonical_verifier_rejects_changed_library_even_after_resealing(canonical, tmp_path):
    root = tmp_path / 'runtime'
    payload(root)
    library = root / 'runner/native/lifetime_watchdog.dylib'
    assert library.is_file(), 'Packaged native watchdog was not generated'
    library.write_bytes(library.read_bytes() + b'changed native implementation')
    package.refresh_worker_integrity(root)
    package.integrity.validate_worker(root)  # Internal hashes alone are insufficient.
    before = (root / 'runtime-manifest.json').read_bytes()
    with pytest.raises(ValueError, match='current build sources'):
        package.verify_current_worker(root)
    assert (root / 'runtime-manifest.json').read_bytes() == before


def test_packaging_without_canonical_watchdog_source_fails_closed(canonical, tmp_path):
    (canonical / 'runner/native/lifetime_watchdog.c').unlink()
    root = tmp_path / 'runtime'; root.mkdir()
    with pytest.raises((ValueError, FileNotFoundError, subprocess.CalledProcessError)):
        package.copy_worker_sources(root)
    assert not (root / 'runtime-manifest.json').exists()


def test_checkout_binary_is_ignored_and_rebuilt(canonical, tmp_path):
    (canonical / 'runner/native/lifetime_watchdog.dylib').write_bytes(b'not a library')
    root = tmp_path / 'runtime'
    payload(root)
    assert (root / 'runner/native/lifetime_watchdog.dylib').read_bytes() != b'not a library'
    assert package.verify_current_worker(root)['status'] == 'matches-current-source'
