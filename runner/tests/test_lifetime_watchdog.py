"""Native EOF guard remains effective while Python's GIL is held by C."""
import errno
import hashlib
import stat
import os
from pathlib import Path
import selectors
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform != 'darwin', reason='macOS native watchdog contract')
ROOT = Path(__file__).resolve().parents[2]


_COMPILE_OPTIONS = ('-dynamiclib', '-pthread', '-Wall', '-Wextra', '-Werror', '-mmacosx-version-min=14.0')
_BLOCKER_SOURCE = '#include <unistd.h>\nvoid hold_gil_forever(void) { (void)write(1, "GIL_HELD\\n", 9); for (;;) pause(); }\n'
_CACHE_KEY = pytest.StashKey()


def _file_pin(path):
    before = path.lstat()
    assert stat.S_ISREG(before.st_mode), 'Native fixture must remain a regular file'
    pin = hashlib.sha256(path.read_bytes()).hexdigest()
    after = path.lstat()
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    assert identity(before) == identity(after), 'Native fixture changed during verification'
    return pin


def _input_identity():
    compiler = Path(subprocess.check_output(['xcrun', '--find', 'clang'], text=True).strip()).resolve(strict=True)
    info = compiler.stat()
    # Installed compiler metadata binds its process-local identity without hashing
    # the large Apple toolchain for every importing test module.
    compiler_pin = (str(compiler), info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    source = ROOT / 'runner/native/lifetime_watchdog.c'
    environment = tuple((key, os.environ.get(key)) for key in ('DEVELOPER_DIR','TOOLCHAINS','SDKROOT','MACOSX_DEPLOYMENT_TARGET','CPATH','C_INCLUDE_PATH','LIBRARY_PATH'))
    return (compiler_pin, _COMPILE_OPTIONS, _file_pin(source), hashlib.sha256(_BLOCKER_SOURCE.encode()).hexdigest(), environment)


def _verify_cached(cached, identity):
    assert cached['identity'] == identity, 'Native fixture compiler/source/options changed during the test session'
    for path, pin in zip(cached['paths'], cached['pins']):
        assert _file_pin(path) == pin, 'Immutable compiled native fixture changed'
    return cached['paths']


@pytest.fixture(scope='module')
def libraries(tmp_path_factory, request):
    # Pytest registers imported fixtures per module. Only the compiled immutable
    # libraries live in config.stash; every authority, lease, child and pipe stays
    # freshly constructed by its existing test. Each xdist process has its own stash.
    identity = _input_identity()
    cached = request.config.stash.get(_CACHE_KEY, None)
    if cached is not None:
        return _verify_cached(cached, identity)
    folder = tmp_path_factory.mktemp('lifetime-native')
    watchdog, blocker = folder / 'watchdog.dylib', folder / 'blocker.dylib'
    fixture = folder / 'blocker.c'; fixture.write_text(_BLOCKER_SOURCE)
    for source, output in [(ROOT / 'runner/native/lifetime_watchdog.c', watchdog), (fixture, blocker)]:
        result = subprocess.run(['xcrun', 'clang', *_COMPILE_OPTIONS, str(source), '-o', str(output)],
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
    assert _input_identity() == identity, 'Native fixture inputs changed during compilation'
    cached = dict(identity=identity, paths=(watchdog, blocker), pins=(_file_pin(watchdog), _file_pin(blocker)))
    request.config.stash[_CACHE_KEY] = cached
    return _verify_cached(cached, identity)


def ready(process):
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ)
        assert selector.select(timeout=10), 'Watchdog fixture failed to acknowledge startup'
        assert process.stdout.readline() == b'READY\n', process.stderr.read() if process.poll() is not None else b''
        assert selector.select(timeout=10), 'Native GIL-holding call was not entered'
        assert process.stdout.readline() == b'GIL_HELD\n'
    assert process.poll() is None


def cleanup(process):
    if process.poll() is None:
        process.kill()
    process.communicate(timeout=10)


@pytest.mark.parametrize('close_original', [False, True])
def test_eof_exits_while_native_call_holds_gil(libraries, close_original):
    watchdog, blocker = libraries
    reader, writer = os.pipe()
    script = '''
import ctypes,os,sys
watchdog=ctypes.PyDLL(sys.argv[1]);watchdog.kwip_lifetime_arm.argtypes=[ctypes.c_int];watchdog.kwip_lifetime_arm.restype=ctypes.c_int
fd=int(sys.argv[3]);assert watchdog.kwip_lifetime_arm(fd)==0
if sys.argv[4]=='yes':os.close(fd)
print('READY',flush=True)
ctypes.PyDLL(sys.argv[2]).hold_gil_forever()
'''
    process = subprocess.Popen([sys.executable, '-I', '-c', script, str(watchdog), str(blocker),
                                str(reader), 'yes' if close_original else 'no'],
                               pass_fds=(reader,), stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    os.close(reader)
    try:
        ready(process)
        # A pipe reader without output proves no unexpected exit while parent lives.
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            assert not selector.select(timeout=0.1)
        os.close(writer); writer = None
        assert process.wait(timeout=10) == 125
    finally:
        if writer is not None: os.close(writer)
        cleanup(process)


def test_unexpected_lifetime_bytes_fail_closed(libraries):
    watchdog, blocker = libraries
    reader, writer = os.pipe()
    script = '''
import ctypes,sys
lib=ctypes.PyDLL(sys.argv[1]);lib.kwip_lifetime_arm.argtypes=[ctypes.c_int];lib.kwip_lifetime_arm.restype=ctypes.c_int
assert lib.kwip_lifetime_arm(int(sys.argv[3]))==0
print('READY',flush=True)
ctypes.PyDLL(sys.argv[2]).hold_gil_forever()
'''
    process = subprocess.Popen([sys.executable, '-I', '-c', script, str(watchdog), str(blocker), str(reader)],
                               pass_fds=(reader,), stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    os.close(reader)
    try:
        ready(process)
        os.write(writer, b'not a heartbeat')
        assert process.wait(timeout=10) == 125
    finally:
        os.close(writer); cleanup(process)


def test_invalid_descriptors_do_not_consume_one_arm_and_repeat_rejected(libraries, tmp_path):
    watchdog, _ = libraries
    reader, writer = os.pipe()
    regular = tmp_path / 'ordinary-file'; regular.write_text('test')
    script = '''
import ctypes,errno,os,sys
lib=ctypes.PyDLL(sys.argv[1]);arm=lib.kwip_lifetime_arm;arm.argtypes=[ctypes.c_int];arm.restype=ctypes.c_int
reader,writer=int(sys.argv[2]),int(sys.argv[3])
assert arm(-1)==errno.EBADF
for fd in [0,1,2]:assert arm(fd)==errno.EBADF
fd=os.open(sys.argv[4],os.O_RDONLY)
assert arm(fd)==errno.EINVAL
os.close(fd)
assert arm(fd)==errno.EBADF
assert arm(writer)==errno.EINVAL
assert arm(reader)==0
assert arm(reader)==errno.EALREADY
print('READY',flush=True)
'''
    process = subprocess.Popen([sys.executable, '-I', '-c', script, str(watchdog), str(reader), str(writer), str(regular)],
                               pass_fds=(reader, writer), stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    os.close(reader)
    try:
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, stderr.decode()
        assert stdout == b'READY\n'
    finally:
        os.close(writer); cleanup(process)


@pytest.mark.parametrize('mutation',['source','compiler','options','library','symlink'])
def test_shared_native_fixture_refuses_changed_inputs_or_outputs(tmp_path,mutation):
    first=tmp_path/'watchdog.dylib';second=tmp_path/'blocker.dylib'
    first.write_bytes(b'guard fixture');second.write_bytes(b'blocker fixture')
    identity=('compiler identity',('flags',),'source hash','blocker hash')
    cached=dict(identity=identity,paths=(first,second),pins=(_file_pin(first),_file_pin(second)))
    assert _verify_cached(cached,identity)==(first,second)
    changed=identity
    if mutation in ('source','compiler','options'):
        index={'source':2,'compiler':0,'options':1}[mutation]
        values=list(identity);values[index]='changed';changed=tuple(values)
    elif mutation=='library':first.write_bytes(b'changed bytes')
    else:first.unlink();first.symlink_to(second)
    with pytest.raises(AssertionError):_verify_cached(cached,changed)
