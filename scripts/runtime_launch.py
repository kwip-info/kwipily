"""Bundled worker bootstrap; invoke with bundled Python using -I -B.

Worker checksums are validated before importing runner code. The signed containing
bundle supplies authenticity; this manifest alone is not a cryptographic signer.
"""
from pathlib import Path
import hashlib
import json
import os
import re
import runpy
import stat
import sys

MAX_FILES=512
MAX_FILE_BYTES=10_000_000
MAX_TOTAL_BYTES=32_000_000


def bounded_regular(path,maximum):
    descriptor=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        info=os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size>maximum:raise ValueError('expected a bounded regular file')
        with os.fdopen(descriptor,'rb') as stream:
            descriptor=None
            raw=stream.read(maximum+1)
        if len(raw)>maximum:raise ValueError('file exceeds size limit')
        return raw
    finally:
        if descriptor is not None:os.close(descriptor)


def worker_source_hashes(root):
    root=Path(root);paths=[root/'launch.py',root/'components.json'];pending=[root/'runner'];entries=0
    while pending:
        directory=pending.pop()
        if not stat.S_ISDIR(directory.lstat().st_mode):raise ValueError('worker directories must not be symbolic links')
        with os.scandir(directory) as listing:
            for entry in listing:
                entries+=1
                if entries>MAX_FILES:raise ValueError('worker inventory exceeds entry limit')
                if entry.is_symlink():raise ValueError('worker symbolic links are not accepted')
                if entry.is_dir(follow_symlinks=False):pending.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):paths.append(Path(entry.path))
                else:raise ValueError('worker contains a nonregular file')
    hashes={};total=0
    for path in sorted(paths):
        raw=bounded_regular(path,MAX_FILE_BYTES);total+=len(raw)
        if total>MAX_TOTAL_BYTES:raise ValueError('worker source total exceeds size limit')
        hashes[path.relative_to(root).as_posix()]=hashlib.sha256(raw).hexdigest()
    if 'runner/main.py' not in hashes:raise ValueError('worker entry point missing')
    return hashes


def validate_worker(root):
    manifest=json.loads(bounded_regular(root/'runtime-manifest.json',65536))
    if not isinstance(manifest,dict) or type(manifest.get('schemaVersion')) is not int or manifest['schemaVersion']!=1 or type(manifest.get('workerIntegrityVersion')) is not int or manifest['workerIntegrityVersion']!=1:raise ValueError('missing or unsupported worker integrity version')
    expected=manifest.get('workerSourceHashes')
    if not isinstance(expected,dict) or not 3<=len(expected)<=MAX_FILES+2:raise ValueError('missing or oversized worker checksum inventory')
    for name,value in expected.items():
        if not isinstance(name,str) or len(name)>512 or name.startswith('/') or any(p in ('','..','.') for p in name.split('/')) or (name not in ('launch.py','components.json') and not name.startswith('runner/')):raise ValueError('unsafe worker inventory path')
        if not isinstance(value,str) or re.fullmatch('[0-9a-f]{64}',value) is None:raise ValueError('invalid worker checksum')
    actual=worker_source_hashes(root)
    if actual!=expected:raise ValueError('worker files are missing, changed or unlisted')
    for field,name in (('entryHash','launch.py'),('registryHash','components.json')):
        if field in manifest and manifest[field]!=actual[name]:raise ValueError('worker checksum metadata disagrees')


def main():
    root=Path(__file__).resolve().parent
    try:
        validate_worker(root)
        registry_values=[]
        for index,argument in enumerate(sys.argv[1:],1):
            if argument=='--registry':
                if index+1>=len(sys.argv):raise ValueError('bundled registry argument is incomplete')
                registry_values.append(sys.argv[index+1])
            elif argument.startswith('--registry='):registry_values.append(argument.split('=',1)[1])
        if len(registry_values)>1 or any(Path(value).resolve()!=root/'components.json' for value in registry_values):raise ValueError('packaged registry must be the verified bundled registry')
    except (OSError,ValueError,TypeError,RecursionError) as error:
        print(json.dumps({'event':'error','message':f'Bundled worker integrity check failed: {error}. Reinstall the app or rebuild its runtime payload; development fallback is disabled.'}),flush=True)
        return 1
    # No shell-enabled per-operation fallback and no checkout module injection.
    os.environ.pop('PYTORCH_ENABLE_MPS_FALLBACK',None)
    sys.path.insert(0,str(root/'runner'))
    internal = any(argument.startswith('--kwip-') for argument in sys.argv[1:])
    if not registry_values and not internal:
        sys.argv.extend(['--registry',str(root/'components.json')])
    runpy.run_path(str(root/'runner/main.py'),run_name='__main__', init_globals={'_KWIP_RUNTIME_CONTEXT': {
        'entry': root/'launch.py', 'native_library': root/'runner/native/lifetime_watchdog.dylib',
        'registry': root/'components.json', 'isolated': True}})
    return 0

if __name__=='__main__':sys.exit(main())
