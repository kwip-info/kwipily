#!/usr/bin/env python3
"""Prepare a separate build-2 runtime from the reviewed build-1 payload.

Use the official same-version OpenBLAS wheel, preserve the exact reviewed GCC
runtime/source correspondence, and omit the unused Tk GUI closure. Signing and
regenerating license/source inventories are subsequent explicit build steps.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

WHEEL = 'scipy-1.17.0-cp313-cp313-macosx_12_0_arm64.whl'
WHEEL_SHA256 = '1f9586a58039d7229ce77b52f8472c972448cded5736eaf102d5658bbac4c269'
SITE = Path('python/lib/python3.13/site-packages')
GCC = {
    'libgcc_s.1.1.dylib': '9249afe2a8590d183b718e4a11e6f4406fdfdf4dbe22bc053531122b62d40925',
    'libgfortran.5.dylib': '14c8b7f0f8f210161825325290b5c60ea0c576f4eef9206cd9211b34df22bac4',
    'libquadmath.0.dylib': '78deaac2e5741f2da41fb2f967ec0b248e44fd41df34867467c950ffb41e54e9',
}
OMIT = ('python/lib/libtk8.6.dylib', 'python/lib/tk8.6',
        'python/lib/python3.13/tkinter', 'python/lib/python3.13/lib-dynload/_tkinter.cpython-313-darwin.so')


def prepare(source, wheel, output):
    source, wheel, output = Path(source), Path(wheel), Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError('Destination must not exist')
    if source.is_symlink() or not source.is_dir() or output.resolve().is_relative_to(source.resolve()):
        raise ValueError('Source must be a real directory and destination separate')
    if wheel.name != WHEEL or hashlib.sha256(wheel.read_bytes()).hexdigest() != WHEEL_SHA256:
        raise ValueError('Expected exact reviewed official SciPy OpenBLAS wheel')
    for p in source.rglob('*'):
        if p.is_symlink() and (not p.exists() or not p.resolve().is_relative_to(source.resolve())):
            raise ValueError('Source contains a broken or escaping symlink')
    libraries = {}
    for name, expected in GCC.items():
        data = (source / SITE / 'scipy/.dylibs' / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('Reviewed GCC runtime differs: ' + name)
        libraries[name] = data
    for relative in OMIT:
        if not (source / relative).exists():
            raise ValueError('Expected reviewed Tk closure: ' + relative)
    with tempfile.TemporaryDirectory(prefix='.review-repair-', dir=output.parent) as temporary:
        staging = Path(temporary) / 'runtime'
        shutil.copytree(source, staging, symlinks=True)
        for name in ('scipy', 'scipy-1.17.0.dist-info'):
            shutil.rmtree(staging / SITE / name)
        with zipfile.ZipFile(wheel) as archive:
            for member in archive.infolist():
                path = Path(member.filename)
                if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0] not in {'scipy', 'scipy-1.17.0.dist-info'}:
                    raise ValueError('Unexpected wheel member')
            archive.extractall(staging / SITE)
        for name, data in libraries.items():
            (staging / SITE / 'scipy/.dylibs' / name).write_bytes(data)
        for relative in OMIT:
            path = staging / relative
            if path.is_dir(): shutil.rmtree(path)
            else: path.unlink()
        # Existing disclosure describes the old SciPy license inventory. Rebuild
        # it against the retained source materials before production packaging.
        shutil.rmtree(staging / 'source-disclosure')
        receipt = {'schemaVersion': 1, 'wheel': WHEEL, 'wheelSHA256': WHEEL_SHA256,
                   'retainedGCCSHA256': GCC, 'omittedUnusedTkClosure': list(OMIT),
                   'status': 'requiresSigningLicenseInventoryAndSourceDisclosure'}
        (staging / 'app-review-repair.json').write_text(json.dumps(receipt, indent=2) + '\n')
        staging.rename(output)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-runtime', required=True)
    parser.add_argument('--scipy-wheel', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_runtime, args.scipy_wheel, args.output)))
