#!/usr/bin/env python3
"""Stage attributed upstream privacy resources before app signing; no SDK certification."""
import argparse
import ctypes
import errno
import sys
import hashlib
import json
import os
from pathlib import Path
import plistlib
import shutil
import stat
import tempfile

ASSETS = Path('docs/production/vendor-privacy-20261009')
PROVENANCE_SHA256 = '832d352459d3ee724c1d03b8a30a1d1f0a8329b1d4a3881b8f7b9e3ca2c45a3f'
SITE = Path('python/lib/python3.13/site-packages')
MAX_INPUT = 128 * 1024
BUNDLES = {
    'abseil-20250814.0': ('AbseilPrivacy', 'tensorflow/libtensorflow_framework.2.dylib', 'lts_family_headers'),
    'protobuf-7.36.2': ('PythonProtobufPrivacy', 'google/_upb/_message.abi3.so', 'installed_python_release'),
    'grpc-1.84.0': ('GRPCPrivacy', 'grpc/_cython/cygrpc.cpython-313-darwin.so', 'installed_release_objectivec_manifest'),
    'openssl-3.0.16': ('OpenSSLPrivacy', None, 'cpython_vendor_recipe_and_observed_version'),
    'boringssl-grpc-submodule': ('GRPCBoringSSLPrivacy', 'grpc/_cython/cygrpc.cpython-313-darwin.so', 'conditional_release_dependency'),
    'protobuf-tensorflow-v31.1': ('TensorFlowProtobufPrivacy', 'tensorflow/libtensorflow_framework.2.dylib', 'tensorflow_headers_source_family'),
}


def safe_path(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('unsafe relative path')
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f'symlink input: {current}')
    if not current.is_file():
        raise ValueError(f'missing regular input: {current}')
    return current


def read_bounded(path):
    if path.stat().st_size > MAX_INPUT:
        raise ValueError(f'oversized input: {path}')
    with path.open('rb') as stream:
        data = stream.read(MAX_INPUT + 1)
    if len(data) > MAX_INPUT:
        raise ValueError('oversized input')
    return data


def validate_manifest(data):
    try:
        value = plistlib.loads(data)
    except Exception as error:
        raise ValueError('malformed privacy plist') from error
    keys = {'NSPrivacyTracking', 'NSPrivacyCollectedDataTypes', 'NSPrivacyTrackingDomains', 'NSPrivacyAccessedAPITypes'}
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError('unexpected privacy keys')
    if value['NSPrivacyTracking'] is not False or value['NSPrivacyCollectedDataTypes'] != [] or value['NSPrivacyTrackingDomains'] != []:
        raise ValueError('unexpected vendor collection declaration')
    apis = value['NSPrivacyAccessedAPITypes']
    expected = [{'NSPrivacyAccessedAPIType': 'NSPrivacyAccessedAPICategoryFileTimestamp', 'NSPrivacyAccessedAPITypeReasons': ['C617.1']}]
    if apis != [] and apis != expected:
        raise ValueError('unexpected vendor accessed API declaration')
    return value


def installed_release(runtime, name, version):
    relative = SITE / f'{name}-{version}.dist-info/METADATA'
    text = read_bounded(safe_path(runtime, relative)).decode('utf-8')
    if f'Name: {name}\n' not in text or f'Version: {version}\n' not in text:
        raise ValueError(f'installed version mismatch: {name}')


def publish_exclusive(staging, destination):
    # macOS renamex_np atomically refuses even an empty destination directory.
    if sys.platform != 'darwin':
        raise OSError('vendor privacy publication requires macOS renamex_np')
    libc = ctypes.CDLL(None, use_errno=True)
    rename = libc.renamex_np
    rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(os.fsencode(staging), os.fsencode(destination), 4) != 0:  # RENAME_EXCL
        code = ctypes.get_errno()
        if code in (errno.EEXIST, errno.ENOTEMPTY):
            raise FileExistsError(code, 'refusing existing output', str(destination))
        raise OSError(code, os.strerror(code), str(destination))


def stage_vendor_privacy(repo_root, runtime_root, destination):
    repo, runtime, dest = Path(repo_root).resolve(), Path(runtime_root).resolve(), Path(destination)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError(f'refusing existing output: {dest}')
    # Refuse symlink destination ancestors; /var may be an OS alias, so use canonical parent.
    if dest.parent.is_symlink():
        raise ValueError('symlink destination parent')
    provenance_bytes = read_bounded(safe_path(repo, ASSETS / 'PROVENANCE.json'))
    if hashlib.sha256(provenance_bytes).hexdigest() != PROVENANCE_SHA256:
        raise ValueError('provenance digest mismatch')
    provenance = json.loads(provenance_bytes)
    if {r['name'] for r in provenance['artifacts']} != set(BUNDLES) or len(provenance['artifacts']) != 6:
        raise ValueError('unexpected artifact inventory')
    for name, version in [('protobuf', '7.36.2'), ('grpcio', '1.84.0'), ('tensorflow', '2.21.0')]:
        installed_release(runtime, name, version)
    options = read_bounded(safe_path(runtime, SITE / 'tensorflow/include/absl/base/options.h'))
    header = read_bounded(safe_path(runtime, SITE / 'tensorflow/include/google/protobuf/empty.pb.h'))
    if b'lts_20250814' not in options or b'PROTOBUF_VERSION != 6031001' not in header:
        raise ValueError('TensorFlow dependency header identity mismatch')
    prepared = []
    for source in provenance['artifacts']:
        title, owner, status = BUNDLES[source['name']]
        owner_relative = SITE / owner if owner else Path('python/lib/libpython3.13.dylib')
        safe_path(runtime, owner_relative)  # existence only; never read/modify native binaries
        artifacts = {}
        for artifact in source['artifacts']:
            relative = Path(artifact['path'])
            if relative.parent != ASSETS / source['name'] or relative.name not in ('PrivacyInfo.xcprivacy', 'LICENSE.upstream'):
                raise ValueError('unexpected artifact path')
            data = read_bounded(safe_path(repo, relative))
            if len(data) != artifact['bytes'] or hashlib.sha256(data).hexdigest() != artifact['sha256']:
                raise ValueError(f'artifact digest mismatch: {relative}')
            artifacts[relative.name] = data
        if set(artifacts) != {'PrivacyInfo.xcprivacy', 'LICENSE.upstream'}:
            raise ValueError('incomplete artifact pair')
        validate_manifest(artifacts['PrivacyInfo.xcprivacy'])
        row = dict(bundle=title + '.bundle', source_manifest=str(ASSETS / source['name'] / 'PrivacyInfo.xcprivacy'), binding_status=status,
                   native_owners=[str(owner_relative)], upstream=source, vendor_signature_provided=False,
                   apple_archive_recognition_verified=False)
        if source['name'] == 'boringssl-grpc-submodule':
            row['limitation'] = 'Exact gRPC release submodule; installed wheel build options and embedded revision not independently attested.'
        if source['name'] == 'openssl-3.0.16':
            row['limitation'] = 'CPython20250814 vendor recipe and prior recorded libpython version inspection; not a new binary attestation.'
        prepared.append((title, row, artifacts))
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.vendor-privacy-', dir=dest.parent))
    try:
        for title, row, artifacts in prepared:
            contents = staging / row['bundle'] / 'Contents'
            resources = contents / 'Resources'
            resources.mkdir(parents=True)
            info = dict(CFBundleIdentifier='info.kwip.kwipml.vendorprivacy.' + title.lower(), CFBundleName=title,
                        CFBundlePackageType='BNDL', CFBundleVersion='1', CFBundleShortVersionString='1.0')
            (contents / 'Info.plist').write_bytes(plistlib.dumps(info, sort_keys=True))
            for filename, data in artifacts.items():
                (resources / filename).write_bytes(data)
            (resources / 'PROVENANCE.json').write_text(json.dumps(row, indent=2) + '\n')
        result = dict(format_version=1, purpose='Build-only vendor resource candidate; no vendor signature or Apple acceptance claim', bundles=[r for _, r, _ in prepared])
        (staging / 'MAPPING.json').write_text(json.dumps(result, indent=2) + '\n')
        if dest.exists() or dest.is_symlink():
            raise FileExistsError('output appeared while staging')
        publish_exclusive(staging, dest)
        return result
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def resource_inventory(root):
    if root.is_symlink() or not root.is_dir():
        raise ValueError('missing or symlink vendor privacy directory')
    inventory = {}
    total = 0
    pending = [root]
    while pending:
        directory = pending.pop()
        for path in sorted(directory.iterdir()):
            mode = path.lstat().st_mode
            relative = str(path.relative_to(root))
            if stat.S_ISLNK(mode):
                raise ValueError('symlink vendor privacy resource')
            if stat.S_ISDIR(mode):
                inventory[relative] = ('directory', None)
                pending.append(path)
            elif stat.S_ISREG(mode):
                data = read_bounded(path)
                total += len(data)
                inventory[relative] = ('file', data)
            else:
                raise ValueError('non-regular vendor privacy resource')
            if len(inventory) > 100 or total > 2 * 1024 * 1024:
                raise ValueError('oversized vendor privacy inventory')
    return inventory


def verify_vendor_privacy(repo_root, runtime_root, destination):
    """Verify exact already-staged resources without touching destination contents."""
    destination = Path(destination)
    actual = resource_inventory(destination)
    with tempfile.TemporaryDirectory(prefix='kwipml-vendor-privacy-verify-') as temporary:
        expected_path = Path(temporary) / 'VendorPrivacy'
        result = stage_vendor_privacy(repo_root, runtime_root, expected_path)
        expected = resource_inventory(expected_path)
        if actual != expected:
            raise ValueError('stale or conflicting vendor privacy resources')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--runtime-root', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--verify-existing', action='store_true', help='Require exact existing resource set; never overwrite it')
    args = parser.parse_args()
    action = verify_vendor_privacy if args.verify_existing else stage_vendor_privacy
    result = action(args.repo_root, args.runtime_root, args.destination)
    verb = 'Verified' if args.verify_existing else 'Staged'
    print(f"{verb} {len(result['bundles'])} attributed vendor privacy resource bundles (archive recognition unverified).")
