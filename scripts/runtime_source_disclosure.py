#!/usr/bin/env python3
"""Build-only retained source/notices index; never edits worker/native binaries."""
import argparse
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
NOTICES = ROOT / 'scripts/runtime_notices'
SITE = 'python/lib/python3.13/site-packages/'
DEFAULT_GCC_DIRECTORY = None
PROVENANCE_SHA256 = 'ebe53478a5a8a3831665cbce3293604a32715d4659b6488a9ada9e6f89a478bc'
EIGEN_PROVENANCE_SHA256='9879c0982f0135abd400b338b8c74ce46546b11b599272b2d07a324931fceaf4'
PINS = {
 'lgpl': '32434afcc8666ba060e111d715bfdb6c2d5dd8a35fa4d3ab8ad67d8f850d2f2b',
 'homebrew': 'f80329e58613ad669c0e73cb132d8060b9b2c55e339c73848068e4d1567f4627',
 'replacementGuide': '481b160fc815b09baacebfeb102025bf7b6f4d53923e8460acf632bb76aca1e7',
 'buildContext': '866e660b3622668931113b6b7c182df1286ad5b378fb38ac3773aba09514d304',
 'formulaPatches': '5322586e284f50e2df2806aa805a33b3bdf7364edbe835a5303745b40f4dbe65',
 'runnerImages': '9e48ecfa18e2b15169746a3c97beda4d1d6c6796097038498ca434ca7e0ccd44',
 'certifiSource': '9cc2a774b5198dcff14d9be1e66091f538975d867ce029a96bce15a55dfd730f',
 'certifiLicense': 'e93716da6b9c0d5a4a1df60fe695b370f0695603d21f6f83f053e42cfc10caf7',
 'tensorflowNotices': 'b9cc03566b8e0e074f718e64f96f49b6fe070e3f777c41b3d4bb0df7113ea634',
 'tensorflowLicense': 'c79a7fea0e3cac04cd43f20e7b648e5a0ff8fa5344e644b0ee09ca1162b62747',
 'eigenTree': 'ff22f14e7cc05854bd317f8c6f595b01e5c7e77181466fad4479015cb2c67165',
 'eigenFiles': 412,
 'eigenUnsupportedFiles':191,
 'eigenUnsupportedTree':'6d850519cbbfe9928d4347e429ce2aa5d44cc5f510420ccb5d7a5fbbeb10ad43',
 # Derived from the exact reviewed gccNative bytes after codesign --remove-signature
 # on temporary copies; permit certificate rotation without accepting changed code.
 'gccNativeUnsigned': {
  'libgcc_s.1.1.dylib': '61505ade1a4197775bfab0d618596a4908e6b782ac648fb5ef47f61a5523b024',
  'libgfortran.5.dylib': '4375a24f18364bafd08c39f2e3b6d82c8e11ec7ab8a6e5ebb6ea33cd09732b5e',
  'libquadmath.0.dylib': '03e4e6a4760e009ebb7410c2dc3815cda02820249e530067d8d024324e1c4431',
 },
 'gccNative': {
  'libgcc_s.1.1.dylib': '9249afe2a8590d183b718e4a11e6f4406fdfdf4dbe22bc053531122b62d40925',
  'libgfortran.5.dylib': '14c8b7f0f8f210161825325290b5c60ea0c576f4eef9206cd9211b34df22bac4',
  'libquadmath.0.dylib': '78deaac2e5741f2da41fb2f967ec0b248e44fd41df34867467c950ffb41e54e9'}}


def sha(data): return hashlib.sha256(data).hexdigest()
def canonical(value): return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def read(root, relative, limit=8*1024*1024):
    root, relative = Path(root), Path(relative)
    if relative.is_absolute() or not relative.parts or '..' in relative.parts:
        raise ValueError('Unsafe disclosure input path')
    current = root
    if current.is_symlink() or not current.is_dir(): raise ValueError('Input root must be a real directory')
    for part in relative.parts:
        current = current / part
        if current.is_symlink(): raise ValueError('Disclosure inputs cannot be symbolic links')
    descriptor = os.open(current, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= limit: raise ValueError('Input must be a bounded nonempty regular file')
        value = stream.read(limit + 1)
    if len(value) > limit: raise ValueError('Input exceeds its size bound')
    return value


def pinned(root, relative, expected, limit=8*1024*1024):
    value = read(root, relative, limit)
    if sha(value) != expected: raise ValueError('Source/notice identity mismatch: ' + str(relative))
    return value


def pinned_native(root, relative, name, expected):
    value = read(root, relative)
    if sha(value) == expected: return value
    expected_unsigned = PINS.get('gccNativeUnsigned', {}).get(name)
    if not expected_unsigned: raise ValueError('Source/notice identity mismatch: '+str(relative))
    # Snapshot bounded bytes; never strip or alter the delivered binary itself.
    with tempfile.TemporaryDirectory(prefix='kwip-native-correspondence-') as directory:
        captured = Path(directory)/name
        captured.write_bytes(value)
        subprocess.run(['/usr/bin/codesign','--remove-signature',str(captured)],
                       check=True,capture_output=True,timeout=30)
        if sha(captured.read_bytes()) != expected_unsigned:
            raise ValueError('Source/notice identity mismatch: '+str(relative))
    return value


def tree_records(root, relative):
    base = Path(root) / relative
    if base.is_symlink() or not base.is_dir(): raise ValueError('Source header directory is missing')
    records, total, count = [], 0, 0
    for path in sorted(base.rglob('*')):
        count += 1
        if count > 2048 or path.is_symlink(): raise ValueError('Source tree exceeds limits or contains symbolic links')
        if path.is_dir(): continue
        value = read(root, path.relative_to(root)); total += len(value)
        if total > 32*1024*1024: raise ValueError('Source tree exceeds byte limit')
        records.append({'path': path.relative_to(base).as_posix(), 'bytes':len(value), 'sha256':sha(value)})
    return records


def prepare_source_disclosure(runtime_root, gcc_source_directory=None):
    runtime = Path(runtime_root)
    gcc = gcc_source_directory or DEFAULT_GCC_DIRECTORY
    if gcc is None: raise ValueError('Reviewed GCC source directory is required')
    gcc = Path(gcc)
    inventory_bytes = read(runtime, 'license-inventory.json', 2*1024*1024)
    inventory = json.loads(inventory_bytes)
    distributions = inventory.get('distributions', [])
    if inventory.get('schemaVersion') != 1 or not 1 <= len(distributions) <= 128 or inventory.get('unresolvedNativeComponentNotices', []):
        raise ValueError('License inventory is incomplete or unsupported')
    identities = [(row['name'].lower(), row['version']) for row in distributions]
    if len(set(name for name,_ in identities)) != len(identities): raise ValueError('Duplicate distribution identity')
    for name,version in [('certifi','2026.7.22'),('tensorflow','2.21.0'),('scipy','1.17.0')]:
        if (name,version) not in identities: raise ValueError('Unsupported distribution release: ' + name)
        metadata = read(runtime, SITE + f'{name}-{version}.dist-info/METADATA').decode()
        if f'Name: {name}\n' not in metadata or f'Version: {version}\n' not in metadata: raise ValueError('Installed release differs from inventory')
    paths = set()
    for row in distributions:
        if not row.get('noticeFiles'): raise ValueError('Distribution notice is missing')
        paths.update(row['noticeFiles'])
    paths.add(inventory['pythonNotice'])
    paths.update(row['copiedTo'] for row in inventory.get('nativeComponentNotices', []))
    if not 1 <= len(paths) <= 512 or any(not isinstance(path,str) or not path.startswith('licenses/') for path in paths):
        raise ValueError('Unsafe notice inventory')
    notices, total = [], 0
    for path in sorted(paths):
        value = read(runtime, path); total += len(value)
        if total > 32*1024*1024: raise ValueError('Notice inventory exceeds byte bound')
        notices.append({'path':path,'bytes':len(value),'sha256':sha(value)})
    standalone_raw = read(runtime, 'standalone-license-evidence/inventory.json', 2*1024*1024)
    standalone = json.loads(standalone_raw)
    if standalone.get('schemaVersion') != 1 or not 1 <= len(standalone.get('notices',[])) <= 128: raise ValueError('Standalone notice inventory is invalid')
    for row in standalone['notices']:
        if not row['path'].startswith('licenses/'): raise ValueError('Unsafe standalone notice path')
        path = 'standalone-license-evidence/' + row['path']
        value = pinned(runtime,path,row['sha256'])
        notices.append({'path':path,'bytes':len(value),'sha256':sha(value)})
    ca_path = SITE + 'certifi/cacert.pem'
    ca = pinned(runtime,ca_path,PINS['certifiSource'])
    pinned(runtime,SITE+'certifi-2026.7.22.dist-info/licenses/LICENSE',PINS['certifiLicense'])
    tf_notice = pinned(runtime,SITE+'tensorflow/THIRD_PARTY_NOTICES.txt',PINS['tensorflowNotices'])
    pinned(runtime,SITE+'tensorflow/include/LICENSE',PINS['tensorflowLicense'])
    if b'20f4d829087ef6ed6a502859c6fd093065e806d0' not in tf_notice: raise ValueError('Eigen source notice pointer is absent')
    headers_path = SITE + 'tensorflow/include/Eigen'
    headers = tree_records(runtime,headers_path)
    if len(headers) != PINS['eigenFiles'] or sha(canonical(headers)) != PINS['eigenTree']: raise ValueError('Eigen header source identity mismatch')
    unsupported_path=SITE+'tensorflow/include/unsupported/Eigen'
    unsupported=tree_records(runtime,unsupported_path)
    if len(unsupported)!=PINS['eigenUnsupportedFiles'] or sha(canonical(unsupported))!=PINS['eigenUnsupportedTree']: raise ValueError('Unsupported Eigen source identity mismatch')
    eigen_proof_raw=pinned(NOTICES,'eigen-SOURCE_PROVENANCE.json',EIGEN_PROVENANCE_SHA256,131072)
    eigen_proof=json.loads(eigen_proof_raw)
    actual_header_hashes={'Eigen/'+row['path']:row['sha256'] for row in headers}
    actual_header_hashes.update({'unsupported/Eigen/'+row['path']:row['sha256'] for row in unsupported})
    if eigen_proof['headerSHA256']!=actual_header_hashes or eigen_proof['actualFiles']!=len(actual_header_hashes): raise ValueError('Eigen source provenance does not match bundled headers')
    eigen_payloads={'sources/eigen/SOURCE_PROVENANCE.json':eigen_proof_raw}
    for row in eigen_proof['sourceMaterials']:
        name=row['file']
        if not isinstance(name,str) or Path(name).name!=name: raise ValueError('Unsafe Eigen source notice path')
        value=pinned(NOTICES,'eigen-'+name,row['sha256'],65536)
        if len(value)!=row['bytes']: raise ValueError('Eigen source notice size mismatch')
        eigen_payloads['sources/eigen/'+name]=value
    provenance_raw = pinned(NOTICES,'gcc-13.4-source-provenance.json',PROVENANCE_SHA256,65536)
    if read(gcc,'SOURCE_PROVENANCE.json',65536) != provenance_raw: raise ValueError('GCC supplier correspondence proof differs from reviewed evidence')
    provenance = json.loads(provenance_raw)
    payloads = {'RUNTIME_MODIFICATIONS.md':pinned(NOTICES,'RUNTIME_MODIFICATIONS.md',PINS['replacementGuide'],65536),
                'COPYING.LIB':pinned(NOTICES,'gcc-13.4.0-libquadmath-COPYING.LIB',PINS['lgpl'],65536),
                'sources/Homebrew-LICENSE.txt':pinned(NOTICES,'homebrew-core-gcc13-LICENSE.txt',PINS['homebrew'],65536),
                'sources/BUILD_CONTEXT.md':pinned(NOTICES,'gcc-13.4-BUILD_CONTEXT.md',PINS['buildContext'],65536),
                'sources/HomebrewPatches-LICENSE.txt':pinned(NOTICES,'homebrew-formula-patches-LICENSE.txt',PINS['formulaPatches'],65536),
                'sources/RunnerImages-LICENSE.txt':pinned(NOTICES,'actions-runner-images-LICENSE.txt',PINS['runnerImages'],65536),
                'certifi/cacert.pem':ca,'sources/SOURCE_PROVENANCE.json':provenance_raw}
    payloads.update(eigen_payloads)
    source_records, source_total = [], 0
    for row in provenance['sourceMaterials']:
        if Path(row['file']).name != row['file']: raise ValueError('Unsafe GCC source member')
        value = pinned(gcc,row['file'],row['sha256'],128*1024*1024)
        source_total += len(value)
        if len(value) != row['bytes'] or source_total > 128*1024*1024: raise ValueError('GCC source size differs from reviewed material')
        payloads['sources/'+row['file']] = value
        source_records.append(dict(row, bundledPath='sources/'+row['file']))
    natives=[]
    for name,expected in sorted(PINS['gccNative'].items()):
        path=SITE+'scipy/.dylibs/'+name;value=pinned_native(runtime,path,name,expected)
        natives.append({'path':path,'sha256':sha(value),'bytes':len(value)})
    index={'schemaVersion':1,'redistributionReviewComplete':False,'gccCorrespondingSourceComplete':True,
           'sourceProvenanceVerified':True,'sourceIncluded':True,'replacementRuntimeVerified':True,
           'replacementVerificationScope':'Local ad-hoc copy with separate bundle identifier and sandboxed app execution; ABI-identical supplier library with changed loader identity. Rebuilt numerical logic, other Macs and an App Store processed copy were not tested.',
           'replacementGuide':{'bundledPath':'RUNTIME_MODIFICATIONS.md','sha256':PINS['replacementGuide']},
           'companyRedistributionApproval':False,'appleAcceptanceVerified':False,
           'licenseInventorySHA256':sha(inventory_bytes),'standaloneInventorySHA256':sha(standalone_raw),
           'supplementalLicenses':[{'bundledPath':name,'sha256':sha(payloads[name]),'bytes':len(payloads[name]),'sourceURL':url} for name,url in [
               ('COPYING.LIB','https://raw.githubusercontent.com/gcc-mirror/gcc/releases/gcc-13.4.0/libquadmath/COPYING.LIB'),
               ('sources/Homebrew-LICENSE.txt','https://raw.githubusercontent.com/Homebrew/homebrew-core/22c7493c23f279ae6c84e0a5c7d83e6a242fe9ec/LICENSE.txt'),
               ('sources/HomebrewPatches-LICENSE.txt','https://raw.githubusercontent.com/Homebrew/formula-patches/8bbfd6c920e63f6ed1bf143771bfbfed0a3aaa1d/LICENSE'),
               ('sources/RunnerImages-LICENSE.txt','https://raw.githubusercontent.com/actions/runner-images/1932eb5b1a45b0778ed8d71b97bc0bc691478c60/LICENSE')]],
           'buildContext':{'bundledPath':'sources/BUILD_CONTEXT.md','sha256':PINS['buildContext']},
           'notices':notices,'gcc':{'provenanceSHA256':PROVENANCE_SHA256,'nativeBindings':natives,'materials':source_records},
           'certifi':{'version':'2026.7.22','runtimePath':ca_path,'bundledPath':'certifi/cacert.pem','sha256':sha(ca),
                      'sourceURL':'https://raw.githubusercontent.com/certifi/python-certifi/f4bc676bc101fe2235846e37044e8c693d6cbaf4/certifi/cacert.pem'},
           'eigen':{'noticeDeclaredVersion':'3.3.90','noticeDeclaredCommit':'20f4d829087ef6ed6a502859c6fd093065e806d0','bundledSourceRevisionVerified':True,'actualSourceCommit':eigen_proof['actualEigenCommit'],'actualSourceVersion':eigen_proof['actualEigenVersion'],'actualSourceURL':eigen_proof['archiveURL'],'provenance':{'bundledPath':'sources/eigen/SOURCE_PROVENANCE.json','sha256':EIGEN_PROVENANCE_SHA256},'runtimePath':headers_path,
                    'fileCount':len(headers),'treeSHA256':PINS['eigenTree'],'files':headers,
                    'unsupported':{'runtimePath':unsupported_path,'fileCount':len(unsupported),'treeSHA256':PINS['eigenUnsupportedTree'],'files':unsupported},
                    'noticeSourceURL':'https://third-party-mirror.googlesource.com/eigen/+/20f4d829087ef6ed6a502859c6fd093065e806d0'}}
    readme='''# Open-source notices and source materials

In Finder, choose Show Package Contents on Kwip ML, then open
Contents/XPCServices/DatabaseSourceService.xpc/Contents/Resources/KwipMLRuntime/source-disclosure/README.md.
Older editions use Contents/Resources/KwipMLRuntime/source-disclosure/README.md.
All paths below are relative to KwipMLRuntime unless stated otherwise.

Third-party licenses govern their covered components. Kwip ML terms do not restrict
rights those licenses grant to use, copy, modify, redistribute or inspect covered
materials. Read each complete license for its conditions and warranty disclaimers.
This index is an access guide, not a replacement license or new written source offer.

## Notices

Full retained notices: `licenses/` and `standalone-license-evidence/licenses/`.
`SOURCE_INDEX.json` records every inventoried notice location and SHA-256.

## Mozilla CA source (Certifi 2026.7.22)

The exact unchanged MPL-2.0 CA source is `source-disclosure/certifi/cacert.pem`,
also at `python/lib/python3.13/site-packages/certifi/cacert.pem`.
Its full license remains in the Certifi notice under `licenses/certifi/`.
Exact source: https://raw.githubusercontent.com/certifi/python-certifi/f4bc676bc101fe2235846e37044e8c693d6cbaf4/certifi/cacert.pem
MPL-2.0 terms: https://www.mozilla.org/MPL/2.0/.

## Eigen source in TensorFlow 2.21.0

Bundled source headers are at `python/lib/python3.13/site-packages/tensorflow/include/Eigen`
and `python/lib/python3.13/site-packages/tensorflow/include/unsupported/Eigen`
(including Tensor functionality). TensorFlow third-party notices, including MPL
terms and other applicable component terms, remain under `licenses/tensorflow/`.
That notice declares version 3.3.90 and commit
20f4d829087ef6ed6a502859c6fd093065e806d0; this historical notice pointer is not an
attestation of the installed headers' actual revision. The installed headers identify
5.0.1-dev+master and match TensorFlow's pinned source commit
dcbaf2d608f306450f1e74949eb87e9a22a7ef4b byte for byte in both trees.
`sources/eigen/SOURCE_PROVENANCE.json` retains the exact correspondence evidence,
recipe and source URL. Full MPL/BSD and MINPACK attribution are in `sources/eigen/`.
SOURCE_INDEX.json pins the complete bundled source bytes. This identifies the
provided source form; it does not promise reproducible builds with arbitrary tools
or relicense other sources.

## GCC 13.4 runtime libraries and libquadmath

Complete source and retained build context are bundled in `source-disclosure/sources/`:
`gcc-13.4.0.tar.xz`, the exact Darwin patch, the supplier Homebrew formula,
SciPy wheel workflow and runner-image context. The archive includes GCC GPL terms,
GCC Runtime Library Exception and libquadmath LGPL terms. `source-disclosure/COPYING.LIB`
is the full unchanged LGPL-2.1 text. Homebrew BSD-2-Clause and runner-images MIT
attribution are in `sources/Homebrew-LICENSE.txt`, `sources/HomebrewPatches-LICENSE.txt` and
`sources/RunnerImages-LICENSE.txt`. Extraction, patch application and build recipe
access instructions are in `sources/BUILD_CONTEXT.md`.
SciPy workflow attribution remains in the existing SciPy BSD notice.

SOURCE_PROVENANCE.json records verified supplier/build correspondence for libgcc,
libgfortran and libquadmath. These files are provided with the app; no download
service or three-year written source offer is substituted for them. Applicable
licenses permit modification under their conditions, including rights relating to
use of a modified LGPL library and debugging those modifications.

## Local library replacement

Read `RUNTIME_MODIFICATIONS.md` for the reviewed local-copy replacement procedure
and its precise verification limits. A separate ad-hoc signed application identity
with the sandbox preserved completed the real logistic-regression workflow using
an ABI-identical supplier library with a changed loader identity and signature.
This verifies that local replacement mechanism on the tested Mac; rebuilt numerical
logic, other Macs and an App Store processed copy were not tested. Company terms
review and Apple acceptance remain separate pending release gates. Source inclusion
and this bounded experiment do not certify either.
'''
    payloads['README.md']=readme.encode();payloads['SOURCE_INDEX.json']=json.dumps(index,indent=2,sort_keys=True).encode()+b'\n'
    return index,payloads


def publish_exclusive(staging,destination):
    if sys.platform!='darwin': raise OSError('Exclusive disclosure publication currently requires macOS')
    libc=ctypes.CDLL(None,use_errno=True);rename=libc.renamex_np
    rename.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_uint];rename.restype=ctypes.c_int
    if rename(os.fsencode(staging),os.fsencode(destination),4)!=0:
        code=ctypes.get_errno()
        if code in (errno.EEXIST,errno.ENOTEMPTY):raise FileExistsError('Disclosure destination already exists')
        raise OSError(code,os.strerror(code))


def stage_source_disclosure(runtime_root,gcc_source_directory=None):
    runtime=Path(runtime_root);dest=runtime/'source-disclosure'
    if dest.exists() or dest.is_symlink():raise FileExistsError('Refusing existing source disclosure')
    index,payloads=prepare_source_disclosure(runtime,gcc_source_directory)
    staging=Path(tempfile.mkdtemp(prefix='.source-disclosure-',dir=runtime))
    try:
        for relative,data in payloads.items():
            path=staging/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        publish_exclusive(staging,dest)
        return index
    finally:
        shutil.rmtree(staging,ignore_errors=True)


def verify_source_disclosure(runtime_root):
    runtime=Path(runtime_root);dest=runtime/'source-disclosure'
    if dest.is_symlink() or not dest.is_dir():raise ValueError('Existing source disclosure is missing')
    index,payloads=prepare_source_disclosure(runtime,dest/'sources')
    actual=set();directories=set()
    for path in dest.rglob('*'):
        if path.is_symlink():raise ValueError('Disclosure contains symbolic links')
        if path.is_dir():directories.add(path.relative_to(dest).as_posix());continue
        relative=path.relative_to(dest).as_posix();actual.add(relative)
        if relative not in payloads or read(dest,relative,128*1024*1024)!=payloads[relative]:raise ValueError('Existing source disclosure is stale or altered')
    expected_dirs={parent.as_posix() for name in payloads for parent in Path(name).parents if parent.as_posix()!='.'}
    if actual!=set(payloads) or directories!=expected_dirs:raise ValueError('Existing disclosure inventory differs from reviewed output')
    return index


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--runtime-root',type=Path,required=True)
    parser.add_argument('--gcc-source-directory',type=Path);parser.add_argument('--verify-existing',action='store_true')
    args=parser.parse_args()
    if args.verify_existing:
        if args.gcc_source_directory:parser.error('--verify-existing reads bundled source; do not supply external GCC material')
        result=verify_source_disclosure(args.runtime_root)
        print('Verified retained sources and notices; replacement, company terms and Apple gates remain separate.')
    else:
        if args.gcc_source_directory is None:parser.error('--gcc-source-directory is required for staging')
        result=stage_source_disclosure(args.runtime_root,args.gcc_source_directory)
        print('Staged retained corresponding sources and notices; no overall redistribution approval claimed.')
