"""Build-time inventory of installed wheel license declarations and shipped notices."""
import hashlib
import importlib.metadata
import os
import re
import stat
import json
from pathlib import Path
import shutil
import sys

# Reviewed installed-wheel layout, deliberately version scoped. Its declaration
# alone cannot substitute for the actual license and bundled third-party notice.
TENSORFLOW_NOTICES = ('tensorflow/include/LICENSE', 'tensorflow/THIRD_PARTY_NOTICES.txt')

def tensorflow_notices(dist, root):
    if dist.metadata['Name'].lower() != 'tensorflow' or dist.version != '2.21.0':
        return []
    base = Path(dist.locate_file('.')).resolve()
    snapshots = []
    for member in TENSORFLOW_NOTICES:
        source = Path(dist.locate_file(member))
        try:
            source.resolve().relative_to(base)
            descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(descriptor, 'rb') as handle:
                info = os.fstat(handle.fileno())
                if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= 8 * 1024 * 1024:
                    raise ValueError('Notice must be a bounded nonempty regular file')
                data = handle.read(8 * 1024 * 1024 + 1)
                if not 0 < len(data) <= 8 * 1024 * 1024:
                    raise ValueError('Notice exceeds size bound')
        except (OSError, ValueError) as error:
            raise ValueError(f'TensorFlow 2.21.0 required notice unavailable: {member}') from error
        snapshots.append((member, data))
    records = []
    for member, data in snapshots:
        target = root / 'licenses' / 'tensorflow' / member
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        records.append({'source': member, 'copiedTo': str(target.relative_to(root)),
                        'sha256': hashlib.sha256(data).hexdigest()})
    return records

# Exact upstream notice sources retained verbatim for deterministic offline builds.
NOTICE_ROOT = Path(__file__).resolve().parent / 'runtime_notices'
PROVENANCE = {
    'tokenizers': {'version':'0.23.3','asset':'tokenizers-0.23.3-LICENSE.txt',
        'sha256':'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4',
        'sourceURL':'https://raw.githubusercontent.com/huggingface/tokenizers/301277072263fb3830eb66eec7e47adadf76f1dc/LICENSE',
        'sourceTag':'v0.23.3','sourceCommit':'301277072263fb3830eb66eec7e47adadf76f1dc','sourceLicenseExpression':'Apache-2.0'},
    'flatbuffers': {'version': '25.12.19', 'asset': 'flatbuffers-25.12.19-LICENSE.txt',
        'sha256': 'cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30',
        'sourceURL': 'https://raw.githubusercontent.com/google/flatbuffers/7e163021e59cca4f8e1e35a7c828b5c6b7915953/LICENSE',
        'sourceTag': 'v25.12.19', 'sourceCommit': '7e163021e59cca4f8e1e35a7c828b5c6b7915953',
        'sourceLicenseExpression': 'Apache-2.0'},
    'namex': {'version': '0.1.0', 'asset': 'namex-0.1.0-LICENSE.txt',
        'sha256': '29aea94ee8d7f78b22d2f7ea626fa513dce47feb17f86c1430ec0d56e3e40a7f',
        'sourceURL': 'https://files.pythonhosted.org/packages/0c/c0/ee95b28f029c73f8d49d8f52edaed02a1d4a9acb8b69355737fdb1faa191/namex-0.1.0.tar.gz',
        'sourceArchiveSHA256': '117f03ccd302cc48e3f5c58a296838f6b89c83455ab8683a1e85f2a430aa4306',
        'sourceMember': 'namex-0.1.0/LICENSE', 'sourceLicenseExpression': 'Apache-2.0'},
    'itcl': {'version': '4.2.4', 'asset': 'itcl-4.2.4-license.terms',
        'sha256': 'b61edfaeead97546bc62b1f205b046f2e440b83bfc517bc0932b93bc3d505865',
        'sourceURL': 'https://raw.githubusercontent.com/tcltk/itcl/a3e8b43c671afc7f0d82bd892325d42c813791ab/license.terms',
        'sourceTag': 'itcl-4-2-4', 'sourceCommit': 'a3e8b43c671afc7f0d82bd892325d42c813791ab'},
    'thread': {'version': '2.8.9', 'asset': 'thread-2.8.9-license.terms',
        'sha256': '0a03981c40f7813ce6ddbdfce9882020bcfd696ea044b21ef07619ed9b86abae',
        'sourceURL': 'https://raw.githubusercontent.com/tcltk/thread/204032ab9d9e557cfe335670eccd38682ce398e2/license.terms',
        'sourceTag': 'thread-2-8-9', 'sourceCommit': '204032ab9d9e557cfe335670eccd38682ce398e2'},
}


def bounded_notice(path):
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, 'rb') as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= 65536:
                raise ValueError('notice must be a bounded nonempty regular file')
            data = handle.read(65537)
            if not 0 < len(data) <= 65536:
                raise ValueError('notice exceeds bound')
            return data
    except OSError as error:
        raise ValueError(f'Required notice unavailable: {path.name}') from error


def copy_provenance_notice(name, root, notice_root):
    evidence = PROVENANCE[name]
    data = bounded_notice(Path(notice_root) / evidence['asset'])
    if hashlib.sha256(data).hexdigest() != evidence['sha256']:
        raise ValueError(f'Upstream notice checksum mismatch: {name}')
    target = Path(root) / 'licenses' / name / 'upstream' / evidence['asset']
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return {**evidence, 'component': name, 'copiedTo': str(target.relative_to(root)),
            'redistributionReviewComplete': False}


def distribution_provenance_notices(dist, root, *, notice_root=NOTICE_ROOT):
    name = dist.metadata['Name'].lower()
    if name not in ('flatbuffers', 'namex', 'tokenizers') or dist.version != PROVENANCE[name]['version']:
        return []
    return [copy_provenance_notice(name, root, notice_root)]


def native_provenance_notices(root, *, notice_root=NOTICE_ROOT):
    root = Path(root)
    records, unresolved = [], []
    for name, package in [('itcl', 'Itcl'), ('thread', 'Thread')]:
        expected = PROVENANCE[name]['version']
        for folder in sorted((root / 'python/lib').glob(name + '*')):
            relative = str(folder.relative_to(root))
            try:
                if folder.is_symlink() or not folder.is_dir() or folder.name != name + expected:
                    raise ValueError('Unreviewed native addon path/version')
                raw = bounded_notice(folder / 'pkgIndex.tcl')
                versions = re.findall(rb'package\s+ifneeded\s+' + package.encode() + rb'\s+([^\s]+)', raw)
                if not versions or any(version != expected.encode() for version in versions):
                    raise ValueError('Native addon package index does not match reviewed version')
            except ValueError as error:
                unresolved.append({'path': relative, 'reason': str(error)})
                continue
            record = copy_provenance_notice(name, root, notice_root)
            record.update(installedPath=relative, observedPackageIndexSHA256=hashlib.sha256(raw).hexdigest())
            records.append(record)
    return records, unresolved


root=Path(sys.argv[1]).resolve(); entries=[]
for dist in sorted(importlib.metadata.distributions(),key=lambda d:d.metadata['Name'].lower()):
    name=dist.metadata['Name']; notices=[]
    for member in dist.files or []:
        lower=str(member).lower(); basename=Path(lower).name
        if '.dist-info/' not in lower or not any(word in basename for word in ('license','notice','copying','copyright')): continue
        source=Path(dist.locate_file(member))
        if not source.is_file(): continue
        target=root/'licenses'/name/str(member)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        notices.append(str(target.relative_to(root)))
    supplemental=distribution_provenance_notices(dist,root)
    notices.extend(record['copiedTo'] for record in supplemental)
    source_expression=next((record.get('sourceLicenseExpression') for record in supplemental if record.get('sourceLicenseExpression')),None)
    mapped=tensorflow_notices(dist,root)
    notices.extend(record['copiedTo'] for record in mapped)
    expression=dist.metadata.get('License-Expression')
    legacy=dist.metadata.get('License')
    classifiers=[v.removeprefix('License :: ') for v in dist.metadata.get_all('Classifier',[]) if v.startswith('License ::')]
    entries.append({'name':name,'version':dist.version,'licenseExpression':expression,'legacyLicense':legacy,'licenseClassifiers':classifiers,'noticeFiles':notices,'frameworkNoticeSources':mapped,'supplementalNoticeSources':supplemental,'sourceLicenseExpression':source_expression,'reviewRequired':not(bool(expression or legacy or classifiers or source_expression) and bool(notices))})
stdlib=root/'python/lib'/f'python{sys.version_info.major}.{sys.version_info.minor}'/'LICENSE.txt'
if stdlib.exists():
    target=root/'licenses/CPython/LICENSE.txt';target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(stdlib,target)
native_notices,unresolved_native_notices=native_provenance_notices(root)
print(json.dumps({'schemaVersion':1,'distributions':entries,'nativeComponentNotices':native_notices,'unresolvedNativeComponentNotices':unresolved_native_notices,'redistributionReviewComplete':False,'pythonNotice':'licenses/CPython/LICENSE.txt','limitations':['Wheel notices and declarations are inventory evidence, not legal clearance.','Transitive native libraries in standalone Python and framework wheels need separate redistribution review.','Astral standalone distribution uses third-party builds, not a python.org macOS framework install.']},indent=2))
