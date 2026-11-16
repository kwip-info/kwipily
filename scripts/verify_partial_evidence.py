#!/usr/bin/env python3
"""Offline integrity checking only; never grants capability acceptance or coverage."""
import argparse
import hashlib
import json
import os
import stat
import re
import subprocess
from pathlib import Path, PurePosixPath

MAX_BYTES = 8 * 1024 * 1024
HEX64 = re.compile(r'[0-9a-f]{64}\Z')
COMMIT = re.compile(r'(?:[0-9a-f]{40}|[0-9a-f]{64})\Z')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def shape(value, required, optional=()):
    require(isinstance(value, dict), 'Expected object')
    require(set(required) <= value.keys() and value.keys() <= set(required) | set(optional), 'Missing or unknown fields')


def text(value):
    require(isinstance(value, str) and bool(value.strip()) and len(value) <= 4096, 'Expected bounded nonempty text')


def texts(value):
    require(isinstance(value, list) and 0 < len(value) <= 512, 'Expected nonempty bounded list')
    for item in value:
        text(item)
    require(len(set(value)) == len(value), 'Duplicate list values')


def git(root, *args):
    try:
        return subprocess.check_output(['git', '--no-replace-objects', '-C', str(root), *args], stderr=subprocess.PIPE, timeout=15)
    except (subprocess.SubprocessError, OSError) as exc:
        raise ValueError('Historical Git object is unavailable') from exc


def revision(root, value):
    require(isinstance(value, str) and COMMIT.fullmatch(value), 'Use a full immutable commit identity')
    require(git(root, 'cat-file', '-t', value).strip() == b'commit', 'Expected commit object')


def bounded_regular_read(root, parts):
    """Descriptor-relative traversal: never reopen a checked pathname to read it."""
    require(bool(parts), 'Expected a regular artifact path')
    directory=None;descriptor=None
    try:
        directory=os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        for part in parts[:-1]:
            child=os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory);directory=child
        descriptor=os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        before=os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), 'Reference must be a regular file')
        require(before.st_size<=MAX_BYTES, 'Reference exceeds size bound')
        chunks=[];total=0
        while total<=MAX_BYTES:
            chunk=os.read(descriptor,min(65536,MAX_BYTES+1-total))
            if not chunk:break
            chunks.append(chunk);total+=len(chunk)
        require(total<=MAX_BYTES, 'Reference exceeds size bound')
        after=os.fstat(descriptor)
        require((before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_size,after.st_mtime_ns,after.st_ctime_ns), 'Reference changed during read')
        return b''.join(chunks)
    except OSError as error:
        raise ValueError('Missing, nonregular, or symlink artifact reference') from error
    finally:
        if descriptor is not None:os.close(descriptor)
        if directory is not None:os.close(directory)


def reference(root, value, forced_revision=None):
    shape(value, ['path', 'sha256'], ['gitCommit'])
    path = value['path']; text(path)
    parts = PurePosixPath(path)
    require(not parts.is_absolute() and '..' not in parts.parts and '\\' not in path and ':' not in path,
            'Reference must be a contained repository-relative path')
    require(isinstance(value['sha256'], str) and HEX64.fullmatch(value['sha256']), 'Expected SHA-256')
    if 'gitCommit' in value:
        require(isinstance(value['gitCommit'], str) and COMMIT.fullmatch(value['gitCommit']), 'Use a full immutable commit identity')
    commit = value.get('gitCommit', forced_revision)
    require(not forced_revision or commit == forced_revision, 'Source revision differs from implementation')
    if commit:
        revision(root, commit)
        entry = git(root, 'ls-tree', commit, '--', path)
        require(entry.startswith((b'100644 blob ', b'100755 blob ')), 'Historical reference must be a regular blob')
        object_name = f'{commit}:{path}'
        size = int(git(root, 'cat-file', '-s', object_name))
        require(size <= MAX_BYTES, 'Reference exceeds size bound')
        data = git(root, 'cat-file', 'blob', object_name)
    else:
        data = bounded_regular_read(root, parts.parts)
    require(hashlib.sha256(data).hexdigest() == value['sha256'], f'Hash mismatch: {path}')


def references(root, values, commit=None):
    require(isinstance(values, list) and 0 < len(values) <= 512, 'Expected bounded references')
    paths = []
    for item in values:
        reference(root, item, commit); paths.append(item['path'])
    require(len(set(paths)) == len(paths), 'Duplicate reference paths')


def verify(manifest, root):
    root = Path(root).resolve()
    shape(manifest, ['schemaVersion', 'capabilityID', 'evidenceRevision', 'status', 'catalogVersion',
                     'implementation', 'acceptanceScope', 'contracts', 'environment', 'upstream',
                     'observations', 'openRequirements', 'review'])
    require(type(manifest['schemaVersion']) is int and manifest['schemaVersion'] == 1, 'Unsupported schema')
    require(manifest['status'] == 'partial', 'Only partial evidence is supported; acceptance is separate')
    for key in ['capabilityID', 'evidenceRevision', 'catalogVersion']: text(manifest[key])
    require(manifest['capabilityID'] == 'pytorch.torch.nn.Linear', 'This bounded verifier supports only Linear evidence')
    implementation = manifest['implementation']
    shape(implementation, ['gitCommit', 'sourceFiles'])
    commit = implementation['gitCommit']; revision(root, commit)
    references(root, implementation['sourceFiles'], commit)
    reference(root, manifest['acceptanceScope'])
    shape(manifest['contracts'], ['registry', 'executionDescriptor'])
    for item in manifest['contracts'].values(): reference(root, item)
    environment = manifest['environment']
    shape(environment, ['python', 'torch', 'macOS', 'architecture', 'availableDevices'])
    for key in ['python', 'torch', 'macOS', 'architecture']: text(environment[key])
    texts(environment['availableDevices'])
    upstream = manifest['upstream']
    shape(upstream, ['documentationVersion', 'alignmentStatus'], ['snapshotRefs', 'runtimeSource'])
    text(upstream['documentationVersion'])
    require(upstream['alignmentStatus'] in ['unreviewed', 'partial', 'reviewed'], 'Unknown alignment status')
    if 'snapshotRefs' in upstream: references(root, upstream['snapshotRefs'])
    if 'runtimeSource' in upstream: reference(root, upstream['runtimeSource'])
    texts(manifest['openRequirements'])
    shape(manifest['review'], ['status'])
    require(manifest['review']['status'] == 'pending', 'Verifier does not attest review acceptance')
    observations = manifest['observations']
    require(isinstance(observations, list) and 0 < len(observations) <= 512, 'Expected bounded observations')
    ids = []; counts = {'passed': 0, 'failed': 0, 'skipped': 0}
    for item in observations:
        shape(item, ['id', 'kind', 'modes', 'dimensions', 'cases', 'outcome', 'commandOrProcedure',
                     'implementationRevision', 'evidenceFiles'], ['reason', 'candidateIdentity'])
        text(item['id']); ids.append(item['id']); text(item['commandOrProcedure'])
        for key in ['modes', 'dimensions', 'cases']: texts(item[key])
        require(set(item['modes']) <= {'inference', 'training'}, 'Unknown Linear mode')
        require(set(item['dimensions']) <= {'configuration', 'execution', 'export', 'validation', 'diagnostics', 'lesson', 'example', 'ui', 'persistence', 'failure_handling'}, 'Unknown evidence dimension')
        require(item['kind'] in ['automated', 'nativeUAT'], 'Unknown observation kind')
        require(item['implementationRevision'] == commit, 'Separate manifests for distinct implementation revisions')
        require(isinstance(item['outcome'], str) and item['outcome'] in counts, 'Unknown outcome')
        counts[item['outcome']] += 1
        if item['outcome'] != 'passed': text(item.get('reason'))
        references(root, item['evidenceFiles'])
        if item['kind'] == 'nativeUAT':
            candidate = item.get('candidateIdentity')
            shape(candidate, ['id', 'appManifest', 'workerManifest'])
            text(candidate['id'])
            reference(root, candidate['appManifest']); reference(root, candidate['workerManifest'])
        else:
            require('candidateIdentity' not in item, 'Native candidate identity belongs to native UAT')
    require(len(set(ids)) == len(ids), 'Duplicate observation IDs')
    return {'integrityValid': True, 'accepted': False, 'coverageCredit': 0,
            'observationCounts': counts, 'warning': 'Integrity verifies references, not observation truth or capability acceptance.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        manifest_bytes=bounded_regular_read(args.manifest.parent.resolve(), (args.manifest.name,))
        result = verify(json.loads(manifest_bytes), args.root)
        print(json.dumps(result, indent=2))
    except (ValueError, OSError) as exc:
        print(json.dumps({'integrityValid': False, 'accepted': False, 'coverageCredit': 0, 'error': str(exc)}))
        return 2
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
