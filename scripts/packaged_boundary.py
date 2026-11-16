#!/usr/bin/env python3
"""Retain signed-app observations before submitting a new candidate to Apple.

This validates evidence identity, not the truth of a human observation. Never turn
source tests or a POC launch into a packaged native observation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat

CASES = ('launch-resources', 'export-model-paired', 'export-trained-package',
         'export-evaluation-replay', 'export-program-package', 'export-code-package',
         'api-database-brokers', 'train-inspect-generate-save',
         'history-replay-offline', 'canvas-interaction')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def tree_digest(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Expected a real app directory')
    records = []
    for path in sorted(root.rglob('*')):
        info = path.lstat()
        kind = stat.S_IFMT(info.st_mode)
        if stat.S_ISLNK(info.st_mode):
            content = os.readlink(path)
        elif stat.S_ISREG(info.st_mode):
            content = digest(path)
        elif stat.S_ISDIR(info.st_mode):
            content = None
        else:
            raise ValueError('Unexpected special file in app')
        records.append((str(path.relative_to(root)), kind, stat.S_IMODE(info.st_mode), content))
    return hashlib.sha256(json.dumps(records, separators=(',', ':')).encode()).hexdigest()


def read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= 1024 * 1024:
        raise ValueError('Expected bounded regular evidence')
    return json.loads(path.read_text())


def evidence_file(folder, name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('Evidence must be retained beneath the boundary directory')
    result = folder / path
    if any(p.is_symlink() for p in [result, *result.parents]) or not result.is_file() or result.stat().st_size == 0:
        raise ValueError('Missing or indirect native evidence')
    return result


def validate(value, release, app, folder):
    if value.get('schemaVersion') != 1 or value.get('coordinates') != release['coordinates']:
        raise ValueError('Boundary source/runtime/build differs')
    if (value.get('appZipSHA256') != release['appZipSHA256']
            or value.get('appTreeSHA256') != release.get('assembledAppTreeSHA256')
            or value.get('appTreeSHA256') != tree_digest(app)):
        raise ValueError('Signed payload changed after native observations')
    cases = value.get('cases', {})
    if set(cases) != set(CASES):
        raise ValueError('Every packaged boundary case must be observed')
    for name, case in cases.items():
        if case.get('result') != 'passed' or not isinstance(case.get('observation'), str) or not case['observation'].strip():
            raise ValueError('Native boundary failed or unobserved: ' + name)
        if digest(evidence_file(folder, case['evidence'])) != case.get('sha256'):
            raise ValueError('Native evidence changed: ' + name)
    return value


def require_gate(out, release):
    if release.get('packagedBoundaryRequired') is not True:
        return  # Already assembled historical candidates retain their original contract.
    folder = Path(out) / 'packaged-boundary'
    receipt = folder / 'observations.json'
    value = validate(read(receipt), release, Path(out) / 'Kwip ML.app', folder)
    if digest(Path(out) / 'app-notary.zip') != release['appZipSHA256']:
        raise ValueError('Submission archive changed')
    return digest(receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=['start', 'record', 'verify'])
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--case', choices=CASES)
    parser.add_argument('--evidence')
    parser.add_argument('--observation')
    parser.add_argument('--result', choices=['passed', 'failed'])
    args = parser.parse_args()
    release = read(args.output / 'release.json')
    if release['phase'] != 'assembled':
        raise ValueError('Early boundary observations require the assembled, unsubmitted app')
    folder = args.output / 'packaged-boundary'
    path = folder / 'observations.json'
    if args.step == 'start':
        if tree_digest(args.output / 'Kwip ML.app') != release.get('assembledAppTreeSHA256'):
            raise ValueError('App differs from assembled artifact before observations')
        folder.mkdir(exist_ok=True)
        value = dict(schemaVersion=1, coordinates=release['coordinates'], appZipSHA256=release['appZipSHA256'],
                     appTreeSHA256=tree_digest(args.output / 'Kwip ML.app'), cases={},
                     startedAt=datetime.now(timezone.utc).isoformat(), finalInstallerUAT=False)
        with path.open('x') as stream:
            json.dump(value, stream, indent=2)
    elif args.step == 'record':
        if not all([args.case, args.evidence, args.observation, args.result]):
            parser.error('record requires case, retained evidence, observation and explicit result')
        value = read(path)
        if args.case in value['cases']:
            raise ValueError('Preserve the previous observation; investigate rather than overwrite it')
        value['cases'][args.case] = dict(evidence=args.evidence, sha256=digest(evidence_file(folder, args.evidence)),
                                       observation=args.observation, result=args.result,
                                       observedAt=datetime.now(timezone.utc).isoformat())
        temporary = path.with_suffix('.tmp')
        with temporary.open('x') as stream:
            json.dump(value, stream, indent=2)
        temporary.replace(path)
    else:
        validate(read(path), release, args.output / 'Kwip ML.app', folder)
        print('Packaged observations verified; final installer and distribution acceptance remain required.')


if __name__ == '__main__':
    main()
