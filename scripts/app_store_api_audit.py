#!/usr/bin/env python3
"""Fail closed on exact native symbol strings rejected in Kwip ML build 1.

This targeted regression check is not a complete private-API database or Apple
validation. Namespaced OpenBLAS implementations are distinct from Apple imports.
"""
import argparse
import json
import mmap
from pathlib import Path

MAGICS = {bytes.fromhex(value) for value in ('cffaedfe', 'cefaedfe', 'feedfacf', 'feedface', 'cafebabe', 'cafebabf', 'bebafeca', 'bfbafeca')}
REJECTED = ('_NSWindowDidOrderOnScreenNotification', '_dcabs1_', '_lsame_', '_xerbla_array__')


def inspect_native(path):
    """Return None for non-native files, otherwise exact rejected C strings."""
    with Path(path).open('rb') as stream:
        if stream.read(4) not in MAGICS:
            return None
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as content:
            return [symbol for symbol in REJECTED if content.find(b'\0' + symbol.encode('ascii') + b'\0') >= 0]


def audit(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Audit root must be a real directory')
    root = root.resolve()
    count, findings = 0, []
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            if not path.exists() or not path.resolve().is_relative_to(root):
                raise ValueError('Broken or escaping audit symlink: ' + str(path.relative_to(root)))
            continue
        if not path.is_file():
            continue
        hits = inspect_native(path)
        if hits is None:
            continue
        count += 1
        if hits:
            findings.append({'path': str(path.relative_to(root)), 'symbols': hits})
    if findings:
        raise ValueError('App Store rejected API references: ' + json.dumps(findings))
    if not count:
        raise ValueError('No native files found for API audit')
    return {'schemaVersion': 1, 'nativeFilesScanned': count, 'rejectedSymbols': list(REJECTED), 'findings': []}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('--report')
    args = parser.parse_args()
    result = audit(args.root)
    if args.report:
        Path(args.report).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
