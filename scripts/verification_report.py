#!/usr/bin/env python3
"""Read-only local verification dashboard; no checks are run or inferred."""
import argparse
import json
from pathlib import Path


def render(receipts, inventory=None, release=None):
    lines = ['# Verification evidence', '', 'Focused success is not full-baseline, native or distribution acceptance.', '',
             '| Feature | Commit | Executed | Wall seconds | Resources |', '|---|---|---:|---:|---|']
    for value in receipts:
        if any(value.get(key) is not False for key in ('fullBaseline','nativeUAT','releaseAcceptance')):
            raise ValueError('Expected a focused verification receipt')
        clean = not value['source']['untracked'] and value['source']['diffSHA256'] == __import__('hashlib').sha256(b'').hexdigest()
        lines.append('| {feature} | {commit}{dirty} | {count} | {seconds:.2f} | {resources} |'.format(
            feature=value['feature'], commit=value['source']['commit'][:12], dirty='' if clean else ' + local changes',
            count=sum(r['executed'] for r in value['reports'].values()), seconds=value['wallSeconds'], resources=', '.join(value['selection']['resources'])))
    if inventory:
        lines += ['', '## Selection coverage', '',
                  f"Unclassified Python files: {len(inventory['unclassifiedPythonFiles'])}. Unclassified Swift files: {len(inventory['unclassifiedSwiftFiles'])}.",
                  'These remain in complete qualification. Reviewed partitions must reconcile their exact discovered test union; this is not a reduced release suite.']
    if release:
        lines += ['', '## Candidate (observed receipt only)', '', f"Build {release['coordinates']['build']}: {release['phase']}. No publication is performed by this report.", '',
                  '| Observed phase | UTC timestamp |', '|---|---|']
        lines += [f"| {e['phase']} | {e['observedAt']} |" for e in release.get('phaseEvents', [])]
        if not release.get('phaseEvents'): lines += ['No structured timing exists for this historical receipt; none has been invented.']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', action='append', required=True, type=Path)
    parser.add_argument('--inventory', type=Path)
    parser.add_argument('--release', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    def read(path): return json.loads(path.read_text()) if path else None
    text = render([read(p) for p in args.receipt], read(args.inventory), read(args.release))
    with args.output.open('x') as stream: stream.write(text)


if __name__ == '__main__': main()
