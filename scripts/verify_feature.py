#!/usr/bin/env python3
"""Focused development checks; never substitutes for complete qualification or native UAT."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
try:
    from .verification_lock import compute_lock, lock_path
except ImportError:
    from verification_lock import compute_lock, lock_path


def validate_manifest(value, root=ROOT):
    if value.get('schemaVersion') != 1 or not value.get('features'):
        raise ValueError('Unknown verification boundary manifest')
    for feature, item in value['features'].items():
        if not re.fullmatch('[a-z][a-z0-9-]+', feature) or not item.get('boundary'):
            raise ValueError('Invalid feature identity')
        for key in ('python', 'swift', 'nativeCases', 'resources', 'requiredEnvironment'):
            if not isinstance(item.get(key), list) or len(set(item[key])) != len(item[key]):
                raise ValueError('Missing or duplicate boundary metadata')
        for path in item['python']:
            if Path(path).is_absolute() or '..' in Path(path).parts or not (root / path).is_file():
                raise ValueError('Unknown Python selection: ' + path)
        for name in item['swift']:
            if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]+(?:/[A-Za-z][A-Za-z0-9_]+)?', name) or not list((root / 'Tests').rglob(name.split('/')[0] + '.swift')):
                raise ValueError('Unknown Swift test class: ' + name)
    return value


def audit_junit(path):
    cases = list(ET.parse(path).iter('testcase'))
    if not cases:
        raise ValueError('Selection executed no tests')
    identities = [(c.get('classname'), c.get('name')) for c in cases]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate test identities')
    if any(c.find(tag) is not None for c in cases for tag in ('failure', 'error', 'skipped')):
        raise ValueError('Required feature selection failed or skipped; no success receipt')
    return {'executed': len(cases), 'seconds': sum(float(c.get('time', 0)) for c in cases),
            'slowest': sorted([{'id': '.'.join(i), 'seconds': float(c.get('time', 0))}
                               for i, c in zip(identities, cases)], key=lambda x: -x['seconds'])[:20]}


def swift_selection(output, selectors):
    discovered = [line.strip() for line in output.splitlines()
                  if re.fullmatch(r"[A-Za-z0-9_]+\.[A-Za-z0-9_]+/[^\s]+", line.strip())]
    selected, covered = [], set()
    for node in discovered:
        short = node.split('.', 1)[1]
        matches = {s for s in selectors if short == s or ('/' not in s and short.split('/')[0] == s)}
        if matches:
            selected.append(node)
            covered.update(matches)
    if len(set(selected)) != len(selected) or covered != set(selectors):
        raise ValueError('Missing or duplicate Swift selection')
    return selected


def audit_partitions(data, discovery):
    from collections import Counter
    for parent, children in data.get('partitions', {}).items():
        whole = data['features'][parent]
        parts = [data['features'][name] for name in children]
        if Counter(whole['python']) != Counter(p for part in parts for p in part['python']):
            raise ValueError('Python partition omits or duplicates files')
        expected = swift_selection(discovery, whole['swift'])
        actual = [node for part in parts for node in swift_selection(discovery, part['swift'])]
        if Counter(expected) != Counter(actual):
            raise ValueError('Swift partition omits or duplicates discovered tests')
    return True


def swift_junit(log, expected, destination):
    # Serial XCTest on this supported Mac does not emit SwiftPM xUnit output.
    # Reconcile actual start/terminal events against fresh runtime discovery.
    starts = re.findall(r"^Test Case '-\[([^ ]+) ([^]]+)\]' started\.", log, re.M)
    results = re.findall(r"^Test Case '-\[([^ ]+) ([^]]+)\]' (passed|failed|skipped) \(([0-9.]+) seconds\)\.", log, re.M)
    from collections import Counter
    if Counter(a + '/' + b for a, b in starts) != Counter(expected) or Counter(a + '/' + b for a, b, _, _ in results) != Counter(expected):
        raise ValueError('Swift discovery/start/completion identities differ')
    suite = ET.Element('testsuite', name='focused-serial-xctest')
    for cls, name, outcome, duration in results:
        case = ET.SubElement(suite, 'testcase', classname=cls, name=name, time=duration)
        if outcome != 'passed':
            ET.SubElement(case, 'skipped' if outcome == 'skipped' else 'failure')
    ET.ElementTree(suite).write(destination, encoding='utf-8', xml_declaration=True)


def source_identity():
    return {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'diffSHA256': hashlib.sha256(subprocess.check_output(['git', 'diff', 'HEAD', '--binary'], cwd=ROOT)).hexdigest(),
            'untracked': {str(p): hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                          for p in subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard'], cwd=ROOT, text=True).splitlines()
                          if (ROOT / p).is_file()}}


def fixture_identity(selection):
    result = {}
    for key in selection['requiredEnvironment']:
        root = Path(os.environ[key]).resolve(strict=True)
        if not root.is_dir():
            raise ValueError('Expected explicit development fixture root: ' + key)
        files = sorted(p for p in (root / 'runner').rglob('*') if p.is_file() and p.suffix in {'.py', '.json'})
        if not files or not (root / 'requirements-runner.txt').is_file():
            raise ValueError('Missing development worker fixture: ' + key)
        files.append(root / 'requirements-runner.txt')
        inventory = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        result[key] = {'root': str(root), 'workerAndRequirementsSHA256': hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature')
    parser.add_argument('--inventory', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    manifest = ROOT / 'scripts/verification_boundaries.json'
    data = validate_manifest(json.loads(manifest.read_text()))
    if args.inventory:
        known = {path for feature in data['features'].values() for path in feature['python']}
        known_swift = {name.split('/')[0] + '.swift' for feature in data['features'].values() for name in feature['swift']}
        unknown_python = sorted(str(p.relative_to(ROOT)) for directory in ('scripts/tests', 'scripts/runtime_tests', 'runner/tests')
                                for p in (ROOT / directory).rglob('test_*.py') if str(p.relative_to(ROOT)) not in known)
        unknown_swift = sorted(str(p.relative_to(ROOT)) for p in (ROOT / 'Tests').rglob('*Tests.swift') if p.name not in known_swift)
        print(json.dumps({'features': data['features'], 'unclassifiedPythonFiles': unknown_python,
                          'unclassifiedSwiftFiles': unknown_swift, 'unknownPolicy': data['unknownPolicy']}, indent=2))
        return
    if args.feature not in data['features']:
        parser.error('Unclassified feature: choose explicit tests and retain complete qualification; do not silently omit it')
    selection = data['features'][args.feature]
    for key in selection['requiredEnvironment']:
        if not os.environ.get(key):
            parser.error('Missing required fixture environment: ' + key)
    if args.plan:
        print(json.dumps(selection, indent=2))
        return
    if args.output is None:
        parser.error('--output is required for execution')
    args.output = args.output.resolve()
    with compute_lock(lock_path(ROOT), 'feature') as compute_fd:
        args.output.mkdir(parents=True, exist_ok=False)
        source = source_identity()
        fixtures = fixture_identity(selection)
        started = time.monotonic()
        reports = {}
        commands = []
        if selection['python']:
            commands.append(('python', [sys.executable, '-m', 'pytest', '-q', '--strict-markers', *selection['python'], '--junitxml', str(args.output / 'python.xml')]))
        if selection['swift']:
            with (args.output / 'swift-collection.log').open('w') as log:
                subprocess.run(['swift', 'test', 'list'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, pass_fds=(compute_fd,))
            discovery = (args.output / 'swift-collection.log').read_text()
            audit_partitions(data, discovery)
            expected_swift = swift_selection(discovery, selection['swift'])
            (args.output / 'swift-selection.json').write_text(json.dumps(expected_swift, indent=2))
            commands.append(('swift', ['swift', 'test', '--skip-build', '--filter', '(' + '|'.join('^' + re.escape(node) + '$' for node in expected_swift) + ')']))
        for lane, command in commands:
            with (args.output / (lane + '.log')).open('w') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, pass_fds=(compute_fd,))
            if lane == 'swift':
                swift_junit((args.output / 'swift.log').read_text(), expected_swift, args.output / 'swift.xml')
            reports[lane] = audit_junit(args.output / (lane + '.xml'))
        if source_identity() != source or fixture_identity(selection) != fixtures:
            raise ValueError('Source changed during verification')
        receipt = dict(schemaVersion=1, feature=args.feature, source=source, fixtures=fixtures, selection=selection,
                       manifestSHA256=hashlib.sha256(manifest.read_bytes()).hexdigest(), reports=reports,
                       fullBaseline=False, nativeUAT=False, releaseAcceptance=False,
                       wallSeconds=time.monotonic()-started, completedAt=datetime.now(timezone.utc).isoformat())
        (args.output / 'FEATURE_PASSED.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
