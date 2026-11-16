#!/usr/bin/env python3
"""Local interaction profiling and matched before/after comparisons. No uploads."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import plistlib
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MATCH_FIELDS = ('fixtureVersion', 'nodes', 'iterations', 'configuration', 'machine', 'os', 'cpu',
                'toolchain', 'mode', 'instrumentationVersion', 'harnessSHA256', 'renderEnvironment')


def command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True, env=build_environment()).strip()


def build_environment():
    env = os.environ.copy()
    env.setdefault('DEVELOPER_DIR', '/Applications/Xcode.app/Contents/Developer')
    return env


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source_identity():
    changed = subprocess.check_output(['git', 'diff', '--binary', 'HEAD'], cwd=ROOT)
    untracked = command('git', 'ls-files', '--others', '--exclude-standard').splitlines()
    for name in sorted(untracked):
        path = ROOT / name
        if path.is_file():
            changed += name.encode() + bytes.fromhex(digest(path.read_bytes()))
    return dict(commit=command('git', 'rev-parse', 'HEAD'), dirty=bool(changed), workingChangesSHA256=digest(changed))


def metadata(args, mode):
    harness = b''.join((ROOT / p).read_bytes() for p in (
        'Tests/KwipMLAppTests/InteractionProfileScenarioTests.swift',
        'Sources/KwipMLCore/InteractionProfiler.swift', 'Sources/KwipMLApp/InteractionProfileSession.swift', 'scripts/interaction_profile.py'))
    return dict(fixtureVersion=2, nodes=getattr(args, 'nodes', None), iterations=getattr(args, 'iterations', None),
                configuration=getattr(args, 'configuration', 'manual-unknown'), machine=command('sysctl', '-n', 'hw.model'),
                os=command('sw_vers', '-productVersion') + '/' + command('sw_vers', '-buildVersion'),
                cpu=command('sysctl', '-n', 'hw.logicalcpu') + ' cores/' + command('sysctl', '-n', 'hw.memsize') + ' bytes/' + platform.machine(),
                toolchain=command('swift', '--version'), mode=mode, instrumentationVersion=1, harnessSHA256=digest(harness),
                source=source_identity(), workload='synthetic editor and pipe burst; no ML training' if mode == 'scripted' else 'manual development app; workload not controlled')


def summarize(capture):
    if capture.get('schemaVersion') != 1:
        raise ValueError('Unsupported capture schema')
    if type(capture.get('droppedSamples')) is not int or capture['droppedSamples'] < 0:
        raise ValueError('Invalid dropped sample count')
    samples = capture.get('samples')
    if not isinstance(samples, list) or len(samples) > 100_000:
        raise ValueError('Invalid sample collection')
    groups = {}
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError('Invalid sample record')
        value = sample.get('milliseconds')
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid duration')
        key = (sample.get('scenario'), sample.get('metric'))
        if not all(isinstance(s, str) and s.isalnum() and len(s) <= 40 for s in key):
            raise ValueError('Invalid measurement label')
        groups.setdefault(key, []).append(value)
    rows = []
    for (scenario, metric), values in sorted(groups.items()):
        values.sort()
        rows.append(dict(scenario=scenario, metric=metric, count=len(values), medianMS=statistics.median(values),
                         p95MS=values[math.ceil(len(values)*.95)-1], maxMS=values[-1], totalMS=sum(values)))
    return rows


def compare(before, after):
    a, b = before['metadata'], after['metadata']
    if a.get('mode') != 'scripted' or b.get('mode') != 'scripted':
        raise ValueError('Only scripted matched workloads can be compared automatically')
    mismatches = [key for key in MATCH_FIELDS if key not in a or key not in b or a[key] != b[key]]
    if mismatches:
        raise ValueError('incompatible profiles: ' + ', '.join(mismatches))
    if before['capture'].get('droppedSamples') or after['capture'].get('droppedSamples'):
        raise ValueError('truncated capture cannot establish a comparison')
    left = {(r['scenario'],r['metric']):r for r in summarize(before['capture'])}
    right = {(r['scenario'],r['metric']):r for r in summarize(after['capture'])}
    if not left or left.keys() != right.keys():
        raise ValueError('measurement coverage differs; inspect reports before comparing')
    rows = []
    for key, base in left.items():
        current = right[key]
        delta = current['p95MS'] - base['p95MS']
        rows.append(dict(scenario=key[0], metric=key[1], baselineCount=base['count'], candidateCount=current['count'],
                         baselineP95MS=base['p95MS'], candidateP95MS=current['p95MS'], p95DeltaMS=delta,
                         p95DeltaPercent=delta/base['p95MS']*100 if base['p95MS'] else None,
                         baselineTotalMS=base['totalMS'], candidateTotalMS=current['totalMS']))
    return rows


def read_receipt(path):
    path = Path(path)
    if path.stat().st_size > 32_000_000:
        raise ValueError('Report exceeds 32 MB')
    return json.loads(path.read_text())


def finish(folder, info, raw):
    capture = read_receipt(raw)
    rows = summarize(capture)
    if not rows:
        raise ValueError('Empty profile; no success receipt')
    receipt = dict(metadata=info, capture=capture)
    (folder/'report.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    lines = ['# Interaction profile', '', 'Host-wall observations; not display-frame or pointer-to-paint latency.',
             'Nested durations overlap: do not add different metrics. Counts describe work as well as time.',
             f"Mode: {info['mode']}; build: {info['configuration']}; dropped samples: {capture['droppedSamples']}.",
             f"Source: `{info['source']['commit']}`; dirty: {info['source']['dirty']}; changes: `{info['source']['workingChangesSHA256']}`.", '',
             '| Scenario | Metric | Count | Median ms | p95 ms | Max ms | Total ms |', '|---|---|---:|---:|---:|---:|---:|']
    lines += [f"| {r['scenario']} | {r['metric']} | {r['count']} | {r['medianMS']:.3f} | {r['p95MS']:.3f} | {r['maxMS']:.3f} | {r['totalMS']:.3f} |" for r in rows]
    (folder/'summary.md').write_text('\n'.join(lines)+'\n')
    print(folder/'summary.md')


def run(args):
    if sys.platform != 'darwin':
        raise ValueError('Native canvas profiling requires a local Mac')
    folder = Path(args.output).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    info = metadata(args, 'scripted')
    raw = folder/'capture.json'
    env = build_environment()
    env.update(KWIPML_PROFILE_OUTPUT=str(raw), KWIPML_PROFILE_NODES=str(args.nodes), KWIPML_PROFILE_ITERATIONS=str(args.iterations))
    with (folder/'execution.log').open('w') as log:
        result = subprocess.run(['swift','test','--configuration',args.configuration,'--filter','InteractionProfileScenarioTests'], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise ValueError(f"Profile scenario failed; inspect {folder/'execution.log'}; no success receipt")
    if source_identity() != info['source']:
        raise ValueError('Source changed during capture; no success receipt')
    info['renderEnvironment'] = read_receipt(folder/'fixture.json')
    finish(folder, info, raw)


def capture(args):
    app = Path(args.app).resolve()
    with (app/'Contents/Info.plist').open('rb') as f:
        plist = plistlib.load(f)
    if plist.get('CFBundleIdentifier') != 'info.kwip.KwipML.POC' or plist.get('CFBundleExecutable') != 'KwipML':
        raise ValueError('Manual recording requires the development POC app identity')
    folder = Path(args.output).resolve(); folder.mkdir(parents=True, exist_ok=False)
    info = metadata(args, 'manual')
    # Bind the observed binary, rather than asserting it matches current source.
    executable = app/'Contents/MacOS/KwipML'
    info['binarySHA256'] = digest(executable.read_bytes())
    raw = folder/'capture.json'
    env = build_environment()
    env.update(KWIPML_INTERACTION_PROFILE=str(raw), KWIPML_INTERACTION_PROFILE_SECONDS=str(args.seconds))
    process = subprocess.Popen([str(executable)], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f'Recording development app PID {process.pid} for {args.seconds}s. Exercise the target interactions. The tool leaves the app open.', flush=True)
    deadline = time.monotonic()+args.seconds+30
    while not raw.exists() and time.monotonic() < deadline:
        if process.poll() is not None:
            break
        time.sleep(.2)
    if not raw.exists():
        raise ValueError('No capture produced; app may be uninstrumented or blocked. No receipt created.')
    finish(folder, info, raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('run', help='Repeatable temporary hosted canvas; local only')
    p.add_argument('--output', required=True); p.add_argument('--nodes', type=int, choices=[16,64,256], default=64)
    p.add_argument('--iterations', type=int, choices=range(3,31), default=8)
    p.add_argument('--configuration', choices=['debug','release'], default='release'); p.set_defaults(action=run)
    p = sub.add_parser('capture', help='Record manual interactions in an already-built development app')
    p.add_argument('--app', required=True); p.add_argument('--output', required=True)
    p.add_argument('--seconds', type=int, choices=range(1,301), default=60); p.set_defaults(action=capture)
    p = sub.add_parser('compare', help='Refuse mismatched/truncated/manual captures')
    p.add_argument('baseline'); p.add_argument('candidate')
    p.set_defaults(action=lambda args: print(json.dumps(compare(read_receipt(args.baseline),read_receipt(args.candidate)),indent=2,allow_nan=False)))
    args = parser.parse_args()
    try:
        args.action(args)
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
