#!/usr/bin/env python3
"""Prepare/resume a local direct candidate. Never publishes, promotes, or installs."""
import argparse
from contextlib import contextmanager
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASELINE_SCOPE = 'local Swift and pinned Python authoring/worker/runtime baseline; see output.log for skips'
FIELDS = {'schemaVersion', 'version', 'build', 'sourceCommit', 'runtimeManifestSHA256',
          'minimumSystemVersion', 'architecture', 'notesReviewed', 'changes', 'compatibility', 'knownLimits'}
COORDINATES = ('version', 'build', 'sourceCommit', 'runtimeManifestSHA256')
PHASES = {'assembled', 'appSubmitted', 'appAccepted', 'dmgBuilt', 'dmgSubmitted', 'verified'}


def read_json(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= 65536:
        raise ValueError('Expected a bounded regular JSON receipt/input')
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_manifest(value):
    if not isinstance(value, dict) or set(value) != FIELDS:
        raise ValueError('Candidate manifest has missing or unknown fields')
    if type(value['schemaVersion']) is not int or value['schemaVersion'] != 1:
        raise ValueError('Unknown candidate manifest schema')
    if not isinstance(value['version'], str) or not re.fullmatch(r'1\.(?:0|[1-9][0-9]{0,3})\.(?:0|[1-9][0-9]{0,3})', value['version']):
        raise ValueError('Only perpetual v1 versions are supported')
    if type(value['build']) is not int or not 1 <= value['build'] <= 999999999:
        raise ValueError('Canonical positive build required')
    for key, length in [('sourceCommit', 40), ('runtimeManifestSHA256', 64)]:
        if not isinstance(value[key], str) or not re.fullmatch('[0-9a-f]{' + str(length) + '}', value[key]):
            raise ValueError('Exact source/runtime pins required')
    if value['architecture'] != 'arm64' or value['minimumSystemVersion'] != '14.0':
        raise ValueError('Manifest cannot expand reviewed platform scope')
    if value['notesReviewed'] is not True:
        raise ValueError('Review actual shipped release notes before candidate preparation')
    for key in ['changes', 'compatibility', 'knownLimits']:
        notes = value[key]
        if not isinstance(notes, list) or len(notes) > 30 or any(
                not isinstance(note, str) or not note.strip() or len(note) > 2000 or
                any(ord(c) < 32 and c not in '\n\t' for c in note) for note in notes):
            raise ValueError('Release notes must be bounded plain-text lists')
        if key != 'knownLimits' and not notes:
            raise ValueError('Changes and compatibility notes are required')
    return value


def validate_baseline(path, commit):
    value = read_json(path)
    if not isinstance(value, dict) or value.get('commit') != commit or value.get('scope') != BASELINE_SCOPE:
        raise ValueError('Full local baseline must match the exact candidate commit')
    if any(value.get(key) is not False for key in ['hostedResult', 'nativeUAT', 'releaseAcceptance']):
        raise ValueError('Baseline cannot claim hosted/native UAT/release acceptance')
    try:
        timestamp = datetime.fromisoformat(value['completedAt'])
        if timestamp.tzinfo is None:
            raise ValueError('Missing baseline timezone')
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('Invalid baseline completion timestamp') from error
    log = Path(path).parent / 'output.log'
    if log.is_symlink() or not log.is_file() or log.stat().st_size == 0:
        raise ValueError('Retain the actual local verification log alongside PASSED.json')
    return value


def validate_previous(value, previous):
    try:
        old = previous['coordinates']
        old_version = tuple(map(int, old['version'].split('.')))
        version = tuple(map(int, value['version'].split('.')))
        valid = (previous['phase'] == 'verified' and len(old_version) == 3 and old_version[0] == 1
                 and type(old['build']) is int and value['build'] > old['build'] and version >= old_version)
    except (KeyError, TypeError, ValueError, AttributeError):
        valid = False
    if not valid:
        raise ValueError('Previous release must be verified; build must increase and version must not regress')


def binding(value, native, baseline_hash):
    if native.get('diagnosticOnly') is True:
        raise ValueError('Diagnostic-only packaged smoke cannot become a release candidate')
    coordinates = {key: value[key] for key in COORDINATES}
    if native.get('coordinates') != coordinates or native.get('phase') not in PHASES or native.get('publicLaunch') is not False:
        raise ValueError('Native release differs from reviewed candidate coordinates')
    if native['phase'] == 'verified' and (not isinstance(native.get('dmgSHA256'), str)
            or not re.fullmatch('[0-9a-f]{64}', native['dmgSHA256'])
            or type(native.get('dmgBytes')) is not int or native['dmgBytes'] <= 0):
        raise ValueError('Verified candidate requires exact final artifact hash and size')
    return {'schemaVersion': 1, 'coordinates': coordinates,
            'manifestSHA256': hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest(),
            'baselineReceiptSHA256': baseline_hash,
            'notes': {key: value[key] for key in ['changes', 'compatibility', 'knownLimits']},
            'minimumSystemVersion': value['minimumSystemVersion'], 'architecture': value['architecture'],
            'phase': native['phase'], 'artifactSHA256': native.get('dmgSHA256'),
            'artifactBytes': native.get('dmgBytes'), 'published': False, 'promoted': False,
            'nativeUAT': False, 'publicLaunch': False}


def render_notes(value):
    sections = [f"Kwip ML {value['version']} (build {value['build']})"]
    for key, heading in [('changes', 'Changes'), ('compatibility', 'Compatibility'), ('knownLimits', 'Known limits')]:
        sections.append(heading + '\n' + ('\n'.join('- ' + note for note in value[key]) or 'None recorded.'))
    return '\n\n'.join(sections) + '\n'


def validate_authority(native, identity, profile):
    profile = Path(profile)
    if (not re.fullmatch('[A-F0-9]{40}', identity) or native.get('identity') != identity
            or profile.is_symlink() or not profile.is_file() or not 0 < profile.stat().st_size <= 4 * 1024 * 1024
            or sha256(profile) != native.get('profileSHA256')):
        raise ValueError('Signing identity/profile differs from the existing candidate; preserve this output')


def save_candidate(path, snapshot):
    # Metadata is reproducible, unlike an uncertain provider upload. A unique
    # scratch file prevents an interrupted local write from poisoning resume.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', prefix='.candidate-', suffix='.tmp',
                                         dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(snapshot, indent=2, sort_keys=True) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def validate_binding(saved, snapshot):
    if any(saved.get(key) != snapshot[key] for key in ['coordinates', 'manifestSHA256', 'baselineReceiptSHA256']):
        raise ValueError('Candidate notes, baseline, or source changed; preserve this output')
    if saved.get('artifactSHA256') is not None and any(saved.get(key) != snapshot[key]
            for key in ['artifactSHA256', 'artifactBytes', 'phase']):
        raise ValueError('Previously verified candidate artifact changed; preserve this output')


@contextmanager
def release_lock(path):
    # flock is released by the OS on exit/crash; a leftover file is not a live lock.
    flags = os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW
    with os.fdopen(os.open(path, flags, 0o600), 'a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Another candidate operator holds this output lock') from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def drive(read, step, *, wait_seconds=0, poll_seconds=30, now=time.monotonic, sleep=time.sleep):
    if not 0 <= wait_seconds <= 600 or not 5 <= poll_seconds <= 60:
        raise ValueError('Wait must be 0..600 seconds; poll interval 5..60 seconds')
    deadline = now() + wait_seconds
    for _ in range(130):
        before = read()
        if before.get('pendingUpload'):
            raise ValueError('Uncertain prior upload: reconcile the existing Apple submission; never resubmit blindly')
        phase = before.get('phase')
        if phase == 'verified':
            return before
        if phase not in PHASES:
            raise ValueError('Unknown native candidate phase')
        step('notary')
        after = read()
        if after.get('phase') == phase:
            if phase not in {'appSubmitted', 'dmgSubmitted'}:
                raise ValueError('Native phase did not advance; inspect retained evidence')
            remaining = deadline - now()
            if remaining <= 0:
                return after
            sleep(min(poll_seconds, remaining))
    raise ValueError('Bounded candidate advancement limit exceeded')


def require_source(commit):
    def git(*args):
        return subprocess.run(['git', *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    if git('rev-parse', 'HEAD') != commit or git('status', '--porcelain'):
        raise ValueError('Candidate preparation requires the clean exact source commit, including orchestration code')
    if os.environ.get('KWIPML_COMMERCE_STAGING') == '1':
        raise ValueError('Production candidate preparation refuses staging configuration')


def output_path(path):
    if '..' in Path(path).parts:
        raise ValueError('Output cannot traverse parent directories')
    path = Path(os.path.abspath(path))
    allowed = ROOT / '.local/direct-releases'
    if '..' in Path(path).parts or not path.is_relative_to(allowed) or path == allowed:
        raise ValueError('Use a named .local/direct-releases output in this source checkout')
    if any(parent.is_symlink() for parent in [path, *path.parents]):
        raise ValueError('Output cannot traverse symlinks')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', choices=['plan', 'assemble', 'advance'], default='plan')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--previous-release', type=Path)
    parser.add_argument('--initial-release', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--identity', required=True)
    parser.add_argument('--developer-dir', default='/Applications/Xcode.app/Contents/Developer')
    parser.add_argument('--keychain-profile', default='kwipml-direct')
    parser.add_argument('--runtime-developer-dir')
    parser.add_argument('--notary-developer-dir', default='/Applications/Xcode.app/Contents/Developer')
    parser.add_argument('--wait-seconds', type=int, default=0)
    args = parser.parse_args()
    try:
        value = validate_manifest(read_json(args.manifest))
        validate_baseline(args.baseline, value['sourceCommit'])
        require_source(value['sourceCommit'])
        if bool(args.previous_release) == args.initial_release:
            raise ValueError('Choose previous verified release for an update, or explicit initial release')
        if args.previous_release:
            validate_previous(value, read_json(args.previous_release))
        if not 0 <= args.wait_seconds <= 600:
            raise ValueError('Wait must be 0..600 seconds')
        out = output_path(args.output)
        baseline_hash = sha256(args.baseline)
        command = [sys.executable, str(ROOT / 'scripts/direct_release.py'), '--output', str(out),
                   '--runtime', str(args.runtime), '--runtime-manifest-sha256', value['runtimeManifestSHA256'],
                   '--version', value['version'], '--build', str(value['build']), '--commit', value['sourceCommit'],
                   '--profile', str(args.profile), '--identity', args.identity,
                   '--developer-dir', args.developer_dir, '--keychain-profile', args.keychain_profile]
        command += ['--notary-developer-dir', args.notary_developer_dir]
        if args.runtime_developer_dir:
            command += ['--runtime-developer-dir', args.runtime_developer_dir]
        def invoke(step):
            subprocess.run([*command, '--step', step], cwd=ROOT, check=True)
        def read():
            native = read_json(out / 'release.json')
            validate_authority(native, args.identity, args.profile)
            snapshot = binding(value, native, baseline_hash)
            saved_path = out / 'candidate.json'
            if saved_path.exists():
                saved = read_json(saved_path)
                validate_binding(saved, snapshot)
            save_candidate(saved_path, snapshot)
            notes = render_notes(value)
            notes_path = out / 'release-notes.txt'
            if notes_path.exists():
                if notes_path.is_symlink() or notes_path.read_text() != notes:
                    raise ValueError('Candidate release notes changed; preserve this output')
            else:
                with notes_path.open('x') as stream:
                    stream.write(notes)
            return native
        if args.step == 'plan':
            invoke('plan')
            print(json.dumps({'step': 'plan', 'notesReviewed': True, 'baselineCommit': value['sourceCommit'],
                              'publishes': False, 'promotes': False, 'signs': False}))
            return 0
        out.parent.mkdir(parents=True, exist_ok=True)
        with release_lock(out.with_name(out.name + '.candidate.lock')):
            if not out.exists():
                invoke('assemble')
            elif not (out / 'release.json').is_file():
                raise ValueError('Incomplete assembly output: inspect it; do not overwrite/restart blindly')
            native = read()
            if args.step == 'assemble':
                print(json.dumps({'phase': native['phase'], 'stoppedBeforeNotary': native['phase'] == 'assembled', 'noSubmissionPerformed': True, 'publishes': False}))
                return 0
            if native['phase'] == 'verified':
                invoke('verify')
            else:
                native = drive(read, invoke, wait_seconds=args.wait_seconds)
            read()
            print(json.dumps({'phase': native['phase'], 'coordinates': native['coordinates'],
                              'published': False, 'promoted': False, 'nativeUAT': False, 'publicLaunch': False}, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print('Candidate stopped: ' + str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
