#!/usr/bin/env python3
"""Separate MAS material preflight. Never uses the POC identity or ad-hoc signing.

The default command only emits a reviewable plan. Synthetic unit-test inputs are
not signing evidence; CLI identities come from security's valid private-key list.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import plistlib
import re
import shutil
import sys
import stat
import subprocess
import tempfile
import database_broker_package

ROOT = Path(__file__).resolve().parents[1]
MAX_PROFILE = 4 * 1024 * 1024
APP_ENTITLEMENTS = {'com.apple.security.app-sandbox': True,
                    'com.apple.security.files.user-selected.read-write': True,
                    'com.apple.security.files.bookmarks.app-scope': True}
WORKER_ENTITLEMENTS = {'com.apple.security.app-sandbox': True,
                       'com.apple.security.inherit': True}


def bounded_file(path, limit):
    path = Path(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError('Input must be a bounded regular file')
        value = stream.read(limit + 1)
    if len(value) > limit:
        raise ValueError('Input exceeds its size limit')
    return value


def parse_identities(text):
    if len(text) > 131072:
        raise ValueError('Signing identity inventory exceeds bounds')
    result = {}
    for fingerprint, label in re.findall(r'^\s*\d+\)\s+([A-Fa-f0-9]{40})\s+"([^"\n]+)"\s*$', text, re.M):
        fingerprint = fingerprint.upper()
        if fingerprint in result:
            raise ValueError('Duplicate signing identity fingerprint')
        result[fingerprint] = label
    return result


def decode_profile(path, *, run=subprocess.run):
    try:
        raw = bounded_file(path, MAX_PROFILE)
        # Snapshot first: never let a later path replacement change the decoded authority.
        with tempfile.TemporaryDirectory(prefix='kwipml-profile-') as directory:
            captured = Path(directory) / 'profile.provisionprofile'
            captured.write_bytes(raw)
            result = run(['/usr/bin/security', 'cms', '-D', '-i', str(captured)],
                         check=True, capture_output=True, timeout=20)
        if len(result.stdout) > MAX_PROFILE:
            raise ValueError('Decoded profile exceeds size bound')
        value = plistlib.loads(result.stdout)
        if not isinstance(value, dict):
            raise ValueError('Decoded profile is not a dictionary')
        return value, raw
    except (OSError, subprocess.SubprocessError, plistlib.InvalidFileException, ValueError) as error:
        raise ValueError('Cannot decode the supplied regular CMS provisioning profile') from error


def utc(value):
    if not isinstance(value, datetime):
        raise ValueError('Profile dates are required')
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def prepare_plan(config, profile, identities, app_entitlements, worker_entitlements, *,
                 now, confirmed_team, app_identity, installer_identity):
    expected = {'schemaVersion', 'bundleIdentifier', 'version', 'build', 'displayName',
                'executable', 'minimumSystemVersion', 'teamIdentifier', 'teamConfirmed'}
    if not isinstance(config, dict) or set(config) != expected or type(config['schemaVersion']) is not int or config['schemaVersion'] != 1:
        raise ValueError('Unsupported production configuration')
    exact = {'bundleIdentifier': 'info.kwip.KwipML', 'version': '1.0.0', 'build': '2',
             'displayName': 'Kwip ML', 'executable': 'KwipML', 'minimumSystemVersion': '14.0'}
    if any(config[key] != value for key, value in exact.items()) or type(config['teamConfirmed']) is not bool:
        raise ValueError('Production identity/version differs from approved V1 scope')
    if config['teamConfirmed'] is not True:
        raise ValueError('Production team has not been owner-confirmed in configuration')
    team = config['teamIdentifier']
    if not isinstance(team, str) or not re.fullmatch('[A-Z0-9]{10}', team) or confirmed_team != team:
        raise ValueError('Explicit owner-confirmed team is required')
    app_identity, installer_identity = app_identity.upper(), installer_identity.upper()
    if not re.fullmatch('[A-F0-9]{40}', app_identity) or not re.fullmatch('[A-F0-9]{40}', installer_identity):
        raise ValueError('Select exact distribution identity fingerprints')
    app_name, installer_name = identities.get(app_identity, ''), identities.get(installer_identity, '')
    if not (app_name.startswith(('Apple Distribution: ', '3rd Party Mac Developer Application: ')) and app_name.endswith(' (' + team + ')')):
        raise ValueError('Matching Mac App Store application distribution private key is unavailable')
    if not (installer_name.startswith('3rd Party Mac Developer Installer: ') and installer_name.endswith(' (' + team + ')')):
        raise ValueError('Matching Mac App Store installer distribution private key is unavailable')
    if app_entitlements != APP_ENTITLEMENTS or worker_entitlements != WORKER_ENTITLEMENTS:
        raise ValueError('Sandbox entitlements differ from reviewed application/child sets')
    if not isinstance(profile, dict) or profile.get('TeamIdentifier') != [team] or profile.get('Platform') != ['OSX']:
        raise ValueError('Profile must belong to the confirmed macOS team')
    if 'ProvisionedDevices' in profile or profile.get('ProvisionsAllDevices', False) is not False:
        raise ValueError('Development/ad-hoc/Developer ID profiles cannot authorize MAS packaging')
    if not utc(profile.get('CreationDate')) <= utc(now) < utc(profile.get('ExpirationDate')):
        raise ValueError('Provisioning profile is not currently valid')
    entitlement = profile.get('Entitlements', {})
    if not isinstance(entitlement, dict) or entitlement.get('com.apple.application-identifier') != team + '.' + config['bundleIdentifier'] or entitlement.get('com.apple.developer.team-identifier') != team:
        raise ValueError('Profile application identifier/team does not exactly match production')
    if entitlement.get('com.apple.security.get-task-allow', False) is not False or entitlement.get('get-task-allow', False) is not False:
        raise ValueError('Debuggable provisioning profile cannot be used for distribution')
    certificates = profile.get('DeveloperCertificates')
    if not isinstance(certificates, list) or not 1 <= len(certificates) <= 32 or any(not isinstance(c, bytes) or not 1 <= len(c) <= 65536 for c in certificates):
        raise ValueError('Profile certificate inventory is invalid')
    if app_identity not in {hashlib.sha1(c).hexdigest().upper() for c in certificates}:
        raise ValueError('Selected application private key is not authorized by the profile certificate')
    profile_id = profile.get('UUID')
    if not isinstance(profile_id, str) or not re.fullmatch('[A-Fa-f0-9-]{36}', profile_id):
        raise ValueError('Provisioning profile UUID is invalid')
    info = {'CFBundleIdentifier': config['bundleIdentifier'], 'CFBundleName': config['displayName'],
            'CFBundleDisplayName': config['displayName'], 'CFBundleExecutable': config['executable'],
            'CFBundlePackageType': 'APPL', 'CFBundleShortVersionString': config['version'],
            'CFBundleVersion': config['build'], 'CFBundleIconFile': 'KwipML.icns',
            'LSMinimumSystemVersion': config['minimumSystemVersion'], 'NSHighResolutionCapable': True,
            'CFBundleSupportedPlatforms': ['MacOSX']}
    app = dict(app_entitlements)
    app.update({'com.apple.application-identifier': team + '.' + config['bundleIdentifier'],
                'com.apple.developer.team-identifier': team})
    return {'schemaVersion': 1, 'status': 'materialPreflightPassedNotSigned', 'archiveReady': False,
            'teamIdentifier': team, 'profileUUID': profile_id, 'applicationIdentitySHA1': app_identity,
            'installerIdentitySHA1': installer_identity, 'infoPlist': info,
            'appEntitlements': app, 'workerEntitlements': dict(worker_entitlements),
            'executionBlockers': ['Provide a runtime produced with explicit MAS signing selection, including the sealed watchdog and reviewed interpreter entitlements before attestation sealing; default ad-hoc payloads are insufficient.',
                                  'Validate the final signed app and installer with Apple; preflight does not establish App Store acceptance.'],
            'commandsAfterBlockersResolved': [
                {'purpose': 'Verify staged worker bytes against reviewed sources', 'argv': ['python3', 'scripts/runtime_package.py', '--verify-production-worker', '<verified-staged-runtime>', '--sign-identity', app_identity, '--team-id', team]},
                {'purpose': 'Sign containing application after signing nested code', 'argv': ['/usr/bin/codesign', '--force', '--sign', app_identity, '--entitlements', '<staging>/App.entitlements', '<staged-app>']},
                {'purpose': 'Verify containing app and nested signatures', 'argv': ['/usr/bin/codesign', '--verify', '--deep', '--strict', '<staged-app>']},
                {'purpose': 'Create Mac App Store installer', 'argv': ['/usr/bin/productbuild', '--component', '<staged-app>', '/Applications', '--sign', installer_identity, '<new-output>.pkg']}]}



def normalize_package_access(root):
    """Make the staged public payload readable without changing sealed bytes."""
    root = Path(root)
    for path in [root, *root.rglob('*')]:
        if path.is_symlink():
            continue
        mode = path.stat().st_mode
        if stat.S_ISDIR(mode):
            path.chmod((mode & 0o777) | 0o555)
        elif stat.S_ISREG(mode):
            path.chmod((mode & 0o777) | 0o444 | (0o111 if mode & 0o111 else 0))


def execute_package(plan, app, runtime, profile_raw, output, *, run=subprocess.run):
    """Stage a separate app, sign its root and build a signed MAS installer.

    Every nested runtime Mach-O must already have the selected production
    signature. This function never re-signs sealed worker code or creates an
    xcarchive. The source app and runtime are never modified.
    """
    app, runtime, output = Path(app), Path(runtime), Path(output)
    if plan.get('status') != 'materialPreflightPassedNotSigned':
        raise ValueError('Successful signing-material preflight is required')
    if not isinstance(profile_raw, bytes) or not 1 <= len(profile_raw) <= MAX_PROFILE or plan.get('profileSHA256') != hashlib.sha256(profile_raw).hexdigest():
        raise ValueError('Captured profile differs from material preflight')
    for source in [app, runtime]:
        if source.is_symlink() or not source.is_dir():
            raise ValueError('Application/runtime input must be a real directory')
        for candidate in source.rglob('*'):
            if candidate.is_symlink() and (not candidate.resolve().is_relative_to(source.resolve()) or not candidate.exists()):
                raise ValueError('Input contains an escaping or broken symbolic link')
    if output.suffix != '.pkg' or output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise ValueError('Package destination must be a new .pkg in an existing directory')
    if any(output.resolve().is_relative_to(source.resolve()) for source in [app, runtime]):
        raise ValueError('Package destination cannot be inside an input')
    source_info = plistlib.loads(bounded_file(app / 'Contents/Info.plist', 65536))
    if not isinstance(source_info, dict) or not isinstance(source_info.get('DTSDKName'), str) or not re.fullmatch(r'macosx[0-9]+(?:\.[0-9]+){0,2}', source_info['DTSDKName']):
        raise ValueError('Source app must supply its actual compiler DTSDKName before production packaging')
    category = source_info.get('LSApplicationCategoryType')
    if not isinstance(category, str) or not re.fullmatch(r'public\.app-category\.[a-z-]+', category):
        raise ValueError('Source app must supply its reviewed LSApplicationCategoryType before production packaging')
    if source_info.get('NSHumanReadableCopyright') != 'Copyright © 2026 KWIP LLC':
        raise ValueError('Source app must supply the approved NSHumanReadableCopyright before production packaging')
    build_fields = {'NSHumanReadableCopyright', 'DTCompiler', 'DTPlatformBuild', 'DTPlatformName', 'DTPlatformVersion', 'DTSDKBuild', 'DTSDKName', 'DTXcode', 'DTXcodeBuild', 'BuildMachineOSBuild', 'LSApplicationCategoryType'}
    staged_info = {key: source_info[key] for key in build_fields if key in source_info}
    if any(not isinstance(value, str) or not 1 <= len(value) <= 256 for value in staged_info.values()):
        raise ValueError('Source compiler metadata has unsupported values')
    staged_info.update(plan['infoPlist'])
    executable = bounded_file(app / 'Contents/MacOS/KwipML', 256 * 1024 * 1024)
    if executable[:4] not in {b'\xcf\xfa\xed\xfe', b'\xce\xfa\xed\xfe', b'\xca\xfe\xba\xbe', b'\xca\xfe\xba\xbf'}:
        raise ValueError('Application executable is not Mach-O')
    if bounded_file(app / 'Contents/Resources/KwipML.icns', 16 * 1024 * 1024)[:4] != b'icns':
        raise ValueError('Production application icon is missing')
    app_sha, installer_sha, team = plan['applicationIdentitySHA1'], plan['installerIdentitySHA1'], plan['teamIdentifier']
    def command(argv, timeout=120):
        return run([str(value) for value in argv], check=True, capture_output=True, timeout=timeout)
    def verify_runtime(path, report):
        command([sys.executable, ROOT / 'scripts/app_store_api_audit.py', path], timeout=1800)
        command([sys.executable, ROOT / 'scripts/runtime_source_disclosure.py',
                 '--runtime-root', path, '--verify-existing'])
        command([sys.executable, ROOT / 'scripts/runtime_package.py', '--verify-production-worker', path,
                 '--sign-identity', app_sha, '--team-id', team])
        command([sys.executable, ROOT / 'scripts/runtime_audit.py', path, '--require-signatures',
                 '--require-identity', app_sha, '--require-team', team, '--report', report], timeout=1800)
        manifest = json.loads(bounded_file(Path(path) / 'runtime-manifest.json', 65536))
        relative = manifest.get('pythonExecutable')
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or '..' in Path(relative).parts or '\\' in relative:
            raise ValueError('Runtime interpreter path is not a contained relative file')
        interpreter = (Path(path) / relative).resolve(strict=True)
        if not interpreter.is_relative_to(Path(path).resolve()) or not interpreter.is_file():
            raise ValueError('Runtime interpreter escapes the verified payload')
        result = command(['/usr/bin/codesign', '--display', '--entitlements', ':-', interpreter])
        try:
            worker_entitlements = plistlib.loads(result.stdout)
        except (plistlib.InvalidFileException, ValueError) as error:
            raise ValueError('Runtime worker entitlements cannot be verified') from error
        if worker_entitlements != WORKER_ENTITLEMENTS:
            raise ValueError('Runtime interpreter must have exactly sandbox and inheritance entitlements')

    # Missing producer support or credentials refuses before any app/package staging.
    with tempfile.TemporaryDirectory(prefix='kwipml-production-preflight-') as checking:
        verify_runtime(runtime, Path(checking) / 'runtime-audit.json')
    with tempfile.TemporaryDirectory(prefix='.kwipml-production-', dir=output.parent) as staging:
        staging = Path(staging)
        staged_app = staging / 'Kwip ML.app'
        command(['/usr/bin/ditto', app, staged_app], timeout=600)
        database_helper=staged_app/'Contents/XPCServices/DatabaseSourceService.xpc'
        existing_runtime=database_broker_package.locate_runtime(staged_app)
        staged_runtime = database_broker_package.runtime_destination(staged_app) if database_helper.is_dir() else staged_app/'Contents/Resources/KwipMLRuntime'
        if existing_runtime != staged_runtime and existing_runtime.exists():shutil.rmtree(existing_runtime)
        staged_runtime.parent.mkdir(parents=True,exist_ok=True)
        if staged_runtime.is_symlink():
            raise ValueError('Existing nested runtime cannot be a symbolic link')
        if staged_runtime.exists(): shutil.rmtree(staged_runtime)
        command(['/usr/bin/ditto', runtime, staged_runtime], timeout=600)
        (staged_app / 'Contents/Info.plist').write_bytes(plistlib.dumps(staged_info))
        (staged_app / 'Contents/embedded.provisionprofile').write_bytes(profile_raw)
        breadcrumb = staged_app / 'Contents/Resources/ProjectRoot.txt'
        if breadcrumb.exists() or breadcrumb.is_symlink(): breadcrumb.unlink()
        entitlements = staging / 'App.entitlements'
        entitlements.write_bytes(plistlib.dumps(plan['appEntitlements']))
        verify_runtime(staged_runtime, staging / 'runtime-audit.json')
        privacy_destination = staged_app / 'Contents/Resources/VendorPrivacy'
        privacy_command = [sys.executable, ROOT / 'scripts/vendor_privacy.py', '--runtime-root', staged_runtime,
                           '--destination', privacy_destination]
        if privacy_destination.exists() or privacy_destination.is_symlink():
            privacy_command.append('--verify-existing')
        command(privacy_command)
        normalize_package_access(staged_app)
        command([sys.executable, ROOT / 'scripts/app_store_api_audit.py', staged_app], timeout=1800)
        database_helper=staged_app/'Contents/XPCServices/DatabaseSourceService.xpc'
        if database_helper.is_dir():
            command(['/usr/bin/codesign','--force','--sign',app_sha,'--entitlements',ROOT/'packaging/DatabaseSourceService.entitlements',database_helper])
        command(['/usr/bin/codesign', '--force', '--sign', app_sha, '--entitlements', entitlements, staged_app])
        command(['/usr/bin/codesign', '--verify', '--deep', '--strict', staged_app])
        verify_runtime(staged_runtime, staging / 'runtime-audit-after-signing.json')
        package = staging / 'KwipML.pkg'
        command(['/usr/bin/productbuild', '--component', staged_app, '/Applications', '--sign', installer_sha, package], timeout=1800)
        command(['/usr/sbin/pkgutil', '--check-signature', package])
        info = package.lstat()
        if not stat.S_ISREG(info.st_mode) or not 1 <= info.st_size <= 8 * 1024**3:
            raise ValueError('Generated package is missing or exceeds the release size bound')
        digest = hashlib.sha256()
        with package.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''): digest.update(chunk)
        # Same-volume link commits atomically and refuses an intervening destination.
        os.link(package, output, follow_symlinks=False)
        return {'schemaVersion': 1, 'status': 'packageCreatedNotUploaded', 'archiveCreated': False,
                'packageBytes': info.st_size, 'packageSHA256': digest.hexdigest(),
                'bundleIdentifier': plan['infoPlist']['CFBundleIdentifier'], 'teamIdentifier': team,
                'appleValidationComplete': False}


def atomic_json(path, value):
    path = Path(path)
    if not path.parent.is_dir() or path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError('Output must be a regular file in an existing directory')
    data = json.dumps(value, indent=2).encode() + b'\n'
    descriptor, temporary = tempfile.mkstemp(prefix='.kwipml-production-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(data)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'packaging/production-config.json'))
    parser.add_argument('--profile', required=True)
    parser.add_argument('--confirm-team', '--team', dest='confirm_team', required=True)
    parser.add_argument('--app-identity', required=True)
    parser.add_argument('--installer-identity', required=True)
    parser.add_argument('--output')
    parser.add_argument('--execute', action='store_true', help='Explicitly sign a separate staged app and build a new installer')
    parser.add_argument('--app')
    parser.add_argument('--runtime')
    args = parser.parse_args()
    try:
        config = json.loads(bounded_file(args.config, 16384))
        profile, profile_raw = decode_profile(args.profile)
        inventory = subprocess.run(['/usr/bin/security', 'find-identity', '-v'], check=True,
                                   capture_output=True, text=True, timeout=20)
        identities = parse_identities(inventory.stdout)
        plan = prepare_plan(config, profile, identities,
                            plistlib.loads(bounded_file(ROOT / 'packaging/App.entitlements', 16384)),
                            plistlib.loads(bounded_file(ROOT / 'packaging/Worker.entitlements', 16384)),
                            now=datetime.now(timezone.utc), confirmed_team=args.confirm_team,
                            app_identity=args.app_identity, installer_identity=args.installer_identity)
        plan['profileSHA256'] = hashlib.sha256(profile_raw).hexdigest()
        if args.execute:
            if not args.app or not args.runtime or not args.output:
                raise ValueError('Execution requires --app, --runtime and a new --output .pkg')
            result = execute_package(plan, args.app, args.runtime, profile_raw, args.output)
            print(json.dumps(result, indent=2))
        elif args.app or args.runtime:
            raise ValueError('--app/--runtime require explicit --execute; default mode only emits the plan')
        elif args.output: atomic_json(args.output, plan)
        else: print(json.dumps(plan, indent=2))
        return 0
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f'Production preflight refused: {error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
