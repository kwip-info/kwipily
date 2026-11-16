"""Synthetic signing-material controls; these never create signatures or Apple records."""
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import plistlib
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 10, 9, tzinfo=timezone.utc)
TEAM = 'MT68X9Z6NQ'
CERT = b'synthetic-profile-certificate-not-an-actual-identity'
APP_SHA = hashlib.sha1(CERT).hexdigest().upper()
INSTALLER_SHA = 'B' * 40


def module():
    spec = importlib.util.spec_from_file_location('production_package', ROOT / 'scripts/production_package.py')
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def inputs():
    config = {'schemaVersion': 1, 'bundleIdentifier': 'info.kwip.KwipML',
              'version': '1.0.0', 'build': '2', 'displayName': 'Kwip ML',
              'executable': 'KwipML', 'minimumSystemVersion': '14.0',
              'teamIdentifier': TEAM, 'teamConfirmed': True}
    profile = {'UUID': '11111111-2222-3333-4444-555555555555', 'Name': 'Synthetic MAS profile',
               'TeamIdentifier': [TEAM], 'Platform': ['OSX'],
               'CreationDate': NOW - timedelta(days=1), 'ExpirationDate': NOW + timedelta(days=30),
               'DeveloperCertificates': [CERT],
               'Entitlements': {'com.apple.application-identifier': TEAM + '.info.kwip.KwipML',
                                'com.apple.developer.team-identifier': TEAM}}
    identities = {APP_SHA: 'Apple Distribution: Synthetic Owner (' + TEAM + ')',
                  INSTALLER_SHA: '3rd Party Mac Developer Installer: Synthetic Owner (' + TEAM + ')'}
    app = plistlib.loads((ROOT / 'packaging/App.entitlements').read_bytes())
    worker = plistlib.loads((ROOT / 'packaging/Worker.entitlements').read_bytes())
    return config, profile, identities, app, worker


def plan(mod, values=None, **kwargs):
    config, profile, identities, app, worker = values or inputs()
    result = mod.prepare_plan(config, profile, identities, app, worker, now=NOW,
                            confirmed_team=kwargs.get('confirmed_team', TEAM),
                            app_identity=APP_SHA, installer_identity=INSTALLER_SHA)
    result['profileSHA256'] = hashlib.sha256(b'synthetic-profile').hexdigest()
    return result


def test_manifest_plist_and_plan_are_explicitly_unsigned_without_archive_claim():
    result = plan(module())
    info = result['infoPlist']
    assert (info['CFBundleIdentifier'], info['CFBundleShortVersionString'], info['CFBundleVersion']) == ('info.kwip.KwipML', '1.0.0', '2')
    assert info['CFBundleDisplayName'] == 'Kwip ML'
    assert info['CFBundleIconFile'] == 'KwipML.icns'
    assert info['CFBundleSupportedPlatforms'] == ['MacOSX']
    assert result['status'] == 'materialPreflightPassedNotSigned'
    assert result['archiveReady'] is False
    assert result['executionBlockers']
    assert result['appEntitlements']['com.apple.application-identifier'] == TEAM + '.info.kwip.KwipML'
    assert result['workerEntitlements'] == inputs()[4]
    commands = result['commandsAfterBlockersResolved']
    assert '--verify-production-worker' in commands[0]['argv']
    assert APP_SHA in commands[0]['argv'] and TEAM in commands[0]['argv']
    assert commands[-1]['argv'][0] == '/usr/bin/productbuild'
    assert INSTALLER_SHA in commands[-1]['argv']
    assert not any('xcarchive' in str(command) for command in commands)
    assert not any('Developer ID' in str(command) or ('--deep' in command['argv'] and '--sign' in command['argv']) for command in commands)


@pytest.mark.parametrize('mutation', ['missingAppKey', 'missingInstallerKey', 'development', 'developerID', 'wrongTeam', 'profileCert'])
def test_private_key_identity_and_certificate_family_must_match(mutation):
    values = list(inputs()); identities = values[2]
    if mutation == 'missingAppKey': identities.pop(APP_SHA)
    elif mutation == 'missingInstallerKey': identities.pop(INSTALLER_SHA)
    elif mutation == 'development': identities[APP_SHA] = f'Apple Development: Synthetic ({TEAM})'
    elif mutation == 'developerID': identities[APP_SHA] = f'Developer ID Application: Synthetic ({TEAM})'
    elif mutation == 'wrongTeam': identities[APP_SHA] = 'Apple Distribution: Synthetic (OTHER12345)'
    else: values[1]['DeveloperCertificates'] = [b'another-certificate']
    with pytest.raises(ValueError): plan(module(), values)


@pytest.mark.parametrize('mutation', ['expired', 'future', 'team', 'bundle', 'developmentDevices', 'allDevices', 'debug', 'platform'])
def test_profile_is_exact_current_mac_app_store_authority(mutation):
    values = list(inputs()); profile = values[1]
    if mutation == 'expired': profile['ExpirationDate'] = NOW
    elif mutation == 'future': profile['CreationDate'] = NOW + timedelta(days=1)
    elif mutation == 'team': profile['TeamIdentifier'] = ['OTHER12345']
    elif mutation == 'bundle': profile['Entitlements']['com.apple.application-identifier'] = TEAM + '.*'
    elif mutation == 'developmentDevices': profile['ProvisionedDevices'] = []
    elif mutation == 'allDevices': profile['ProvisionsAllDevices'] = True
    elif mutation == 'debug': profile['Entitlements']['com.apple.security.get-task-allow'] = True
    else: profile['Platform'] = ['iOS']
    with pytest.raises(ValueError): plan(module(), values)


def test_owner_team_confirmation_required_and_config_is_not_mutated():
    values = inputs(); before = copy.deepcopy(values)
    with pytest.raises(ValueError): plan(module(), values, confirmed_team=None)
    plan(module(), values)
    assert values == before
    assert values[0]['teamConfirmed'] is True


@pytest.mark.parametrize('mutation', ['sandbox', 'network', 'workerInherit', 'workerFileAccess'])
def test_entitlements_are_reviewed_exact_sets(mutation):
    values = list(inputs())
    if mutation == 'sandbox': values[3]['com.apple.security.app-sandbox'] = False
    elif mutation == 'network': values[3]['com.apple.security.network.client'] = True
    elif mutation == 'workerInherit': values[4]['com.apple.security.inherit'] = False
    else: values[4]['com.apple.security.files.user-selected.read-write'] = True
    with pytest.raises(ValueError): plan(module(), values)


def test_identity_inventory_parser_excludes_unavailable_private_keys():
    mod = module()
    text = f'  1) {APP_SHA} "Apple Distribution: Synthetic ({TEAM})"\n  2) {INSTALLER_SHA} "3rd Party Mac Developer Installer: Synthetic ({TEAM})"\n     2 valid identities found\n'
    assert mod.parse_identities(text) == inputs()[2] | {APP_SHA: f'Apple Distribution: Synthetic ({TEAM})', INSTALLER_SHA: f'3rd Party Mac Developer Installer: Synthetic ({TEAM})'}
    assert mod.parse_identities('0 valid identities found') == {}
    with pytest.raises(ValueError): mod.parse_identities(text + f'3) {APP_SHA} "Different"\n')


def test_cli_invalid_inputs_preserve_existing_output_and_poc_identity(tmp_path):
    output = tmp_path / 'plan.json'; output.write_text('previous')
    proc = subprocess.run([sys.executable, str(ROOT / 'scripts/production_package.py'), '--profile', str(tmp_path / 'missing'),
                           '--app-identity', APP_SHA, '--installer-identity', INSTALLER_SHA,
                           '--output', str(output)], capture_output=True, text=True)
    assert proc.returncode != 0
    assert output.read_text() == 'previous'
    assert 'info.kwip.KwipML.POC' in (ROOT / 'scripts/build-app.sh').read_text()


def test_profile_decoder_failure_never_yields_a_plan(tmp_path):
    mod = module(); profile = tmp_path / 'bad.provisionprofile'; profile.write_bytes(b'not-CMS')
    def refuse(argv, **kwargs):
        assert argv[:3] == ['/usr/bin/security', 'cms', '-D']
        raise subprocess.CalledProcessError(1, argv)
    with pytest.raises(ValueError): mod.decode_profile(profile, run=refuse)
    linked = tmp_path / 'alias'; linked.symlink_to(profile)
    with pytest.raises(ValueError): mod.decode_profile(linked, run=refuse)


def execution_fixture(tmp_path):
    import shutil
    app = tmp_path / 'Input.app'; (app / 'Contents/MacOS').mkdir(parents=True)
    (app / 'Contents/Resources').mkdir()
    (app / 'Contents/MacOS/KwipML').write_bytes(b'\xcf\xfa\xed\xfeexecutable')
    (app / 'Contents/Resources/KwipML.icns').write_bytes(b'icnsicon')
    (app / 'Contents/Resources/ProjectRoot.txt').write_text('/private/build/machine')
    (app / 'Contents/Info.plist').write_bytes(plistlib.dumps({'CFBundleIdentifier': 'info.kwip.KwipML.POC', 'DTSDKName': 'macosx26.0', 'DTXcodeBuild': 'synthetic-build', 'LSApplicationCategoryType': 'public.app-category.developer-tools', 'NSHumanReadableCopyright': 'Copyright © 2026 KWIP LLC'}))
    runtime = tmp_path / 'Runtime'; runtime.mkdir()
    (runtime / 'runtime-manifest.json').write_text(json.dumps({'pythonExecutable': 'python/bin/python3.13'}))
    interpreter = runtime / 'python/bin/python3.13'; interpreter.parent.mkdir(parents=True); interpreter.write_bytes(b'python')
    site = runtime / 'python/lib/python3.13/site-packages'
    for name, version in [('protobuf', '7.36.2'), ('grpcio', '1.84.0'), ('tensorflow', '2.21.0')]:
        folder = site / f'{name}-{version}.dist-info'; folder.mkdir(parents=True)
        (folder / 'METADATA').write_text(f'Name: {name}\nVersion: {version}\n')
    for relative in ['python/lib/libpython3.13.dylib', 'python/lib/python3.13/site-packages/google/_upb/_message.abi3.so', 'python/lib/python3.13/site-packages/grpc/_cython/cygrpc.cpython-313-darwin.so', 'python/lib/python3.13/site-packages/tensorflow/libtensorflow_framework.2.dylib']:
        native = runtime / relative; native.parent.mkdir(parents=True, exist_ok=True); native.write_bytes(b'synthetic-native-owner')
    headers = site / 'tensorflow/include'
    (headers / 'absl/base').mkdir(parents=True); (headers / 'absl/base/options.h').write_text('lts_20250814')
    (headers / 'google/protobuf').mkdir(parents=True); (headers / 'google/protobuf/empty.pb.h').write_text('PROTOBUF_VERSION != 6031001')

    output = tmp_path / 'Release.pkg'; calls = []
    def tool(argv, **kwargs):
        calls.append(argv)
        if len(argv) > 1 and Path(argv[1]).name == 'vendor_privacy.py':
            return subprocess.run(argv, **kwargs)
        if argv[0] == '/usr/bin/ditto': shutil.copytree(argv[1], argv[2], symlinks=True)
        elif argv[0] == '/usr/bin/productbuild': Path(argv[-1]).write_bytes(b'new-synthetic-package')
        elif argv[0] == '/usr/bin/codesign' and '--sign' in argv:
            staged = Path(argv[-1])
            metadata = plistlib.loads((staged / 'Contents/Info.plist').read_bytes())
            assert metadata['CFBundleIdentifier'] == 'info.kwip.KwipML'
            assert metadata['DTSDKName'] == 'macosx26.0'
            assert metadata['DTXcodeBuild'] == 'synthetic-build'
            assert metadata['LSApplicationCategoryType'] == 'public.app-category.developer-tools'
            assert metadata['NSHumanReadableCopyright'] == 'Copyright © 2026 KWIP LLC'
            assert (staged / 'Contents/embedded.provisionprofile').read_bytes() == b'synthetic-profile'
            assert not (staged / 'Contents/Resources/ProjectRoot.txt').exists()
            privacy = staged / 'Contents/Resources/VendorPrivacy'
            assert len(list(privacy.glob('*.bundle'))) == 6
            assert (privacy / 'MAPPING.json').is_file()
        elif argv[0] == '/usr/bin/codesign' and '--entitlements' in argv:
            return subprocess.CompletedProcess(argv, 0, stdout=plistlib.dumps(inputs()[4]), stderr=b'')
        return subprocess.CompletedProcess(argv, 0, stdout=b'{}', stderr=b'')
    return app, runtime, output, calls, tool


def test_executor_preflights_before_staging_and_never_overwrites_or_edits_input(tmp_path):
    mod = module(); app, runtime, output, calls, tool = execution_fixture(tmp_path)
    original = (app / 'Contents/Info.plist').read_bytes()
    result = mod.execute_package(plan(mod), app, runtime, b'synthetic-profile', output, run=tool)
    assert output.read_bytes() == b'new-synthetic-package'
    assert result['status'] == 'packageCreatedNotUploaded'
    assert result['archiveCreated'] is False
    assert (app / 'Contents/Info.plist').read_bytes() == original
    assert (app / 'Contents/Resources/ProjectRoot.txt').exists()
    first_copy = next(i for i, command in enumerate(calls) if command[0] == '/usr/bin/ditto')
    assert any('--verify-production-worker' in command for command in calls[:first_copy])
    assert any('--require-identity' in command and APP_SHA in command for command in calls[:first_copy])
    assert any('--require-team' in command and TEAM in command for command in calls[:first_copy])
    sign = next(command for command in calls if command[0] == '/usr/bin/codesign' and '--sign' in command)
    assert '--deep' not in sign and APP_SHA in sign
    assert not list(tmp_path.glob('.kwipml-production-*'))
    with pytest.raises(ValueError): mod.execute_package(plan(mod), app, runtime, b'synthetic-profile', output, run=tool)
    assert output.read_bytes() == b'new-synthetic-package'


def test_production_source_disclosure_refusal_prevents_staging_or_signing(tmp_path):
    mod = module(); app, runtime, output, calls, good = execution_fixture(tmp_path)
    def refuse_disclosure(argv, **kwargs):
        if len(argv) > 1 and Path(argv[1]).name == 'runtime_source_disclosure.py':
            raise subprocess.CalledProcessError(1, argv)
        return good(argv, **kwargs)
    with pytest.raises(subprocess.CalledProcessError):
        mod.execute_package(plan(mod), app, runtime, b'synthetic-profile', output, run=refuse_disclosure)
    assert not output.exists()
    assert not any(command[0] == '/usr/bin/ditto' for command in calls)


def test_production_verifies_source_disclosure_before_copy_and_after_signing(tmp_path):
    mod = module(); app, runtime, output, calls, good = execution_fixture(tmp_path)
    mod.execute_package(plan(mod), app, runtime, b'synthetic-profile', output, run=good)
    checks = [i for i, command in enumerate(calls)
              if len(command) > 1 and Path(command[1]).name == 'runtime_source_disclosure.py']
    assert len(checks) == 3
    assert all('--verify-existing' in calls[i] for i in checks)
    first_copy = next(i for i, command in enumerate(calls) if command[0] == '/usr/bin/ditto')
    sign = next(i for i, command in enumerate(calls) if command[0] == '/usr/bin/codesign' and '--sign' in command)
    assert checks[0] < first_copy < checks[1] < sign < checks[2]


@pytest.mark.parametrize('phase', ['preflight', 'copy', 'sign', 'productbuild', 'verifyPackage'])
def test_executor_failure_preserves_existing_files_and_cleans_staging(tmp_path, phase):
    mod = module(); app, runtime, output, calls, good = execution_fixture(tmp_path)
    def fail(argv, **kwargs):
        bad = ((phase == 'preflight' and '--verify-production-worker' in argv)
               or (phase == 'copy' and argv[0] == '/usr/bin/ditto')
               or (phase == 'sign' and argv[0] == '/usr/bin/codesign' and '--sign' in argv)
               or (phase == 'productbuild' and argv[0] == '/usr/bin/productbuild')
               or (phase == 'verifyPackage' and argv[0] == '/usr/sbin/pkgutil'))
        if bad: raise subprocess.CalledProcessError(1, argv)
        return good(argv, **kwargs)
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        mod.execute_package(plan(mod), app, runtime, b'synthetic-profile', output, run=fail)
    assert not output.exists()
    assert not list(tmp_path.glob('.kwipml-production-*'))
    assert (app / 'Contents/Resources/ProjectRoot.txt').read_text() == '/private/build/machine'
    if phase == 'preflight': assert not any(command[0] == '/usr/bin/ditto' for command in calls)


@pytest.mark.parametrize('entitlements', [{}, {'com.apple.security.app-sandbox': True},
    {'com.apple.security.app-sandbox': True, 'com.apple.security.inherit': True, 'com.apple.security.network.client': True}])
def test_executor_refuses_worker_entitlement_mismatch_before_staging(tmp_path, entitlements):
    mod = module(); app, runtime, output, calls, good = execution_fixture(tmp_path)
    def mismatched(argv, **kwargs):
        if argv[0] == '/usr/bin/codesign' and '--entitlements' in argv:
            return subprocess.CompletedProcess(argv, 0, stdout=plistlib.dumps(entitlements), stderr=b'')
        return good(argv, **kwargs)
    with pytest.raises(ValueError): mod.execute_package(plan(mod), app, runtime, b'synthetic-profile', output, run=mismatched)
    assert not output.exists()
    assert not any(command[0] == '/usr/bin/ditto' for command in calls)


def test_executor_rejects_changed_profile_and_escaping_input_links(tmp_path):
    mod = module(); app, runtime, output, calls, good = execution_fixture(tmp_path)
    with pytest.raises(ValueError): mod.execute_package(plan(mod), app, runtime, b'different-profile', output, run=good)
    external = tmp_path / 'outside'; external.write_text('private')
    (app / 'Contents/Resources/escape').symlink_to(external)
    with pytest.raises(ValueError): mod.execute_package(plan(mod), app, runtime, b'synthetic-profile', output, run=good)
    assert not output.exists() and calls == []


def test_unconfirmed_config_refuses_even_with_matching_command_line_team():
    values = list(inputs()); values[0]['teamConfirmed'] = False
    with pytest.raises(ValueError, match='confirmed'):
        plan(module(), values, confirmed_team=TEAM)


@pytest.mark.parametrize('field', ['DTSDKName', 'LSApplicationCategoryType', 'NSHumanReadableCopyright'])
def test_executor_requires_actual_build_sdk_and_reviewed_category_before_staging(tmp_path, field):
    mod = module(); app, runtime, output, calls, good = execution_fixture(tmp_path)
    metadata = plistlib.loads((app / 'Contents/Info.plist').read_bytes()); metadata.pop(field)
    (app / 'Contents/Info.plist').write_bytes(plistlib.dumps(metadata))
    with pytest.raises(ValueError): mod.execute_package(plan(mod), app, runtime, b'synthetic-profile', output, run=good)
    assert not output.exists() and calls == []


@pytest.mark.parametrize('configuration', [None, 'debug', 'release'])
def test_build_script_records_actual_sdk_and_uses_selected_configuration_with_mock_tools(tmp_path, configuration):
    import os
    import shutil
    project = tmp_path / 'Project'; (project / 'scripts').mkdir(parents=True)
    shutil.copy2(ROOT / 'scripts/build-app.sh', project / 'scripts/build-app.sh')
    binary = project / 'mock-bin'; binary.mkdir()
    (binary / 'KwipML').write_bytes(b'synthetic-executable')
    (binary / 'KwipML_KwipMLCore.bundle').mkdir()
    (binary / 'KwipML_KwipMLApp.bundle').mkdir()
    tools = tmp_path / 'tools'; tools.mkdir(); log = tmp_path / 'calls.jsonl'
    source = f'''#!{sys.executable}
import json,os,sys
from pathlib import Path
name=Path(sys.argv[0]).name
with open(os.environ['MOCK_BUILD_LOG'],'a') as stream: stream.write(json.dumps([name]+sys.argv[1:])+'\\n')
if name=='swift' and '--show-bin-path' in sys.argv: print(os.environ['MOCK_BUILD_BIN'])
if name=='xcrun': print('26.1' if '--show-sdk-version' in sys.argv else '25B999')
'''
    for name in ['swift', 'xcrun', 'codesign', 'build-python']:
        path = tools / name; path.write_text(source); path.chmod(0o755)
    environment = os.environ | {'PATH': str(tools) + ':' + os.environ['PATH'],
        'KWIPML_BUILD_PYTHON': str(tools / 'build-python'),
        'MOCK_BUILD_LOG': str(log), 'MOCK_BUILD_BIN': str(binary)}
    environment.pop('KWIPML_BUILD_CONFIGURATION', None)
    if configuration is not None: environment['KWIPML_BUILD_CONFIGURATION'] = configuration
    result = subprocess.run(['/bin/zsh', str(project / 'scripts/build-app.sh')], env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    info = plistlib.loads((project / '.local/KwipML POC.app/Contents/Info.plist').read_bytes())
    assert info['CFBundleIdentifier'] == 'info.kwip.KwipML.POC'
    assert info['CFBundleIconFile'] == 'KwipML.icns'
    assert info['DTSDKName'] == 'macosx26.1' and info['DTSDKBuild'] == '25B999'
    assert info['CFBundleSupportedPlatforms'] == ['MacOSX']
    assert info['LSApplicationCategoryType'] == 'public.app-category.developer-tools'
    assert info['NSHumanReadableCopyright'] == 'Copyright © 2026 KWIP LLC'
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    swift = [call for call in calls if call[0] == 'swift']
    assert len(swift) == 2 and all(call[call.index('--configuration') + 1] == (configuration or 'debug') for call in swift)


def test_build_script_rejects_unknown_configuration_before_tool_invocation(tmp_path):
    import os
    import shutil
    project = tmp_path / 'Project'; (project / 'scripts').mkdir(parents=True)
    shutil.copy2(ROOT / 'scripts/build-app.sh', project / 'scripts/build-app.sh')
    result = subprocess.run(['/bin/zsh', str(project / 'scripts/build-app.sh')],
        env=os.environ | {'KWIPML_BUILD_CONFIGURATION': 'unknown', 'PATH': str(tmp_path)},
        capture_output=True, text=True)
    assert result.returncode != 0 and 'must be debug or release' in result.stderr
    assert not (project / '.local').exists()


def test_executor_refuses_unapproved_copyright_without_staging(tmp_path):
    mod = module(); app, runtime, output, calls, good = execution_fixture(tmp_path)
    metadata = plistlib.loads((app / 'Contents/Info.plist').read_bytes()); metadata['NSHumanReadableCopyright'] = 'Unapproved owner'
    (app / 'Contents/Info.plist').write_bytes(plistlib.dumps(metadata))
    with pytest.raises(ValueError, match='Copyright'): mod.execute_package(plan(mod), app, runtime, b'synthetic-profile', output, run=good)
    assert not output.exists() and calls == []


def test_production_vendor_privacy_missing_runtime_binding_refuses_without_package(tmp_path):
    mod = module(); app, runtime, output, calls, good = execution_fixture(tmp_path)
    metadata = runtime / 'python/lib/python3.13/site-packages/grpcio-1.84.0.dist-info/METADATA'
    metadata.unlink()
    with pytest.raises(subprocess.CalledProcessError):
        mod.execute_package(plan(mod), app, runtime, b'synthetic-profile', output, run=good)
    assert not output.exists()
    assert not any(command[0] == '/usr/bin/codesign' and '--sign' in command for command in calls)
    assert not list(tmp_path.glob('.kwipml-production-*'))


def stage_existing_privacy(app, runtime):
    destination = app / 'Contents/Resources/VendorPrivacy'
    subprocess.run([sys.executable, str(ROOT / 'scripts/vendor_privacy.py'), '--runtime-root', str(runtime),
                    '--destination', str(destination)], check=True, capture_output=True)
    return destination


def test_production_verifies_existing_fresh_privacy_without_rewriting_input(tmp_path):
    mod = module(); app, runtime, output, calls, good = execution_fixture(tmp_path)
    privacy = stage_existing_privacy(app, runtime)
    before = {str(p.relative_to(privacy)): (p.read_bytes(), p.stat().st_mtime_ns) for p in privacy.rglob('*') if p.is_file()}
    mod.execute_package(plan(mod), app, runtime, b'synthetic-profile', output, run=good)
    assert output.exists()
    assert before == {str(p.relative_to(privacy)): (p.read_bytes(), p.stat().st_mtime_ns) for p in privacy.rglob('*') if p.is_file()}
    assert any('--verify-existing' in command for command in calls)


@pytest.mark.parametrize('mutation', ['stale', 'missing'])
def test_production_refuses_existing_stale_or_missing_privacy_bundle_without_overwrite(tmp_path, mutation):
    mod = module(); app, runtime, output, calls, good = execution_fixture(tmp_path)
    privacy = stage_existing_privacy(app, runtime)
    manifest = next(privacy.glob('*.bundle/Contents/Resources/PrivacyInfo.xcprivacy'))
    if mutation == 'stale': manifest.write_bytes(b'stale-input-preserved')
    else: manifest.unlink()
    before = {str(p.relative_to(privacy)): p.read_bytes() for p in privacy.rglob('*') if p.is_file()}
    with pytest.raises(subprocess.CalledProcessError):
        mod.execute_package(plan(mod), app, runtime, b'synthetic-profile', output, run=good)
    assert before == {str(p.relative_to(privacy)): p.read_bytes() for p in privacy.rglob('*') if p.is_file()}
    assert not output.exists()
    assert not any(command[0] == '/usr/bin/codesign' and '--sign' in command for command in calls)
    assert not list(tmp_path.glob('.kwipml-production-*'))


@pytest.mark.parametrize('existing', ['absent', 'fresh', 'stale'])
def test_packaged_shell_stages_or_verifies_privacy_before_mocked_signing(tmp_path, existing):
    import os
    import shutil
    app, runtime, _, _, _ = execution_fixture(tmp_path)
    if existing != 'absent':
        privacy = stage_existing_privacy(app, runtime)
        if existing == 'stale': next(privacy.glob('*.bundle/Contents/Resources/PrivacyInfo.xcprivacy')).write_bytes(b'stale')
    project = tmp_path / 'PackagedProject'; (project / 'scripts').mkdir(parents=True)
    (project / '.local').mkdir()
    shutil.copytree(app, project / '.local/KwipML POC.app')
    assets = project / 'docs/production/vendor-privacy-20261009'
    shutil.copytree(ROOT / 'docs/production/vendor-privacy-20261009', assets)
    shutil.copy2(ROOT / 'scripts/vendor_privacy.py', project / 'scripts/vendor_privacy.py')
    build = project / 'scripts/build-app.sh'; build.write_text('#!/bin/zsh\nexit 0\n'); build.chmod(0o755)
    old = project / '.local/KwipML Candidate.app'; old.mkdir(); (old / 'previous').write_text('retained candidate')
    calls = tmp_path / 'packaged-calls.jsonl'
    router = tmp_path / 'build-python'
    router.write_text(f'''#!{sys.executable}
import json,os,sys
from pathlib import Path
with open(os.environ['MOCK_PACKAGE_CALLS'],'a') as stream: stream.write(json.dumps(sys.argv[1:])+'\\n')
if Path(sys.argv[1]).name=='vendor_privacy.py': os.execv(sys.executable,[sys.executable]+sys.argv[1:])
'''); router.chmod(0o755)
    signer = tmp_path / 'mock-codesign'
    signer.write_text(f'''#!{sys.executable}
import json,os,sys
from pathlib import Path
with open(os.environ['MOCK_PACKAGE_CALLS'],'a') as stream: stream.write(json.dumps(['MOCK_CODESIGN']+sys.argv[1:])+'\\n')
if '--sign' in sys.argv:
    location=Path(sys.argv[-1]); app=next(p for p in [location,*location.parents] if p.suffix=='.app')
    assert len(list((app/'Contents/Resources/VendorPrivacy').glob('*.bundle')))==6
'''); signer.chmod(0o755)
    # Replace only the signing executable in a disposable script copy. No real
    # Swift, compiler, runtime attestation producer or codesign is invoked.
    script = project / 'scripts/build-packaged-app.sh'
    script.write_text((ROOT / 'scripts/build-packaged-app.sh').read_text().replace('/usr/bin/codesign', str(signer)))
    result = subprocess.run(['/bin/zsh', str(script), str(runtime)],
        env=os.environ | {'KWIPML_BUILD_PYTHON': str(router), 'MOCK_PACKAGE_CALLS': str(calls)},
        capture_output=True, text=True)
    invoked = [json.loads(line) for line in calls.read_text().splitlines()]
    privacy_index = next(i for i, call in enumerate(invoked) if Path(call[0]).name == 'vendor_privacy.py')
    if existing == 'stale':
        assert result.returncode != 0
        assert not any(call[0] == 'MOCK_CODESIGN' for call in invoked)
        assert (old / 'previous').read_text() == 'retained candidate'
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        signing_index = next(i for i, call in enumerate(invoked) if call[0] == 'MOCK_CODESIGN')
        assert privacy_index < signing_index
        assert len(list((old / 'Contents/Resources/VendorPrivacy').glob('*.bundle'))) == 6
        previous = list((project / '.local').glob('KwipML Candidate previous-*.app/previous'))
        assert len(previous) == 1 and previous[0].read_text() == 'retained candidate'
    assert not list((project / '.local').glob('packaged-stage-*'))


def test_package_access_normalization_preserves_bytes_and_executable_bits(tmp_path):
    mod = module()
    app = tmp_path / 'App'; app.mkdir(mode=0o700)
    resource = app / 'source-disclosure'; resource.mkdir(mode=0o700)
    notice = resource / 'NOTICE'; notice.write_bytes(b'exact notice'); notice.chmod(0o600)
    executable = app / 'worker'; executable.write_bytes(b'unchanged code'); executable.chmod(0o700)
    link = app / 'notice-link'; link.symlink_to('source-disclosure/NOTICE')
    mod.normalize_package_access(app)
    assert app.stat().st_mode & 0o777 == 0o755
    assert resource.stat().st_mode & 0o777 == 0o755
    assert notice.stat().st_mode & 0o777 == 0o644
    assert executable.stat().st_mode & 0o777 == 0o755
    assert notice.read_bytes() == b'exact notice' and executable.read_bytes() == b'unchanged code'
    assert link.is_symlink()
