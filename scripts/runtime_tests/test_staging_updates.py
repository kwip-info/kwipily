"""Closed staging updater recipe; no signing, dependency build or installed update."""
import importlib.util
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
import sparkle_build as build
import sparkle_package as package
import staging_package as staging


def test_staging_peer_patch_preserves_exact_parent_and_never_admits_production_or_workers():
    original=(ROOT/'packaging/sparkle/peer-boundary.patch').read_bytes()
    assert build.patch_bytes('production') == original
    staged=build.patch_bytes('staging').decode()
    assert 'identifier \\"info.kwip.KwipML.POC\\"' in staged
    assert 'identifier \\"info.kwip.KwipML\\"' not in staged
    assert 'info.kwip.KwipML.POC.Sparkle.InstallerStatus' in staged
    assert 'identifier \\"python' not in staged
    assert 'NSAllowsLocalNetworking' in staged
    assert 'completionHandler(nil)' in staged and 'ephemeralSessionConfiguration' in staged
    for invalid in ['test','production-or-staging','']:
        with pytest.raises(ValueError):build.patch_bytes(invalid)


def test_staging_build_and_package_identity_are_separate_from_default_production(tmp_path):
    commands=build.commands(tmp_path/'source',tmp_path/'derived',tmp_path/'sdk',edition='staging')
    assert 'XPC_SERVICE_BUNDLE_ID_PREFIX=info.kwip.KwipML.POC.Sparkle' in commands[0]
    assert 'XPC_SERVICE_BUNDLE_ID_PREFIX=info.kwip.KwipML.Sparkle' in build.commands(tmp_path/'s',tmp_path/'d',tmp_path/'f')[0]
    stage=package.info_keys(edition='staging')
    assert stage['SUFeedURL']=='http://127.0.0.1:8768/updates/v1/arm64/appcast.xml'
    assert stage['SUPublicEDKey']==package.info_keys()['SUPublicEDKey']
    assert stage['SURequireSignedFeed'] and stage['SUVerifyUpdateBeforeExtraction']
    assert stage['SUSignedFeedFailureExpirationInterval']==0
    commands=package.signing_commands(tmp_path/'Sparkle.framework','D'*40,tmp_path/'ent',edition='staging')
    assert all('info.kwip.KwipML.Sparkle.' not in ' '.join(command) for command in commands)
    assert any('info.kwip.KwipML.POC.Sparkle.DownloaderService' in command for command in commands)


def test_stage_and_production_dependency_receipts_cannot_cross():
    record=dict(upstreamCommit=build.COMMIT,architecture='arm64',minimumMacOS='14.0',
                patchSHA256='a'*64,licenseSHA256='b'*64,xcframeworkSHA256='c'*64,contentSHA256='d'*64)
    assert package.validate_record(record,'a'*64,'b'*64)==record
    with pytest.raises(ValueError):package.validate_record(record,'a'*64,'b'*64,edition='staging')
    stage={**record,'edition':'staging','parentBundleIdentifier':'info.kwip.KwipML.POC'}
    assert package.validate_record(stage,'a'*64,'b'*64,edition='staging')==stage
    with pytest.raises(ValueError):package.validate_record(stage,'a'*64,'b'*64)


def test_staging_coordinates_and_updater_flags_remain_explicit_and_closed():
    assert staging.build_environment('/xcode',updates=True)['KWIPML_COMMERCE_STAGING']=='1'
    assert staging.build_environment('/xcode',updates=True)['KWIPML_DIRECT_UPDATES']=='1'
    assert staging.build_environment('/xcode',updates=False)['KWIPML_DIRECT_UPDATES']=='0'
    assert staging.release.coordinates('1.1.0','2','a'*40,'b'*64)['build']==2


def test_opted_in_assembly_refuses_unbound_sdk_before_profile_output_or_signing(tmp_path,monkeypatch):
    from types import SimpleNamespace
    calls=[]
    monkeypatch.setattr(staging.release,'require_clean',lambda commit:calls.append('source'))
    def refuse(*,edition):
        calls.append(edition)
        raise ValueError('Updater edition differs')
    monkeypatch.setattr(staging.updater,'preflight',refuse)
    monkeypatch.setattr(staging,'decode_profile',lambda path:pytest.fail('Profile access preceded updater refusal'))
    args=SimpleNamespace(commit='a'*40,version='1.1.0',build='2',runtime_manifest_sha256='b'*64,
                         updates=True,output=tmp_path/'must-not-exist')
    with pytest.raises(ValueError,match='Updater edition differs'):staging.assemble(args)
    assert calls==['source','staging']
    assert not args.output.exists()
