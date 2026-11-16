"""Release policy/state controls; synthetic inputs never establish signed release evidence."""
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
from pathlib import Path
import plistlib
import sys
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))


def module():
    spec=importlib.util.spec_from_file_location('direct_release',ROOT/'scripts/direct_release.py')
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def inputs():
    now=datetime.now(timezone.utc);cert=b'synthetic certificate';identity=hashlib.sha1(cert).hexdigest().upper()
    profile={'UUID':'11111111-2222-3333-4444-555555555555','TeamIdentifier':['MT68X9Z6NQ'],
      'ApplicationIdentifierPrefix':['MT68X9Z6NQ'],'Platform':['OSX'],'ProvisionsAllDevices':True,
      'CreationDate':now-timedelta(days=1),'ExpirationDate':now+timedelta(days=30),
      'DeveloperCertificates':[cert], 'Entitlements':{'com.apple.application-identifier':'MT68X9Z6NQ.info.kwip.KwipML',
        'com.apple.developer.team-identifier':'MT68X9Z6NQ','keychain-access-groups':['MT68X9Z6NQ.*']}}
    return profile,identity,{identity:'Developer ID Application: Synthetic (MT68X9Z6NQ)'}


def test_profile_requires_exact_existing_developer_id_and_private_key_authority():
    mod=module();profile,identity,inventory=inputs()
    ent=mod.validate_profile(profile,identity,inventory)
    assert ent=={**mod.APP_ENTITLEMENTS,'com.apple.application-identifier':'MT68X9Z6NQ.info.kwip.KwipML','com.apple.developer.team-identifier':'MT68X9Z6NQ'}
    assert 'com.apple.security.network.client' not in ent and 'keychain-access-groups' not in ent
    for change in [ {'ProvisionsAllDevices':False},{'ProvisionedDevices':['any']},{'TeamIdentifier':['OTHER12345']},
      {'ExpirationDate':datetime.now(timezone.utc)-timedelta(seconds=1)},{'DeveloperCertificates':[b'other']} ]:
        with pytest.raises(ValueError):mod.validate_profile({**profile,**change},identity,inventory)
    for key,value in [('get-task-allow',True),('com.apple.security.get-task-allow',True),('com.apple.application-identifier','MT68X9Z6NQ.*')]:
        bad=copy.deepcopy(profile);bad['Entitlements'][key]=value
        with pytest.raises(ValueError):mod.validate_profile(bad,identity,inventory)
    with pytest.raises(ValueError):mod.validate_profile(profile,identity,{identity:'Apple Distribution: Synthetic (MT68X9Z6NQ)'})


def test_release_coordinates_and_captured_runtime_are_exact_and_bounded():
    mod=module()
    assert mod.coordinates('1.0.0','7','a'*40,'b'*64)=={'version':'1.0.0','build':7,'sourceCommit':'a'*40,'runtimeManifestSHA256':'b'*64}
    for args in [('2.0.0','7','a'*40,'b'*64),('1.0.0','07','a'*40,'b'*64),('1.0.0','7','short','b'*64),('1.0.0','7','a'*40,'bad'),('1.0.0','0','a'*40,'b'*64)]:
        with pytest.raises(ValueError):mod.coordinates(*args)


def test_symbol_uuids_must_match_actual_architecture_and_never_fabricate_third_party_symbols():
    mod=module();native='UUID: 11111111-2222-3333-4444-555555555555 (arm64) /private/path/KwipML\n'
    assert mod.symbols_match(native,native.replace('/private/path/KwipML','KwipML.dSYM'))
    assert not mod.symbols_match(native,native.replace('11111111','99999999'))
    assert not mod.symbols_match(native,native.replace('arm64','x86_64'))
    assert not mod.symbols_match(native,'')
    assert not mod.symbols_match('', '')


def test_notary_status_requires_matching_submission_id_and_accepted_status():
    mod=module();job='11111111-2222-3333-4444-555555555555'
    assert mod.accepted_notary({'id':job,'status':'Accepted'},job)
    for result in [{'id':job,'status':'In Progress'},{'id':job,'status':'Invalid'},{'id':'another','status':'Accepted'},{}]:
        assert not mod.accepted_notary(result,job)


def test_stage_plan_signs_only_fixed_inner_xpc_then_parent_and_never_uses_deep_signing(tmp_path):
    mod=module();commands=mod.signing_commands(tmp_path/'Kwip ML.app','A'*40,tmp_path/'App.entitlements')
    assert len(commands)==2 and commands[0][-1].endswith('/NetworkService.xpc')
    assert commands[1][-1].endswith('/Kwip ML.app')
    for command in commands:
        assert '--timestamp' in command and command[command.index('--options')+1]=='runtime'
        assert '--deep' not in command
    assert str(ROOT/'packaging/NetworkService.entitlements') in commands[0]


def test_receipt_rejects_unexpected_schema_and_path_or_digest_rebinding():
    mod=module();receipt={'schemaVersion':1,'coordinates':mod.coordinates('1.0.0','7','a'*40,'b'*64),'phase':'assembled','identity':'A'*40,'profileSHA256':'c'*64,'appEntitlements':{**mod.APP_ENTITLEMENTS,'com.apple.application-identifier':'MT68X9Z6NQ.info.kwip.KwipML','com.apple.developer.team-identifier':'MT68X9Z6NQ'}}
    assert mod.validate_receipt(receipt)==receipt
    for changes in [{'schemaVersion':True},{'phase':'published'},{'identity':'-'},{'profileSHA256':'bad'},{'appEntitlements':{**receipt['appEntitlements'],'com.apple.security.network.client':True}},{'coordinates':{**receipt['coordinates'],'build':False}}]:
        with pytest.raises(ValueError):mod.validate_receipt({**receipt,**changes})


def test_output_path_and_receipt_validation_refuse_symlink_escape_and_existing_evidence(tmp_path,monkeypatch):
    mod=module();monkeypatch.setattr(mod,'ROOT',tmp_path)
    allowed=tmp_path/'.local/direct-releases';allowed.mkdir(parents=True)
    assert mod.output_path(allowed/'v1-build7')==allowed/'v1-build7'
    with pytest.raises(ValueError):mod.output_path(tmp_path/'outside')
    (allowed/'link').symlink_to(tmp_path,target_is_directory=True)
    with pytest.raises(ValueError):mod.output_path(allowed/'link'/'escape')
    receipt=allowed/'receipt.json';mod.save(receipt,{'version':1});assert receipt.read_text().endswith('\n')
    receipt.with_name('receipt.json.tmp').write_text('interrupted previous write')
    with pytest.raises(FileExistsError):mod.save(receipt,{'version':2})
    assert '"version": 1' in receipt.read_text()


def test_notary_uncertain_upload_never_silently_repeats(tmp_path,monkeypatch):
    from types import SimpleNamespace
    mod=module();artifact=tmp_path/'app-notary.zip';artifact.write_bytes(b'synthetic')
    receipt={'phase':'assembled','appZipSHA256':mod.digest(artifact),'pendingUpload':'app'}
    monkeypatch.setattr(mod.subprocess,'run',lambda *a,**k:pytest.fail('Must not repeat an uncertain external upload'))
    with pytest.raises(ValueError,match='Uncertain prior upload'):mod.notary(SimpleNamespace(),tmp_path,receipt)


def test_plan_cli_is_unsigned_and_creates_no_output(tmp_path):
    import subprocess
    destination=ROOT/'.local/direct-releases/plan-test-only'
    assert not destination.exists()
    result=subprocess.run([sys.executable,str(ROOT/'scripts/direct_release.py'),'--output',str(destination),
      '--runtime',str(tmp_path),'--runtime-manifest-sha256','b'*64,'--build','7','--commit','a'*40,'--profile','not-opened','--identity','A'*40],capture_output=True,text=True)
    assert result.returncode==0 and '"phase": "plan"' in result.stdout
    assert not destination.exists()


def test_assembly_refuses_unprepared_updater_before_creating_artifacts(tmp_path,monkeypatch):
    from types import SimpleNamespace
    mod=module();monkeypatch.setattr(mod,'require_clean',lambda commit:None)
    def refuse():raise ValueError('Prepared local updater dependency required')
    monkeypatch.setattr(mod.sparkle_package,'preflight',refuse)
    out=tmp_path/'release'
    args=SimpleNamespace(version='1.0.0',build='8',commit='a'*40,runtime_manifest_sha256='b'*64)
    with pytest.raises(ValueError,match='Prepared local updater'):mod.assemble(args,out)
    assert not out.exists()


def test_production_assembly_refuses_staging_build_environment_before_any_work(tmp_path, monkeypatch):
    from types import SimpleNamespace
    mod = module()
    monkeypatch.setenv('KWIPML_COMMERCE_STAGING', '1')
    monkeypatch.setattr(mod, 'require_clean', lambda commit: pytest.fail('Must refuse before source/build/signing work'))
    with pytest.raises(ValueError, match='staging'):
        mod.assemble(SimpleNamespace(version='1.0.0', build='12', commit='a'*40, runtime_manifest_sha256='b'*64), tmp_path/'release')
