"""Packaging boundary controls; fake native inputs do not establish release acceptance."""
import importlib.util
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))

def module():
    spec=importlib.util.spec_from_file_location('sparkle_package',ROOT/'scripts/sparkle_package.py')
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

def test_only_downloader_receives_network_and_all_seven_images_are_signed_inside_out(tmp_path):
    m=module();commands=m.signing_commands(tmp_path/'Sparkle.framework','A'*40,tmp_path/'Downloader.entitlements')
    assert len(commands)==7 and commands[-1][-1].endswith('/Sparkle.framework')
    privileged=[c for c in commands if '--entitlements' in c]
    assert len(privileged)==1 and privileged[0][-1].endswith('/Downloader.xpc')
    assert m.DOWNLOADER_ENTITLEMENTS=={'com.apple.security.app-sandbox':True,'com.apple.security.network.client':True}
    for c in commands:
        assert '--deep' not in c and '--timestamp' in c and c[c.index('--options')+1]=='runtime'

def test_delivery_configuration_keeps_fixed_signed_feed_and_disables_fallback_and_silent_install():
    m=module();info=m.info_keys()
    assert info['SUFeedURL']=='https://apps.kwip.info/updates/v1/arm64/appcast.xml'
    assert info['SURequireSignedFeed'] is True and info['SUSignedFeedFailureExpirationInterval']==0
    assert info.get('SUVerifyUpdateBeforeExtraction') is True
    assert info['SUAllowsAutomaticUpdates'] is False and info['SUAutomaticallyUpdate'] is False
    assert info['SUEnableAutomaticChecks'] is True and info['SUSendProfileInfo'] is False
    assert len([x for x in info if x.startswith('SUEnable') and x.endswith('Service')])==4

def test_correspondence_detects_resource_changes_extra_files_and_symlink_escape(tmp_path):
    m=module();framework=tmp_path/'Sparkle.framework';framework.mkdir()
    resource=framework/'notice';resource.write_bytes(b'original')
    first=m.content_digest(framework,native_paths=())
    resource.write_bytes(b'altered');assert first!=m.content_digest(framework,native_paths=())
    resource.write_bytes(b'original');extra=framework/'extra';extra.write_bytes(b'new')
    assert first!=m.content_digest(framework,native_paths=())
    extra.unlink();(framework/'escape').symlink_to(tmp_path)
    with pytest.raises(ValueError):m.content_digest(framework,native_paths=())

def test_signature_exclusions_are_exact_not_arbitrary_code_resource_directories(tmp_path):
    m=module();framework=tmp_path/'Sparkle.framework';framework.mkdir();(framework/'notice').write_bytes(b'notice')
    first=m.content_digest(framework,native_paths=())
    legitimate=framework/'Versions/B/_CodeSignature/CodeResources';legitimate.parent.mkdir(parents=True);legitimate.write_bytes(b'signature')
    assert first==m.content_digest(framework,native_paths=())
    arbitrary=framework/'unreviewed/_CodeSignature/CodeResources';arbitrary.parent.mkdir(parents=True);arbitrary.write_bytes(b'extra')
    assert first!=m.content_digest(framework,native_paths=())

def test_receipt_rejects_wrong_source_patch_architecture_or_rebound_tree():
    m=module();receipt={'upstreamCommit':m.COMMIT,'architecture':'arm64','minimumMacOS':'14.0',
      'patchSHA256':'a'*64,'licenseSHA256':'b'*64,'xcframeworkSHA256':'c'*64,'contentSHA256':'d'*64}
    assert m.validate_record(receipt,'a'*64,'b'*64)==receipt
    for change in [{'upstreamCommit':'0'*40},{'architecture':'x86_64'},{'minimumMacOS':'13.0'},
      {'patchSHA256':'e'*64},{'licenseSHA256':'e'*64},{'contentSHA256':'bad'},{'xcframeworkSHA256':True}]:
        with pytest.raises(ValueError):m.validate_record({**receipt,**change},'a'*64,'b'*64)

def test_normalization_ignores_only_read_only_linkedit_reservation_not_code_bytes(tmp_path):
    import struct
    m=module();data=bytearray(256)
    struct.pack_into('<IIIIIIII',data,0,0xfeedfacf,0x100000c,0,2,1,72,0,0)
    struct.pack_into('<II16sQQQQIIII',data,32,0x19,72,b'__LINKEDIT',16384,16384,128,128,1,1,0,0)
    path=tmp_path/'native';path.write_bytes(data);first=m.normalized_native_hash(path)
    struct.pack_into('<Q',data,64,32768);path.write_bytes(data)
    assert m.normalized_native_hash(path)==first
    data[120]=1;path.write_bytes(data);assert m.normalized_native_hash(path)!=first
    struct.pack_into('<I',data,88,3);path.write_bytes(data)
    with pytest.raises(ValueError):m.normalized_native_hash(path)

def test_app_linkage_rejects_developer_paths_and_requires_embedded_framework_path():
    m=module()
    good='cmd LC_RPATH\n cmdsize 48\n path @executable_path/../Frameworks (offset 12)\n'
    m.validate_app_linkage(good,'@rpath/Sparkle.framework/Versions/B/Sparkle')
    for bad in [good+'cmd LC_RPATH\n cmdsize 48\n path /private/build/PackageFrameworks (offset 12)\n','']:
        with pytest.raises(ValueError):m.validate_app_linkage(bad,'@rpath/Sparkle.framework/Versions/B/Sparkle')
    with pytest.raises(ValueError):m.validate_app_linkage(good,'/private/build/Sparkle.framework/Sparkle')
