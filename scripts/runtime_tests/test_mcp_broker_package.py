"""Synthetic packaging controls, never native signing or release acceptance."""
import json,plistlib,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import mcp_broker_package as package

def test_helper_embeds_exact_parent_identity_and_only_listener_authority(tmp_path):
    app=tmp_path/'Preview.app';(app/'Contents').mkdir(parents=True)
    (app/'Contents/Info.plist').write_bytes(plistlib.dumps({'CFBundleIdentifier':'info.kwip.KwipML.POC'}))
    products=tmp_path/'products';products.mkdir();(products/'KwipMLMCPService').write_bytes(b'synthetic executable')
    helper=package.embed(app,products,development=True)
    info=plistlib.loads((helper/'Contents/Info.plist').read_bytes())
    assert info['CFBundleIdentifier']=='info.kwip.KwipML.POC.MCPService'
    assert info['XPCService']['ServiceType']=='Application'
    assert plistlib.loads(package.ENTITLEMENTS.read_bytes())=={'com.apple.security.app-sandbox':True,'com.apple.security.network.server':True}
    for name in ('App','Worker'):
        assert not plistlib.loads((ROOT/'packaging'/f'{name}.entitlements').read_bytes()).get('com.apple.security.network.server')
    notices=app/'Contents/Resources/MCPNotices'
    inventory=json.loads((notices/'DEPENDENCIES.json').read_text())
    assert {item['identity'] for item in inventory['packages']}=={'swift-sdk','swift-system','swift-log','eventsource'}
    assert (notices/'swift-sdk/LICENSE').read_bytes()==(ROOT/'vendor/mcp-notices/swift-sdk/LICENSE').read_bytes()

def test_development_identity_cannot_replace_production_helper(tmp_path):
    app=tmp_path/'Release.app';(app/'Contents').mkdir(parents=True)
    (app/'Contents/Info.plist').write_bytes(plistlib.dumps({'CFBundleIdentifier':'info.kwip.KwipML'}))
    with pytest.raises(ValueError):package.embed(app,tmp_path,development=True)
    with pytest.raises(ValueError):package.embed(app,tmp_path,identifier='another.MCPService')

def test_mcp_is_signed_before_parent_and_never_with_blanket_network_permission(tmp_path):
    import direct_release
    app=tmp_path/'Preview.app';helper=app/'Contents/XPCServices/MCPService.xpc';helper.mkdir(parents=True)
    commands=direct_release.signing_commands(app,'A'*40,tmp_path/'App.entitlements')
    assert commands[-2][-1]==str(helper) and commands[-1][-1]==str(app)
    assert str(package.ENTITLEMENTS) in commands[-2] and '--deep' not in commands[-2]
