import importlib.util
from pathlib import Path
import plistlib
import sys
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import database_broker_package as package


def test_embed_uses_parent_identity_and_network_authority_stays_in_helper(tmp_path):
    app=tmp_path/'Preview.app';(app/'Contents').mkdir(parents=True)
    (app/'Contents/Info.plist').write_bytes(plistlib.dumps({'CFBundleIdentifier':'info.kwip.KwipML.POC','CFBundleVersion':'1'}))
    products=tmp_path/'products';products.mkdir();(products/package.PRODUCT).write_bytes(b'fixture executable')
    helper=package.embed(app,products)
    info=plistlib.loads((helper/'Contents/Info.plist').read_bytes())
    assert info['CFBundleIdentifier']=='info.kwip.KwipML.POC.DatabaseSourceService'
    assert info['CFBundleExecutable']=='DatabaseSourceService'
    assert info['KwipMLParentExecutable']=='KwipML'
    assert plistlib.loads(package.ENTITLEMENTS.read_bytes())=={'com.apple.security.app-sandbox':True,'com.apple.security.network.client':True}
    assert 'com.apple.security.network.client' not in plistlib.loads((ROOT/'packaging/App.entitlements').read_bytes())
    assert 'com.apple.security.network.client' not in plistlib.loads((ROOT/'packaging/Worker.entitlements').read_bytes())


def test_signing_includes_database_helper_before_parent_only_when_present(tmp_path):
    import direct_release
    app=tmp_path/'Preview.app';products=tmp_path/'products'
    (app/'Contents').mkdir(parents=True);products.mkdir()
    (app/'Contents/Info.plist').write_bytes(plistlib.dumps({'CFBundleIdentifier':'info.kwip.KwipML'}))
    (products/package.PRODUCT).write_bytes(b'fixture')
    package.embed(app,products)
    commands=direct_release.signing_commands(app,'A'*40,tmp_path/'App.entitlements')
    assert len(commands)==3
    assert commands[1][-1].endswith('DatabaseSourceService.xpc') and commands[2][-1]==str(app)
    assert all('--deep' not in command for command in commands)


def test_helper_product_symlink_is_rejected(tmp_path):
    app=tmp_path/'Preview.app';(app/'Contents').mkdir(parents=True)
    (app/'Contents/Info.plist').write_bytes(plistlib.dumps({'CFBundleIdentifier':'info.kwip.KwipML'}))
    products=tmp_path/'products';products.mkdir();(products/'outside').write_bytes(b'fixture');(products/package.PRODUCT).symlink_to('outside')
    with pytest.raises(ValueError):package.embed(app,products)

@pytest.mark.parametrize('name',['../Other','/tmp/Other','..','a\\b',''])
def test_embedded_parent_executable_metadata_cannot_escape_bundle(tmp_path,name):
    app=tmp_path/'Preview.app';(app/'Contents').mkdir(parents=True)
    (app/'Contents/Info.plist').write_bytes(plistlib.dumps({'CFBundleIdentifier':'info.kwip.KwipML','CFBundleExecutable':name}))
    with pytest.raises(ValueError,match='basename'):package.embed(app,tmp_path/'products')

def test_debug_read_exception_is_exact_bundle_only_and_not_production(tmp_path):
    app=tmp_path/'Probe.app';(app/'Contents').mkdir(parents=True)
    (app/'Contents/Info.plist').write_bytes(plistlib.dumps({'CFBundleIdentifier':'info.kwip.KwipML.POC','CFBundlePackageType':'APPL'}))
    value=package.development_entitlements(app)
    paths=value['com.apple.security.temporary-exception.files.absolute-path.read-only']
    assert paths[0]==str(app.resolve())+'/'
    assert str(ROOT/'runner')+'/' in paths
    assert str(ROOT/'.venv')+'/' in paths
    assert str(ROOT/'.local/native-watchdog')+'/' in paths
    assert str(ROOT)+'/' not in paths and str(ROOT/'.local')+'/' not in paths
    assert str(Path.home())+'/' not in paths and '/' not in paths
    assert len(value)==3 and value['com.apple.security.app-sandbox'] and value['com.apple.security.network.client']
    assert len(plistlib.loads(package.ENTITLEMENTS.read_bytes()))==2
    app_link=tmp_path/'Linked.app';app_link.symlink_to(app)
    with pytest.raises(ValueError):package.development_entitlements(app_link)
    for invalid in (Path('/'),Path.home(),ROOT):
        with pytest.raises(ValueError):package.development_entitlements(invalid)

def test_runtime_location_supports_single_nested_and_legacy_layout(tmp_path):
    app=tmp_path/'Example.app';legacy=app/'Contents/Resources/KwipMLRuntime'
    nested=package.runtime_destination(app)
    assert package.locate_runtime(app)==nested
    legacy.mkdir(parents=True)
    assert package.locate_runtime(app)==legacy
    nested.mkdir(parents=True)
    with pytest.raises(ValueError,match='multiple'):package.locate_runtime(app)
    legacy.rmdir()
    assert package.locate_runtime(app)==nested
    nested.rmdir();nested.symlink_to(tmp_path)
    with pytest.raises(ValueError,match='physical'):package.locate_runtime(app)

def test_debug_packaged_runtime_needs_only_bundle_reads_and_fixed_relocation(tmp_path):
    app=tmp_path/'Probe.app';(app/'Contents').mkdir(parents=True)
    (app/'Contents/Info.plist').write_bytes(plistlib.dumps({'CFBundleIdentifier':'info.kwip.KwipML.POC','CFBundlePackageType':'APPL'}))
    package.runtime_destination(app).mkdir(parents=True)
    final=ROOT/'.local/KwipML Candidate.app'
    paths=package.development_entitlements(app,final)['com.apple.security.temporary-exception.files.absolute-path.read-only']
    assert paths==[str(app.resolve())+'/',str(final.resolve())+'/']
    with pytest.raises(ValueError,match='fixed local'):package.development_entitlements(app,tmp_path/'Other.app')
