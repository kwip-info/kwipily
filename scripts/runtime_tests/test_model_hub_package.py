import importlib.util
import plistlib
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[2]

@pytest.fixture
def package(monkeypatch):
    # Match direct script execution without relying on another test to have
    # inserted scripts/ into this process's import path during collection.
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    spec=importlib.util.spec_from_file_location('model_hub_package',ROOT/'scripts/model_hub_broker_package.py')
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value

def test_development_grant_is_only_the_exact_poc_app(tmp_path, package):
    app=tmp_path/'Development.app';contents=app/'Contents';contents.mkdir(parents=True)
    info=contents/'Info.plist';info.write_bytes(plistlib.dumps({'CFBundleIdentifier':'info.kwip.KwipML.POC'}))
    rights=package.development_entitlements(app)
    assert rights['com.apple.security.app-sandbox'] is True
    assert rights['com.apple.security.temporary-exception.files.absolute-path.read-only']==[str(app)+'/']
    assert not any('write' in name for name in rights)
    info.write_bytes(plistlib.dumps({'CFBundleIdentifier':'info.kwip.KwipML'}))
    with pytest.raises(ValueError):package.development_entitlements(app)
    link=tmp_path/'Alias.app';link.symlink_to(app,target_is_directory=True)
    with pytest.raises(ValueError):package.development_entitlements(link)


def test_embedded_helper_has_a_human_identity(tmp_path, package):
    app=tmp_path/'Development.app';contents=app/'Contents';contents.mkdir(parents=True)
    (contents/'Info.plist').write_bytes(plistlib.dumps({'CFBundleIdentifier':'info.kwip.KwipML.POC','CFBundleExecutable':'KwipML'}))
    products=tmp_path/'products';products.mkdir();(products/'KwipMLNetworkService').write_bytes(b'fixture')
    helper=package.embed(app,products);info=plistlib.loads((helper/'Contents/Info.plist').read_bytes())
    import json
    name=json.loads((ROOT/'Sources/KwipMLCore/Resources/ProductBrand.json').read_text())['name']
    assert info['CFBundleName']==info['CFBundleDisplayName']==name+' Network Service'
    assert info['CFBundleIdentifier']=='info.kwip.KwipML.POC.NetworkService'
    assert info['CFBundleExecutable']=='NetworkService' and info['KwipMLParentExecutable']=='KwipML'
    assert info['XPCService']['RunLoopType']=='NSRunLoop'
