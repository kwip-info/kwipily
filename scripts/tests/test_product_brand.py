"""Visible naming must never migrate compatibility or paid identities."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_shared_brand_resource_is_presentation_only():
    brand = json.loads((ROOT / 'Sources/KwipMLCore/Resources/ProductBrand.json').read_text())
    assert brand == {
        'name': 'Kwipily', 'formerName': 'Kwip ML', 'publisher': 'KWIP LLC',
        'tagline': 'Let AI build tools without code',
        'businessTagline': 'AI & BI. Intelligence Evolved.',
    }
    identity = (ROOT / 'Sources/KwipMLCore/DesktopDeployment.swift').read_text()
    for stable in ['info.kwip.KwipML', '"kwipml"', '"KwipML"']:
        assert stable in identity


def test_packaging_uses_shared_visible_brand_and_keeps_legacy_payload_path():
    direct = (ROOT / 'scripts/direct_release.py').read_text()
    assert "'CFBundleDisplayName':product_brand.NAME" in direct
    assert "app=out/'Kwip ML.app'" in direct
    assert "'CFBundleExecutable':'KwipML'" in direct
    helper = (ROOT / 'scripts/model_hub_broker_package.py').read_text()
    assert "product_brand.NAME+' Network Service'" in helper


def test_isolated_mcp_helper_embeds_brand_without_extra_resource_grants(tmp_path):
    import sys
    sys.path.insert(0, str(ROOT / 'scripts'))
    import plistlib
    import mcp_broker_package
    app = tmp_path / 'Test.app'
    (app / 'Contents').mkdir(parents=True)
    (app / 'Contents/Info.plist').write_bytes(plistlib.dumps({'CFBundleIdentifier': 'info.kwip.KwipML.POC'}))
    products = tmp_path / 'products'
    products.mkdir()
    (products / 'KwipMLMCPService').write_bytes(b'test fixture, not a runnable helper')
    helper = mcp_broker_package.embed(app, products, development=True)
    info = plistlib.loads((helper / 'Contents/Info.plist').read_bytes())
    assert info['KwipMLProductName'] == 'Kwipily'
    assert info['CFBundleIdentifier'] == 'info.kwip.KwipML.POC.MCPService'
    assert info['CFBundleExecutable'] == 'MCPService'
    assert not (helper / 'Contents/Resources/KwipML_KwipMLCore.bundle').exists()
