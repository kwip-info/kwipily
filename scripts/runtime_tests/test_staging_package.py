"""Private staging assembly must never borrow the production app identity."""
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))

def module():
    spec = importlib.util.spec_from_file_location('staging_package', ROOT/'scripts/staging_package.py')
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value

def profile():
    cert = b'synthetic certificate'; identity = hashlib.sha1(cert).hexdigest().upper()
    now = datetime.now(timezone.utc)
    return {'TeamIdentifier':['MT68X9Z6NQ'], 'ApplicationIdentifierPrefix':['MT68X9Z6NQ'],
        'Platform':['OSX'], 'ProvisionsAllDevices':True, 'CreationDate':now-timedelta(days=1),
        'ExpirationDate':now+timedelta(days=1), 'DeveloperCertificates':[cert],
        'Entitlements':{'com.apple.application-identifier':'MT68X9Z6NQ.info.kwip.KwipML.POC',
            'com.apple.developer.team-identifier':'MT68X9Z6NQ', 'keychain-access-groups':['MT68X9Z6NQ.*']}}, identity

def test_staging_plan_requires_exact_poc_profile_and_preserves_worker_boundary():
    mod = module(); value, identity = profile()
    inventory = {identity:'Developer ID Application: Synthetic (MT68X9Z6NQ)'}
    with pytest.raises(ValueError): mod.release.validate_profile(value, identity, inventory)
    plan = mod.plan(value, identity, inventory)
    assert plan['bundleIdentifier'] == 'info.kwip.KwipML.POC'
    assert plan['callbackScheme'] == 'kwipml-staging'
    assert plan['origin'] == 'http://127.0.0.1:8768'
    assert plan['appEntitlements']['com.apple.application-identifier'] == 'MT68X9Z6NQ.info.kwip.KwipML.POC'
    assert 'com.apple.security.network.client' not in plan['appEntitlements']
    assert plan['published'] is False and plan['notarized'] is False
    value['Entitlements']['com.apple.application-identifier'] = 'MT68X9Z6NQ.info.kwip.KwipML'
    with pytest.raises(ValueError): mod.plan(value, identity, inventory)

def test_staging_output_cannot_overwrite_release_or_installed_app(tmp_path, monkeypatch):
    mod = module(); monkeypatch.setattr(mod, 'ROOT', tmp_path)
    allowed = tmp_path/'.local/staging-apps'; allowed.mkdir(parents=True)
    assert mod.output_path(allowed/'acceptance-1') == allowed/'acceptance-1'
    for path in [tmp_path/'.local/direct-releases/rc', tmp_path/'Applications/Kwip ML.app', allowed/'..'/'direct-releases'/'escape']:
        with pytest.raises(ValueError): mod.output_path(path)
    (allowed/'link').symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError): mod.output_path(allowed/'link'/'escape')
