#!/usr/bin/env python3
"""Assemble a private, signed staging app from current source; never publish or notarize."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys

import direct_release as release
import sparkle_package as updater
import database_broker_package
import mcp_broker_package
from production_package import bounded_file, decode_profile, normalize_package_access, parse_identities

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = 'info.kwip.KwipML.POC'


def plan(profile, identity, inventory):
    entitlements = release.validate_profile(profile, identity, inventory, bundle=BUNDLE)
    return {'schemaVersion':1, 'environment':'staging', 'bundleIdentifier':BUNDLE,
            'origin':'http://127.0.0.1:8768', 'callbackScheme':'kwipml-staging',
            'appEntitlements':entitlements, 'published':False, 'notarized':False}


def output_path(value):
    if '..' in Path(value).parts:raise ValueError('Parent traversal staging output refused')
    path = Path(value).absolute()
    allowed = ROOT/'.local/staging-apps'
    if not path.is_relative_to(allowed) or path == allowed:
        raise ValueError('Private staging output must be under .local/staging-apps')
    if any(parent.is_symlink() for parent in [path, *path.parents]):
        raise ValueError('Symlink staging output refused')
    if path.exists():raise ValueError('Preserve existing staging evidence; use a fresh output')
    return path


def build_environment(developer_dir,*,updates=False):
    return {**os.environ,'DEVELOPER_DIR':developer_dir,'KWIPML_COMMERCE_STAGING':'1',
            'KWIPML_DIRECT_UPDATES':'1' if updates else '0'}

def assemble(args):
    release.require_clean(args.commit)
    coordinates=release.coordinates(args.version,args.build,args.commit,args.runtime_manifest_sha256)
    dependency=updater.preflight(edition='staging') if args.updates else None
    profile, raw = decode_profile(args.profile)
    prepared = plan(profile, args.identity, parse_identities(release.run(['/usr/bin/security','find-identity','-v'])))
    out = output_path(args.output)
    runtime = Path(args.runtime)
    release.real_tree(runtime)
    if release.digest(runtime/'runtime-manifest.json') != args.runtime_manifest_sha256:
        raise ValueError('Runtime manifest does not match supplied pin')
    if shutil.disk_usage(ROOT).free < 10*2**30:raise ValueError('At least10GiB free space required')
    out.mkdir(parents=True); os.chmod(out,0o700)
    release.audit_runtime(runtime,args.identity,out/'input-runtime-audit.json')
    env = build_environment(args.developer_dir,updates=args.updates)
    command = ['/usr/bin/xcrun','swift','build','--configuration','release',
               '--disable-experimental-strip-products','-debug-info-format','dwarf','--scratch-path',str(out/'build')]
    for product in ['KwipML','KwipMLNetworkService','KwipMLDatabaseSourceService','KwipMLMCPService']:
        result = subprocess.run(command+['--product',product],cwd=ROOT,env=env,capture_output=True,text=True,timeout=1800)
        (out/(product+'-build.log')).write_text(result.stdout+result.stderr)
        result.check_returncode()
    products = Path(release.run(command+['--show-bin-path'],env=env))
    app = out/'Kwip ML Staging.app'
    macos = app/'Contents/MacOS'; resources = app/'Contents/Resources'
    macos.mkdir(parents=True); resources.mkdir()
    shutil.copy2(products/'KwipML',macos/'KwipML')
    for name in ['KwipML_KwipMLCore.bundle','KwipML_KwipMLApp.bundle']:
        shutil.copytree(products/name,resources/name)
    bundled_runtime=database_broker_package.runtime_destination(app)
    bundled_runtime.parent.mkdir(parents=True,exist_ok=True)
    release.run(['/usr/bin/ditto',runtime,bundled_runtime],timeout=300)
    release.run([sys.executable,ROOT/'scripts/app_icon.py','--source',ROOT/'Sources/KwipMLApp/Resources/KwipMLWelcomeIcon.png','--output',resources/'KwipML.icns'])
    xpc = app/'Contents/XPCServices/NetworkService.xpc'; (xpc/'Contents/MacOS').mkdir(parents=True)
    shutil.copy2(products/'KwipMLNetworkService',xpc/'Contents/MacOS/NetworkService')
    info = {'CFBundleIdentifier':BUNDLE,'CFBundleName':'Kwip ML Staging','CFBundleDisplayName':'Kwip ML Staging',
        'CFBundleExecutable':'KwipML','CFBundlePackageType':'APPL','CFBundleShortVersionString':coordinates['version'],
        'CFBundleVersion':str(coordinates['build']),'CFBundleIconFile':'KwipML.icns','LSMinimumSystemVersion':'14.0',
        'NSHighResolutionCapable':True,'CFBundleSupportedPlatforms':['MacOSX'],
        'CFBundleURLTypes':[{'CFBundleURLName':BUNDLE+'.staging.auth','CFBundleURLSchemes':['kwipml-staging']}],
        'KwipMLSourceCommit':args.commit,'KwipMLCommerceEnvironment':'staging'}
    if args.updates:
        info.update(updater.info_keys(edition='staging'))
        info['NSAppTransportSecurity']={'NSAllowsLocalNetworking':True,'NSAllowsArbitraryLoads':False}
        updater.prepare_app_linkage(macos/'KwipML',products,args.developer_dir)
        updater.embed(app,dependency,edition='staging')
    (app/'Contents/Info.plist').write_bytes(plistlib.dumps(info))
    helper = {'CFBundleIdentifier':BUNDLE+'.NetworkService','CFBundleExecutable':'NetworkService','CFBundleName':'Kwip ML Network Service','CFBundleDisplayName':'Kwip ML Network Service',
        'CFBundlePackageType':'XPC!','CFBundleShortVersionString':coordinates['version'],'CFBundleVersion':str(coordinates['build']),
        'LSMinimumSystemVersion':'14.0','XPCService':{'RunLoopType':'NSRunLoop','ServiceType':'Application'},
        'NSAppTransportSecurity':{'NSAllowsLocalNetworking':True}}
    (xpc/'Contents/Info.plist').write_bytes(plistlib.dumps(helper))
    database_xpc=database_broker_package.embed(app,products)
    mcp_xpc=mcp_broker_package.embed(app,products)
    (app/'Contents/embedded.provisionprofile').write_bytes(raw)
    ent_path = out/'App.entitlements'; ent_path.write_bytes(plistlib.dumps(prepared['appEntitlements']))
    release.run([sys.executable,ROOT/'scripts/vendor_privacy.py','--runtime-root',bundled_runtime,'--destination',resources/'VendorPrivacy'])
    normalize_package_access(app)
    release.run([sys.executable,ROOT/'scripts/app_store_api_audit.py',app],timeout=180)
    if args.updates:
        downloader_ent=out/'Downloader.entitlements';downloader_ent.write_bytes(plistlib.dumps(updater.DOWNLOADER_ENTITLEMENTS))
        for command in updater.signing_commands(app/'Contents/Frameworks/Sparkle.framework',args.identity,downloader_ent,edition='staging'):release.run(command,timeout=180)
    for command in release.signing_commands(app,args.identity,ent_path):release.run(command,timeout=180)
    release.run(['/usr/bin/codesign','--verify','--deep','--strict',app],timeout=180)
    # Observe signed authority rather than infer it from the inputs.
    for target, expected in [(app,prepared['appEntitlements']), (xpc,plistlib.loads(bounded_file(ROOT/'packaging/NetworkService.entitlements',16384))), (database_xpc,plistlib.loads(bounded_file(ROOT/'packaging/DatabaseSourceService.entitlements',16384))), (mcp_xpc,plistlib.loads(bounded_file(mcp_broker_package.ENTITLEMENTS,16384)))]:
        result = subprocess.run(['/usr/bin/codesign','-d','--entitlements',':-',target],capture_output=True,check=True)
        if plistlib.loads(result.stdout) != expected:raise ValueError('Unexpected signed authority')
    release.audit_runtime(bundled_runtime,args.identity,out/'assembled-runtime-audit.json')
    if args.updates:
        updater.audit_app_linkage(macos/'KwipML')
        updater.audit(app,dependency,args.identity,release.TEAM,edition='staging')
    release.require_clean(args.commit)
    receipt = {**prepared,'sourceCommit':args.commit,'runtimeManifestSHA256':args.runtime_manifest_sha256,
               'profileSHA256':hashlib.sha256(raw).hexdigest(),'identity':args.identity,'appSHA256':release.digest(macos/'KwipML'),
               'version':coordinates['version'],'build':coordinates['build'],
               'databaseSourceBroker':True,'mcpSourceBroker':True,'nativeUAT':False,'offlineAcceptance':False,'updatesEnabled':args.updates}
    if dependency is not None:receipt['sparkleDependency']=dependency
    release.save(out/'staging.json',receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['output','runtime','runtime-manifest-sha256','commit','profile','identity']:
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--version',default='1.0.0')
    parser.add_argument('--build',default='1')
    parser.add_argument('--updates',action='store_true',help='Use only the separately prepared staging updater SDK and fixed loopback feed')
    parser.add_argument('--developer-dir',default='/Applications/Xcode.app/Contents/Developer')
    args = parser.parse_args()
    try:
        release.coordinates(args.version,args.build,args.commit,args.runtime_manifest_sha256)
        result = assemble(args)
        print(json.dumps({k:result[k] for k in ['environment','bundleIdentifier','sourceCommit','published','notarized']},indent=2))
        return 0
    except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError) as error:
        print('Staging assembly stopped: '+str(error),file=sys.stderr); return 1

if __name__ == '__main__':raise SystemExit(main())
