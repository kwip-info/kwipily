"""Development embedding of the existing authenticated network-only helper.

Production assembly already embeds this helper. No parent/worker entitlement is added.
"""
from pathlib import Path
import plistlib
import shutil
import product_brand

def embed(app,products):
    app=Path(app);products=Path(products)
    info=plistlib.loads((app/'Contents/Info.plist').read_bytes())
    if info['CFBundleIdentifier']!='info.kwip.KwipML.POC':raise ValueError('This helper assembly is only for the POC identity')
    helper=app/'Contents/XPCServices/NetworkService.xpc';binary=helper/'Contents/MacOS/NetworkService';binary.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(products/'KwipMLNetworkService',binary)
    (helper/'Contents/Info.plist').write_bytes(plistlib.dumps(dict(CFBundleIdentifier=info['CFBundleIdentifier']+'.NetworkService',CFBundleExecutable='NetworkService',CFBundleName=product_brand.NAME+' Network Service',CFBundleDisplayName=product_brand.NAME+' Network Service',CFBundlePackageType='XPC!',CFBundleVersion='1',CFBundleShortVersionString='1',LSMinimumSystemVersion='14.0',KwipMLParentExecutable=info['CFBundleExecutable'],XPCService={'RunLoopType':'NSRunLoop','ServiceType':'Application'})))
    return helper

def development_entitlements(app):
    app=Path(app).absolute()
    if app.suffix!='.app' or app.is_symlink() or not app.is_dir():raise ValueError('Expected exact development app')
    info=plistlib.loads((app/'Contents/Info.plist').read_bytes())
    if info.get('CFBundleIdentifier')!='info.kwip.KwipML.POC':raise ValueError('Development peer access is POC-only')
    # The debug listener reads the exact enclosing executable's designated signature.
    # No runtime, user model folder, project root, or write permission is granted.
    return {'com.apple.security.app-sandbox':True,'com.apple.security.network.client':True,
        'com.apple.security.temporary-exception.files.absolute-path.read-only':[str(app)+'/']}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--app',required=True,type=Path);p.add_argument('--products',required=True,type=Path);p.add_argument('--development-entitlements',type=Path);a=p.parse_args();print(embed(a.app,a.products))
    if a.development_entitlements:a.development_entitlements.write_bytes(plistlib.dumps(development_entitlements(a.app)))
