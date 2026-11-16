#!/usr/bin/env python3
"""Embed a separately sandboxed MCP listener; keep parent/worker permissions intact."""
from pathlib import Path
import plistlib,shutil
import product_brand
ROOT=Path(__file__).resolve().parent.parent
ENTITLEMENTS=ROOT/'packaging/MCPService.entitlements'

def embed(app,products,development=False,identifier=None):
    app=Path(app);products=Path(products)
    app_id=plistlib.loads((app/'Contents/Info.plist').read_bytes())['CFBundleIdentifier']
    if development and app_id!='info.kwip.KwipML.POC':raise ValueError('Development helper requires POC app identity')
    expected=app_id+'.MCPService'
    if identifier is not None and identifier!=expected:raise ValueError('MCP helper identity differs from its enclosing app')
    helper=app/'Contents/XPCServices/MCPService.xpc';mac=helper/'Contents/MacOS';mac.mkdir(parents=True,exist_ok=True)
    shutil.copy2(products/'KwipMLMCPService',mac/'MCPService')
    (helper/'Contents/Info.plist').write_bytes(plistlib.dumps({'KwipMLProductName':product_brand.NAME,'CFBundleName':product_brand.NAME+' MCP Service','CFBundleDisplayName':product_brand.NAME+' MCP Service','CFBundleIdentifier':expected,'CFBundleExecutable':'MCPService','CFBundlePackageType':'XPC!','CFBundleVersion':'1','CFBundleShortVersionString':'1','LSMinimumSystemVersion':'14.0','XPCService':{'ServiceType':'Application'}}))
    notices=app/'Contents/Resources/MCPNotices'
    shutil.copytree(ROOT/'vendor/mcp-notices',notices,dirs_exist_ok=True)
    return helper

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--app',required=True,type=Path)
    parser.add_argument('--products',required=True,type=Path)
    parser.add_argument('--development',action='store_true')
    args=parser.parse_args()
    print(embed(args.app,args.products,args.development))
