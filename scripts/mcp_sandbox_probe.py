#!/usr/bin/env python3
"""Assemble a new signed sandbox probe only; no installed app or owner grants."""
import argparse,plistlib,shutil,subprocess
from pathlib import Path
from mcp_broker_package import embed,ENTITLEMENTS,ROOT

def assemble(output,products):
    if output.exists():raise ValueError('Probe output must be new')
    app=output/'Kwip ML MCP Probe.app';mac=app/'Contents/MacOS';mac.mkdir(parents=True)
    shutil.copy2(products/'KwipMLMCPProbe',mac/'Probe')
    (app/'Contents/Info.plist').write_bytes(plistlib.dumps({'CFBundleIdentifier':'info.kwip.KwipML.POC','CFBundleExecutable':'Probe','CFBundlePackageType':'APPL','CFBundleVersion':'1','LSMinimumSystemVersion':'14.0'}))
    helper=embed(app,products,development=True)
    subprocess.run(['/usr/bin/codesign','--force','--sign','-','--entitlements',str(ENTITLEMENTS),str(helper)],check=True,capture_output=True)
    subprocess.run(['/usr/bin/codesign','--force','--sign','-','--entitlements',str(ROOT/'packaging/App.entitlements'),str(app)],check=True,capture_output=True)
    subprocess.run(['/usr/bin/codesign','--verify','--deep','--strict',str(app)],check=True,capture_output=True)
    return mac/'Probe'
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--products',type=Path,required=True);a=p.parse_args();print(assemble(a.output,a.products))
