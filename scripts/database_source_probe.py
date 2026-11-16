#!/usr/bin/env python3
"""Assemble a disposable DEBUG XPC database harness; never touch an installed app.

Run its printed executable with a new output directory and synthetic connection
settings/credentials on stdin. Normal application signing/release is separate.
"""
import argparse
from pathlib import Path
import plistlib
import shutil
import subprocess
from database_broker_package import embed,development_entitlements
ROOT=Path(__file__).resolve().parent.parent


def assemble(output,products):
    output=Path(output);products=Path(products)
    if output.exists():raise ValueError('Probe output must be new')
    app=output/'Kwip ML Database Probe.app';macos=app/'Contents/MacOS';resources=app/'Contents/Resources'
    macos.mkdir(parents=True);resources.mkdir()
    executable=macos/'DatabaseProbe'
    shutil.copy2(products/'KwipMLDatabaseSourceProbe',executable)
    info={'CFBundleIdentifier':'info.kwip.KwipML.POC','CFBundleExecutable':'DatabaseProbe','CFBundlePackageType':'APPL',
          'CFBundleVersion':'1','CFBundleShortVersionString':'0.1.0','LSMinimumSystemVersion':'14.0'}
    (app/'Contents/Info.plist').write_bytes(plistlib.dumps(info))
    (resources/'ProjectRoot.txt').write_text(str(ROOT)+'\n')
    helper=embed(app,products)
    entitlement_file=output/'DatabaseProbeHelper.entitlements'
    entitlement_file.write_bytes(plistlib.dumps(development_entitlements(app)))
    subprocess.run(['/usr/bin/codesign','--force','--sign','-','--entitlements',str(entitlement_file),str(helper)],check=True,capture_output=True)
    subprocess.run(['/usr/bin/codesign','--force','--sign','-',str(app)],check=True,capture_output=True)
    return executable


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path);parser.add_argument('--products',required=True,type=Path)
    args=parser.parse_args();print(assemble(args.output,args.products));return 0

if __name__=='__main__':raise SystemExit(main())
