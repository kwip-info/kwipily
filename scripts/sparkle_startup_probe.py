#!/usr/bin/env python3
"""Sign and validate real SDK startup configuration without running update cycles."""
import argparse
import json
import os
from pathlib import Path
import plistlib
import subprocess

import sparkle_package as package
from sparkle_build import ROOT, DEVELOPER, output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--identity', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--developer-dir', default=DEVELOPER)
    parser.add_argument('--omit-verify-before-extraction', action='store_true',
                        help='Deliberate isolated negative control; never a release option')
    args = parser.parse_args()
    record = package.preflight()
    out = output_path(ROOT, args.output)
    out.mkdir(parents=True, mode=0o700)
    app = out/'Startup Probe.app'
    contents = app/'Contents'
    (contents/'MacOS').mkdir(parents=True)
    info = {'CFBundleIdentifier':'info.kwip.KwipML', 'CFBundleExecutable':'Probe',
            'CFBundleName':'Kwip ML startup probe', 'CFBundlePackageType':'APPL',
            'CFBundleVersion':'9', 'CFBundleShortVersionString':'1.0.0',
            'LSMinimumSystemVersion':'14.0', **package.info_keys()}
    if args.omit_verify_before_extraction:
        info.pop('SUVerifyUpdateBeforeExtraction', None)
    (contents/'Info.plist').write_bytes(plistlib.dumps(info))
    framework = package.embed(app, record)
    def run(command):
        return subprocess.run([str(x) for x in command], check=True, capture_output=True,
                              text=True, timeout=120, env={**os.environ,'DEVELOPER_DIR':args.developer_dir})
    run(['xcrun','clang','-fobjc-arc','-arch','arm64','-mmacosx-version-min=14.0',
         '-framework','Foundation','-framework','AppKit','-framework','Sparkle',
         '-F'+str(framework.parent),'-Wl,-rpath,@executable_path/../Frameworks',
         ROOT/'packaging/sparkle/startup-probe.m','-o',contents/'MacOS/Probe'])
    entitlements = out/'Downloader.entitlements'
    entitlements.write_bytes(plistlib.dumps(package.DOWNLOADER_ENTITLEMENTS))
    for command in package.signing_commands(framework,args.identity,entitlements):
        run(command)
    run(['/usr/bin/codesign','--force','--sign',args.identity,'--options','runtime','--timestamp',
         '--entitlements',ROOT/'packaging/App.entitlements',app])
    run(['/usr/bin/codesign','--verify','--deep','--strict',app])
    if package.content_digest(framework) != record['contentSHA256']:
        raise ValueError('Probe dependency correspondence changed')
    result = run([contents/'MacOS/Probe'])
    observed = json.loads(result.stdout)
    if args.omit_verify_before_extraction:
        passed = observed.get('started') is False and all(
            key in observed.get('error','') for key in ['SUVerifyUpdateBeforeExtraction','SURequireSignedFeed'])
    else:
        passed = observed == {'started':True,'error':'','domain':'','code':0}
        package.audit(app,record,args.identity,'MT68X9Z6NQ')
    receipt = {'passed':passed,'negativeControl':args.omit_verify_before_extraction,
               'observed':observed,'dependency':record,'stderr':result.stderr,
               'updateCycleRun':False,'installedAppChanged':False,'publicLaunch':False}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
