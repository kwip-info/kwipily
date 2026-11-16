#!/usr/bin/env python3
"""Build a separately signed local gate06 harness; never modifies an installed app."""
import argparse
import json
from pathlib import Path
import plistlib
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent.parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--identity', required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise SystemExit('Output must be new; preserve earlier evidence')
    out.mkdir(parents=True)
    app = out / 'Kwip ML Isolation Probe.app'
    contents = app / 'Contents'
    macos = contents / 'MacOS'
    helpers = contents / 'Helpers'
    xpc = contents / 'XPCServices' / 'NetworkService.xpc'
    for folder in (macos, helpers, xpc / 'Contents/MacOS'):
        folder.mkdir(parents=True)
    for src, dst in [(ROOT / '.build/debug/KwipMLIsolationProbe', macos / 'Probe'),
                     (ROOT / '.build/debug/KwipMLIsolationProbe', helpers / 'WorkerProbe'),
                     (ROOT / '.build/debug/KwipMLNetworkService', xpc / 'Contents/MacOS/NetworkService')]:
        shutil.copy2(src, dst)
    def info(path, identifier, executable, package_type):
        data = {'CFBundleIdentifier': identifier, 'CFBundleExecutable': executable,
                'CFBundlePackageType': package_type, 'CFBundleVersion': '1',
                'CFBundleShortVersionString': '1.0', 'LSMinimumSystemVersion': '14.0'}
        if package_type == 'XPC!':
            data['XPCService'] = {'ServiceType': 'Application', 'RunLoopType': 'NSRunLoop'}
        path.write_bytes(plistlib.dumps(data))
    info(contents / 'Info.plist', 'info.kwip.KwipML', 'Probe', 'APPL')
    info(xpc / 'Contents/Info.plist', 'info.kwip.KwipML.NetworkService', 'NetworkService', 'XPC!')
    def sign(path, entitlements, identifier=None):
        command = ['codesign', '--force', '--sign', args.identity, '--options', 'runtime',
                   '--timestamp', '--entitlements', str(ROOT / 'packaging' / entitlements)]
        if identifier:
            command += ['--identifier', identifier]
        subprocess.run(command + [str(path)], check=True, capture_output=True)
    sign(helpers / 'WorkerProbe', 'Worker.entitlements', 'info.kwip.KwipML.WorkerProbe')
    sign(xpc, 'NetworkService.entitlements')
    sign(app, 'App.entitlements')
    subprocess.run(['codesign', '--verify', '--deep', '--strict', str(app)], check=True)
    sentinel = out / 'ungranted.txt'
    sentinel.write_text('read access must be denied\n')
    result = subprocess.run([str(macos / 'Probe'), str(sentinel)], capture_output=True, text=True, timeout=35)
    receipt = {'stdout': result.stdout, 'stderr': result.stderr, 'exit_code': result.returncode,
               'developer_id': args.identity, 'customer_app_modified': False}
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
    if result.returncode:
        raise SystemExit(result.returncode)

if __name__ == '__main__':
    main()
