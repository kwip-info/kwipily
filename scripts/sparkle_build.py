#!/usr/bin/env python3
"""Build the exact patched sandboxed updater dependency; no signing or app integration."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess

ROOT=Path(__file__).resolve().parents[1]
COMMIT='eef1a539a373c1f1a320624b1130fc5de7b2e100'
DEVELOPER='/Applications/Xcode.app/Contents/Developer'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_digest(root):
    root=root.resolve(strict=True);records=[]
    for path in sorted(root.rglob('*')):
        kind=path.lstat().st_mode;name=str(path.relative_to(root))
        if stat.S_ISLNK(kind):
            try:target=path.resolve(strict=True)
            except OSError:raise ValueError('Broken dependency symlink') from None
            if not target.is_relative_to(root):raise ValueError('Dependency symlink escapes its bundle')
            records.append([name,'link',os.readlink(path)])
        elif stat.S_ISREG(kind):records.append([name,'file',sha(path),stat.S_IMODE(kind)])
        elif not stat.S_ISDIR(kind):raise ValueError('Unexpected dependency file type')
    if not records:raise ValueError('Empty dependency')
    return hashlib.sha256(json.dumps(records,separators=(',',':')).encode()).hexdigest()


def output_path(root,path):
    root=root.resolve();path=Path(os.path.abspath(path))
    if any(parent.is_symlink() for parent in [path,*path.parents]):raise ValueError('Output cannot traverse symlinks')
    if path.exists() or path==root/'.local' or not path.is_relative_to(root/'.local'):raise ValueError('New ignored local output directory required')
    return path


def parent_bundle(edition):
    if edition not in {'production','staging'}:raise ValueError('Unknown updater edition')
    return 'info.kwip.KwipML.POC' if edition=='staging' else 'info.kwip.KwipML'


def patch_bytes(edition='production'):
    parent=parent_bundle(edition)
    raw=(ROOT/'packaging/sparkle/peer-boundary.patch').read_bytes()
    if edition=='production':return raw
    return raw.replace(b'info.kwip.KwipML',parent.encode())+(ROOT/'packaging/sparkle/staging-local-network.patch').read_bytes()


def commands(source,derived,framework,*,edition='production'):
    parent=parent_bundle(edition)
    return [
      ['xcodebuild','-project',str(source/'Sparkle.xcodeproj'),'-scheme','Sparkle','-configuration','Release',
       '-derivedDataPath',str(derived),'ARCHS=arm64','ONLY_ACTIVE_ARCH=YES','MACOSX_DEPLOYMENT_TARGET=14.0',
       'DOWNLOADER_SANDBOXED_ENTITLEMENTS=Downloader/Downloader.entitlements',
       'XPC_SERVICE_BUNDLE_ID_PREFIX='+parent+'.Sparkle','SPARKLE_EMBED_INSTALLER_CONNECTION_XPC_SERVICE=1',
       'SPARKLE_EMBED_INSTALLER_STATUS_XPC_SERVICE=1','CODE_SIGNING_ALLOWED=NO','build'],
      ['xcodebuild','-create-xcframework','-framework',str(derived/'Build/Products/Release/Sparkle.framework'),'-output',str(framework)]]


def run(command,cwd,log=None,timeout=60):
    result=subprocess.run(command,cwd=cwd,env={**os.environ,'DEVELOPER_DIR':DEVELOPER},capture_output=True,text=True,timeout=timeout)
    if log:log.write_text(result.stdout+result.stderr)
    if result.returncode:raise ValueError('Dependency command failed; inspect its local log')
    return result.stdout.strip()


def build(source,out,*,edition='production'):
    parent=parent_bundle(edition)
    if not (source/'.git').exists():raise ValueError('Local upstream Git checkout required')
    if run(['git','rev-parse','HEAD'],source)!=COMMIT:raise ValueError('Wrong upstream commit')
    out.mkdir(parents=True,mode=0o700);clone=out/'source'
    patch=out/'peer-boundary.patch';patch.write_bytes(patch_bytes(edition))
    run(['git','clone','--no-hardlinks','--no-checkout',str(source),str(clone)],ROOT,out/'clone.log')
    run(['git','checkout','--detach',COMMIT],clone,out/'checkout.log')
    if run(['git','status','--porcelain'],clone):raise ValueError('Source clone not clean')
    run(['git','apply','--check',str(patch)],clone)
    run(['git','apply',str(patch)],clone)
    expected=run(['git','diff','--binary'],clone)
    source_status=run(['git','status','--porcelain'],clone)
    derived=out/'derived';framework=out/'Sparkle.xcframework'
    for index,command in enumerate(commands(clone,derived,framework,edition=edition)):
        run(command,clone,out/f'build-{index}.log',timeout=900)
    if run(['git','diff','--binary'],clone)!=expected or run(['git','status','--porcelain'],clone)!=source_status:raise ValueError('Source changed during dependency build')
    run(['git','apply','--reverse','--check',str(patch)],clone)
    # The linked binary and all delivered native helpers must be arm64, never host-default universal/x86.
    binaries=[]
    for path in framework.rglob('*'):
        if not path.is_file() or path.is_symlink():continue
        description=run(['/usr/bin/file','-b',str(path)],ROOT)
        if 'Mach-O' in description:
            if run(['/usr/bin/lipo','-archs',str(path)],ROOT)!='arm64':raise ValueError('Unexpected updater architecture')
            binaries.append(str(path.relative_to(framework)))
    if len(binaries)!=7:raise ValueError('Unexpected updater native image inventory')
    # Include the exact upstream notices; no dependency signature or release authority is claimed.
    (out/'LICENSE').write_bytes((clone/'LICENSE').read_bytes())
    receipt={'schemaVersion':1,'upstreamCommit':COMMIT,'patchSHA256':sha(patch),'licenseSHA256':sha(out/'LICENSE'),
      'architecture':'arm64','minimumMacOS':'14.0','toolchain':run(['xcodebuild','-version'],ROOT),
      'xcframeworkSHA256':tree_digest(framework),'nativeImages':binaries,'signed':False,'embedded':False,'publicLaunch':False}
    if edition=='staging':receipt.update(edition='staging',parentBundleIdentifier=parent)
    with (out/'dependency.json').open('x') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--edition',choices=['production','staging'],default='production')
    args=parser.parse_args()
    try:
        out=output_path(ROOT,args.output);print(json.dumps(build(args.source.resolve(),out,edition=args.edition),indent=2));return 0
    except (ValueError,OSError,subprocess.SubprocessError):
        print('Updater dependency build refused. Preserve local logs; no keys, app or installed release changed.');return 1

if __name__=='__main__':raise SystemExit(main())
