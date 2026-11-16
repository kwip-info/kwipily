"""Pinned direct updater packaging. Never grants networking to the app or ML worker."""
import hashlib
import json
import os
from pathlib import Path
import plistlib
import re
import shutil
import stat
import struct
import subprocess
import tempfile

from sparkle_build import COMMIT, tree_digest, parent_bundle, patch_bytes
from production_package import bounded_file
import runtime_audit

ROOT=Path(__file__).resolve().parents[1]
DOWNLOADER_ENTITLEMENTS={'com.apple.security.app-sandbox':True,'com.apple.security.network.client':True}
SERVICES={'Downloader':'DownloaderService','Installer':'InstallerLauncher',
          'InstallerConnection':'InstallerConnection','InstallerStatus':'InstallerStatus'}
NATIVE_PATHS=tuple(['Versions/B/Autoupdate','Versions/B/Sparkle','Versions/B/Updater.app/Contents/MacOS/Updater']+
    [f'Versions/B/XPCServices/{name}.xpc/Contents/MacOS/{name}' for name in SERVICES])
SIGNATURE_FILES={'Versions/B/_CodeSignature/CodeResources','Versions/B/Updater.app/Contents/_CodeSignature/CodeResources'}|{
    f'Versions/B/XPCServices/{name}.xpc/Contents/_CodeSignature/CodeResources' for name in SERVICES}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def run(argv):
    return subprocess.run([str(x) for x in argv],check=True,capture_output=True,text=True,timeout=120).stdout.strip()

def info_keys(*,edition='production'):
    parent_bundle(edition)
    origin='http://127.0.0.1:8768' if edition=='staging' else 'https://apps.kwip.info'
    return {'KWIPMLDirectUpdatesEnabled':True,'SUFeedURL':origin+'/updates/v1/arm64/appcast.xml',
      'SUPublicEDKey':'maob1E5oufy9KNFYZjxnd7/rEb/Ld13LqW6KFmc3aVc=',
      'SURequireSignedFeed':True,'SUVerifyUpdateBeforeExtraction':True,'SUSignedFeedFailureExpirationInterval':0,
      'SUEnableAutomaticChecks':True,'SUAllowsAutomaticUpdates':False,'SUAutomaticallyUpdate':False,
      'SUSendProfileInfo':False,'SUShowReleaseNotes':False,
      **{key:True for key in ['SUEnableDownloaderService','SUEnableInstallerLauncherService',
                             'SUEnableInstallerConnectionService','SUEnableInstallerStatusService']}}

def validate_record(value,patch,license_hash,*,edition='production'):
    parent=parent_bundle(edition)
    if not isinstance(value,dict):raise ValueError('Invalid updater receipt')
    if edition=='staging':
        if value.get('edition')!='staging' or value.get('parentBundleIdentifier')!=parent:raise ValueError('Updater edition differs')
    elif 'edition' in value or 'parentBundleIdentifier' in value:raise ValueError('Production updater must use production authority')
    if not isinstance(value,dict) or any(value.get(k)!=v for k,v in {
      'upstreamCommit':COMMIT,'architecture':'arm64','minimumMacOS':'14.0',
      'patchSHA256':patch,'licenseSHA256':license_hash}.items()):raise ValueError('Updater dependency authority differs')
    for key in ['patchSHA256','licenseSHA256','xcframeworkSHA256','contentSHA256']:
        if not isinstance(value.get(key),str) or not re.fullmatch('[a-f0-9]{64}',value[key]):raise ValueError('Invalid updater correspondence digest')
    return value

def normalized_native_hash(path):
    # codesign removal leaves __LINKEDIT's signature-expanded VM reservation intact.
    # Canonicalize ONLY this read-only segment's reservation, never code/data/load paths.
    data=bytearray(path.read_bytes())
    if len(data)<32 or data[:4]!=b'\xcf\xfa\xed\xfe':raise ValueError('Thin 64-bit updater image required')
    count,size=struct.unpack_from('<II',data,16);end=32+size;offset=32;found=False
    if end>len(data) or count>1024:raise ValueError('Malformed updater load commands')
    for _ in range(count):
        if offset+8>end:raise ValueError('Truncated updater load command')
        command,length=struct.unpack_from('<II',data,offset)
        if length<8 or offset+length>end:raise ValueError('Invalid updater load command')
        if command==0x19 and length>=72 and bytes(data[offset+8:offset+24]).rstrip(b'\0')==b'__LINKEDIT':
            vm_size,file_offset,file_size=struct.unpack_from('<QQQ',data,offset+32)
            maxprot,initprot=struct.unpack_from('<II',data,offset+56)
            if found or maxprot!=1 or initprot!=1 or vm_size<file_size or vm_size%16384 or file_offset+file_size>len(data):
                raise ValueError('Unexpected updater link-edit segment')
            struct.pack_into('<Q',data,offset+32,(file_size+16383)//16384*16384);found=True
        offset+=length
    if offset!=end or not found:raise ValueError('Missing updater link-edit segment')
    return hashlib.sha256(data).hexdigest()

def rpaths(loads):
    return re.findall(r'cmd LC_RPATH\s+cmdsize \d+\s+path (.+?) \(offset \d+\)',loads)

def validate_app_linkage(loads,dependencies):
    paths=rpaths(loads)
    if '@executable_path/../Frameworks' not in paths or any(x not in {'/usr/lib/swift','@executable_path/../Frameworks'} for x in paths):
        raise ValueError('Application framework lookup contains a development/unreviewed path')
    if '@rpath/Sparkle.framework/Versions/B/Sparkle' not in dependencies.splitlines():
        raise ValueError('Application must link the embedded updater')

def prepare_app_linkage(binary,products,developer_dir='/Applications/Xcode.app/Contents/Developer'):
    paths=rpaths(run(['/usr/bin/otool','-l',binary]))
    expected=str(products/'PackageFrameworks')
    toolchain=str(Path(developer_dir)/'Toolchains/XcodeDefault.xctoolchain/usr/lib')
    removable={expected,'@loader_path'}|{x for x in paths if re.fullmatch(re.escape(toolchain)+r'/swift(?:-[0-9.]+)?/macosx',x)}
    if any(x not in {'/usr/lib/swift','@executable_path/../Frameworks'}|removable for x in paths):
        raise ValueError('Unexpected SwiftPM framework lookup path')
    for path in paths:
        if path in removable:run(['/usr/bin/install_name_tool','-delete_rpath',path,binary])
    if '@executable_path/../Frameworks' not in paths:run(['/usr/bin/install_name_tool','-add_rpath','@executable_path/../Frameworks',binary])
    audit_app_linkage(binary)

def audit_app_linkage(binary):
    dependencies='\n'.join(line.strip().split(' (',1)[0] for line in run(['/usr/bin/otool','-L',binary]).splitlines()[1:])
    validate_app_linkage(run(['/usr/bin/otool','-l',binary]),dependencies)

def content_digest(framework,*,native_paths=NATIVE_PATHS):
    """Strip signatures only on disposable copies; resources and links remain byte exact."""
    if framework.is_symlink() or not framework.is_dir():raise ValueError('Real updater framework required')
    root=framework.resolve(strict=True);records=[]
    with tempfile.TemporaryDirectory(prefix='kwip-updater-correspondence-') as directory:
        for path in sorted(root.rglob('*')):
            name=path.relative_to(root).as_posix();mode=path.lstat().st_mode
            if stat.S_ISLNK(mode):
                if not path.resolve(strict=True).is_relative_to(root):raise ValueError('Updater symlink escapes framework')
                records.append([name,'link',os.readlink(path)])
            elif stat.S_ISREG(mode):
                if name in SIGNATURE_FILES:continue
                material=path
                if name in native_paths:
                    material=Path(directory)/str(len(records));shutil.copyfile(path,material)
                    run(['/usr/bin/codesign','--remove-signature',material])
                records.append([name,'file',normalized_native_hash(material) if name in native_paths else sha(material)])
            elif not stat.S_ISDIR(mode):raise ValueError('Unexpected updater file type')
    if not records:raise ValueError('Empty updater framework')
    return hashlib.sha256(json.dumps(records,separators=(',',':')).encode()).hexdigest()

def inspect_native(framework):
    actual=[]
    for path in framework.rglob('*'):
        if path.is_symlink() or not path.is_file():continue
        with path.open('rb') as stream:magic=stream.read(4)
        if magic not in runtime_audit.MAGICS:continue
        name=path.relative_to(framework).as_posix();actual.append(name)
        if run(['/usr/bin/lipo','-archs',path])!='arm64':raise ValueError('Unexpected updater architecture')
        loads=run(['/usr/bin/otool','-l',path])
        if not re.search(r'cmd LC_BUILD_VERSION\s+cmdsize \d+\s+platform 1\s+minos 14\.0\s',loads):raise ValueError('Unexpected updater platform/minimum OS')
        if rpaths(loads):raise ValueError('Unexpected updater helper framework lookup')
        dependencies=run(['/usr/bin/otool','-L',path]).splitlines()[1:]
        for line in dependencies:
            dependency=line.strip().split(' (',1)[0]
            if not dependency.startswith(('/usr/lib/','/System/Library/','@rpath/Sparkle.framework/','@loader_path/')):
                raise ValueError('Updater links an unreviewed dependency path')
    if set(actual)!=set(NATIVE_PATHS) or len(actual)!=7:raise ValueError('Unexpected updater native inventory')

def dependency_path(edition):
    parent_bundle(edition)
    return ROOT/('.local/sparkle-staging' if edition=='staging' else '.local/sparkle')

def preflight(*,edition='production'):
    dependency=dependency_path(edition);framework=dependency/'Sparkle.xcframework/macos-arm64/Sparkle.framework'
    for path in [dependency,dependency/'Sparkle.xcframework',framework]:
        if path.is_symlink() or not path.is_dir():raise ValueError('Prepared local updater dependency required')
    receipt=json.loads(bounded_file(dependency/'dependency.json',65536))
    patch_hash=hashlib.sha256(patch_bytes(edition)).hexdigest();license_path=ROOT/'packaging/sparkle/LICENSE'
    if sha(dependency/'LICENSE')!=sha(license_path):raise ValueError('Updater notices differ')
    value={**receipt,'contentSHA256':content_digest(framework)}
    validate_record(value,patch_hash,sha(license_path),edition=edition)
    if type(receipt.get('schemaVersion')) is not int or receipt['schemaVersion']!=1 or any(receipt.get(k) is not False for k in ['signed','embedded','publicLaunch']):
        raise ValueError('Unrecognized updater build receipt')
    expected={'macos-arm64/Sparkle.framework/'+x for x in NATIVE_PATHS}
    if not isinstance(receipt.get('nativeImages'),list) or len(receipt['nativeImages'])!=7 or set(receipt['nativeImages'])!=expected:
        raise ValueError('Updater receipt native inventory differs')
    if tree_digest(dependency/'Sparkle.xcframework')!=receipt['xcframeworkSHA256']:raise ValueError('Updater dependency tree changed')
    inspect_native(framework)
    keys=['upstreamCommit','patchSHA256','licenseSHA256','architecture','minimumMacOS','xcframeworkSHA256','contentSHA256']
    if edition=='staging':keys+=['edition','parentBundleIdentifier']
    return {k:value[k] for k in keys}

def embed(app,record,*,edition='production'):
    validate_record(record,hashlib.sha256(patch_bytes(edition)).hexdigest(),sha(ROOT/'packaging/sparkle/LICENSE'),edition=edition)
    source=dependency_path(edition)/'Sparkle.xcframework/macos-arm64/Sparkle.framework'
    framework=app/'Contents/Frameworks/Sparkle.framework';framework.parent.mkdir(parents=True,exist_ok=True)
    shutil.copytree(source,framework,symlinks=True)
    notices=app/'Contents/Resources/ThirdParty/Sparkle';notices.mkdir(parents=True)
    shutil.copyfile(ROOT/'packaging/sparkle/LICENSE',notices/'LICENSE')
    (notices/'peer-boundary.patch').write_bytes(patch_bytes(edition))
    (notices/'dependency.json').write_text(json.dumps(record,sort_keys=True,indent=2)+'\n')
    return framework

def signing_commands(framework,identity,entitlements,*,edition='production'):
    parent=parent_bundle(edition)
    targets=[(framework/f'Versions/B/XPCServices/{name}.xpc',parent+'.Sparkle.'+identifier)
      for name,identifier in SERVICES.items()]+[
      (framework/'Versions/B/Autoupdate','org.sparkle-project.Sparkle.Autoupdate'),
      (framework/'Versions/B/Updater.app','org.sparkle-project.Sparkle.Updater'),
      (framework,'org.sparkle-project.Sparkle')]
    commands=[]
    for target,identifier in targets:
        command=['/usr/bin/codesign','--force','--sign',identity,'--timestamp','--options','runtime','--identifier',identifier]
        if target.name=='Downloader.xpc':command+=['--entitlements',str(entitlements)]
        commands.append(command+[str(target)])
    return commands

def audit(app,record,identity,team,*,edition='production'):
    validate_record(record,hashlib.sha256(patch_bytes(edition)).hexdigest(),sha(ROOT/'packaging/sparkle/LICENSE'),edition=edition)
    framework=app/'Contents/Frameworks/Sparkle.framework';inspect_native(framework)
    if content_digest(framework)!=record['contentSHA256']:raise ValueError('Signed updater differs from compiled dependency')
    info=plistlib.loads(bounded_file(app/'Contents/Info.plist',65536))
    if any(type(info.get(k)) is not type(v) or info.get(k)!=v for k,v in info_keys(edition=edition).items()):raise ValueError('Updater configuration differs')
    notices=app/'Contents/Resources/ThirdParty/Sparkle'
    if sha(notices/'LICENSE')!=record['licenseSHA256'] or sha(notices/'peer-boundary.patch')!=record['patchSHA256'] or json.loads(bounded_file(notices/'dependency.json',65536))!=record:
        raise ValueError('Delivered updater source/notices differ')
    for command in signing_commands(framework,identity,Path('unused'),edition=edition):
        target=Path(command[-1]);identifier=command[command.index('--identifier')+1]
        runtime_audit.verify_distribution_signature(target,identity,team,channel='direct')
        details=subprocess.run(['/usr/bin/codesign','--display','--verbose=4',str(target)],check=True,capture_output=True,text=True,timeout=30)
        if ('Identifier='+identifier) not in (details.stdout+'\n'+details.stderr).splitlines():raise ValueError('Updater helper identity differs')
        output=subprocess.run(['/usr/bin/codesign','--display','--entitlements',':-',str(target)],check=True,capture_output=True,timeout=30).stdout
        ent=plistlib.loads(output) if output.strip() else {}
        expected=DOWNLOADER_ENTITLEMENTS if target.name=='Downloader.xpc' else {}
        if ent!=expected:raise ValueError('Updater helper entitlements differ')
