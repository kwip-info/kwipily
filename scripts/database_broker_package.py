#!/usr/bin/env python3
"""Embed the isolated database source helper; never change app/worker entitlements."""
import argparse
from pathlib import Path
import plistlib
import shutil
import json
import subprocess

ROOT=Path(__file__).resolve().parent.parent
ENTITLEMENTS=ROOT/'packaging/DatabaseSourceService.entitlements'
PRODUCT='KwipMLDatabaseSourceService'
SERVICE='DatabaseSourceService'
RUNTIME_RELATIVE=Path('Contents/XPCServices/DatabaseSourceService.xpc/Contents/Resources/KwipMLRuntime')


def runtime_destination(app):return Path(app)/RUNTIME_RELATIVE


def locate_runtime(app):
    app=Path(app);nested=runtime_destination(app);legacy=app/'Contents/Resources/KwipMLRuntime'
    present=lambda path:path.exists() or path.is_symlink()
    if present(nested) and present(legacy):raise ValueError('Application contains multiple runtime payloads')
    result=nested if present(nested) else legacy if present(legacy) else nested
    if result.is_symlink() or (result.exists() and not result.is_dir()):raise ValueError('Application runtime must be a physical directory')
    if result.exists() and not result.resolve().is_relative_to((app/'Contents').resolve()):raise ValueError('Application runtime escapes bundle contents')
    return result


def development_runtime_read_paths(root=ROOT):
    root=Path(root).resolve()
    executable=root/'.venv/bin/python'
    script="import json,sys,sysconfig;print(json.dumps({'base':sys.base_prefix,'executable':sys._base_executable,'stdlib':sysconfig.get_path('stdlib'),'platstdlib':sysconfig.get_path('platstdlib'),'libdir':sysconfig.get_config_var('LIBDIR'),'framework':sysconfig.get_config_var('PYTHONFRAMEWORK')}))"
    result=subprocess.run([str(executable),'-I','-B','-c',script],check=True,capture_output=True,text=True,timeout=10,env={'PATH':'/usr/bin:/bin'})
    if len(result.stdout)>16_384:raise ValueError('Development runtime identity exceeds its bound')
    info=json.loads(result.stdout);base=Path(info['base']).resolve()
    if base in (Path('/'),Path.home().resolve(),root):raise ValueError('Unsafe development interpreter base')
    values=[]
    for relative in ('runner','.venv','.local/native-watchdog'):
        path=(root/relative).resolve()
        if not path.is_dir() or not path.is_relative_to(root) or path==root:raise ValueError('Development code directory escapes checkout')
        values.append(str(path)+'/')
    registry=root/'Sources/KwipMLCore/Resources/components.json'
    if registry.is_symlink() or not registry.is_file():raise ValueError('Development registry is not fixed regular code')
    values.append(str(registry))
    actual=Path(info['executable']).resolve()
    if not actual.is_file() or not actual.is_relative_to(base):raise ValueError('Development interpreter escapes declared base')
    values.append(str(actual))
    for key in ('stdlib','platstdlib','libdir'):
        if not info.get(key):continue
        path=Path(info[key]).resolve()
        # platstdlib can point into the virtualenv; it is already covered.
        if path.is_relative_to(root/'.venv'):continue
        if not path.is_dir() or not path.is_relative_to(base) or path==base:raise ValueError('Development standard library escapes declared base')
        values.append(str(path)+'/')
    if info.get('framework'):
        library=base/info['framework']
        if library.is_file():values.append(str(library.resolve()))
    return sorted(set(values))


def development_entitlements(app,relocated_app=None):
    """Debug assembly only: read the exact enclosing code bundle for peer trust.

    Production/staging signing always uses the checked-in ENTITLEMENTS instead.
    Only fixed interpreter/runner code is readable, never the whole checkout,
    home directory, source project, credential store, or writable data.
    """
    app=Path(app)
    if app.suffix!='.app' or app.is_symlink() or not app.is_dir():raise ValueError('Expected an exact development app bundle')
    physical=app.resolve()
    if physical.suffix!='.app' or physical==Path.home().resolve() or physical==ROOT.resolve():raise ValueError('Development exception cannot cover a home or checkout directory')
    info_path=app/'Contents/Info.plist'
    if info_path.is_symlink() or info_path.stat().st_size>65_536:raise ValueError('Invalid development app declaration')
    info=plistlib.loads(info_path.read_bytes())
    if info.get('CFBundlePackageType')!='APPL' or not str(info.get('CFBundleIdentifier','')).startswith('info.kwip.KwipML'):raise ValueError('Development exception requires a declared KWIP app')
    values=plistlib.loads(ENTITLEMENTS.read_bytes())
    paths=[str(physical)+'/']
    if relocated_app is not None:
        final=Path(relocated_app)
        expected=ROOT/'.local/KwipML Candidate.app'
        if final.is_symlink() or final.absolute()!=expected.absolute():raise ValueError('Only the fixed local Candidate.app relocation is allowed')
        paths.append(str(final.resolve())+'/')
    if not runtime_destination(app).is_dir():paths+=development_runtime_read_paths()
    values['com.apple.security.temporary-exception.files.absolute-path.read-only']=paths
    return values


def embed(app,products,*,version=None,build=None):
    app=Path(app);products=Path(products)
    info_path=app/'Contents/Info.plist'
    info=plistlib.loads(info_path.read_bytes())
    identifier=info.get('CFBundleIdentifier')
    if not isinstance(identifier,str) or not identifier.startswith('info.kwip.KwipML'):raise ValueError('Database helper requires a KWIP parent app identity')
    parent_executable=info.get('CFBundleExecutable','KwipML')
    if not isinstance(parent_executable,str) or not parent_executable or parent_executable in ('.','..') or '/' in parent_executable or '\\' in parent_executable:raise ValueError('Database helper requires a safe parent executable basename')
    source=products/PRODUCT
    if source.is_symlink() or not source.is_file():raise ValueError('Database helper product is missing or not a regular file')
    helper=app/'Contents/XPCServices'/f'{SERVICE}.xpc'
    if helper.is_symlink():raise ValueError('Database helper destination cannot be a link')
    binary=helper/'Contents/MacOS'/SERVICE
    binary.parent.mkdir(parents=True,exist_ok=True)
    if binary.is_symlink():raise ValueError('Database helper executable cannot be a link')
    shutil.copy2(source,binary)
    values={'CFBundleIdentifier':identifier+'.'+SERVICE,'CFBundleExecutable':SERVICE,'CFBundlePackageType':'XPC!',
        'CFBundleShortVersionString':version or info.get('CFBundleShortVersionString','1.0'),
        'CFBundleVersion':str(build or info.get('CFBundleVersion','1')),'LSMinimumSystemVersion':'14.0',
        'XPCService':{'RunLoopType':'NSRunLoop','ServiceType':'Application'},
        'KwipMLParentExecutable':parent_executable}
    (helper/'Contents/Info.plist').write_bytes(plistlib.dumps(values))
    return helper


def signing_target(app):
    return Path(app)/'Contents/XPCServices/DatabaseSourceService.xpc',ENTITLEMENTS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app',required=True,type=Path);parser.add_argument('--products',type=Path)
    parser.add_argument('--development-entitlements',type=Path,help='DEBUG assembly only: write exact enclosing app read permission')
    parser.add_argument('--development-relocation',type=Path,help='DEBUG packaged Candidate.app final location only')
    args=parser.parse_args()
    if args.products:print(embed(args.app,args.products))
    elif not args.development_entitlements:parser.error('Provide --products or --development-entitlements')
    if args.development_relocation and not args.development_entitlements:parser.error('Relocation requires DEBUG entitlements')
    if args.development_entitlements:args.development_entitlements.write_bytes(plistlib.dumps(development_entitlements(args.app,args.development_relocation)))
    return 0

if __name__=='__main__':raise SystemExit(main())
