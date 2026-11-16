#!/usr/bin/env python3
"""Build-only packager for the isolated, relocatable KwipML runtime payload."""
import argparse
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import stat
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('kwip_runtime_integrity',ROOT/'scripts/runtime_launch.py')
integrity=importlib.util.module_from_spec(spec);spec.loader.exec_module(integrity)
worker_source_hashes=integrity.worker_source_hashes
class _DistributionAudit:
    # Legacy isolated worker-verification fixtures need no production signing module.
    # Production paths fail closed if this build-owned verifier is missing.
    def __getattr__(self, name):
        spec=importlib.util.spec_from_file_location('kwip_distribution_audit',ROOT/'scripts/runtime_audit.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return getattr(module,name)

distribution_audit=_DistributionAudit()


def build_native_watchdog(destination, signing_identity=None, team_id=None, channel="mas"):
    if signing_identity or team_id or channel != "mas": distribution_audit.validate_distribution_selection(signing_identity,team_id,channel)
    """Build-host only; sign before inventory sealing, never at app launch."""
    if sys.platform!='darwin' or os.uname().machine!='arm64':
        raise ValueError('Native worker packaging currently requires an arm64 macOS build host')
    native=Path(destination)/'runner/native'
    source=native/'lifetime_watchdog.c'
    # Refuse symlink inputs anywhere in the copied worker before invoking tools.
    worker_source_hashes(destination)
    integrity.bounded_regular(source,integrity.MAX_FILE_BYTES)
    output=native/'lifetime_watchdog.dylib'
    if output.exists() or output.is_symlink():
        raise ValueError('Native watchdog output must not already exist')
    subprocess.run(['/usr/bin/xcrun','clang','-dynamiclib','-arch','arm64','-pthread',
                    '-Wall','-Wextra','-Werror','-O2','-mmacosx-version-min=14.0',
                    '-Wl,-install_name,@rpath/lifetime_watchdog.dylib',
                    str(source),'-o',str(output)],check=True,capture_output=True,timeout=60)
    subprocess.run(['/usr/bin/codesign','--force','--sign',signing_identity or '-',
                    '--identifier','info.kwip.KwipML.lifetime-watchdog',
                    *(['--timestamp','--options','runtime'] if channel == 'direct' else ['--timestamp=none']),str(output)],check=True,capture_output=True,timeout=30)
    subprocess.run(['/usr/bin/codesign','--verify','--strict',str(output)],
                   check=True,capture_output=True,timeout=30)

    if signing_identity: distribution_audit.verify_distribution_signature(output,signing_identity,team_id,**({"channel":channel} if channel != "mas" else {}))

def copy_worker_sources(destination, signing_identity=None, team_id=None, channel="mas"):
    """Copy canonical build inputs, excluding tests/caches; never used at app launch."""
    if channel != "mas": distribution_audit.validate_distribution_selection(signing_identity,team_id,channel)
    destination=Path(destination)
    shutil.copytree(ROOT/'runner',destination/'runner',symlinks=True,
                    ignore=shutil.ignore_patterns('tests','__pycache__','*.pyc','.pytest_cache',
                                                 'lifetime_watchdog.dylib'))
    shutil.copy2(ROOT/'Sources/KwipMLCore/Resources/components.json',destination/'components.json',follow_symlinks=False)
    shutil.copy2(ROOT/'scripts/runtime_launch.py',destination/'launch.py',follow_symlinks=False)
    build_native_watchdog(destination,signing_identity,team_id,**({"channel":channel} if channel != "mas" else {}))


def prepare_development_watchdog(destination):
    """Explicit build step for development; the app only verifies this pair."""
    destination=Path(destination)
    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        raise ValueError('Development watchdog destination must be a real directory')
    with tempfile.TemporaryDirectory(prefix='kwip-development-watchdog-') as staging:
        copy_worker_sources(staging)
        native=Path(staging)/'runner/native'
        library=(native/'lifetime_watchdog.dylib').read_bytes()
        manifest={'schemaVersion':1,
                  'sourceSHA256':hashlib.sha256((native/'lifetime_watchdog.c').read_bytes()).hexdigest(),
                  'librarySHA256':hashlib.sha256(library).hexdigest()}
        destination.mkdir(parents=True,exist_ok=True)
        for name,raw in [('lifetime_watchdog.dylib',library),
                         ('lifetime_watchdog.json',json.dumps(manifest,sort_keys=True).encode())]:
            target=destination/name
            if target.is_symlink():raise ValueError('Development watchdog files must not be symbolic links')
            descriptor,temporary=tempfile.mkstemp(prefix='.native-',dir=destination)
            try:
                with os.fdopen(descriptor,'wb') as stream:
                    stream.write(raw);stream.flush();os.fsync(stream.fileno())
                os.replace(temporary,target)
            finally:
                if os.path.exists(temporary):os.unlink(temporary)
        return manifest


def verify_current_worker(root):
    """Read-only build gate: internally valid old payloads must not pass as current."""
    root=Path(root)
    integrity.validate_worker(root)
    actual=worker_source_hashes(root)
    with tempfile.TemporaryDirectory(prefix='kwip-worker-source-') as staging:
        copy_worker_sources(staging)
        expected=worker_source_hashes(staging)
    if actual!=expected:
        changed=sorted(key for key in set(actual)|set(expected) if actual.get(key)!=expected.get(key))
        raise ValueError('Bundled worker differs from current build sources: '+', '.join(changed))
    return {'status':'matches-current-source','files':len(actual),'workerSourceHashes':actual}


def verify_production_worker(root, signing_identity, team_id, channel="mas"):
    """Read-only current-source verification across signed generated machine code."""
    distribution_audit.validate_distribution_selection(signing_identity, team_id, channel)
    root=Path(root)
    integrity.validate_worker(root)
    actual=worker_source_hashes(root)
    native='runner/native/lifetime_watchdog.dylib'
    if native not in actual: raise ValueError('Generated watchdog is missing from sealed inventory')
    distribution_audit.verify_distribution_signature(root/native,signing_identity,team_id,**({"channel":channel} if channel != "mas" else {}))
    with tempfile.TemporaryDirectory(prefix='kwip-production-source-') as staging:
        staging=Path(staging)
        expected_root=staging/'expected'
        copy_worker_sources(expected_root)
        expected=worker_source_hashes(expected_root)
        changed=sorted(key for key in set(actual)|set(expected) if key != native and actual.get(key)!=expected.get(key))
        if changed: raise ValueError('Bundled worker differs from current build sources: '+', '.join(changed))
        for label,source in [('actual',root/native),('expected',expected_root/native)]:
            target=staging/(label+'.dylib')
            shutil.copyfile(source,target,follow_symlinks=False)
            subprocess.run(['/usr/bin/codesign','--remove-signature',str(target)],check=True,capture_output=True,timeout=30)
        if (staging/'actual.dylib').read_bytes() != (staging/'expected.dylib').read_bytes():
            raise ValueError('Bundled generated watchdog differs from independently compiled current code')
    return {'status':'matches-current-source','files':len(actual),'workerSourceHashes':actual,'identity':signing_identity,'teamID':team_id}


def refresh_worker_integrity(root):
    """Refresh only explicit build-owned payloads before signing; never called at launch."""
    root=Path(root)
    manifest=json.loads(integrity.bounded_regular(root/'runtime-manifest.json',65536))
    if manifest.get('schemaVersion')!=1:raise ValueError('Unsupported runtime manifest schema')
    if integrity.bounded_regular(root/'launch.py',integrity.MAX_FILE_BYTES)!=(ROOT/'scripts/runtime_launch.py').read_bytes():raise ValueError('Copy the current canonical bootstrap to the payload before refreshing worker integrity')
    hashes=worker_source_hashes(root)
    manifest.update(workerIntegrityVersion=1,workerSourceHashes=hashes,entryHash=hashes['launch.py'],registryHash=hashes['components.json'])
    raw=json.dumps(manifest,indent=2).encode()
    if len(raw)>65536:raise ValueError('Runtime manifest exceeds 64 KB')
    temporary=root/'runtime-manifest.json.tmp';temporary.write_bytes(raw);os.replace(temporary,root/'runtime-manifest.json')
    return manifest


def require_empty_destination(destination):
    destination=Path(destination)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError('Runtime destination already exists and must be empty; existing content is preserved.')

def audit_links(root):
    root=Path(root).resolve()
    for path in root.rglob('*'):
        if path.is_symlink():
            target=path.resolve()
            if not target.is_relative_to(root): raise ValueError(f'Runtime symlink points outside payload: {path.relative_to(root)}')
            if not target.exists(): raise ValueError(f'Runtime symlink is broken: {path.relative_to(root)}')

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def clean_vendor_documentation(site_packages):
    """Remove the owner-excluded optional NetworkX blog citation before signing.

    This is documentation only; preserve executable code and license notices.
    """
    path=Path(site_packages)/'networkx/algorithms/operators/product.py'
    if not path.is_file(): return
    text=path.read_text()
    citation=('    [2] A. Faraji, "Corona Product in Graph Theory," Ali Faraji, May 11, 2021.\n'
              '        https://blog.alifaraji.ir/math/graph-theory/corona-product.html (accessed Dec. 07, 2021).\n')
    if 'blog.alifaraji.ir' not in text: return
    if text.count(citation)!=1:
        raise ValueError('NetworkX excluded citation changed; review documentation cleanup before packaging.')
    path.write_text(text.replace(citation,''))
    # An installed wheel may have bytecode containing the original docstring.
    for cached in (path.parent/'__pycache__').glob('product.*.pyc'):
        cached.unlink()


def _build(destination,python_version,standalone_archive,signing_identity=None,team_id=None):
    destination=Path(destination).resolve();require_empty_destination(destination)
    found=subprocess.check_output(['uv','python','find','--managed-python',python_version],text=True).strip()
    source=Path(found).resolve().parent.parent
    if not (source/'BUILD').is_file(): raise ValueError('Expected an Astral-managed python-build-standalone install with BUILD identity.')
    if python_version!='3.13.6' or os.uname().machine!='arm64' or (source/'BUILD').read_text().strip()!='20250814':raise ValueError('Standalone evidence currently binds only arm64 Python 3.13.6 build 20250814; other distributions remain unresolved.')
    destination.mkdir(parents=True,exist_ok=True)
    shutil.copytree(source,destination/'python',symlinks=True)
    python=destination/'python/bin'/f'python{python_version.rsplit(".",1)[0]}'
    # uv relocates LC_ID_DYLIB to its own install; restore a bundle-relative identity on the copy.
    library=destination/'python/lib'/f'libpython{python_version.rsplit(".",1)[0]}.dylib'
    subprocess.run(['/usr/bin/install_name_tool','-id','@rpath/'+library.name,str(library)],check=True)
    subprocess.run(['/usr/bin/codesign','--force','--sign','-','--timestamp=none',str(library)],check=True)
    # The copied standalone payload is ours to assemble; do not alter uv's source interpreter.
    marker=destination/'python/lib'/f'python{python_version.rsplit(".",1)[0]}'/'EXTERNALLY-MANAGED'
    if marker.exists(): marker.unlink()
    subprocess.run(['uv','pip','install','--python',str(python),'--build-constraints',str(ROOT/'requirements-runner.txt'),'-r',str(ROOT/'requirements-runner.txt')],check=True)
    clean_vendor_documentation(destination/'python/lib'/f'python{python_version.rsplit(".",1)[0]}'/'site-packages')
    copy_worker_sources(destination,signing_identity,team_id)
    # Console tools installed by pip have build-path shebangs and are not runtime APIs.
    for path in (destination/'python/bin').iterdir():
        if path.name not in {'python','python3',python.name}:
            path.unlink()
    # No generated bytecode with build-machine source paths is shipped or written at run time.
    for path in list(destination.rglob('__pycache__')):
        if path.is_dir(): shutil.rmtree(path)
    audit_links(destination)
    inventory=subprocess.check_output([str(python),'-I','-B',str(ROOT/'scripts/runtime_licenses.py'),str(destination)],text=True)
    licenses=json.loads(inventory)
    (destination/'license-inventory.json').write_text(json.dumps(licenses,indent=2))
    packages={item['name']:item['version'] for item in licenses['distributions']}
    runtime_version=subprocess.check_output([str(python),'-I','-B','-c','import platform;print(platform.python_version())'],text=True).strip()
    manifest={'schemaVersion':1,'pythonExecutable':str(python.relative_to(destination)),'entryPoint':'launch.py','registry':'components.json','pythonVersion':runtime_version,'packages':packages,'distribution':'Astral python-build-standalone','distributionBuild':(source/'BUILD').read_text().strip(),'requirementsHash':digest(ROOT/'requirements-runner.txt'),'entryHash':digest(destination/'launch.py'),'registryHash':digest(destination/'components.json'),'architecture':os.uname().machine,'licenseInventory':'license-inventory.json','signing':'unsigned payload; sign native code and containing bundle before distribution'}
    manifest['standaloneLicenseEvidence']=attach_standalone_licenses(destination,standalone_archive,python,library,manifest['distributionBuild'])
    (destination/'runtime-manifest.json').write_text(json.dumps(manifest,indent=2))
    manifest=refresh_worker_integrity(destination)
    if signing_identity: verify_production_worker(destination,signing_identity,team_id)
    else: verify_current_worker(destination)
    return manifest

KNOWN_STANDALONE_SHA256='f169b076da928b34ceff3578438fe68b65ef8009cae64fbc7150515a0e61d007'


def reconcile_standalone_zlib(inventory,*,distribution_build,binary_links,runtime_version):
    missing='licenses/LICENSE.zlib-ng.txt'
    entries=inventory.get('extensions',{}).get('zlib',[])
    exact=(inventory.get('archiveSHA256')==KNOWN_STANDALONE_SHA256
        and inventory.get('pythonVersion')=='3.13.6' and inventory.get('targetTriple')=='aarch64-apple-darwin'
        and distribution_build=='20250814' and missing in inventory.get('missingReferencedNotices',[])
        and len(entries)==1 and entries[0].get('links')==[{'name':'z','system':True}]
        and binary_links==['/usr/lib/libz.1.dylib'] and runtime_version=='1.2.12')
    return {'schemaVersion':1,'status':'explainedMetadataOvermatch' if exact else 'unresolved',
        'archiveSHA256':inventory.get('archiveSHA256'),'targetTriple':inventory.get('targetTriple'),
        'distributionBuild':distribution_build,'missingReference':missing,'binaryZlibLinks':binary_links,
        'observedZlibRuntimeVersion':runtime_version,'redistributionReviewComplete':False,
        'basis':'Inference from exact distribution metadata and observed binary linkage. Upstream matches every download sharing library name z, including system links; not a blanket license conclusion.',
        'sources':['https://raw.githubusercontent.com/astral-sh/python-build-standalone/20250814/pythonbuild/downloads.py','https://raw.githubusercontent.com/astral-sh/python-build-standalone/20250814/pythonbuild/utils.py'],
        'remainingMissingReferences':[p for p in inventory.get('missingReferencedNotices',[]) if not (exact and p==missing)]}


def attach_standalone_licenses(destination,archive,python,library,distribution_build):
    spec=importlib.util.spec_from_file_location('runtime_standalone_license_inventory',ROOT/'scripts/runtime_standalone_licenses.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    folder=Path(destination)/'standalone-license-evidence'
    inventory=module.inventory(archive,KNOWN_STANDALONE_SHA256,folder)
    output=subprocess.check_output(['/usr/bin/otool','-L',str(library)],text=True)
    links=[line.strip().split(' (',1)[0] for line in output.splitlines()[1:] if 'libz' in line]
    runtime_version=subprocess.check_output([str(python),'-I','-B','-c','import zlib;print(zlib.ZLIB_RUNTIME_VERSION)'],text=True).strip()
    result=reconcile_standalone_zlib(inventory,distribution_build=distribution_build,binary_links=links,runtime_version=runtime_version)
    path=folder/'reconciliation.json';path.write_text(json.dumps(result,indent=2)+'\n')
    return {'inventory':'standalone-license-evidence/inventory.json','reconciliation':'standalone-license-evidence/reconciliation.json','inventorySHA256':digest(folder/'inventory.json'),'reconciliationSHA256':digest(path),'redistributionReviewComplete':False}


def build(destination,python_version='3.13.6',standalone_archive=None,signing_identity=None,team_id=None):
    if signing_identity or team_id: distribution_audit.validate_distribution_selection(signing_identity,team_id)
    if standalone_archive is None:raise ValueError('An explicit verified standalone archive is required for a fresh runtime build.')
    # Snapshot a bounded regular descriptor before the archive is hashed or parsed.
    # A later external pathname replacement cannot change the bytes inventoried.
    descriptor=os.open(standalone_archive,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        info=os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size>128*1024*1024:raise ValueError('Standalone archive must be a bounded regular file.')
        with tempfile.TemporaryDirectory(prefix='kwip-runtime-archive-') as directory:
            snapshot=Path(directory)/'archive.tar.zst';h=hashlib.sha256();total=0
            with snapshot.open('wb') as target:
                while True:
                    chunk=os.read(descriptor,min(1024*1024,128*1024*1024+1-total))
                    if not chunk:break
                    total+=len(chunk)
                    if total>128*1024*1024:raise ValueError('Standalone archive exceeds size bound.')
                    h.update(chunk);target.write(chunk)
            if h.hexdigest()!=KNOWN_STANDALONE_SHA256:raise ValueError('Unknown standalone archive digest; dependency reconciliation remains unresolved.')
            return _build(destination,python_version,snapshot,signing_identity,team_id)
    finally:os.close(descriptor)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--output')
    mode.add_argument('--refresh-worker-integrity')
    mode.add_argument('--verify-current-worker')
    mode.add_argument('--verify-production-worker')
    mode.add_argument('--copy-worker-sources')
    mode.add_argument('--prepare-development-watchdog')
    parser.add_argument('--python-version',default='3.13.6')
    parser.add_argument('--standalone-archive')
    parser.add_argument('--sign-identity')
    parser.add_argument('--team-id')
    parser.add_argument('--channel',choices=['mas','direct'],default='mas')
    args=parser.parse_args()
    if args.channel == "direct" and not (args.sign_identity and args.team_id and (args.copy_worker_sources or args.verify_production_worker)):
        parser.error("Direct channel currently requires explicit signing selection and worker copy or verification")
    if (args.sign_identity or args.team_id) and not (args.verify_production_worker or args.copy_worker_sources or args.output): parser.error('Distribution signing selection requires --output, --copy-worker-sources or --verify-production-worker')
    if args.verify_production_worker:result=verify_production_worker(args.verify_production_worker,args.sign_identity,args.team_id,args.channel)
    elif args.copy_worker_sources:
        require_empty_destination(args.copy_worker_sources)
        copy_worker_sources(args.copy_worker_sources,args.sign_identity,args.team_id,args.channel)
        result={'status':'copied-worker-sources','signing':args.sign_identity or 'ad-hoc'}
    elif args.prepare_development_watchdog:result=prepare_development_watchdog(args.prepare_development_watchdog)
    elif args.verify_current_worker:result=verify_current_worker(args.verify_current_worker)
    elif args.refresh_worker_integrity:result=refresh_worker_integrity(args.refresh_worker_integrity)
    else:result=build(args.output,args.python_version,args.standalone_archive,args.sign_identity,args.team_id)
    print(json.dumps(result,indent=2))
