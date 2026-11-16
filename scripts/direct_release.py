#!/usr/bin/env python3
"""Local direct-channel assembly and resumable Apple notarization. Never publishes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import plistlib
import re
import shutil
import subprocess
import sys
import tempfile
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from production_package import APP_ENTITLEMENTS, bounded_file, decode_profile, parse_identities, normalize_package_access, utc
import runtime_audit
import sparkle_package
import database_broker_package
import mcp_broker_package
import packaged_boundary
import product_brand
from release_operation_events import measured
from verification_lock import compute_lock, lock_path

TEAM='MT68X9Z6NQ'
BUNDLE='info.kwip.KwipML'
PHASES={'assembled','appSubmitted','appAccepted','dmgBuilt','dmgSubmitted','verified'}

def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def coordinates(version,build,commit,runtime_hash):
    if not isinstance(version,str) or not re.fullmatch(r'1\.(?:0|[1-9][0-9]{0,3})\.(?:0|[1-9][0-9]{0,3})',version):raise ValueError('Only perpetual v1 versions are supported')
    if not isinstance(build,str) or not re.fullmatch(r'[1-9][0-9]{0,8}',build):raise ValueError('Build must be a canonical positive integer')
    if not isinstance(commit,str) or not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('Exact source commit required')
    if not isinstance(runtime_hash,str) or not re.fullmatch('[0-9a-f]{64}',runtime_hash):raise ValueError('Exact runtime manifest hash required')
    return {'version':version,'build':int(build),'sourceCommit':commit,'runtimeManifestSHA256':runtime_hash}


def validate_profile(profile,identity,inventory,*,bundle=BUNDLE):
    if bundle not in {BUNDLE, "info.kwip.KwipML.POC"}:raise ValueError("Unreviewed app identity")
    now=datetime.now(timezone.utc)
    if not re.fullmatch('[A-F0-9]{40}',identity) or not inventory.get(identity,'').startswith('Developer ID Application: ') or not inventory[identity].endswith(' ('+TEAM+')'):
        raise ValueError('Exact existing Developer ID private-key identity required')
    if profile.get('TeamIdentifier')!=[TEAM] or profile.get('ApplicationIdentifierPrefix')!=[TEAM] or profile.get('Platform')!=['OSX'] or profile.get('ProvisionsAllDevices') is not True or 'ProvisionedDevices' in profile:
        raise ValueError('Exact macOS Developer ID profile required')
    if not utc(profile.get('CreationDate'))<=now<utc(profile.get('ExpirationDate')):raise ValueError('Profile is not currently valid')
    ent=profile.get('Entitlements',{})
    if ent.get('com.apple.application-identifier')!=TEAM+'.'+bundle or ent.get('com.apple.developer.team-identifier')!=TEAM or any(ent.get(x,False) is not False for x in ['get-task-allow','com.apple.security.get-task-allow']):
        raise ValueError('Profile does not authorize the reviewed nondebuggable app identity')
    if ent.get('keychain-access-groups')!=[TEAM+'.*']:raise ValueError('Unexpected profile Keychain authority')
    certs=profile.get('DeveloperCertificates',[])
    if not isinstance(certs,list) or not 1<=len(certs)<=32 or any(not isinstance(x,bytes) or not 1<=len(x)<=65536 for x in certs) or identity not in {hashlib.sha1(x).hexdigest().upper() for x in certs}:
        raise ValueError('Profile does not authorize the selected certificate')
    return {**APP_ENTITLEMENTS,'com.apple.application-identifier':TEAM+'.'+bundle,'com.apple.developer.team-identifier':TEAM}


def symbols_match(binary,dsym):
    pattern=r'^UUID: ([A-Fa-f0-9-]{36}) \((arm64|x86_64)\) .+$'
    def parse(text):
        lines=text.strip().splitlines();result=[re.fullmatch(pattern,line) for line in lines]
        if not result or any(x is None for x in result):return None
        return sorted((x[1].upper(),x[2]) for x in result)
    return parse(binary) is not None and parse(binary)==parse(dsym)


def accepted_notary(result,submission):
    return isinstance(result,dict) and result.get('id')==submission and result.get('status')=='Accepted'


def signing_commands(app,identity,entitlements):
    helpers=[(app/'Contents/XPCServices/NetworkService.xpc',ROOT/'packaging/NetworkService.entitlements')]
    database=database_broker_package.signing_target(app)
    if database[0].exists():helpers.append(database)
    mcp=app/'Contents/XPCServices/MCPService.xpc'
    if mcp.exists():helpers.append((mcp,mcp_broker_package.ENTITLEMENTS))
    return [['/usr/bin/codesign','--force','--sign',identity,'--timestamp','--options','runtime','--entitlements',str(ent),str(target)]
      for target,ent in helpers+[(app,entitlements)]]


def validate_receipt(value):
    if not isinstance(value,dict) or type(value.get('schemaVersion')) is not int or value['schemaVersion']!=1 or value.get('phase') not in PHASES:raise ValueError('Invalid release receipt')
    c=value.get('coordinates',{})
    if not isinstance(c,dict) or type(c.get('build')) is not int or c!=coordinates(c.get('version'),str(c.get('build')),c.get('sourceCommit'),c.get('runtimeManifestSHA256')):raise ValueError('Invalid release coordinates')
    if not re.fullmatch('[A-F0-9]{40}',value.get('identity','')) or not re.fullmatch('[0-9a-f]{64}',value.get('profileSHA256','')):raise ValueError('Invalid release authority')
    expected_ent={**APP_ENTITLEMENTS,'com.apple.application-identifier':TEAM+'.'+BUNDLE,'com.apple.developer.team-identifier':TEAM}
    if value.get('appEntitlements')!=expected_ent:raise ValueError('Receipt cannot expand parent entitlements')
    return value


def run(argv,*,env=None,timeout=120):
    return subprocess.run([str(x) for x in argv],cwd=ROOT,env=env,check=True,capture_output=True,text=True,timeout=timeout).stdout.strip()


def save(path,value):
    # Record transitions once, not each queue poll. Historical receipts are not
    # assigned invented timestamps for phases that predate this instrumentation.
    if path.name == 'release.json' and value.get('phase') in PHASES:
        events = value.setdefault('phaseEvents', [])
        if not events or events[-1]['phase'] != value['phase']:
            events.append({'phase': value['phase'], 'observedAt': datetime.now(timezone.utc).isoformat()})
    raw=json.dumps(value,indent=2,sort_keys=True)+'\n'
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('x') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def require_clean(commit):
    if run(['git','status','--porcelain']) or run(['git','rev-parse','HEAD'])!=commit:raise ValueError('Release requires a clean exact source commit')


def real_tree(path):
    if path.is_symlink() or not path.is_dir():raise ValueError('Input must be a real directory')
    for child in path.rglob('*'):
        if child.is_symlink() and (not child.exists() or not child.resolve().is_relative_to(path.resolve())):raise ValueError('Escaping/broken input link')


def output_path(path):
    path=Path(os.path.abspath(path));allowed=ROOT/'.local/direct-releases'
    if not path.is_relative_to(allowed) or path==allowed:raise ValueError('Output must be a named local direct-releases directory')
    for parent in [path,*path.parents]:
        if parent.is_symlink():raise ValueError('Output path cannot traverse a symlink')
    return path


def audit_runtime(runtime,identity,report,developer_dir=None):
    environment = dict(os.environ, DEVELOPER_DIR=developer_dir) if developer_dir else None
    run([sys.executable,ROOT/'scripts/runtime_package.py','--verify-production-worker',runtime,'--channel','direct','--sign-identity',identity,'--team-id',TEAM],env=environment)
    run([sys.executable,ROOT/'scripts/runtime_audit.py',runtime,'--channel','direct','--require-identity',identity,'--require-team',TEAM,'--require-signatures','--report',report],timeout=300)
    run([sys.executable,ROOT/'scripts/runtime_source_disclosure.py','--runtime-root',runtime,'--verify-existing'],timeout=180)


def audit_app(app,receipt,report):
    real_tree(app)
    c=receipt['coordinates'];info=plistlib.loads(bounded_file(app/'Contents/Info.plist',65536))
    if receipt['requirementsSHA256']!=digest(ROOT/'requirements-runner.txt'):raise ValueError('Release requirements differ from current source')
    expected={'CFBundleIdentifier':BUNDLE,'CFBundleShortVersionString':c['version'],'CFBundleVersion':str(c['build']),
      'LSMinimumSystemVersion':'14.0','KwipMLSourceCommit':c['sourceCommit']}
    if any(info.get(k)!=v for k,v in expected.items()):raise ValueError('Application identity differs from release receipt')
    if bool(info.get('KwipMLDiagnosticOnly')) != bool(receipt.get('diagnosticOnly')):raise ValueError('Diagnostic purpose differs from signed application')
    if 'sparkleDependency' not in receipt:raise ValueError('Direct release requires the reviewed isolated updater')
    sparkle_package.audit(app,receipt['sparkleDependency'],receipt['identity'],TEAM)
    sparkle_package.audit_app_linkage(app/'Contents/MacOS/KwipML')
    profile=app/'Contents/embedded.provisionprofile'
    if digest(profile)!=receipt['profileSHA256']:raise ValueError('Embedded profile changed')
    runtime=database_broker_package.locate_runtime(app)
    if digest(runtime/'runtime-manifest.json')!=c['runtimeManifestSHA256']:raise ValueError('Runtime manifest changed')
    manifest=json.loads(bounded_file(runtime/'runtime-manifest.json',65536))
    if manifest.get('requirementsHash')!=receipt['requirementsSHA256']:raise ValueError('Runtime dependencies differ from the recorded source requirements')
    helper,helper_ent=database_broker_package.signing_target(app)
    if receipt.get('databaseSourceBroker') and not helper.is_dir():raise ValueError('Recorded database source broker is missing')
    authorities=[(app,receipt['appEntitlements']), (app/'Contents/XPCServices/NetworkService.xpc',plistlib.loads((ROOT/'packaging/NetworkService.entitlements').read_bytes()))]
    if helper.exists():authorities.append((helper,plistlib.loads(helper_ent.read_bytes())))
    mcp=app/'Contents/XPCServices/MCPService.xpc'
    if receipt.get('mcpSourceBroker') and not mcp.is_dir():raise ValueError('Recorded MCP listener is missing')
    if mcp.exists():
        mcp_info=plistlib.loads(bounded_file(mcp/'Contents/Info.plist',65536))
        if mcp_info.get('CFBundleIdentifier')!=BUNDLE+'.MCPService':raise ValueError('MCP helper identity differs')
        authorities.append((mcp,plistlib.loads(mcp_broker_package.ENTITLEMENTS.read_bytes())))
    for target,ent in authorities:
        run(['/usr/bin/codesign','--verify','--deep','--strict',target])
        runtime_audit.verify_distribution_signature(target,receipt['identity'],TEAM,channel='direct')
        actual=plistlib.loads(subprocess.run(['/usr/bin/codesign','--display','--entitlements',':-',str(target)],check=True,capture_output=True,timeout=30).stdout)
        if actual!=ent:raise ValueError('Delivered app/helper entitlements differ')
    binaries=[app/'Contents/MacOS/KwipML',app/'Contents/XPCServices/NetworkService.xpc/Contents/MacOS/NetworkService']
    if helper.exists():binaries.append(helper/'Contents/MacOS/DatabaseSourceService')
    if mcp.exists():binaries.append(mcp/'Contents/MacOS/MCPService')
    for binary in binaries:
        if run(['/usr/bin/lipo','-archs',binary])!='arm64':raise ValueError('Release architecture differs from Apple Silicon scope')
    if (app/'Contents/Resources/ProjectRoot.txt').exists():raise ValueError('Development path cannot ship')
    run([sys.executable,ROOT/'scripts/app_store_api_audit.py',app],timeout=180)
    audit_runtime(runtime,receipt['identity'],report,receipt.get('toolchainContexts',{}).get('runtimeVerifierDeveloperDir'))
    run([sys.executable,ROOT/'scripts/vendor_privacy.py','--runtime-root',runtime,'--destination',app/'Contents/Resources/VendorPrivacy','--verify-existing'])


def assemble(args,out):
    if os.environ.get("KWIPML_COMMERCE_STAGING") == "1":
        raise ValueError("Production assembly refuses staging build configuration")
    c=coordinates(args.version,args.build,args.commit,args.runtime_manifest_sha256);require_clean(c['sourceCommit'])
    updater=sparkle_package.preflight()
    runtime=Path(args.runtime);real_tree(runtime)
    if digest(runtime/'runtime-manifest.json')!=c['runtimeManifestSHA256']:raise ValueError('Supplied runtime manifest does not match pin')
    profile,raw=decode_profile(args.profile)
    inventory=parse_identities(run(['/usr/bin/security','find-identity','-v']))
    ent=validate_profile(profile,args.identity,inventory)
    if out.exists():raise FileExistsError('Release destination already exists; preserve it and choose a new build/output')
    if shutil.disk_usage(ROOT).free<10*2**30:raise ValueError('At least10GiB free space required')
    out.mkdir(parents=True);os.chmod(out,0o700)
    run([sys.executable,ROOT/'scripts/sparkle_startup_probe.py','--identity',args.identity,
         '--output',out/'updater-startup','--developer-dir',args.developer_dir],timeout=300)
    startup=json.loads(bounded_file(out/'updater-startup/receipt.json',65536))
    if startup.get('passed') is not True or startup.get('negativeControl') is not False or startup.get('dependency')!=updater:
        raise ValueError('Actual updater startup prerequisite failed')
    runtime_developer_dir = getattr(args,'runtime_developer_dir',None) or os.environ.get('DEVELOPER_DIR') or run(['/usr/bin/xcode-select','-p'])
    audit_runtime(runtime,args.identity,out/'input-runtime-audit.json',runtime_developer_dir)
    env={**os.environ,'DEVELOPER_DIR':args.developer_dir,'KWIPML_DIRECT_UPDATES':'1'}
    toolchain={'swift':run(['/usr/bin/xcrun','swift','--version'],env=env),'xcode':run(['/usr/bin/xcodebuild','-version'],env=env),
      'sdkVersion':run(['/usr/bin/xcrun','--sdk','macosx','--show-sdk-version'],env=env),
      'sdkBuild':run(['/usr/bin/xcrun','--sdk','macosx','--show-sdk-build-version'],env=env),
      'macOS':run(['/usr/bin/sw_vers','-productVersion']),'architecture':os.uname().machine}
    if toolchain['architecture']!='arm64':raise ValueError('Apple Silicon build host required')
    command=['/usr/bin/xcrun','swift','build','--configuration','release','--disable-experimental-strip-products','-debug-info-format','dwarf','--scratch-path',str(out/'build')]
    for product in ['KwipML','KwipMLNetworkService','KwipMLDatabaseSourceService','KwipMLMCPService']:
        result=subprocess.run(command+['--product',product],cwd=ROOT,env=env,capture_output=True,text=True,timeout=1800)
        (out/(product+'-build.log')).write_text(result.stdout+result.stderr)
        result.check_returncode()
    products=Path(run(command+['--show-bin-path'],env=env))
    app=out/'Kwip ML.app';macos=app/'Contents/MacOS';resources=app/'Contents/Resources';macos.mkdir(parents=True);resources.mkdir()
    shutil.copy2(products/'KwipML',macos/'KwipML')
    sparkle_package.prepare_app_linkage(macos/'KwipML',products,args.developer_dir)
    for name in ['KwipML_KwipMLCore.bundle','KwipML_KwipMLApp.bundle']:shutil.copytree(products/name,resources/name)
    bundled_runtime=database_broker_package.runtime_destination(app)
    bundled_runtime.parent.mkdir(parents=True,exist_ok=True)
    run(['/usr/bin/ditto',runtime,bundled_runtime],timeout=300)
    run([sys.executable,ROOT/'scripts/app_icon.py','--source',ROOT/'Sources/KwipMLApp/Resources/KwipMLWelcomeIcon.png','--output',resources/'KwipML.icns'])
    xpc=app/'Contents/XPCServices/NetworkService.xpc';(xpc/'Contents/MacOS').mkdir(parents=True)
    shutil.copy2(products/'KwipMLNetworkService',xpc/'Contents/MacOS/NetworkService')
    info={'CFBundleIdentifier':BUNDLE,'CFBundleName':product_brand.NAME,'CFBundleDisplayName':product_brand.NAME,'CFBundleExecutable':'KwipML',
      'CFBundlePackageType':'APPL','CFBundleShortVersionString':c['version'],'CFBundleVersion':str(c['build']),
      'CFBundleIconFile':'KwipML.icns','LSMinimumSystemVersion':'14.0','NSHighResolutionCapable':True,'CFBundleSupportedPlatforms':['MacOSX'],
      'LSApplicationCategoryType':'public.app-category.developer-tools','NSHumanReadableCopyright':'Copyright © 2026 KWIP LLC',
      'DTPlatformName':'macosx','DTSDKName':'macosx'+toolchain['sdkVersion'],'DTSDKBuild':toolchain['sdkBuild'],
      'CFBundleURLTypes':[{'CFBundleURLName':BUNDLE+'.auth','CFBundleURLSchemes':['kwipml']}],'KwipMLSourceCommit':c['sourceCommit'],
      **sparkle_package.info_keys()}
    if getattr(args,'diagnostic_only',False):
        info['KwipMLDiagnosticOnly'] = True
        info['CFBundleDisplayName'] = product_brand.NAME+' Verification'
    (app/'Contents/Info.plist').write_bytes(plistlib.dumps(info))
    (xpc/'Contents/Info.plist').write_bytes(plistlib.dumps({'CFBundleIdentifier':BUNDLE+'.NetworkService','CFBundleExecutable':'NetworkService','CFBundleName':product_brand.NAME+' Network Service','CFBundleDisplayName':product_brand.NAME+' Network Service',
      'CFBundlePackageType':'XPC!','CFBundleShortVersionString':'1.0','CFBundleVersion':str(c['build']),'LSMinimumSystemVersion':'14.0',
      'XPCService':{'RunLoopType':'NSRunLoop','ServiceType':'Application'}}))
    database_xpc=database_broker_package.embed(app,products)
    mcp_xpc=mcp_broker_package.embed(app,products)
    (app/'Contents/embedded.provisionprofile').write_bytes(raw)
    ent_path=out/'App.entitlements';ent_path.write_bytes(plistlib.dumps(ent))
    framework=sparkle_package.embed(app,updater)
    downloader_ent=out/'Downloader.entitlements';downloader_ent.write_bytes(plistlib.dumps(sparkle_package.DOWNLOADER_ENTITLEMENTS))
    run([sys.executable,ROOT/'scripts/vendor_privacy.py','--runtime-root',bundled_runtime,'--destination',resources/'VendorPrivacy'])
    normalize_package_access(app)
    for command in sparkle_package.signing_commands(framework,args.identity,downloader_ent):run(command,timeout=180)
    for command in signing_commands(app,args.identity,ent_path):run(command,timeout=180)
    symbols=out/'symbols';symbols.mkdir();symbol_records=[]
    for name,binary in [('KwipML',macos/'KwipML'),('KwipMLNetworkService',xpc/'Contents/MacOS/NetworkService'),('KwipMLDatabaseSourceService',database_xpc/'Contents/MacOS/DatabaseSourceService'),('KwipMLMCPService',mcp_xpc/'Contents/MacOS/MCPService')]:
        source=products/(name+'.dSYM')
        if not source.is_dir():run(['/usr/bin/dsymutil',binary,'-o',symbols/(name+'.dSYM')],timeout=180)
        else:shutil.copytree(source,symbols/source.name)
        native_uuid=run(['/usr/bin/dwarfdump','--uuid',binary]);symbol_uuid=run(['/usr/bin/dwarfdump','--uuid',symbols/(name+'.dSYM')])
        if not symbols_match(native_uuid,symbol_uuid):raise ValueError('Missing/mismatched exact native symbols')
        statistics=json.loads(run(['/usr/bin/dwarfdump','--statistics',symbols/(name+'.dSYM')],timeout=120))
        if statistics.get('#functions',0)<1 or statistics.get('#functions with location',0)<1:raise ValueError('Native symbol file contains no usable function debug information')
        symbol_records.append({'product':name,'uuid':re.findall(r'UUID: ([A-Fa-f0-9-]{36}) \(([^)]+)\)',native_uuid)})
    receipt={'schemaVersion':1,'coordinates':c,'phase':'assembled','identity':args.identity,'profileSHA256':hashlib.sha256(raw).hexdigest(),
      'toolchainContexts':{'buildDeveloperDir':args.developer_dir,'runtimeVerifierDeveloperDir':runtime_developer_dir,'notaryDeveloperDir':getattr(args,'notary_developer_dir',None) or args.developer_dir},
      'diagnosticOnly':getattr(args,'diagnostic_only',False),'packagedBoundaryRequired':True,'appEntitlements':ent,'databaseSourceBroker':True,'mcpSourceBroker':True,'toolchain':toolchain,'symbols':symbol_records,'thirdPartySymbols':'Not fabricated; upstream Python/native library dSYMs may be unavailable.',
      'requirementsSHA256':digest(ROOT/'requirements-runner.txt'),'createdAt':datetime.now(timezone.utc).isoformat(),
      'runtimeInput':'Explicit pinned previously produced runtime, independently reverified; no claim of fresh dependency rebuild.',
      'sparkleDependency':updater,'updaterStartupProbeSHA256':digest(out/'updater-startup/receipt.json'),'publicLaunch':False}
    audit_app(app,receipt,out/'assembled-runtime-audit.json');require_clean(c['sourceCommit'])
    run(['/usr/bin/ditto','-c','-k','--keepParent',app,out/'app-notary.zip'],timeout=300)
    receipt['licenseInventorySHA256']=digest(bundled_runtime/'license-inventory.json')
    receipt['sourceIndexSHA256']=digest(bundled_runtime/'source-disclosure/SOURCE_INDEX.json')
    receipt['assembledRuntimeAuditSHA256']=digest(out/'assembled-runtime-audit.json')
    receipt['appZipSHA256']=digest(out/'app-notary.zip')
    receipt['assembledAppTreeSHA256']=packaged_boundary.tree_digest(app)
    save(out/'release.json',receipt)
    return receipt


def notary_environment(args, receipt):
    recorded = receipt.get('toolchainContexts',{}).get('notaryDeveloperDir')
    requested = getattr(args,'notary_developer_dir',None)
    if recorded and requested and recorded != requested:
        raise ValueError('Notary toolchain differs from recorded candidate; reconcile explicitly')
    selected = recorded or requested
    return dict(os.environ, DEVELOPER_DIR=selected) if selected else None


def notary(args,out,receipt):
    signed_info = out / 'Kwip ML.app/Contents/Info.plist'
    signed_diagnostic = signed_info.is_file() and plistlib.loads(bounded_file(signed_info, 1024*1024)).get('KwipMLDiagnosticOnly') is True
    if receipt.get('diagnosticOnly') is True or signed_diagnostic:
        raise ValueError('Diagnostic-only signed smoke apps cannot be submitted or released')
    environment = notary_environment(args, receipt)
    phase=receipt['phase'];app=out/'Kwip ML.app';dmg=out/'KwipML.dmg';file=out/'release.json'
    if phase=='verified':return receipt
    if phase=='assembled':
        gate = packaged_boundary.require_gate(out, receipt)
        if gate:
            receipt['packagedBoundarySHA256'] = gate
            save(file, receipt)
    if phase in {'assembled','dmgBuilt'}:
        kind='app' if phase=='assembled' else 'dmg';artifact=out/'app-notary.zip' if kind=='app' else dmg
        expected=receipt['appZipSHA256'] if kind=='app' else receipt['dmgPreNotarySHA256']
        if digest(artifact)!=expected:raise ValueError('Notary payload changed')
        response_file=out/(kind+'-submit.json')
        if receipt.get('pendingUpload') and not response_file.exists():raise ValueError('Uncertain prior upload: reconcile Apple history before another submission')
        if not response_file.exists():
            receipt['pendingUpload']=kind;save(file,receipt)
            with response_file.open('x') as output:
                result=subprocess.run(['/usr/bin/xcrun','notarytool','submit',str(artifact),'--keychain-profile',args.keychain_profile,'--output-format','json'],cwd=ROOT,stdout=output,stderr=subprocess.PIPE,text=True,timeout=900,env=environment)
            if result.returncode:raise ValueError('Notary upload failed; inspect local submission receipt and credentials, do not repeat blindly')
        result=json.loads(bounded_file(response_file,65536));submission=result.get('id')
        if not isinstance(submission,str) or str(uuid.UUID(submission))!=submission:raise ValueError('Notary upload has no canonical submission ID; reconcile Apple history')
        receipt[kind+'SubmissionID']=submission;receipt['phase']=kind+'Submitted';receipt.pop('pendingUpload',None);save(file,receipt)
        return receipt
    if phase in {'appSubmitted','dmgSubmitted'}:
        kind='app' if phase=='appSubmitted' else 'dmg';submission=receipt[kind+'SubmissionID']
        result=json.loads(run(['/usr/bin/xcrun','notarytool','info',submission,'--keychain-profile',args.keychain_profile,'--output-format','json'],timeout=60,env=environment))
        save(out/(kind+'-notary-info.json'),result)
        if result.get('id')!=submission:raise ValueError('Notary status identifies a different submission')
        if not accepted_notary(result,submission):
            if result.get('status')=='In Progress':return receipt
            raise ValueError('Notary submission not Accepted; inspect Apple log before changing the artifact')
        run(['/usr/bin/xcrun','notarytool','log',submission,'--keychain-profile',args.keychain_profile,out/(kind+'-notary-log.json')],timeout=60,env=environment)
        log=json.loads(bounded_file(out/(kind+'-notary-log.json'),2*1024*1024))
        if log.get('jobId')!=submission or log.get('status')!='Accepted':raise ValueError('Notary log does not match the accepted submission')
        receipt[kind+'NotaryIssues']=log.get('issues',[])
        require_clean(receipt['coordinates']['sourceCommit'])
        run(['/usr/bin/xcrun','stapler','staple',app if kind=='app' else dmg],timeout=120)
        run(['/usr/bin/xcrun','stapler','validate',app if kind=='app' else dmg],timeout=60)
        receipt['phase']='appAccepted' if kind=='app' else 'verified'
        if kind=='dmg':
            # Mark verified only after the independently extracted round-trip checks succeed.
            receipt['phase']='dmgSubmitted';final_verify(out,receipt);receipt['phase']='verified'
            receipt['dmgSHA256']=digest(dmg);receipt['dmgBytes']=dmg.stat().st_size
        save(file,receipt);return receipt
    if phase=='appAccepted':
        require_clean(receipt['coordinates']['sourceCommit']);audit_app(app,receipt,out/'stapled-runtime-audit.json')
        payload=out/'disk-image';payload.mkdir();run(['/usr/bin/ditto',app,payload/'Kwip ML.app'],timeout=300)
        (payload/'Applications').symlink_to('/Applications',target_is_directory=True)
        run(['/usr/bin/hdiutil','create','-srcfolder',payload,'-volname',product_brand.NAME,'-format','UDZO',dmg],timeout=600)
        run(['/usr/bin/codesign','--sign',receipt['identity'],'--timestamp',dmg],timeout=120)
        run(['/usr/bin/codesign','--verify','--strict',dmg])
        receipt['dmgPreNotarySHA256']=digest(dmg);receipt['phase']='dmgBuilt';save(file,receipt);return receipt
    raise ValueError('Unexpected phase')


def final_verify(out,receipt):
    dmg=out/'KwipML.dmg'
    run(['/usr/bin/hdiutil','verify',dmg],timeout=180)
    run(['/usr/sbin/spctl','--assess','--type','open','--context','context:primary-signature','--verbose=2',dmg])
    with tempfile.TemporaryDirectory(prefix='kwipml-release-mount-') as directory:
        mount=Path(directory)/'mounted';mount.mkdir()
        run(['/usr/bin/hdiutil','attach','-readonly','-nobrowse','-mountpoint',mount,dmg],timeout=120)
        try:
            extracted=Path(directory)/'extracted.app';run(['/usr/bin/ditto',mount/'Kwip ML.app',extracted],timeout=300)
            audit_app(extracted,receipt,out/'extracted-runtime-audit.json')
            run(['/usr/bin/xcrun','stapler','validate',extracted]);run(['/usr/sbin/spctl','--assess','--type','execute','--verbose=2',extracted])
        finally:run(['/usr/bin/hdiutil','detach',mount],timeout=60)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step',choices=['plan','assemble','notary','verify'],default='plan')
    parser.add_argument('--diagnostic-only',action='store_true',help='Signed packaged development smoke; permanently ineligible for notarization/promotion')
    parser.add_argument('--output',required=True);parser.add_argument('--runtime');parser.add_argument('--runtime-manifest-sha256')
    parser.add_argument('--version',default='1.0.0');parser.add_argument('--build');parser.add_argument('--commit');parser.add_argument('--profile')
    parser.add_argument('--identity');parser.add_argument('--developer-dir',default='/Applications/Xcode.app/Contents/Developer')
    parser.add_argument('--keychain-profile',default='kwipml-direct')
    parser.add_argument('--runtime-developer-dir',help='Compiler used to independently reconstruct the sealed watchdog')
    parser.add_argument('--notary-developer-dir',default='/Applications/Xcode.app/Contents/Developer')
    args=parser.parse_args()
    try:
        out=output_path(args.output)
        if args.step in {'plan','assemble'}:
            if not all([args.runtime,args.runtime_manifest_sha256,args.build,args.commit,args.profile,args.identity]):raise ValueError('Assembly requires exact source/build/runtime/profile/identity inputs')
            if args.step=='plan':
                result={'coordinates':coordinates(args.version,args.build,args.commit,args.runtime_manifest_sha256),'output':str(out),'signed':False,'published':False}
            else:
                with compute_lock(lock_path(ROOT), 'assembly'):
                    result=measured(out.with_name(out.name+'.operations.jsonl'), 'compute', 'assemble-sign-audit', lambda: assemble(args,out))
        else:
            receipt=validate_receipt(json.loads(bounded_file(out/'release.json',65536)))
            if args.step=='notary':
                phase = receipt['phase']
                category = 'provider' if phase in {'assembled','appSubmitted','dmgBuilt','dmgSubmitted'} else 'compute'
                result=measured(out.with_name(out.name+'.operations.jsonl'), category, 'notary-'+phase, lambda: notary(args,out,receipt))
            else:
                if receipt['phase']!='verified' or digest(out/'KwipML.dmg')!=receipt.get('dmgSHA256'):raise ValueError('Final accepted/stapled digest required')
                require_clean(receipt['coordinates']['sourceCommit']);final_verify(out,receipt);result=receipt
        print(json.dumps({'phase':result.get('phase','plan'),'coordinates':result['coordinates'],'publicLaunch':False},indent=2));return 0
    except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError) as error:
        print('Direct release stopped: '+str(error),file=sys.stderr);return 1


if __name__=='__main__':raise SystemExit(main())
