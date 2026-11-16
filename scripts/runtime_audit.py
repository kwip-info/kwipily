#!/usr/bin/env python3
"""Inspect runtime Mach-O files; sign ad-hoc or with an explicit MAS/direct identity."""
import argparse
import json
import hashlib
import re
import tempfile
import plistlib
from pathlib import Path
import subprocess

MAGICS={b'\xcf\xfa\xed\xfe',b'\xce\xfa\xed\xfe',b'\xfe\xed\xfa\xcf',b'\xfe\xed\xfa\xce',b'\xca\xfe\xba\xbe',b'\xca\xfe\xba\xbf'}

def validate_distribution_selection(identity, team_id, channel="mas"):
    if channel not in {"mas", "direct"}: raise ValueError("Unknown distribution channel")
    prefixes = ("Developer ID Application: ",) if channel == "direct" else ("Apple Distribution: ", "3rd Party Mac Developer Application: ")
    if not isinstance(team_id, str) or not re.fullmatch(r'[A-Z0-9]{10}', team_id):
        raise ValueError('Explicit distribution team must contain ten uppercase letters/digits')
    if not isinstance(identity, str) or not (re.fullmatch(r'[A-Fa-f0-9]{40}', identity) or
            (identity.startswith(prefixes) and identity.endswith(' (' + team_id + ')'))):
        raise ValueError('Explicit '+channel+' application distribution identity and matching team are required')


def verify_distribution_signature(path, identity, team_id, channel="mas"):
    validate_distribution_selection(identity, team_id, channel)
    subprocess.run(['/usr/bin/codesign', '--verify', '--strict', str(path)], check=True, capture_output=True, text=True)
    result = subprocess.run(['/usr/bin/codesign', '--display', '--verbose=4', str(path)], check=True, capture_output=True, text=True)
    fields = (result.stdout + '\n' + result.stderr).splitlines()
    authorities = [line.removeprefix('Authority=') for line in fields if line.startswith('Authority=')]
    teams = [line.removeprefix('TeamIdentifier=') for line in fields if line.startswith('TeamIdentifier=')]
    if teams != [team_id]: raise ValueError('Native signature team differs from selected team')
    prefixes = ('Developer ID Application: ',) if channel == 'direct' else ('Apple Distribution: ', '3rd Party Mac Developer Application: ')
    if not authorities or not authorities[0].startswith(prefixes):
        raise ValueError('Native signature identity is not a '+channel+' distribution identity')
    if channel == 'direct':
        directories = [line for line in fields if line.startswith('CodeDirectory ')]
        match = re.search(r'flags=0x([0-9a-fA-F]+)', directories[0]) if len(directories) == 1 else None
        if not match or not (int(match.group(1),16) & 0x10000):
            raise ValueError('Direct native signature requires hardened runtime')
        if not any(line.startswith('Timestamp=') and line.removeprefix('Timestamp=').strip() for line in fields):
            raise ValueError('Direct native signature requires a secure timestamp')
    if re.fullmatch(r'[A-Fa-f0-9]{40}', identity):
        with tempfile.TemporaryDirectory(prefix='kwip-signature-cert-') as temporary:
            prefix = str(Path(temporary) / 'certificate')
            subprocess.run(['/usr/bin/codesign', '--display', '--extract-certificates='+prefix, str(path)], check=True, capture_output=True)
            leaf = Path(prefix + '0')
            if leaf.is_symlink() or not leaf.is_file() or leaf.stat().st_size > 65536:
                raise ValueError('Native signature certificate is absent or invalid')
            fingerprint = hashlib.sha1(leaf.read_bytes()).hexdigest().upper()
            if fingerprint != identity.upper(): raise ValueError('Native signature identity fingerprint differs from selected identity')
    elif authorities[0] != identity:
        raise ValueError('Native signature identity differs from selected identity')
    return {'identity': identity, 'authority': authorities[0], 'teamID': team_id, 'channel':channel}


def audit(root,sign=False,require_signatures=False,signing_identity=None,team_id=None,require_identity=None,require_team=None,worker_entitlements=None,channel="mas"):
    if channel not in {"mas", "direct"}: raise ValueError("Unknown distribution channel")
    if channel == "direct" and (sign or not (signing_identity or require_identity)):
        raise ValueError("Direct channel requires an explicit Developer ID selection")
    channel_options = {"channel":channel} if channel != "mas" else {}
    if signing_identity and sign: raise ValueError("Choose ad-hoc or distribution signing, not both")
    if signing_identity or team_id: validate_distribution_selection(signing_identity,team_id,channel)
    if require_identity or require_team: validate_distribution_selection(require_identity,require_team,channel)
    if signing_identity and require_identity and (signing_identity != require_identity or team_id != require_team): raise ValueError("Signing and required identities differ")
    if worker_entitlements and not signing_identity: raise ValueError("Worker entitlements require explicit distribution signing")
    mutate = sign or signing_identity is not None
    required_identity = require_identity or signing_identity
    required_team = require_team or team_id
    root=Path(root).resolve();records=[]
    worker_executable=None
    if worker_entitlements:
        entitlement_path=Path(worker_entitlements)
        if entitlement_path.is_symlink() or not entitlement_path.is_file() or entitlement_path.stat().st_size > 65536:
            raise ValueError('Worker entitlements must be a bounded regular file')
        if plistlib.loads(entitlement_path.read_bytes()) != {'com.apple.security.app-sandbox':True,'com.apple.security.inherit':True}:
            raise ValueError('Worker entitlements must contain only sandbox and inherit')
        manifest_path=root/'runtime-manifest.json'
        if manifest_path.is_symlink() or not manifest_path.is_file() or manifest_path.stat().st_size > 65536:
            raise ValueError('Worker signing requires a bounded regular runtime manifest')
        selected=json.loads(manifest_path.read_text()).get('pythonExecutable')
        if not isinstance(selected,str) or Path(selected).is_absolute() or '..' in Path(selected).parts:
            raise ValueError('Worker executable must be a contained manifest-relative path')
        worker_executable=(root/selected).resolve(strict=True)
        if not worker_executable.is_relative_to(root) or not worker_executable.is_file():
            raise ValueError('Worker executable must resolve to a contained regular file')
    for path in sorted(root.rglob('*')):
        if path.is_symlink() or not path.is_file(): continue
        with path.open('rb') as source: magic=source.read(4)
        if magic not in MAGICS: continue
        sealed_worker=path.relative_to(root).parts[0] == 'runner'
        original_id=None
        identities=subprocess.check_output(['/usr/bin/otool','-D',str(path)],text=True).splitlines()[1:]
        if identities:
            own_id=identities[0].strip()
            if own_id.startswith('/') and not own_id.startswith(('/usr/lib/','/System/Library/')) and mutate:
                if sealed_worker:
                    raise ValueError(f'Cannot normalize sealed worker identity at {path.relative_to(root)}')
                original_id=own_id
                subprocess.run(['/usr/bin/install_name_tool','-id','@rpath/'+path.name,str(path)],check=True)
        linked=subprocess.check_output(['/usr/bin/otool','-L',str(path)],text=True)
        dependencies=[]
        for line in linked.splitlines()[1:]:
            if line.startswith('\t'): dependencies.append(line.strip().split(' (compatibility')[0])
        external=[value for value in dependencies if value.startswith('/') and not value.startswith(('/usr/lib/','/System/Library/'))]
        if external: raise ValueError(f'External runtime dependency at {path.relative_to(root)}: {external}')
        commands=subprocess.check_output(['/usr/bin/otool','-l',str(path)],text=True)
        header=subprocess.check_output(['/usr/bin/otool','-hv',str(path)],text=True)
        executable=bool(re.search(r'\bEXECUTE\b', header))
        minos=[line.strip().split()[-1] for line in commands.splitlines() if line.strip().startswith('minos ')]
        # Worker attestation seals exact bytes, including its canonical signature.
        # Dependency normalization/signing must not reseal or repair worker code.
        if mutate and not sealed_worker:
            args=['/usr/bin/codesign','--force','--sign',signing_identity or '-', *(['--timestamp','--options','runtime'] if channel == 'direct' else ['--timestamp=none'])]
            if worker_entitlements:
                if path == worker_executable or executable:
                    args += ['--entitlements', str(worker_entitlements)]
            subprocess.run(args+[str(path)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        verified=subprocess.run(['/usr/bin/codesign','--verify','--strict',str(path)],capture_output=True,text=True)
        if require_signatures and verified.returncode != 0:
            raise ValueError(f'Invalid native signature at {path.relative_to(root)}')
        distribution = verify_distribution_signature(path,required_identity,required_team,**channel_options) if required_identity else None
        if required_identity and ((executable and not mutate) or channel == "direct"):
            result=subprocess.run(['/usr/bin/codesign','--display','--entitlements',':-',str(path)],check=True,capture_output=True)
            actual = plistlib.loads(result.stdout) if result.stdout.strip() else {}
            expected = {'com.apple.security.app-sandbox':True,'com.apple.security.inherit':True} if executable else {}
            if actual != expected:
                raise ValueError(f'Native entitlements must exactly '+('inherit the app sandbox: ' if executable else 'match empty library entitlements: ')+str(path.relative_to(root)))
        records.append({'distributionSignature':distribution,'path':str(path.relative_to(root)),'dependencies':dependencies,'minimumOS':minos,'signatureVerified':verified.returncode==0,'normalizedInstallID':original_id is not None})
    if mutate and (root/'runtime-manifest.json').exists():
        manifest=json.loads((root/'runtime-manifest.json').read_text());manifest['signing']=(('MAS' if channel == 'mas' else 'Developer ID')+' distribution native identity '+signing_identity+' team '+team_id) if signing_identity else 'Ad-hoc signed native payload; not a distribution or App Store signature'
        (root/'runtime-manifest.json').write_text(json.dumps(manifest,indent=2))
    return {'schemaVersion':1,'machOFiles':records,'fileCount':len(records),'sizeBytes':sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink()),'signing':(('MAS' if channel == 'mas' else 'Developer ID')+' distribution native identity '+signing_identity+' team '+team_id) if signing_identity else ('ad-hoc, no hardened runtime or sandbox inheritance' if sign else 'inspection only')}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('runtime');parser.add_argument('--sign-ad-hoc',action='store_true');parser.add_argument('--report',required=True);parser.add_argument('--require-signatures',action='store_true')
    parser.add_argument('--channel',choices=['mas','direct'],default='mas');parser.add_argument('--sign-identity');parser.add_argument('--team-id');parser.add_argument('--require-identity');parser.add_argument('--require-team');parser.add_argument('--worker-entitlements')
    args=parser.parse_args();result=audit(args.runtime,args.sign_ad_hoc,args.require_signatures,args.sign_identity,args.team_id,args.require_identity,args.require_team,args.worker_entitlements,args.channel);Path(args.report).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='machOFiles'}))
