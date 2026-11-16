#!/usr/bin/env python3
"""Explicit offline model gate; never converts skipped integration tests into acceptance."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET


def source_identity(root):
    """Fingerprint Git source only; never walk ignored model/build artifacts."""
    def git(*args):
        return subprocess.check_output(['git',*args],cwd=root)
    commit = git('rev-parse','HEAD').decode().strip()
    diff = git('diff','--binary','HEAD','--')
    names = git('ls-files','--others','--exclude-standard','-z','--',
                'Sources','Tests','runner','scripts','Package.swift','Package.resolved',
                'requirements*.txt').decode().split('\0')
    untracked = {}; total = 0
    for name in sorted(filter(None,names)):
        path = root/name
        if path.is_symlink() or not path.is_file():
            raise ValueError('Qualification source must be a regular file')
        size = path.stat().st_size; total += size
        if size > 32_000_000 or total > 256_000_000:
            raise ValueError('Qualification source inventory exceeds bounds')
        untracked[name] = {'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                           'mode':path.stat().st_mode & 0o777}
    return {'commit':commit,'diffSHA256':hashlib.sha256(diff).hexdigest(),
            'untrackedSource':untracked}


def require_unchanged_source(root, expected):
    if source_identity(root) != expected:
        raise ValueError('Source changed during qualification')


def verify_fixtures(spec, manifest):
    result = {}
    for name, contract in spec.items():
        if name not in manifest:
            raise ValueError(f'Missing fixture {name}')
        entry = manifest[name]; root = Path(entry['path']); pins = entry['files']
        if not set(contract.get('requiredFiles', [])) <= set(pins) or not pins:
            raise ValueError(f'{name}: required files missing')
        if any(pins.get(k) != v for k,v in contract.get('qualifiedPins', {}).items()):
            raise ValueError(f'{name}: fixture differs from reviewed qualification pins')
        if root.is_symlink() or not root.exists():
            raise ValueError(f'{name}: regular fixture required')
        files = {}
        for relative, expected in pins.items():
            part = Path(relative)
            if part.is_absolute() or '..' in part.parts:
                raise ValueError('Unsafe fixture relative path')
            path = root / part if contract['kind'] == 'directory' else root
            if path.is_symlink() or not path.is_file() or any(p.is_symlink() for p in path.parents):
                raise ValueError(f'{name}: regular file required')
            digest = hashlib.sha256(); size = 0
            with path.open('rb') as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b''):
                    size += len(chunk); digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(f'{name}/{relative}: hash pin mismatch')
            files[relative] = {'sha256': expected, 'bytes': size}
        if contract['kind'] == 'directory':
            actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
            if actual != set(pins):
                raise ValueError(f'{name}: unpinned fixture files')
        result[name] = {'path': str(root.resolve()), 'files': files}
    return result


def verify_python_result(path, expected):
    cases = list(ET.parse(path).iter('testcase'))
    if any(list(case) and any(c.tag in ('skipped','failure','error') for c in case) for case in cases):
        raise ValueError('Python qualification skipped or failed')
    if Counter(c.attrib['name'] for c in cases) != Counter(expected):
        raise ValueError('Python qualification cases missing or duplicated')
    return len(cases)


def verify_swift_result(log, expected):
    for selector in expected:
        cls, method = selector.split('/')
        expression = rf"Test Case '[^\n']*{re.escape(cls)}[ .]{re.escape(method)}\]?' passed \("
        if len(re.findall(expression, log)) != 1:
            raise ValueError(f'Swift qualification missing unique PASS: {selector}')
    return len(expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, required=True, help='Private JSON paths and SHA256 pins, see inventory')
    parser.add_argument('--group', action='append')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    args.output = args.output.resolve()
    root = Path(__file__).resolve().parents[1]
    inventory = json.loads((root/'scripts/required_model_qualification.json').read_text())
    groups = args.group or list(inventory)
    if len(set(groups)) != len(groups) or set(groups)-set(inventory):
        parser.error('Unknown or duplicate group')
    spec = {k:v for group in groups for k,v in inventory[group]['fixtures'].items()}
    manifest = json.loads(args.fixtures.read_text())
    fixtures = verify_fixtures(spec, manifest)
    if args.preflight:
        print(json.dumps({'scope':'fixture-preflight-only','fixtures':fixtures},indent=2)); return
    args.output.mkdir(parents=True, exist_ok=False)
    source = source_identity(root)
    commit = source['commit']
    env = os.environ.copy()
    env.update({k:v['path'] for k,v in fixtures.items()})
    env.update(KWIPML_TEST_DEVELOPMENT_ROOT=str(root), KWIPML_SKLEARN_TEST_PYTHON=str(root/'.venv/bin/python'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    results = []
    for group in groups:
        entry = inventory[group]
        for lane in ('python','swift'):
            selected = entry.get(lane, [])
            if not selected: continue
            log = args.output/f'{group}-{lane}.log'
            junit = args.output/f'{group}.xml'
            command = ([str(root/'.venv/bin/python'),'-m','pytest',*selected,'--junitxml',str(junit),'-q'] if lane=='python' else ['swift','test','--filter','|'.join(re.escape(s) for s in selected)])
            with log.open('w') as stream:
                subprocess.run(command,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=1800,check=True)
            count = verify_python_result(junit,entry['pythonCases']) if lane=='python' else verify_swift_result(log.read_text(),selected)
            results.append({'group':group,'lane':lane,'cases':count,'command':command,'log':str(log)})
    require_unchanged_source(root, source)
    if fixtures != verify_fixtures(spec,manifest):
        raise ValueError('Qualification inputs changed')
    receipt = {'scope':'required-model-qualification','allRequiredGroups':set(groups)==set(inventory),'commit':commit,'sourceIdentity':source,'workingTreeDirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()),'fixtures':fixtures,'executions':results,'nativeUAT':False,'releaseAcceptance':False}
    (args.output/'PASSED.json').write_text(json.dumps(receipt,indent=2)+'\n')

if __name__ == '__main__':
    main()
