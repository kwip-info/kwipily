#!/usr/bin/env python3
"""Inventory exact upstream standalone archive notices without extracting archive paths."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess


def notice_paths(metadata):
    paths={metadata['license_path']}
    for variants in metadata['build_info']['extensions'].values():
        for variant in variants:
            paths.update(variant.get('license_paths',[]))
    for path in paths:
        if not path.startswith('licenses/') or '..' in PurePosixPath(path).parts:
            raise ValueError('Unsafe upstream license path')
    return sorted(paths)


def inventory(archive,expected_sha256,output):
    archive=Path(archive);output=Path(output)
    actual=hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()
    if actual!=expected_sha256: raise ValueError('Archive checksum differs from expected upstream digest')
    if output.exists() and any(output.iterdir()): raise ValueError('Output must be empty')
    listing=set(subprocess.check_output(['/usr/bin/tar','-tf',str(archive)],text=True).splitlines())
    metadata=json.loads(subprocess.check_output(['/usr/bin/tar','-xOf',str(archive),'python/PYTHON.json']))
    output.mkdir(parents=True,exist_ok=True)
    (output/'PYTHON.json').write_text(json.dumps(metadata,indent=2))
    notices=[];missing=[]
    for path in notice_paths(metadata):
        if 'python/'+path not in listing:
            missing.append(path);continue
        content=subprocess.check_output(['/usr/bin/tar','-xOf',str(archive),'python/'+path])
        target=output/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
        notices.append({'path':path,'sha256':hashlib.sha256(content).hexdigest()})
    extensions={name:[{key:value for key,value in variant.items() if key in {'licenses','license_paths','license_public_domain','links'}} for variant in variants] for name,variants in metadata['build_info']['extensions'].items() if any(v.get('license_paths') for v in variants)}
    result={'schemaVersion':1,'archiveSHA256':actual,'pythonVersion':metadata['python_version'],'targetTriple':metadata['target_triple'],'pythonLicenses':metadata['licenses'],'extensions':extensions,'notices':notices,'missingReferencedNotices':missing,'redistributionReviewComplete':False}
    (output/'inventory.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('archive');parser.add_argument('--sha256',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(inventory(args.archive,args.sha256,args.output),indent=2))
