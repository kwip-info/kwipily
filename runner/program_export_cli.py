"""Pinned retained-stage bridge. Native supervision owns one-shot launch authority.

Inputs are trusted, app-generated trained packages. Hashes establish consistency,
not safety of third-party Python code or serialized programs.
"""
import os
from pathlib import Path
import tempfile
import types
import torch
if __package__:
    from .execution_budget import validate_versioned_request
    from .worker_authorization import canonical_uuid, directory, hash_value, object_json, read_regular, require, sha
    from .program_export_configuration import normalize_program_export
    from .program_export import build_package
else:
    from execution_budget import validate_versioned_request
    from worker_authorization import canonical_uuid, directory, hash_value, object_json, read_regular, require, sha
    from program_export_configuration import normalize_program_export
    from program_export import build_package

REQUEST_FIELDS = {'schemaVersion','kind','projectID','exportID','requestIdentity','sourceDirectory','sourceManifestSHA256','configuration'}
ALLOWED = {'model.py','checkpoint.pt','inference.py','input-contract.json','README.md','requirements.txt','training_recipe.py','rate-trace.json','objective_recipe.py','initialization_recipe.py','gradient_policy.py','trainability_recipe.py','data_loading_recipe.py'}
REQUIRED = {'model.py','checkpoint.pt','inference.py','input-contract.json'}


def execute_export(request_path, expected_sha256, emit, *, check_cancelled=lambda: None):
    path = Path(request_path)
    require(path.is_absolute() and (path.name == 'request.json' or (path.name == 'program-export.json' and path.parent.name == 'input')), 'Expected retained program request')
    hash_value(expected_sha256)
    managed = path.name == 'program-export.json'
    stage = path.parent.parent if managed else path.parent
    if managed:
        authority = getattr(check_cancelled,'__self__',None)
        if __package__: from .operation_authorization import OperationAuthority
        else: from operation_authorization import OperationAuthority
        require(isinstance(authority,OperationAuthority) and authority.request.get('toolMode') == 'programExport' and str(authority._path) == str(stage), 'Program export requires its managed operation authority')
        authority.check()
    stage_fd = directory(str(stage))
    source_fd = None
    try:
        request_fd = directory(str(path.parent))
        try: request_raw = read_regular(request_fd, path.name, 262144)
        finally: os.close(request_fd)
        require(sha(request_raw) == expected_sha256, 'Program request pin differs')
        request = object_json(request_raw); validate_versioned_request(request, REQUEST_FIELDS)
        require(request['kind'] == 'programExport', 'Invalid program export request kind')
        for key in ('projectID','exportID'): canonical_uuid(request[key])
        for key in ('requestIdentity','sourceManifestSHA256'): hash_value(request[key])
        require(stage.name == request['exportID'] and stage.parent.name == request['projectID'] and stage.parent.parent.name == ('workflow-operations' if managed else 'workflow-program-exports'), 'Program stage identity differs')
        require(request['sourceDirectory'] == str(stage/'source'), 'Expected private source inside retained stage')
        require(not os.path.lexists(stage/'package'), 'Program output already exists')
        settings = normalize_program_export(request['configuration'])
        source_fd = directory(request['sourceDirectory'])
        manifest_raw = read_regular(source_fd,'export-manifest.json',2_000_000)
        require(sha(manifest_raw) == request['sourceManifestSHA256'], 'Trained source manifest pin differs')
        manifest = object_json(manifest_raw)
        require(type(manifest.get('schemaVersion')) is int and manifest['schemaVersion'] == 1 and manifest.get('kind') == 'kwipmlTrainedModel', 'Unsupported trained source manifest')
        canonical_uuid(manifest.get('runID'))
        for key in ('requestIdentity','modelManifestSHA256'): hash_value(manifest.get(key))
        rows = manifest.get('files')
        require(type(rows) is list and 4 <= len(rows) <= len(ALLOWED), 'Invalid trained source inventory')
        captured = {}
        for row in rows:
            require(type(row) is dict and set(row) == {'path','bytes','sha256'}, 'Invalid source inventory row')
            name = row['path']; require(type(name) is str and name in ALLOWED and name not in captured, 'Invalid source file name')
            limit = 100_000_000 if name == 'checkpoint.pt' else 4_000_000 if name == 'rate-trace.json' else 131_072 if name == 'data_loading_recipe.py' else 2_000_000
            require(type(row['bytes']) is int and 0 < row['bytes'] <= limit, 'Invalid source file size'); hash_value(row['sha256'])
            raw = read_regular(source_fd,name,limit)
            require(len(raw) == row['bytes'] and sha(raw) == row['sha256'], 'Source file integrity differs: '+name)
            captured[name] = raw
        require(REQUIRED <= captured.keys(), 'Missing trained source files')
        contract = object_json(captured['input-contract.json'])
        require(type(contract.get('schemaVersion')) is int and contract['schemaVersion'] in (1,2), 'Bag packet programs require the existing packet inference export')
        spec = contract.get('input',{}); axes = spec.get('shape')
        require(type(axes) is list and len(axes) >= 2 and axes[0] == 'N', 'Invalid trained input axes')
        input_contract = {'dtype':spec.get('dtype'),'sampleShape':axes[1:]}
        identity = {'runID':manifest['runID'],'requestIdentity':manifest['requestIdentity'],'modelManifestSHA256':manifest['modelManifestSHA256'],'exportManifestSHA256':request['sourceManifestSHA256']}
        def recheck():
            check_cancelled()
            for location, retained in ((str(stage),stage_fd),(request['sourceDirectory'],source_fd)):
                current = directory(location)
                try:
                    a, b = os.fstat(retained), os.fstat(current)
                    require((a.st_dev,a.st_ino) == (b.st_dev,b.st_ino), 'Retained directory identity changed')
                finally: os.close(current)
            current_request = directory(str(path.parent))
            try: require(read_regular(current_request,path.name,262144) == request_raw, 'Program request changed')
            finally: os.close(current_request)
            require(read_regular(source_fd,'export-manifest.json',2_000_000) == manifest_raw, 'Source manifest changed')
            for name, raw in captured.items():
                require(read_regular(source_fd,name,max(len(raw),1)) == raw, 'Source artifact changed')
        recheck()
        emit({'type':'programExportStarted','projectID':request['projectID'],'exportID':request['exportID'],'requestIdentity':request['requestIdentity']})
        # All executable reads below use a private immutable byte snapshot, not
        # mutable original paths. The pinned helper owns original schema checks.
        with tempfile.TemporaryDirectory(prefix='kwipml-program-source-') as temporary:
            folder = Path(temporary)
            for name,raw in dict(captured, **{'export-manifest.json':manifest_raw}).items():
                (folder/name).write_bytes(raw)
            namespace = types.ModuleType('kwipml_owned_program_source')
            namespace.__file__ = str(folder/'inference.py')
            exec(compile(captured['inference.py'], namespace.__file__, 'exec'), namespace.__dict__)
            with torch.random.fork_rng(devices=[]):
                predictor = namespace.load_model()
                model = predictor._model
                if settings['device'] == 'mps' and not torch.backends.mps.is_available():
                    raise ValueError('MPS unavailable; choose CPU explicitly')
                model = model.to(settings['device'])
                example = torch.zeros((settings['example_batch_size'],*input_contract['sampleShape']),dtype=getattr(torch,input_contract['dtype']),device=settings['device'])
                result = build_package(model,example,input_contract,settings,stage/'package',source_identity=identity,check_cancelled=recheck)
        recheck()
        emit({'type':'programExportCompleted','projectID':request['projectID'],'exportID':request['exportID'],'requestIdentity':request['requestIdentity'],'manifestSHA256':sha((stage/'package'/'manifest.json').read_bytes())})
        return result
    finally:
        if source_fd is not None: os.close(source_fd)
        os.close(stage_fd)


def main():
    import argparse
    import json
    import sys
    parser = argparse.ArgumentParser(description='Execute an app-owned pinned inference-program export request')
    parser.add_argument('--request', required=True)
    parser.add_argument('--request-sha256', required=True)
    args = parser.parse_args()
    def emit(value): print(json.dumps(value,sort_keys=True,allow_nan=False),flush=True)
    try:
        execute_export(args.request,args.request_sha256,emit)
    except Exception as error:
        emit({'type':'programExportFailed','error':str(error)[:1600]})
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
