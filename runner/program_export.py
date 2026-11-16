"""Bounded, trusted inference-only torch.export programs. Never load untrusted programs."""
import copy
import hashlib
import inspect
import io
import json
import math
from pathlib import Path
import re
import uuid
import zipfile
import torch
if __package__:
    from .program_export_configuration import DEFAULTS, normalize_program_export
else:
    from program_export_configuration import DEFAULTS, normalize_program_export


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _identity(value):
    if type(value) is not dict or set(value) != {'runID', 'requestIdentity', 'modelManifestSHA256', 'exportManifestSHA256'}:
        raise ValueError('Invalid program source identity')
    try:
        if str(uuid.UUID(value['runID'])).upper() != value['runID']: raise ValueError()
    except (ValueError, TypeError, AttributeError):
        raise ValueError('Invalid program source run identity') from None
    if any(type(value[k]) is not str or re.fullmatch('[0-9a-f]{64}', value[k]) is None for k in value if k != 'runID'):
        raise ValueError('Invalid program source hashes')


def _contract(value):
    if isinstance(value, dict) and value.get('kind') == 'bagPacket':
        raise ValueError('Bag packet input requires the trained packet inference export')
    if type(value) is not dict or set(value) != {'dtype', 'sampleShape'} or value['dtype'] not in ('float32', 'int64'):
        raise ValueError('Invalid ordinary tensor input/output contract')
    shape = value['sampleShape']
    if type(shape) is not list or not 1 <= len(shape) <= 7 or any(type(n) is not int or not 1 <= n <= 4096 for n in shape):
        raise ValueError('Invalid fixed sample axes')
    return dict(value)


def _tensor(x, spec, settings):
    if not isinstance(x, torch.Tensor) or x.layout != torch.strided or x.dtype != getattr(torch, spec['dtype']):
        raise ValueError('Program input dtype/layout differs')
    if x.device.type != settings['device']:
        raise ValueError('Program input device differs from captured device')
    if x.ndim != len(spec['sampleShape'])+1 or list(x.shape[1:]) != spec['sampleShape']:
        raise ValueError('Program input sample shape differs')
    if not settings['minimum_batch_size'] <= x.shape[0] <= settings['maximum_batch_size']:
        raise ValueError('Program input batch is outside captured bounds')
    if x.numel() > 2_000_000 or (x.is_floating_point() and not torch.isfinite(x).all().item()):
        raise ValueError('Program input must be finite and bounded')


def _state(ep):
    rows = []
    parameters = set(ep.graph_signature.parameters)
    buffers = set(ep.graph_signature.buffers)
    for name, value in sorted(dict(ep.state_dict, **ep.constants).items()):
        if not isinstance(value, torch.Tensor) or value.device.type == 'meta' or value.dtype not in (torch.float32, torch.int64, torch.bool):
            raise ValueError('Program state contains unsupported tensors')
        if value.numel() > 2_000_000 or (value.is_floating_point() and not torch.isfinite(value).all().item()):
            raise ValueError('Program state must be finite and bounded')
        rows.append({'name': name, 'kind': 'parameter' if name in parameters else 'buffer' if name in buffers else 'constant', 'dtype': str(value.dtype).removeprefix('torch.'), 'shape': list(value.shape)})
    for kind in ('parameter', 'buffer', 'constant'):
        if sum(math.prod(row['shape']) for row in rows if row['kind'] == kind) > 2_000_000:
            raise ValueError('Program state exceeds element budget')
    return rows


def _ranges(ep):
    return [{'symbol': str(k), 'minimum': int(v.lower), 'maximum': int(v.upper)} for k, v in sorted(ep.range_constraints.items(), key=lambda p: str(p[0]))]


def _signature(ep, manifest):
    nodes = {node.name: node for node in ep.graph.nodes}
    for field, names in (('input', ep.graph_signature.user_inputs), ('output', ep.graph_signature.user_outputs)):
        if len(names) != 1 or not isinstance(names[0], str) or names[0] not in nodes:
            raise ValueError('Program signature requires one tensor input/output')
        value = nodes[names[0]].meta.get('val')
        spec = manifest[field]
        if not isinstance(value, torch.Tensor) or str(value.dtype).removeprefix('torch.') != spec['dtype'] or value.device.type != manifest['runtime']['device'] or len(value.shape) != len(spec['sampleShape'])+1:
            raise ValueError('Program native signature differs')
        if any(not isinstance(axis, int) or axis != expected for axis,expected in zip(value.shape[1:],spec['sampleShape'])):
            raise ValueError('Program native sample signature differs')
        batch = value.shape[0]
        settings = manifest['configuration']
        if settings['batch_mode'] == 'static':
            if not isinstance(batch, int) or batch != settings['example_batch_size']:
                raise ValueError('Program static signature differs')
        elif not isinstance(batch, torch.SymInt) or str(batch) not in {str(k) for k in ep.range_constraints}:
            raise ValueError('Program bounded signature differs')


def _archive(raw):
    if type(raw) is not bytes or not 0 < len(raw) <= 100_000_000:
        raise ValueError('Program archive size exceeds budget')
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            rows = archive.infolist(); names = [row.filename for row in rows]
            if len(rows) > 512 or len(set(names)) != len(names) or sum(row.file_size for row in rows) > 100_000_000:
                raise ValueError('Program archive inventory exceeds budget')
            for row in rows:
                path = Path(row.filename)
                if path.is_absolute() or '..' in path.parts or '\\' in row.filename or row.flag_bits & 1 or (row.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('Unsafe program archive entry')
    except zipfile.BadZipFile as error:
        raise ValueError('Invalid program archive') from error


def _decode(raw, manifest, expected_manifest_sha256):
    if type(manifest) is not dict or len(_canonical(manifest)) > 2_000_000 or _sha(_canonical(manifest)) != expected_manifest_sha256:
        raise ValueError('Program manifest hash differs')
    if set(manifest) != {'schemaVersion','kind','runtime','configuration','input','output','sourceIdentity','shapeConstraints','rangeConstraints','state','files'} or type(manifest['schemaVersion']) is not int or manifest['schemaVersion'] != 1 or manifest['kind'] != 'kwipmlExportedProgram':
        raise ValueError('Unknown program manifest schema')
    settings = normalize_program_export(manifest['configuration'])
    if manifest['runtime'] != {'torch': torch.__version__, 'device': settings['device']}:
        raise ValueError('Program runtime differs; use the exact captured Torch version and device')
    if settings['device'] == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS is unavailable; recapture explicitly on CPU')
    spec = _contract(manifest['input']); output = _contract(manifest['output']); _identity(manifest['sourceIdentity'])
    if output['dtype'] != 'float32': raise ValueError('Program output must be float32')
    expected = {'mode': settings['batch_mode'], 'leadingAxis': 'batch', 'minimum': settings['minimum_batch_size'], 'maximum': settings['maximum_batch_size'], 'example': settings['example_batch_size']}
    if manifest['shapeConstraints'] != expected: raise ValueError('Program shape constraints differ')
    files = manifest['files']
    if type(files) is not list or len(files) != 4 or {r.get('path') for r in files if isinstance(r, dict)} != {'program.pt2','inference.py','README.md','requirements.txt'}:
        raise ValueError('Program file inventory differs')
    for row in files:
        if set(row) != {'path','sha256','bytes'} or type(row['bytes']) is not int or row['bytes'] <= 0 or type(row['sha256']) is not str or re.fullmatch('[0-9a-f]{64}', row['sha256']) is None:
            raise ValueError('Invalid program file binding')
    row = next(r for r in files if r['path'] == 'program.pt2')
    if row['sha256'] != _sha(raw) or row['bytes'] != len(raw): raise ValueError('Program bytes differ')
    for value in (spec, output):
        if math.prod(value['sampleShape'])*settings['maximum_batch_size'] > 2_000_000: raise ValueError('Program tensor budget exceeded')
    _archive(raw)
    ep = torch.export.load(io.BytesIO(raw))
    _signature(ep, manifest)
    if _state(ep) != manifest['state']: raise ValueError('Program state metadata differs')
    if _ranges(ep) != manifest['rangeConstraints']: raise ValueError('Program native constraints differ')
    if settings['batch_mode'] == 'static' and ep.range_constraints: raise ValueError('Static program has symbolic axes')
    if settings['batch_mode'] == 'bounded' and (len(ep.range_constraints) != 1 or next(iter(ep.range_constraints.values())).upper != settings['maximum_batch_size'] or next(iter(ep.range_constraints.values())).lower != settings['minimum_batch_size']):
        raise ValueError('Program native bounds differ')
    if ep.graph_signature.buffers_to_mutate or ep.graph_signature.user_inputs_to_mutate:
        raise ValueError('Program mutates state or inputs')
    module = ep.module()
    class Inference:
        def __call__(self, x):
            _tensor(x, spec, settings)
            with torch.no_grad(): result = module(x)
            _tensor(result, output, settings)
            if result.shape[0] != x.shape[0]: raise ValueError('Program output observation count differs')
            return result
        predict = __call__
    return Inference()


def _portable_source():
    functions = [_canonical, _sha, _identity, _contract, _tensor, _state, _ranges, _signature, _archive, _decode]
    return ('"""Trusted app-generated inference only. Never load untrusted programs."""\nimport hashlib, io, json, math, re, uuid, zipfile\nfrom pathlib import Path\nimport torch\nDEFAULTS = '+repr(DEFAULTS)+'\n'+inspect.getsource(normalize_program_export)+'\n'+'\n'.join(inspect.getsource(f) for f in functions)+'''\ndef load_model():
    root = Path(__file__).resolve().parent
    def read(name):
        p = root/name
        if p.is_symlink() or not p.is_file(): raise ValueError('Unsafe package file')
        return p.read_bytes()
    manifest_raw = read('manifest.json')
    manifest = json.loads(manifest_raw)
    if manifest_raw != _canonical(manifest): raise ValueError('Noncanonical program manifest')
    if type(manifest.get('files')) is not list or len(manifest['files']) != 4 or {row.get('path') for row in manifest['files'] if isinstance(row, dict)} != {'program.pt2','inference.py','README.md','requirements.txt'}:
        raise ValueError('Program file inventory differs')
    for row in manifest['files']:
        raw = read(row['path'])
        if len(raw) != row['bytes'] or _sha(raw) != row['sha256']: raise ValueError('Package file hash differs')
    return _decode(read('program.pt2'), manifest, _sha(manifest_raw))
''').encode()


def _support():
    return {'inference.py': _portable_source(), 'README.md': ('Inference-only torch.export program. Requires exact Torch '+torch.__version__+'.\nNever load untrusted programs or helpers. Learned weights and buffers, synthetic trace input, and graph names/debug metadata are included. No training checkpoint or original Python model source is needed. Device is fixed at capture; no fallback.\n').encode(), 'requirements.txt': ('torch=='+torch.__version__+'\n').encode()}


def export_program(model, example, input_contract, configuration, *, source_identity, check_cancelled=lambda: None):
    settings = normalize_program_export(configuration); spec = _contract(input_contract); _identity(source_identity)
    if settings['device'] == 'mps' and not torch.backends.mps.is_available(): raise ValueError('MPS unavailable; choose CPU explicitly')
    _tensor(example, spec, settings)
    if example.shape[0] != settings['example_batch_size']: raise ValueError('Example batch differs')
    if math.prod(spec['sampleShape'])*settings['maximum_batch_size'] > 2_000_000: raise ValueError('Input budget exceeded')
    check_cancelled()
    mps_rng = torch.mps.get_rng_state() if settings['device'] == 'mps' else None
    try:
        with torch.random.fork_rng(devices=[]):
            clone = copy.deepcopy(model).eval()
            before = {k:v.detach().clone() for k,v in clone.state_dict().items()}
            for v in before.values():
                if v.device.type != settings['device'] or v.numel() > 2_000_000: raise ValueError('Model state device/budget differs')
                if v.is_floating_point() and not torch.isfinite(v).all().item(): raise ValueError('Model state must be finite')
            trace = example.detach().clone()
            with torch.no_grad(): result = clone(trace)
            if any(not torch.equal(v, clone.state_dict()[k]) for k,v in before.items()) or not torch.equal(trace, example): raise ValueError('Inference forward mutates state or input')
            if not isinstance(result, torch.Tensor) or result.dtype != torch.float32 or result.ndim < 2 or result.shape[0] != example.shape[0]: raise ValueError('Export requires one float32 output tensor preserving observations')
            output = _contract({'dtype':'float32','sampleShape':list(result.shape[1:])}); _tensor(result, output, settings)
            if math.prod(output['sampleShape'])*settings['maximum_batch_size'] > 2_000_000: raise ValueError('Output budget exceeded')
            dynamic = None if settings['batch_mode'] == 'static' else ({0:torch.export.Dim('batch', min=settings['minimum_batch_size'], max=settings['maximum_batch_size'])},)
            check_cancelled()
            try:
                ep = torch.export.export(clone, (example.detach().clone(),), dynamic_shapes=dynamic, strict=settings['strict'], prefer_deferred_runtime_asserts_over_guards=settings['prefer_deferred_runtime_asserts'])
            except Exception as error:
                raise ValueError('Export cannot trace this model on the selected device: '+str(error)[:1200]) from error
            if len(list(ep.graph.nodes)) > 4096: raise ValueError('Export graph exceeds node budget')
            for node in ep.graph.nodes:
                if node.op == 'call_function' and isinstance(node.target, torch._ops.OpOverload) and node.target.namespace not in ('aten','prims'):
                    raise ValueError('Export custom operators are unsupported')
            buffer = io.BytesIO(); check_cancelled(); torch.export.save(ep, buffer); raw = buffer.getvalue(); _archive(raw)
            support = _support(); files = dict(support, **{'program.pt2':raw})
            manifest = {'schemaVersion':1,'kind':'kwipmlExportedProgram','runtime':{'torch':torch.__version__,'device':settings['device']},'configuration':settings,'input':spec,'output':output,'sourceIdentity':dict(source_identity),'shapeConstraints':{'mode':settings['batch_mode'],'leadingAxis':'batch','minimum':settings['minimum_batch_size'],'maximum':settings['maximum_batch_size'],'example':settings['example_batch_size']},'rangeConstraints':_ranges(ep),'state':_state(ep),'files':[{'path':name,'sha256':_sha(data),'bytes':len(data)} for name,data in sorted(files.items())]}
            check_cancelled(); restored = reload_program(raw, manifest, expected_manifest_sha256=_sha(_canonical(manifest)))
            torch.testing.assert_close(restored(example), result)
            return raw, manifest
    finally:
        if mps_rng is not None: torch.mps.set_rng_state(mps_rng)


def reload_program(program_bytes, manifest, *, expected_manifest_sha256):
    return _decode(program_bytes, manifest, expected_manifest_sha256)


def build_package(model, example, input_contract, configuration, destination, *, source_identity, check_cancelled=lambda: None):
    destination = Path(destination)
    if destination.exists(): raise ValueError('Program destination already exists')
    raw, manifest = export_program(model, example, input_contract, configuration, source_identity=source_identity, check_cancelled=check_cancelled)
    destination.mkdir(parents=False)
    for name,data in dict(_support(), **{'program.pt2':raw,'manifest.json':_canonical(manifest)}).items():
        check_cancelled()
        with (destination/name).open('xb') as file: file.write(data)
    return manifest
