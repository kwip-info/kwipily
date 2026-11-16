"""Bounded native starting-value policy, isolated from the training RNG."""
import hashlib
import json
import math
import struct
import torch

MAX_RECIPE_BYTES = 32768
MAX_RULES = 128
MAX_PARAMETERS = 200000
SEED_VERSION = 'kwip.initialization.owner-seed.v1'
DEFAULTS = {
    'xavier_uniform': {'gain': 1.0},
    'kaiming_uniform': {'a': 0.0, 'mode': 'fan_in', 'nonlinearity': 'leaky_relu'},
    'normal': {'mean': 0.0, 'std': 1.0},
    'zeros': {},
}


def _exact(value, keys, message):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(message)


def _integer(value, maximum):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError('Initialization requires an exact integer seed in the declared range.')
    return int(value)


def _number(value, minimum, maximum):
    if type(value) not in (int, float) or not minimum <= value <= maximum:
        raise ValueError('Initialization settings require finite numbers in the declared range.')
    try:
        native = struct.unpack('f', struct.pack('f', value))[0]
    except (OverflowError, struct.error):
        raise ValueError('Initialization setting is not representable as float32.') from None
    if not math.isfinite(native) or (value != 0 and native == 0):
        raise ValueError('Initialization setting is not representable as nonzero finite float32.')
    return float(value)


def validate_recipe(value):
    _exact(value, ('schemaVersion', 'configurationVersion', 'componentID', 'config'), 'Choose a versioned initialization recipe with exactly its declared fields.')
    if type(value['schemaVersion']) is not int or value['schemaVersion'] != 1 or type(value['configurationVersion']) is not int or value['configurationVersion'] != 1 or value['componentID'] != 'initialization':
        raise ValueError('Unsupported initialization recipe identity/version.')
    config = value['config']
    _exact(config, ('seed', 'rules'), 'Initialization accepts a seed and selected-owner rules.')
    seed = _integer(config['seed'], 2**63 - 1)
    if type(config['rules']) is not list or not 1 <= len(config['rules']) <= MAX_RULES:
        raise ValueError('Choose from 1 to 128 initialization rules.')
    rules = []; seen = set()
    for rule in config['rules']:
        _exact(rule, ('owner', 'algorithm', 'settings'), 'Initialization rules need owner, algorithm and settings.')
        owner = rule['owner']
        _exact(owner, ('node', 'parameter'), 'Select an exact model node and relative native parameter name.')
        if any(type(owner[key]) is not str or not owner[key] or '\0' in owner[key] for key in ('node', 'parameter')):
            raise ValueError('Select nonempty actual model parameter owners.')
        identity = (owner['node'], owner['parameter'])
        if identity in seen: raise ValueError('Duplicate initialization owner.')
        seen.add(identity)
        algorithm = rule['algorithm']
        if type(algorithm) is not str or algorithm not in DEFAULTS:
            raise ValueError('Choose Xavier uniform, Kaiming uniform, normal or zeros.')
        supplied = rule['settings']
        if type(supplied) is not dict or set(supplied) != set(DEFAULTS[algorithm]):
            raise ValueError('Initialization accepts only the selected algorithm settings.')
        settings = {**DEFAULTS[algorithm], **supplied}
        for key in ('gain', 'a'):
            if key in settings: settings[key] = _number(settings[key], 0, 100)
        if 'mean' in settings: settings['mean'] = _number(settings['mean'], -1e6, 1e6)
        if 'std' in settings: settings['std'] = _number(settings['std'], 0, 1e6)
        if 'mode' in settings and settings['mode'] not in ('fan_in', 'fan_out'):
            raise ValueError('Choose fan_in or fan_out for Kaiming initialization.')
        if 'nonlinearity' in settings and settings['nonlinearity'] not in ('relu', 'leaky_relu'):
            raise ValueError('Choose relu or leaky_relu for Kaiming initialization.')
        rules.append({'owner': dict(owner), 'algorithm': algorithm, 'settings': settings})
    rules.sort(key=lambda rule: (rule['owner']['node'], rule['owner']['parameter']))
    result = dict(schemaVersion=1, configurationVersion=1, componentID='initialization', config=dict(seed=seed, rules=rules))
    if len(json.dumps(result, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')) > MAX_RECIPE_BYTES:
        raise ValueError('Initialization recipe exceeds its 32768-byte portable limit.')
    return result


def _resolve(model, recipe):
    recipe = validate_recipe(recipe)
    # COMPOSITION_RESOLUTION_BEGIN
    if hasattr(model,'composition_root'):
        if __package__:from .native_ownership import native_owners
        else:from native_ownership import native_owners
        inventory={(m['node'],m['parameter']):(m,p) for m,p in native_owners(model)}
        resolved=[]
        for rule in recipe['config']['rules']:
            key=(rule['owner']['node'],rule['owner']['parameter'])
            if key not in inventory:raise ValueError('Initialization selects actual native parameters, not buffers or foreign owners.')
            metadata,parameter=inventory[key]
            if rule['algorithm'] in ('xavier_uniform','kaiming_uniform') and parameter.ndim<2:raise ValueError('Fan-based initialization requires at least two directions.')
            resolved.append(({k:v for k,v in metadata.items() if k!='requiresGrad'},parameter,rule))
        return recipe,resolved
    # COMPOSITION_RESOLUTION_END
    if not isinstance(model, torch.nn.Module) or not isinstance(getattr(model, 'layers', None), torch.nn.ModuleDict):
        raise ValueError('Initialization requires the selected model native module ownership.')
    parameters = list(model.named_parameters(remove_duplicate=False))
    if any(isinstance(parameter, torch.nn.parameter.UninitializedParameter) for _, parameter in parameters):
        raise ValueError('Materialize all lazy model parameters before initialization.')
    unique = {id(parameter): parameter for _, parameter in parameters}
    if sum(parameter.numel() for parameter in unique.values()) > MAX_PARAMETERS:
        raise ValueError('Whole model exceeds the 200000-parameter initialization budget.')
    if any(parameter.dtype != torch.float32 or parameter.layout != torch.strided or parameter.device.type not in ('cpu', 'mps') for parameter in unique.values()):
        raise ValueError('The selected model requires dense float32 CPU/MPS learned parameters.')
    if any(not torch.isfinite(parameter.detach()).all().item() for parameter in unique.values()):
        raise ValueError('Model parameters must be finite before initialization.')
    mapping = getattr(model, 'layer_keys', None)
    resolved = []; seen = set()
    for rule in recipe['config']['rules']:
        node, relative = rule['owner']['node'], rule['owner']['parameter']
        if mapping is not None:
            if type(mapping) is not dict or node not in mapping:
                raise ValueError('Initialization owner is not in the selected model node ownership.')
            key = mapping[node]
        else: key = node
        if key not in model.layers:
            raise ValueError('Initialization owner is not an actual selected model module.')
        available = dict(model.layers[key].named_parameters(remove_duplicate=False))
        if relative not in available:
            raise ValueError('Initialization selects learned parameters, not buffers or foreign names.')
        parameter = available[relative]
        if id(parameter) in seen:
            raise ValueError('Initialization owners alias the same parameter identity.')
        seen.add(id(parameter))
        if parameter.dtype != torch.float32 or parameter.layout != torch.strided or parameter.device.type not in ('cpu', 'mps'):
            raise ValueError('Initialization requires dense float32 CPU/MPS learned parameters.')
        if not parameter.numel() or any(extent <= 0 for extent in parameter.shape):
            raise ValueError('Initialization requires positive parameter extents.')
        if rule['algorithm'] in ('xavier_uniform', 'kaiming_uniform') and parameter.ndim < 2:
            raise ValueError('Fan-based initialization requires a stored parameter with at least two directions.')
        metadata = dict(node=node, parameter=relative, name='layers.' + key + '.' + relative, shape=list(parameter.shape), dtype='float32')
        resolved.append((metadata, parameter, rule))
    return recipe, resolved


def resolve_owners(model, recipe):
    """Return canonical actual ownership metadata; never expose tensor values."""
    return [metadata for metadata, _, _ in _resolve(model, recipe)[1]]


def apply_initialization(model, recipe, check_cancelled=lambda: None):
    """Generate/validate all candidates before copying; do not replace Parameters."""
    check_cancelled()
    recipe, resolved = _resolve(model, recipe)
    candidates = []
    for metadata, parameter, rule in resolved:
        check_cancelled()
        encoded = json.dumps([SEED_VERSION, recipe['config']['seed'], metadata['node'], metadata['parameter']], sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
        seed = int.from_bytes(hashlib.sha256(encoded).digest()[:8], 'big') % (2**63)
        generator = torch.Generator(device='cpu').manual_seed(seed)
        candidate = torch.empty(tuple(metadata['shape']), dtype=torch.float32, device='cpu')
        function = getattr(torch.nn.init, rule['algorithm'] + '_')
        if rule['algorithm'] == 'zeros': function(candidate)
        else: function(candidate, **rule['settings'], generator=generator)
        if not torch.isfinite(candidate).all().item():
            raise ValueError('Initialization generated nonfinite values; no learned parameters were changed.')
        candidates.append(candidate.to(device=parameter.device))
    check_cancelled()
    previous = [parameter.detach().clone() for _, parameter, _ in resolved]
    with torch.no_grad():
        try:
            for (_, parameter, _), candidate in zip(resolved, candidates): parameter.copy_(candidate)
        except Exception:
            for (_, parameter, _), original in zip(resolved, previous): parameter.copy_(original)
            raise
    return [metadata for metadata, _, _ in resolved]


# standalone export boundary

def initialization_recipe_code(recipe,composition=False):
    """Return a standalone Torch/stdlib policy helper; no automatic application."""
    from pathlib import Path
    canonical = validate_recipe(recipe)
    source = Path(__file__).read_text(encoding='utf-8').split('# standalone export boundary', 1)[0]
    if composition:
        if __package__:from .native_ownership import portable_source
        else:from native_ownership import portable_source
        source=source.replace("        if __package__:from .native_ownership import native_owners\n        else:from native_ownership import native_owners\n",'')+portable_source()
    else:
        start=source.index('    # COMPOSITION_RESOLUTION_BEGIN');end=source.index('    # COMPOSITION_RESOLUTION_END')+len('    # COMPOSITION_RESOLUTION_END\n')
        source=source[:start]+source[end:]
    code = source + '\nINITIALIZATION_RECIPE = ' + repr(canonical) + '\n'
    if len(code.encode('utf-8')) > 131072:
        raise ValueError('Standalone initialization helper exceeds its bounded artifact size.')
    return code
