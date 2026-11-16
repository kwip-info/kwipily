"""Versioned path-free audio/prepared-tensor artifacts in the owned run inventory.

Image v1 containers retain their existing codec and digest. This codec adds typed
assets; it grants no source access. A publisher/reader is supplied by the same
operation that owns table publication and its resource reservation.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import math
import re
import struct

import numpy as np

MAGIC = b'KWIPMEDIA1\n'
MAX_HEADER_BYTES = 8192
MAX_ARTIFACT_BYTES = 50_000_000
SIZES = {'uint8': 1, 'uint16': 2, 'int16': 2, 'int32': 4, 'float32': 4, 'int64': 8}
KINDS = {'audio': ('audioSamples', 'audioSamplesRef'),
         'tensor': ('preparedTensor', 'preparedTensorRef')}
AXES = {'samples', 'channels', 'height', 'width', 'features', 'tokens', 'frequency', 'time', 'frames'}
REF_KEYS = {'schemaVersion', 'kind', 'sha256', 'shape', 'axes', 'dtype', 'metadata'}
_scope = ContextVar('owned_media_artifact_context', default=None)


def _canonical_numbers(value):
    if type(value) is float and value.is_integer() and abs(value)<=2**53-1:return int(value)
    if type(value) is list:return [_canonical_numbers(x) for x in value]
    if type(value) is dict:return {k:_canonical_numbers(v) for k,v in value.items()}
    return value


def _json(value):
    return json.dumps(_canonical_numbers(value), sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('ascii')


def _require(condition, message):
    if not condition: raise ValueError(message)


def _pin(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def _geometry(shape, axes, dtype):
    _require(type(shape) is list and 1 <= len(shape) <= 8 and
             all(type(v) is int and v > 0 for v in shape) and
             type(axes) is list and len(axes) == len(shape) and
             all(type(a) is str and a in AXES for a in axes) and
             len(set(axes)) == len(axes) and type(dtype) is str and dtype in SIZES,
             'Media tensor shape, named axes or dtype is invalid.')
    count = SIZES[dtype]
    for dimension in shape:
        _require(dimension <= MAX_ARTIFACT_BYTES // count,
                 'Decoded media tensor exceeds its byte budget.')
        count *= dimension
    return count


def _metadata(kind, metadata, shape, axes, dtype):
    _require(type(metadata) is dict, 'Media metadata must be an object.')
    if kind == 'audio':
        _require(set(metadata) == {'sampleRate', 'sourceBits'} and
                 type(metadata['sampleRate']) is int and 1 <= metadata['sampleRate'] <= 768000 and
                 type(metadata['sourceBits']) is int and metadata['sourceBits'] in (8, 16, 24, 32) and
                 axes == ['samples', 'channels'] and len(shape) == 2 and
                 dtype == {8: 'uint8', 16: 'int16', 24: 'int32', 32: 'int32'}[metadata['sourceBits']],
                 'Audio metadata must preserve PCM precision, rate and sample/channel axes.')
    else:
        allowed = {'parentSHA256', 'recipeSHA256', 'sourceStart', 'sourceEnd',
                   'sourceRate', 'validSamples', 'sampleRate', 'sourceBits', 'geometry', 'maskConvention', 'spectral', 'validFrames'}
        _require(not set(metadata) - allowed, 'Unsupported prepared media metadata.')
        for key, value in metadata.items():
            if key.endswith('SHA256'):
                _require(_pin(value), 'Prepared media provenance requires exact SHA-256 pins.')
            elif key == 'sourceBits':
                _require(type(value) is int and value in (8,16,24,32) and
                         (axes==['channels','samples'] and dtype=={8:'uint8',16:'int16',24:'int32',32:'int32'}[value] or
                          axes==['samples'] and dtype=='int64' and 'maskConvention' in metadata),
                         'Prepared PCM precision differs from its waveform or mask.')
            elif key == 'spectral':
                if __package__:from .audio_spectral import validate_metadata
                else:from audio_spectral import validate_metadata
                validate_metadata(value)
                _require(axes==['frames'] and dtype=='int64' or axes==['channels','frames','frequency'] and dtype=='float32','Spectral axes/type do not match their descriptor.')
            elif key == 'maskConvention':
                _require(value in ('valid','blocked'), 'Mask convention must state whether one means valid or blocked.')
            elif key == 'geometry':
                _require(type(value) is list and len(value) == 9 and
                         all(type(v) in (int, float) and math.isfinite(v) for v in value),
                         'Image geometry mapping requires a finite 3×3 transform.')
            else:
                _require(type(value) is int and 0 <= value <= 2**53 - 1,
                         'Media sample/length metadata requires nonnegative exact integers.')
        if 'sourceStart' in metadata or 'sourceEnd' in metadata:
            _require({'sourceStart', 'sourceEnd', 'sourceRate'} <= set(metadata) and
                     metadata['sourceEnd'] >= metadata['sourceStart'] and metadata['sourceRate'] > 0,
                     'Media source interval must have ordered sample offsets and a positive rate.')
        if 'spectral' in metadata:
            spec=metadata['spectral'];rate=metadata.get('sampleRate')
            _require(type(rate) is int and rate>0 and spec['maximumHz']<=rate/2,'Spectral frequency range exceeds its sample rate.')
            if axes==['channels','frames','frequency']:_require(shape[2]==(spec['melBands'] if spec['representation']=='logMel' else spec['fftSize']//2+1),'Spectral bins differ from their recipe.')
        if 'validFrames' in metadata:
            _require('frames' in axes and 0<=metadata['validFrames']<=shape[axes.index('frames')] and 'spectral' in metadata,'Valid frames exceed the spectral extent.')
        if 'validSamples' in metadata:
            _require('samples' in axes and metadata['validSamples'] <= shape[axes.index('samples')],
                     'Valid samples exceed the prepared tensor extent.')
    _require(len(_json(metadata)) <= 4096, 'Media metadata exceeds its byte budget.')


def _kind(reference_kind):
    for kind, (_, ref) in KINDS.items():
        if reference_kind == ref: return kind
    raise ValueError('Unsupported media reference kind.')


def validate_reference(value):
    _require(type(value) is dict and set(value) == REF_KEYS and
             type(value.get('schemaVersion')) is int and value['schemaVersion'] == 1 and
             _pin(value.get('sha256')), 'Invalid path-free media reference.')
    kind = _kind(value['kind'])
    _geometry(value['shape'], value['axes'], value['dtype'])
    _metadata(kind, value['metadata'], value['shape'], value['axes'], value['dtype'])
    return json.loads(_json(value))


def encode_asset(samples, *, kind, shape, axes, dtype, metadata):
    _require(type(kind) is str and kind in KINDS, 'Unsupported media asset kind.')
    size = _geometry(shape, axes, dtype)
    _metadata(kind, metadata, shape, axes, dtype)
    _require(type(samples) is bytes and len(samples) == size,
             'Media sample bytes differ from their declared geometry.')
    if dtype == 'float32':
        _require(np.isfinite(np.frombuffer(samples, dtype='<f4')).all().item(),
                 'Media tensor samples must be finite.')
    header = dict(schemaVersion=1, kind=KINDS[kind][0], shape=shape, axes=axes,
                  dtype=dtype, metadata=metadata, byteOrder='little')
    raw = _json(header)
    _require(len(raw) <= MAX_HEADER_BYTES and len(MAGIC) + 4 + len(raw) + size <= MAX_ARTIFACT_BYTES,
             'Media container exceeds its byte budget.')
    content = MAGIC + struct.pack('>I', len(raw)) + raw + samples
    reference = dict(schemaVersion=1, kind=KINDS[kind][1],
                     sha256=hashlib.sha256(content).hexdigest(), shape=shape, axes=axes,
                     dtype=dtype, metadata=metadata)
    return content, validate_reference(reference)


def _pairs(items):
    result = {}
    for key, value in items:
        _require(key not in result, 'Duplicate media container header field.')
        result[key] = value
    return result


def decode_asset(content, expected=None):
    _require(type(content) is bytes and len(MAGIC)+4 < len(content) <= MAX_ARTIFACT_BYTES and
             content.startswith(MAGIC), 'Unsupported media container.')
    start = len(MAGIC) + 4
    length = struct.unpack('>I', content[len(MAGIC):start])[0]
    _require(1 <= length <= MAX_HEADER_BYTES and start+length <= len(content),
             'Invalid media container header length.')
    try:
        header = json.loads(content[start:start+length].decode('ascii'), object_pairs_hook=_pairs,
                            parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite media metadata.')))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError('Malformed media container header.') from error
    _require(type(header) is dict and set(header) ==
             {'schemaVersion', 'kind', 'shape', 'axes', 'dtype', 'metadata', 'byteOrder'} and
             type(header['schemaVersion']) is int and header['schemaVersion'] == 1 and
             header['byteOrder'] == 'little', 'Unsupported media container header.')
    kind = next((key for key, (name, _) in KINDS.items() if header['kind'] == name), None)
    _require(kind is not None, 'Unsupported media container payload.')
    # Re-encoding verifies canonical metadata, payload length and finite values.
    samples = content[start+length:]
    canonical, reference = encode_asset(samples, kind=kind,
        **{key: header[key] for key in ('shape', 'axes', 'dtype', 'metadata')})
    _require(canonical == content, 'Media container is not canonical.')
    if expected is not None:
        _require(validate_reference(expected) == reference,
                 'Media artifact differs from its immutable reference.')
    return header, samples


@contextmanager
def artifact_scope(publisher, reader=None):
    _require(callable(publisher) and (reader is None or callable(reader)),
             'Media publisher/reader must be callable.')
    token = _scope.set((publisher, reader, set()))
    try: yield
    finally: _scope.reset(token)


def publish_encoded(content, reference, check_cancelled=lambda: None):
    context = _scope.get()
    _require(context is not None, 'Media requires an owned artifact publisher.')
    check_cancelled(); reference = validate_reference(reference)
    decode_asset(content, expected=reference)
    if reference['sha256'] not in context[2]:
        context[0](content, reference); context[2].add(reference['sha256'])
    check_cancelled()
    return reference


def read_asset(reference, check_cancelled=lambda: None):
    context = _scope.get()
    _require(context is not None and context[1] is not None,
             'Media requires an owned artifact reader.')
    check_cancelled(); reference = validate_reference(reference)
    content = context[1](reference)
    check_cancelled()
    return decode_asset(content, expected=reference)


def filename(reference):
    reference = validate_reference(reference)
    return 'media-' + reference['sha256'] + '.kwipmedia'


def is_reference(value, dtype=None):
    try:
        reference = validate_reference(value)
        return dtype is None or _kind(reference['kind']) == dtype
    except (ValueError, TypeError): return False


def references(table):
    indices = [(i, c['dtype']) for i, c in enumerate(table.columns) if c['dtype'] in KINDS]
    # Typed scalar tables have no media references. Do not reconstruct their
    # native rows merely to execute an empty inner loop. Media columns retain
    # the complete reference/kind validation below.
    if not indices: return
    for row in table.rows:
        for i, dtype in indices:
            if row[i] is None: continue
            reference = validate_reference(row[i])
            _require(_kind(reference['kind']) == dtype, 'Media reference kind differs from its table column.')
            yield reference


def validate_inventory(manifest, root, check_cancelled=lambda: None):
    if __package__:
        from .dataflow_storage import bounded_regular_read, verify_regular_file
        from .dataflow_table_reader import read_table_artifact
        from .image_artifacts import _image_columns
    else:
        from dataflow_storage import bounded_regular_read, verify_regular_file
        from dataflow_table_reader import read_table_artifact
        from image_artifacts import _image_columns
    from pathlib import Path
    root = Path(root); assets = {}
    all_pins = {entry.get('sha256') for entry in manifest['artifacts'] if entry.get('kind') in ('imagePixels','audioSamples','preparedTensor')}
    for entry in manifest['artifacts']:
        if entry.get('kind') not in ('audioSamples', 'preparedTensor'): continue
        check_cancelled(); ref = validate_reference(entry.get('media'))
        _require(entry['kind'] == KINDS[_kind(ref['kind'])][0] and
                 entry.get('path') == filename(ref) and entry.get('sha256') == ref['sha256'] and
                 ref['sha256'] not in assets, 'Invalid or duplicate media artifact inventory.')
        content = bounded_regular_read(root / entry['path'], MAX_ARTIFACT_BYTES)
        _require(len(content) == entry['bytes'], 'Media inventory size differs from the container.')
        decode_asset(content, expected=ref)
        parent = ref['metadata'].get('parentSHA256')
        _require(parent is None or parent in all_pins, 'Prepared media parent is missing from retained inventory.')
        assets[ref['sha256']] = ref
    for entry in manifest['artifacts']:
        if entry.get('kind') != 'table': continue
        path = root / entry['path']; check_cancelled()
        if _image_columns(path, check_cancelled, column_types=set(KINDS)) is False: continue
        verify_regular_file(path, entry['bytes'], entry['sha256'], check_cancelled)
        table = read_table_artifact(path, entry['sha256'], check_cancelled=check_cancelled)
        try:
            for ref in references(table):
                _require(assets.get(ref['sha256']) == ref, 'Media table reference lacks its immutable owned artifact.')
        finally:
            table.rows.store.close(); table.row_ids.store.close()
    return assets


def inventory_reader(manifest, files, check_cancelled=lambda:None):
    """Resolve only media admitted by completed_inventory; never a source path."""
    assets={e['sha256']:e for e in manifest['artifacts'] if e.get('kind') in ('audioSamples','preparedTensor')}
    def read(reference):
        check_cancelled();reference=validate_reference(reference);entry=assets.get(reference['sha256'])
        _require(entry is not None and entry.get('media')==reference and entry['path']==filename(reference),
                 'Media is outside its admitted immutable run inventory.')
        content=files[entry['path']];check_cancelled()
        return content
    return read
