from io import BytesIO
import struct
import wave

import numpy as np
import pytest

from runner.file_input_audio import decode_audio


def wav(payload, width=2, channels=2, rate=48000):
    out = BytesIO()
    with wave.open(out, 'wb') as stream:
        stream.setnchannels(channels); stream.setsampwidth(width)
        stream.setframerate(rate); stream.writeframes(payload)
    return out.getvalue()


@pytest.mark.parametrize('width,payload,expected,dtype', [
    (1, bytes([0, 128, 255, 1]), [0, 128, 255, 1], 'uint8'),
    (2, struct.pack('<4h', -32768, 0, 32767, -1), [-32768, 0, 32767, -1], 'int16'),
    (3, b'\x00\x00\x80\xff\xff\x7f\x00\x00\x00\xff\xff\xff',
     [-8388608, 8388607, 0, -1], 'int32'),
    (4, struct.pack('<4i', -2147483648, 0, 2147483647, -1),
     [-2147483648, 0, 2147483647, -1], 'int32'),
])
def test_pcm_keeps_exact_integer_samples_channels_rate_and_source_bits(width, payload, expected, dtype):
    decoded = decode_audio(wav(payload, width=width))
    assert decoded['shape'] == [2, 2]
    assert decoded['axes'] == ['samples', 'channels']
    assert decoded['metadata'] == {'sampleRate': 48000, 'sourceBits': width * 8}
    assert np.frombuffer(decoded['samples'], dtype=np.dtype(dtype).newbyteorder('<')).tolist() == expected


def test_refuses_truncation_unsupported_encoding_and_admission_before_decode():
    content = wav(bytes(16))
    for raw in (content[:-1], b'not audio', content[:20] + b'\x03\x00' + content[22:]):
        with pytest.raises(ValueError): decode_audio(raw)
    with pytest.raises(ValueError, match='budget'):
        decode_audio(content, maximum_decoded_bytes=8)


def test_cancellation_and_no_conversion_options():
    def cancel(): raise InterruptedError('cancelled')
    with pytest.raises(InterruptedError): decode_audio(wav(bytes(8)), check_cancelled=cancel)
    with pytest.raises(ValueError): decode_audio(wav(bytes(8)), options={'sampleRate': 16000})


@pytest.mark.parametrize('engine', ['disk', 'polars'])
def test_file_input_audio_owns_reference_closure_and_reopens_without_original(tmp_path, engine):
    import hashlib
    import json
    from runner.dataflow_cli import run_dataflow
    from runner.dataflow_storage import completed_inventory
    from runner.file_input_options import defaults
    from runner.media_artifacts import decode_asset
    from runner.dataflow_table_reader import read_table_artifact
    source = tmp_path / 'sample.wav'
    source.write_bytes(wav(struct.pack('<4h', -123, 123, 0, 1)))
    config = defaults(); config['extensions'] = ['wav']
    config['readers']['audio'] = {'format': 'pcmWav'}
    graph = {'schemaVersion': 1, 'nodes': [dict(id='audio', operator='kwip.data.sourceFiles', config=config)], 'edges': []}
    path = tmp_path / 'graph.json'; path.write_text(json.dumps(graph))
    out = tmp_path / 'output'
    run_dataflow(path, {'audio': str(source)}, out, lambda _: None, engine=engine)
    source.unlink()
    manifest, files, _ = completed_inventory(out / 'manifest.json')
    media = next(entry for entry in manifest['artifacts'] if entry['kind'] == 'audioSamples')
    _, restored = decode_asset(files[media['path']], expected=media['media'])
    assert restored == struct.pack('<4h', -123, 123, 0, 1)
    table_entry = next(entry for entry in manifest['artifacts'] if entry['kind'] == 'table')
    table = read_table_artifact(out / table_entry['path'], table_entry['sha256'])
    try:
        ref = table.rows[0][table.index('audio_content')]
        assert ref == media['media'] and len(table.rows) == 1
    finally: table.rows.store.close()
    # Rehashing a manifest cannot authorize changing or dropping referenced assets.
    missing = {**manifest, 'artifacts': [entry for entry in manifest['artifacts'] if entry != media]}
    (out / 'manifest.json').write_text(json.dumps(missing))
    with pytest.raises(ValueError): completed_inventory(out / 'manifest.json')


def test_odd_final_pcm_payload_and_invalid_frame_geometry():
    decoded = decode_audio(wav(b'\x80', width=1, channels=1))
    assert decoded['shape'] == [1, 1] and decoded['samples'] == b'\x80'
    content = bytearray(wav(bytes(8)))
    # PCM block alignment and byte rate are authoritative, not decoder hints.
    content[32:34] = struct.pack('<H', 1)
    with pytest.raises(ValueError): decode_audio(bytes(content))
    content = bytearray(wav(bytes(8)))
    content[28:32] = struct.pack('<I', 1)
    with pytest.raises(ValueError): decode_audio(bytes(content))
    with pytest.raises(ValueError): decode_audio(wav(bytes(3), channels=1))


def test_old_file_input_defaults_do_not_start_decoding_audio():
    from runner.file_input_options import validate_configuration, defaults
    current = defaults()
    assert 'wav' not in current['extensions']
    current['readers'].pop('audio')
    restored = validate_configuration(current)
    assert restored['readers']['audio'] == {'format': 'disabled'}
