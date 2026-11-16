"""Strict bounded PCM WAV decoding of already captured bytes, without I/O grants."""
from io import BytesIO
import struct
import wave

import numpy as np

MAX_ENCODED_BYTES = 64 * 1024 * 1024
MAX_DECODED_BYTES = 49_990_000


def _require(condition, message):
    if not condition: raise ValueError(message)


def decode_audio(raw, options=None, check_cancelled=lambda: None,
                 maximum_decoded_bytes=MAX_DECODED_BYTES):
    _require(options is None or options == {} or options == {'format': 'pcmWav'},
             'Audio reading preserves channels/rate; choose preparation to transform samples.')
    _require(type(maximum_decoded_bytes) is int and 0 < maximum_decoded_bytes <= MAX_DECODED_BYTES,
             'Decoded audio byte budget is invalid.')
    _require(type(raw) is bytes and 44 <= len(raw) <= MAX_ENCODED_BYTES and
             raw[:4] == b'RIFF' and raw[8:12] == b'WAVE' and
             struct.unpack('<I', raw[4:8])[0] + 8 == len(raw),
             'Audio must be a complete bounded RIFF PCM WAV container.')
    check_cancelled()
    # Scan RIFF chunks first: the decoder must not accept a truncated data chunk,
    # ambiguous duplicate data/format chunk or an unbounded chunk walk.
    position = 12; seen = set(); chunks = 0; format_bytes = None; data_length = None
    while position < len(raw):
        check_cancelled(); chunks += 1
        _require(chunks <= 10000 and position+8 <= len(raw), 'Malformed or oversized WAV chunk inventory.')
        name, length = raw[position:position+4], struct.unpack('<I', raw[position+4:position+8])[0]
        end = position+8+length
        _require(end <= len(raw), 'WAV chunk is truncated.')
        if name in (b'fmt ', b'data'):
            _require(name not in seen, 'Duplicate WAV format/data chunk.')
            seen.add(name)
        if name == b'fmt ': format_bytes = raw[position+8:end]
        if name == b'data': data_length = length
        # A final odd data payload is emitted without a padding byte by common
        # PCM writers (including Python wave). Never skip a byte inside chunks.
        position = end if name == b'data' and end == len(raw) else end + length % 2
    _require(position == len(raw) and seen == {b'fmt ', b'data'}, 'Incomplete PCM WAV chunks.')
    _require(format_bytes is not None and len(format_bytes) in (16, 18),
             'Only standard integer PCM WAV format is qualified; extensible codecs are not yet supported.')
    encoding, channels, rate, byte_rate, alignment, bits = struct.unpack('<HHIIHH', format_bytes[:16])
    _require(encoding == 1 and bits in (8, 16, 24, 32) and channels > 0 and
             alignment == channels*(bits//8) and byte_rate == rate*alignment and
             data_length is not None and data_length > 0 and data_length % alignment == 0 and
             (len(format_bytes) == 16 or format_bytes[16:] == b'\0\0'),
             'Invalid or unsupported PCM WAV frame geometry.')
    try:
        with wave.open(BytesIO(raw), 'rb') as stream:
            channels, frames, rate, width = (stream.getnchannels(), stream.getnframes(),
                                            stream.getframerate(), stream.getsampwidth())
            _require(stream.getcomptype() == 'NONE' and width in (1, 2, 3, 4) and
                     channels > 0 and frames > 0 and 1 <= rate <= 768000,
                     'Only nonempty uncompressed 8/16/24/32-bit integer PCM WAV is supported.')
            # 24-bit PCM uses exact sign-extended int32 in the canonical artifact.
            output_width = 4 if width == 3 else width
            _require(frames <= maximum_decoded_bytes // output_width // channels,
                     'Decoded audio exceeds its byte budget; use a smaller source.')
            blocks = []; remaining = frames
            while remaining:
                check_cancelled(); take = min(remaining, max(1, 262144 // channels // width))
                block = stream.readframes(take)
                _require(len(block) == take*channels*width, 'PCM WAV sample payload is truncated.')
                if width == 3:
                    values = np.frombuffer(block, dtype=np.uint8).reshape(-1, 3)
                    signed = (values[:, 0].astype(np.int32) | values[:, 1].astype(np.int32) << 8 |
                              values[:, 2].astype(np.int32) << 16)
                    signed = (signed ^ 0x800000) - 0x800000
                    block = signed.astype('<i4', copy=False).tobytes()
                blocks.append(block); remaining -= take
            check_cancelled()
            return dict(samples=b''.join(blocks), shape=[frames, channels],
                        axes=['samples', 'channels'], dtype={1: 'uint8', 2: 'int16', 3: 'int32', 4: 'int32'}[width],
                        metadata={'sampleRate': rate, 'sourceBits': width*8})
    except (wave.Error, EOFError, struct.error) as error:
        raise ValueError('Unsupported or malformed PCM WAV encoding.') from error
