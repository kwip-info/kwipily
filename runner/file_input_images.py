"""Bounded byte-only image decoding for captured File Input members.

No filesystem/network access or implicit resize, normalization or color conversion.
Frames are Pillow's displayed/composited frames, including GIF/APNG disposal.
"""
from array import array
from io import BytesIO
import math
import sys
from struct import error as struct_error
import warnings

from PIL import Image, ImageFile, UnidentifiedImageError

if __package__:
    from .image_artifacts import pixel_header
else:
    from image_artifacts import pixel_header

MAX_PIXELS = 16_000_000
MAX_DIMENSION = 16_384
MAX_FRAME_BYTES = 50_000_000
MAX_AGGREGATE_BYTES = 200_000_000
MAX_ENCODED_BYTES = 64 * 1024 * 1024
MAX_FRAMES = 1000
_FORMATS = {'PNG', 'JPEG', 'TIFF', 'BMP', 'GIF', 'WEBP'}
_TRANSPOSE = {2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180,
              4: Image.Transpose.FLIP_TOP_BOTTOM, 5: Image.Transpose.TRANSPOSE,
              6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE,
              8: Image.Transpose.ROTATE_90}
_TIFF_INVERSE = dict(_TRANSPOSE)
_TIFF_INVERSE[6] = Image.Transpose.ROTATE_90
_TIFF_INVERSE[8] = Image.Transpose.ROTATE_270
_DEFAULTS = {'frames': 'first', 'orientation': 'stored', 'colorMode': 'preserve',
             'palette': 'expandRGBA', 'layout': 'HWC', 'frameStart': 0, 'frameEnd': 0}
_MODES = {'L': ('uint8', 1, 1), 'LA': ('uint8', 2, 1), 'RGB': ('uint8', 3, 1),
          'RGBA': ('uint8', 4, 1), 'CMYK': ('uint8', 4, 1),
          'I;16': ('uint16', 1, 2), 'I;16L': ('uint16', 1, 2),
          'I;16B': ('uint16', 1, 2), 'I;16N': ('uint16', 1, 2),
          'I': ('int32', 1, 4), 'F': ('float32', 1, 4)}


class ImageDecodeError(ValueError):
    """Actionable bounded decoder refusal; never contains input contents or paths."""


def _require(condition, message):
    if not condition:
        raise ImageDecodeError(message)


def normalize_options(options):
    _require(type(options) is dict and not (set(options) - set(_DEFAULTS) - {'frameStart', 'frameEnd'}),
             'Image options contain an unsupported field.')
    result = dict(_DEFAULTS, **options)
    for name, choices in (('frames', {'first', 'all', 'range'}),
                          ('orientation', {'stored', 'applyEXIF'}),
                          ('colorMode', {'preserve', 'RGB', 'RGBA', 'L'}),
                          ('palette', {'expandRGBA', 'reject'}), ('layout', {'HWC', 'CHW'})):
        _require(type(result[name]) is str and result[name] in choices, 'Invalid image ' + name + ' option.')
    start, end = result['frameStart'], result['frameEnd']
    _require(type(start) is int and type(end) is int and 0 <= start < MAX_FRAMES and 0 <= end < MAX_FRAMES,
             'Image frame indices must be zero-based integers below 1000.')
    if result['frames'] == 'range':
        _require(start <= end, 'Image frame range must have ordered zero-based indices below 1000.')
    # Dormant validated bounds are retained by the inspector when switching its
    # frame policy. They have no effect on first/all selection.
    return result


def _geometry(image, mode):
    width, height = image.size
    _require(type(width) is int and type(height) is int and 0 < width <= MAX_DIMENSION
             and 0 < height <= MAX_DIMENSION and width * height <= MAX_PIXELS,
             'Image geometry exceeds the 16 megapixel / 16384 dimension limit.')
    _require(mode in _MODES, 'Image mode ' + str(mode) + ' cannot be preserved; choose RGB, RGBA or L explicitly.')
    dtype, channels, itemsize = _MODES[mode]
    size = width * height * channels * itemsize
    _require(size <= MAX_FRAME_BYTES, 'Decoded image bytes exceed the 50 MB frame limit.')
    return width, height, channels, dtype, size


def _decoded_mode(image, options):
    if options['colorMode'] != 'preserve':
        return options['colorMode']
    if image.mode == 'P':
        _require(options['palette'] == 'expandRGBA', 'Image palette is rejected; choose Expand RGBA to decode it.')
        return 'RGBA'
    return image.mode


def _source_precision(image, raw):
    if image.format == 'PNG':
        return raw[24] if len(raw) > 24 else 8
    if image.format == 'TIFF':
        bits = image.tag_v2.get(258, (8,))
        bits = (bits,) if type(bits) is int else bits
        _require(isinstance(bits, (tuple, list)) and bits and all(type(value) is int and 1 <= value <= 64 for value in bits),
                 'TIFF sample precision is invalid.')
        return max(bits)
    return _MODES.get(image.mode, ('uint8', 1, 1))[2] * 8


def _check_precision(image, raw, options):
    precision = _source_precision(image, raw)
    native_width = _MODES.get(image.mode, ('uint8', 1, 1))[2] * 8
    _require(options['colorMode'] != 'preserve' or precision <= native_width,
             'This multichannel image precision cannot be preserved by the decoder; choose RGB, RGBA or L explicitly to accept uint8 conversion.')
    return precision


def _little_endian(pixels, mode):
    swap = mode == 'I;16B' or (sys.byteorder == 'big' and mode in {'I;16N', 'I', 'F'})
    if not swap:
        return pixels
    values = array('H' if mode.startswith('I;16') else 'I')
    values.frombytes(pixels); values.byteswap()
    return values.tobytes()


def _chw(pixels, channels, itemsize, check):
    if channels == 1:
        return pixels
    # Current multichannel preserved/converted modes are all uint8.
    _require(itemsize == 1, 'Multichannel image byte layout is unsupported.')
    planes = []
    for channel in range(channels):
        check()
        planes.append(pixels[channel::channels])
    return b''.join(planes)


def decode_images(raw, options, check=lambda: None):
    """Yield selected frame dictionaries, with immutable raw little-endian pixels.

    Options are strict and normalized before opening bytes. ``policy`` is the
    pixel-container policy, including actual decoded color mode. ``notices`` are
    structured conversion disclosures for the source adapter's diagnostics.
    ``duration`` is frame duration in milliseconds, or None if absent.
    """
    options = normalize_options(options)
    _require(type(raw) is bytes and 0 < len(raw) <= MAX_ENCODED_BYTES,
             'Encoded image must be nonempty and at most 64 MiB.')
    check()
    _require(not ImageFile.LOAD_TRUNCATED_IMAGES, 'Strict image decoding is unavailable while truncated-image loading is enabled.')
    image = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            image = Image.open(BytesIO(raw))
            _require(image.format in _FORMATS, 'Unsupported image format; choose PNG, JPEG, TIFF, BMP, GIF or WebP.')
            # Header geometry is checked before a decoder is asked for pixels.
            _geometry(image, _decoded_mode(image, options))
            _check_precision(image, raw, options)
            # Pillow may load a PNG whose final IEND CRC is truncated. Admit a
            # complete container before any selected frame's pixel allocation.
            if image.format == 'PNG':
                _require(raw.endswith(b'\x00\x00\x00\x00IEND\xaeB`\x82'), 'PNG data is truncated or lacks its complete end marker.')
            elif image.format == 'JPEG':
                _require(raw.endswith(b'\xff\xd9'), 'JPEG data is truncated or lacks its complete end marker.')
            elif image.format == 'GIF':
                _require(raw.endswith(b';'), 'GIF data is truncated or lacks its complete end marker.')
            elif image.format == 'WEBP':
                _require(len(raw) >= 12 and int.from_bytes(raw[4:8], 'little') + 8 == len(raw), 'WebP container length differs from its captured bytes.')
            image.verify()
            image.close()
            image = Image.open(BytesIO(raw))
            check()
            count = getattr(image, 'n_frames', 1)
            _require(type(count) is int and 1 <= count <= MAX_FRAMES,
                     'Image frame count exceeds the 1000-frame limit.')
            check()
            start, end = (options['frameStart'], options['frameEnd']) if options['frames'] == 'range' else (0, count - 1 if options['frames'] == 'all' else 0)
            _require(end < count, 'Selected image frame range extends beyond the available frames.')
            aggregate = 0
            for index in range(end + 1):
                check()
                image.seek(index)
                mode = _decoded_mode(image, options)
                geometry = _geometry(image, mode)
                source_precision = _check_precision(image, raw, options)
                if index < start:
                    continue
                width, height, channels, dtype, size = geometry
                _require(aggregate + size <= MAX_AGGREGATE_BYTES,
                         'Decoded image aggregate exceeds the 200 MB selected-frame limit.')
                source_mode = image.mode
                actual_mode = 'I;16' if mode.startswith('I;16') else mode
                policy = {'orientation': options['orientation'], 'colorMode': options['colorMode'],
                          'palette': options['palette'], 'actualMode': actual_mode}
                try:
                    pixel_header(width=width, height=height, channels=channels, dtype=dtype,
                                 layout=options['layout'], policy=policy)
                except ValueError as error:
                    raise ImageDecodeError('Decoded frame exceeds its bounded pixel artifact contract.') from error
                original_orientation = image.getexif().get(274)
                _require(original_orientation is None or (type(original_orientation) is int and 1 <= original_orientation <= 8),
                         'Image EXIF orientation is invalid.')
                image.load()
                check()
                working = image
                notices = []
                try:
                    # Pillow TIFF applies EXIF orientation during load, even
                    # when no transpose was requested. Undo that decoder action
                    # for stored pixels; applyEXIF must not transpose twice.
                    transform = (_TIFF_INVERSE.get(original_orientation) if image.format == 'TIFF'
                                 and options['orientation'] == 'stored' else
                                 _TRANSPOSE.get(original_orientation) if image.format != 'TIFF'
                                 and options['orientation'] == 'applyEXIF' else None)
                    if transform is not None:
                        working = image.transpose(transform)
                    if mode != working.mode:
                        converted = working.convert(mode)
                        if working is not image:
                            working.close()
                        working = converted
                        loses_depth = source_precision > 8
                        notices.append({'code': 'imageColorConversion', 'sourceMode': source_mode,
                                        'actualMode': mode, 'changesBitDepth': loses_depth,
                                        'message': 'Explicit color conversion changes bit depth to uint8.' if loses_depth else 'Selected color policy converts the decoded image.'})
                    if source_precision > 8 and dtype == 'uint8' and not any(item['changesBitDepth'] for item in notices):
                        notices.append({'code': 'imageColorConversion', 'sourceMode': source_mode,
                                        'actualMode': mode, 'changesBitDepth': True,
                                        'message': 'Explicit color conversion changes bit depth to uint8.'})
                    orientation = original_orientation if options['orientation'] == 'applyEXIF' else None
                    width, height, channels, dtype, size = _geometry(working, mode)
                    check()
                    pixels = _little_endian(working.tobytes(), mode)
                    _require(len(pixels) == size, 'Decoded pixel bytes do not match the image geometry.')
                    if options['layout'] == 'CHW':
                        pixels = _chw(pixels, channels, _MODES[mode][2], check)
                    check()
                    duration = image.info.get('duration')
                    _require(duration is None or (type(duration) in (int, float) and math.isfinite(duration) and duration >= 0),
                             'Image frame duration is invalid.')
                    frame = {'pixels': pixels, 'width': width, 'height': height, 'channels': channels,
                             'dtype': dtype, 'layout': options['layout'], 'frame_index': index,
                             'duration': duration, 'orientation_applied': orientation,
                             'policy': policy,
                             'notices': notices}
                    aggregate += size
                finally:
                    if working is not image:
                        working.close()
                yield frame
    except (UnidentifiedImageError, Image.DecompressionBombWarning, Image.DecompressionBombError,
            OSError, EOFError, SyntaxError, struct_error) as error:
        raise ImageDecodeError('Image data is invalid, truncated or unsafe to decode.') from error
    finally:
        if image is not None:
            image.close()
