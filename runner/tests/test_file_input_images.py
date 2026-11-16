"""Byte-only image reader: pixels, explicit policy, and allocation guards."""
import io
import struct
from unittest.mock import patch

import pytest
from PIL import Image

from runner.file_input_images import ImageDecodeError, decode_images


def encoded(image, format, **options):
    stream = io.BytesIO()
    image.save(stream, format=format, **options)
    return stream.getvalue()


def frames(raw, **options):
    return list(decode_images(raw, options, lambda: None))


@pytest.mark.parametrize('format', ['PNG', 'JPEG', 'TIFF', 'BMP', 'GIF', 'WEBP'])
def test_supported_formats_have_typed_raw_pixels(format):
    image = Image.new('RGB', (3, 2), (71, 121, 211))
    result = frames(encoded(image, format))[0]
    assert (result['width'], result['height'], result['channels'], result['dtype'], result['layout']) == (3, 2, 4 if format == 'GIF' else 3, 'uint8', 'HWC')
    assert len(result['pixels']) == 3 * 2 * result['channels']
    assert result['frame_index'] == 0
    if format not in ('JPEG', 'WEBP'):
        assert result['pixels'][:3] == bytes((71, 121, 211))


@pytest.mark.parametrize('format', ['PNG', 'TIFF'])
def test_sixteen_bit_values_preserved_little_endian(format):
    image = Image.frombytes('I;16', (3, 1), struct.pack('<3H', 1, 1025, 65535))
    result = frames(encoded(image, format))[0]
    assert result['dtype'] == 'uint16'
    assert result['channels'] == 1
    assert result['pixels'] == struct.pack('<3H', 1, 1025, 65535)
    assert result['notices'] == []


@pytest.mark.parametrize('mode,dtype,values,packing', [('I', 'int32', [-1000, 3, 100000], 'i'), ('F', 'float32', [-0.5, 1.25, 10.0], 'f')])
def test_signed_and_float_tiff_preserved(mode, dtype, values, packing):
    image = Image.new(mode, (3, 1)); image.putdata(values)
    result = frames(encoded(image, 'TIFF'))[0]
    assert result['dtype'] == dtype
    assert result['pixels'] == struct.pack('<3' + packing, *values)


def test_big_endian_tiff_normalizes_only_byte_order():
    image = Image.frombytes('I;16B', (2, 1), struct.pack('>2H', 1025, 65534))
    result = frames(encoded(image, 'TIFF'))[0]
    assert result['dtype'] == 'uint16'
    assert result['pixels'] == struct.pack('<2H', 1025, 65534)


def test_explicit_conversion_reports_bit_depth_change_and_cmyk_preserved():
    image = Image.frombytes('I;16', (2, 1), struct.pack('<2H', 0, 1025))
    result = frames(encoded(image, 'PNG'), colorMode='RGB')[0]
    assert result['dtype'] == 'uint8' and result['channels'] == 3
    assert any(notice['changesBitDepth'] for notice in result['notices'])
    raw = encoded(Image.new('CMYK', (2, 1)), 'JPEG')
    preserved = frames(raw)[0]
    assert preserved['channels'] == 4 and preserved['policy']['actualMode'] == 'CMYK'
    assert frames(raw, colorMode='RGB')[0]['channels'] == 3


def test_alpha_palette_and_chw_are_explicit():
    image = Image.frombytes('RGBA', (2, 1), bytes((1, 2, 3, 4, 5, 6, 7, 8)))
    raw = encoded(image, 'PNG')
    assert frames(raw)[0]['pixels'] == bytes((1, 2, 3, 4, 5, 6, 7, 8))
    assert frames(raw, layout='CHW')[0]['pixels'] == bytes((1, 5, 2, 6, 3, 7, 4, 8))
    palette = Image.new('P', (2, 1)); palette.putpalette([255, 0, 0, 0, 255, 0] + [0] * 762)
    palette.putdata([0, 1]); palette.info['transparency'] = 1
    palraw = encoded(palette, 'PNG')
    assert frames(palraw)[0]['pixels'] == bytes((255, 0, 0, 255, 0, 255, 0, 0))
    with pytest.raises(ImageDecodeError, match='palette'):
        frames(palraw, palette='reject')


def test_orientation_is_stored_unless_requested():
    image = Image.new('RGB', (2, 3), (30, 50, 90)); exif = Image.Exif(); exif[274] = 6
    raw = encoded(image, 'JPEG', exif=exif)
    assert (frames(raw)[0]['width'], frames(raw)[0]['height']) == (2, 3)
    applied = frames(raw, orientation='applyEXIF')[0]
    assert (applied['width'], applied['height']) == (3, 2)
    assert applied['orientation_applied'] == 6


@pytest.mark.parametrize('format', ['GIF', 'TIFF', 'WEBP', 'PNG'])
def test_first_all_range_frames_duration_and_composited_pixels(format):
    images = [Image.new('RGB', (2, 2), color) for color in ('red', 'green', 'blue')]
    raw = encoded(images[0], format, save_all=True, append_images=images[1:], duration=[70, 90, 110], loop=0, lossless=True)
    assert [f['frame_index'] for f in frames(raw)] == [0]
    all_frames = frames(raw, frames='all')
    assert [f['frame_index'] for f in all_frames] == [0, 1, 2]
    assert all_frames[2]['pixels'][:3] == bytes((0, 0, 255))
    assert [f['frame_index'] for f in frames(raw, frames='range', frameStart=1, frameEnd=2)] == [1, 2]
    with pytest.raises(ImageDecodeError, match='range'):
        frames(raw, frames='range', frameStart=1, frameEnd=3)


def test_gif_disposal_restores_background_before_next_frame():
    base = Image.new('RGBA', (3, 1), (0, 0, 0, 0)); first = base.copy(); first.putpixel((0, 0), (255, 0, 0, 255))
    second = base.copy(); second.putpixel((1, 0), (0, 255, 0, 255))
    third = base.copy(); third.putpixel((2, 0), (0, 0, 255, 255))
    raw = encoded(first, 'GIF', save_all=True, append_images=[second, third], duration=80, disposal=[2, 2, 2], transparency=0, optimize=False)
    result = frames(raw, frames='all')
    assert result[2]['pixels'][3] == 0
    assert result[2]['pixels'][-4:] == bytes((0, 0, 255, 255))


@pytest.mark.parametrize('options', [{'bad': True}, {'frames': 'random'}, {'frameStart': True}, {'frames': 'range', 'frameStart': 2, 'frameEnd': 1}, {'frames': 'range', 'frameStart': True, 'frameEnd': 2}, {'orientation': 'auto'}, {'layout': 'WHC'}, {'colorMode': 'normalize'}, {'palette': 'keep'}])
def test_invalid_options_refused_before_open(options):
    with patch('runner.file_input_images.Image.open') as opened:
        with pytest.raises(ImageDecodeError):
            frames(b'invalid', **options)
        opened.assert_not_called()


def test_invalid_and_truncated_data_refused():
    with pytest.raises(ImageDecodeError):
        frames(b'not an image')
    raw = encoded(Image.new('RGB', (30, 30)), 'PNG')
    with pytest.raises(ImageDecodeError):
        frames(raw[:50])


def test_geometry_and_bytes_refused_before_pixel_load():
    raw = encoded(Image.new('RGB', (4, 4)), 'PNG')
    with patch('runner.file_input_images.MAX_PIXELS', 15), patch('PIL.Image.Image.load') as load:
        with pytest.raises(ImageDecodeError, match='geometry'):
            frames(raw)
        load.assert_not_called()
    with patch('runner.file_input_images.MAX_FRAME_BYTES', 47), patch('PIL.Image.Image.load') as load:
        with pytest.raises(ImageDecodeError, match='bytes'):
            frames(raw)
        load.assert_not_called()


def test_aggregate_limit_and_cancellation_between_frames():
    raw = encoded(Image.new('RGB', (4, 4), 'red'), 'TIFF', save_all=True, append_images=[Image.new('RGB', (4, 4), 'blue')])
    iterator = decode_images(raw, {'frames': 'all'}, lambda: None)
    with patch('runner.file_input_images.MAX_AGGREGATE_BYTES', 60):
        assert next(iterator)['frame_index'] == 0
        with pytest.raises(ImageDecodeError, match='aggregate'):
            next(iterator)
    class Cancelled(Exception): pass
    cancelled = False
    def check():
        if cancelled: raise Cancelled()
    iterator = decode_images(raw, {'frames': 'all'}, check)
    assert next(iterator)['frame_index'] == 0
    cancelled = True
    with pytest.raises(Cancelled): next(iterator)


def test_truncated_final_png_marker_and_permissive_host_policy_refused():
    from PIL import ImageFile
    raw = encoded(Image.new('RGB', (4, 4)), 'PNG')
    for missing in (1, 8, 12):
        with pytest.raises(ImageDecodeError, match='truncated'):
            frames(raw[:-missing])
    with patch.object(ImageFile, 'LOAD_TRUNCATED_IMAGES', True):
        with pytest.raises(ImageDecodeError, match='Strict image'):
            frames(raw)


def test_la_preserves_alpha_and_container_admission_precedes_load():
    image = Image.frombytes('LA', (2, 1), bytes((55, 70, 90, 120)))
    result = frames(encoded(image, 'PNG'))[0]
    assert result['pixels'] == bytes((55, 70, 90, 120))
    assert result['channels'] == 2 and result['policy']['actualMode'] == 'LA'
    raw = encoded(image, 'PNG')
    with patch('runner.file_input_images.pixel_header', side_effect=ValueError('too large')), patch('PIL.Image.Image.load') as load:
        with pytest.raises(ImageDecodeError, match='artifact'):
            frames(raw)
        load.assert_not_called()


def test_range_outside_available_frames_refused_without_pixel_decode():
    raw = encoded(Image.new('RGB', (4, 4)), 'PNG')
    with patch('PIL.Image.Image.load') as load:
        with pytest.raises(ImageDecodeError, match='range'):
            frames(raw, frames='range', frameStart=0, frameEnd=1)
        load.assert_not_called()


def test_dormant_frame_bounds_are_retained_but_never_change_first_or_all():
    raw = encoded(Image.new('RGB', (2, 2), 'red'), 'TIFF', save_all=True,
                  append_images=[Image.new('RGB', (2, 2), 'blue')])
    assert [item['frame_index'] for item in frames(raw, frameStart=999, frameEnd=100)] == [0]
    assert [item['frame_index'] for item in frames(raw, frames='all', frameStart=999, frameEnd=100)] == [0, 1]
    for options in ({'frameStart': -1}, {'frameEnd': 1000}, {'frameStart': True}):
        with pytest.raises(ImageDecodeError): frames(raw, **options)


def test_one_bit_preserve_requires_an_explicit_channel_conversion():
    raw = encoded(Image.new('1', (2, 2), 1), 'PNG')
    with pytest.raises(ImageDecodeError, match='choose RGB, RGBA or L explicitly'):
        frames(raw)
    result = frames(raw, colorMode='L')[0]
    assert result['pixels'] == bytes([255] * 4)
    assert result['policy']['actualMode'] == 'L' and result['notices']


def test_oversized_png_header_refused_without_decoding_or_allocating_pixels():
    import zlib
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    raw = (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 16001, 1001, 8, 2, 0, 0, 0))
           + chunk(b'IDAT', b'') + chunk(b'IEND', b''))
    with patch('PIL.Image.Image.load') as load:
        with pytest.raises(ImageDecodeError, match='geometry'):
            frames(raw)
        load.assert_not_called()


@pytest.mark.parametrize('orientation', [2, 3, 4, 5, 6, 7, 8])
def test_tiff_pillow_implicit_orientation_is_undone_for_stored_pixels(orientation):
    from PIL import TiffImagePlugin, ImageOps
    image = Image.frombytes('L', (2, 3), bytes([1, 2, 3, 4, 5, 6]))
    tags = TiffImagePlugin.ImageFileDirectory_v2(); tags[274] = orientation
    raw = encoded(image, 'TIFF', tiffinfo=tags)
    stored = frames(raw)[0]
    assert (stored['width'], stored['height'], stored['pixels']) == (2, 3, bytes([1, 2, 3, 4, 5, 6]))
    applied = frames(raw, orientation='applyEXIF')[0]
    image.getexif()[274] = orientation
    expected = ImageOps.exif_transpose(image)
    assert (applied['width'], applied['height'], applied['pixels']) == (expected.width, expected.height, expected.tobytes())
    assert applied['orientation_applied'] == orientation


@pytest.mark.parametrize('format', ['PNG', 'TIFF'])
def test_multichannel_sixteen_bit_cannot_silently_become_rgb_eight_bit(format):
    if format == 'PNG':
        import zlib
        def chunk(kind, data):
            return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
        raw = (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 2, 1, 16, 2, 0, 0, 0))
               + chunk(b'IDAT', zlib.compress(b'\x00' + struct.pack('>6H', 1, 1025, 65535, 2, 4000, 3000))) + chunk(b'IEND', b''))
    else:
        offset = 8 + 2 + 10 * 12 + 4
        tags = [(256,4,1,2), (257,4,1,1), (258,3,3,offset), (259,3,1,1), (262,3,1,2),
                (273,4,1,offset+6), (277,3,1,3), (278,4,1,1), (279,4,1,12), (284,3,1,1)]
        raw = (b'II' + struct.pack('<HIH', 42, 8, len(tags))
               + b''.join(struct.pack('<HHII', *tag) for tag in tags) + struct.pack('<I', 0)
               + struct.pack('<3H', 16, 16, 16) + struct.pack('<6H', 1, 1025, 65535, 2, 4000, 3000))
    with patch('PIL.Image.Image.load') as load:
        with pytest.raises(ImageDecodeError, match='precision'):
            frames(raw)
        load.assert_not_called()
    result = frames(raw, colorMode='RGB')[0]
    assert result['dtype'] == 'uint8'
    assert any(notice['changesBitDepth'] for notice in result['notices'])
