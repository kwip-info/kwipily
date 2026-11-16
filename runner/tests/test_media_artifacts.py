import hashlib
import json
import struct

import pytest

from runner import media_artifacts as media


def audio_metadata():
    return dict(shape=[3, 2], axes=['samples', 'channels'], dtype='int16',
                metadata={'sampleRate': 48000, 'sourceBits': 16})


def test_audio_container_pins_exact_samples_geometry_rate_and_precision():
    samples = struct.pack('<6h', -32768, 32767, 0, 123, -123, 1)
    content, ref = media.encode_asset(samples, kind='audio', **audio_metadata())
    assert ref['sha256'] == hashlib.sha256(content).hexdigest()
    header, restored = media.decode_asset(content, expected=ref)
    assert restored == samples
    assert header['metadata'] == {'sampleRate': 48000, 'sourceBits': 16}
    assert media.encode_asset(samples, kind='audio', **{**audio_metadata(),
        'metadata': {'sampleRate': 16000, 'sourceBits': 16}})[1] != ref


def test_tensor_binds_axes_and_preparation_to_parent_without_a_path():
    metadata = dict(parentSHA256='a' * 64, recipeSHA256='b' * 64,
                    sourceStart=0, sourceEnd=3, sourceRate=48000, validSamples=1)
    content, ref = media.encode_asset(struct.pack('<f', .25), kind='tensor',
        shape=[1, 1], axes=['channels', 'samples'], dtype='float32', metadata=metadata)
    assert media.decode_asset(content, expected=ref)[1] == struct.pack('<f', .25)
    for update in ({'path': '/private/file'}, {'shape': [True, 1]},
                   {'dtype': 'object'}, {'axes': ['samples', 'samples']}):
        with pytest.raises(ValueError):
            media.validate_reference({**ref, **update})


def test_rejects_malformed_tampered_nonfinite_and_oversized_assets():
    content, ref = media.encode_asset(bytes(12), kind='audio', **audio_metadata())
    with pytest.raises(ValueError):
        media.decode_asset(content[:-1] + b'x', expected=ref)
    with pytest.raises(ValueError):
        media.encode_asset(bytes(11), kind='audio', **audio_metadata())
    with pytest.raises(ValueError):
        media.encode_asset(b'', kind='tensor', shape=[10**20], axes=['samples'],
                           dtype='float32', metadata={})
    with pytest.raises(ValueError):
        media.encode_asset(struct.pack('<f', float('nan')), kind='tensor',
                           shape=[1], axes=['samples'], dtype='float32', metadata={})
    length = struct.unpack('>I', content[len(media.MAGIC):len(media.MAGIC)+4])[0]
    header = json.loads(content[len(media.MAGIC)+4:len(media.MAGIC)+4+length])
    header['metadata']['path'] = '/private/file'
    raw = json.dumps(header).encode()
    with pytest.raises(ValueError):
        media.decode_asset(media.MAGIC + struct.pack('>I', len(raw)) + raw + bytes(12))


def test_publication_requires_owned_scope_and_deduplicates():
    content, ref = media.encode_asset(bytes(12), kind='audio', **audio_metadata())
    with pytest.raises(ValueError, match='publisher'):
        media.publish_encoded(content, ref)
    published = []
    with media.artifact_scope(lambda data, ref: published.append((data, ref))):
        assert media.publish_encoded(content, ref) == ref
        assert media.publish_encoded(content, ref) == ref
    assert len(published) == 1


def test_asset_reader_requires_exact_published_reference():
    content, ref = media.encode_asset(bytes(12), kind='audio', **audio_metadata())
    with pytest.raises(ValueError, match='reader'):
        media.read_asset(ref)
    with media.artifact_scope(lambda *_: None, reader=lambda _: content):
        assert media.read_asset(ref)[1] == bytes(12)

@pytest.mark.parametrize('bits,dtype',[(8,'uint8'),(16,'int16'),(24,'int32'),(32,'int32')])
def test_prepared_integer_precision_and_mask_are_versioned_and_validated(bits,dtype):
 size=media.SIZES[dtype]
 metadata=dict(sampleRate=16000,sourceBits=bits,validSamples=2,sourceStart=0,sourceEnd=2,sourceRate=16000)
 raw,ref=media.encode_asset(bytes(4*size),kind='tensor',shape=[1,4],axes=['channels','samples'],dtype=dtype,metadata=metadata)
 assert media.decode_asset(raw,expected=ref)[0]['metadata']['sourceBits']==bits
 mask,_=media.encode_asset(bytes(32),kind='tensor',shape=[4],axes=['samples'],dtype='int64',metadata=dict(metadata,maskConvention='valid'))
 assert mask
 for update in [{'sourceBits':12},{'sourceBits':True},{'sourceBits':16 if dtype!='int16' else 8}]:
  with pytest.raises(ValueError):media.encode_asset(bytes(4*size),kind='tensor',shape=[1,4],axes=['channels','samples'],dtype=dtype,metadata=dict(metadata,**update))
 legacy=dict(metadata);legacy.pop('sourceBits')
 # Historical captures remain readable; preparation separately refuses unknown normalization precision.
 assert media.encode_asset(bytes(4*size),kind='tensor',shape=[1,4],axes=['channels','samples'],dtype=dtype,metadata=legacy)
