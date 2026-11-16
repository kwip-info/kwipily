import hashlib
import json
import struct
import pytest
from runner import image_artifacts as ia


def options(dtype='uint8',channels=3):
    mode={'uint8':{1:'L',2:'LA',3:'RGB',4:'RGBA'}[channels],'uint16':'I;16','int32':'I','float32':'F'}[dtype]
    return dict(width=2,height=3,channels=channels,dtype=dtype,layout='HWC',policy={'orientation':'stored','colorMode':'preserve','palette':'expandRGBA','actualMode':mode})


@pytest.mark.parametrize('dtype,size,channels',[('uint8',1,3),('uint16',2,1),('int32',4,1),('float32',4,1)])
@pytest.mark.parametrize('layout',['HWC','CHW'])
def test_pixel_container_binds_geometry_dtype_layout_policy_and_exact_bytes(dtype,size,channels,layout):
    settings=options(dtype,channels);settings['layout']=layout;pixels=bytes(range(6*channels*size))
    content,reference=ia.encode_pixels(pixels,**settings)
    assert reference=={'schemaVersion':1,'kind':'imagePixelsRef','sha256':hashlib.sha256(content).hexdigest(),'width':2,'height':3,'channels':channels,'dtype':dtype,'layout':layout}
    header,restored=ia.decode_pixels(content,expected_sha256=reference['sha256'])
    assert restored==pixels and header['policy']==settings['policy'] and header['byteOrder']=='little'
    changed={**settings,'width':3,'height':2}
    assert ia.encode_pixels(pixels,**changed)[1]['sha256']!=reference['sha256']


def test_same_rgba_and_cmyk_bytes_have_distinct_interpretation_identity():
    settings=options(channels=4);pixels=bytes(24)
    _,rgba=ia.encode_pixels(pixels,**settings)
    settings['policy']['actualMode']='CMYK'
    _,cmyk=ia.encode_pixels(pixels,**settings)
    assert rgba['sha256']!=cmyk['sha256']


def test_refuses_unbounded_private_reference_and_mismatched_or_tampered_container():
    content,reference=ia.encode_pixels(bytes(18),**options())
    for change in ({'path':'/tmp/image'},{'sha256':'A'*64},{'width':True},{'channels':5},{'layout':'HW'}):
        with pytest.raises(ValueError):ia.validate_reference({**reference,**change})
    with pytest.raises(ValueError):ia.decode_pixels(content[:-1]+b'1',expected_sha256=reference['sha256'])
    with pytest.raises(ValueError):ia.encode_pixels(bytes(17),**options())
    with pytest.raises(ValueError):ia.encode_pixels(b'',**{**options(),'width':16385})
    with pytest.raises(ValueError):ia.encode_pixels(bytes(18),**{**options(),'policy':{**options()['policy'],'path':'secret'}})
    length=struct.unpack('>I',content[len(ia.MAGIC):len(ia.MAGIC)+4])[0]
    header=json.loads(content[len(ia.MAGIC)+4:len(ia.MAGIC)+4+length]);header['width']=3
    altered=json.dumps(header,sort_keys=True,separators=(',',':')).encode()
    with pytest.raises(ValueError):ia.decode_pixels(ia.MAGIC+struct.pack('>I',len(altered))+altered+bytes(18))


def test_publication_requires_owned_publisher_and_deduplicates_exact_asset():
    with pytest.raises(ValueError,match='publisher'):ia.publish_image_pixels(bytes(18),**options())
    output=[]
    with ia.artifact_scope(lambda content,reference:output.append((content,reference))):
        first=ia.publish_image_pixels(bytes(18),**options())
        second=ia.publish_image_pixels(bytes(18),**options())
    assert first==second and len(output)==1
    assert ia.decode_pixels(output[0][0],expected_sha256=first['sha256'])[1]==bytes(18)
