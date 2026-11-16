"""Path-free typed image cells and immutable bounded pixel containers.

No decoder or file authority lives here. Producers publish through a caller-owned
artifact sink; readers must independently pin the owned inventory before decoding.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import re
import struct
import os
import stat
from pathlib import Path

MAGIC=b'KWIPIMG1\n'
MAX_HEADER_BYTES=4096
MAX_ARTIFACT_BYTES=50_000_000
MAX_PIXELS=16_000_000
MAX_DIMENSION=16384
SIZES={'uint8':1,'uint16':2,'int32':4,'float32':4}
MODES={'L':(1,'uint8'),'LA':(2,'uint8'),'RGB':(3,'uint8'),'RGBA':(4,'uint8'),'CMYK':(4,'uint8'),'I;16':(1,'uint16'),'I':(1,'int32'),'F':(1,'float32')}
REF_KEYS={'schemaVersion','kind','sha256','width','height','channels','dtype','layout'}
HEADER_KEYS={'schemaVersion','kind','width','height','channels','dtype','layout','byteOrder','policy'}
_scope=ContextVar('owned_image_artifact_publisher',default=None)


def _json(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')


def geometry(width,height,channels,dtype,layout):
    if any(type(v) is not int for v in (width,height,channels)) or type(dtype) is not str or type(layout) is not str or not 1<=width<=MAX_DIMENSION or not 1<=height<=MAX_DIMENSION or not 1<=channels<=4 or width*height>MAX_PIXELS or dtype not in SIZES or layout not in ('HWC','CHW'):
        raise ValueError('Image geometry, dtype or layout is outside its bounded pixel contract.')
    size=width*height*channels*SIZES[dtype]
    if size>MAX_ARTIFACT_BYTES:raise ValueError('Decoded image exceeds its 50 MB artifact allowance.')
    return size


def validate_reference(value):
    if type(value) is not dict or set(value)!=REF_KEYS or type(value['schemaVersion']) is not int or value['schemaVersion']!=1 or value['kind']!='imagePixelsRef' or type(value['sha256']) is not str or re.fullmatch('[0-9a-f]{64}',value['sha256']) is None:
        raise ValueError('Invalid immutable image reference; paths and URLs are not image authority.')
    geometry(value['width'],value['height'],value['channels'],value['dtype'],value['layout'])
    return dict(value)


def is_reference(value):
    try:validate_reference(value);return True
    except (ValueError,TypeError):return False


def pixel_header(*,width,height,channels,dtype,layout,policy):
    size=geometry(width,height,channels,dtype,layout)
    if type(policy) is not dict or set(policy)!={'orientation','colorMode','palette','actualMode'} or any(type(value) is not str for value in policy.values()) or policy['orientation'] not in ('stored','applyEXIF') or policy['colorMode'] not in ('preserve','RGB','RGBA','L') or policy['palette'] not in ('expandRGBA','reject') or MODES.get(policy['actualMode'])!=(channels,dtype):
        raise ValueError('Image decoding policy or actual pixel mode is unsupported.')
    header={'schemaVersion':1,'kind':'imagePixels','width':width,'height':height,'channels':channels,'dtype':dtype,'layout':layout,'byteOrder':'little','policy':dict(policy)}
    encoded=_json(header)
    if len(encoded)>MAX_HEADER_BYTES or len(MAGIC)+4+len(encoded)+size>MAX_ARTIFACT_BYTES:
        raise ValueError('Decoded image including its header exceeds its 50 MB artifact allowance.')
    return header,encoded,size


def encode_pixels(pixels,*,width,height,channels,dtype,layout,policy):
    _,encoded,size=pixel_header(width=width,height=height,channels=channels,dtype=dtype,layout=layout,policy=policy)
    if not isinstance(pixels,(bytes,bytearray,memoryview)) or len(pixels)!=size:raise ValueError('Image pixel byte length differs from declared geometry.')
    content=MAGIC+struct.pack('>I',len(encoded))+encoded+bytes(pixels)
    reference={'schemaVersion':1,'kind':'imagePixelsRef','sha256':hashlib.sha256(content).hexdigest(),'width':width,'height':height,'channels':channels,'dtype':dtype,'layout':layout}
    return content,reference


def _pairs(items):
    result={}
    for key,value in items:
        if key in result:raise ValueError('Duplicate image container header field.')
        result[key]=value
    return result


def decode_pixels(content,expected_sha256=None):
    if type(content) is not bytes or not len(MAGIC)+4<len(content)<=MAX_ARTIFACT_BYTES or not content.startswith(MAGIC):raise ValueError('Unsupported image pixel container.')
    if expected_sha256 is not None and (type(expected_sha256) is not str or re.fullmatch('[0-9a-f]{64}',expected_sha256) is None or hashlib.sha256(content).hexdigest()!=expected_sha256):raise ValueError('Image pixel artifact differs from its immutable SHA-256 pin.')
    length=struct.unpack('>I',content[len(MAGIC):len(MAGIC)+4])[0];start=len(MAGIC)+4
    if not 1<=length<=MAX_HEADER_BYTES or start+length>len(content):raise ValueError('Image container header exceeds its byte bound.')
    try:header=json.loads(content[start:start+length].decode('ascii'),object_pairs_hook=_pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite header value')))
    except (UnicodeError,json.JSONDecodeError) as error:raise ValueError('Invalid image container header.') from error
    if type(header) is not dict or set(header)!=HEADER_KEYS or type(header['schemaVersion']) is not int or header['schemaVersion']!=1 or header['kind']!='imagePixels' or header['byteOrder']!='little':raise ValueError('Unsupported image container header.')
    _,encoded,size=pixel_header(**{key:header[key] for key in ('width','height','channels','dtype','layout','policy')})
    if encoded!=content[start:start+length] or len(content)-start-length!=size:raise ValueError('Image container geometry or canonical header differs from its pixels.')
    return header,content[start+length:]


@contextmanager
def artifact_scope(publisher):
    if not callable(publisher):raise ValueError('Image artifact publisher must be callable.')
    token=_scope.set((publisher,set()))
    try:yield
    finally:_scope.reset(token)


def publish_image_pixels(pixels,*,check_cancelled=lambda:None,**metadata):
    context=_scope.get()
    if context is None:raise ValueError('Image pixels require an owned artifact publisher.')
    check_cancelled();content,reference=encode_pixels(pixels,**metadata)
    return publish_encoded(content,reference,check_cancelled=check_cancelled)


def publish_encoded(content,reference,check_cancelled=lambda:None):
    context=_scope.get()
    if context is None:raise ValueError('Image pixels require an owned artifact publisher.')
    reference=validate_reference(reference);check_cancelled()
    header,pixels=decode_pixels(content,expected_sha256=reference['sha256'])
    if any(header[key]!=reference[key] for key in ('width','height','channels','dtype','layout')):raise ValueError('Image reference differs from encoded pixel geometry.')
    del pixels
    if reference['sha256'] not in context[1]:
        context[0](content,dict(reference));context[1].add(reference['sha256'])
    check_cancelled();return dict(reference)


publish_pixels=publish_image_pixels


def table_schema_version(columns):
    if any(column.get('dtype') in ('audio','tensor') for column in columns):return 5
    if any(column.get('dtype')=='int64Sequence' for column in columns):return 4
    if any(column.get('dtype') in ('date','datetime','time') for column in columns):return 3
    return 2 if any(column.get('dtype')=='image' for column in columns) else 1


def validate_table_version(version,columns):
    if type(version) is not int or version!=table_schema_version(columns):raise ValueError('Table schema version differs from its typed payload.')


def references(table):
    selected=[index for index,column in enumerate(table.columns) if column['dtype']=='image']
    if not selected:return
    for row in table.rows:
        for index in selected:
            if row[index] is not None:yield validate_reference(row[index])


def _image_columns(path,check,column_types=None):
    """Read only bounded leading metadata on canonical tables, not their rows."""
    try:descriptor=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    except OSError as error:raise ValueError('Image table artifact must be a readable regular file.') from error
    try:
        info=os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size>MAX_ARTIFACT_BYTES:raise ValueError('Image table must be a bounded regular artifact.')
        check();raw=os.read(descriptor,min(info.st_size,262144))
    finally:os.close(descriptor)
    try:
        text=raw.decode('utf-8');decoder=json.JSONDecoder();index=0
        def whitespace(i):
            while i<len(text) and text[i].isspace():i+=1
            return i
        index=whitespace(index)
        if text[index]!='{':return None
        index+=1
        while True:
            index=whitespace(index);key,index=decoder.raw_decode(text,index);index=whitespace(index)
            if text[index]!=':':return None
            index=whitespace(index+1)
            if key in ('rows','lineage'):return None
            value,index=decoder.raw_decode(text,index)
            if key=='columns':return any(type(column) is dict and column.get('dtype') in (column_types or {'image'}) for column in value) if type(value) is list else None
            index=whitespace(index)
            if text[index]!=',':return None
            index+=1
    except (ValueError,IndexError,UnicodeError):return None


def validate_inventory(manifest,root,check_cancelled=lambda:None):
    """Prove every typed reference resolves to a pinned geometry-bound asset.

    Scalar canonical tables need only leading metadata. Image tables stream their
    complete reference closure through the existing bounded table reader.
    """
    if __package__:
        from .dataflow_storage import bounded_regular_read,verify_regular_file
        from .dataflow_table_reader import read_table_artifact
    else:
        from dataflow_storage import bounded_regular_read,verify_regular_file
        from dataflow_table_reader import read_table_artifact
    root=Path(root);assets={}
    for entry in manifest['artifacts']:
        if entry.get('kind')!='imagePixels':continue
        check_cancelled();reference=validate_reference(entry.get('image'))
        if entry.get('sha256')!=reference['sha256'] or entry.get('path')!='image-'+reference['sha256']+'.kwipimg' or reference['sha256'] in assets:raise ValueError('Invalid or duplicate image artifact inventory.')
        content=bounded_regular_read(root/entry['path'],MAX_ARTIFACT_BYTES)
        if len(content)!=entry['bytes']:raise ValueError('Image artifact size differs from its inventory.')
        header,pixels=decode_pixels(content,expected_sha256=reference['sha256'])
        if any(header[key]!=reference[key] for key in ('width','height','channels','dtype','layout')):raise ValueError('Image inventory geometry differs from its pixel container.')
        assets[reference['sha256']]=reference
        del content,pixels
    for entry in manifest['artifacts']:
        if entry.get('kind')!='table':continue
        check_cancelled();path=root/entry['path']
        if _image_columns(path,check_cancelled) is False:continue
        verify_regular_file(path,entry['bytes'],entry['sha256'],check_cancelled)
        table=read_table_artifact(path,entry['sha256'],check_cancelled=check_cancelled)
        try:
            for reference in references(table):
                if assets.get(reference['sha256'])!=reference:raise ValueError('Image table reference is missing from its immutable artifact inventory.')
        finally:
            table.rows.store.close();table.row_ids.store.close()
    return assets
