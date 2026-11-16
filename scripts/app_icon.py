#!/usr/bin/env python3
"""Package approved PNG artwork as a macOS icon with system tools, offline.

Build helper only: no artwork generation, signing, app mutation or installation.
An existing destination is replaced atomically only after conversion succeeds.
"""
import argparse
import os
from pathlib import Path
import stat
import struct
import subprocess
import tempfile
import time
import zlib

MAX_INPUT_BYTES=16*1024*1024
MAX_OUTPUT_BYTES=16*1024*1024
ICON_SIZES=(16,32,128,256,512)


def png_dimensions(data):
    if len(data)>MAX_INPUT_BYTES or data[:8]!=b'\x89PNG\r\n\x1a\n':
        raise ValueError('Icon source must be a bounded valid PNG.')
    cursor=8;dimensions=None;saw_data=False;ended=False
    while cursor<len(data):
        if cursor+12>len(data):raise ValueError('Truncated PNG icon source.')
        length=struct.unpack('>I',data[cursor:cursor+4])[0]
        kind=data[cursor+4:cursor+8];stop=cursor+12+length
        if stop>len(data):raise ValueError('Truncated PNG icon chunk.')
        payload=data[cursor+8:cursor+8+length]
        crc=struct.unpack('>I',data[cursor+8+length:stop])[0]
        if zlib.crc32(kind+payload)&0xffffffff!=crc:raise ValueError('Invalid PNG icon checksum.')
        if dimensions is None:
            if kind!=b'IHDR' or length!=13:raise ValueError('PNG icon requires its header first.')
            dimensions=struct.unpack('>II',payload[:8])
        elif kind==b'IHDR':raise ValueError('Duplicate PNG icon header.')
        if kind==b'IDAT':saw_data=True
        if kind==b'IEND':
            if length or stop!=len(data):raise ValueError('Invalid PNG icon ending.')
            ended=True
        cursor=stop
    if not ended or not saw_data or dimensions is None or not all(0<n<=4096 for n in dimensions):
        raise ValueError('PNG icon must have complete bounded image data.')
    return dimensions


def _source_bytes(source):
    fd=os.open(source,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0)|getattr(os,'O_NONBLOCK',0))
    with os.fdopen(fd,'rb') as stream:
        info=os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0<info.st_size<=MAX_INPUT_BYTES:
            raise ValueError('Icon source must be a bounded regular PNG file.')
        data=stream.read(MAX_INPUT_BYTES+1)
    dimensions=png_dimensions(data)
    if dimensions[0]!=dimensions[1] or dimensions[0]<1024:
        raise ValueError('Icon source must be square and at least1024 pixels.')
    return data


def _destination_state(path):
    try:info=path.lstat()
    except FileNotFoundError:return None
    if not stat.S_ISREG(info.st_mode):raise ValueError('Icon destination must be a regular file or new path.')
    return (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns)


def build_icon(source, output):
    source=Path(source);output=Path(output)
    data=_source_bytes(source)
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise ValueError('Icon destination parent must be an existing directory.')
    previous=_destination_state(output)
    deadline=time.monotonic()+45
    def convert(arguments):
        remaining=deadline-time.monotonic()
        if remaining<=0:raise ValueError('Icon conversion exceeded its bounded time.')
        try:subprocess.run(arguments,check=True,capture_output=True,timeout=min(15,remaining))
        except (subprocess.SubprocessError,OSError) as error:
            raise ValueError('Icon conversion failed; the previous icon was preserved.') from error
    with tempfile.TemporaryDirectory(prefix='.kwipml-icon-',dir=output.parent) as temporary:
        staging=Path(temporary)
        snapshot=staging/'source.png';snapshot.write_bytes(data)
        iconset=staging/'KwipML.iconset';iconset.mkdir()
        for size in ICON_SIZES:
            for scale,suffix in ((1,''),(2,'@2x')):
                pixels=size*scale
                target=iconset/f'icon_{size}x{size}{suffix}.png'
                convert(['/usr/bin/sips','-z',str(pixels),str(pixels),str(snapshot),'--out',str(target)])
                if png_dimensions(target.read_bytes())!=(pixels,pixels):raise ValueError('Icon conversion produced incorrect dimensions.')
        converted=staging/'KwipML.icns'
        convert(['/usr/bin/iconutil','-c','icns',str(iconset),'-o',str(converted)])
        info=converted.lstat()
        if not stat.S_ISREG(info.st_mode) or not 8<info.st_size<=MAX_OUTPUT_BYTES:
            raise ValueError('Icon conversion produced invalid output.')
        content=converted.read_bytes()
        if content[:4]!=b'icns' or struct.unpack('>I',content[4:8])[0]!=len(content):
            raise ValueError('Icon conversion produced an invalid ICNS header.')
        cursor=8
        while cursor<len(content):
            if cursor+8>len(content):raise ValueError('Icon conversion produced truncated ICNS data.')
            length=struct.unpack('>I',content[cursor+4:cursor+8])[0]
            if length<8 or cursor+length>len(content):raise ValueError('Icon conversion produced invalid ICNS data.')
            cursor+=length
        if _destination_state(output)!=previous:
            raise ValueError('Icon destination changed during conversion; existing content was preserved.')
        os.chmod(converted,0o644)
        os.replace(converted,output)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    try:build_icon(args.source,args.output)
    except (ValueError,OSError) as error:parser.exit(1,str(error)+'\n')


if __name__=='__main__':main()
