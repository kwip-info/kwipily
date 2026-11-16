"""Real macOS icon packaging and refusal/preservation controls, no app build/signing."""
import importlib.util
from pathlib import Path
import subprocess
import struct
import zlib
import sys
import pytest

ROOT=Path(__file__).resolve().parents[2]
ASSET=ROOT/'Sources/KwipMLApp/Resources/KwipMLWelcomeIcon.png'

def tool():
    spec=importlib.util.spec_from_file_location('app_icon_tool',ROOT/'scripts/app_icon.py')
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

def test_approved_asset_dimensions_and_actual_iconutil_conversion(tmp_path):
    value=tool()
    assert value.png_dimensions(ASSET.read_bytes()) == (1254,1254)
    output=tmp_path/'KwipML.icns';output.write_bytes(b'previous icon');value.build_icon(ASSET,output)
    assert output.read_bytes()[:4] == b'icns'
    unpacked=tmp_path/'decoded.iconset'
    subprocess.run(['/usr/bin/iconutil','-c','iconset',str(output),'-o',str(unpacked)],check=True,capture_output=True,timeout=20)
    expected={f'icon_{size}x{size}{suffix}.png':(size*scale,size*scale) for size in (16,32,128,256,512) for suffix,scale in (('',1),('@2x',2))}
    assert {p.name:value.png_dimensions(p.read_bytes()) for p in unpacked.glob('*.png')} == expected
    assert not list(tmp_path.glob('.kwipml-icon-*'))

@pytest.mark.parametrize('kind',['missing','symlink','nonpng','truncated','oversized','directory'])
def test_input_refusal_preserves_existing_output(tmp_path,kind):
    value=tool();source=tmp_path/'input.png';output=tmp_path/'KwipML.icns';output.write_bytes(b'old icon')
    if kind=='symlink':source.symlink_to(ASSET)
    elif kind=='nonpng':source.write_bytes(b'not an image')
    elif kind=='truncated':source.write_bytes(ASSET.read_bytes()[:40])
    elif kind=='oversized':source.write_bytes(b'x'*(value.MAX_INPUT_BYTES+1))
    elif kind=='directory':source.mkdir()
    with pytest.raises((ValueError,OSError)):value.build_icon(source,output)
    assert output.read_bytes()==b'old icon'
    assert not list(tmp_path.glob('.kwipml-icon-*'))

def test_conversion_failure_preserves_previous_icon_and_cleans_staging(tmp_path,monkeypatch):
    value=tool();output=tmp_path/'KwipML.icns';output.write_bytes(b'old icon')
    def failure(*args,**kwargs):raise subprocess.CalledProcessError(1,args[0],stderr=b'conversion failed')
    monkeypatch.setattr(value.subprocess,'run',failure)
    with pytest.raises(ValueError,match='conversion'):value.build_icon(ASSET,output)
    assert output.read_bytes()==b'old icon'
    assert not list(tmp_path.glob('.kwipml-icon-*'))

def test_symlink_destination_is_not_followed_or_replaced(tmp_path):
    value=tool();target=tmp_path/'owned';target.write_bytes(b'untouched');output=tmp_path/'KwipML.icns';output.symlink_to(target)
    with pytest.raises(ValueError):value.build_icon(ASSET,output)
    assert output.is_symlink() and target.read_bytes()==b'untouched'

def test_cli_uses_explicit_source_and_output(tmp_path):
    result=subprocess.run([sys.executable,str(ROOT/'scripts/app_icon.py'),'--source',str(ASSET),'--output',str(tmp_path/'KwipML.icns')],capture_output=True,text=True,timeout=45)
    assert result.returncode==0,result.stderr
    assert (tmp_path/'KwipML.icns').is_file()

@pytest.mark.parametrize('width,height',[(512,512),(1024,512),(5000,5000)])
def test_invalid_source_dimensions_refuse_before_conversion(tmp_path,width,height):
    value=tool();data=bytearray(ASSET.read_bytes())
    data[16:24]=struct.pack('>II',width,height)
    data[29:33]=struct.pack('>I',zlib.crc32(data[12:29])&0xffffffff)
    source=tmp_path/'dimensions.png';source.write_bytes(data)
    output=tmp_path/'KwipML.icns';output.write_bytes(b'previous icon')
    with pytest.raises(ValueError):value.build_icon(source,output)
    assert output.read_bytes()==b'previous icon'
    assert not list(tmp_path.glob('.kwipml-icon-*'))

def test_corrupt_png_checksum_is_rejected(tmp_path):
    value=tool();data=bytearray(ASSET.read_bytes());data[29]^=1
    source=tmp_path/'corrupt.png';source.write_bytes(data)
    with pytest.raises(ValueError,match='checksum'):value.build_icon(source,tmp_path/'KwipML.icns')
    assert not (tmp_path/'KwipML.icns').exists()
