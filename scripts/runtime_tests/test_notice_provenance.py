"""Pinned upstream notices: deterministic offline provenance, not legal clearance."""
import importlib.metadata
import runpy
from pathlib import Path
from email.message import Message
import sys
import pytest

ROOT = Path(__file__).resolve().parents[2]

@pytest.fixture
def tool(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(importlib.metadata, 'distributions', lambda: [])
    monkeypatch.setattr(sys, 'argv', ['runtime_licenses.py', str(tmp_path/'runtime')])
    result = runpy.run_path(str(ROOT/'scripts/runtime_licenses.py'))
    capsys.readouterr()
    return result

class Dist:
    def __init__(self, name, version):
        self.metadata = Message(); self.metadata['Name'] = name
        self.version = version

@pytest.mark.parametrize('name,version', [('flatbuffers','25.12.19'),('namex','0.1.0')])
def test_exact_distribution_upstream_notices_have_verified_sources(tool,tmp_path,name,version):
    records = tool['distribution_provenance_notices'](Dist(name,version),tmp_path/'out')
    assert len(records) == 1
    record = records[0]
    assert record['version'] == version and record['sha256']
    assert record['sourceURL'].startswith('https://')
    assert record['sourceLicenseExpression'] == 'Apache-2.0'
    assert (tmp_path/'out'/record['copiedTo']).is_file()
    assert record['redistributionReviewComplete'] is False


def test_unknown_version_never_inherits_notice(tool,tmp_path):
    assert tool['distribution_provenance_notices'](Dist('flatbuffers','99.0'),tmp_path/'out') == []

@pytest.mark.parametrize('mode',['missing','changed','symlink','oversized'])
def test_pinned_notice_source_failures_stop_copy(tool,tmp_path,mode):
    sources=tmp_path/'notices';sources.mkdir()
    asset=sources/'flatbuffers-25.12.19-LICENSE.txt'
    if mode=='changed':asset.write_text('changed notice')
    elif mode=='symlink':asset.symlink_to(ROOT/'scripts/runtime_notices/flatbuffers-25.12.19-LICENSE.txt')
    elif mode=='oversized':asset.write_bytes(b'x'*65537)
    with pytest.raises(ValueError,match='notice'):
        tool['distribution_provenance_notices'](Dist('flatbuffers','25.12.19'),tmp_path/'out',notice_root=sources)
    assert not (tmp_path/'out').exists()


def test_exact_native_addons_use_their_own_versioned_notices(tool,tmp_path):
    for dirname,package,version in [('itcl4.2.4','Itcl','4.2.4'),('thread2.8.9','Thread','2.8.9')]:
        folder=tmp_path/'python/lib'/dirname;folder.mkdir(parents=True)
        (folder/'pkgIndex.tcl').write_text(f'package ifneeded {package} {version} [list load [file join $dir libexample.dylib]]\n')
    records,unresolved=tool['native_provenance_notices'](tmp_path)
    assert len(records)==2 and not unresolved
    assert {r['version'] for r in records}=={'4.2.4','2.8.9'}
    assert all((tmp_path/r['copiedTo']).is_file() for r in records)
    assert all(not r['redistributionReviewComplete'] for r in records)


def test_native_declared_version_mismatch_and_unknown_directory_stay_unresolved(tool,tmp_path):
    for dirname,package,version in [('itcl4.2.4','Itcl','4.2.5'),('thread9.0','Thread','9.0')]:
        folder=tmp_path/'python/lib'/dirname;folder.mkdir(parents=True)
        (folder/'pkgIndex.tcl').write_text(f'package ifneeded {package} {version} {{}}\n')
    records,unresolved=tool['native_provenance_notices'](tmp_path)
    assert not records and len(unresolved)==2
    assert not (tmp_path/'licenses').exists()
