"""Explicit installed TensorFlow notice mapping; no framework execution/install."""
import importlib.metadata
import json
from email.message import Message
from pathlib import Path
import runpy
import sys
import hashlib
import pytest

ROOT = Path(__file__).resolve().parents[2]


class Distribution:
    def __init__(self, site, version='2.21.0'):
        self.site = site
        self.version = version
        self.metadata = Message()
        self.metadata['Name'] = 'tensorflow'
        self.metadata['Version'] = version
        self.metadata['License'] = 'Apache 2.0'
        self.files = [Path(f'tensorflow-{version}.dist-info/METADATA')]
    def locate_file(self, member):
        return self.site / member


def generate(tmp_path, monkeypatch, capsys, *, missing=None, symlink=False, version='2.21.0'):
    site = tmp_path / 'site'
    for name, data in [('tensorflow/include/LICENSE', 'exact framework license\n'), ('tensorflow/THIRD_PARTY_NOTICES.txt', 'exact third party notice\n')]:
        path = site / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name != missing:
            path.write_text(data)
    if symlink:
        path = site / 'tensorflow/include/LICENSE'
        path.unlink()
        path.symlink_to(site / 'tensorflow/THIRD_PARTY_NOTICES.txt')
    dist = Distribution(site, version)
    monkeypatch.setattr(importlib.metadata, 'distributions', lambda: [dist])
    monkeypatch.setattr(sys, 'argv', ['runtime_licenses.py', str(tmp_path / 'runtime')])
    runpy.run_path(str(ROOT / 'scripts/runtime_licenses.py'), run_name='__main__')
    return json.loads(capsys.readouterr().out), site


def test_exact_tensorflow_maps_both_installed_notices_with_source_hashes(tmp_path, monkeypatch, capsys):
    result, site = generate(tmp_path, monkeypatch, capsys)
    entry = result['distributions'][0]
    assert len(entry['noticeFiles']) == 2
    assert not entry['reviewRequired']
    assert len(entry['frameworkNoticeSources']) == 2
    for record in entry['frameworkNoticeSources']:
        source = site / record['source']
        copied = tmp_path / 'runtime' / record['copiedTo']
        assert copied.read_bytes() == source.read_bytes()
        assert record['sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert any('not legal clearance' in text for text in result['limitations'])


@pytest.mark.parametrize('missing', ['tensorflow/include/LICENSE', 'tensorflow/THIRD_PARTY_NOTICES.txt'])
def test_missing_explicit_notice_fails_closed(tmp_path, monkeypatch, capsys, missing):
    with pytest.raises(ValueError, match='TensorFlow.*notice'):
        generate(tmp_path, monkeypatch, capsys, missing=missing)


def test_symlink_notice_is_not_promoted(tmp_path, monkeypatch, capsys):
    with pytest.raises(ValueError, match='TensorFlow.*notice'):
        generate(tmp_path, monkeypatch, capsys, symlink=True)


def test_unreviewed_tensorflow_version_does_not_inherit_mapping(tmp_path, monkeypatch, capsys):
    result, _ = generate(tmp_path, monkeypatch, capsys, version='2.22.0')
    entry = result['distributions'][0]
    assert entry['noticeFiles'] == [] and entry['reviewRequired']
    assert not entry.get('frameworkNoticeSources')
