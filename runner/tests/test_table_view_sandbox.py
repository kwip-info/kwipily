"""Exercise View under a real Seatbelt denial of ancestor directory reads."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from runner.table_view import _open_directory

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(sys.platform != 'darwin', reason='Darwin Seatbelt and O_NOFOLLOW_ANY')
def test_view_build_and_query_need_only_authorized_leaf(tmp_path):
    base = tmp_path.resolve()
    ancestor = base / 'denied-ancestor'
    leaf = ancestor / 'authorized-leaf'
    leaf.mkdir(parents=True)
    script = base / 'probe.py'
    script.write_text(f'''
import os, sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, {str(ROOT)!r})
from runner.table_view import build_index, query_index, _open_directory
ancestor = Path({str(ancestor)!r})
leaf = Path({str(leaf)!r})
try:
    fd = os.open(ancestor, os.O_RDONLY | os.O_DIRECTORY)
except PermissionError:
    pass
else:
    os.close(fd)
    raise AssertionError('Seatbelt must deny ancestor read-data')
table = SimpleNamespace(columns=({{'name':'id','dtype':'int64','nullable':False}},),
                        rows=[(i,) for i in range(120)], row_ids=[str(i) for i in range(120)])
path = leaf / 'view.sqlite3'
receipt = build_index(table, path, source_sha256='a'*64)
page = query_index(path, receipt['index']['sha256'], offset=100, source_sha256='a'*64)
assert page['rows'] == [[i] for i in range(100,120)]
assert page['totalRows'] == 120 and page['nextOffset'] is None
(leaf / 'nested').mkdir()
alias = leaf / 'alias'
alias.symlink_to(leaf, target_is_directory=True)
for target in (alias, alias / 'nested'):
    try:
        fd = _open_directory(target)
    except OSError:
        pass
    else:
        os.close(fd)
        raise AssertionError('Symlink traversal accepted')
print('VIEW_BUILD_QUERY_PASSED')
''')
    policy = '(version 1) (allow default) (deny file-read-data (literal ' + json.dumps(str(ancestor)) + '))'
    result = subprocess.run(['/usr/bin/sandbox-exec', '-p', policy, sys.executable, '-I', '-B', str(script)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == 'VIEW_BUILD_QUERY_PASSED'


@pytest.mark.parametrize('suffix', ['/./leaf', '/leaf/../leaf'])
def test_directory_refuses_noncanonical_components(tmp_path, suffix):
    (tmp_path / 'leaf').mkdir()
    with pytest.raises(ValueError, match='canonical'):
        _open_directory(str(tmp_path.resolve()) + suffix)


def test_directory_refuses_symlink_ancestor(tmp_path):
    base = tmp_path.resolve()
    (base / 'real' / 'leaf').mkdir(parents=True)
    (base / 'alias').symlink_to(base / 'real', target_is_directory=True)
    with pytest.raises(OSError):
        _open_directory(base / 'alias' / 'leaf')


def test_non_darwin_walk_keeps_ancestor_symlink_denial(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from runner import table_view
    monkeypatch.setattr(table_view, 'sys', SimpleNamespace(platform='linux'))
    base = tmp_path.resolve()
    leaf = base / 'real' / 'leaf'
    leaf.mkdir(parents=True)
    descriptor = table_view._open_directory(leaf)
    try:
        assert os.fstat(descriptor).st_ino == leaf.stat().st_ino
    finally:
        os.close(descriptor)
    (base / 'alias').symlink_to(base / 'real', target_is_directory=True)
    with pytest.raises(OSError):
        table_view._open_directory(base / 'alias' / 'leaf')
