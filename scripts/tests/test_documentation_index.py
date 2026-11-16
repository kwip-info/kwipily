"""Exercise the index CLI against temporary repositories, including drift failures."""
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'documentation_index.py'


def run(root, *args):
    return subprocess.run([sys.executable, str(SCRIPT), '--root', str(root), *args],
                          capture_output=True, text=True)


def test_indexes_navigate_nested_paths_and_escape_titles(tmp_path):
    folder = tmp_path / 'docs' / 'Deep folder'
    folder.mkdir(parents=True)
    (folder / 'a #1.md').write_text('# A | useful [title]\n', encoding='utf-8')
    assert run(tmp_path).returncode == 0
    root = (tmp_path / 'docs/INDEX.md').read_text()
    child = (folder / 'INDEX.md').read_text()
    assert 'Deep%20folder/INDEX.md' in root
    assert 'a%20%231.md' in child
    assert r'A \| useful \[title\]' in child
    assert '../INDEX.md' in child
    assert 'a #1.md' in child  # Filename disambiguates duplicate document titles.
    assert run(tmp_path, '--check').returncode == 0


def test_check_refuses_missing_and_changed_indexes_without_writing(tmp_path):
    docs = tmp_path / 'docs'
    docs.mkdir()
    source = docs / 'guide.md'
    source.write_text('# Before\n')
    assert run(tmp_path, '--check').returncode == 1
    assert not (docs / 'INDEX.md').exists()
    assert run(tmp_path).returncode == 0
    before = (docs / 'INDEX.md').read_bytes()
    source.write_text('# After\n')
    assert run(tmp_path, '--check').returncode == 1
    assert (docs / 'INDEX.md').read_bytes() == before
    assert run(tmp_path).returncode == 0
    assert run(tmp_path, '--check').returncode == 0


def test_deleted_document_and_empty_folder_have_no_stale_links(tmp_path):
    folder = tmp_path / 'docs' / 'old'
    folder.mkdir(parents=True)
    source = folder / 'old.md'
    source.write_text('No heading\n')
    assert run(tmp_path).returncode == 0
    source.unlink()
    assert run(tmp_path, '--check').returncode == 1
    assert run(tmp_path).returncode == 0
    assert not (folder / 'INDEX.md').exists()
    assert 'old/INDEX.md' not in (tmp_path / 'docs/INDEX.md').read_text()


def test_generation_is_deterministic_and_does_not_overwrite_readme(tmp_path):
    docs = tmp_path / 'docs'
    docs.mkdir()
    readme = docs / 'README.md'
    readme.write_text('# Curated start\nKeep me.\n')
    for name in ('B_TWO.md', 'A_ONE.md', 'A_TWO.md'):
        (docs / name).write_text('# Same title\n')
    assert run(tmp_path).returncode == 0
    first = (docs / 'INDEX.md').read_bytes()
    assert first.endswith(b'\n') and not first.endswith(b'\n\n')
    assert run(tmp_path).returncode == 0
    assert (docs / 'INDEX.md').read_bytes() == first
    assert readme.read_text() == '# Curated start\nKeep me.\n'


def test_refuses_to_overwrite_an_authored_index(tmp_path):
    docs = tmp_path / 'docs'
    docs.mkdir()
    (docs / 'guide.md').write_text('# Guide\n')
    index = docs / 'INDEX.md'
    index.write_text('# Owner-authored index\n')
    assert run(tmp_path).returncode == 2
    assert index.read_text() == '# Owner-authored index\n'
