import importlib.util
import hashlib
from pathlib import Path
import zipfile
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('repair', ROOT / 'scripts/repair_review_runtime.py')
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


def fixture(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    for name in ('scipy', 'scipy-1.17.0.dist-info'):
        (source / repair.SITE / name).mkdir(parents=True)
    libs = source / repair.SITE / 'scipy/.dylibs'
    libs.mkdir()
    (libs / 'gcc.dylib').write_bytes(b'reviewed GCC')
    monkeypatch.setattr(repair, 'GCC', {'gcc.dylib': hashlib.sha256(b'reviewed GCC').hexdigest()})
    for name in repair.OMIT:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'unused Tk')
    (source / 'source-disclosure').mkdir()
    (source / 'source-disclosure/old').write_text('stale')
    wheel = tmp_path / repair.WHEEL
    with zipfile.ZipFile(wheel, 'w') as z:
        z.writestr('scipy/.dylibs/gcc.dylib', b'different supplier')
        z.writestr('scipy/.dylibs/openblas.dylib', b'OpenBLAS')
        z.writestr('scipy-1.17.0.dist-info/LICENSE.txt', 'retained license')
    monkeypatch.setattr(repair, 'WHEEL_SHA256', hashlib.sha256(wheel.read_bytes()).hexdigest())
    return source, wheel, tmp_path / 'output'


def test_repair_preserves_reviewed_gcc_source_and_wheel_notice(tmp_path, monkeypatch):
    source, wheel, output = fixture(tmp_path, monkeypatch)
    repair.prepare(source, wheel, output)
    assert (output / repair.SITE / 'scipy/.dylibs/gcc.dylib').read_bytes() == b'reviewed GCC'
    assert (output / repair.SITE / 'scipy-1.17.0.dist-info/LICENSE.txt').read_text() == 'retained license'
    assert all(not (output / name).exists() for name in repair.OMIT)
    assert all((source / name).exists() for name in repair.OMIT)
    assert not (output / 'source-disclosure').exists()
    assert (source / 'source-disclosure/old').exists()


@pytest.mark.parametrize('failure', ['wheel', 'gcc', 'existing', 'link'])
def test_refuses_changed_inputs_or_existing_output(tmp_path, monkeypatch, failure):
    source, wheel, output = fixture(tmp_path, monkeypatch)
    if failure == 'wheel': wheel.write_bytes(b'wrong')
    if failure == 'gcc': (source / repair.SITE / 'scipy/.dylibs/gcc.dylib').write_bytes(b'wrong')
    if failure == 'existing': output.mkdir()
    if failure == 'link': (source / 'escape').symlink_to('/usr/lib')
    with pytest.raises(ValueError): repair.prepare(source, wheel, output)
    assert not (output / 'app-review-repair.json').exists()
