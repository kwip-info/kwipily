import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]


def module():
    spec = importlib.util.spec_from_file_location('api_audit', ROOT / 'scripts/app_store_api_audit.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.mark.parametrize('symbol', ['_NSWindowDidOrderOnScreenNotification', '_dcabs1_', '_lsame_', '_xerbla_array__'])
def test_rejects_each_reported_reference_even_in_renamed_native_file(tmp_path, symbol):
    binary = tmp_path / 'nested' / 'no-extension'
    binary.parent.mkdir()
    binary.write_bytes(b'\xcf\xfa\xed\xfe\0' + symbol.encode() + b'\0')
    with pytest.raises(ValueError, match=symbol):
        module().audit(tmp_path)


def test_allows_namespaced_openblas_definitions_and_documentation(tmp_path):
    (tmp_path / 'math').write_bytes(b'\xcf\xfa\xed\xfe\0_scipy_dcabs1_\0_scipy_lsame_\0_scipy_xerbla_array__\0')
    (tmp_path / 'notice.txt').write_text('_dcabs1_')
    assert module().audit(tmp_path)['nativeFilesScanned'] == 1


def test_refuses_escaping_link(tmp_path):
    (tmp_path / 'escape').symlink_to('/usr/lib')
    with pytest.raises(ValueError, match='symlink'):
        module().audit(tmp_path)


def test_production_gate_checks_runtime_and_complete_staged_app():
    source = (ROOT / 'scripts/production_package.py').read_text()
    assert "ROOT / 'scripts/app_store_api_audit.py', path" in source
    assert "ROOT / 'scripts/app_store_api_audit.py', staged_app" in source
    assert source.index("ROOT / 'scripts/app_store_api_audit.py', staged_app") < source.index("command(['/usr/bin/codesign', '--force'")
