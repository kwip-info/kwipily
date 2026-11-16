import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('standalone',Path(__file__).parents[1]/'runtime_standalone_licenses.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def test_references_are_unique_and_include_python():
    data={'license_path':'licenses/python.txt','build_info':{'extensions':{'x':[{'license_paths':['licenses/x.txt','licenses/x.txt']}]}}}
    assert module.notice_paths(data)==['licenses/python.txt','licenses/x.txt']

def test_escaping_license_path_rejected():
    with pytest.raises(ValueError,match='Unsafe'):
        module.notice_paths({'license_path':'../outside','build_info':{'extensions':{}}})
