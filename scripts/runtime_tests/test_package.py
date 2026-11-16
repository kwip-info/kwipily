import importlib.util
from pathlib import Path
import pytest
from types import SimpleNamespace

MODULE=Path(__file__).resolve().parents[1]/'runtime_package.py'
spec=importlib.util.spec_from_file_location('runtime_package',MODULE)
package=importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)

def test_absolute_and_escaping_runtime_links_are_rejected(tmp_path):
    (tmp_path/'bad').symlink_to('/usr/bin/true')
    with pytest.raises(ValueError,match='outside'): package.audit_links(tmp_path)

def test_relative_internal_runtime_links_survive(tmp_path):
    (tmp_path/'actual').write_text('content')
    (tmp_path/'link').symlink_to('actual')
    package.audit_links(tmp_path)

def test_existing_output_is_preserved(tmp_path):
    marker=tmp_path/'keep';marker.write_text('user data')
    with pytest.raises(ValueError,match='empty|exists'): package.require_empty_destination(tmp_path)
    assert marker.read_text()=='user data'

def test_runtime_source_build_uses_pinned_backend_constraints(tmp_path,monkeypatch):
    """Exercise the real assembly path through install without invoking build tools."""
    source=tmp_path/'managed';(source/'bin').mkdir(parents=True)
    (source/'bin/python3.13').write_text('synthetic interpreter fixture')
    (source/'BUILD').write_text('20250814')
    monkeypatch.setattr(package.os,'uname',lambda:SimpleNamespace(machine='arm64'))
    monkeypatch.setattr(package.subprocess,'check_output',lambda *args,**kwargs:str(source/'bin/python3.13'))
    calls=[]
    class InstallCaptured(Exception):pass
    def run(command,**kwargs):
        if command[:3]==['uv','pip','install']:
            calls.append((command,kwargs));raise InstallCaptured
    monkeypatch.setattr(package.subprocess,'run',run)
    with pytest.raises(InstallCaptured):
        package._build(tmp_path/'assembled','3.13.6',tmp_path/'unused.tar.gz')
    command,kwargs=calls[0]
    # Source packages must not silently resolve a newer backend in isolation.
    assert '--build-constraints' in command
    constraints=Path(command[command.index('--build-constraints')+1])
    requirements=Path(command[command.index('-r')+1])
    assert constraints==requirements==package.ROOT/'requirements-runner.txt'
    assert 'setuptools==84.0.0' in constraints.read_text().splitlines()
    assert kwargs['check'] is True


def test_vendor_blog_cleanup_preserves_code_and_license_and_is_idempotent(tmp_path):
    import ast
    # Synthetic pinned docstring excerpt; no installed dependency prerequisite.
    citation=('    [2] A. Faraji, "Corona Product in Graph Theory," Ali Faraji, May 11, 2021.\n'
              '        https://blog.alifaraji.ir/math/graph-theory/corona-product.html (accessed Dec. 07, 2021).\n')
    path=tmp_path/'networkx/algorithms/operators/product.py';path.parent.mkdir(parents=True)
    before='def corona_product():\n    """Graph operation.\n'+citation+'    """\n    return 42\n'
    path.write_text(before)
    notice=tmp_path/'LICENSE.txt';notice.write_text('Copyright retained; BSD-3-Clause')
    cached=path.parent/'__pycache__/product.cpython-313.pyc';cached.parent.mkdir();cached.write_bytes(b'old docstring')
    package.clean_vendor_documentation(tmp_path)
    after=path.read_text();package.clean_vendor_documentation(tmp_path)
    assert 'alifaraji' not in after and path.read_text()==after and not cached.exists()
    first,second=ast.parse(before),ast.parse(after)
    first.body[0].body.pop(0);second.body[0].body.pop(0)
    assert ast.dump(first)==ast.dump(second)
    assert notice.read_text()=='Copyright retained; BSD-3-Clause'
