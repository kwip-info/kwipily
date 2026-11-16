"""Source/notice disclosure controls; no app builds, native edits or signing."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[2]


def helper():
    spec = importlib.util.spec_from_file_location('source_disclosure', ROOT / 'scripts/runtime_source_disclosure.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def digest(data): return hashlib.sha256(data).hexdigest()


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    module = helper(); runtime = tmp_path / 'runtime'; runtime.mkdir()
    files = {
        'python/lib/python3.13/site-packages/certifi/cacert.pem': b'EXACT SYNTHETIC CA SOURCE\n',
        'python/lib/python3.13/site-packages/certifi-2026.7.22.dist-info/licenses/LICENSE': b'synthetic certifi MPL notice\n',
        'python/lib/python3.13/site-packages/tensorflow/THIRD_PARTY_NOTICES.txt': b'Eigen 3.3.90 20f4d829087ef6ed6a502859c6fd093065e806d0 Mozilla Public License\n',
        'python/lib/python3.13/site-packages/tensorflow/include/LICENSE': b'synthetic TF license\n',
        'python/lib/python3.13/site-packages/tensorflow/include/Eigen/Core': b'synthetic Eigen source\n',
        'python/lib/python3.13/site-packages/tensorflow/include/unsupported/Eigen/Tensor': b'synthetic tensor source\n',
        'licenses/certifi/LICENSE': b'synthetic certifi MPL notice\n',
        'licenses/tensorflow/NOTICE': b'synthetic TF third party notice\n',
        'licenses/scipy/LICENSE': b'synthetic SciPy notice LGPL2.1\n',
        'licenses/CPython/LICENSE.txt': b'synthetic CPython notice\n',
        'standalone-license-evidence/licenses/LICENSE.example': b'standalone notice\n',
        'runner/main.py': b'worker unchanged\n',
        'python/lib/python3.13/site-packages/scipy/.dylibs/libquadmath.0.dylib': b'native unchanged\n',
    }
    for relative, data in files.items():
        path = runtime / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
    for name, version in [('certifi', '2026.7.22'), ('tensorflow', '2.21.0'), ('scipy', '1.17.0')]:
        path = runtime / f'python/lib/python3.13/site-packages/{name}-{version}.dist-info/METADATA'
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(f'Name: {name}\nVersion: {version}\n')
    inventory = {'schemaVersion': 1, 'distributions': [
        {'name': name, 'version': version, 'noticeFiles': ['licenses/' + name + '/' + ('NOTICE' if name == 'tensorflow' else 'LICENSE')]}
        for name, version in [('certifi','2026.7.22'),('tensorflow','2.21.0'),('scipy','1.17.0')]],
        'nativeComponentNotices': [], 'pythonNotice': 'licenses/CPython/LICENSE.txt', 'unresolvedNativeComponentNotices': []}
    (runtime / 'license-inventory.json').write_text(json.dumps(inventory))
    standalone = {'schemaVersion': 1, 'notices': [{'path': 'licenses/LICENSE.example', 'sha256': digest(b'standalone notice\n')}]}
    (runtime / 'standalone-license-evidence/inventory.json').write_text(json.dumps(standalone))
    tree = [{'path':'Core','bytes':len(b'synthetic Eigen source\n'),'sha256':digest(b'synthetic Eigen source\n')}]
    pins = dict(module.PINS)
    for key, relative in [('certifiSource','python/lib/python3.13/site-packages/certifi/cacert.pem'),
                          ('certifiLicense','python/lib/python3.13/site-packages/certifi-2026.7.22.dist-info/licenses/LICENSE'),
                          ('tensorflowNotices','python/lib/python3.13/site-packages/tensorflow/THIRD_PARTY_NOTICES.txt'),
                          ('tensorflowLicense','python/lib/python3.13/site-packages/tensorflow/include/LICENSE')]:
        pins[key] = digest(files[relative])
    pins['eigenTree'] = digest(json.dumps(tree, sort_keys=True, separators=(',',':')).encode())
    pins['eigenFiles'] = 1
    unsupported=[{'path':'Tensor','bytes':len(b'synthetic tensor source\n'),'sha256':digest(b'synthetic tensor source\n')}]
    pins['eigenUnsupportedTree']=digest(json.dumps(unsupported,sort_keys=True,separators=(',',':')).encode())
    pins['eigenUnsupportedFiles']=1
    pins['gccNative']={'libquadmath.0.dylib':digest(b'native unchanged\n')}
    monkeypatch.setattr(module, 'PINS', pins)
    gcc=tmp_path/'gcc';gcc.mkdir();retained=tmp_path/'retained';retained.mkdir()
    for name in ['gcc-13.4.0-libquadmath-COPYING.LIB','homebrew-core-gcc13-LICENSE.txt','actions-runner-images-LICENSE.txt','homebrew-formula-patches-LICENSE.txt','gcc-13.4-BUILD_CONTEXT.md','RUNTIME_MODIFICATIONS.md','eigen-SOURCE_PROVENANCE.json','eigen-tensorflow-eigen-workspace.bzl','eigen-tensorflow-eigen-archive.BUILD','eigen-COPYING.MPL2','eigen-COPYING.BSD','eigen-CopyrightMINPACK.txt']:
        shutil.copy2(ROOT/'scripts/runtime_notices'/name,retained/name)
    eigen_proof=json.loads((retained/'eigen-SOURCE_PROVENANCE.json').read_text())
    eigen_proof['headerSHA256']={'Eigen/Core':digest(b'synthetic Eigen source\n'),'unsupported/Eigen/Tensor':digest(b'synthetic tensor source\n')}
    eigen_proof['actualFiles']=2
    eigen_raw=json.dumps(eigen_proof).encode();(retained/'eigen-SOURCE_PROVENANCE.json').write_bytes(eigen_raw)
    monkeypatch.setattr(module,'EIGEN_PROVENANCE_SHA256',digest(eigen_raw))
    materials=[]
    for name in ['gcc-13.4.0.tar.xz','gcc-13.4.0-darwin.diff','gcc@13.rb','scipy-wheel-workflow.yml','runner-image-20260105.md']:
        data=('synthetic source '+name+'\n').encode();(gcc/name).write_bytes(data)
        materials.append({'file':name,'sourceURL':'https://example.invalid/'+name,'sha256':digest(data),'bytes':len(data)})
    provenance={'schemaVersion':1,'status':'verifiedSupplierAndCorrespondingSourceMaterials','sourceMaterials':materials,
                'replacementRuntimeVerified':False,'companyRedistributionApproval':False}
    raw=json.dumps(provenance,sort_keys=True).encode();(gcc/'SOURCE_PROVENANCE.json').write_bytes(raw)
    (retained/'gcc-13.4-source-provenance.json').write_bytes(raw)
    monkeypatch.setattr(module,'NOTICES',retained);monkeypatch.setattr(module,'PROVENANCE_SHA256',digest(raw))
    monkeypatch.setattr(module,'DEFAULT_GCC_DIRECTORY',gcc)
    return module, runtime


def test_stage_exact_source_index_license_and_rights_without_native_or_worker_changes(candidate):
    module, runtime = candidate
    before = {str(p.relative_to(runtime)):p.read_bytes() for p in runtime.rglob('*') if p.is_file()}
    index = module.stage_source_disclosure(runtime)
    output = runtime / 'source-disclosure'
    assert index['redistributionReviewComplete'] is False
    assert index['gccCorrespondingSourceComplete'] is True
    assert index['sourceIncluded'] is True
    assert index['replacementRuntimeVerified'] is True
    assert len(index['notices']) == 5
    assert (output / 'COPYING.LIB').read_bytes() == (ROOT / 'scripts/runtime_notices/gcc-13.4.0-libquadmath-COPYING.LIB').read_bytes()
    assert digest((output / 'COPYING.LIB').read_bytes()) == '32434afcc8666ba060e111d715bfdb6c2d5dd8a35fa4d3ab8ad67d8f850d2f2b'
    assert (output / 'certifi/cacert.pem').read_bytes() == b'EXACT SYNTHETIC CA SOURCE\n'
    assert json.loads((output / 'SOURCE_INDEX.json').read_text()) == index
    readme = (output / 'README.md').read_text()
    assert 'licenses/' in readme and 'cacert.pem' in readme and 'Eigen' in readme
    assert 'pending' in readme.lower() and 'do not restrict' in readme
    assert str(runtime) not in readme and str(runtime) not in json.dumps(index)
    for relative, data in before.items(): assert (runtime / relative).read_bytes() == data


def test_deterministic_stage_and_readonly_existing_verification(candidate, tmp_path):
    module, runtime = candidate
    module.stage_source_disclosure(runtime)
    output = runtime / 'source-disclosure'
    before = {str(p.relative_to(output)):(p.read_bytes(),p.stat().st_mtime_ns) for p in output.rglob('*') if p.is_file()}
    module.verify_source_disclosure(runtime)
    assert before == {str(p.relative_to(output)):(p.read_bytes(),p.stat().st_mtime_ns) for p in output.rglob('*') if p.is_file()}
    with pytest.raises(FileExistsError): module.stage_source_disclosure(runtime)
    shutil.rmtree(output); module.stage_source_disclosure(runtime)
    assert {k:v[0] for k,v in before.items()} == {str(p.relative_to(output)):p.read_bytes() for p in output.rglob('*') if p.is_file()}


@pytest.mark.parametrize('relative', ['python/lib/python3.13/site-packages/certifi/cacert.pem',
    'python/lib/python3.13/site-packages/tensorflow/include/Eigen/Core',
    'python/lib/python3.13/site-packages/tensorflow/THIRD_PARTY_NOTICES.txt',
    'standalone-license-evidence/licenses/LICENSE.example'])
@pytest.mark.parametrize('mutation', ['tamper','missing','symlink'])
def test_source_and_notice_identity_failures_refuse_before_output(candidate, relative, mutation):
    module, runtime = candidate; path = runtime / relative
    if mutation == 'tamper': path.write_bytes(b'changed')
    elif mutation == 'missing': path.unlink()
    else:
        moved = path.with_name(path.name + '.original'); path.rename(moved); path.symlink_to(moved)
    with pytest.raises((ValueError,OSError)): module.stage_source_disclosure(runtime)
    assert not (runtime / 'source-disclosure').exists()
    assert not list(runtime.glob('.source-disclosure-*'))


def test_inventory_path_escape_and_missing_notice_refuse(candidate):
    module, runtime = candidate; path=runtime/'license-inventory.json'; value=json.loads(path.read_text())
    value['distributions'][0]['noticeFiles']=['../private'];path.write_text(json.dumps(value))
    with pytest.raises(ValueError): module.stage_source_disclosure(runtime)
    assert not (runtime/'source-disclosure').exists()


def test_existing_tamper_and_extra_files_refuse_without_repair(candidate):
    module,runtime=candidate;module.stage_source_disclosure(runtime);output=runtime/'source-disclosure'
    (output/'README.md').write_text('stale preserved')
    with pytest.raises(ValueError):module.verify_source_disclosure(runtime)
    assert (output/'README.md').read_text()=='stale preserved'
    shutil.rmtree(output);module.stage_source_disclosure(runtime);(output/'extra').write_text('not allowed')
    with pytest.raises(ValueError):module.verify_source_disclosure(runtime)
    assert (output/'extra').read_text()=='not allowed'


def test_actual_current_runtime_source_identity_readonly(tmp_path):
    runtime = ROOT/'.local/dist-production-runtime-20261009/KwipMLRuntime'
    if not (runtime/'python/lib/python3.13/site-packages/certifi-2026.7.22.dist-info/METADATA').is_file(): pytest.skip('Complete local production runtime fixture is not available')
    index, payloads = helper().prepare_source_disclosure(runtime,ROOT/'.local/dist-gcc-provenance-20261009')
    assert len(index['notices']) == 203
    assert index['eigen']['fileCount'] == 412
    assert index['certifi']['sha256'] == '9cc2a774b5198dcff14d9be1e66091f538975d867ce029a96bce15a55dfd730f'
    assert index['gccCorrespondingSourceComplete'] is True
    assert index['sourceIncluded'] is True
    assert index['replacementRuntimeVerified'] is True
    assert len(payloads['COPYING.LIB']) == 26440


@pytest.mark.parametrize('member',['gcc-13.4.0.tar.xz','gcc-13.4.0-darwin.diff','gcc@13.rb','SOURCE_PROVENANCE.json'])
def test_exact_gcc_source_or_correspondence_tamper_is_refused(candidate,member):
    module,runtime=candidate;(module.DEFAULT_GCC_DIRECTORY/member).write_bytes(b'changed source material')
    with pytest.raises(ValueError):module.stage_source_disclosure(runtime)
    assert not (runtime/'source-disclosure').exists()


def test_current_native_binding_cannot_silently_attribute_new_binary(candidate):
    module,runtime=candidate
    (runtime/'python/lib/python3.13/site-packages/scipy/.dylibs/libquadmath.0.dylib').write_bytes(b'new unknown library')
    with pytest.raises(ValueError):module.stage_source_disclosure(runtime)
    assert not (runtime/'source-disclosure').exists()


def test_partial_output_failure_is_cleaned_and_inputs_preserved(candidate,monkeypatch):
    module,runtime=candidate;original=Path.write_bytes
    def refuse(self,value):
        if self.name=='README.md' and '.source-disclosure-' in str(self): raise OSError('simulated write refusal')
        return original(self,value)
    monkeypatch.setattr(Path,'write_bytes',refuse)
    with pytest.raises(OSError):module.stage_source_disclosure(runtime)
    assert not (runtime/'source-disclosure').exists()
    assert not list(runtime.glob('.source-disclosure-*'))
    assert (runtime/'runner/main.py').read_bytes()==b'worker unchanged\n'


def test_retained_full_license_cannot_be_replaced_by_short_link(candidate):
    module,runtime=candidate;(module.NOTICES/'gcc-13.4.0-libquadmath-COPYING.LIB').write_bytes(b'LGPL: see URL')
    with pytest.raises(ValueError):module.stage_source_disclosure(runtime)
    assert not (runtime/'source-disclosure').exists()


def test_actual_cli_stage_and_offline_verify_on_disposable_current_runtime_projection(tmp_path):
    import subprocess
    import sys
    runtime=ROOT/'.local/dist-production-runtime-20261009/KwipMLRuntime'
    gcc=ROOT/'.local/dist-gcc-provenance-20261009'
    if not (runtime/'python/lib/python3.13/site-packages/certifi-2026.7.22.dist-info/METADATA').is_file() or not gcc.exists():pytest.skip('Complete reviewed local source fixtures are not available')
    module=helper();index,_=module.prepare_source_disclosure(runtime,gcc)
    paths={row['path'] for row in index['notices']}|{row['path'] for row in index['gcc']['nativeBindings']}
    paths.update('python/lib/python3.13/site-packages/tensorflow/include/Eigen/'+row['path'] for row in index['eigen']['files'])
    paths.update('python/lib/python3.13/site-packages/tensorflow/include/unsupported/Eigen/'+row['path'] for row in index['eigen']['unsupported']['files'])
    paths.update(['license-inventory.json','standalone-license-evidence/inventory.json',
        'python/lib/python3.13/site-packages/certifi/cacert.pem',
        'python/lib/python3.13/site-packages/certifi-2026.7.22.dist-info/licenses/LICENSE',
        'python/lib/python3.13/site-packages/tensorflow/THIRD_PARTY_NOTICES.txt',
        'python/lib/python3.13/site-packages/tensorflow/include/LICENSE'])
    paths.update('python/lib/python3.13/site-packages/'+name+'-'+version+'.dist-info/METADATA'
        for name,version in [('certifi','2026.7.22'),('tensorflow','2.21.0'),('scipy','1.17.0')])
    projection=tmp_path/'projection';projection.mkdir()
    for relative in paths:
        target=projection/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(runtime/relative,target)
    command=[sys.executable,str(ROOT/'scripts/runtime_source_disclosure.py'),'--runtime-root',str(projection)]
    staged=subprocess.run(command+['--gcc-source-directory',str(gcc)],capture_output=True,text=True,timeout=30)
    assert staged.returncode==0,staged.stderr
    source=projection/'source-disclosure/sources/gcc-13.4.0.tar.xz'
    assert source.stat().st_size==88530720
    assert digest(source.read_bytes())=='9c4ce6dbb040568fdc545588ac03c5cbc95a8dbf0c7aa490170843afb59ca8f5'
    verified=subprocess.run(command+['--verify-existing'],capture_output=True,text=True,timeout=30)
    assert verified.returncode==0,verified.stderr
    (projection/'source-disclosure/sources/BUILD_CONTEXT.md').write_text('tampered instructions')
    refused=subprocess.run(command+['--verify-existing'],capture_output=True,text=True,timeout=30)
    assert refused.returncode!=0
    assert (projection/'source-disclosure/sources/BUILD_CONTEXT.md').read_text()=='tampered instructions'


def test_verified_local_replacement_guide_has_explicit_bounded_scope(candidate):
    module,runtime=candidate
    index=module.stage_source_disclosure(runtime)
    output=runtime/'source-disclosure'
    guide=(output/'RUNTIME_MODIFICATIONS.md').read_bytes()
    assert guide==(ROOT/'scripts/runtime_notices/RUNTIME_MODIFICATIONS.md').read_bytes()
    assert index['replacementRuntimeVerified'] is True
    assert index['replacementGuide']['sha256']==digest(guide)
    assert index['replacementGuide']['bundledPath']=='RUNTIME_MODIFICATIONS.md'
    assert 'separate bundle identifier' in index['replacementVerificationScope']
    assert 'App Store processed' in index['replacementVerificationScope']
    assert index['companyRedistributionApproval'] is False
    assert index['appleAcceptanceVerified'] is False
    assert 'RUNTIME_MODIFICATIONS.md' in (output/'README.md').read_text()


@pytest.mark.parametrize('damage',['missing','tampered','symlink'])
def test_unreviewed_replacement_instructions_refused_before_output(candidate,damage):
    module,runtime=candidate
    guide=module.NOTICES/'RUNTIME_MODIFICATIONS.md'
    if damage=='missing':guide.unlink()
    elif damage=='tampered':guide.write_text('unverified instructions')
    else:
        original=guide.read_bytes();guide.unlink()
        elsewhere=module.NOTICES/'alternate-guide';elsewhere.write_bytes(original);guide.symlink_to(elsewhere)
    with pytest.raises((ValueError,OSError)):
        module.stage_source_disclosure(runtime)
    assert not (runtime/'source-disclosure').exists()


def test_eigen_distinguishes_notice_from_bundled_source_and_covers_tensor_headers(candidate):
    module,runtime=candidate
    index=module.stage_source_disclosure(runtime)
    eigen=index['eigen']
    assert eigen['noticeDeclaredVersion']=='3.3.90'
    assert eigen['noticeDeclaredCommit']=='20f4d829087ef6ed6a502859c6fd093065e806d0'
    assert 'version' not in eigen and 'commit' not in eigen
    assert eigen['bundledSourceRevisionVerified'] is True
    assert eigen['actualSourceCommit']=='dcbaf2d608f306450f1e74949eb87e9a22a7ef4b'
    assert eigen['actualSourceVersion']=='5.0.1-dev+master'
    assert (runtime/'source-disclosure/sources/eigen/CopyrightMINPACK.txt').read_bytes()==(ROOT/'scripts/runtime_notices/eigen-CopyrightMINPACK.txt').read_bytes()
    assert eigen['unsupported']['fileCount']==1
    assert eigen['unsupported']['files']==[{'path':'Tensor','bytes':24,'sha256':digest(b'synthetic tensor source\n')}]


@pytest.mark.parametrize('damage',['missing','tampered','symlink'])
def test_unsupported_eigen_source_refused_before_output(candidate,damage):
    module,runtime=candidate
    header=runtime/'python/lib/python3.13/site-packages/tensorflow/include/unsupported/Eigen/Tensor'
    if damage=='missing':header.unlink()
    elif damage=='tampered':header.write_text('unreviewed tensor')
    else:
        header.unlink();header.symlink_to(runtime/'runner/main.py')
    with pytest.raises((ValueError,OSError)):
        module.stage_source_disclosure(runtime)
    assert not (runtime/'source-disclosure').exists()


@pytest.mark.parametrize('asset',['eigen-SOURCE_PROVENANCE.json','eigen-COPYING.MPL2','eigen-CopyrightMINPACK.txt'])
def test_eigen_reviewed_proof_and_license_tampering_refused(candidate,asset):
    module,runtime=candidate
    (module.NOTICES/asset).write_text('tampered reviewed source evidence')
    with pytest.raises((ValueError,OSError)):
        module.stage_source_disclosure(runtime)
    assert not (runtime/'source-disclosure').exists()
