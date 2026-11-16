import copy
import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('packager_zlib',ROOT/'scripts/runtime_package.py')
package=importlib.util.module_from_spec(spec);spec.loader.exec_module(package)

def evidence():return json.loads((ROOT/'docs/runtime/standalone-license-inventory.json').read_text())

def test_known_archive_system_link_is_reconciled_without_erasing_missing_notice():
    original=evidence();r=package.reconcile_standalone_zlib(original,distribution_build='20250814',binary_links=['/usr/lib/libz.1.dylib'],runtime_version='1.2.12')
    assert r['status']=='explainedMetadataOvermatch'
    assert not r['redistributionReviewComplete']
    assert original['missingReferencedNotices']==['licenses/LICENSE.zlib-ng.txt']
    assert r['missingReference']=='licenses/LICENSE.zlib-ng.txt'

@pytest.mark.parametrize('mutation',['archive','target','python','static','local','build','binary','version'])
def test_unknown_or_non_system_evidence_stays_unresolved(mutation):
    i=evidence();build='20250814';links=['/usr/lib/libz.1.dylib'];version='1.2.12'
    if mutation=='archive':i['archiveSHA256']='0'*64
    if mutation=='target':i['targetTriple']='x86_64-apple-darwin'
    if mutation=='python':i['pythonVersion']='3.13.7'
    if mutation=='static':i['extensions']['zlib'][0]['links']=[{'name':'z','path_static':'build/lib/libz.a'}]
    if mutation=='local':links=['@rpath/libz.1.dylib']
    if mutation=='build':build='20250901'
    if mutation=='binary':links=['/usr/lib/libz.1.dylib','@rpath/libz-ng.dylib']
    if mutation=='version':version='2.2.4'
    r=package.reconcile_standalone_zlib(i,distribution_build=build,binary_links=links,runtime_version=version)
    assert r['status']=='unresolved' and not r['redistributionReviewComplete']

def test_canonical_build_requires_explicit_archive_before_mutation(tmp_path):
    with pytest.raises(ValueError,match='standalone archive'):package.build(tmp_path/'never-created')
    assert not (tmp_path/'never-created').exists()

def test_unrelated_missing_notice_is_not_closed():
    i=evidence();i['missingReferencedNotices'].append('licenses/other.txt')
    r=package.reconcile_standalone_zlib(i,distribution_build='20250814',binary_links=['/usr/lib/libz.1.dylib'],runtime_version='1.2.12')
    assert r['remainingMissingReferences']==['licenses/other.txt']

def test_changed_archive_and_symlink_fail_before_destination_creation(tmp_path):
    archive=tmp_path/'bad.tar.zst';archive.write_bytes(b'unknown')
    with pytest.raises(ValueError,match='unresolved'):package.build(tmp_path/'output',standalone_archive=archive)
    link=tmp_path/'link';link.symlink_to(archive)
    with pytest.raises(OSError):package.build(tmp_path/'output',standalone_archive=link)
    assert not (tmp_path/'output').exists()

def test_retained_exact_archive_attachment_into_isolated_temporary_output(tmp_path):
    archive=ROOT/'.local/runtime-feasibility/cpython-full.tar.zst'
    runtime=ROOT/'.local/runtime-feasibility/payload5/KwipMLRuntime'
    if not archive.exists() or not runtime.exists():pytest.skip('Exact retained archive/runtime unavailable; integration not exercised')
    result=package.attach_standalone_licenses(tmp_path,archive,runtime/'python/bin/python3.13',runtime/'python/lib/libpython3.13.dylib','20250814')
    inventory=json.loads((tmp_path/result['inventory']).read_text())
    reconciliation=json.loads((tmp_path/result['reconciliation']).read_text())
    assert inventory['missingReferencedNotices']==['licenses/LICENSE.zlib-ng.txt']
    assert reconciliation['status']=='explainedMetadataOvermatch'
    assert not result['redistributionReviewComplete']
    assert (tmp_path/'standalone-license-evidence/licenses/LICENSE.zlib.txt').is_file()
    assert not (tmp_path/'standalone-license-evidence/licenses/LICENSE.zlib-ng.txt').exists()
