"""A report shares one file snapshot; another report must observe changed evidence."""
import copy
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('coverage_snapshot', Path(__file__).resolve().parents[2] / 'scripts/coverage_report.py')
coverage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coverage)

@pytest.fixture
def fixture(tmp_path):
    proof = tmp_path / 'proof.txt'
    proof.write_text('verified original evidence\n')
    (tmp_path / 'alias.txt').symlink_to(proof)
    inventory = tmp_path / 'inventory.json'
    inventory.write_text(json.dumps([{'symbol':'lib.Op','canonical_symbol':'lib.Op','disposition':'mapped','capability_ids':['op'],'reason':'Selected'}]))
    digest = hashlib.sha256(proof.read_bytes()).hexdigest()
    modes = ['forward', 'input-gradients', 'shape-contract']
    evidence = {dimension:[{'path':path,'sha256':digest,'status':'verified','scope':'op:'+mode}
                          for mode in modes for path in ['proof.txt','./proof.txt','alias.txt']]
                for dimension in coverage.REQUIRED_EVIDENCE}
    catalog = {'schema_version':1,'catalog_version':'1','framework':'test','framework_version':'1','documentation_version':'1',
               'denominator_status':'frozen','approval':{'reference':'fixture approval'},
               'inventory':{'path':'inventory.json','sha256':hashlib.sha256(inventory.read_bytes()).hexdigest(),'scope':'test','framework_completeness':'not_established'},
               'required_families':['activation'],'thresholds':{'minimum_percent':90,'target_percent':95},
               'capabilities':[{'id':'op','family':'activation','canonical_symbol':'lib.Op','mode_scope':modes,
                                'configuration_contract':'verified','mode_scope_status':'reviewed',
                                'backend_matrix':{'status':'verified','cases':['cpu-float32']},'evidence':evidence}]}
    return tmp_path, catalog, proof

def test_each_unique_resolved_file_is_hashed_and_checked_once_per_report(fixture, monkeypatch):
    root, catalog, proof = fixture
    hashes, resolutions, checks = Counter(), Counter(), Counter()
    original_sha, original_path, original_is_file = coverage._sha, coverage._path, Path.is_file
    def sha(path):
        hashes[path] += 1
        return original_sha(path)
    def resolve(root, value):
        resolutions[value] += 1
        return original_path(root, value)
    def is_file(path):
        checks[path] += 1
        return original_is_file(path)
    monkeypatch.setattr(coverage, '_sha', sha)
    monkeypatch.setattr(coverage, '_path', resolve)
    monkeypatch.setattr(Path, 'is_file', is_file)
    result = coverage.report_catalog(catalog, root)
    assert result['complete_ids'] == ['op'] and result['release_gate_passed']
    assert hashes == {root/'inventory.json':1, proof:1}
    assert checks == {root/'inventory.json':1, proof:1}
    assert resolutions == {'inventory.json':1,'proof.txt':1,'./proof.txt':1,'alias.txt':1}

def test_next_report_rehashes_changed_bytes_and_observes_deleted_file(fixture, monkeypatch):
    root, catalog, proof = fixture
    hashes = Counter()
    original = coverage._sha
    def sha(path):
        hashes[path] += 1
        return original(path)
    monkeypatch.setattr(coverage, '_sha', sha)
    assert coverage.report_catalog(catalog, root)['complete_ids'] == ['op']
    proof.write_text('changed evidence with same catalog digest\n')
    changed = coverage.report_catalog(catalog, root)
    assert changed['complete_ids'] == []
    assert hashes[proof] == 2
    assert len([reason for reason in changed['incomplete'][0]['reasons'] if 'stale verified evidence' in reason]) == 90
    proof.unlink()
    assert coverage.report_catalog(catalog, root)['complete_ids'] == []
    assert hashes[proof] == 2  # Missing paths are never hashed.
    assert hashes[root/'inventory.json'] == 3

def test_every_mode_dimension_and_row_status_still_controls_credit(fixture):
    root, catalog, _ = fixture
    assert coverage.report_catalog(catalog, root)['complete_count'] == 1
    for dimension in coverage.REQUIRED_EVIDENCE:
        for mode in catalog['capabilities'][0]['mode_scope']:
            changed = copy.deepcopy(catalog)
            rows = changed['capabilities'][0]['evidence'][dimension]
            for row in rows:
                if row['scope'] == 'op:'+mode:
                    row['status'] = 'present'
            assert coverage.report_catalog(changed, root)['complete_count'] == 0
    changed = copy.deepcopy(catalog)
    changed['capabilities'][0]['evidence']['execution'][0]['sha256'] = '0'*64
    assert coverage.report_catalog(changed, root)['complete_count'] == 0
    assert coverage.report_catalog(catalog, root)['complete_count'] == 1

@pytest.mark.parametrize('path', ['../outside-proof.txt', '/tmp/outside-proof.txt'])
def test_unverified_rows_cannot_hide_escaping_paths(fixture, path):
    root, catalog, _ = fixture
    catalog['capabilities'][0]['evidence']['ui'].append({'path':path,'status':'present'})
    with pytest.raises(ValueError, match='repository'):
        coverage.report_catalog(catalog, root)

def test_later_report_resolves_retargeted_symlink_again(fixture, tmp_path):
    root, catalog, _ = fixture
    assert coverage.report_catalog(catalog, root)['complete_count'] == 1
    outside = tmp_path.parent / (tmp_path.name+'-outside.txt')
    outside.write_text('outside proof')
    try:
        alias = root/'alias.txt'
        alias.unlink()
        alias.symlink_to(outside)
        with pytest.raises(ValueError, match='escapes repository'):
            coverage.report_catalog(catalog, root)
    finally:
        outside.unlink()

def test_missing_inventory_never_matches_a_null_digest(fixture):
    root, catalog, _ = fixture
    (root/'inventory.json').unlink()
    catalog['inventory']['sha256'] = None
    with pytest.raises(ValueError, match='Inventory snapshot missing or hash mismatch'):
        coverage.report_catalog(catalog, root)
