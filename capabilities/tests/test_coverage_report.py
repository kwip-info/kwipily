"""Coverage claims must be earned by complete, version-bound evidence."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "coverage_report.py"
spec = importlib.util.spec_from_file_location("coverage_report", SCRIPT)
coverage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coverage)

class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.evidence = self.root / "proof.txt"
        self.evidence.write_text("verified fixture evidence\n")
        self.digest = hashlib.sha256(self.evidence.read_bytes()).hexdigest()
        self.records = [
            {"symbol":"lib.Op", "canonical_symbol":"lib.Op", "disposition":"mapped", "capability_ids":["op"], "reason":"Selected", "doc_url":"https://official.example/Op"},
            {"symbol":"lib.Alias", "canonical_symbol":"lib.Op", "disposition":"alias", "capability_ids":["op"], "reason":"Documented alias", "doc_url":"https://official.example/Op"},
            {"symbol":"lib.Other", "canonical_symbol":"lib.Other", "disposition":"unsupported", "capability_ids":[], "reason":"Future family", "doc_url":"https://official.example/Other"},
        ]
        self.snapshot = self.root / "inventory.json"
        self.snapshot.write_text(json.dumps(self.records))
        evidence = {key:[{"path":"proof.txt", "sha256":self.digest,"status":"verified","scope":"op:default"}]
                    for key in coverage.REQUIRED_EVIDENCE}
        self.catalog = {"schema_version":1,"catalog_version":"1.0.0","framework":"test","framework_version":"1.2.3",
            "documentation_version":"1.2.3","denominator_status":"frozen","approval":{"reference":"fixture approval"},
            "inventory":{"path":"inventory.json","sha256":hashlib.sha256(self.snapshot.read_bytes()).hexdigest(),
                "scope":"published test index","framework_completeness":"not_established"},
            "required_families":["layers"],"thresholds":{"minimum_percent":90,"target_percent":95},
            "capabilities":[{"id":"op","family":"layers","canonical_symbol":"lib.Op","mode_scope":["default"],
                "configuration_contract":"verified","mode_scope_status":"reviewed","backend_matrix":{"status":"verified","cases":["cpu-float32"]},
                "evidence":evidence}],"poc_components":[]}

    def report(self):
        return coverage.report_catalog(self.catalog,self.root)

    def test_complete_evidence_counts_one_unit_not_alias(self):
        r=self.report()
        self.assertEqual(r["capability_count"],1)
        self.assertEqual(r["complete_count"],1)
        self.assertEqual(r["capability_percent"],100.0)
        self.assertEqual(r["api"]["symbol_count"],3)
        self.assertEqual(r["api"]["canonical_count"],2)
        self.assertIsNone(r["api"]["framework_wide_percent"])
        self.assertTrue(r["release_gate_passed"])

    def test_every_evidence_dimension_is_required(self):
        for field in coverage.REQUIRED_EVIDENCE:
            with self.subTest(field=field):
                original=self.catalog["capabilities"][0]["evidence"].pop(field)
                self.assertEqual(self.report()["complete_count"],0)
                self.catalog["capabilities"][0]["evidence"][field]=original

    def test_existing_file_does_not_prove_verification(self):
        self.catalog["capabilities"][0]["evidence"]["execution"][0]["status"]="present"
        self.assertEqual(self.report()["complete_count"],0)

    def test_changed_evidence_is_not_current(self):
        self.evidence.write_text("changed after verification")
        self.assertEqual(self.report()["complete_count"],0)

    def test_missing_evidence_is_not_current(self):
        self.evidence.unlink()
        self.assertEqual(self.report()["complete_count"],0)

    def test_unbounded_or_unverified_configuration_not_complete(self):
        self.catalog["capabilities"][0]["configuration_contract"]="pending"
        self.assertEqual(self.report()["complete_count"],0)

    def test_unverified_backend_matrix_not_complete(self):
        self.catalog["capabilities"][0]["backend_matrix"]["status"]="pending"
        self.assertEqual(self.report()["complete_count"],0)

    def test_draft_denominator_withholds_percentage_and_release(self):
        self.catalog["denominator_status"]="proposed"
        self.catalog["approval"]=None
        r=self.report()
        self.assertIsNone(r["capability_percent"])
        self.assertFalse(r["release_gate_passed"])
        self.assertEqual(r["complete_count"],1)

    def test_empty_denominator_is_unknown_not_hundred_percent(self):
        self.catalog["capabilities"]=[]
        self.records[0]["disposition"]="unsupported"
        for row in self.records: row["capability_ids"]=[]
        self.rewrite_inventory()
        r=self.report()
        self.assertIsNone(r["capability_percent"])
        self.assertFalse(r["release_gate_passed"])

    def test_missing_mandatory_family_fails_even_at_hundred_percent(self):
        self.catalog["required_families"].append("recurrent")
        r=self.report()
        self.assertEqual(r["capability_percent"],100.0)
        self.assertFalse(r["release_gate_passed"])

    def test_reject_duplicate_capability_identity(self):
        self.catalog["capabilities"].append(copy.deepcopy(self.catalog["capabilities"][0]))
        with self.assertRaises(ValueError):self.report()

    def test_reject_alias_as_extra_operation(self):
        extra=copy.deepcopy(self.catalog["capabilities"][0]);extra["id"]="alias-op";extra["canonical_symbol"]="lib.Alias"
        self.catalog["capabilities"].append(extra)
        with self.assertRaises(ValueError):self.report()

    def test_reject_same_api_and_modes_under_different_id(self):
        extra=copy.deepcopy(self.catalog["capabilities"][0]);extra["id"]="duplicate-op"
        self.catalog["capabilities"].append(extra)
        with self.assertRaises(ValueError):self.report()

    def test_evidence_cannot_escape_repository(self):
        self.catalog["capabilities"][0]["evidence"]["execution"][0]["path"]="../outside.txt"
        with self.assertRaises(ValueError):self.report()

    def test_evidence_scope_must_match_modes(self):
        self.catalog["capabilities"][0]["evidence"]["execution"][0]["scope"]="another:default"
        self.assertEqual(self.report()["complete_count"],0)

    def test_inventory_integrity_and_dispositions_required(self):
        self.snapshot.write_text("[]")
        with self.assertRaises(ValueError):self.report()

    def test_unexplained_exclusion_rejected(self):
        self.records[2]["disposition"]="excluded";self.records[2]["reason"]=""
        self.rewrite_inventory()
        with self.assertRaises(ValueError):self.report()

    def test_frozen_catalog_requires_approval_record(self):
        self.catalog["approval"]=None
        with self.assertRaises(ValueError):self.report()

    def test_proposed_modes_cannot_be_complete(self):
        self.catalog["capabilities"][0]["mode_scope_status"]="proposed"
        self.assertEqual(self.report()["complete_count"],0)

    def test_selected_unit_requires_reciprocal_inventory_mapping(self):
        self.records[0]["capability_ids"]=[]
        self.rewrite_inventory()
        with self.assertRaises(ValueError):self.report()

    def test_display_rounding_cannot_pass_release_threshold(self):
        self.assertFalse(coverage.meets_threshold(89999,100000,90))
        self.assertTrue(coverage.meets_threshold(90000,100000,90))
        self.assertFalse(coverage.meets_threshold(0,0,90))

    def rewrite_inventory(self):
        self.snapshot.write_text(json.dumps(self.records))
        self.catalog["inventory"]["sha256"]=hashlib.sha256(self.snapshot.read_bytes()).hexdigest()

if __name__=="__main__":unittest.main()
