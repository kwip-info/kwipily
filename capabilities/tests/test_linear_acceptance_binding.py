import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
RECEIPT = "docs/production/TORCH_LINEAR_ACCEPTANCE_20261008.md"
MODES = ["inference", "training", "input-gradients", "state-roundtrip"]
spec = importlib.util.spec_from_file_location("linear_binding_coverage", ROOT / "scripts/coverage_report.py")
coverage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coverage)


class LinearAcceptanceBindingTests(unittest.TestCase):
    def setUp(self):
        self.catalog = json.loads((ROOT / "capabilities/catalogs/v1/pytorch.json").read_text())
        self.unit = next(x for x in self.catalog["capabilities"] if x["id"] == "pytorch.torch.nn.Linear")

    def testReviewedUnitHasCurrentEvidenceAcrossAllFourPreservedModes(self):
        result = coverage.report_catalog(self.catalog, ROOT)
        self.assertIn(self.unit["id"], result["complete_ids"])
        self.assertEqual(self.unit["mode_scope"], MODES)
        self.assertEqual(self.unit["mode_scope_status"], "reviewed")
        self.assertEqual(self.unit["configuration_contract"], "verified")
        self.assertEqual(self.unit["backend_matrix"]["status"], "verified")
        self.assertEqual(self.unit["backend_matrix"]["reference"], RECEIPT)
        self.assertEqual(result["minimum_complete_units"], 80)
        self.assertFalse(result["release_gate_passed"])

    def testFrozenReceiptIsCurrentAndBoundAcrossEveryEvidenceDimensionAndMode(self):
        path = ROOT / RECEIPT
        self.assertTrue(path.is_file(), "Linear requires its own frozen acceptance receipt")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        for dimension in coverage.REQUIRED_EVIDENCE:
            for mode in MODES:
                with self.subTest(dimension=dimension, mode=mode):
                    matching = [item for item in self.unit["evidence"].get(dimension, [])
                                if item["path"] == RECEIPT and item.get("scope") == f'{self.unit["id"]}:{mode}']
                    self.assertTrue(matching)
                    self.assertTrue(all(item.get("status") == "verified" and item.get("sha256") == digest for item in matching))

    def testStaleEngineOrNumericalTestCannotHideBehindUnchangedReceipt(self):
        for path in ["runner/engine.py", "runner/tests/test_linear_axes_independent.py"]:
            with self.subTest(path=path):
                catalog = copy.deepcopy(self.catalog)
                unit = next(x for x in catalog["capabilities"] if x["id"] == self.unit["id"])
                source = next(x for x in unit["evidence"].get("execution", []) if x["path"] == path)
                source["sha256"] = "0" * 64
                self.assertNotIn(unit["id"], coverage.report_catalog(catalog, ROOT)["complete_ids"])

    def testMissingGradientLessonOrStateRoundtripEvidenceWithholdsCredit(self):
        for dimension, mode in [("lesson", "input-gradients"), ("persistence", "state-roundtrip")]:
            with self.subTest(dimension=dimension, mode=mode):
                catalog = copy.deepcopy(self.catalog)
                unit = next(x for x in catalog["capabilities"] if x["id"] == self.unit["id"])
                unit["evidence"][dimension] = [x for x in unit["evidence"].get(dimension, [])
                                               if x.get("scope") != f'{unit["id"]}:{mode}']
                self.assertNotIn(unit["id"], coverage.report_catalog(catalog, ROOT)["complete_ids"])

    def testUnreviewedModeBoundaryWithholdsCredit(self):
        self.unit["mode_scope_status"] = "pending-operation-boundary-review"
        self.assertNotIn(self.unit["id"], coverage.report_catalog(self.catalog, ROOT)["complete_ids"])
