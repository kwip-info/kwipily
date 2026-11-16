import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("softmax_binding_coverage", ROOT / "scripts/coverage_report.py")
coverage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coverage)

class SoftmaxAcceptanceBindingTests(unittest.TestCase):
    def setUp(self):
        self.catalog = json.loads((ROOT / "capabilities/catalogs/v1/pytorch.json").read_text())
        self.unit = next(x for x in self.catalog["capabilities"] if x["id"] == "pytorch.torch.nn.Softmax")
    def testReviewedUnitHasCurrentEvidenceAcrossItsThreeModes(self):
        result = coverage.report_catalog(self.catalog, ROOT)
        self.assertIn(self.unit["id"], result["complete_ids"])
        self.assertEqual(self.unit["mode_scope"], ["forward", "input-gradients", "shape-contract"])
        self.assertEqual(result["minimum_complete_units"], 80)
        self.assertFalse(result["release_gate_passed"])
    def testStaleVerifiedSourceCannotHideBehindAnUnchangedReceipt(self):
        source = next(x for x in self.unit["evidence"].get("execution", []) if x["path"] == "runner/engine.py")
        source["sha256"] = "0" * 64
        result = coverage.report_catalog(self.catalog, ROOT)
        self.assertNotIn(self.unit["id"], result["complete_ids"])
    def testUnreviewedModeOrMissingGradientTeachingEvidenceWithholdsCredit(self):
        self.unit["evidence"]["lesson"] = [x for x in self.unit["evidence"].get("lesson", []) if not x["scope"].endswith(":input-gradients")]
        result = coverage.report_catalog(self.catalog, ROOT)
        self.assertNotIn(self.unit["id"], result["complete_ids"])
