import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
RECEIPT = "docs/production/TORCH_TENSOR_LAYOUT_ACCEPTANCE_20261008.md"
MODES = ["forward", "input-gradients", "shape-contract"]
OPERATIONS = ["reshape", "transpose", "flatten"]
spec = importlib.util.spec_from_file_location("tensor_layout_binding_coverage", ROOT / "scripts/coverage_report.py")
coverage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coverage)


class TensorLayoutAcceptanceBindingTests(unittest.TestCase):
    def setUp(self):
        self.catalog = json.loads((ROOT / "capabilities/catalogs/v1/pytorch.json").read_text())
        self.units = {operation: next(x for x in self.catalog["capabilities"]
                                     if x["id"] == "pytorch.torch." + operation)
                      for operation in OPERATIONS}

    def testThreeUnitsRetainExactReviewedParameterlessModesAndBackendBoundaries(self):
        result = coverage.report_catalog(self.catalog, ROOT)
        self.assertEqual(result["minimum_complete_units"], 80)
        self.assertFalse(result["release_gate_passed"])
        for operation, unit in self.units.items():
            with self.subTest(operation=operation):
                self.assertIn(unit["id"], result["complete_ids"])
                self.assertEqual(unit["mode_scope"], MODES)
                self.assertEqual(unit["mode_scope_status"], "reviewed")
                self.assertEqual(unit["configuration_contract"], "verified")
                self.assertEqual(unit["backend_matrix"]["status"], "verified")
                self.assertEqual(unit["backend_matrix"]["reference"], RECEIPT)
                self.assertIn("cpu-float32", unit["backend_matrix"]["cases"])
                self.assertIn("mps-float32", unit["backend_matrix"]["cases"])
                self.assertEqual(unit["lesson_id"], "assessment.tensor-" + operation)
                self.assertEqual(unit["example_id"], "kwip.example.tensor-layout-classifier.v1")

    def testSharedFrozenReceiptHasSeparateCurrentUnitModeBindingsAcrossEveryDimension(self):
        path = ROOT / RECEIPT
        self.assertTrue(path.is_file(), "The shared family receipt must explicitly accept each separate unit")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        for operation, unit in self.units.items():
            for dimension in coverage.REQUIRED_EVIDENCE:
                for mode in MODES:
                    with self.subTest(operation=operation, dimension=dimension, mode=mode):
                        matching = [item for item in unit["evidence"].get(dimension, [])
                                    if item["path"] == RECEIPT and item.get("scope") == f'{unit["id"]}:{mode}']
                        self.assertTrue(matching)
                        self.assertTrue(all(item.get("status") == "verified" and item.get("sha256") == digest for item in matching))

    def testChangedWorkerNumericalManifestOrLessonDependencyCannotHideBehindReceipt(self):
        dependencies = [("execution", "runner/tensor_layout.py"),
                        ("execution", "runner/tests/test_tensor_layout_axes_independent.py"),
                        ("configuration", "components/tensor-layout/bundle.json"),
                        ("lesson", "Sources/KwipMLCore/TensorTransformAssessment.swift")]
        for operation in OPERATIONS:
            for dimension, path in dependencies:
                with self.subTest(operation=operation, dimension=dimension, path=path):
                    catalog = copy.deepcopy(self.catalog)
                    unit = next(x for x in catalog["capabilities"] if x["id"] == self.units[operation]["id"])
                    matching = [x for x in unit["evidence"].get(dimension, []) if x["path"] == path]
                    self.assertTrue(matching, "Acceptance must bind the actual dependency, not only a narrative")
                    matching[0]["sha256"] = "0" * 64
                    self.assertNotIn(unit["id"], coverage.report_catalog(catalog, ROOT)["complete_ids"])

    def testMissingOwnGradientLessonEvidenceWithholdsOnlyThatUnitsCredit(self):
        for operation in OPERATIONS:
            with self.subTest(operation=operation):
                catalog = copy.deepcopy(self.catalog)
                unit = next(x for x in catalog["capabilities"] if x["id"] == self.units[operation]["id"])
                unit["evidence"]["lesson"] = [x for x in unit["evidence"].get("lesson", [])
                                               if x.get("scope") != f'{unit["id"]}:input-gradients']
                result = coverage.report_catalog(catalog, ROOT)
                self.assertNotIn(unit["id"], result["complete_ids"])
                for other in OPERATIONS:
                    if other != operation:
                        self.assertIn(self.units[other]["id"], result["complete_ids"], "Another operation's proof must remain separately accounted")

    def testUnreviewedUnitModeCannotBorrowAnotherOperationsAcceptance(self):
        for operation in OPERATIONS:
            with self.subTest(operation=operation):
                catalog = copy.deepcopy(self.catalog)
                unit = next(x for x in catalog["capabilities"] if x["id"] == self.units[operation]["id"])
                unit["mode_scope_status"] = "pending-operation-boundary-review"
                self.assertNotIn(unit["id"], coverage.report_catalog(catalog, ROOT)["complete_ids"])
