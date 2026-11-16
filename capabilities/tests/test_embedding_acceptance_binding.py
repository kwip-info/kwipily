import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
RECEIPT = "docs/production/TORCH_EMBEDDING_ACCEPTANCE_20261008.md"
MODES = ["inference", "training", "parameter-gradients", "state-roundtrip"]
spec = importlib.util.spec_from_file_location("embedding_binding_coverage", ROOT / "scripts/coverage_report.py")
coverage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coverage)


class EmbeddingAcceptanceBindingTests(unittest.TestCase):
    def setUp(self):
        self.catalog = json.loads((ROOT / "capabilities/catalogs/v1/pytorch.json").read_text())
        self.unit = next(x for x in self.catalog["capabilities"] if x["id"] == "pytorch.torch.nn.Embedding")

    def testReviewedUnitHasCurrentEvidenceAcrossAllFourPreservedModes(self):
        result = coverage.report_catalog(self.catalog, ROOT)
        self.assertIn(self.unit["id"], result["complete_ids"])
        self.assertEqual(self.unit["mode_scope"], MODES)
        self.assertEqual(self.unit["mode_scope_status"], "reviewed")
        self.assertEqual(self.unit["configuration_contract"], "verified")
        self.assertEqual(self.unit["backend_matrix"]["status"], "verified")
        self.assertEqual(self.unit["backend_matrix"]["reference"], RECEIPT)
        self.assertEqual(self.unit["lesson_id"], "assessment.embedding-gather")
        self.assertEqual(self.unit["lesson_ids"], ["assessment.embedding-gather", "assessment.embedding-padding-frequency", "assessment.embedding-renormalize"])
        self.assertEqual(self.unit["example_id"], "prepared-embedding-classifier")
        self.assertIn("cpu-float32", self.unit["backend_matrix"]["cases"])
        self.assertIn("mps-float32", self.unit["backend_matrix"]["cases"])
        self.assertEqual(result["minimum_complete_units"], 80)
        self.assertFalse(result["release_gate_passed"])

    def testFrozenReceiptIsCurrentAndBoundAcrossEveryEvidenceDimensionAndMode(self):
        path = ROOT / RECEIPT
        self.assertTrue(path.is_file(), "Embedding requires its own frozen acceptance receipt")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        for dimension in coverage.REQUIRED_EVIDENCE:
            for mode in MODES:
                with self.subTest(dimension=dimension, mode=mode):
                    matching = [item for item in self.unit["evidence"].get(dimension, [])
                                if item["path"] == RECEIPT and item.get("scope") == f'{self.unit["id"]}:{mode}']
                    self.assertTrue(matching)
                    self.assertTrue(all(item.get("status") == "verified" and item.get("sha256") == digest for item in matching))

    def testStaleWorkerNumericalManifestAssessmentOrBridgeCannotHideBehindReceipt(self):
        dependencies = [("execution", "runner/embedding.py"),
                        ("execution", "runner/tests/test_embedding_axes_independent.py"),
                        ("configuration", "components/embeddings/bundle.json"),
                        ("lesson", "Sources/KwipMLCore/EmbeddingAssessment.swift"),
                        ("lesson", "runner/learning_embedding.py"),
                        ("validation", "Sources/KwipMLCore/EmbeddingCapabilityBridge.swift")]
        for dimension, path in dependencies:
            with self.subTest(dimension=dimension, path=path):
                catalog = copy.deepcopy(self.catalog)
                unit = next(x for x in catalog["capabilities"] if x["id"] == self.unit["id"])
                sources = [x for x in unit["evidence"].get(dimension, []) if x["path"] == path]
                self.assertTrue(sources, "Acceptance must bind the actual dependency, not only its receipt")
                sources[0]["sha256"] = "0" * 64
                self.assertNotIn(unit["id"], coverage.report_catalog(catalog, ROOT)["complete_ids"])

    def testMissingParameterGradientLessonOrStateRoundtripEvidenceWithholdsCredit(self):
        for dimension, mode in [("lesson", "parameter-gradients"), ("persistence", "state-roundtrip")]:
            with self.subTest(dimension=dimension, mode=mode):
                catalog = copy.deepcopy(self.catalog)
                unit = next(x for x in catalog["capabilities"] if x["id"] == self.unit["id"])
                unit["evidence"][dimension] = [x for x in unit["evidence"].get(dimension, [])
                                               if x.get("scope") != f'{unit["id"]}:{mode}']
                self.assertNotIn(unit["id"], coverage.report_catalog(catalog, ROOT)["complete_ids"])

    def testUnreviewedModeBoundaryWithholdsCredit(self):
        self.unit["mode_scope_status"] = "pending-operation-boundary-review"
        self.assertNotIn(self.unit["id"], coverage.report_catalog(self.catalog, ROOT)["complete_ids"])
