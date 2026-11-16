import XCTest
import Foundation
import KwipMLCore
@testable import KwipMLApp

/// Replays pinned real worker artifacts. Fixture identity is not a fabricated
/// app archive; the separate verified-History tests cover that trust bridge.
final class MeasuredGeneralizationTrainedCorpusTests: XCTestCase {
    func testActualLabelShiftAndHealthyControlReplayCurrentRegistry() throws {
        guard let path = ProcessInfo.processInfo.environment["KWIPML_GENERALIZATION_CORPUS"] else {
            throw XCTSkip("Generate generalization_training_fixtures.py and set KWIPML_GENERALIZATION_CORPUS.")
        }
        let root = URL(fileURLWithPath: path, isDirectory: true)
        let corpusBytes = try Data(contentsOf: root.appendingPathComponent("corpus.json"))
        XCTAssertEqual(WorkflowArchiveCodec.hash(corpusBytes), "ff9a7e6f0dff03dd9cd2838f352a3f5fe7bbf25579365326fd8ae250e07e922f")
        let corpus = try XCTUnwrap(JSONSerialization.jsonObject(with: corpusBytes) as? [String: Any])
        XCTAssertEqual(corpus["corpusID"] as? String, "kwip.cpu-trained-generalization-controls.v1")
        let files = try XCTUnwrap(corpus["files"] as? [[String: Any]])
        var verified: [String: Data] = [:]
        for pin in files {
            let relative = try XCTUnwrap(pin["path"] as? String)
            XCTAssertFalse(relative.hasPrefix("/")); XCTAssertFalse(relative.split(separator: "/").contains(".."))
            let url = root.appendingPathComponent(relative)
            XCTAssertTrue(url.resolvingSymlinksInPath().path.hasPrefix(root.resolvingSymlinksInPath().path + "/"))
            let data = try Data(contentsOf: url)
            XCTAssertEqual(data.count, pin["bytes"] as? Int)
            XCTAssertEqual(WorkflowArchiveCodec.hash(data), pin["sha256"] as? String)
            XCTAssertNil(verified.updateValue(data, forKey: relative))
        }
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        for pin in try XCTUnwrap(corpus["sourcePins"] as? [[String: Any]]) {
            let relative = try XCTUnwrap(pin["path"] as? String)
            XCTAssertEqual(WorkflowArchiveCodec.hash(try Data(contentsOf: repo.appendingPathComponent(relative))), pin["sha256"] as? String)
        }
        let cases = try XCTUnwrap(corpus["cases"] as? [[String: Any]])
        XCTAssertEqual(Set(cases.compactMap { $0["id"] as? String }), ["heldout-label-shift", "consistent-relation-control"])
        for fixture in cases {
            func artifact(_ key: String) throws -> Data {
                try XCTUnwrap(verified[try XCTUnwrap(fixture[key] as? String)])
            }
            let id = try XCTUnwrap(fixture["id"] as? String)
            let manifestBytes = try artifact("modelManifest")
            let manifest = try XCTUnwrap(JSONSerialization.jsonObject(with: manifestBytes) as? [String: Any])
            let runID = try XCTUnwrap(fixture["runID"] as? String)
            XCTAssertEqual(manifest["runID"] as? String, runID)
            XCTAssertEqual(manifest["status"] as? String, "completed")
            let rows = try XCTUnwrap(JSONSerialization.jsonObject(with: artifact("metricsArtifact")) as? [[String: Any]])
            let metrics = try rows.map { row in
                EpochMetric(epoch: try XCTUnwrap(row["epoch"] as? Int), trainLoss: try XCTUnwrap(row["trainLoss"] as? Double), validationLoss: try XCTUnwrap(row["validationLoss"] as? Double), accuracy: try XCTUnwrap(row["accuracy"] as? Double))
            }
            XCTAssertEqual(metrics.map(\.epoch), Array(1...20))
            let identity = WorkflowArchiveCodec.hash(manifestBytes)
            let input = MeasuredDiagnosticInput(runID: runID, requestSHA256: identity, metrics: metrics)
            let context = MeasuredGeneralizationCheck.VerifiedContext(runID: runID, requestSHA256: identity,
                partitionSHA256: WorkflowArchiveCodec.hash(try artifact("preparationManifest")),
                evaluationSHA256: identity, completeEpochCapture: true, unchangedPartitions: true,
                unchangedEvaluation: true, binaryCrossEntropy: true)
            let report = try MeasuredDiagnostics.evaluateCurrent(input, context: context)
            XCTAssertEqual(report.registryVersion, 2)
            let positive = id == "heldout-label-shift"
            XCTAssertEqual(report.checks.first { $0.ruleID == MeasuredGeneralizationCheck.rule.id }?.status, positive ? .emitted : .notTriggered)
            XCTAssertEqual(report.findings.contains { $0.ruleID == MeasuredGeneralizationCheck.rule.id }, positive)
            XCTAssertEqual(DiagnosticPresentationPreference.hiddenLegacyRuleIDs(measured: report), positive ? ["trend.generalization-gap.v1"] : [])
            XCTAssertEqual(try MeasuredDiagnosticReport.decode(JSONEncoder().encode(report)), report)
            let unavailable = try MeasuredDiagnostics.evaluateCurrent(input)
            XCTAssertEqual(unavailable.checks.first { $0.ruleID == MeasuredGeneralizationCheck.rule.id }?.status, .withheld)
            XCTAssertTrue(DiagnosticPresentationPreference.hiddenLegacyRuleIDs(measured: unavailable).isEmpty)
        }
    }
}
