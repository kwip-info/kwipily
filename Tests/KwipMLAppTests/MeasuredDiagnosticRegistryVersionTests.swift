import XCTest
import KwipMLCore
@testable import KwipMLApp

final class MeasuredDiagnosticRegistryVersionTests: XCTestCase {
    private func input() -> MeasuredDiagnosticInput {
        .init(runID: "00000000-0000-0000-0000-000000000001", requestSHA256: String(repeating: "a", count: 64), metrics: (1...20).map {
            .init(epoch: $0, trainLoss: $0 <= 5 ? 1 : 0.7, validationLoss: $0 <= 5 ? 1.1 : 1.4, accuracy: 0.5)
        })
    }
    private func context(_ input: MeasuredDiagnosticInput) -> MeasuredGeneralizationCheck.VerifiedContext {
        .init(runID: input.runID, requestSHA256: input.requestSHA256, partitionSHA256: String(repeating: "b", count: 64), evaluationSHA256: String(repeating: "c", count: 64), completeEpochCapture: true, unchangedPartitions: true, unchangedEvaluation: true, binaryCrossEntropy: true)
    }
    func testPinnedV1EvaluationAndStrictArchiveRemainUnchanged() throws {
        let value = input()
        let v1 = try MeasuredDiagnostics.evaluate(value)
        XCTAssertEqual(v1.registryVersion, 1)
        XCTAssertEqual(v1.checks.count, 9)
        XCTAssertEqual(v1.inputSHA256, try WorkflowArchiveCodec.hash(WorkflowArchiveCodec.encode(value)))
        XCTAssertEqual(try MeasuredDiagnosticReport.decode(JSONEncoder().encode(v1)), v1)
        XCTAssertFalse(v1.checks.contains { $0.ruleID == MeasuredGeneralizationCheck.rule.id })
        XCTAssertEqual(try MeasuredDiagnostics.rules(forVersion: 1), MeasuredDiagnostics.rules)
        XCTAssertEqual(try MeasuredDiagnostics.rules(forVersion: 2), MeasuredDiagnostics.currentRules)
        XCTAssertThrowsError(try MeasuredDiagnostics.rules(forVersion: 3))
    }
    func testCurrentV2IncludesDivergenceWithMeasuredFactsAndRoundTrips() throws {
        let value = input()
        let report = try MeasuredDiagnostics.evaluateCurrent(value, context: context(value))
        XCTAssertEqual(report.registryVersion, 2)
        XCTAssertEqual(report.checks.map(\.ruleID), MeasuredDiagnostics.currentRules.map(\.id))
        XCTAssertEqual(report.findings.last?.ruleID, MeasuredGeneralizationCheck.rule.id)
        XCTAssertEqual(try MeasuredDiagnosticReport.decode(JSONEncoder().encode(report)), report)
    }
    func testEveryVerifiedContextFieldChangesTheV2Fingerprint() throws {
        let value = input(), original = context(input())
        let baseline = try MeasuredDiagnostics.evaluateCurrent(value, context: original).inputSHA256
        var variants: [MeasuredGeneralizationCheck.VerifiedContext] = []
        var changed = original; changed.runID = "00000000-0000-0000-0000-000000000002"; variants.append(changed)
        changed = original; changed.requestSHA256 = String(repeating: "d", count: 64); variants.append(changed)
        changed = original; changed.partitionSHA256 = String(repeating: "d", count: 64); variants.append(changed)
        changed = original; changed.evaluationSHA256 = String(repeating: "d", count: 64); variants.append(changed)
        changed = original; changed.completeEpochCapture = false; variants.append(changed)
        changed = original; changed.unchangedPartitions = false; variants.append(changed)
        changed = original; changed.unchangedEvaluation = false; variants.append(changed)
        changed = original; changed.binaryCrossEntropy = false; variants.append(changed)
        for variant in variants {
            XCTAssertNotEqual(try MeasuredDiagnostics.evaluateCurrent(value, context: variant).inputSHA256, baseline)
        }
        let missing = try MeasuredDiagnostics.evaluateCurrent(value)
        XCTAssertNotEqual(missing.inputSHA256, baseline)
        XCTAssertEqual(missing.checks.last?.status, .withheld)
        XCTAssertNotEqual(missing.inputSHA256, try MeasuredDiagnostics.evaluate(value).inputSHA256)
        XCTAssertEqual(missing.inputSHA256, try MeasuredDiagnostics.evaluateCurrent(value).inputSHA256)
    }
    func testVersionIdentityOrderUnknownChecksAndStrictKeysAreRejected() throws {
        let v1 = try MeasuredDiagnostics.evaluate(input())
        let v2 = try MeasuredDiagnostics.evaluateCurrent(input(), context: context(input()))
        func mutated(_ report: MeasuredDiagnosticReport, _ change: (inout [String: Any]) -> Void) throws -> Data {
            var object = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(report)) as? [String: Any])
            change(&object)
            return try JSONSerialization.data(withJSONObject: object)
        }
        XCTAssertThrowsError(try MeasuredDiagnosticReport.decode(mutated(v1) { $0["registryVersion"] = 2 }))
        XCTAssertThrowsError(try MeasuredDiagnosticReport.decode(mutated(v2) { $0["registryVersion"] = 1 }))
        XCTAssertThrowsError(try MeasuredDiagnosticReport.decode(mutated(v2) { $0["registryVersion"] = 99 }))
        XCTAssertThrowsError(try MeasuredDiagnosticReport.decode(mutated(v2) { $0["unexpected"] = true }))
        XCTAssertThrowsError(try MeasuredDiagnosticReport.decode(mutated(v2) {
            var checks = $0["checks"] as! [[String: Any]]; checks.swapAt(0, 1); $0["checks"] = checks
        }))
        XCTAssertThrowsError(try MeasuredDiagnosticReport.decode(mutated(v2) {
            var checks = $0["checks"] as! [[String: Any]]; checks[0]["ruleID"] = "unknown.v1"; $0["checks"] = checks
        }))
        XCTAssertThrowsError(try MeasuredDiagnosticReport.decode(mutated(v2) {
            var findings = $0["findings"] as! [[String: Any]]; findings[0]["ruleID"] = "unknown.v1"; $0["findings"] = findings
        }))
    }
}
