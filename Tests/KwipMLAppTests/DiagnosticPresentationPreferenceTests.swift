import XCTest
@testable import KwipMLApp

final class DiagnosticPresentationPreferenceTests: XCTestCase {
    func testOnlyVerifiedCurrentDivergenceReplacesItsLiveLegacyCard() throws {
        let run = "00000000-0000-0000-0000-000000000001"
        let request = String(repeating: "a", count: 64)
        let input = MeasuredDiagnosticInput(runID: run, requestSHA256: request, metrics: (1...20).map {
            .init(epoch: $0, trainLoss: $0 <= 5 ? 1 : 0.7, validationLoss: $0 <= 5 ? 1 : 1.4, accuracy: 0.5)
        })
        let context = MeasuredGeneralizationCheck.VerifiedContext(runID: run, requestSHA256: request,
            partitionSHA256: String(repeating: "b", count: 64), evaluationSHA256: String(repeating: "c", count: 64),
            completeEpochCapture: true, unchangedPartitions: true, unchangedEvaluation: true, binaryCrossEntropy: true)
        let archive = try MeasuredDiagnostics.evaluate(input)
        let current = try MeasuredDiagnostics.evaluateCurrent(input, context: context)
        let unavailable = try MeasuredDiagnostics.evaluateCurrent(input)
        XCTAssertEqual(DiagnosticPresentationPreference.hiddenLegacyRuleIDs(measured: current), ["trend.generalization-gap.v1"])
        XCTAssertTrue(DiagnosticPresentationPreference.hiddenLegacyRuleIDs(measured: archive).isEmpty)
        XCTAssertTrue(DiagnosticPresentationPreference.hiddenLegacyRuleIDs(measured: unavailable).isEmpty)
        XCTAssertTrue(DiagnosticPresentationPreference.hiddenLegacyRuleIDs(measured: nil).isEmpty)
        XCTAssertTrue(archive.findings.isEmpty, "Presentation preference never rewrites the earlier report")
    }
}
