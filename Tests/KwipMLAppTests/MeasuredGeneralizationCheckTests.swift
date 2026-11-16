import XCTest
@testable import KwipMLApp

final class MeasuredGeneralizationCheckTests: XCTestCase {
    private func input(_ rows: [EpochMetric]? = nil) -> MeasuredDiagnosticInput {
        MeasuredDiagnosticInput(runID: "00000000-0000-0000-0000-000000000001", requestSHA256: String(repeating: "a", count: 64), metrics: rows ?? (1...20).map {
            EpochMetric(epoch: $0, trainLoss: $0 <= 5 ? 1 : 0.7, validationLoss: $0 <= 5 ? 1.1 : 1.4, accuracy: 0.5)
        })
    }
    private func context(_ value: MeasuredDiagnosticInput) -> MeasuredGeneralizationCheck.VerifiedContext {
        .init(runID: value.runID, requestSHA256: value.requestSHA256,
              partitionSHA256: String(repeating: "b", count: 64), evaluationSHA256: String(repeating: "c", count: 64),
              completeEpochCapture: true, unchangedPartitions: true, unchangedEvaluation: true,
              binaryCrossEntropy: true)
    }
    func testMeasuredDivergenceCarriesActualWindowFactsWithoutProvenCause() throws {
        let value = input()
        let result = try MeasuredGeneralizationCheck.evaluate(value, context: context(value))
        XCTAssertEqual(result.check.status, .emitted)
        let finding = try XCTUnwrap(result.finding)
        XCTAssertEqual(finding.ruleID, "generalization.divergence.v1")
        XCTAssertEqual(finding.values["trainingStartMean"]!, 1, accuracy: 1e-12)
        XCTAssertEqual(finding.values["validationEndMean"]!, 1.4, accuracy: 1e-12)
        XCTAssertEqual(finding.references.count, 20)
        XCTAssertTrue(finding.interpretation.contains("does not prove"))
    }
    func testAbsentUnboundOrUnverifiedContextWithholds() throws {
        let value = input()
        XCTAssertEqual(try MeasuredGeneralizationCheck.evaluate(value).check.status, .withheld)
        var evidence = context(value)
        evidence.requestSHA256 = String(repeating: "d", count: 64)
        XCTAssertEqual(try MeasuredGeneralizationCheck.evaluate(value, context: evidence).check.status, .withheld)
        evidence = context(value); evidence.completeEpochCapture = false
        XCTAssertEqual(try MeasuredGeneralizationCheck.evaluate(value, context: evidence).check.status, .withheld)
        evidence = context(value); evidence.unchangedPartitions = false
        XCTAssertEqual(try MeasuredGeneralizationCheck.evaluate(value, context: evidence).check.status, .withheld)
        evidence = context(value); evidence.unchangedEvaluation = false
        XCTAssertEqual(try MeasuredGeneralizationCheck.evaluate(value, context: evidence).check.status, .withheld)
        evidence = context(value); evidence.binaryCrossEntropy = false
        XCTAssertEqual(try MeasuredGeneralizationCheck.evaluate(value, context: evidence).check.status, .withheld)
        evidence = context(value); evidence.partitionSHA256 = "missing"
        XCTAssertEqual(try MeasuredGeneralizationCheck.evaluate(value, context: evidence).check.status, .withheld)
    }
    func testContinuationBackendMissingDuplicateAndReorderedEpochsWithhold() throws {
        let initial = input()
        var variants: [MeasuredDiagnosticInput] = []
        var changed = initial; changed.continuation = true; variants.append(changed)
        changed = initial; changed.backend = "tensorflow"; variants.append(changed)
        changed = initial; changed.metrics.remove(at: 7); variants.append(changed)
        changed = initial; changed.metrics[8] = changed.metrics[7]; variants.append(changed)
        changed = initial; changed.metrics.swapAt(7, 8); variants.append(changed)
        for value in variants {
            XCTAssertEqual(try MeasuredGeneralizationCheck.evaluate(value, context: context(value)).check.status, .withheld)
        }
    }
    func testInvalidMetricsWithhold() throws {
        for bad in [Double.nan, Double.infinity, -1] {
            var value = input()
            value.metrics[9] = MeasuredMetric(.init(epoch: 10, trainLoss: bad, validationLoss: 1, accuracy: 0.5))
            XCTAssertEqual(try MeasuredGeneralizationCheck.evaluate(value, context: context(value)).check.status, .withheld)
        }
        var value = input()
        value.metrics[9] = MeasuredMetric(.init(epoch: 10, trainLoss: 1, validationLoss: 1, accuracy: 1.1))
        XCTAssertEqual(try MeasuredGeneralizationCheck.evaluate(value, context: context(value)).check.status, .withheld)
    }
    func testOverflowingDerivedGapAndLateStartingCaptureWithhold() throws {
        let large = Double.greatestFiniteMagnitude / 1.5
        let rows = (1...20).map { EpochMetric(epoch: $0, trainLoss: $0 <= 5 ? large : 0, validationLoss: $0 <= 5 ? 0 : large, accuracy: 0.5) }
        let value = input(rows)
        XCTAssertEqual(try MeasuredGeneralizationCheck.evaluate(value, context: context(value)).check.status, .withheld)
        let late = input((2...21).map { EpochMetric(epoch: $0, trainLoss: 1, validationLoss: 2, accuracy: 0.5) })
        XCTAssertEqual(try MeasuredGeneralizationCheck.evaluate(late, context: context(late)).check.status, .withheld)
    }
    func testConstantGapAndJointlyFallingLossAreNotDivergence() throws {
        for rows in [
            (1...20).map { EpochMetric(epoch: $0, trainLoss: 0.2, validationLoss: 0.9, accuracy: 0.5) },
            (1...20).map { EpochMetric(epoch: $0, trainLoss: $0 <= 5 ? 1 : 0.7, validationLoss: $0 <= 5 ? 1.4 : 1.1, accuracy: 0.5) }
        ] {
            let value = input(rows)
            let result = try MeasuredGeneralizationCheck.evaluate(value, context: context(value))
            XCTAssertEqual(result.check.status, .notTriggered)
            XCTAssertNil(result.finding)
        }
    }
    func testBothRelativeChangesMustMeetTheHeuristic() throws {
        for (trainEnd, validationEnd, status) in [(0.89, 1.09, DiagnosticEvaluationStatus.notTriggered), (0.91, 1.11, .notTriggered), (0.89, 1.11, .emitted)] {
            let value = input((1...20).map { EpochMetric(epoch: $0, trainLoss: $0 <= 5 ? 1 : trainEnd, validationLoss: $0 <= 5 ? 1 : validationEnd, accuracy: 0.5) })
            XCTAssertEqual(try MeasuredGeneralizationCheck.evaluate(value, context: context(value)).check.status, status)
        }
    }
    func testLatestTwentyConsecutiveEpochsAndDisclosedThresholds() throws {
        let rows = (1...30).map { EpochMetric(epoch: $0, trainLoss: $0 <= 15 ? 1 : 0.7, validationLoss: $0 <= 15 ? 1.1 : 1.4, accuracy: 0.5) }
        let value = input(rows)
        let result = try MeasuredGeneralizationCheck.evaluate(value, context: context(value))
        XCTAssertEqual(result.finding?.references.first, "metrics.json:epoch=11")
        XCTAssertEqual(MeasuredGeneralizationCheck.rule.thresholds["observations"], 20)
        XCTAssertEqual(MeasuredGeneralizationCheck.rule.thresholds["minimumGapIncrease"], 0.05)
        XCTAssertTrue(MeasuredGeneralizationCheck.rule.confounders.contains { $0.contains("heuristic") })
    }
}
