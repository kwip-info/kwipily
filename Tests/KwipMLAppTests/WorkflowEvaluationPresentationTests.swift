import XCTest
@testable import KwipMLApp

final class WorkflowEvaluationPresentationTests:XCTestCase {
    func testUndefinedMetricsExplainBothDenominators() {
        let value=WorkflowEvaluationClassMetric(classIndex:1,precision:nil,precisionUnavailableReason:"noPredictedObservations",recall:nil,recallUnavailableReason:"noActualObservations",support:0)
        XCTAssertEqual(WorkflowEvaluationPresentation.explanations(value),[
            "Precision is unavailable because no observations were predicted as class 1.",
            "Recall is unavailable because the test data contains no class 1 observations."
        ])
        let defined=WorkflowEvaluationClassMetric(classIndex:0,precision:0.5,precisionUnavailableReason:nil,recall:1,recallUnavailableReason:nil,support:2)
        XCTAssertTrue(WorkflowEvaluationPresentation.explanations(defined).isEmpty)
    }
    func testConfusionAccessibilityDescribesEachCountExactlyOnce() {
        XCTAssertEqual(WorkflowEvaluationPresentation.confusionDescription([[11,0],[12,0]]),
            "Test confusion counts. Actual class 0: predicted class 0, 11; predicted class 1, 0. Actual class 1: predicted class 0, 12; predicted class 1, 0.")
    }
}
