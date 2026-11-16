import XCTest
@testable import KwipMLApp

final class RunDiagnosticsTests:XCTestCase {
    private func series(_ count:Int,train:(Int)->Double,validation:(Int)->Double)->[EpochMetric] {
        (0..<count).map {EpochMetric(epoch:$0+1,trainLoss:train($0),validationLoss:validation($0),accuracy:0.75)}
    }
    func testHealthyAndInsufficientDataDoNotTriggerTrendClaims() {
        let healthy=series(30,train:{1-Double($0)*0.02},validation:{1.1-Double($0)*0.02})
        XCTAssertTrue(RunDiagnostics.evaluate(healthy).isEmpty)
        let stable=series(30,train:{_ in 0.4},validation:{_ in 0.8})
        XCTAssertTrue(RunDiagnostics.evaluate(stable).isEmpty,"A static gap alone is not evidence of rising validation loss.")
        let short=series(9,train:{1-Double($0)*0.09},validation:{0.2+Double($0)})
        XCTAssertTrue(RunDiagnostics.evaluate(short).isEmpty)
        XCTAssertTrue(RunDiagnostics.evaluate([]).isEmpty)
    }
    func testDivergenceReportsQuantifiedEvidenceWithoutDiagnosisAsFact() {
        let values=series(20,train:{1-Double($0)*0.03},validation:{0.5+Double($0)*0.04})
        let diagnostics=RunDiagnostics.evaluate(values)
        let issue=diagnostics.first {$0.id == "trend.generalization-gap.v1"}
        XCTAssertNotNil(issue)
        XCTAssertTrue(issue?.title.contains("Possible") == true)
        XCTAssertTrue(issue?.evidence.contains("→") == true)
        XCTAssertTrue(issue?.evidence.contains("20") == true)
        XCTAssertFalse(issue?.suggestion.lowercased().contains("softmax") == true)
    }
    func testLossGrowthDoesNotClaimUnmeasuredGradients() {
        let values=series(20,train:{_ in 0.3},validation:{_ in 0.4})
        var growing=values
        for i in 15..<20 {growing[i]=EpochMetric(epoch:i+1,trainLoss:2,validationLoss:2.2,accuracy:0.4)}
        let issue=RunDiagnostics.evaluate(growing).first {$0.id == "trend.loss-growth.v1"}
        XCTAssertNotNil(issue)
        XCTAssertTrue(issue?.evidence.contains("0.3000 → 2.0000") == true)
        XCTAssertTrue(issue?.suggestion.contains("not measured") == true)
    }
    func testNonfiniteMeasurementsReportedWithoutTrendAnalysis() {
        var values=series(20,train:{_ in 0.5},validation:{_ in 0.6})
        values[5]=EpochMetric(epoch:6,trainLoss:.infinity,validationLoss:.nan,accuracy:0.5)
        let issues=RunDiagnostics.evaluate(values)
        XCTAssertEqual(issues.map(\.id),["measurement.nonfinite.v1"])
        XCTAssertTrue(issues.first?.evidence.contains("6") == true)
    }
    func testKnownErrorsPreserveExactEvidenceAndUnknownErrorsStayGeneric() {
        for (error,id) in [("CSV row 7: features and target must be numeric.","error.data.v1"),("branch-a: receives 8 features but expects 3.","error.shape.v1"),("MPS backend unavailable on this machine.","error.device.v1"),("opaque problem xyz","error.worker.v1")] {
            let issues=RunDiagnostics.evaluate([],error:error)
            XCTAssertEqual(issues.count,1);XCTAssertEqual(issues.first?.id,id);XCTAssertEqual(issues.first?.evidence,error)
        }
        XCTAssertTrue(RunDiagnostics.evaluate([],error:"  ").isEmpty)
    }
}
