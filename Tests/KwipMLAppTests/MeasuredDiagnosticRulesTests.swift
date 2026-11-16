import XCTest
import KwipMLCore
@testable import KwipMLApp

final class MeasuredDiagnosticRulesTests:XCTestCase {
    func testReviewedMeasuredRegistryAndMissingEvidence()throws {
        let report=try MeasuredDiagnostics.evaluate(input())
        XCTAssertEqual(report.registryVersion,1)
        XCTAssertEqual(report.checks.count,9)
        XCTAssertTrue(report.findings.isEmpty)
        XCTAssertTrue(report.checks.allSatisfy{$0.status == .withheld})
        XCTAssertEqual(try MeasuredDiagnosticReport.decode(JSONEncoder().encode(report)),report)
        var modified=report;modified.registryVersion=2
        XCTAssertThrowsError(try modified.validate())
        XCTAssertTrue(MeasuredDiagnostics.rules.allSatisfy{!$0.prerequisites.isEmpty && !$0.confounders.isEmpty && !$0.thresholds.isEmpty && !$0.experiment.isEmpty})
    }
    func testGradientFactsRequireMeasuredEvidenceAndSuppressOverlappingAdvice()throws {
        var observed=input();observed.gradients=[batch(nonfinite:1)]
        XCTAssertEqual(try ids(observed),["gradient.nonfinite.v1"])
        observed.gradients=[batch(missing:1)]
        XCTAssertEqual(try ids(observed),["gradient.missing.v1"])
        observed.gradients=(1...3).map{batch(epoch:$0,zeros:10,norm:0)}
        XCTAssertEqual(try ids(observed),["gradient.zero.v1"])
        observed.gradients=(1...2).map{batch(epoch:$0,zeros:10,norm:0)}
        XCTAssertFalse(try ids(observed).contains("gradient.zero.v1"))
        observed.gradients=(1...3).map{batch(epoch:$0,zeros:0,norm:1)}
        XCTAssertTrue(try ids(observed).isEmpty)
        observed.gradients=[batch(nonfinite:1,missing:1)]
        XCTAssertEqual(try ids(observed),["gradient.nonfinite.v1"],"Nonfinite evidence takes precedence over lower-specificity gradient advice")
    }
    func testGradientGrowthIsRelativeToObservedBaselineNotInventedUniversalNorm()throws {
        var observed=input()
        observed.gradients=(1...10).map{batch(epoch:$0,norm:$0<=5 ? 0.01:2)}
        XCTAssertEqual(try ids(observed),["gradient.growth.v1"])
        observed.gradients=(1...10).map{batch(epoch:$0,norm:500)}
        XCTAssertTrue(try ids(observed).isEmpty,"Large stable norms are not growth")
        observed.gradients=(1...10).map{batch(epoch:$0,norm:$0<=5 ? 0:0.1)}
        XCTAssertTrue(try ids(observed).isEmpty,"Near-zero denominator must not create growth advice")
    }
    func testActivationFindingsNameSmallSamplesWithoutClaimingDeadNetworks()throws {
        var observed=input();observed.activations=[.init(nodeID:"r",component:"relu",values:Array(repeating:0,count:8))]
        let zero=try MeasuredDiagnostics.evaluate(observed)
        XCTAssertEqual(Set(zero.findings.map(\.ruleID)),["activation.zero-sample.v1"])
        let finding=try XCTUnwrap(zero.findings.first)
        XCTAssertTrue(finding.interpretation.contains("sample"))
        XCTAssertFalse(finding.interpretation.contains("dead neuron"))
        observed.activations=[.init(nodeID:"r",component:"relu",values:[0,0,0,0,0,0,0,1])]
        XCTAssertTrue(try ids(observed).isEmpty)
        observed.activations=[.init(nodeID:"t",component:"tanh",values:Array(repeating:0.999,count:8))]
        XCTAssertEqual(try ids(observed),["activation.saturation-sample.v1"])
        observed.activations=[.init(nodeID:"t",component:"tanh",values:Array(repeating:0.5,count:8))]
        XCTAssertTrue(try ids(observed).isEmpty)
        observed.activations=[.init(nodeID:"t",component:"tanh",values:[])]
        XCTAssertTrue(try ids(observed).isEmpty,"Privacy-omitted samples must withhold")
    }
    func testPlateauAndSmallHoldoutUseDisclosedMeasuredBaselines()throws {
        var observed=input(metrics:(1...20).map{EpochMetric(epoch:$0,trainLoss:0.7,validationLoss:0.7,accuracy:0.5)})
        XCTAssertEqual(try ids(observed),["optimization.plateau.v1"])
        observed.continuation=true
        XCTAssertTrue(try ids(observed).isEmpty,"Continuation boundary cannot be treated as one controlled history")
        observed=input(metrics:(1...20).map{EpochMetric(epoch:$0,trainLoss:0.7-Double($0)*0.02,validationLoss:0.7-Double($0)*0.01,accuracy:0.6)})
        XCTAssertTrue(try ids(observed).isEmpty)
        observed=input();observed.validationRows=10
        XCTAssertEqual(try ids(observed),["data.coarse-holdout.v1"])
        observed.validationRows=100
        XCTAssertTrue(try ids(observed).isEmpty)
    }
    func testMemoryErrorRetainsFactWithoutAssumingCauseAndRejectsIncidentalMemoryText()throws {
        var observed=input();observed.rawError="MPS backend out of memory: allocation failed"
        XCTAssertEqual(try ids(observed),["runtime.memory.v1"])
        observed.rawError="memory policy loaded; invalid CSV column"
        XCTAssertTrue(try ids(observed).isEmpty)
        observed.rawError="ResourceExhaustedError: dataset iterator quota exceeded"
        XCTAssertTrue(try ids(observed).isEmpty,"A framework resource error without allocation evidence is not necessarily memory exhaustion")
    }
    func testInputPinsChangeWithEvidenceAndMalformedCountsCannotBecomeFindings()throws {
        let a=try MeasuredDiagnostics.evaluate(input())
        var changed=input();changed.rawError="different evidence"
        XCTAssertNotEqual(a.inputSHA256,try MeasuredDiagnostics.evaluate(changed).inputSHA256)
        changed=input();changed.gradients=[batch(zeros:11)]
        XCTAssertThrowsError(try MeasuredDiagnostics.evaluate(changed))
        changed=input();changed.activations=[.init(nodeID:"x",component:"relu",values:[.infinity])]
        XCTAssertThrowsError(try MeasuredDiagnostics.evaluate(changed))
    }
    private func input(metrics:[EpochMetric]=[])->MeasuredDiagnosticInput {
        .init(runID:"9C6FD620-44F3-4A20-810C-7D2B6085D0DE",requestSHA256:String(repeating:"a",count:64),metrics:metrics)
    }
    private func batch(epoch:Int=1,nonfinite:Int=0,missing:Int=0,zeros:Int=0,norm:Double=1)->MeasuredGradientBatch {
        .init(epoch:epoch,batch:0,presentElements:10,nonfiniteElements:nonfinite,zeroElements:zeros,missingParameters:missing,l2Norm:nonfinite>0 ? nil:norm)
    }
    private func ids(_ input:MeasuredDiagnosticInput)throws->Set<String> {Set(try MeasuredDiagnostics.evaluate(input).findings.map(\.ruleID))}
}
