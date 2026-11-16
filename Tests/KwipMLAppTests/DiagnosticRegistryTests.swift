import XCTest
@testable import KwipMLApp

final class DiagnosticRegistryTests:XCTestCase {
    private func metrics(_ count:Int=20)->[EpochMetric] {(0..<count).map {EpochMetric(epoch:$0+1,trainLoss:1-Double($0)*0.03,validationLoss:0.5+Double($0)*0.04,accuracy:0.7)}}
    func testReviewedRegistryPinsSevenRulesAndRejectsDrift() throws {
        let registry=DiagnosticRuleRegistry.builtIn
        XCTAssertEqual(registry.rules.count,7)
        XCTAssertEqual(Set(registry.rules.map(\.id)).count,7)
        XCTAssertNoThrow(try registry.validate())
        XCTAssertTrue(registry.rules.allSatisfy {!$0.prerequisites.isEmpty && !$0.confounders.isEmpty && !$0.missingEvidence.isEmpty})
        var bad=registry;bad.schemaVersion=2;XCTAssertThrowsError(try bad.validate())
        bad=registry;bad.version=2;XCTAssertThrowsError(try bad.validate())
        bad=registry;bad.rules.append(bad.rules[0]);XCTAssertThrowsError(try bad.validate())
        bad=registry;bad.rules[1].thresholds["minimumObservations"]=9;XCTAssertThrowsError(try bad.validate())
        XCTAssertThrowsError(try RunDiagnostics.report(metrics(),registry:bad))
    }
    func testRawErrorAndInputFingerprintAreExactAndReportRoundTrips() throws {
        let error=" \nMPS BACKEND shape error\t "
        let report=try RunDiagnostics.report(metrics(),error:error,context:DiagnosticInputContext(runID:"run-local",sampling:.sampled))
        XCTAssertEqual(report.findings.first?.id,"error.device.v1")
        XCTAssertEqual(report.findings.first?.evidence,error)
        XCTAssertEqual(report.findings.first?.provenance?.context.sampling,.sampled)
        XCTAssertEqual(report.findings.first?.provenance?.registryVersion,1)
        XCTAssertEqual(report.findings.first?.provenance?.inputSHA256,report.inputSHA256)
        XCTAssertNotEqual(report.inputSHA256,try RunDiagnostics.report(metrics(),error:error.trimmingCharacters(in:.whitespacesAndNewlines)).inputSHA256)
        XCTAssertNotEqual(report.inputSHA256,try RunDiagnostics.report(Array(metrics().reversed()),error:error).inputSHA256)
        XCTAssertEqual(try DiagnosticReport.decode(JSONEncoder().encode(report)),report)
        var invalid=report;invalid.schemaVersion=99
        XCTAssertThrowsError(try DiagnosticReport.decode(JSONEncoder().encode(invalid)))
        invalid=report;invalid.registryVersion=99
        XCTAssertThrowsError(try invalid.validate())
    }
    func testMissingEvidenceIsExplicitAndNonfiniteReportIsJSONSafe() throws {
        let short=try RunDiagnostics.report(metrics(9))
        XCTAssertTrue(short.findings.isEmpty)
        XCTAssertEqual(short.evaluations.first {$0.ruleID == "trend.generalization-gap.v1"}?.reason,.insufficientObservations)
        XCTAssertEqual(short.evaluations.first {$0.ruleID == "error.worker.v1"}?.reason,.missingError)
        var nonfinite=metrics();nonfinite[0]=EpochMetric(epoch:1,trainLoss:.nan,validationLoss:1,accuracy:0.5)
        let report=try RunDiagnostics.report(nonfinite)
        XCTAssertEqual(report.findings.map(\.id),["measurement.nonfinite.v1"])
        XCTAssertEqual(report.evaluations.first {$0.ruleID == "trend.loss-growth.v1"}?.reason,.nonfiniteMeasurements)
        XCTAssertNoThrow(try DiagnosticReport.decode(JSONEncoder().encode(report)))
    }
    func testDerivedWindowReferencesLatestDuplicateAndNeverInterpolates() throws {
        var values=metrics()
        values.append(EpochMetric(epoch:20,trainLoss:0.1,validationLoss:2,accuracy:0.7))
        values.insert(EpochMetric(epoch:0,trainLoss:0,validationLoss:0,accuracy:1),at:0)
        let report=try RunDiagnostics.report(values)
        let finding=try XCTUnwrap(report.findings.first {$0.id == "trend.generalization-gap.v1"})
        let provenance=try XCTUnwrap(finding.provenance)
        XCTAssertEqual(provenance.observations.count,20)
        XCTAssertEqual(provenance.observations.last?.inputIndex,21)
        XCTAssertEqual(provenance.observations.last?.epoch,20)
        XCTAssertEqual(provenance.observedInputCount,22)
        XCTAssertEqual(provenance.derivedSignals["trainingEndMean"]!,0.424,accuracy:0.000001)
        XCTAssertTrue(finding.suggestion.contains("heuristic"))
        XCTAssertTrue(report.limitations.contains {$0.contains("uncalibrated")})
    }
    func testRejectsContradictoryEvaluationAndInventedSignalsOrUnknownSchemaFields() throws {
        let original=try RunDiagnostics.report(metrics())
        var malformed=original
        malformed.evaluations[0].reason = .matched
        XCTAssertThrowsError(try malformed.validate())
        malformed=original
        malformed.findings[0].provenance?.derivedSignals["gradientNorm"]=999
        XCTAssertThrowsError(try malformed.validate())
        malformed=original
        malformed.findings[0].provenance?.usesRawError=true
        XCTAssertThrowsError(try malformed.validate())
        var object=try XCTUnwrap(JSONSerialization.jsonObject(with:JSONEncoder().encode(original)) as? [String:Any])
        object["unreviewedConclusion"]="healthy"
        XCTAssertThrowsError(try DiagnosticReport.decode(JSONSerialization.data(withJSONObject:object)))
        var registry=try XCTUnwrap(JSONSerialization.jsonObject(with:JSONEncoder().encode(DiagnosticRuleRegistry.builtIn)) as? [String:Any])
        registry["pluginCode"]="unknown"
        XCTAssertThrowsError(try DiagnosticRuleRegistry.decode(JSONSerialization.data(withJSONObject:registry)))
    }
    func testGrowthBoundaryDomainAndDerivedOverflowWithholding() throws {
        func growth(_ end:Double)->[EpochMetric] {(0..<10).map {EpochMetric(epoch:$0+1,trainLoss:$0<5 ? 0.25:end,validationLoss:0.3,accuracy:0.5)}}
        XCTAssertTrue(try RunDiagnostics.report(growth(0.75)).findings.contains {$0.id == "trend.loss-growth.v1"})
        XCTAssertFalse(try RunDiagnostics.report(growth(0.749999)).findings.contains {$0.id == "trend.loss-growth.v1"})
        var invalid=growth(0.75);invalid[8]=EpochMetric(epoch:9,trainLoss:1,validationLoss:1,accuracy:1.1)
        XCTAssertEqual(try RunDiagnostics.report(invalid).evaluations.first {$0.ruleID == "trend.loss-growth.v1"}?.reason,.invalidMetricDomain)
        let overflow=(0..<10).map {EpochMetric(epoch:$0+1,trainLoss:$0<5 ? 1e308:0,validationLoss:$0<5 ? 0:1e308,accuracy:0.5)}
        let report=try RunDiagnostics.report(overflow)
        XCTAssertTrue(report.findings.isEmpty)
        XCTAssertEqual(report.evaluations.first {$0.ruleID == "trend.generalization-gap.v1"}?.reason,.nonfiniteDerivedSignals)
        XCTAssertNoThrow(try DiagnosticReport.decode(JSONEncoder().encode(report)))
    }

}
