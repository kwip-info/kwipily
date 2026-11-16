import XCTest
import CryptoKit
import KwipMLCore
@testable import KwipMLApp

final class OptionalEpochAccuracyTests:XCTestCase {
    private func unknown(_ epoch:Int=1,train:Double=0.7,validation:Double=0.7)throws->EpochMetric {
        let data=try JSONSerialization.data(withJSONObject:["epoch":epoch,"trainBits":String(train.bitPattern,radix:16),"validationBits":String(validation.bitPattern,radix:16)])
        return try XCTUnwrap(JSONDecoder().decode(MeasuredMetric.self,from:data).metric)
    }
    private func digest(_ data:Data)->String {SHA256.hash(data:data).map {String(format:"%02x",$0)}.joined()}
    func testMissingAccuracyRoundTripsWithoutZeroBitsOrNonfiniteClaim()throws {
        let metric=try unknown(train:-0.03,validation:-0.02)
        let accuracy:Double?=metric.accuracy
        XCTAssertNil(accuracy)
        let encoded=try JSONEncoder().encode(MeasuredMetric(metric))
        let object=try XCTUnwrap(JSONSerialization.jsonObject(with:encoded) as? [String:Any])
        XCTAssertNil(object["accuracyBits"])
        XCTAssertEqual(try XCTUnwrap(JSONDecoder().decode(MeasuredMetric.self,from:encoded).metric).trainLoss,-0.03)
        let report=try RunDiagnostics.report([metric])
        XCTAssertFalse(report.findings.contains {$0.id == "measurement.nonfinite.v1"})
        XCTAssertNotEqual(report.inputSHA256,try RunDiagnostics.report([.init(epoch:1,trainLoss:-0.03,validationLoss:-0.02,accuracy:0)]).inputSHA256)
    }
    func testKnownAccuracyKeepsHistoricalBytesAndFingerprint()throws {
        let metric=EpochMetric(epoch:1,trainLoss:0.5,validationLoss:0.6,accuracy:0.7)
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys,.withoutEscapingSlashes]
        let expected="{\"accuracyBits\":\"3fe6666666666666\",\"epoch\":1,\"trainBits\":\"3fe0000000000000\",\"validationBits\":\"3fe3333333333333\"}"
        XCTAssertEqual(String(decoding:try encoder.encode(MeasuredMetric(metric)),as:UTF8.self),expected)
        let snapshot="{\"context\":{\"sampling\":\"unspecified\"},\"observations\":["+expected+"],\"schemaVersion\":1}"
        XCTAssertEqual(try RunDiagnostics.report([metric]).inputSHA256,digest(Data(snapshot.utf8)))
    }
    func testBinaryChecksWithholdUnknownAccuracyButKeepMeasuredGradientFacts()throws {
        let metrics=try (1...20).map {try unknown($0)}
        var input=MeasuredDiagnosticInput(runID:UUID().uuidString,requestSHA256:String(repeating:"a",count:64),metrics:metrics)
        input.validationRows=10
        input.gradients=[.init(epoch:1,batch:0,presentElements:10,nonfiniteElements:1,zeroElements:0,missingParameters:0,l2Norm:nil)]
        let report=try MeasuredDiagnostics.evaluate(input)
        XCTAssertTrue(report.findings.contains {$0.ruleID == "gradient.nonfinite.v1"})
        for id in ["optimization.plateau.v1","data.coarse-holdout.v1"] {XCTAssertEqual(report.checks.first {$0.ruleID == id}?.status,.withheld)}
        let legacy=try RunDiagnostics.report(metrics)
        XCTAssertTrue(legacy.findings.isEmpty)
        let trends=legacy.evaluations.filter {["trend.generalization-gap.v1","trend.loss-growth.v1"].contains($0.ruleID)}
        XCTAssertEqual(trends.count,2);XCTAssertTrue(trends.allSatisfy {$0.status == .withheld})
        let context=MeasuredGeneralizationCheck.VerifiedContext(runID:input.runID,requestSHA256:input.requestSHA256,partitionSHA256:String(repeating:"b",count:64),evaluationSHA256:String(repeating:"c",count:64),completeEpochCapture:true,unchangedPartitions:true,unchangedEvaluation:true,binaryCrossEntropy:true)
        XCTAssertEqual(try MeasuredGeneralizationCheck.evaluate(input,context:context).check.status,.withheld)
    }
    @MainActor func testInterruptedArchivePersistsMissingAccuracyAndColdReplays()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let store=WorkspaceStore(rootOverride:root,detectCapabilities:false);store.beginProject(.init(name:"Numeric objective"))
        store.metrics=[try unknown()];store.latestReceivedMetric=try unknown(2,train:0.6,validation:0.65)
        store.recordExperiment(status:"cancelled",directory:root.appendingPathComponent("unused"),error:nil)
        let record=try XCTUnwrap(store.experiments.first)
        XCTAssertNil(record.accuracy);XCTAssertNil(record.diagnosticError)
        guard case .verified(let saved)=store.loadExperimentDiagnostics(record) else {return XCTFail("Missing-accuracy archive must retain a verified report")}
        let cold=WorkspaceStore(rootOverride:root,detectCapabilities:false);cold.activeProjectID=store.activeProjectID;cold.refreshExperiments()
        let reopened=try XCTUnwrap(cold.experiments.first)
        XCTAssertNil(reopened.accuracy)
        guard case .verified(let report)=cold.loadExperimentDiagnostics(reopened) else {return XCTFail("Cold diagnostic replay refused omitted accuracy")}
        XCTAssertEqual(report,saved)
        XCTAssertEqual(saved.observedInputCount,2)
    }
}
