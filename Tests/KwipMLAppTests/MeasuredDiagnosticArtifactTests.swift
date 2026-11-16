import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp

final class MeasuredDiagnosticArtifactTests:XCTestCase {
    func testRealAutogradControlsPassStrictArtifactDecoderAndExpectedRules()throws {
        guard let path=ProcessInfo.processInfo.environment["KWIPML_DIAGNOSTIC_CONTROLS"] else {throw XCTSkip("Generate the actual PyTorch backward controls first.")}
        struct Fixture:Decodable {let id:String;let file:String;let sha256:String;let runID:String;let expectedRules:[String];let activations:[MeasuredActivationSample]}
        struct Manifest:Decodable {let schemaVersion:Int;let corpusID:String;let cases:[Fixture]}
        let directory=URL(fileURLWithPath:path),manifest=try JSONDecoder().decode(Manifest.self,from:Data(contentsOf:directory.appendingPathComponent("manifest.json")))
        XCTAssertEqual(manifest.schemaVersion,1);XCTAssertEqual(manifest.corpusID,"kwip.pytorch-backward-controls.v1");XCTAssertEqual(manifest.cases.count,8)
        for fixture in manifest.cases {
            let bytes=try Data(contentsOf:directory.appendingPathComponent(fixture.file))
            XCTAssertEqual(WorkflowArchiveCodec.hash(bytes),fixture.sha256)
            let artifact=try GradientCaptureCodec.decodeArtifact(bytes,expectedRunID:fixture.runID,expectedDevice:"cpu")
            var input=MeasuredDiagnosticInput(runID:fixture.runID,requestSHA256:fixture.sha256,metrics:[])
            try input.installGradients(artifact,expectedWorkerRunID:fixture.runID)
            input.activations=fixture.activations
            XCTAssertEqual(Set(try MeasuredDiagnostics.evaluate(input).findings.map(\.ruleID)),Set(fixture.expectedRules),fixture.id)
            XCTAssertThrowsError(try input.installGradients(artifact,expectedWorkerRunID:UUID().uuidString),"A foreign worker's observations cannot be borrowed")
        }
    }
    @MainActor func testActualVerifiedWorkflowFeedsMeasuredDiagnosticsWithoutInventedCoverage()async throws {
        guard let development=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {throw XCTSkip("Select the actual development worker for artifact integration.")}
        let root=try canonicalBackendTemporaryDirectory();defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:URL(fileURLWithPath:development),storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace)
        XCTAssertTrue(workspace.loadPreparedWorkflowExample(device:"cpu"),workspace.statusMessage)
        let host=try XCTUnwrap(workspace.workflowSession)
        let setup=try XCTUnwrap(host.owner.document.nodes.first{if case .train=$0.payload{return true};return false})
        guard case .train(var config)=setup.payload else {return XCTFail("Missing Train")}
        var runSetup=try XCTUnwrap(WorkflowTrainRunSetup.configuration(config,schemaVersion:host.owner.document.schemaVersion))
        runSetup["gradientCapture"]=try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(GradientCaptureConfiguration(everyEpochs:1,maxBatchesPerEpoch:1,maxCaptures:2)))
        config=try WorkflowTrainRunSetup.setting(runSetup,in:config,schemaVersion:host.owner.document.schemaVersion)
        XCTAssertNotNil(host.owner.apply(.configure(address:setup.address,config:config),lease:host.owner.lease()))
        let done=expectation(description:"Actual gradient training completes")
        let observation=host.$outcomes.dropFirst().filter{!$0.isEmpty}.prefix(1).sink{_ in done.fulfill()};defer {observation.cancel()}
        XCTAssertTrue(host.run(train:WorkflowExamples.trainAddress),host.statusMessage)
        await fulfillment(of:[done],timeout:60)
        if workspace.isBusy {host.cancel()}
        let outcome=try XCTUnwrap(host.outcomes.first)
        XCTAssertEqual(outcome.status,.completed,outcome.error ?? host.statusMessage)
        let reference=WorkflowHistoryReference(projectID:host.owner.projectID,directoryID:try XCTUnwrap(UUID(uuidString:outcome.directory.lastPathComponent)))
        guard case .verified(let verified)=WorkflowHistoryReader.load(reference,root:root) else {return XCTFail("Actual archive verification failed")}
        let input=try MeasuredDiagnosticInput.from(verified)
        XCTAssertEqual(input.workflowSHA256,verified.capture.workflow.sha256)
        XCTAssertEqual(input.metrics.count,3)
        XCTAssertEqual(input.gradients.count,2,"Capture budget exhaustion must not invent a third batch")
        XCTAssertEqual(input.validationRows,verified.modelContent?.partitionEvidence?.validationCount)
        let report=try MeasuredDiagnostics.evaluate(input)
        XCTAssertEqual(report.checks.first{$0.ruleID=="gradient.growth.v1"}?.status,.withheld)
        XCTAssertEqual(report.checks.first{$0.ruleID=="optimization.plateau.v1"}?.status,.withheld)
        XCTAssertTrue(report.findings.contains{$0.ruleID=="data.coarse-holdout.v1"})
        XCTAssertFalse(report.findings.contains{$0.ruleID=="gradient.nonfinite.v1"})
        let encoded=try JSONEncoder().encode(report)
        XCTAssertEqual(try MeasuredDiagnosticReport.decode(encoded),report)
    }
    @MainActor func testSuppressionStorageSurvivesRestartAndRejectsCorruptAnnotations()throws {
        let root=try canonicalBackendTemporaryDirectory();defer {try? FileManager.default.removeItem(at:root)}
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        let project=UUID().uuidString,now=Date()
        var input=MeasuredDiagnosticInput(runID:UUID().uuidString,requestSHA256:String(repeating:"d",count:64),metrics:[]);input.validationRows=10
        let report=try MeasuredDiagnostics.evaluate(input),finding=try XCTUnwrap(report.findings.first)
        let item=try MeasuredDiagnosticSuppression(report:report,finding:finding,reason:"Additional held-out data scheduled",now:now,expiresAt:now.addingTimeInterval(3600))
        let store=MeasuredDiagnosticSuppressionStore(root:root,projectID:project)
        try store.save(item,now:now)
        let reopened=MeasuredDiagnosticSuppressionStore(root:root,projectID:project)
        XCTAssertNil(reopened.error);XCTAssertEqual(reopened.entries,[item])
        try Data("broken".utf8).write(to:root.appendingPathComponent("diagnostic-workbench/\(project)/suppressions.json"))
        reopened.reload();XCTAssertNotNil(reopened.error)
        XCTAssertThrowsError(try reopened.save(item,now:now),"A failed read must not silently overwrite annotations")
    }
}
