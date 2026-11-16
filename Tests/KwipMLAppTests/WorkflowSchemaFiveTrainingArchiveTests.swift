import XCTest
import Combine
import Darwin
import KwipMLCore
@testable import KwipMLApp

final class WorkflowSchemaFiveTrainingArchiveTests:XCTestCase {
    @MainActor func testLegacySourceUpgradesWithRecoveryAndCurrentTrainingCapture()async throws {
        guard let development=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {
            throw XCTSkip("Set KWIPML_TEST_DEVELOPMENT_ROOT for actual schema-five training/archive replay.")
        }
        let canonical=try XCTUnwrap(realpath(FileManager.default.temporaryDirectory.path,nil));defer {free(canonical)}
        let root=URL(fileURLWithPath:String(cString:canonical)).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        let workspace=WorkspaceStore(rootOverride:URL(fileURLWithPath:development),storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace)
        var document=try WorkflowCodec.migrateToVersion5(WorkflowExamples.preparedParallelClassifier(device:"cpu"))
        document.nodes.append(.init(address:.init(domain:.control,localID:"frozen-unbound"),payload:.frozenModel(config:[:])))
        document.nodes.append(.init(address:.init(domain:.control,localID:"evaluate-unconnected"),payload:.evaluateTest(config:[:])))
        let originalRecord=try workspace.projectLibrary.saveSource(.workflow(document),id:document.projectID)
        XCTAssertTrue(workspace.beginWorkflowProject(id:document.projectID),workspace.statusMessage)
        let host=try XCTUnwrap(workspace.workflowSession)
        let upgraded=host.owner.document
        let backup=URL(fileURLWithPath:originalRecord.path).appendingPathExtension("before-retired-tools-r\(document.revision)")
        XCTAssertEqual(try WorkflowCodec.decode(Data(contentsOf:backup)),document)
        defer {if workspace.isBusy {host.cancel()};if !workspace.isBusy {try? FileManager.default.removeItem(at:root)}}
        let csv=root.appendingPathComponent("observations.csv")
        try FileManager.default.copyItem(at:URL(fileURLWithPath:development).appendingPathComponent("Sources/KwipMLApp/Resources/workflow-observations.csv"),to:csv)
        XCTAssertTrue(host.bindings.setPath(csv.path,for:WorkflowExamples.sourceAddress,lease:host.owner.lease(),expectedRevision:host.bindings.revision))
        let done=expectation(description:"Schema-five actual training terminates")
        let observation=host.$outcomes.filter {!$0.isEmpty}.prefix(1).sink {_ in done.fulfill()};defer {observation.cancel()}
        guard host.run(train:WorkflowExamples.trainAddress) else {return XCTFail(host.statusMessage)}
        await fulfillment(of:[done],timeout:60)
        XCTAssertFalse(workspace.isBusy)
        let outcome=try XCTUnwrap(host.outcomes.first)
        guard outcome.status == .completed else {return XCTFail("Schema-five training did not complete: \(outcome.error ?? "No worker diagnostic")")}
        XCTAssertNil(outcome.recordError)
        let captureURL=outcome.directory.appendingPathComponent("capture.json")
        let captureBytes=try Data(contentsOf:captureURL)
        let capture=try WorkflowArchiveCodec.decodeCapture(captureBytes)
        XCTAssertEqual(capture.schemaVersion,18);XCTAssertEqual(capture.compilerContractVersion,16)
        let archived=try WorkflowCodec.decode(Data(contentsOf:outcome.directory.appendingPathComponent("workflow.json")))
        XCTAssertEqual(archived,upgraded)
        let reference=WorkflowHistoryReference(projectID:document.projectID,directoryID:try XCTUnwrap(UUID(uuidString:outcome.directory.lastPathComponent)))
        guard case .verified(let verified)=WorkflowHistoryReader.load(reference,root:root) else {return XCTFail("Schema-five source did not independently replay its training archive")}
        XCTAssertEqual(verified.outcome.status,.completed)
        XCTAssertTrue(try capture.plan.matches(verified.plan,compilerContractVersion:14))
        for (schema,compiler) in [(5,5),(5,3),(4,3),(3,4),(2,4),(1,4)] {
            var wrong=capture;wrong.schemaVersion=schema;wrong.compilerContractVersion=compiler
            XCTAssertThrowsError(try WorkflowArchiveCodec.encodeCapture(wrong),"Rejected pair \(schema)/\(compiler)")
        }
        // Downgrading current composition evidence cannot create a legacy receipt.
        var legacy=capture;legacy.schemaVersion=4;legacy.compilerContractVersion=4
        XCTAssertThrowsError(try WorkflowArchiveCodec.encodeCapture(legacy))
        if case .verified=WorkflowHistoryReader.load(reference,root:root) {} else {XCTFail("Untouched exact archive must remain verified")}
    }
}
