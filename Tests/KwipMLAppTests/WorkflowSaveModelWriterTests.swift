import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp

final class WorkflowSaveModelWriterTests:XCTestCase {
    /// Uses the real frozen History verifier and real staging; no custom writer seam.
    @MainActor func testActualCapturedModelSavesBothFormatsAndRejectsChangedPins()async throws {
        guard let development=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {throw XCTSkip("Enable actual workflow runtime")}
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).resolvingSymlinksInPath()
        let external=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).resolvingSymlinksInPath()
        try FileManager.default.createDirectory(at:external,withIntermediateDirectories:false,attributes:[.posixPermissions:0o700])
        defer {try? FileManager.default.removeItem(at:root);try? FileManager.default.removeItem(at:external)}
        let workspace=WorkspaceStore(rootOverride:URL(fileURLWithPath:development),storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace)
        XCTAssertTrue(workspace.loadPreparedWorkflowExample(device:"cpu"))
        let host=try XCTUnwrap(workspace.workflowSession),done=expectation(description:"Training complete")
        let observation=host.$outcomes.dropFirst().filter {!$0.isEmpty}.prefix(1).sink {_ in done.fulfill()}
        XCTAssertTrue(host.run(train:WorkflowExamples.trainAddress));await fulfillment(of:[done],timeout:60);observation.cancel()
        let outcome=try XCTUnwrap(host.outcomes.first);XCTAssertEqual(outcome.status,.completed)
        let history=WorkflowHistoryReference(projectID:host.owner.projectID,directoryID:try XCTUnwrap(UUID(uuidString:outcome.directory.lastPathComponent)))
        let reference=try WorkflowFrozenModelSnapshot.select(history,root:root)
        let originalOutcome=try Data(contentsOf:outcome.directory.appendingPathComponent("outcome.json"))
        let originalCheckpoint=try Data(contentsOf:outcome.directory.appendingPathComponent("training/artifacts/checkpoint.pt"))
        let writer=WorkflowSaveModelWriter()
        let checkpoint=try await writer.stage(reference:reference,format:.checkpoint,root:root,
            destination:external.appendingPathComponent("model.pt"),accountingRoot:nil,storageLimitBytes:Int64.max)
        defer {checkpoint.close()}
        try checkpoint.commit()
        XCTAssertEqual(try Data(contentsOf:external.appendingPathComponent("model.pt")),originalCheckpoint)
        XCTAssertEqual(checkpoint.receipt.files.map(\.path),["model.pt"])
        let saved=expectation(description:"Package session finished")
        let packageSession=WorkflowSaveModelSession(reference:reference,format:.package,root:root,storageLimitBytes:Int64.max,
            reservation:.init(acquire:{UUID()},release:{_ in saved.fulfill()}),io:writer,isCurrent:{true})
        XCTAssertTrue(packageSession.save(to:external.appendingPathComponent("model"),access:external))
        await fulfillment(of:[saved],timeout:10)
        XCTAssertNotNil(packageSession.receipt,packageSession.statusMessage)
        XCTAssertEqual(try Data(contentsOf:external.appendingPathComponent("model/checkpoint.pt")),originalCheckpoint)
        XCTAssertTrue(FileManager.default.fileExists(atPath:external.appendingPathComponent("model/model.py").path))
        let contract=try String(contentsOf:external.appendingPathComponent("model/input-contract.json"),encoding:.utf8)
        XCTAssertTrue(contract.contains("alreadyPrepared"));XCTAssertFalse(contract.contains(root.path))
        XCTAssertEqual(try Data(contentsOf:outcome.directory.appendingPathComponent("outcome.json")),originalOutcome,"Saving must not rewrite History")
        let wrong=WorkflowFrozenModelReference(projectID:reference.projectID,directoryID:reference.directoryID,
            runID:reference.runID,requestIdentity:reference.requestIdentity,outcomeSHA256:reference.outcomeSHA256,
            modelManifestSHA256:reference.modelManifestSHA256,preparationManifestSHA256:String(repeating:"0",count:64),checkpointSHA256:reference.checkpointSHA256)
        do {
            _=try await writer.stage(reference:wrong,format:.checkpoint,root:root,destination:external.appendingPathComponent("wrong.pt"),accountingRoot:nil,storageLimitBytes:Int64.max)
            XCTFail("An incorrect preparation parent must not save")
        } catch {XCTAssertFalse(FileManager.default.fileExists(atPath:external.appendingPathComponent("wrong.pt").path))}
        let checkpointURL=outcome.directory.appendingPathComponent("training/artifacts/checkpoint.pt")
        try Data("tampered".utf8).write(to:checkpointURL)
        do {
            _=try await writer.stage(reference:reference,format:.checkpoint,root:root,destination:external.appendingPathComponent("tampered.pt"),accountingRoot:nil,storageLimitBytes:Int64.max)
            XCTFail("Changed captured checkpoint bytes must not save")
        } catch {XCTAssertFalse(FileManager.default.fileExists(atPath:external.appendingPathComponent("tampered.pt").path))}
        XCTAssertEqual(Set(try FileManager.default.contentsOfDirectory(atPath:external.path)),["model.pt","model"])
    }
}
