import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowCheckpointStorageTests: XCTestCase {
    func testExactAdmissionManagedStageAndLifetimePreserveParentArchive() async throws {
        guard let development = ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else { throw XCTSkip("Requires actual runtime") }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: URL(fileURLWithPath: development), storageOverride: root, detectCapabilities: false)
        await verifyRuntimeForIntegration(workspace)
        XCTAssertTrue(workspace.loadPreparedWorkflowExample(device: "cpu"))
        let host = try XCTUnwrap(workspace.workflowSession)
        let done = expectation(description: "Verified parent")
        let observation = host.$outcomes.dropFirst().filter { !$0.isEmpty }.prefix(1).sink { _ in done.fulfill() }
        XCTAssertTrue(host.run(train: WorkflowExamples.trainAddress))
        await fulfillment(of: [done], timeout: 60); observation.cancel()
        let outcome = try XCTUnwrap(host.outcomes.first)
        XCTAssertEqual(outcome.status, .completed, outcome.error ?? "")
        let reference = WorkflowHistoryReference(projectID: host.owner.projectID,
            directoryID: try XCTUnwrap(UUID(uuidString: outcome.directory.lastPathComponent)))
        let selected = WorkflowCheckpointReference(projectID: reference.projectID, directoryID: reference.directoryID.uuidString,
            runID: outcome.runID.uuidString, requestIdentity: outcome.request.requestIdentity,
            modelManifestSHA256: try XCTUnwrap(outcome.modelManifestSHA256))
        let original = try Dictionary(uniqueKeysWithValues: WorkflowCheckpointSnapshot.artifactLimits.keys.map { name in
            (name, try Data(contentsOf: outcome.directory.appendingPathComponent("training/artifacts/" + name)))
        })
        let exact = Int64(original.values.reduce(0) { $0 + $1.count })
        let ledger = try WorkflowStorageAdmission(workspace: root)
        let before = try ledger.snapshot().committedBytes
        XCTAssertThrowsError(try WorkflowHistoryReader.stageCheckpoint(selected, root: root, storageLimitBytes: before + exact - 1))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("checkpoint-staging").path))
        var snapshot: WorkflowCheckpointSnapshot? = try WorkflowHistoryReader.stageCheckpoint(selected, root: root, storageLimitBytes: before + exact)
        let directory = try XCTUnwrap(snapshot?.directory)
        XCTAssertEqual(directory.deletingLastPathComponent().lastPathComponent, "checkpoint-staging")
        XCTAssertNotNil(UUID(uuidString: directory.lastPathComponent))
        XCTAssertEqual(try ledger.snapshot().committedBytes, before + exact)
        for (name, bytes) in original { XCTAssertEqual(try Data(contentsOf: directory.appendingPathComponent(name)), bytes) }
        XCTAssertNoThrow(try snapshot?.parentRecord())
        snapshot?.remove(); snapshot?.remove()
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.path))
        XCTAssertEqual(try ledger.snapshot().committedBytes, before)
        snapshot = try WorkflowHistoryReader.stageCheckpoint(reference, root: root, storageLimitBytes: before + exact)
        let abandoned = try XCTUnwrap(snapshot?.directory)
        snapshot = nil
        XCTAssertFalse(FileManager.default.fileExists(atPath: abandoned.path))
        XCTAssertEqual(try ledger.snapshot().committedBytes, before)
        for (name, bytes) in original {
            XCTAssertEqual(try Data(contentsOf: outcome.directory.appendingPathComponent("training/artifacts/" + name)), bytes)
        }
    }
}
