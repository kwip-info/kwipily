import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class CheckpointLifecycleAdmissionTests: XCTestCase {
    private var repository: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    }
    private final class Captures: @unchecked Sendable {
        private let lock = NSLock()
        private var value = 0
        func record() { lock.lock(); value += 1; lock.unlock() }
        var count: Int { lock.lock(); defer { lock.unlock() }; return value }
    }
    private func fixture(_ mode: WorkflowTrainingMode) throws -> (WorkspaceStore, WorkflowWorkspaceSession, Captures) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
        var document = try WorkflowExamples.preparedParallelClassifier(device: "cpu")
        document.schemaVersion = max(3, document.schemaVersion)
        let train = try XCTUnwrap(document.nodes.firstIndex { $0.address == WorkflowExamples.trainAddress })
        document.nodes[train].payload = .train(config: ["trainingMode": .string(mode.rawValue)])
        _ = try workspace.projectLibrary.saveSource(.workflow(document), id: document.projectID)
        XCTAssertTrue(workspace.beginWorkflowProject(id: document.projectID), workspace.statusMessage)
        let host = try XCTUnwrap(workspace.workflowSession)
        try installFixtureExecutionContract(workspace, version: "2.14.1")
        let csv = root.appendingPathComponent("observations.csv")
        try FileManager.default.copyItem(at: repository.appendingPathComponent("Sources/KwipMLApp/Resources/workflow-observations.csv"), to: csv)
        try workspace.sourceAccess.remember(csv)
        XCTAssertTrue(host.bindings.setPath(csv.path, for: WorkflowExamples.sourceAddress,
            lease: host.owner.lease(), expectedRevision: host.bindings.revision))
        let parent = WorkflowCheckpointReference(projectID: document.projectID, directoryID: UUID().uuidString,
            runID: UUID().uuidString, requestIdentity: String(repeating: "a", count: 64),
            modelManifestSHA256: String(repeating: "b", count: 64))
        XCTAssertTrue(host.checkpointBindings.setReference(parent, for: WorkflowExamples.trainAddress,
            lease: host.owner.lease(), expectedRevision: host.checkpointBindings.revision))
        let captures = Captures()
        host.captureRunSources = { _, _, _, _ in captures.record(); throw CancellationError() }
        host.captureCheckpoint = { _, _, _ in captures.record(); throw CancellationError() }
        host.executionLaunch = { _, _, _, _, _ in XCTFail("Admission must refuse before worker launch"); throw CancellationError() }
        return (workspace, host, captures)
    }
    func testMissingLifecycleDiscoveryRefusesBothRestoreModesBeforeCaptureOrReservation() async throws {
        for mode in [WorkflowTrainingMode.exactResume, .fineTune] {
            let (workspace, host, captures) = try fixture(mode)
            defer { host.cancel(); workspace.retireWorkflowSession() }
            XCTAssertFalse(host.run(), host.statusMessage)
            try await Task.sleep(for: .milliseconds(20))
            XCTAssertTrue(host.statusMessage.localizedCaseInsensitiveContains("checkpoint") &&
                host.statusMessage.localizedCaseInsensitiveContains("runtime"), host.statusMessage)
            XCTAssertEqual(captures.count, 0, "Unavailable restore must refuse before any source or checkpoint staging")
            XCTAssertFalse(workspace.isBusy)
            XCTAssertFalse(host.isStagingCheckpoint)
            XCTAssertNil(host.queuedRequestID)
        }
    }
    func testMissingLifecycleDiscoveryDoesNotBlockFreshTrainingCapture() async throws {
        let (workspace, host, captures) = try fixture(.fresh)
        defer { host.cancel(); workspace.retireWorkflowSession() }
        XCTAssertTrue(host.run(), host.statusMessage)
        let deadline = Date().addingTimeInterval(3)
        while captures.count == 0 && Date() < deadline { try await Task.sleep(for: .milliseconds(1)) }
        XCTAssertEqual(captures.count, 1)
    }
    private func install(_ workspace: WorkspaceStore, lifecycle: Data?) throws {
        let base = try BuiltInCapabilityBridge.executionContract(registry: workspace.registry,
            runtimeVersion: "2.14.1", availableDevices: [.cpu])
        try fixtureInstallDiscoveredRuntimeContract(workspace,version: "2.14.1", expectedVersion: nil,
            devices: ["cpu"], descriptor: JSONEncoder().encode(base),
            tensorCompositionDescriptors: fixtureReviewedCompositionDescriptors(), lifecycleDescriptor: lifecycle, moduleCompositionDescriptor: fixtureReviewedModuleCompositionDescriptor())
    }
    func testExactDiscoveryIsSeparateFromComponentsAndRefreshRevokesIt() throws {
        let (workspace, host, _) = try fixture(.fineTune)
        defer { host.cancel(); workspace.retireWorkflowSession() }
        let count = workspace.verifiedRuntimeComponentCount
        let valid = try JSONEncoder().encode(BundledLifecycleCapabilities.entry.runtimeContract)
        try install(workspace, lifecycle: valid)
        XCTAssertTrue(workspace.lifecycleRuntimeCapability.isAvailable)
        XCTAssertNil(host.checkpointRestoreUnavailableReason)
        XCTAssertEqual(workspace.verifiedRuntimeComponentCount, count)
        var wrong = try XCTUnwrap(JSONSerialization.jsonObject(with: valid) as? [String: Any])
        wrong["externalImport"] = true
        try install(workspace, lifecycle: JSONSerialization.data(withJSONObject: wrong))
        XCTAssertFalse(workspace.lifecycleRuntimeCapability.isAvailable)
        XCTAssertNotNil(host.checkpointRestoreUnavailableReason)
        try install(workspace, lifecycle: valid)
        try install(workspace, lifecycle: nil)
        XCTAssertFalse(workspace.lifecycleRuntimeCapability.isAvailable)
        try install(workspace, lifecycle: valid)
        XCTAssertThrowsError(try fixtureInstallDiscoveredRuntimeContract(workspace,version: "2.14.1",
            expectedVersion: nil, devices: ["cpu"], descriptor: Data("{}".utf8), lifecycleDescriptor: valid))
        XCTAssertFalse(workspace.lifecycleRuntimeCapability.isAvailable)
        XCTAssertNil(workspace.runtimeExecutionContract)
        try install(workspace, lifecycle: valid)
        XCTAssertThrowsError(try workspace.installRuntimeContract(version: "bad", expectedVersion: nil))
        XCTAssertFalse(workspace.lifecycleRuntimeCapability.isAvailable)
    }
    func testCapabilityProbeReadsOnlyLifecycleSiblingAndFailedProbeRevokes() throws {
        let (workspace, host, _) = try fixture(.fineTune)
        defer { host.cancel(); workspace.retireWorkflowSession() }
        let base = try BuiltInCapabilityBridge.executionContract(registry: workspace.registry,
            runtimeVersion: "2.14.1", availableDevices: [.cpu])
        let lifecycle = try JSONSerialization.jsonObject(with: JSONEncoder().encode(BundledLifecycleCapabilities.entry.runtimeContract))
        func probe(sibling: Bool, failed: Bool = false) throws {
            var event: [String: Any] = ["event": "capabilities", "torchVersion": "2.14.1", "devices": ["cpu"],
                "defaultDevice": "cpu", "executionContract": try JSONSerialization.jsonObject(with: JSONEncoder().encode(base))]
            event[sibling ? "lifecycleCapabilities" : "componentCapabilities"] = ["checkpoint_lifecycle": lifecycle]
            let line = String(decoding: try JSONSerialization.data(withJSONObject: event), as: UTF8.self)
            workspace.capabilityLaunch = { _, _, _, _, onLine, onExit in
                onLine(line); onExit(failed ? 1 : 0, failed ? "Fixture discovery refusal" : "")
                return .init(cancel: {})
            }
            var completed: Bool?
            workspace.refreshCapabilities { completed = $0 }
            XCTAssertEqual(completed, !failed)
        }
        try probe(sibling: false)
        XCTAssertFalse(workspace.lifecycleRuntimeCapability.isAvailable)
        try probe(sibling: true)
        XCTAssertTrue(workspace.lifecycleRuntimeCapability.isAvailable)
        try probe(sibling: true, failed: true)
        XCTAssertFalse(workspace.lifecycleRuntimeCapability.isAvailable)
        XCTAssertNil(workspace.runtimeExecutionContract)
    }

}
