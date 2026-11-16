import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class TensorFlowSessionTests: XCTestCase {
    func testRealTensorFlowCanvasPreparationFitHistoryAndOutput() async throws {
        guard let python = ProcessInfo.processInfo.environment["KWIPML_BACKEND_TEST_PYTHON"] else { throw XCTSkip("Select an isolated real optional-framework interpreter.") }
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root = try canonicalBackendTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
        await verifyRuntimeForIntegration(workspace)
        XCTAssertTrue(workspace.loadBackendWorkflowExample(backendID: "tensorflow"), workspace.statusMessage)
        let host = try XCTUnwrap(workspace.workflowSession)
        // Real worker execution below uses the isolated optional dependency environment;
        // package/runtime discovery has its own strict capability tests.
        workspace.tensorflowRuntimeCapability = try .decode(JSONEncoder().encode(TensorFlowRuntimeCapabilities.expected))
        host.executionLaunch = { _, arguments, workingDirectory, onLine, onExit in
            try WorkflowExecutionSession.launchWorker(URL(fileURLWithPath: python), arguments, workingDirectory, onLine, onExit)
        }
        XCTAssertTrue(host.run(train: WorkflowExamples.tensorFlowEstimatorAddress), host.statusMessage)
        let deadline = Date().addingTimeInterval(60)
        while host.backendFit?.isRunning == true && Date() < deadline { try await Task.sleep(nanoseconds: 100_000_000) }
        let session = try XCTUnwrap(host.backendFit)
        XCTAssertFalse(session.isRunning, session.status)
        XCTAssertEqual(session.phase, .completed, session.status + "\n" + session.messages.joined(separator: "\n"))
        XCTAssertNotNil(session.output)
        XCTAssertFalse(host.owner.isBusy)
        XCTAssertFalse(workspace.isBusy)
        // Resolve the connected provider: consolidation moved both partitions
        // onto Preprocess; there is no separate node with the old "training" ID.
        let training=try XCTUnwrap(host.owner.document.edges.first{$0.to == .init(address:WorkflowExamples.tensorFlowEstimatorAddress,port:"training")}?.from)
        XCTAssertNotNil(host.snapshot(for:training.address))
        XCTAssertTrue(host.owner.select(.init(nodeAddresses:[training.address]),lease:host.owner.lease()))
        let viewer=try XCTUnwrap(host.capturedTableViewSession(for:training))
        XCTAssertTrue(viewer.load())
        let queryDeadline=Date().addingTimeInterval(20)
        while viewer.isBusy && Date()<queryDeadline {try await Task.sleep(for:.milliseconds(20))}
        XCTAssertNotNil(viewer.page,viewer.statusMessage)
        XCTAssertGreaterThan(viewer.page?.totalRows ?? 0,8)
        XCTAssertEqual(host.backendHistory.entries.count, 1, host.backendHistory.status)
        if let entry = host.backendHistory.entries.first {
            let verified = try await host.backendHistory.verify(entry)
            XCTAssertTrue(FileManager.default.fileExists(atPath: verified.artifactDirectory.appendingPathComponent("portable/inference.py").path))
            let restored = BackendFitHistoryStore(root: root, projectID: host.owner.projectID)
            await restored.reloadAndWait()
            XCTAssertEqual(restored.entries.map(\.id), [entry.id], restored.status)
        }
    }
}
