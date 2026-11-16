import XCTest
import KwipMLCore
@testable import KwipMLApp

final class SupervisedPreparationActionsTests: XCTestCase {
    @MainActor private func finish(_ store: DataFlowWorkspaceStore) async throws {
        let deadline = Date().addingTimeInterval(30)
        while store.isBusy && Date() < deadline { try await Task.sleep(nanoseconds: 20_000_000) }
        XCTAssertFalse(store.isBusy, store.statusMessage)
    }
    @MainActor func testClosingBeforeCaptureStartsReleasesReservation() async throws {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let storage = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: storage) }
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: storage, detectCapabilities: false)
        workspace.loadExample()
        var store: DataFlowWorkspaceStore? = DataFlowWorkspaceStore(workspace: workspace)
        store?.loadBundledSensorExample()
        try await finish(try XCTUnwrap(store))
        store?.runPreparation(); store?.cancelPreparation(); store = nil
        let deadline = Date().addingTimeInterval(3)
        while workspace.isPreparingData && Date() < deadline { try await Task.sleep(nanoseconds: 20_000_000) }
        XCTAssertFalse(workspace.isPreparingData)
        XCTAssertNil(workspace.cancelDataPreparation)
    }
    @MainActor func testClosingPreparationRetainsReservationUntilWorkerExit() async throws {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let storage = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: storage) }
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: storage, detectCapabilities: false)
        await verifyRuntimeForIntegration(workspace)
        workspace.loadExample()
        var store: DataFlowWorkspaceStore? = DataFlowWorkspaceStore(workspace: workspace)
        store?.loadBundledSensorExample()
        try await finish(try XCTUnwrap(store))
        var pendingExit: (() -> Void)?
        store?.preparationWorkerLauncher = { executable, args, directory, line, exit in
            try WorkflowExecutionSession.launchWorker(executable, args, directory, line, { code, stderr in
                pendingExit = { exit(code, stderr) }
            })
        }
        store?.runPreparation()
        let deadline = Date().addingTimeInterval(20)
        while pendingExit == nil && Date() < deadline { try await Task.sleep(nanoseconds: 20_000_000) }
        XCTAssertNotNil(pendingExit)
        store?.cancelPreparation()
        store = nil
        XCTAssertTrue(workspace.isPreparingData)
        pendingExit?(); pendingExit = nil
        let done = Date().addingTimeInterval(5)
        while workspace.isPreparingData && Date() < done { try await Task.sleep(nanoseconds: 20_000_000) }
        XCTAssertFalse(workspace.isPreparingData, "Closing the editor must not strand the workspace reservation")
        XCTAssertNil(workspace.cancelDataPreparation)
    }
    @MainActor func testNativePreparationSupervisesCacheReuseAndPreservesResultsOnCancel() async throws {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let storage = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: storage) }
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: storage, detectCapabilities: false)
        await verifyRuntimeForIntegration(workspace)
        workspace.activeProjectID = UUID().uuidString
        let store = DataFlowWorkspaceStore(workspace: workspace)
        store.loadExample(columns: ["x", "y"])
        try FileManager.default.createDirectory(at: storage, withIntermediateDirectories: true)
        let source = storage.appendingPathComponent("observations.csv")
        try (["x,y"] + (0..<40).map { "\($0),\($0 * 2)" }).joined(separator: "\n").write(to: source, atomically: true, encoding: .utf8)
        store.bindSource(source.path, nodeID: "source")
        var arguments: [String] = []
        store.preparationWorkerLauncher = { executable, args, directory, line, exit in
            arguments = args
            return try WorkflowExecutionSession.launchWorker(executable, args, directory, line, exit)
        }
        store.runPreparation()
        XCTAssertTrue(store.isPreparing)
        try await finish(store)
        let index = try XCTUnwrap(arguments.firstIndex(of: "--kwip-operation-supervise"), store.statusMessage)
        let operation = URL(fileURLWithPath: arguments[index + 1])
        XCTAssertEqual(store.manifestURL?.path, operation.appendingPathComponent("artifacts/manifest.json").path)
        XCTAssertFalse(FileManager.default.fileExists(atPath: operation.appendingPathComponent("input").path))
        XCTAssertEqual(store.completedResult?.schemaVersion, 2)
        store.setCachePolicy(.reuse); store.runPreparation(); try await finish(store)
        XCTAssertFalse(try XCTUnwrap(store.cacheReport).reusedNodeIDs.isEmpty, store.statusMessage)
        let manifest = try XCTUnwrap(store.manifestURL)
        let reference = try Data(contentsOf: store.completedResultURL)
        store.runPreparation(); store.cancelPreparation()
        XCTAssertTrue(store.isPreparing, "Busy state remains held until capture/worker cancellation is joined")
        try await finish(store)
        XCTAssertEqual(store.manifestURL, manifest)
        XCTAssertEqual(try Data(contentsOf: store.completedResultURL), reference)
        try FileManager.default.removeItem(at: source)
        let reopened = DataFlowWorkspaceStore(workspace: workspace); try await finish(reopened)
        XCTAssertEqual(reopened.manifestURL?.standardizedFileURL.path, manifest.standardizedFileURL.path)
        XCTAssertTrue(reopened.restoredResult, reopened.statusMessage)
    }
}
