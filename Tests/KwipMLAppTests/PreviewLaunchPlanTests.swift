import XCTest
import KwipMLCore
@testable import KwipMLApp

final class PreviewLaunchPlanTests: XCTestCase {
    func testCSVPlanCapturesAuthorizedBytesAndKeepsBasename() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let csv = root.appendingPathComponent("my observations.csv")
        let bytes = Data("a,b,label\n1,2,0\n".utf8); try bytes.write(to: csv)
        let settings = ExecutionSettings(device: "cpu", datasetCSV: csv.path, featureColumns: ["a", "b"], targetColumn: "label")
        let plan = try PreviewLaunchPlan.capture(graph: ExampleFlows.parallelClassifier, settings: settings,
            selected: "output", resume: false, cache: nil, checkpoint: nil, hashes: [:], sourcePermissions: root.appendingPathComponent("grants"))
        let captured = try WorkflowPreviewCapture.stage(workspace: root, inputs: plan.inputs)
        defer { captured.remove() }
        XCTAssertTrue(captured.files.contains { $0.path == "input/data/my observations.csv" && $0.sha256 == WorkflowArchiveCodec.hash(bytes) })
        let args = plan.arguments(in: root)
        XCTAssertTrue(args.contains(root.appendingPathComponent("input/data/my observations.csv").path))
        XCTAssertFalse(args.contains(csv.path)); XCTAssertFalse(args.contains("--registry"))
    }
    func testRetainedResumeRequiresBothWeightAndCachePins() throws {
        let settings = ExecutionSettings(device: "cpu")
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        XCTAssertThrowsError(try PreviewLaunchPlan.capture(graph: ExampleFlows.parallelClassifier, settings: settings,
            selected: "output", resume: true, cache: root.appendingPathComponent("cache.pt").path,
            checkpoint: root.appendingPathComponent("checkpoint.pt").path, hashes: [:], sourcePermissions: root))
    }
}

final class SupervisedPreviewActionsTests: XCTestCase {
    @MainActor func testNativePreviewUsesSupervisorAndCleansInputsBeforePublishing() async throws {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let storage = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: storage) }
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: storage, detectCapabilities: false)
        await verifyRuntimeForIntegration(workspace)
        workspace.loadExample(); workspace.setExecution { $0.device = "cpu" }
        var arguments: [String] = []
        workspace.previewWorkerLauncher = { executable, args, directory, line, exit in
            arguments = args
            return try WorkflowExecutionSession.launchWorker(executable, args, directory, line, exit)
        }
        workspace.selectNode("output"); workspace.previewSelected()
        XCTAssertTrue(workspace.isPreviewing)
        let deadline = Date().addingTimeInterval(25)
        while workspace.isPreviewing && Date() < deadline { try await Task.sleep(nanoseconds: 20_000_000) }
        XCTAssertFalse(workspace.isPreviewing, workspace.statusMessage)
        let index = try XCTUnwrap(arguments.firstIndex(of: "--kwip-operation-supervise"), workspace.statusMessage)
        let operation = URL(fileURLWithPath: arguments[index + 1])
        XCTAssertFalse(FileManager.default.fileExists(atPath: operation.appendingPathComponent("input").path))
        XCTAssertEqual(workspace.previewCachePath, operation.appendingPathComponent("artifacts/activation-cache.pt").path)
        XCTAssertTrue(workspace.cacheValid, workspace.statusMessage)
        let previous = workspace.previewCachePath
        workspace.previewSelected(); workspace.cancelRun()
        XCTAssertTrue(workspace.isPreviewing, "Cancellation must join pending capture before releasing busy state")
        let cancelDeadline = Date().addingTimeInterval(10)
        while workspace.isPreviewing && Date() < cancelDeadline { try await Task.sleep(nanoseconds: 20_000_000) }
        XCTAssertFalse(workspace.isPreviewing)
        XCTAssertEqual(workspace.previewCachePath, previous)
        XCTAssertTrue(workspace.statusMessage.lowercased().contains("cancel"))
    }
}
