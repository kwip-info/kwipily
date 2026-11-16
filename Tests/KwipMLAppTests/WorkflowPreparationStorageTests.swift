import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowPreparationStorageTests: XCTestCase {
    func testOneReservationCoversSourceCaptureOwnedInputsWorkerAndFinalVerification() async throws {
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: repo, storageOverride: root, detectCapabilities: false)
        let runtime = try XCTUnwrap(workspace.runtime), projectID = UUID().uuidString
        let graph = try DataFlowCodec.decodeGraph(Data(contentsOf: repo.appendingPathComponent("examples/dataflow/scaler.json")))
        let source = root.appendingPathComponent("observations.csv")
        try Data(contentsOf: repo.appendingPathComponent("examples/dataflow/observations.csv")).write(to: source)
        let plan = try PreparationLaunchPlan.capture(graph: graph, bindings: ["source": source.path], labels: [:],
            cachePolicy: .materialize, cacheReference: nil, workspace: root, sourcePermissions: root.appendingPathComponent("grants"),
            projectID: projectID, storageLimitBytes: 305_000_000)
        defer { plan.removeSourceStage() }
        let admission = try XCTUnwrap(plan.storageReservation)
        let ledger = try WorkflowStorageAdmission(workspace: root)
        let admitted = try ledger.snapshot().committedBytes
        XCTAssertGreaterThan(try ledger.snapshot().reservedGrowthBytes, 0)
        let window = try ExecutionBudgetWindow(policy: .resolve(global: .factory, projectMaximumRunSeconds: nil), startedAtUptime: ProcessInfo.processInfo.systemUptime)
        let finished = expectation(description: "verified preparation"), workerExited = expectation(description: "held terminal")
        var deliverExit: (() -> Void)?, result: URL?
        let handle = try await WorkflowPreparationExecution.launch(workspace: root, projectID: projectID,
            runtime: runtime, window: window, storageAdmission: admission, inputs: plan.inputs, arguments: plan.arguments,
            launcher: { executable, args, directory, line, exit in
                try WorkflowExecutionSession.launchWorker(executable, args, directory, line, { code, message in
                    deliverExit = { exit(code, message) }; workerExited.fulfill()
                })
            }, onLine: { _ in }, onExit: { code, message, output in
                XCTAssertEqual(code, 0, message); result = output; finished.fulfill()
            })
        _ = handle
        plan.removeSourceStage()
        XCTAssertEqual(try ledger.snapshot().committedBytes, admitted, "Source-to-operation handoff must not acquire another full reservation")
        await fulfillment(of: [workerExited], timeout: 30)
        XCTAssertGreaterThan(try ledger.snapshot().reservedGrowthBytes, 0, "Held terminal still awaits verification and cleanup")
        deliverExit?(); deliverExit = nil
        await fulfillment(of: [finished], timeout: 10)
        let output = try XCTUnwrap(result)
        XCTAssertTrue(FileManager.default.fileExists(atPath: output.appendingPathComponent("manifest.json").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: output.deletingLastPathComponent().appendingPathComponent("input").path))
        XCTAssertEqual(try ledger.snapshot().reservedGrowthBytes, 0)
        XCTAssertGreaterThan(try ledger.snapshot().retainedBytes, 0)
    }
}
