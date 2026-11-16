import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowCapturedStorageTests: XCTestCase {
    private func window() throws -> ExecutionBudgetWindow {
        try .init(policy: .resolve(global: .factory, projectMaximumRunSeconds: nil), startedAtUptime: ProcessInfo.processInfo.systemUptime)
    }
    private func arguments(_ directory: URL, training: Bool) -> [String] {
        let common = ["--graph", directory.appendingPathComponent("input/graph.json").path,
                      "--output", directory.appendingPathComponent("artifacts").path, "--device", "cpu", "--seed", "42"]
        let sampling = ["--sample-limit", "20", "--split-seed", "42", "--validation-percent", "20"]
        return common + (training ? ["--epochs", "1", "--batch-size", "8", "--learning-rate", "0.03"] : ["--preview", "--stop-at", "hidden"]) + sampling
    }
    func testInsufficientStorageRefusesPreviewAndTrainingBeforePrivateCapture() async throws {
        for training in [false, true] {
            let (root, _, _, runtime) = try WorkflowEvaluationRequestTests().fixture()
            defer { try? FileManager.default.removeItem(at: root) }
            var launched = false
            do {
                _ = try await WorkflowCapturedToolExecution.launch(mode: training ? .modelTraining : .preview,
                    workspace: root, projectID: UUID().uuidString, runtime: runtime, window: window(),
                    storageLimitBytes: 1, inputs: { _ in [.bytes(path: "input/graph.json", data: Data("{}".utf8))] },
                    arguments: { self.arguments($0, training: training) },
                    launcher: { _, _, _, _, _ in launched = true; return .init(cancel: {}) },
                    onLine: { _ in }, onExit: { _, _, _ in })
                XCTFail("Insufficient storage was admitted")
            } catch { XCTAssertTrue(error.localizedDescription.lowercased().contains("storage"), error.localizedDescription) }
            XCTAssertFalse(launched)
            for name in ["workflow-operations", "preview-staging", "model-training-staging"] {
                XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent(name).path))
            }
            XCTAssertEqual(try WorkflowStorageAdmission(workspace: root).snapshot().committedBytes, 0)
        }
    }
    func testCapturedReservationSurvivesCancellationUntilActualExitAndCleanup() async throws {
        for training in [false, true] {
            let (root, _, _, runtime) = try WorkflowEvaluationRequestTests().fixture()
            defer { try? FileManager.default.removeItem(at: root) }
            let ledger = try WorkflowStorageAdmission(workspace: root)
            var exit: ((Int32, String) -> Void)?, directory: URL?
            let finished = expectation(description: "cancelled captured operation")
            let handle = try await WorkflowCapturedToolExecution.launch(mode: training ? .modelTraining : .preview,
                workspace: root, projectID: UUID().uuidString, runtime: runtime, window: window(),
                storageLimitBytes: 500_000_000, inputs: { _ in [.bytes(path: "input/graph.json", data: Data("{}".utf8))] },
                arguments: { self.arguments($0, training: training) },
                launcher: { _, args, _, _, terminal in
                    let index = try XCTUnwrap(args.firstIndex(of: "--kwip-operation-supervise"))
                    directory = URL(fileURLWithPath: args[index + 1]); exit = terminal
                    return .init(cancel: {})
                }, onLine: { _ in }, onExit: { status, _, output in
                    XCTAssertEqual(status, 130); XCTAssertNil(output); finished.fulfill()
                })
            let committed = try ledger.snapshot().committedBytes
            XCTAssertGreaterThan(try ledger.snapshot().reservedGrowthBytes, 0)
            handle.cancel()
            XCTAssertEqual(try ledger.snapshot().committedBytes, committed)
            let input = try XCTUnwrap(directory).appendingPathComponent("input")
            XCTAssertTrue(FileManager.default.fileExists(atPath: input.path))
            exit?(130, "")
            await fulfillment(of: [finished], timeout: 5)
            XCTAssertFalse(FileManager.default.fileExists(atPath: input.path))
            XCTAssertEqual(try ledger.snapshot().reservedGrowthBytes, 0)
            XCTAssertGreaterThan(try ledger.snapshot().retainedBytes, 0)
        }
    }
}
