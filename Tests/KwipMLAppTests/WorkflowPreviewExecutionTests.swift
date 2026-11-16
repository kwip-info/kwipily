import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowPreviewExecutionTests: XCTestCase {
    private func arguments(_ directory: URL, checkpoint: Bool) -> [String] {
        var result = ["--graph", directory.appendingPathComponent("input/graph.json").path,
            "--output", directory.appendingPathComponent("artifacts").path, "--device", "cpu", "--seed", "42",
            "--preview", "--stop-at", "hidden", "--sample-limit", "0", "--split-seed", "42", "--validation-percent", "20"]
        if checkpoint { result += ["--checkpoint", directory.appendingPathComponent("input/weights/checkpoint.pt").path] }
        return result
    }
    private func output(arguments: [String], line: (String) -> Void, corrupt: Bool = false) throws -> URL {
        let index = try XCTUnwrap(arguments.firstIndex(of: "--kwip-operation-supervise"))
        let directory = URL(fileURLWithPath: arguments[index + 1])
        let request = try WorkflowToolRequest.decode(Data(contentsOf: directory.appendingPathComponent("request.json")))
        let ready: [String: Any] = ["event": "operationReady", "operationID": request.operationID,
            "toolMode": "modelPreview", "requestSHA256": arguments[index + 2], "ownershipSHA256": arguments[index + 3]]
        line(String(decoding: try JSONSerialization.data(withJSONObject: ready), as: UTF8.self))
        let output = directory.appendingPathComponent("artifacts")
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: false)
        let cache = Data("opaque cache bytes".utf8)
        try cache.write(to: output.appendingPathComponent("activation-cache.pt"))
        let weights = request.arguments.contains("--checkpoint") ? "checkpoint" : "initialized"
        let hash = String(repeating: "a", count: 64)
        let event: [String: Any] = ["event": "previewCompleted", "initialWeightsHash": hash, "finalWeightsHash": hash,
            "recipeVersion": "test", "previews": ["hidden": ["shape": [1, 1], "dtype": "float32", "values": [1.0]]],
            "executedNodeIDs": ["hidden"], "reusedNodeIDs": [], "cachePath": output.appendingPathComponent("activation-cache.pt").path,
            "outputDir": output.path, "weights": weights, "device": "cpu", "sourceHash": NSNull(),
            "stopAt": "hidden", "inputProvenance": ["count": 1]]
        let metadata: [String: Any] = ["schemaVersion": 1, "recipeVersion": "test", "signature": hash,
            "fileHash": corrupt ? String(repeating: "f", count: 64) : WorkflowArchiveCodec.hash(cache),
            "weights": weights, "device": "cpu", "executedNodeIDs": ["hidden"], "reusedNodeIDs": [],
            "stopAt": "hidden", "inferenceOnly": true, "inputProvenance": ["count": 1]]
        try JSONSerialization.data(withJSONObject: metadata).write(to: output.appendingPathComponent("activation-cache.json"))
        try JSONSerialization.data(withJSONObject: event).write(to: output.appendingPathComponent("preview.json"))
        line(String(decoding: try JSONSerialization.data(withJSONObject: event), as: UTF8.self))
        return directory
    }
    func testVerifiedExitRetainsWeightsThenDeletesPrivateInputs() async throws {
        let (root, _, _, runtime) = try WorkflowEvaluationRequestTests().fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let policy = try ResolvedExecutionBudget.resolve(global: .factory, projectMaximumRunSeconds: nil)
        let window = try ExecutionBudgetWindow(policy: policy, startedAtUptime: ProcessInfo.processInfo.systemUptime)
        let graph = Data("{}".utf8), weights = Data("checkpoint".utf8)
        var exit: ((Int32, String) -> Void)?, directory: URL?, received: URL?
        let finished = expectation(description: "verified preview exit")
        _ = try await WorkflowPreviewExecution.launch(workspace: root, projectID: UUID().uuidString,
            runtime: runtime, window: window, storageLimitBytes: 40_000_000, inputs: [.bytes(path: "input/graph.json", data: graph),
                .bytes(path: "input/weights/checkpoint.pt", data: weights), .bytes(path: "input/weights/graph.json", data: graph)],
            arguments: { self.arguments($0, checkpoint: true) },
            launcher: { _, args, _, line, onExit in directory = try self.output(arguments: args, line: line); exit = onExit; return .init(cancel: {}) },
            onLine: { _ in }, onExit: { code, message, output in XCTAssertEqual(code, 0, message); received = output; finished.fulfill() })
        let owned = try XCTUnwrap(directory)
        let ledger = try WorkflowStorageAdmission(workspace: root)
        XCTAssertGreaterThan(try ledger.snapshot().reservedGrowthBytes, 0)
        XCTAssertThrowsError(try WorkflowFileLease(directory: owned, name: "owner.lock", mode: .shared))
        exit?(0, "")
        await fulfillment(of: [finished], timeout: 5)
        XCTAssertEqual(received, owned.appendingPathComponent("artifacts"))
        XCTAssertEqual(try ledger.snapshot().reservedGrowthBytes, 0)
        XCTAssertGreaterThan(try ledger.snapshot().retainedBytes, 0)
        XCTAssertFalse(FileManager.default.fileExists(atPath: owned.appendingPathComponent("input").path))
        XCTAssertEqual(try Data(contentsOf: owned.appendingPathComponent("artifacts/weights/checkpoint.pt")), weights)
        XCTAssertEqual(try Data(contentsOf: owned.appendingPathComponent("artifacts/weights/graph.json")), graph)
        let released = try WorkflowFileLease(directory: owned, name: "owner.lock", mode: .exclusive); released.close()
    }
    func testTimeoutCancellationAndCorruptCacheNeverPublishUsableURL() async throws {
        for scenario in ["timeout", "cancel", "corrupt"] {
            let (root, _, _, runtime) = try WorkflowEvaluationRequestTests().fixture()
            defer { try? FileManager.default.removeItem(at: root) }
            let policy = try ResolvedExecutionBudget.resolve(global: .factory, projectMaximumRunSeconds: nil)
            let window = try ExecutionBudgetWindow(policy: policy, startedAtUptime: ProcessInfo.processInfo.systemUptime)
            var exit: ((Int32, String) -> Void)?, directory: URL?
            let finished = expectation(description: scenario)
            let handle = try await WorkflowPreviewExecution.launch(workspace: root, projectID: UUID().uuidString,
                runtime: runtime, window: window, inputs: [.bytes(path: "input/graph.json", data: Data("{}".utf8))],
                arguments: { self.arguments($0, checkpoint: false) },
                launcher: { _, args, _, line, onExit in directory = try self.output(arguments: args, line: line, corrupt: scenario == "corrupt"); exit = onExit; return .init(cancel: {}) },
                onLine: { _ in }, onExit: { code, _, output in XCTAssertNotEqual(code, 0); XCTAssertNil(output); finished.fulfill() })
            if scenario == "cancel" { handle.cancel() }
            let owned = try XCTUnwrap(directory)
            XCTAssertTrue(FileManager.default.fileExists(atPath: owned.appendingPathComponent("input").path))
            exit?(scenario == "timeout" ? 124 : 0, "")
            await fulfillment(of: [finished], timeout: 5)
            XCTAssertFalse(FileManager.default.fileExists(atPath: owned.appendingPathComponent("input").path))
        }
    }
    func testExhaustedWindowRefusesCaptureLaunchAndLeavesNoPrivateStage() async throws {
        let (root, _, _, runtime) = try WorkflowEvaluationRequestTests().fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let policy = try ResolvedExecutionBudget.resolve(global: .init(maximumRunSeconds: 1), projectMaximumRunSeconds: nil)
        let window = try ExecutionBudgetWindow(policy: policy, startedAtUptime: ProcessInfo.processInfo.systemUptime - 2)
        var launches = 0
        do {
            _ = try await WorkflowPreviewExecution.launch(workspace: root, projectID: UUID().uuidString,
                runtime: runtime, window: window, inputs: [.bytes(path: "input/graph.json", data: Data("{}".utf8))],
                arguments: { self.arguments($0, checkpoint: false) },
                launcher: { _, _, _, _, _ in launches += 1; return .init(cancel: {}) }, onLine: { _ in }, onExit: { _, _, _ in })
            XCTFail("Exhausted operation launched")
        } catch { guard case ExecutionBudgetError.wallTimeLimitExceeded = error else { return XCTFail(error.localizedDescription) } }
        XCTAssertEqual(launches, 0)
        let stages = root.appendingPathComponent("preview-staging")
        if FileManager.default.fileExists(atPath: stages.path) { XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: stages.path).isEmpty) }
    }
    func testNativeRemainingDeadlineCancelsWorkerAndExplicitCancellationWins() async throws {
        for explicitCancel in [false, true] {
            let (root, _, _, runtime) = try WorkflowEvaluationRequestTests().fixture()
            defer { try? FileManager.default.removeItem(at: root) }
            let policy = try ResolvedExecutionBudget.resolve(global: .init(maximumRunSeconds: 1), projectMaximumRunSeconds: nil)
            let window = try ExecutionBudgetWindow(policy: policy, startedAtUptime: ProcessInfo.processInfo.systemUptime)
            let stopped = expectation(description: "remaining deadline cancels worker")
            let finished = expectation(description: "deadline result")
            var exit: ((Int32, String) -> Void)?
            let handle = try await WorkflowPreviewExecution.launch(workspace: root, projectID: UUID().uuidString,
                runtime: runtime, window: window, inputs: [.bytes(path: "input/graph.json", data: Data("{}".utf8))],
                arguments: { self.arguments($0, checkpoint: false) },
                launcher: { _, _, _, _, onExit in exit = onExit; return .init(cancel: { stopped.fulfill() }) },
                onLine: { _ in }, onExit: { code, _, output in
                    XCTAssertEqual(code, explicitCancel ? 130 : 124); XCTAssertNil(output); finished.fulfill()
                })
            await fulfillment(of: [stopped], timeout: 5)
            if explicitCancel { handle.cancel() }
            exit?(130, "")
            await fulfillment(of: [finished], timeout: 5)
        }
    }
    func testRelayRetainsOwnerAndStickyCancellationWhileFinalizing() async throws {
        let (root, _, _, runtime) = try WorkflowEvaluationRequestTests().fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let project = UUID().uuidString, operation = UUID().uuidString
        let request = WorkflowToolRequest(operationID: operation, projectID: project, requestIdentity: String(repeating: "a", count: 64),
            toolMode: .learningRuntime, arguments: ["--learning-runtime"], inputFiles: [],
            executionBudget: try .resolve(global: .factory, projectMaximumRunSeconds: nil), remainingSeconds: 10)
        let owner = try WorkflowOperationOwnership.create(workspace: root, request: request)
        try owner.publishRequest(request)
        let pin = try owner.prepareLaunch()
        let entered = expectation(description: "finalizer entered"), finished = expectation(description: "finalizer finished")
        var continuation: CheckedContinuation<Void, Never>?
        let handle = try WorkflowToolExecution.launchPrepared(owner: owner, request: request, runtime: runtime,
            ownershipSHA256: pin,
            launcher: { _, _, _, line, exit in
                let ready: [String: Any] = ["event": "operationReady", "operationID": operation, "toolMode": "learningRuntime",
                    "requestSHA256": owner.metadata.requestSHA256, "ownershipSHA256": pin]
                line(String(decoding: try JSONSerialization.data(withJSONObject: ready), as: UTF8.self)); exit(0, "")
                return .init(cancel: {})
            }, finalize: { _, _ in await withCheckedContinuation { continuation = $0; entered.fulfill() } },
            onLine: { _ in }, onExit: { code, _ in XCTAssertEqual(code, 130); finished.fulfill() })
        await fulfillment(of: [entered], timeout: 5)
        XCTAssertThrowsError(try WorkflowFileLease(directory: owner.directory, name: "owner.lock", mode: .shared))
        handle.cancel(); continuation?.resume()
        await fulfillment(of: [finished], timeout: 5)
        let released = try WorkflowFileLease(directory: owner.directory, name: "owner.lock", mode: .exclusive); released.close()
    }
}
