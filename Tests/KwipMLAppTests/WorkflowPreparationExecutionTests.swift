import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowPreparationExecutionTests: XCTestCase {
    private let fixtureDigest = String(repeating: "a", count: 64)
    private func arguments(_ root: URL) -> [String] {
        ["--execute-dataflow", root.appendingPathComponent("input/graph.json").path,
         "--source-bindings", root.appendingPathComponent("input/bindings.json").path,
         "--dataflow-node-labels", root.appendingPathComponent("input/node-labels.json").path,
         "--output", root.appendingPathComponent("artifacts").path, "--dataflow-cache-policy", "off"]
    }
    private nonisolated static func inputs(_ root: URL) throws -> [WorkflowPreviewCapture.Input] {
        let graph = try DataFlowCodec.encodeGraph(.init(nodes: [.init(id: "source", operatorID: "kwip.data.sourceCSV")]))
        let csv = Data("x,y\n1,2\n".utf8), path = "input/sources/\(WorkflowArchiveCodec.hash(csv))/source.csv"
        return [.bytes(path: "input/graph.json", data: graph), .bytes(path: "input/bindings.json", data: try JSONEncoder().encode(["source": root.appendingPathComponent(path).path])),
                .bytes(path: "input/node-labels.json", data: Data("{}".utf8)), .bytes(path: path, data: csv)]
    }
    private func publish(_ args: [String], line: (String) -> Void, scenario: String) throws -> URL {
        let index = try XCTUnwrap(args.firstIndex(of: "--kwip-operation-supervise")), root = URL(fileURLWithPath: args[index + 1])
        let request = try WorkflowToolRequest.decode(Data(contentsOf: root.appendingPathComponent("request.json")))
        let ready: [String: Any] = ["event": "operationReady", "operationID": request.operationID, "toolMode": "dataPreparation", "requestSHA256": args[index + 2], "ownershipSHA256": args[index + 3]]
        line(String(decoding: try JSONSerialization.data(withJSONObject: ready), as: UTF8.self))
        let output = root.appendingPathComponent("artifacts")
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: false)
        let expectedGraph = try Data(contentsOf: root.appendingPathComponent("input/graph.json"))
        let graph = try scenario == "wrongGraph" ? DataFlowCodec.encodeGraph(.init(nodes: [.init(id: "another", operatorID: "kwip.data.sourceCSV")])) : expectedGraph
        let graphDigest = WorkflowArchiveCodec.hash(expectedGraph)
        try graph.write(to: output.appendingPathComponent("dataflow.json"))
        let cache: [String: Any] = ["schemaVersion": 1, "policy": "off", "implementationDigest": fixtureDigest,
            "sourceManifestSHA256": NSNull(), "nodes": [], "executedNodeIDs": ["source"], "reusedNodeIDs": []]
        let manifest: [String: Any] = ["schemaVersion": 1, "kind": "dataflowExecution", "status": "completed", "graphDigest": graphDigest,
            "artifacts": [["path": "dataflow.json", "bytes": graph.count, "sha256": WorkflowArchiveCodec.hash(graph), "kind": "graph"]], "cache": cache]
        try JSONSerialization.data(withJSONObject: manifest).write(to: output.appendingPathComponent("manifest.json"))
        var event: [String: Any] = ["event": "dataflowCompleted", "outputDir": output.path, "manifestPath": output.appendingPathComponent("manifest.json").path,
            "graphDigest": graphDigest, "artifacts": ["dataflow.json"], "cache": cache]
        if scenario == "wrongPath" { event["outputDir"] = "/tmp/unowned" }
        if scenario == "wrongCache" { var changed = cache; changed["policy"] = "materialize"; event["cache"] = changed }
        if scenario == "tamper" { try Data("{}".utf8).write(to: output.appendingPathComponent("dataflow.json")) }
        line(String(decoding: try JSONSerialization.data(withJSONObject: event), as: UTF8.self))
        return root
    }
    func testSuccessfulOwnedManifestVerificationPrecedesCleanupAndCallback() async throws {
        let (workspace, _, _, runtime) = try WorkflowEvaluationRequestTests().fixture()
        defer { try? FileManager.default.removeItem(at: workspace) }
        let policy = try ResolvedExecutionBudget.resolve(global: .factory, projectMaximumRunSeconds: nil)
        let window = try ExecutionBudgetWindow(policy: policy, startedAtUptime: ProcessInfo.processInfo.systemUptime)
        var root: URL?, exit: ((Int32, String) -> Void)?
        let done = expectation(description: "prepared")
        _ = try await WorkflowPreparationExecution.launch(workspace: workspace, projectID: UUID().uuidString, runtime: runtime, window: window,
            inputs: Self.inputs, arguments: arguments,
            launcher: { _, args, _, line, onExit in root = try self.publish(args, line: line, scenario: "valid"); exit = onExit; return .init(cancel: {}) },
            onLine: { _ in }, onExit: { code, message, output in XCTAssertEqual(code, 0, message); XCTAssertEqual(output, root?.appendingPathComponent("artifacts")); done.fulfill() })
        let directory = try XCTUnwrap(root)
        XCTAssertTrue(FileManager.default.fileExists(atPath: directory.appendingPathComponent("input").path))
        XCTAssertThrowsError(try WorkflowFileLease(directory: directory, name: "owner.lock", mode: .shared))
        exit?(0, "")
        await fulfillment(of: [done], timeout: 5)
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.appendingPathComponent("input").path))
        let lease = try WorkflowFileLease(directory: directory, name: "owner.lock", mode: .exclusive); lease.close()
    }
    func testTamperCacheMismatchWrongPathTimeoutAndCancellationNeverYieldOutput() async throws {
        for scenario in ["tamper", "wrongGraph", "wrongCache", "wrongPath", "timeout", "cancel"] {
            let (workspace, _, _, runtime) = try WorkflowEvaluationRequestTests().fixture()
            defer { try? FileManager.default.removeItem(at: workspace) }
            let window = try ExecutionBudgetWindow(policy: .resolve(global: .factory, projectMaximumRunSeconds: nil), startedAtUptime: ProcessInfo.processInfo.systemUptime)
            var root: URL?, exit: ((Int32, String) -> Void)?
            let done = expectation(description: scenario)
            let handle = try await WorkflowPreparationExecution.launch(workspace: workspace, projectID: UUID().uuidString, runtime: runtime, window: window,
                inputs: Self.inputs, arguments: arguments,
                launcher: { _, args, _, line, onExit in root = try self.publish(args, line: line, scenario: scenario); exit = onExit; return .init(cancel: {}) },
                onLine: { _ in }, onExit: { code, _, output in XCTAssertNotEqual(code, 0); XCTAssertNil(output); done.fulfill() })
            if scenario == "cancel" { handle.cancel() }
            let directory = try XCTUnwrap(root)
            XCTAssertTrue(FileManager.default.fileExists(atPath: directory.appendingPathComponent("input").path))
            exit?(scenario == "timeout" ? 124 : 0, "")
            await fulfillment(of: [done], timeout: 5)
            XCTAssertFalse(FileManager.default.fileExists(atPath: directory.appendingPathComponent("input").path))
        }
    }
}
