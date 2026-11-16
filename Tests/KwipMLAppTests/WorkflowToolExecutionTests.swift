import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowToolExecutionTests: XCTestCase {
    func testCodeExportStagesPinnedGraphAndRetainsItUntilTerminalExit() throws {
        let helper = WorkflowEvaluationRequestTests()
        let (root, _, _, runtime) = try helper.fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let policy = try ResolvedExecutionBudget.resolve(global: .factory, projectMaximumRunSeconds: nil, capabilityMaximumSeconds: 30)
        let window = try ExecutionBudgetWindow(policy: policy, startedAtUptime: ProcessInfo.processInfo.systemUptime)
        let graph = Data("{\"schemaVersion\":1}".utf8)
        var exit: ((Int32, String) -> Void)?, arguments: [String] = [], exits: [Int32] = []
        let handle = try WorkflowToolExecution.launch(workspace: root, projectID: UUID().uuidString,
            runtime: runtime, window: window, toolMode: .codeExport, inputs: ["input/graph.json": graph], maximumSeconds: 30,
            launcher: { _, args, _, _, onExit in arguments = args; exit = onExit; return .init(cancel: {}) },
            onLine: { _ in }, onExit: { code, _ in exits.append(code) })
        let index = try XCTUnwrap(arguments.firstIndex(of: "--kwip-operation-supervise"))
        let directory = URL(fileURLWithPath: arguments[index + 1])
        let request = try WorkflowToolRequest.decode(Data(contentsOf: directory.appendingPathComponent("request.json")))
        XCTAssertEqual(request.toolMode, .codeExport)
        XCTAssertEqual(request.arguments, ["--export-code", "--graph", directory.appendingPathComponent("input/graph.json").path])
        XCTAssertEqual(try Data(contentsOf: directory.appendingPathComponent("input/graph.json")), graph)
        XCTAssertLessThanOrEqual(request.remainingSeconds, 30)
        handle.cancel()
        XCTAssertTrue(exits.isEmpty)
        XCTAssertTrue(FileManager.default.fileExists(atPath: directory.appendingPathComponent("input/graph.json").path))
        exit?(130, "")
        XCTAssertEqual(exits, [130])
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.appendingPathComponent("input").path))
    }
    func testWorkflowCodeExportStagesSnapshotWithoutGraphOrDataGrants() throws {
        let helper = WorkflowEvaluationRequestTests()
        let (root, _, _, runtime) = try helper.fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let policy = try ResolvedExecutionBudget.resolve(global: .factory, projectMaximumRunSeconds: nil, capabilityMaximumSeconds: 30)
        let window = try ExecutionBudgetWindow(policy: policy, startedAtUptime: ProcessInfo.processInfo.systemUptime)
        let snapshot = Data("{\"schemaVersion\":1,\"scope\":\"workflow\"}".utf8)
        var exit: ((Int32, String) -> Void)?, arguments: [String] = [], exits: [Int32] = []
        let handle = try WorkflowToolExecution.launch(workspace: root, projectID: UUID().uuidString,
            runtime: runtime, window: window, toolMode: .codeExport, inputs: ["input/workflow-code.json": snapshot], maximumSeconds: 30,
            launcher: { _, args, _, _, onExit in arguments = args; exit = onExit; return .init(cancel: {}) },
            onLine: { _ in }, onExit: { code, _ in exits.append(code) })
        let index = try XCTUnwrap(arguments.firstIndex(of: "--kwip-operation-supervise"))
        let directory = URL(fileURLWithPath: arguments[index + 1])
        let request = try WorkflowToolRequest.decode(Data(contentsOf: directory.appendingPathComponent("request.json")))
        XCTAssertEqual(request.toolMode, .codeExport)
        XCTAssertEqual(request.arguments, ["--export-workflow-code", "--workflow-code-plan", directory.appendingPathComponent("input/workflow-code.json").path])
        XCTAssertEqual(request.inputFiles.map(\.path), ["input/workflow-code.json"])
        XCTAssertEqual(try Data(contentsOf: directory.appendingPathComponent("input/workflow-code.json")), snapshot)
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.appendingPathComponent("input/graph.json").path))
        handle.cancel()
        XCTAssertTrue(exits.isEmpty)
        XCTAssertTrue(FileManager.default.fileExists(atPath: directory.appendingPathComponent("input/workflow-code.json").path))
        exit?(130, "")
        XCTAssertEqual(exits, [130])
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.appendingPathComponent("input").path))
    }
    func testExhaustedSharedWindowNeverLaunchesAnotherTool() throws {
        let helper = WorkflowEvaluationRequestTests()
        let (root, _, _, runtime) = try helper.fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let policy = try ResolvedExecutionBudget.resolve(global: .init(maximumRunSeconds: 1), projectMaximumRunSeconds: nil)
        let window = try ExecutionBudgetWindow(policy: policy, startedAtUptime: ProcessInfo.processInfo.systemUptime - 2)
        var launches = 0
        XCTAssertThrowsError(try WorkflowToolExecution.launch(workspace: root, projectID: UUID().uuidString,
            runtime: runtime, window: window, toolMode: .learningRuntime, inputs: [:], maximumSeconds: 10,
            launcher: { _, _, _, _, _ in launches += 1; return .init(cancel: {}) }, onLine: { _ in }, onExit: { _, _ in })) { error in
            guard case ExecutionBudgetError.wallTimeLimitExceeded = error else { return XCTFail(error.localizedDescription) }
        }
        XCTAssertEqual(launches, 0)
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("workflow-operations").path))
    }
    func testReadinessAndCancellationRetainAuthorityUntilExit() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let helper = WorkflowEvaluationRequestTests()
        let (fixture, _, _, runtime) = try helper.fixture()
        defer { try? FileManager.default.removeItem(at: fixture) }
        let policy = try ResolvedExecutionBudget.resolve(global: .factory, projectMaximumRunSeconds: nil, capabilityMaximumSeconds: 65)
        let window = try ExecutionBudgetWindow(policy: policy, startedAtUptime: ProcessInfo.processInfo.systemUptime)
        var line: ((String) -> Void)?, exit: ((Int32, String) -> Void)?
        var args: [String] = [], cancelled = 0, observed: [String] = [], exits: [Int32] = []
        let handle = try WorkflowToolExecution.launch(workspace: root, projectID: UUID().uuidString,
            runtime: runtime, window: window, toolMode: .learningRuntime, inputs: [:], maximumSeconds: 10,
            launcher: { _, arguments, _, onLine, onExit in
                args = arguments; line = onLine; exit = onExit
                return .init(cancel: { cancelled += 1 })
            }, onLine: { observed.append($0) }, onExit: { code, _ in exits.append(code) })
        let index = try XCTUnwrap(args.firstIndex(of: "--kwip-operation-supervise"))
        let folder = URL(fileURLWithPath: args[index + 1])
        let request = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: folder.appendingPathComponent("request.json"))) as? [String: Any])
        XCTAssertEqual(request["toolMode"] as? String, "learningRuntime")
        XCTAssertLessThanOrEqual(try XCTUnwrap(request["remainingSeconds"] as? Double), 10)
        let ready: [String: Any] = ["event": "operationReady", "operationID": try XCTUnwrap(request["operationID"]), "toolMode": "learningRuntime",
                                   "requestSHA256": args[index + 2], "ownershipSHA256": args[index + 3]]
        line?(String(decoding: try JSONSerialization.data(withJSONObject: ready), as: UTF8.self))
        line?("{\"event\":\"learningRuntime\",\"versions\":{\"numpy\":\"test\"}}")
        XCTAssertEqual(observed.count, 1, "Internal readiness stays inside the tool service")
        handle.cancel()
        XCTAssertEqual(cancelled, 1)
        XCTAssertTrue(exits.isEmpty)
        XCTAssertTrue(FileManager.default.fileExists(atPath: folder.path))
        exit?(130, "")
        XCTAssertEqual(exits, [130])
        XCTAssertFalse(FileManager.default.fileExists(atPath: folder.appendingPathComponent("input").path))
        exit?(130, "")
        XCTAssertEqual(exits, [130])
    }
}
