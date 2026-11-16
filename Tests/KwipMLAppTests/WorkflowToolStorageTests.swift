import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowToolStorageTests: XCTestCase {
    private func window() throws -> ExecutionBudgetWindow {
        try .init(policy: .resolve(global: .factory, projectMaximumRunSeconds: 30), startedAtUptime: ProcessInfo.processInfo.systemUptime)
    }
    func testStorageRefusalPrecedesOwnerAndInputPublication() throws {
        let (root, _, _, runtime) = try WorkflowEvaluationRequestTests().fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        var launched = false
        XCTAssertThrowsError(try WorkflowToolExecution.launch(workspace: root, projectID: UUID().uuidString,
            runtime: runtime, window: window(), toolMode: .codeExport,
            inputs: ["input/graph.json": Data("{}".utf8)], maximumSeconds: 30, storageLimitBytes: 1,
            launcher: { _, _, _, _, _ in launched = true; return .init(cancel: {}) },
            onLine: { _ in }, onExit: { _, _ in }))
        XCTAssertFalse(launched)
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("workflow-operations").path))
        XCTAssertEqual(try WorkflowStorageAdmission(workspace: root).snapshot().committedBytes, 0)
    }
    func testCancellationAndConcurrentAdmissionKeepReservationUntilActualExit() throws {
        let (root, _, _, runtime) = try WorkflowEvaluationRequestTests().fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let ledger = try WorkflowStorageAdmission(workspace: root)
        var exit: ((Int32, String) -> Void)?, exits = 0
        let handle = try WorkflowToolExecution.launch(workspace: root, projectID: UUID().uuidString,
            runtime: runtime, window: window(), toolMode: .codeExport,
            inputs: ["input/graph.json": Data("{}".utf8)], maximumSeconds: 30, storageLimitBytes: 2_000_000,
            launcher: { _, _, _, _, terminal in exit = terminal; return .init(cancel: {}) },
            onLine: { _ in }, onExit: { _, _ in exits += 1 })
        let live = try ledger.snapshot()
        XCTAssertGreaterThan(live.reservedGrowthBytes, 0)
        handle.cancel()
        XCTAssertEqual(try ledger.snapshot().committedBytes, live.committedBytes)
        XCTAssertEqual(exits, 0)
        var extraLaunched = false
        XCTAssertThrowsError(try WorkflowToolExecution.launch(workspace: root, projectID: UUID().uuidString,
            runtime: runtime, window: window(), toolMode: .codeExport,
            inputs: ["input/graph.json": Data("{}".utf8)], maximumSeconds: 30, storageLimitBytes: live.committedBytes,
            launcher: { _, _, _, _, _ in extraLaunched = true; return .init(cancel: {}) },
            onLine: { _ in }, onExit: { _, _ in }))
        XCTAssertFalse(extraLaunched)
        exit?(130, "")
        XCTAssertEqual(exits, 1)
        let finished = try ledger.snapshot()
        XCTAssertEqual(finished.reservedGrowthBytes, 0)
        XCTAssertGreaterThan(finished.retainedBytes, 0, "Request and ownership evidence remains charged")
        XCTAssertLessThan(finished.committedBytes, live.committedBytes)
    }
    func testAssessmentReservesOneBoundedFixtureUntilActualExit() throws {
        let (root, _, _, runtime) = try WorkflowEvaluationRequestTests().fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let ledger = try WorkflowStorageAdmission(workspace: root)
        let inputs = ["input/assessment.json": Data("{}".utf8), "input/submission.json": Data("{}".utf8)]
        let peak: Int64 = 1_000_000 + 4 + 65_536
        var launched = false
        XCTAssertThrowsError(try WorkflowToolExecution.launch(workspace: root, projectID: UUID().uuidString,
            runtime: runtime, window: window(), toolMode: .learningAssessment, inputs: inputs,
            maximumSeconds: 30, storageLimitBytes: peak - 1,
            launcher: { _, _, _, _, _ in launched = true; return .init(cancel: {}) },
            onLine: { _ in }, onExit: { _, _ in }))
        XCTAssertFalse(launched)
        XCTAssertEqual(try ledger.snapshot().committedBytes, 0)
        var exit: ((Int32, String) -> Void)?
        let handle = try WorkflowToolExecution.launch(workspace: root, projectID: UUID().uuidString,
            runtime: runtime, window: window(), toolMode: .learningAssessment, inputs: inputs,
            maximumSeconds: 30, storageLimitBytes: peak,
            launcher: { _, _, _, _, terminal in launched = true; exit = terminal; return .init(cancel: {}) },
            onLine: { _ in }, onExit: { _, _ in })
        XCTAssertTrue(launched)
        XCTAssertEqual(try ledger.snapshot().committedBytes, peak)
        handle.cancel()
        XCTAssertEqual(try ledger.snapshot().committedBytes, peak)
        exit?(130, "")
        XCTAssertEqual(try ledger.snapshot().reservedGrowthBytes, 0)
        XCTAssertGreaterThan(try ledger.snapshot().retainedBytes, 0)
    }
    func testAsyncAdmissionRefusesBeforeWritesAndRetainsThroughExit() async throws {
        let (root, _, _, runtime) = try WorkflowEvaluationRequestTests().fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        var launched = false
        do {
            _ = try await WorkflowToolExecution.launchAsync(workspace: root, projectID: UUID().uuidString,
                runtime: runtime, window: window(), toolMode: .codeExport, inputs: ["input/graph.json": Data("{}".utf8)],
                maximumSeconds: 30, storageLimitBytes: 1,
                launcher: { _, _, _, _, _ in launched = true; return .init(cancel: {}) }, onLine: { _ in }, onExit: { _, _ in })
            XCTFail("Expected quota refusal")
        } catch { XCTAssertFalse(launched) }
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("workflow-operations").path))
        var exit: ((Int32, String) -> Void)?
        let handle = try await WorkflowToolExecution.launchAsync(workspace: root, projectID: UUID().uuidString,
            runtime: runtime, window: window(), toolMode: .codeExport, inputs: ["input/graph.json": Data("{}".utf8)],
            maximumSeconds: 30, storageLimitBytes: 2_000_000,
            launcher: { _, _, _, _, terminal in XCTAssertTrue(Thread.isMainThread); exit = terminal; return .init(cancel: {}) },
            onLine: { _ in }, onExit: { _, _ in })
        let ledger = try WorkflowStorageAdmission(workspace: root)
        XCTAssertEqual(try ledger.snapshot().committedBytes, 1_000_002)
        handle.cancel()
        XCTAssertEqual(try ledger.snapshot().committedBytes, 1_000_002)
        exit?(143, "")
        XCTAssertEqual(try ledger.snapshot().reservedGrowthBytes, 0)
    }
    func testActualCodeExportCompletesWithBoundReservationAndReleasesOnlyGrowth() async throws {
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: repo, storageOverride: root, detectCapabilities: false)
        let runtime = try XCTUnwrap(workspace.runtime)
        let ledger = try WorkflowStorageAdmission(workspace: root)
        let finished = expectation(description: "storage-admitted code export")
        var code = ""
        let handle = try WorkflowToolExecution.launch(workspace: root, projectID: UUID().uuidString,
            runtime: runtime, window: window(), toolMode: .codeExport,
            inputs: ["input/graph.json": try JSONEncoder().encode(ExampleFlows.parallelClassifier)],
            maximumSeconds: 30, storageLimitBytes: 4_000_000,
            onLine: { line in
                if let event = try? JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any],
                   event["event"] as? String == "codeExport" { code = event["code"] as? String ?? "" }
            }, onExit: { status, message in XCTAssertEqual(status, 0, message); finished.fulfill() })
        _ = handle
        await fulfillment(of: [finished], timeout: 30)
        XCTAssertTrue(code.contains("def forward"))
        XCTAssertEqual(try ledger.snapshot().reservedGrowthBytes, 0)
        XCTAssertGreaterThan(try ledger.snapshot().retainedBytes, 0)
    }
}
