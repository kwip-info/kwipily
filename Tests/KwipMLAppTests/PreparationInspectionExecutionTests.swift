import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class PreparationInspectionExecutionTests: XCTestCase {
    private func fixture() async throws -> (URL, RuntimeLocation, URL, String) {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
        await verifyRuntimeForIntegration(workspace)
        workspace.activeProjectID = UUID().uuidString
        let store = DataFlowWorkspaceStore(workspace: workspace)
        store.loadExample(columns: ["x", "y"])
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let source = root.appendingPathComponent("observations.csv")
        try (["x,y"] + (0..<40).map { "\($0),\($0 * 2)" }).joined(separator: "\n").write(to: source, atomically: true, encoding: .utf8)
        store.bindSource(source.path, nodeID: "source")
        store.runPreparation()
        let deadline = Date().addingTimeInterval(30)
        while store.isBusy && Date() < deadline { try await Task.sleep(nanoseconds: 20_000_000) }
        XCTAssertFalse(store.isBusy, store.statusMessage)
        let manifest = try XCTUnwrap(store.manifestURL, store.statusMessage)
        let pin = WorkflowArchiveCodec.hash(try Data(contentsOf: manifest))
        try FileManager.default.removeItem(at: source)
        return (root, try XCTUnwrap(workspace.runtime), manifest, pin)
    }
    func testStorageRefusalDoesNotCreateWorkerOrInspectionStage() async throws {
        let (root, runtime, manifest, pin) = try await fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let window = try ExecutionBudgetWindow(policy: .resolve(global: .factory, projectMaximumRunSeconds: nil), startedAtUptime: ProcessInfo.processInfo.systemUptime)
        var launched = false
        let projectID = UUID().uuidString
        do {
            _ = try await PreparationInspectionExecution.launch(workspace: root, projectID: projectID,
                runtime: runtime, window: window, manifestURL: manifest, manifestSHA256: pin, storageLimitBytes: 1,
                launcher: { _, _, _, _, _ in launched = true; return .init(cancel: {}) }, onExit: { _, _, _ in XCTFail("Refused operation must not launch") })
            XCTFail("Insufficient allowance should refuse")
        } catch { XCTAssertTrue(error is WorkflowStorageAdmission.Failure) }
        XCTAssertFalse(launched)
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("workflow-operations/" + projectID).path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("preparation-inspection-staging").path))
        XCTAssertEqual(try WorkflowStorageAdmission(workspace: root).snapshot().reservedGrowthBytes, 0)
    }
    func testHistoricalInspectionUsesCapturedArchiveAndJoinsExitBeforePublishing() async throws {
        let (root, runtime, manifest, pin) = try await fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let window = try ExecutionBudgetWindow(policy: .resolve(global: .factory, projectMaximumRunSeconds: nil), startedAtUptime: ProcessInfo.processInfo.systemUptime)
        var operation: URL?, pendingExit: (() -> Void)?, callbackCount = 0
        let workerDone = expectation(description: "worker exited, caller still joined")
        let published = expectation(description: "verified saved outputs")
        let handle = try await PreparationInspectionExecution.launch(workspace: root, projectID: UUID().uuidString,
            runtime: runtime, window: window, manifestURL: manifest, manifestSHA256: pin, storageLimitBytes: 300_000_000,
            launcher: { executable, args, directory, line, exit in
                let index = try XCTUnwrap(args.firstIndex(of: "--kwip-operation-supervise"))
                operation = URL(fileURLWithPath: args[index + 1])
                return try WorkflowExecutionSession.launchWorker(executable, args, directory, line, { code, message in
                    pendingExit = { exit(code, message) }; workerDone.fulfill()
                })
            }, onExit: { code, message, result in
                XCTAssertEqual(try? WorkflowStorageAdmission(workspace: root).snapshot().reservedGrowthBytes, 0)
                callbackCount += 1; XCTAssertEqual(code, 0, message)
                XCTAssertFalse(result?.outputs.isEmpty ?? true)
                if let operation { XCTAssertFalse(FileManager.default.fileExists(atPath: operation.appendingPathComponent("input").path)) }
                published.fulfill()
            })
        _ = handle
        await fulfillment(of: [workerDone], timeout: 30)
        let owned = try XCTUnwrap(operation)
        XCTAssertGreaterThan(try WorkflowStorageAdmission(workspace: root).snapshot().reservedGrowthBytes, 0)
        XCTAssertEqual(callbackCount, 0)
        XCTAssertThrowsError(try WorkflowFileLease(directory: owned, name: "owner.lock", mode: .shared))
        XCTAssertTrue(FileManager.default.fileExists(atPath: owned.appendingPathComponent("input/preparation/manifest.json").path))
        // Once privately captured, the old archive may disappear without invalidating the historical sample.
        try FileManager.default.removeItem(at: manifest.deletingLastPathComponent())
        pendingExit?()
        await fulfillment(of: [published], timeout: 10)
        XCTAssertEqual(callbackCount, 1)
        let lease = try WorkflowFileLease(directory: owned, name: "owner.lock", mode: .exclusive); lease.close()
    }
    func testCancellationTimeoutMalformedCompletionAndPrivateTamperNeverPublish() async throws {
        let (root, runtime, manifest, pin) = try await fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        for scenario in ["cancel", "timeout", "wrongCompletion", "tamper"] {
            let window = try ExecutionBudgetWindow(policy: .resolve(global: .factory, projectMaximumRunSeconds: nil), startedAtUptime: ProcessInfo.processInfo.systemUptime)
            var operation: URL?, pendingExit: (() -> Void)?, callbackCount = 0
            let workerDone = expectation(description: "held " + scenario), finished = expectation(description: scenario)
            let handle = try await PreparationInspectionExecution.launch(workspace: root, projectID: UUID().uuidString,
                runtime: runtime, window: window, manifestURL: manifest, manifestSHA256: pin, storageLimitBytes: 300_000_000,
                launcher: { executable, args, directory, line, exit in
                    let index = try XCTUnwrap(args.firstIndex(of: "--kwip-operation-supervise"))
                    operation = URL(fileURLWithPath: args[index + 1])
                    return try WorkflowExecutionSession.launchWorker(executable, args, directory, { text in
                        if scenario == "wrongCompletion", let bytes = text.data(using: .utf8),
                           var event = (try? JSONSerialization.jsonObject(with: bytes)) as? [String: Any],
                           event["event"] as? String == "dataflowInspectionCompleted" {
                            event["manifestSHA256"] = String(repeating: "0", count: 64)
                            line(String(decoding: try! JSONSerialization.data(withJSONObject: event), as: UTF8.self))
                        } else { line(text) }
                    }, { code, message in
                        pendingExit = { exit(scenario == "timeout" ? 124 : code, message) }; workerDone.fulfill()
                    })
                }, onExit: { code, _, result in
                    XCTAssertEqual(try? WorkflowStorageAdmission(workspace: root).snapshot().reservedGrowthBytes, 0)
                    callbackCount += 1; XCTAssertNotEqual(code, 0); XCTAssertNil(result)
                    if scenario == "cancel" { XCTAssertEqual(code, 130) }
                    if scenario == "timeout" { XCTAssertEqual(code, 124) }
                    finished.fulfill()
                })
            await fulfillment(of: [workerDone], timeout: 30)
            let owned = try XCTUnwrap(operation)
            if scenario == "cancel" { handle.cancel() }
            if scenario == "tamper" { try Data("changed".utf8).write(to: owned.appendingPathComponent("input/preparation/manifest.json")) }
            XCTAssertGreaterThan(try WorkflowStorageAdmission(workspace: root).snapshot().reservedGrowthBytes, 0)
        XCTAssertEqual(callbackCount, 0)
            XCTAssertThrowsError(try WorkflowFileLease(directory: owned, name: "owner.lock", mode: .shared))
            pendingExit?()
            await fulfillment(of: [finished], timeout: 10)
            XCTAssertEqual(callbackCount, 1)
            XCTAssertFalse(FileManager.default.fileExists(atPath: owned.appendingPathComponent("input").path))
            let lease = try WorkflowFileLease(directory: owned, name: "owner.lock", mode: .exclusive); lease.close()
        }
    }
}
