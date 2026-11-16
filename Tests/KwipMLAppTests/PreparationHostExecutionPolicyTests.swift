import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class PreparationHostExecutionPolicyTests: XCTestCase {
    func testAcceptedProjectPolicyIsPinnedAndQuotaRefusesBeforeSourceStaging() async throws {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        for refuse in [false, true] {
            let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: root) }
            let workspace = WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
            await verifyRuntimeForIntegration(workspace)
            workspace.activeProjectID = UUID().uuidString
            let project = try XCTUnwrap(workspace.activeProjectID)
            let store = DataFlowWorkspaceStore(workspace: workspace)
            store.loadExample(columns: ["x", "y"])
            let source = root.appendingPathComponent("observations.csv")
            try Data("x,y\n1,2\n2,4\n3,6\n".utf8).write(to: source)
            store.bindSource(source.path, nodeID: "source")
            XCTAssertTrue(workspace.setExecutionPreferences(try .init(maximumRunSeconds: 120, managedStorageLimitBytes: refuse ? 1 : 2_000_000_000)))
            XCTAssertTrue(workspace.setProjectExecutionOverrides(try .init(maximumRunSeconds: 90, cpuThreads: 2, memory: .limit(1_073_741_824)), projectID: project))
            var request: WorkflowToolRequest?
            store.preparationWorkerLauncher = { _, args, _, _, exit in
                let index = try XCTUnwrap(args.firstIndex(of: "--kwip-operation-supervise"))
                request = try WorkflowToolRequest.decode(Data(contentsOf: URL(fileURLWithPath: args[index + 1]).appendingPathComponent("request.json")))
                exit(143, "Test stops after request capture")
                return .init(cancel: {})
            }
            store.runPreparation()
            XCTAssertTrue(workspace.resetProjectExecutionOverrides(projectID: project))
            let deadline = Date().addingTimeInterval(10)
            while store.isBusy && Date() < deadline { try await Task.sleep(for: .milliseconds(10)) }
            XCTAssertFalse(store.isBusy, store.statusMessage)
            XCTAssertFalse(workspace.isPreparingData)
            if refuse {
                XCTAssertNil(request)
                XCTAssertTrue(store.statusMessage.lowercased().contains("storage"), store.statusMessage)
            } else {
                let captured = try XCTUnwrap(request, store.statusMessage)
                XCTAssertEqual(captured.executionBudget.maximumRunSeconds, 90)
                XCTAssertEqual(captured.executionBudget.source, .project)
                XCTAssertEqual(captured.executionBudget.resources?.cpuThreads, 2)
                XCTAssertEqual(captured.executionBudget.resources?.maximumResidentBytes, 1_073_741_824)
            }
            XCTAssertEqual(try WorkflowStorageAdmission(workspace: root).snapshot().reservedGrowthBytes, 0)
        }
    }
    func testRestorationCapturesProjectPolicyAndDefaultPathRefusesTinyQuota() async throws {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
        await verifyRuntimeForIntegration(workspace)
        workspace.activeProjectID = UUID().uuidString
        let project = try XCTUnwrap(workspace.activeProjectID)
        let store = DataFlowWorkspaceStore(workspace: workspace)
        store.loadExample(columns: ["x", "y"])
        let source = root.appendingPathComponent("observations.csv")
        try Data((["x,y"] + (0..<40).map { "\($0),\($0 * 2)" }).joined(separator: "\n").utf8).write(to: source)
        store.bindSource(source.path, nodeID: "source")
        store.runPreparation()
        try await waitForIdle(store)
        let original = try XCTUnwrap(store.manifestURL, store.statusMessage)
        XCTAssertTrue(workspace.setProjectExecutionOverrides(try .init(maximumRunSeconds: 70, cpuThreads: 3), projectID: project))
        var captured: ResolvedExecutionBudget?
        let actual = store.resultRestorationExecute
        store.resultRestorationExecute = { _, _, _, window, _, _, exit in
            captured = window.policy; exit(1, "fixture stopped", nil); return .init(cancel: {})
        }
        store.restoreCompletedOutputs()
        XCTAssertTrue(workspace.resetProjectExecutionOverrides(projectID: project))
        try await waitForIdle(store)
        XCTAssertEqual(captured?.maximumRunSeconds, 70)
        XCTAssertEqual(captured?.resources?.cpuThreads, 3)
        store.resultRestorationExecute = actual
        XCTAssertTrue(workspace.setExecutionPreferences(try .init(managedStorageLimitBytes: 1)))
        store.restoreCompletedOutputs()
        try await waitForIdle(store)
        XCTAssertFalse(store.restoredResult)
        XCTAssertTrue(store.statusMessage.lowercased().contains("storage"), store.statusMessage)
        XCTAssertEqual(store.manifestURL, original)
        XCTAssertEqual(try WorkflowStorageAdmission(workspace: root).snapshot().reservedGrowthBytes, 0)
    }
    private func waitForIdle(_ store: DataFlowWorkspaceStore) async throws {
        let deadline = Date().addingTimeInterval(20)
        while store.isBusy && Date() < deadline { try await Task.sleep(for: .milliseconds(10)) }
        XCTAssertFalse(store.isBusy, store.statusMessage)
    }

}
