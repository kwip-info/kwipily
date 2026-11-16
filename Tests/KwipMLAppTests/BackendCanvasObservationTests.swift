import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp

@MainActor final class BackendCanvasObservationTests: XCTestCase {
    func testBackendChangesReachCanvasWithoutHostFeedbackCycleAndRejectOldSessions() async throws {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
        XCTAssertTrue(workspace.loadBackendWorkflowExample(backendID: "sklearn"), workspace.statusMessage)
        let host = try XCTUnwrap(workspace.workflowSession)
        // This test never launches a worker or requires installed sklearn. Resolve
        // inert file locations only; the isolated sessions remain idle throughout.
        let runtimeRoot = root.appendingPathComponent("inert-runtime")
        for path in [".venv/bin/python", "runner/main.py", "Sources/KwipMLCore/Resources/components.json"] {
            let file = runtimeRoot.appendingPathComponent(path)
            try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data().write(to: file)
        }
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: runtimeRoot.appendingPathComponent(".venv/bin/python").path)
        let runtime = try RuntimeLocation.resolve(resources: nil, developmentRoot: runtimeRoot, allowDevelopment: true)
        let plan = try WorkflowEstimatorCompiler.compile(host.owner.document, estimator: WorkflowExamples.sklearnEstimatorAddress,
            modelRegistry: host.owner.modelRegistry, dataRegistry: host.owner.dataRegistry)
        let policy = try workspace.executionPolicy(for: host.owner.projectID)
        func session() -> BackendFitSession {
            BackendFitSession(workspace: workspace, owner: host.owner, bindings: host.bindings, history: host.backendHistory,
                plan: plan, runtime: runtime, policy: policy, launcher: { _, _, _, _, _ in
                    XCTFail("Observation test must not launch a worker"); throw CancellationError()
                })
        }
        let first = session(), second = session()
        host.observeBackendFit(first)
        XCTAssertEqual(host.currentResult?.runID,first.runID)
        XCTAssertTrue(host.currentBackendFit === first)
        XCTAssertNil(host.currentExecution)
        XCTAssertTrue(host.currentResult?.containsCapturedComponent(plan.estimator) == true)
        XCTAssertFalse(host.currentResult?.containsCapturedComponent(.init(domain:.model,localID:plan.estimator.localID)) == true)
        // Let the example’s independent History reload settle before counting.
        try await Task.sleep(for: .milliseconds(300))
        var canvasChanges = 0, hostChanges = 0
        let canvasObservation = host.canvas.objectWillChange.sink { canvasChanges += 1 }
        let hostObservation = host.objectWillChange.sink { hostChanges += 1 }
        defer { canvasObservation.cancel(); hostObservation.cancel() }
        first.objectWillChange.send()
        try await Task.sleep(for: .milliseconds(250))
        XCTAssertEqual(canvasChanges, 1, "Backend status must invalidate its canvas directly")
        XCTAssertEqual(hostChanges, 1, "Canvas already forwards to host; no duplicate notification or cycle")
        host.observeBackendFit(second)
        XCTAssertEqual(host.currentResult?.runID,second.runID)
        XCTAssertTrue(host.currentBackendFit === second)
        let afterReplacement = canvasChanges
        first.objectWillChange.send()
        try await Task.sleep(for: .milliseconds(250))
        XCTAssertEqual(canvasChanges, afterReplacement, "Replaced sessions lose presentation authority")
        second.objectWillChange.send()
        try await Task.sleep(for: .milliseconds(250))
        XCTAssertEqual(canvasChanges, afterReplacement + 1)
        host.retire()
        XCTAssertNil(host.currentResult)
        XCTAssertNil(host.currentBackendFit)
        let afterRetirement = canvasChanges
        second.objectWillChange.send()
        try await Task.sleep(for: .milliseconds(250))
        XCTAssertEqual(canvasChanges, afterRetirement, "Retired host must cancel delayed backend updates")
    }
}
