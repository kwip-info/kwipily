import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowQueuedSessionTests: XCTestCase {
    private func wait(_ condition: () -> Bool, seconds: Double = 10) async throws {
        let end = Date().addingTimeInterval(seconds)
        while !condition(), Date() < end { try await Task.sleep(for: .milliseconds(10)) }
        XCTAssertTrue(condition(), "Queue boundary timed out")
    }
    func testActualHostCancellationStaleDispatchAndImmediateRefusalReleaseOwnership() async throws {
        let (root, _, runtime) = try WorkflowExecutionRequestTests().setup()
        defer { try? FileManager.default.removeItem(at: root) }
        func host() throws -> (WorkspaceStore, WorkflowWorkspaceSession) {
            let workspace = WorkspaceStore(rootOverride: runtime.workingDirectoryURL, storageOverride: root.appendingPathComponent("shared"), detectCapabilities: false)
            // This injected launcher cannot perform discovery; install its reviewed fixture contract.
            try installFixtureExecutionContract(workspace, version: "2.14.1")
            var document = CurrentExecutionWorkflowTestFixture.make(block:false); document.projectID = UUID().uuidString
            _ = try workspace.projectLibrary.saveSource(.workflow(document))
            XCTAssertTrue(workspace.beginWorkflowProject(id: document.projectID), workspace.statusMessage)
            let host = try XCTUnwrap(workspace.workflowSession)
            let csv = root.appendingPathComponent(document.projectID + ".csv")
            try Data("a,b,label\n1,2,0\n2,3,1\n".utf8).write(to: csv)
            XCTAssertTrue(host.bindings.setPath(csv.path, for: .init(domain: .data, localID: "input"), lease: host.owner.lease(), expectedRevision: 0))
            return (workspace, host)
        }
        let (a, first) = try host(), (b, second) = try host()
        XCTAssertTrue(first.executionQueue === second.executionQueue)
        var firstExit: ((Int32, String) -> Void)?, secondLaunches = 0
        first.executionLaunch = { _, _, _, _, done in firstExit = done; return .init(cancel: { done(143, "Cancelled actual host fixture") }) }
        second.executionLaunch = { _, _, _, _, _ in secondLaunches += 1; throw WorkflowArchiveError("Immediate launcher refusal") }
        XCTAssertTrue(first.run(),first.statusMessage)
        try await wait { firstExit != nil }
        XCTAssertTrue(second.run(),second.statusMessage)
        try await wait { second.queuedRequestID != nil }
        let cancelled = try XCTUnwrap(second.queuedRequestID)
        second.cancel()
        XCTAssertEqual(secondLaunches, 0)
        XCTAssertNil(second.queuedRequestID); XCTAssertFalse(b.isBusy)
        XCTAssertEqual(first.executionQueue.entries.first(where: {$0.id == cancelled})?.state, .cancelled)
        XCTAssertTrue(second.run(),second.statusMessage)
        try await wait { second.queuedRequestID != nil }
        let stale = try XCTUnwrap(second.queuedRequestID)
        XCTAssertTrue(second.owner.rename("Edited after capture", lease: second.owner.lease()))
        first.cancel()
        try await wait { !a.isBusy && !b.isBusy }
        XCTAssertEqual(secondLaunches, 0)
        XCTAssertEqual(first.executionQueue.entries.first(where: {$0.id == stale})?.state, .failed)
        XCTAssertNil(second.queuedRequestID)
        XCTAssertTrue(second.run(),second.statusMessage)
        try await wait { secondLaunches == 1 && !b.isBusy }
        XCTAssertNil(second.queuedRequestID)
        XCTAssertEqual(second.executionQueue.entries.last?.state, .failed)
        a.retireWorkflowSession(); b.retireWorkflowSession()
    }

    func testRealTrainAndExperimentGroupUseCapturedInputsThroughSharedQueue() async throws {
        guard let development = ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else { throw XCTSkip("Enable real workflow worker") }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let a = WorkspaceStore(rootOverride: URL(fileURLWithPath: development), storageOverride: root, detectCapabilities: false)
        let b = WorkspaceStore(rootOverride: URL(fileURLWithPath: development), storageOverride: root, detectCapabilities: false)
        await verifyRuntimeForIntegration(a); await verifyRuntimeForIntegration(b)
        XCTAssertTrue(a.loadPreparedWorkflowExample(device: "cpu")); XCTAssertTrue(b.loadPreparedWorkflowExample(device: "cpu"))
        let first = try XCTUnwrap(a.workflowSession)
        var document = try XCTUnwrap(b.workflowSession).owner.document
        document = try WorkflowCodec.migrateToVersion4(document)
        let setup = try XCTUnwrap(document.nodes.firstIndex { if case .train = $0.payload { return true }; return false })
        guard case .train(let trainConfig) = document.nodes[setup].payload else { return XCTFail("Missing Train") }
        var settings=try XCTUnwrap(WorkflowTrainRunSetup.configuration(trainConfig,schemaVersion:document.schemaVersion))
        settings["preparationCachePolicy"] = .string("materialize")
        settings["experiment"] = .object(["schemaVersion": 1, "axes": .array([.object(["field": .string("seed"), "values": .array([7, 8])])]), "budget": .object(["maximumRuns": 2, "maximumTotalEpochs": 6])])
        document.nodes[setup].payload = .train(config:try WorkflowTrainRunSetup.setting(settings,in:trainConfig,schemaVersion:document.schemaVersion))
        _ = try b.projectLibrary.saveSource(.workflow(document), id: document.projectID)
        XCTAssertTrue(b.beginWorkflowProject(id: document.projectID, forceReload: true), b.statusMessage)
        let second = try XCTUnwrap(b.workflowSession)
        let input = try XCTUnwrap(second.bindings.path(for: WorkflowExamples.sourceAddress))
        let copied = root.appendingPathComponent("mutable-original.csv")
        let original = try Data(contentsOf: URL(fileURLWithPath: input)); try original.write(to: copied)
        try b.sourceAccess.remember(copied)
        XCTAssertTrue(second.bindings.setPath(copied.path, for: WorkflowExamples.sourceAddress, lease: second.owner.lease(), expectedRevision: second.bindings.revision))
        let queue = first.executionQueue
        let gateCapture = try WorkflowQueueCapture(projectID: first.owner.projectID, title: "Test dispatch gate", kind: .training,
            requestIdentities: [String(repeating: "a", count: 64)], document: first.owner.document, sourceHashes: [])
        let gate = try queue.enqueue(capture: gateCapture, start: {}, cancel: {})
        try await wait { queue.entries.first?.state == .running }
        XCTAssertTrue(first.run(),first.statusMessage); XCTAssertTrue(second.runExperiments())
        try await wait { first.queuedRequestID != nil && second.queuedRequestID != nil }
        let firstID = try XCTUnwrap(first.queuedRequestID), secondID = try XCTUnwrap(second.queuedRequestID)
        XCTAssertEqual(queue.entries.first(where: {$0.id == firstID})?.state, .queued)
        XCTAssertEqual(queue.entries.first(where: {$0.id == secondID})?.state, .queued)
        let pinned = try XCTUnwrap(queue.entries.first(where: {$0.id == secondID})).capture
        try Data("Changed original after capture; this must not be read".utf8).write(to: copied)
        XCTAssertTrue(b.setExecutionPreferences(try .init(maximumRunSeconds: 1, preparationCacheLimitBytes: 1)))
        queue.finish(gate, state: .completed, message: "Test gate released; actual workers now dispatch")
        try await wait({ !a.isBusy && !b.isBusy }, seconds: 90)
        XCTAssertEqual(first.outcomes.count, 1); XCTAssertEqual(second.outcomes.count, 2)
        XCTAssertTrue((first.outcomes + second.outcomes).allSatisfy {$0.status == .completed})
        XCTAssertEqual(queue.entries.first(where: {$0.id == secondID})?.state, .completed)
        XCTAssertEqual(Set(second.outcomes.map { $0.request.requestIdentity }), Set(pinned.requestIdentities))
        XCTAssertTrue(second.outcomes.allSatisfy { $0.request.executionBudget?.maximumRunSeconds == pinned.budget?.maximumRunSeconds })
        XCTAssertTrue(second.outcomes.allSatisfy { $0.request.preparationCacheLimitBytes == 200_000_000 })
        let hash = WorkflowPhaseArtifacts.hash(original)
        XCTAssertTrue(second.outcomes.allSatisfy { $0.sources.values.contains(where: {$0.sha256 == hash}) })
        a.retireWorkflowSession(); b.retireWorkflowSession()
    }
}
