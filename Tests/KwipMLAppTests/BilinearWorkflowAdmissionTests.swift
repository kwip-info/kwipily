import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class BilinearWorkflowAdmissionTests: XCTestCase {
    private final class Recorder: @unchecked Sendable {
        private let lock = NSLock()
        private var captures = 0
        private var launches = 0
        func capture() { lock.lock(); captures += 1; lock.unlock() }
        func launch() { lock.lock(); launches += 1; lock.unlock() }
        var counts: (Int, Int) { lock.lock(); defer { lock.unlock() }; return (captures, launches) }
    }
    private struct Fixture {
        let root: URL
        let workspace: WorkspaceStore
        let host: WorkflowWorkspaceSession
        let recorder: Recorder
    }
    private static var compositionProof:ModuleCompositionDiscoveryEvidence?
    private func fixture(bilinearModel: Bool, verified: Bool, malformed: Bool = false, controlledCancellation: Bool = true) async throws -> Fixture {
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: repo, storageOverride: root, detectCapabilities: false)
        _ = try XCTUnwrap(workspace.runtime, workspace.runtimeError ?? "Fixture runtime unavailable")
        var document = try WorkflowCodec.migrateToVersion4(bilinearModel
            ? WorkflowExamples.preparedBilinearClassifier(device: "cpu")
            : WorkflowExamples.preparedParallelClassifier(device: "cpu"))
        if !bilinearModel {
            document.nodes.append(.init(address: .init(domain: .model, localID: "disconnected-bilinear"), name: "Unused bilinear", x: 0, y: 0,
                payload: .modelPrimitive(type: "bilinear", configurationVersion: 2, config: ["in1_features": 3, "in2_features": 5, "out_features": 2, "bias": true])))
        }
        let setup = try XCTUnwrap(document.nodes.firstIndex { if case .runSetup = $0.payload { return true }; return false })
        guard case .runSetup(var settings) = document.nodes[setup].payload else { throw WorkflowArchiveError("Missing setup") }
        settings["experiment"] = .object(["schemaVersion": 1, "axes": .array([.object(["field": .string("seed"), "values": .array([7, 8])])]), "budget": .object(["maximumRuns": 2, "maximumTotalEpochs": 6])])
        document.nodes[setup].payload = .runSetup(config: settings)
        _ = try workspace.projectLibrary.saveSource(.workflow(document), id: document.projectID)
        XCTAssertTrue(workspace.beginWorkflowProject(id: document.projectID), workspace.statusMessage)
        let host = try XCTUnwrap(workspace.workflowSession)
        let csv = root.appendingPathComponent("observations.csv")
        try FileManager.default.copyItem(at: repo.appendingPathComponent("Sources/KwipMLApp/Resources/workflow-observations.csv"), to: csv)
        try workspace.sourceAccess.remember(csv)
        XCTAssertTrue(host.bindings.setPath(csv.path, for: WorkflowExamples.sourceAddress, lease: host.owner.lease(), expectedRevision: host.bindings.revision))
        if Self.compositionProof == nil {
            await verifyRuntimeForIntegration(workspace)
            Self.compositionProof=try XCTUnwrap(workspace.moduleCompositionDiscoveryEvidence)
        }
        let contract = try BuiltInCapabilityBridge.executionContract(registry: workspace.registry, runtimeVersion: "2.14.1", availableDevices: [.cpu])
        let bilinear: [String: Any] = ["version": ["major": malformed ? 2 : 1, "minor": 0, "patch": 0],
            "adapterID": "kwip.runner.graph-v2.bilinear", "graphSchemaVersions": [2], "maximumRank": 8,
            "preservesBatch": true, "minimumSampleRank": 1, "sampleBroadcasting": false,
            "maximumSampleElements": 100_000_000, "maximumParameters": 2_000_000,
            "featureMinimum": 1, "featureMaximum": 512, "biasModes": [false, true], "graphDTypes": ["float32"]]
        try workspace.installDiscoveredRuntimeContract(version: "2.14.1", expectedVersion: nil, devices: ["cpu"], descriptor: JSONEncoder().encode(contract),
            bilinearDescriptor: verified ? JSONSerialization.data(withJSONObject: bilinear) : nil, tensorCompositionDescriptors: fixtureReviewedCompositionDescriptors())
        workspace.moduleCompositionDiscoveryEvidence=Self.compositionProof
        let recorder = Recorder()
        if controlledCancellation {
            host.captureRunSources = { _, _, _, _ in recorder.capture(); throw CancellationError() }
            host.captureExperimentInputs = { _, _ in recorder.capture(); throw CancellationError() }
        }
        host.executionLaunch = { _, _, _, _, _ in recorder.launch(); throw WorkflowArchiveError("No worker should launch in admission tests") }
        return .init(root: root, workspace: workspace, host: host, recorder: recorder)
    }
    private func waitForCaptureToFinish(_ fixture: Fixture) async throws {
        let deadline = Date().addingTimeInterval(5)
        while (fixture.recorder.counts.0 == 0 || fixture.workspace.isBusy), Date() < deadline {
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTAssertEqual(fixture.recorder.counts.0, 1)
        XCTAssertEqual(fixture.recorder.counts.1, 0)
        XCTAssertFalse(fixture.workspace.isBusy)
    }
    func testMissingDescriptorRefusesTrainAndExperimentsBeforeCaptureOrReservation() async throws {
        for experiments in [false, true] {
            let f = try await fixture(bilinearModel: true, verified: false)
            defer { f.host.cancel(); f.workspace.retireWorkflowSession() }
            XCTAssertFalse(experiments ? f.host.runExperiments() : f.host.run(), f.host.statusMessage)
            XCTAssertTrue(f.host.statusMessage.contains("Bilinear"), f.host.statusMessage)
            XCTAssertEqual(f.recorder.counts.0, 0); XCTAssertEqual(f.recorder.counts.1, 0)
            XCTAssertFalse(f.workspace.isBusy); XCTAssertFalse(f.host.isStagingExperiment)
            XCTAssertNil(f.host.queuedRequestID)
            XCTAssertFalse(FileManager.default.fileExists(atPath: f.workspace.workspaceDirectory.appendingPathComponent("workflow-source-staging").path))
        }
    }
    func testMismatchedDescriptorCannotAdmitAnySelectedBilinear() async throws {
        for experiments in [false, true] {
            let f = try await fixture(bilinearModel: true, verified: true, malformed: true)
            defer { f.host.cancel(); f.workspace.retireWorkflowSession() }
            XCTAssertNil(f.workspace.bilinearRuntimeCapability)
            XCTAssertFalse(experiments ? f.host.runExperiments() : f.host.run(), f.host.statusMessage)
            XCTAssertTrue(f.host.statusMessage.contains("Bilinear"), f.host.statusMessage)
            XCTAssertEqual(f.recorder.counts.0, 0); XCTAssertEqual(f.recorder.counts.1, 0)
            XCTAssertFalse(f.workspace.isBusy)
        }
    }
    func testVerifiedDescriptorAllowsTrainAndExperimentCaptureWithControlledCancellation() async throws {
        for experiments in [false, true] {
            let f = try await fixture(bilinearModel: true, verified: true)
            defer { f.host.cancel(); f.workspace.retireWorkflowSession() }
            XCTAssertTrue(experiments ? f.host.runExperiments() : f.host.run(), f.host.statusMessage)
            try await waitForCaptureToFinish(f)
        }
    }
    func testQueuedTrainAndExperimentsRecheckLostSupportBeforeDispatchAndReleaseCapture() async throws {
        func wait(_ condition: () -> Bool) async throws {
            let deadline = Date().addingTimeInterval(10)
            while !condition(), Date() < deadline { try await Task.sleep(for: .milliseconds(10)) }
            XCTAssertTrue(condition(), "Timed out at queued admission boundary")
        }
        for experiments in [false, true] {
            let f = try await fixture(bilinearModel: true, verified: true, controlledCancellation: false)
            defer { f.host.cancel(); f.workspace.retireWorkflowSession() }
            let queue = f.host.executionQueue
            let capture = try WorkflowQueueCapture(projectID: f.host.owner.projectID, title: "Admission test gate", kind: .training,
                requestIdentities: [String(repeating: "a", count: 64)], document: f.host.owner.document, sourceHashes: [])
            let gate = try queue.enqueue(capture: capture, start: {}, cancel: {})
            defer { queue.finish(gate, state: .completed, message: "Test cleanup") }
            try await wait { queue.entries.first(where: { $0.id == gate })?.state == .running }
            let accounting = try WorkflowStorageAdmission(workspace: f.workspace.workspaceDirectory, accountingRoot: f.workspace.applicationDirectory)
            let before = try accounting.snapshot().reservedGrowthBytes
            XCTAssertTrue(experiments ? f.host.runExperiments() : f.host.run(), f.host.statusMessage)
            try await wait { f.host.queuedRequestID != nil }
            let queuedID = try XCTUnwrap(f.host.queuedRequestID)
            let entry = try XCTUnwrap(queue.entries.first { $0.id == queuedID })
            XCTAssertEqual(entry.state, .queued)
            XCTAssertFalse(entry.capture.sourceHashes.isEmpty, "Real source bytes must have been captured before queue admission")
            XCTAssertTrue(f.workspace.isBusy)
            XCTAssertEqual(f.recorder.counts.1, 0)
            f.workspace.bilinearRuntimeCapability = nil
            queue.finish(gate, state: .completed, message: "Release after optional support was revoked")
            try await wait { queue.entries.first(where: { $0.id == queuedID })?.state == .failed && !f.workspace.isBusy }
            let failed = try XCTUnwrap(queue.entries.first { $0.id == queuedID })
            XCTAssertTrue(failed.message.contains("Bilinear"), failed.message)
            XCTAssertTrue(failed.message.contains("Refresh"), failed.message)
            XCTAssertEqual(f.recorder.counts.1, 0)
            XCTAssertNil(f.host.queuedRequestID)
            XCTAssertNil(f.workspace.cancelDataPreparation)
            XCTAssertEqual(try accounting.snapshot().reservedGrowthBytes, before)
            let stage = f.workspace.workspaceDirectory.appendingPathComponent("workflow-source-staging")
            if FileManager.default.fileExists(atPath: stage.path) {
                XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: stage.path).isEmpty,
                              "Temporary captured run sources must be released")
            }
            if experiments {
                let runner = try XCTUnwrap(f.host.experimentRunner)
                XCTAssertTrue(runner.group.record.variants.allSatisfy { $0.status == .cancelled })
                XCTAssertFalse(runner.isRunning)
                // Immutable experiment inputs remain with the cancelled group as
                // historical evidence; they are not a leaked temporary run stage.
            }
        }
    }
    func testExplicitQueuedCancellationReleasesCapturedSourcesAndReservations() async throws {
        func wait(_ condition: () -> Bool) async throws {
            let deadline = Date().addingTimeInterval(10)
            while !condition(), Date() < deadline { try await Task.sleep(for: .milliseconds(10)) }
            XCTAssertTrue(condition(), "Timed out at queued admission boundary")
        }
        for experiments in [false, true] {
            let f = try await fixture(bilinearModel: true, verified: true, controlledCancellation: false)
            defer { f.host.cancel(); f.workspace.retireWorkflowSession() }
            let queue = f.host.executionQueue
            let capture = try WorkflowQueueCapture(projectID: f.host.owner.projectID, title: "Admission test gate", kind: .training,
                requestIdentities: [String(repeating: "a", count: 64)], document: f.host.owner.document, sourceHashes: [])
            let gate = try queue.enqueue(capture: capture, start: {}, cancel: {})
            defer { queue.finish(gate, state: .completed, message: "Test cleanup") }
            try await wait { queue.entries.first(where: { $0.id == gate })?.state == .running }
            let accounting = try WorkflowStorageAdmission(workspace: f.workspace.workspaceDirectory, accountingRoot: f.workspace.applicationDirectory)
            let before = try accounting.snapshot().reservedGrowthBytes
            XCTAssertTrue(experiments ? f.host.runExperiments() : f.host.run(), f.host.statusMessage)
            try await wait { f.host.queuedRequestID != nil }
            let queuedID = try XCTUnwrap(f.host.queuedRequestID)
            let entry = try XCTUnwrap(queue.entries.first { $0.id == queuedID })
            XCTAssertEqual(entry.state, .queued)
            XCTAssertFalse(entry.capture.sourceHashes.isEmpty, "Real source bytes must have been captured before queue admission")
            XCTAssertTrue(f.workspace.isBusy)
            XCTAssertEqual(f.recorder.counts.1, 0)
            f.host.cancel()
            try await wait { queue.entries.first(where: { $0.id == queuedID })?.state == .cancelled && !f.workspace.isBusy }
            let failed = try XCTUnwrap(queue.entries.first { $0.id == queuedID })
            XCTAssertEqual(failed.state, .cancelled)
            XCTAssertEqual(f.recorder.counts.1, 0)
            XCTAssertNil(f.host.queuedRequestID)
            XCTAssertNil(f.workspace.cancelDataPreparation)
            XCTAssertEqual(try accounting.snapshot().reservedGrowthBytes, before)
            let stage = f.workspace.workspaceDirectory.appendingPathComponent("workflow-source-staging")
            if FileManager.default.fileExists(atPath: stage.path) {
                XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: stage.path).isEmpty,
                              "Temporary captured run sources must be released")
            }
            if experiments {
                let runner = try XCTUnwrap(f.host.experimentRunner)
                XCTAssertTrue(runner.group.record.variants.allSatisfy { $0.status == .cancelled })
                XCTAssertFalse(runner.isRunning)
                // Immutable experiment inputs remain with the cancelled group as
                // historical evidence; they are not a leaked temporary run stage.
            }
        }
    }
    func testDisconnectedBilinearDoesNotBlockTheSelectedTrainGraph() async throws {
        for experiments in [false, true] {
            let f = try await fixture(bilinearModel: false, verified: false)
            defer { f.host.cancel(); f.workspace.retireWorkflowSession() }
            let compiled = try WorkflowCompiler.compile(f.host.owner.document, train: WorkflowExamples.trainAddress, modelRegistry: f.host.owner.modelRegistry, dataRegistry: f.host.owner.dataRegistry, contractVersion: max(2, WorkflowArchiveCodec.trainingCompilerContractVersion(forWorkflowSchema: f.host.owner.document.schemaVersion)))
            XCTAssertFalse(compiled.model.graph.nodes.contains { $0.type == "bilinear" })
            XCTAssertTrue(experiments ? f.host.runExperiments() : f.host.run(), f.host.statusMessage)
            try await waitForCaptureToFinish(f)
        }
    }
}
