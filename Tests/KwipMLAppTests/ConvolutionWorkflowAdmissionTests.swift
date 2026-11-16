import XCTest
import KwipMLCore
@testable import KwipMLApp

// Historical class/filter identity retained; covers window operators and configurable activations.
@MainActor final class ConvolutionWorkflowAdmissionTests: XCTestCase {
    private static var compositionProof:ModuleCompositionDiscoveryEvidence?
    private let componentIDs = ["conv1d", "conv2d", "conv3d", "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
                                "max_pool1d", "max_pool2d", "max_pool3d", "avg_pool1d", "avg_pool2d", "adaptive_avg_pool1d", "adaptive_avg_pool2d", "adaptive_max_pool2d", "silu", "leaky_relu", "elu"]
    private final class Recorder: @unchecked Sendable {
        private let lock = NSLock()
        private var captures = 0, launches = 0
        func capture() { lock.lock(); captures += 1; lock.unlock() }
        func launch() { lock.lock(); launches += 1; lock.unlock() }
        var counts: (Int, Int) { lock.lock(); defer { lock.unlock() }; return (captures, launches) }
    }
    @MainActor private struct Fixture {
        let workspace: WorkspaceStore
        let host: WorkflowWorkspaceSession
        let recorder: Recorder
        let type: String
        var title: String { workspace.registry.components.first { $0.id == type }?.title ?? type }
        func revoke() {
            switch type {
            case "conv1d": workspace.conv1dRuntimeCapability = nil
            case "conv2d": workspace.conv2dRuntimeCapability = nil
            case "conv3d": workspace.conv3dRuntimeCapability = nil
            case "conv_transpose1d": workspace.convTranspose1dRuntimeCapability = nil
            case "conv_transpose2d": workspace.convTranspose2dRuntimeCapability = nil
            case "conv_transpose3d": workspace.convTranspose3dRuntimeCapability = nil
            case "silu", "leaky_relu", "elu": workspace.configurableActivationRuntimeCapabilities = .init()
            default: workspace.poolingRuntimeCapabilities = .init()
            }
        }
        var supported: Bool {
            switch type {
            case "conv1d": return workspace.conv1dRuntimeCapability != nil
            case "conv2d": return workspace.conv2dRuntimeCapability != nil
            case "conv3d": return workspace.conv3dRuntimeCapability != nil
            case "conv_transpose1d": return workspace.convTranspose1dRuntimeCapability != nil
            case "conv_transpose2d": return workspace.convTranspose2dRuntimeCapability != nil
            case "conv_transpose3d": return workspace.convTranspose3dRuntimeCapability != nil
            case "silu", "leaky_relu", "elu": return workspace.configurableActivationRuntimeCapabilities.contains(type)
            default: return workspace.poolingRuntimeCapabilities.contains(type)
            }
        }
        func start(_ experiment: Bool) -> Bool { experiment ? host.runExperiments() : host.run() }
        func close() { host.cancel(); workspace.retireWorkflowSession() }
    }
    private func fixture(_ id: String, verified: Bool, malformed: Bool = false,
                         selected: Bool = true, controlledCancellation: Bool = true) async throws -> Fixture {
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: repo, storageOverride: root, detectCapabilities: false)
        _ = try XCTUnwrap(workspace.runtime, workspace.runtimeError ?? "Fixture runtime unavailable")
        let prepared: WorkflowDocument
        if !selected { prepared = try WorkflowExamples.preparedParallelClassifier(device: "cpu") }
        else if BundledConvolutionComponents.metadata(id) != nil { prepared = try BundledConvolutionComponents.example(id, device: "cpu") }
        else if ConfigurableActivationRuntimeCapabilities.operations.contains(id) { prepared = try BundledActivationComponents.example(id, device: "cpu") }
        else { prepared = try BundledPoolingComponents.example(id, device: "cpu") }
        var document = try WorkflowCodec.migrateToVersion4(prepared)
        if !selected {
            document.nodes.append(.init(address: .init(domain: .model, localID: "disconnected-window"), name: "Unused window operator", x: 0, y: 0,
                payload: .modelPrimitive(type: id, configurationVersion: 2, config: [:])))
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
        var extensionContract = try XCTUnwrap(BundledConvolutionComponents.metadata(id)?.runtimeContract ?? BundledPoolingComponents.metadata(id)?.runtimeContract ?? BundledActivationComponents.entries[id]?.runtimeContract)
        if malformed, case .object(var object) = extensionContract {
            object["version"] = .object(["major": 2, "minor": 0, "patch": 0]); extensionContract = .object(object)
        }
        let optional = verified ? try JSONEncoder().encode(extensionContract) : nil
        let layout: [String: Any] = ["version": ["major": 1, "minor": 0, "patch": 0], "adapterID": "kwip.runner.graph-v2.tensor-layout", "graphSchemaVersions": [2], "operations": ["reshape", "transpose", "flatten"], "maximumRank": 8, "preservesBatch": true]
        try workspace.installDiscoveredRuntimeContract(version: "2.14.1", expectedVersion: nil, devices: ["cpu"], descriptor: JSONEncoder().encode(contract),
            tensorLayoutDescriptor: JSONSerialization.data(withJSONObject: layout),
            conv1dDescriptor: id == "conv1d" ? optional : nil,
            conv2dDescriptor: id == "conv2d" ? optional : nil,
            conv3dDescriptor: id == "conv3d" ? optional : nil,
            convTranspose1dDescriptor: id == "conv_transpose1d" ? optional : nil,
            convTranspose2dDescriptor: id == "conv_transpose2d" ? optional : nil,
            convTranspose3dDescriptor: id == "conv_transpose3d" ? optional : nil,
            poolingDescriptors: BundledPoolingComponents.metadata(id) != nil ? (optional.map { [id: $0] } ?? [:]) : [:],
            tensorCompositionDescriptors: fixtureReviewedCompositionDescriptors(), configurableActivationDescriptors: ConfigurableActivationRuntimeCapabilities.operations.contains(id) ? (optional.map { [id: $0] } ?? [:]) : [:])
        // Cache actual selected-worker proof, while retaining this fixture’s controlled optional descriptors.
        workspace.moduleCompositionDiscoveryEvidence=Self.compositionProof
        let recorder = Recorder()
        if controlledCancellation {
            host.captureRunSources = { _, _, _, _ in recorder.capture(); throw CancellationError() }
            host.captureExperimentInputs = { _, _ in recorder.capture(); throw CancellationError() }
        }
        host.executionLaunch = { _, _, _, _, _ in recorder.launch(); throw WorkflowArchiveError("Admission tests must not launch a worker") }
        return .init(workspace: workspace, host: host, recorder: recorder, type: id)
    }
    private func wait(_ condition: () -> Bool) async throws {
        let deadline = Date().addingTimeInterval(10)
        while !condition(), Date() < deadline { try await Task.sleep(for: .milliseconds(10)) }
        guard condition() else {throw WorkflowArchiveError("Timed out at admission boundary")}
    }
    private func captureCancelled(_ f: Fixture) async throws {
        do { try await wait { f.recorder.counts.0 == 1 && !f.workspace.isBusy } }
        catch {throw WorkflowArchiveError("Admission capture did not settle: captures=\(f.recorder.counts.0), busy=\(f.workspace.isBusy), status=\(f.host.statusMessage)")}
        XCTAssertEqual(f.recorder.counts.1, 0)
    }
    private func clean(_ f: Fixture, reserved: Int64? = nil) throws {
        XCTAssertNil(f.host.queuedRequestID)
        XCTAssertNil(f.workspace.cancelDataPreparation)
        XCTAssertFalse(f.workspace.isBusy)
        XCTAssertEqual(f.recorder.counts.1, 0)
        if let reserved {
            let accounting = try WorkflowStorageAdmission(workspace: f.workspace.workspaceDirectory, accountingRoot: f.workspace.applicationDirectory)
            XCTAssertEqual(try accounting.snapshot().reservedGrowthBytes, reserved)
        }
        let stage = f.workspace.workspaceDirectory.appendingPathComponent("workflow-source-staging")
        if FileManager.default.fileExists(atPath: stage.path) { XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: stage.path).isEmpty) }
    }
    private func queueBoundary(_ id: String, experiment: Bool, explicitCancel: Bool) async throws {
        let f = try await fixture(id, verified: true, controlledCancellation: false)
        defer { f.close() }
        let queue = f.host.executionQueue
        let capture = try WorkflowQueueCapture(projectID: f.host.owner.projectID, title: "Admission gate", kind: .training,
            requestIdentities: [String(repeating: "a", count: 64)], document: f.host.owner.document, sourceHashes: [])
        let gate = try queue.enqueue(capture: capture, start: {}, cancel: {})
        defer { queue.finish(gate, state: .completed, message: "Test cleanup") }
        try await wait { queue.entries.first(where: { $0.id == gate })?.state == .running }
        let accounting = try WorkflowStorageAdmission(workspace: f.workspace.workspaceDirectory, accountingRoot: f.workspace.applicationDirectory)
        let before = try accounting.snapshot().reservedGrowthBytes
        XCTAssertTrue(f.start(experiment), f.host.statusMessage)
        try await wait { f.host.queuedRequestID != nil }
        let request = try XCTUnwrap(f.host.queuedRequestID)
        let entry = try XCTUnwrap(queue.entries.first { $0.id == request })
        XCTAssertEqual(entry.state, .queued)
        XCTAssertFalse(entry.capture.sourceHashes.isEmpty)
        XCTAssertTrue(f.workspace.isBusy)
        XCTAssertEqual(f.recorder.counts.1, 0)
        if explicitCancel { f.host.cancel() }
        else { f.revoke(); queue.finish(gate, state: .completed, message: "Support revoked") }
        try await wait { queue.entries.first(where: { $0.id == request })?.state == (explicitCancel ? .cancelled : .failed) && !f.workspace.isBusy }
        if !explicitCancel {
            let result = try XCTUnwrap(queue.entries.first { $0.id == request })
            XCTAssertTrue(result.message.contains(f.title), result.message)
            XCTAssertTrue(result.message.contains("Refresh"), result.message)
        }
        try clean(f, reserved: before)
        if experiment {
            let runner = try XCTUnwrap(f.host.experimentRunner)
            XCTAssertTrue(runner.group.record.variants.allSatisfy { $0.status == .cancelled })
            XCTAssertFalse(runner.isRunning)
            // Retained immutable experiment inputs are historical evidence, not a temporary run-stage leak.
        }
    }
    func testMissingDescriptorsRefuseBeforeCaptureAndReservation() async throws {
        for id in componentIDs { for experiment in [false,true] {
            let f = try await fixture(id, verified: false); defer { f.close() }
            XCTAssertFalse(f.start(experiment), f.host.statusMessage)
            XCTAssertTrue(f.host.statusMessage.contains(f.title), f.host.statusMessage)
            XCTAssertEqual(f.recorder.counts.0, 0)
            XCTAssertFalse(f.host.isStagingExperiment)
            try clean(f)
        } }
    }
    func testMalformedDescriptorsRefuseBeforeCapture() async throws {
        for id in componentIDs { for experiment in [false,true] {
            let f = try await fixture(id, verified: true, malformed: true); defer { f.close() }
            XCTAssertFalse(f.supported)
            XCTAssertFalse(f.start(experiment), f.host.statusMessage)
            XCTAssertTrue(f.host.statusMessage.contains(f.title), f.host.statusMessage)
            XCTAssertEqual(f.recorder.counts.0, 0)
            try clean(f)
        } }
    }
    func testVerifiedDescriptorsAllowControlledCaptureCancellation() async throws {
        for id in componentIDs { for experiment in [false,true] {
            let f = try await fixture(id, verified: true); defer { f.close() }
            XCTAssertTrue(f.supported)
            XCTAssertTrue(f.start(experiment), f.host.statusMessage)
            try await captureCancelled(f)
            if ConfigurableActivationRuntimeCapabilities.operations.contains(id) {
                let verifiedCount = f.workspace.verifiedRuntimeComponentCount
                f.revoke()
                XCTAssertEqual(f.workspace.verifiedRuntimeComponentCount, verifiedCount - 1)
            }
        } }
    }
    func testQueuedSupportRevocationReleasesCapturedSourcesAndReservations() async throws {
        for id in componentIDs { for experiment in [false,true] { try await queueBoundary(id, experiment: experiment, explicitCancel: false) } }
    }
    func testExplicitQueuedCancellationReleasesCapturedSourcesAndReservations() async throws {
        for id in componentIDs { for experiment in [false,true] { try await queueBoundary(id, experiment: experiment, explicitCancel: true) } }
    }
    func testDisconnectedConvolutionsDoNotBlockSelectedGraph() async throws {
        for id in componentIDs { for experiment in [false,true] {
            let f = try await fixture(id, verified: false, selected: false); defer { f.close() }
            let compiled = try WorkflowCompiler.compile(f.host.owner.document, train: WorkflowExamples.trainAddress,
                modelRegistry: f.host.owner.modelRegistry, dataRegistry: f.host.owner.dataRegistry,
                contractVersion: max(2, WorkflowArchiveCodec.trainingCompilerContractVersion(forWorkflowSchema: f.host.owner.document.schemaVersion)))
            XCTAssertFalse(compiled.model.graph.nodes.contains { $0.type == f.type })
            XCTAssertTrue(f.start(experiment), f.host.statusMessage)
            try await captureCancelled(f)
        } }
    }
}
