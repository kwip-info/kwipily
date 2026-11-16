import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class RemainingActivationRuntimeAdmissionTests: XCTestCase {
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
    /// Literal worker descriptors independent of the Core bridge/decoder under test.
    private func descriptor(_ operation:String)throws->Data {
        var d:[String:Any] = ["version":["major":1,"minor":2,"patch":0],"adapterID":"kwip.runner.graph-v2.\(operation)","graphSchemaVersions":[2],"maximumRank":8,"preservesBatch":true,"graphDTypes":["float32"],"maximumTensorElements":100000000]
        switch operation {
        case "prelu":d.merge(["parameterCountMinimum":1,"parameterCountMaximum":4096,"initMinimum":-3.4028234663852886e38,"initMaximum":3.4028234663852886e38,"channelAxis":1,"learnedParameters":true,"zeroInputGradient":"slope"]) {$1}
        case "log_softmax":d.merge(["axisMinimum":-8,"axisMaximum":7,"excludesBatch":true,"outputKind":"logarithmicProbabilities"]) {$1}
        case "dropout":d.merge(["probabilityMinimum":0,"probabilityMaximum":1,"modeSource":"parentModule","stochasticTraining":true,"evaluationIdentity":true,"checkpointGuarantees":["cpu":"epochBoundaryExactCPU","mps":"weightsOnly"]]) {$1}
        default:throw WorkflowArchiveError("Unknown test operator")
        }
        return try JSONSerialization.data(withJSONObject:d)
    }
    private func fixture(activationModel: Bool, operation: String, verified: Bool, malformed: Bool = false, controlledCancellation: Bool = true) async throws -> Fixture {
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: repo, storageOverride: root, detectCapabilities: false)
        _ = try XCTUnwrap(workspace.runtime, workspace.runtimeError ?? "Fixture runtime unavailable")
        var document: WorkflowDocument
        let entry = try XCTUnwrap(BundledActivationComponents.entries[operation])
        if activationModel {
            switch operation {
            case "prelu":document = try WorkflowCodec.migrateToVersion4(WorkflowExamples.preparedPReLUClassifier(device:"cpu"))
            case "log_softmax":document = try WorkflowCodec.migrateToVersion4(WorkflowExamples.preparedLogSoftmaxClassifier(device:"cpu"))
            default:document = try WorkflowCodec.migrateToVersion4(WorkflowExamples.preparedDropoutClassifier(device:"cpu"))
            }
        } else {
            document = try WorkflowCodec.migrateToVersion4(WorkflowExamples.preparedParallelClassifier(device:"cpu"))
            // The original parallel example contains a connected Concatenate; build a
            // simple selected model so the unrelated activation block is truly disconnected.
            document.nodes.removeAll { $0.address.domain == .model }
            document.edges.removeAll { $0.from.address.domain == .model || $0.to.address.domain == .model }
            document.definitions = []
            func address(_ id:String)->WorkflowNodeAddress { .init(domain:.model,localID:id) }
            for (id,type,config) in [("input","input",["features":JSONValue.integer(2)]),("head","linear",["in_features":2,"out_features":2,"bias":true]),("output","output",[:])] {
                document.nodes.append(.init(address:address(id),name:type,x:0,y:0,payload:.modelPrimitive(type:type,configurationVersion:2,config:config)))
            }
            for (source,target) in [("input","head"),("head","output")] {
                document.edges.append(.init(from:.init(address:address(source),port:"out"),to:.init(address:address(target),port:"in"),kind:.tensor))
            }
            document.edges.append(.init(from:.init(address:address("output"),port:"definition"),to:.init(address:.init(domain:.control,localID:"model"),port:"source"),kind:.modelDefinition))
            let config = Dictionary(uniqueKeysWithValues:entry.definition.fields.map { ($0.key,$0.defaultValue) })
            document.nodes.append(.init(address:.init(domain:.model,localID:"disconnected-activation"),name:"Unused activation",x:0,y:0,payload:.modelPrimitive(type:operation,configurationVersion:2,config:config)))
        }
        let specification=WorkflowExperimentSpecification(axes:[.init(field:"seed",values:[7,8])],budget:.init(maximumRuns:2,maximumTotalEpochs:6))
        _ = try workspace.projectLibrary.saveSource(.workflow(document), id: document.projectID)
        XCTAssertTrue(workspace.beginWorkflowProject(id: document.projectID), workspace.statusMessage)
        let host = try XCTUnwrap(workspace.workflowSession)
        // Exercise the actual current Train-owned authoring API after project
        // migration, rather than manually constructing a legacy configuration edge.
        XCTAssertTrue(host.owner.setExperimentSpecification(specification,for:WorkflowExamples.trainAddress,lease:host.owner.lease()),host.owner.error ?? "Could not save experiment")
        let authority=try WorkflowTrainRunSetup.resolve(in:host.owner.document,train:WorkflowExamples.trainAddress)
        XCTAssertEqual(authority.address,WorkflowExamples.trainAddress)
        XCTAssertEqual(try WorkflowExperimentSpecification.decode(XCTUnwrap(authority.configuration["experiment"])),specification)
        let csv = root.appendingPathComponent("observations.csv")
        try FileManager.default.copyItem(at: repo.appendingPathComponent("Sources/KwipMLApp/Resources/workflow-observations.csv"), to: csv)
        try workspace.sourceAccess.remember(csv)
        XCTAssertTrue(host.bindings.setPath(csv.path, for: WorkflowExamples.sourceAddress, lease: host.owner.lease(), expectedRevision: host.bindings.revision))
        let contract = try BuiltInCapabilityBridge.executionContract(registry: workspace.registry, runtimeVersion: "2.14.1", availableDevices: [.cpu])
        var activation = try JSONSerialization.jsonObject(with:descriptor(operation)) as! [String:Any]
        if malformed { activation["maximumRank"] = 9 }
        try fixtureInstallDiscoveredRuntimeContract(workspace,version:"2.14.1",expectedVersion:nil,devices:["cpu"],descriptor:JSONEncoder().encode(contract),
            tensorLayoutDescriptor:JSONSerialization.data(withJSONObject:["version":["major":1,"minor":0,"patch":0],"adapterID":"kwip.runner.graph-v2.tensor-layout","graphSchemaVersions":[2],"operations":["reshape","transpose","flatten"],"maximumRank":8,"preservesBatch":true]),
            remainingActivationDescriptors:verified ? [operation:JSONSerialization.data(withJSONObject:activation)] : [:], moduleCompositionDescriptor:fixtureReviewedModuleCompositionDescriptor())
        // Project opening upgrades Train to current inline settings. Admission
        // therefore needs genuine source-bound composition proof from this runtime,
        // independently of the optional activation descriptors controlled here.
        let selectedActivations=workspace.remainingActivationRuntimeCapabilities
        await verifyRuntimeForIntegration(workspace)
        _ = try XCTUnwrap(workspace.moduleCompositionDiscoveryEvidence,workspace.capabilityStatus)
        workspace.remainingActivationRuntimeCapabilities=selectedActivations
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
        for (experiments, operation) in [false, true].flatMap({ experiments in ["prelu", "log_softmax", "dropout"].map { (experiments, $0) } }) {
            let f = try await fixture(activationModel: true, operation: operation, verified: false)
            defer { f.host.cancel(); f.workspace.retireWorkflowSession() }
            XCTAssertFalse(experiments ? f.host.runExperiments() : f.host.run(), f.host.statusMessage)
            XCTAssertTrue(f.host.statusMessage.contains(BundledActivationComponents.entries[operation]!.definition.title), f.host.statusMessage)
            XCTAssertEqual(f.recorder.counts.0, 0); XCTAssertEqual(f.recorder.counts.1, 0)
            XCTAssertFalse(f.workspace.isBusy); XCTAssertFalse(f.host.isStagingExperiment)
            XCTAssertNil(f.host.queuedRequestID)
            XCTAssertFalse(FileManager.default.fileExists(atPath: f.workspace.workspaceDirectory.appendingPathComponent("workflow-source-staging").path))
        }
    }
    func testMismatchedDescriptorCannotAdmitAnySelectedActivationOperation() async throws {
        for operation in ["prelu", "log_softmax", "dropout"] {
            let f = try await fixture(activationModel: true, operation: operation, verified: true, malformed: true)
            defer { f.host.cancel(); f.workspace.retireWorkflowSession() }
            XCTAssertFalse(f.workspace.remainingActivationRuntimeCapabilities.contains(operation))
            XCTAssertFalse(f.host.run(), f.host.statusMessage)
            XCTAssertTrue(f.host.statusMessage.contains(BundledActivationComponents.entries[operation]!.definition.title), f.host.statusMessage)
            XCTAssertEqual(f.recorder.counts.0, 0); XCTAssertEqual(f.recorder.counts.1, 0)
            XCTAssertFalse(f.workspace.isBusy)
        }
    }
    func testVerifiedDescriptorAllowsTrainAndExperimentCaptureWithControlledCancellation() async throws {
        for (experiments, operation) in [false, true].flatMap({ experiments in ["prelu", "log_softmax", "dropout"].map { (experiments, $0) } }) {
            let f = try await fixture(activationModel: true, operation: operation, verified: true)
            defer { f.host.cancel(); f.workspace.retireWorkflowSession() }
            let admitted=experiments ? f.host.runExperiments() : f.host.run()
            XCTAssertTrue(admitted,f.host.statusMessage)
            guard admitted else {continue} // Report the admission failure without a misleading timeout.
            try await waitForCaptureToFinish(f)
        }
    }
    func testQueuedTrainAndExperimentsRecheckLostSupportBeforeDispatchAndReleaseCapture() async throws {
        func wait(_ condition: () -> Bool) async throws {
            let deadline = Date().addingTimeInterval(10)
            while !condition(), Date() < deadline { try await Task.sleep(for: .milliseconds(10)) }
            XCTAssertTrue(condition(), "Timed out at queued admission boundary")
        }
        for (experiments, operation) in [false, true].flatMap({ experiments in ["prelu", "log_softmax", "dropout"].map { (experiments, $0) } }) {
            let f = try await fixture(activationModel: true, operation: operation, verified: true, controlledCancellation: false)
            defer { f.host.cancel(); f.workspace.retireWorkflowSession() }
            let queue = f.host.executionQueue
            let capture = try WorkflowQueueCapture(projectID: f.host.owner.projectID, title: "Admission test gate", kind: .training,
                requestIdentities: [String(repeating: "a", count: 64)], document: f.host.owner.document, sourceHashes: [])
            let gate = try queue.enqueue(capture: capture, start: {}, cancel: {})
            defer { queue.finish(gate, state: .completed, message: "Test cleanup") }
            try await wait { queue.entries.first(where: { $0.id == gate })?.state == .running }
            let accounting = try WorkflowStorageAdmission(workspace: f.workspace.workspaceDirectory, accountingRoot: f.workspace.applicationDirectory)
            let before = try accounting.snapshot().reservedGrowthBytes
            let admitted=experiments ? f.host.runExperiments() : f.host.run()
            XCTAssertTrue(admitted,f.host.statusMessage)
            guard admitted else {continue} // Report the admission failure without a misleading timeout.
            try await wait { f.host.queuedRequestID != nil || !f.workspace.isBusy }
            let queuedID = try XCTUnwrap(f.host.queuedRequestID,"\(operation), experiments=\(experiments): capture did not reach the blocked queue. \(f.host.statusMessage)")
            let entry = try XCTUnwrap(queue.entries.first { $0.id == queuedID })
            XCTAssertEqual(entry.state, .queued)
            XCTAssertFalse(entry.capture.sourceHashes.isEmpty, "Real source bytes must have been captured before queue admission")
            XCTAssertTrue(f.workspace.isBusy)
            XCTAssertEqual(f.recorder.counts.1, 0)
            f.workspace.remainingActivationRuntimeCapabilities = .init()
            queue.finish(gate, state: .completed, message: "Release after optional support was revoked")
            try await wait { queue.entries.first(where: { $0.id == queuedID })?.state == .failed && !f.workspace.isBusy }
            let failed = try XCTUnwrap(queue.entries.first { $0.id == queuedID })
            XCTAssertTrue(failed.message.contains(BundledActivationComponents.entries[operation]!.definition.title), failed.message)
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
    func testDisconnectedCompositionDoesNotBlockTheSelectedTrainGraph() async throws {
        for (experiments, operation) in [false, true].flatMap({ experiments in ["prelu", "log_softmax", "dropout"].map { (experiments, $0) } }) {
            let f = try await fixture(activationModel: false, operation: operation, verified: false)
            defer { f.host.cancel(); f.workspace.retireWorkflowSession() }
            let compiled = try WorkflowCompiler.compile(f.host.owner.document, train: WorkflowExamples.trainAddress, modelRegistry: f.host.owner.modelRegistry, dataRegistry: f.host.owner.dataRegistry, contractVersion: max(2, WorkflowArchiveCodec.trainingCompilerContractVersion(forWorkflowSchema: f.host.owner.document.schemaVersion)))
            XCTAssertFalse(compiled.model.graph.nodes.contains { $0.type == operation })
            let admitted=experiments ? f.host.runExperiments() : f.host.run()
            XCTAssertTrue(admitted,f.host.statusMessage)
            guard admitted else {continue} // Report the admission failure without a misleading timeout.
            try await waitForCaptureToFinish(f)
        }
    }
    func testPartialDiscoveryCountAndEveryRefreshRevokeMissingIDsWithoutImplicitLegacySupport() throws {
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:storage)}
        let store=WorkspaceStore(rootOverride:repo,storageOverride:storage,detectCapabilities:false)
        let contract=try JSONEncoder().encode(BuiltInCapabilityBridge.executionContract(registry:store.registry,runtimeVersion:"2.14.1",availableDevices:[.cpu]))
        func install(_ ids:[String])throws {
            let descriptors=try Dictionary(uniqueKeysWithValues:ids.map { ($0,try descriptor($0)) })
            try store.installDiscoveredRuntimeContract(version:"2.14.1",expectedVersion:nil,devices:["cpu"],descriptor:contract,remainingActivationDescriptors:descriptors)
        }
        try install([])
        let baseCount=store.verifiedRuntimeComponentCount
        for id in ["prelu","log_softmax","dropout"] {
            try install([id])
            XCTAssertEqual(store.remainingActivationRuntimeCapabilities.componentIDs,[id])
            XCTAssertEqual(store.verifiedRuntimeComponentCount,baseCount+1,id)
        }
        try install(["prelu","log_softmax","dropout"])
        XCTAssertEqual(store.verifiedRuntimeComponentCount,baseCount+3)
        // Existing helper callers omit optional activation descriptors; that must revoke all three.
        try store.installDiscoveredRuntimeContract(version:"2.14.1",expectedVersion:nil,devices:["cpu"],descriptor:contract)
        XCTAssertTrue(store.remainingActivationRuntimeCapabilities.componentIDs.isEmpty)
        XCTAssertEqual(store.verifiedRuntimeComponentCount,baseCount)
        try install(["prelu","log_softmax","dropout"])
        XCTAssertThrowsError(try store.installDiscoveredRuntimeContract(version:"2.14.1",expectedVersion:nil,devices:["cpu"],descriptor:Data("{}".utf8)))
        XCTAssertTrue(store.remainingActivationRuntimeCapabilities.componentIDs.isEmpty)
        XCTAssertEqual(store.verifiedRuntimeComponentCount,0)
        try install(["prelu","log_softmax","dropout"])
        XCTAssertThrowsError(try store.installRuntimeContract(version:"bad",expectedVersion:nil))
        XCTAssertTrue(store.remainingActivationRuntimeCapabilities.componentIDs.isEmpty)
        XCTAssertEqual(store.verifiedRuntimeComponentCount,0)
    }
    func testActualWorkerDiscoversAllThreeExactReviewedCapabilities() async throws {
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:storage)}
        let store=WorkspaceStore(rootOverride:repo,storageOverride:storage,detectCapabilities:false)
        _ = try XCTUnwrap(store.runtime,store.runtimeError ?? "Actual worker required")
        let ready=await withCheckedContinuation { continuation in store.refreshCapabilities { continuation.resume(returning:$0) } }
        XCTAssertTrue(ready,store.capabilityStatus)
        XCTAssertEqual(store.remainingActivationRuntimeCapabilities.componentIDs,["prelu","log_softmax","dropout"])
        for id in ["prelu","log_softmax","dropout"] {
            XCTAssertEqual(store.registry.component(type:id),BundledActivationComponents.entries[id]!.definition)
        }
    }

}
