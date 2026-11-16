import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class TensorRoutingRuntimeAdmissionTests: XCTestCase {
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
    /// Independently authored worker contracts, never encoded from the decoder under test.
    private func descriptor(_ operation:String)throws->Data {
        var d:[String:Any] = ["version":["major":1,"minor":0,"patch":0],"adapterID":"kwip.runner.graph-v2.\(operation)","graphSchemaVersions":[2],"maximumRank":8,"preservesBatch":true,"graphDTypes":["float32"],"maximumTensorElements":100000000,"parameterless":true]
        if ["split","chunk","gather"].contains(operation) { d.merge(["axisMinimum":-8,"axisMaximum":7,"excludesBatch":true]) {$1} }
        if ["split","chunk"].contains(operation) { d.merge(["maximumOutputs":8,"outputPorts":(0..<8).map {"part\($0)"},"views":true]) {$1} }
        switch operation {
        case "split":d.merge(["splitSizeMinimum":1,"splitSizeMaximum":100000000,"sectionCountMinimum":1,"sectionCountMaximum":8]) {$1}
        case "chunk":d.merge(["chunksMinimum":1,"chunksMaximum":8,"mayReturnFewer":true]) {$1}
        case "gather":d.merge(["indicesDType":"int64","broadcast":false,"negativeIndices":false,"sparseGradient":false]) {$1}
        case "einsum":d.merge(["maximumOperands":2,"equationMinimumCharacters":1,"equationMaximumCharacters":128,"asciiOnly":true,"explicitOutput":true,"reservedBatchLabel":"n","maximumContractionTerms":100000000]) {$1}
        default:throw WorkflowArchiveError("Unknown routing test operator")
        }
        return try JSONSerialization.data(withJSONObject:d)
    }
    private func fixture(routingModel: Bool, operation: String, verified: Bool, malformed: Bool = false, controlledCancellation: Bool = true) throws -> Fixture {
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: repo, storageOverride: root, detectCapabilities: false)
        _ = try XCTUnwrap(workspace.runtime, workspace.runtimeError ?? "Fixture runtime unavailable")
        var document: WorkflowDocument
        let entry = try XCTUnwrap(BundledTensorRoutingComponents.entries[operation])
        if routingModel {
            switch operation {
            case "split":document = try WorkflowCodec.migrateToVersion4(WorkflowExamples.preparedSplitClassifier(device:"cpu"))
            case "chunk":document = try WorkflowCodec.migrateToVersion4(WorkflowExamples.preparedChunkClassifier(device:"cpu"))
            case "gather":document = try WorkflowCodec.migrateToVersion4(WorkflowExamples.preparedGatherClassifier(device:"cpu"))
            default:document = try WorkflowCodec.migrateToVersion4(WorkflowExamples.preparedEinsumClassifier(device:"cpu"))
            }
        } else {
            document = try WorkflowCodec.migrateToVersion4(WorkflowExamples.preparedParallelClassifier(device:"cpu"))
            // The original parallel example contains a connected Concatenate; build a
            // simple selected model so the unrelated routing block is truly disconnected.
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
            document.nodes.append(.init(address:.init(domain:.model,localID:"disconnected-routing"),name:"Unused routing",x:0,y:0,payload:.modelPrimitive(type:operation,configurationVersion:2,config:config)))
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
        let contract = try BuiltInCapabilityBridge.executionContract(registry: workspace.registry, runtimeVersion: "2.14.1", availableDevices: [.cpu])
        // The selected Split graph also uses Concatenate; Gather uses typed MaxPool indices.
        // Keep their already-reviewed capabilities independent of routing admission.
        let composition = ["concat":try JSONEncoder().encode(BundledTensorCompositionComponents.entries["concat"]!.runtimeContract)]
        let pooling = ["max_pool1d":try JSONSerialization.data(withJSONObject:[
            "version":["major":1,"minor":0,"patch":0],"adapterID":"kwip.runner.graph-v2.max_pool1d","graphSchemaVersions":[2],"sampleRank":2,"preservesBatch":true,"graphDTypes":["float32"],
            "channelMinimum":1,"channelMaximum":512,"lengthMinimum":1,"lengthMaximum":4096,"kernelMinimum":1,"kernelMaximum":4096,"strideMinimum":1,"strideMaximum":4096,"dilationMinimum":1,"dilationMaximum":4096,"paddingMinimum":0,"paddingMaximum":2048,"paddingKernelDivisor":2,"nullStrideFollowsKernel":true,"ceilModes":[false,true],"returnIndicesModes":[false,true],"outputDTypes":["out":"float32","indices":"int64"]])]
        var routing = try JSONSerialization.jsonObject(with:descriptor(operation)) as! [String:Any]
        if malformed { routing["maximumRank"] = 9 }
        try fixtureInstallDiscoveredRuntimeContract(workspace,version:"2.14.1",expectedVersion:nil,devices:["cpu"],descriptor:JSONEncoder().encode(contract),
            tensorLayoutDescriptor:JSONSerialization.data(withJSONObject:["version":["major":1,"minor":0,"patch":0],"adapterID":"kwip.runner.graph-v2.tensor-layout","graphSchemaVersions":[2],"operations":["reshape","transpose","flatten"],"maximumRank":8,"preservesBatch":true]),
            poolingDescriptors:pooling,tensorCompositionDescriptors:composition,
            tensorRoutingDescriptors:verified ? [operation:JSONSerialization.data(withJSONObject:routing)] : [:], moduleCompositionDescriptor:fixtureReviewedModuleCompositionDescriptor())
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
    func testMissingDescriptorRefusesTrainAndExperimentsBeforeCaptureOrReservation() throws {
        for (experiments, operation) in [false, true].flatMap({ experiments in ["split", "chunk", "gather", "einsum"].map { (experiments, $0) } }) {
            let f = try fixture(routingModel: true, operation: operation, verified: false)
            defer { f.host.cancel(); f.workspace.retireWorkflowSession() }
            XCTAssertFalse(experiments ? f.host.runExperiments() : f.host.run(), f.host.statusMessage)
            XCTAssertTrue(f.host.statusMessage.contains(BundledTensorRoutingComponents.entries[operation]!.definition.title), f.host.statusMessage)
            XCTAssertEqual(f.recorder.counts.0, 0); XCTAssertEqual(f.recorder.counts.1, 0)
            XCTAssertFalse(f.workspace.isBusy); XCTAssertFalse(f.host.isStagingExperiment)
            XCTAssertNil(f.host.queuedRequestID)
            XCTAssertFalse(FileManager.default.fileExists(atPath: f.workspace.workspaceDirectory.appendingPathComponent("workflow-source-staging").path))
        }
    }
    func testMismatchedDescriptorCannotAdmitAnySelectedRoutingOperation() throws {
        for operation in ["split", "chunk", "gather", "einsum"] {
            let f = try fixture(routingModel: true, operation: operation, verified: true, malformed: true)
            defer { f.host.cancel(); f.workspace.retireWorkflowSession() }
            XCTAssertFalse(f.workspace.tensorRoutingRuntimeCapabilities.contains(operation))
            XCTAssertFalse(f.host.run(), f.host.statusMessage)
            XCTAssertTrue(f.host.statusMessage.contains(BundledTensorRoutingComponents.entries[operation]!.definition.title), f.host.statusMessage)
            XCTAssertEqual(f.recorder.counts.0, 0); XCTAssertEqual(f.recorder.counts.1, 0)
            XCTAssertFalse(f.workspace.isBusy)
        }
    }
    func testVerifiedDescriptorAllowsTrainAndExperimentCaptureWithControlledCancellation() async throws {
        for (experiments, operation) in [false, true].flatMap({ experiments in ["split", "chunk", "gather", "einsum"].map { (experiments, $0) } }) {
            let f = try fixture(routingModel: true, operation: operation, verified: true)
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
        for (experiments, operation) in [false, true].flatMap({ experiments in ["split", "chunk", "gather", "einsum"].map { (experiments, $0) } }) {
            let f = try fixture(routingModel: true, operation: operation, verified: true, controlledCancellation: false)
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
            f.workspace.tensorRoutingRuntimeCapabilities = .init()
            queue.finish(gate, state: .completed, message: "Release after optional support was revoked")
            try await wait { queue.entries.first(where: { $0.id == queuedID })?.state == .failed && !f.workspace.isBusy }
            let failed = try XCTUnwrap(queue.entries.first { $0.id == queuedID })
            XCTAssertTrue(failed.message.contains(BundledTensorRoutingComponents.entries[operation]!.definition.title), failed.message)
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
    func testDisconnectedRoutingDoesNotBlockTheSelectedTrainGraph() async throws {
        for (experiments, operation) in [false, true].flatMap({ experiments in ["split", "chunk", "gather", "einsum"].map { (experiments, $0) } }) {
            let f = try fixture(routingModel: false, operation: operation, verified: false)
            defer { f.host.cancel(); f.workspace.retireWorkflowSession() }
            let compiled = try WorkflowCompiler.compile(f.host.owner.document, train: WorkflowExamples.trainAddress, modelRegistry: f.host.owner.modelRegistry, dataRegistry: f.host.owner.dataRegistry, contractVersion: max(2, WorkflowArchiveCodec.trainingCompilerContractVersion(forWorkflowSchema: f.host.owner.document.schemaVersion)))
            XCTAssertFalse(compiled.model.graph.nodes.contains { $0.type == operation })
            XCTAssertTrue(experiments ? f.host.runExperiments() : f.host.run(), f.host.statusMessage)
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
            try store.installDiscoveredRuntimeContract(version:"2.14.1",expectedVersion:nil,devices:["cpu"],descriptor:contract,tensorRoutingDescriptors:descriptors)
        }
        try install([])
        let baseCount=store.verifiedRuntimeComponentCount
        for id in ["split","chunk","gather","einsum"] {
            try install([id])
            XCTAssertEqual(store.tensorRoutingRuntimeCapabilities.componentIDs,[id])
            XCTAssertEqual(store.verifiedRuntimeComponentCount,baseCount+1,id)
        }
        try install(["split","chunk","gather","einsum"])
        XCTAssertEqual(store.verifiedRuntimeComponentCount,baseCount+4)
        // Existing helper callers omit optional routing descriptors; that must revoke all four.
        try store.installDiscoveredRuntimeContract(version:"2.14.1",expectedVersion:nil,devices:["cpu"],descriptor:contract)
        XCTAssertTrue(store.tensorRoutingRuntimeCapabilities.componentIDs.isEmpty)
        XCTAssertEqual(store.verifiedRuntimeComponentCount,baseCount)
        try install(["split","chunk","gather","einsum"])
        XCTAssertThrowsError(try store.installDiscoveredRuntimeContract(version:"2.14.1",expectedVersion:nil,devices:["cpu"],descriptor:Data("{}".utf8)))
        XCTAssertTrue(store.tensorRoutingRuntimeCapabilities.componentIDs.isEmpty)
        XCTAssertEqual(store.verifiedRuntimeComponentCount,0)
        try install(["split","chunk","gather","einsum"])
        XCTAssertThrowsError(try store.installRuntimeContract(version:"bad",expectedVersion:nil))
        XCTAssertTrue(store.tensorRoutingRuntimeCapabilities.componentIDs.isEmpty)
        XCTAssertEqual(store.verifiedRuntimeComponentCount,0)
    }
    func testActualWorkerDiscoversAllFourExactReviewedCapabilities() async throws {
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:storage)}
        let store=WorkspaceStore(rootOverride:repo,storageOverride:storage,detectCapabilities:false)
        _ = try XCTUnwrap(store.runtime,store.runtimeError ?? "Actual worker required")
        let ready=await withCheckedContinuation { continuation in store.refreshCapabilities { continuation.resume(returning:$0) } }
        XCTAssertTrue(ready,store.capabilityStatus)
        XCTAssertEqual(store.tensorRoutingRuntimeCapabilities.componentIDs,["split","chunk","gather","einsum"])
        for id in ["split","chunk","gather","einsum"] {
            XCTAssertEqual(store.registry.component(type:id),BundledTensorRoutingComponents.entries[id]!.definition)
        }
    }

}
