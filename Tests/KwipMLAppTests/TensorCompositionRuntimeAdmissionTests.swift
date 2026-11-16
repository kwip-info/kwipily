import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class TensorCompositionRuntimeAdmissionTests: XCTestCase {
    private var compositionEvidence: Data { Data(#"{"version":"1.0.0","adapterID":"kwip.runner.module-composition","graphSchemaVersions":[2],"planSchemaVersion":1,"modelAuthoringSchemaVersion":2,"compositeSourceSchemaVersion":3,"nativeExpansionContractVersion":2,"workflowSchemaVersion":11,"compilerContractVersion":10,"captureFormatVersion":6,"devices":["cpu","mps"],"modes":["branched-structure","nested-structure","parameter-ownership","train-eval","architecture-export","sequential-structure","ordered-structure","named-structure"],"sourceMembers":["torch.nn.Module","torch.nn.Sequential","torch.nn.ModuleList","torch.nn.ModuleDict"],"containerKinds":["graph","sequential","orderedList","namedMap"],"maximumDefinitions":64,"maximumExpandedNodes":128,"maximumEdges":256,"maximumContainers":129,"maximumChildReferences":256,"maximumBoundaryEndpoints":4096,"maximumDepth":8,"maximumPlanBytes":131072,"maximumSourceBytes":2000000,"maximumModelSourceBytes":2000000,"maximumParameters":200000,"parameterOwnership":"expandedNodeRelativeNativeParameter","stateOwnership":"bijectiveNativePersistentState","sharedParameters":false,"sequentialExecution":"nativeForwardValidatedChain","collectionExecution":"explicitDAG","sourceVerification":"independentCapturedRegistryReExpansion","exactResumeDevices":["cpu"],"fineTuneDevices":["cpu","mps"],"mpsResume":"weightsOnly","portableExport":"selfContainedNativeHierarchy","legacyAbsence":"unchanged"}"#.utf8) }
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
    private func fixture(compositionModel: Bool, operation: String, verified: Bool, malformed: Bool = false, controlledCancellation: Bool = true) throws -> Fixture {
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: repo, storageOverride: root, detectCapabilities: false)
        _ = try XCTUnwrap(workspace.runtime, workspace.runtimeError ?? "Fixture runtime unavailable")
        var document: WorkflowDocument
        let entry = try XCTUnwrap(BundledTensorCompositionComponents.entries[operation])
        if compositionModel {
            document = try WorkflowCodec.migrateToVersion4(BundledTensorCompositionComponents.example(operation, device:"cpu"))
        } else {
            document = try WorkflowCodec.migrateToVersion4(WorkflowExamples.preparedParallelClassifier(device:"cpu"))
            // The original parallel example contains a connected Concatenate; build a
            // simple selected model so the unrelated composition block is truly disconnected.
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
            document.nodes.append(.init(address:.init(domain:.model,localID:"disconnected-composition"),name:"Unused composition",x:0,y:0,payload:.modelPrimitive(type:operation,configurationVersion:2,config:config)))
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
        let version=workspace.runtime?.packageVersions["torch"] ?? "2.14.1"
        let contract = try BuiltInCapabilityBridge.executionContract(registry: workspace.registry, runtimeVersion: version, availableDevices: [.cpu])
        var composition = try JSONSerialization.jsonObject(with:JSONEncoder().encode(entry.runtimeContract)) as! [String:Any]
        if malformed {composition["maximumRank"]=9}
        var optional:[String:Any]=["tensorLayout":["version":["major":1,"minor":0,"patch":0],"adapterID":"kwip.runner.graph-v2.tensor-layout","graphSchemaVersions":[2],"operations":["reshape","transpose","flatten"],"maximumRank":8,"preservesBatch":true]]
        if verified {optional[operation]=composition}
        let event:[String:Any]=["event":"capabilities","torchVersion":version,"devices":["cpu"],"defaultDevice":"cpu",
            "executionContract":try JSONSerialization.jsonObject(with:JSONEncoder().encode(contract)),
            "componentCapabilities":optional,"compositionCapabilities":["moduleComposition":try JSONSerialization.jsonObject(with:compositionEvidence)]]
        let line=String(decoding:try JSONSerialization.data(withJSONObject:event),as:UTF8.self)
        workspace.capabilityLaunch={_,_,_,_,receive,exit in receive(line);exit(0,"");return .init(cancel:{})}
        workspace.refreshCapabilities()
        XCTAssertNotNil(workspace.moduleCompositionDiscoveryEvidence,"Admission requires an owned runtime discovery probe, not a parser-only contract install")
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
        for (experiments, operation) in [false, true].flatMap({ experiments in ["concat", "add", "stack", "permute"].map { (experiments, $0) } }) {
            let f = try fixture(compositionModel: true, operation: operation, verified: false)
            defer { f.host.cancel(); f.workspace.retireWorkflowSession() }
            XCTAssertFalse(experiments ? f.host.runExperiments() : f.host.run(), f.host.statusMessage)
            XCTAssertTrue(f.host.statusMessage.contains(BundledTensorCompositionComponents.definition(operation).title), f.host.statusMessage)
            XCTAssertEqual(f.recorder.counts.0, 0); XCTAssertEqual(f.recorder.counts.1, 0)
            XCTAssertFalse(f.workspace.isBusy); XCTAssertFalse(f.host.isStagingExperiment)
            XCTAssertNil(f.host.queuedRequestID)
            XCTAssertFalse(FileManager.default.fileExists(atPath: f.workspace.workspaceDirectory.appendingPathComponent("workflow-source-staging").path))
        }
    }
    func testMismatchedDescriptorCannotAdmitAnySelectedCompositionOperation() throws {
        for operation in ["concat", "add", "stack", "permute"] {
            let f = try fixture(compositionModel: true, operation: operation, verified: true, malformed: true)
            defer { f.host.cancel(); f.workspace.retireWorkflowSession() }
            XCTAssertFalse(f.workspace.tensorCompositionRuntimeCapabilities.contains(operation))
            XCTAssertFalse(f.host.run(), f.host.statusMessage)
            XCTAssertTrue(f.host.statusMessage.contains(BundledTensorCompositionComponents.definition(operation).title), f.host.statusMessage)
            XCTAssertEqual(f.recorder.counts.0, 0); XCTAssertEqual(f.recorder.counts.1, 0)
            XCTAssertFalse(f.workspace.isBusy)
        }
    }
    func testVerifiedDescriptorAllowsTrainAndExperimentCaptureWithControlledCancellation() async throws {
        for (experiments, operation) in [false, true].flatMap({ experiments in ["concat", "add", "stack", "permute"].map { (experiments, $0) } }) {
            let f = try fixture(compositionModel: true, operation: operation, verified: true)
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
        for (experiments, operation) in [false, true].flatMap({ experiments in ["concat", "add", "stack", "permute"].map { (experiments, $0) } }) {
            let f = try fixture(compositionModel: true, operation: operation, verified: true, controlledCancellation: false)
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
            f.workspace.tensorCompositionRuntimeCapabilities = .init()
            queue.finish(gate, state: .completed, message: "Release after optional support was revoked")
            try await wait { queue.entries.first(where: { $0.id == queuedID })?.state == .failed && !f.workspace.isBusy }
            let failed = try XCTUnwrap(queue.entries.first { $0.id == queuedID })
            XCTAssertTrue(failed.message.contains(BundledTensorCompositionComponents.definition(operation).title), failed.message)
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
        for (experiments, operation) in [false, true].flatMap({ experiments in ["concat", "add", "stack", "permute"].map { (experiments, $0) } }) {
            let f = try fixture(compositionModel: false, operation: operation, verified: false)
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
            let descriptors=try Dictionary(uniqueKeysWithValues:ids.map { ($0,try JSONEncoder().encode(BundledTensorCompositionComponents.entries[$0]!.runtimeContract)) })
            try store.installDiscoveredRuntimeContract(version:"2.14.1",expectedVersion:nil,devices:["cpu"],descriptor:contract,tensorCompositionDescriptors:descriptors)
        }
        try install([])
        let baseCount=store.verifiedRuntimeComponentCount
        for id in ["concat","add","stack","permute"] {
            try install([id])
            XCTAssertEqual(store.tensorCompositionRuntimeCapabilities.componentIDs,[id])
            XCTAssertEqual(store.verifiedRuntimeComponentCount,baseCount+1,id)
        }
        try install(["concat","add","stack","permute"])
        XCTAssertEqual(store.verifiedRuntimeComponentCount,baseCount+4)
        // Existing helper callers omit optional composition descriptors; that must revoke all four.
        try store.installDiscoveredRuntimeContract(version:"2.14.1",expectedVersion:nil,devices:["cpu"],descriptor:contract)
        XCTAssertTrue(store.tensorCompositionRuntimeCapabilities.componentIDs.isEmpty)
        XCTAssertEqual(store.verifiedRuntimeComponentCount,baseCount)
        try install(["concat","add","stack","permute"])
        XCTAssertThrowsError(try store.installDiscoveredRuntimeContract(version:"2.14.1",expectedVersion:nil,devices:["cpu"],descriptor:Data("{}".utf8)))
        XCTAssertTrue(store.tensorCompositionRuntimeCapabilities.componentIDs.isEmpty)
        XCTAssertEqual(store.verifiedRuntimeComponentCount,0)
        try install(["concat","add","stack","permute"])
        XCTAssertThrowsError(try store.installRuntimeContract(version:"bad",expectedVersion:nil))
        XCTAssertTrue(store.tensorCompositionRuntimeCapabilities.componentIDs.isEmpty)
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
        XCTAssertEqual(store.tensorCompositionRuntimeCapabilities.componentIDs,["concat","add","stack","permute"])
        for id in ["concat","add","stack","permute"] {
            XCTAssertEqual(store.registry.component(type:id),BundledTensorCompositionComponents.definition(id))
        }
    }

}
