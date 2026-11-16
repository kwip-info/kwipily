import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class StatefulNormalizationRuntimeAdmissionTests: XCTestCase {
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
    /// Independently authored literal descriptors, not encoded from the Core decoder.
    private func descriptor(_ id:String)throws->Data {
        let ranks=["batch_norm1d":[1,2],"batch_norm2d":[3],"instance_norm1d":[2],"instance_norm2d":[3],"group_norm":[1,2,3,4,5,6,7]]
        var value:[String:Any]=["version":["major":1,"minor":0,"patch":0],"adapterID":"kwip.runner.graph-v2."+id,"graphSchemaVersions":[2],"inputSampleRanks":ranks[id]!,"channelAxis":1,"channelMinimum":1,"channelMaximum":512,"spatialMinimum":1,"spatialMaximum":4096,"preservesBatch":true,"graphDTypes":["float32"],"epsilonMinimum":1.1754943508222875e-38,"epsilonMaximum":3.4028234663852886e38,"affineModes":[false,true],"biasModes":[false,true],"forwardVarianceCorrection":0,"maximumTensorElements":100000000]
        if id == "group_norm" {
            value.merge(["groupMinimum":1,"groupMaximum":512,"requiresChannelDivisibility":true,"minimumGroupMembers":2,"statisticsScope":"observationGroupSpatial","usesCurrentStatistics":true]) {$1}
        } else {
            value.merge(["momentumMinimum":0.0,"momentumMaximum":1.0,"trackingModes":[false,true],"runningVarianceCorrection":1,"runningMeanDType":"float32","runningVarianceDType":"float32","counterDType":"int64","counterMaximum":1000000000,"statisticsScope":id.hasPrefix("batch") ? "batchSpatial":"observationChannelSpatial","evaluationUsesRunningStatisticsWhenTracked":true]) {$1}
            if id.hasPrefix("batch") {value.merge(["averagingModes":["moving","cumulative"],"counterBehavior":"trainingCalls","minimumCurrentStatisticsMembers":2]) {$1}}
            else {value.merge(["counterBehavior":"constantZero","minimumCurrentStatisticsSpatialElements":2]) {$1}}
        }
        return try JSONSerialization.data(withJSONObject:value)
    }
    private func fixture(normalizationModel: Bool, operation: String, verified: Bool, malformed: Bool = false, controlledCancellation: Bool = true, directModel: Bool = false) throws -> Fixture {
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: repo, storageOverride: root, detectCapabilities: false)
        _ = try XCTUnwrap(workspace.runtime, workspace.runtimeError ?? "Fixture runtime unavailable")
        var document: WorkflowDocument
        let entry = try XCTUnwrap(BundledNormalizationComponents.entries[operation])
        if normalizationModel {
            let factories:[String:()throws->WorkflowDocument]=["batch_norm1d":{try WorkflowExamples.preparedBatchNorm1dClassifier(device:"cpu")},"batch_norm2d":{try WorkflowExamples.preparedBatchNorm2dClassifier(device:"cpu")},"instance_norm1d":{try WorkflowExamples.preparedInstanceNorm1dClassifier(device:"cpu")},"instance_norm2d":{try WorkflowExamples.preparedInstanceNorm2dClassifier(device:"cpu")},"group_norm":{try WorkflowExamples.preparedGroupNormClassifier(device:"cpu")}]
            document = try WorkflowCodec.migrateToVersion4(XCTUnwrap(factories[operation])())
        } else {
            document = try WorkflowCodec.migrateToVersion4(WorkflowExamples.preparedParallelClassifier(device:"cpu"))
            // The original parallel example contains a connected Concatenate; build a
            // simple selected model so the unrelated normalization block is truly disconnected.
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
            document.nodes.append(.init(address:.init(domain:.model,localID:"disconnected-normalization"),name:"Unused normalization",x:0,y:0,payload:.modelPrimitive(type:operation,configurationVersion:2,config:config)))
        }
        if directModel {
            // Legal [N,2] normalization with no reshape or other optional operator:
            // admission must be owned by normalization itself, not a neighboring block.
            document.nodes.removeAll {$0.address.domain == .model}
            document.edges.removeAll {$0.from.address.domain == .model || $0.to.address.domain == .model}
            document.definitions=[]
            func address(_ id:String)->WorkflowNodeAddress {.init(domain:.model,localID:id)}
            let norm=try StatefulNormalizationConfiguration(componentID:operation,values:[:]).values
            for (id,type,config) in [("input","input",["features":JSONValue.integer(2)]),("normalize",operation,norm),("head","linear",["in_features":2,"out_features":2,"bias":true]),("output","output",[:])] {
                document.nodes.append(.init(address:address(id),name:type,x:0,y:0,payload:.modelPrimitive(type:type,configurationVersion:2,config:config)))
            }
            for (source,target) in [("input","normalize"),("normalize","head"),("head","output")] {
                document.edges.append(.init(from:.init(address:address(source),port:"out"),to:.init(address:address(target),port:"in"),kind:.tensor))
            }
            document.edges.append(.init(from:.init(address:address("output"),port:"definition"),to:.init(address:.init(domain:.control,localID:"model"),port:"source"),kind:.modelDefinition))
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
        var patch = try JSONSerialization.jsonObject(with:descriptor(operation)) as! [String:Any]
        if malformed { patch["inputSampleRanks"] = [9] }
        try fixtureInstallDiscoveredRuntimeContract(workspace,version:"2.14.1",expectedVersion:nil,devices:["cpu"],descriptor:JSONEncoder().encode(contract),
            tensorLayoutDescriptor:JSONSerialization.data(withJSONObject:["version":["major":1,"minor":0,"patch":0],"adapterID":"kwip.runner.graph-v2.tensor-layout","graphSchemaVersions":[2],"operations":["reshape","transpose","flatten"],"maximumRank":8,"preservesBatch":true]),
            statefulNormalizationDescriptors:verified ? [operation:JSONSerialization.data(withJSONObject:patch)] : [:], moduleCompositionDescriptor:fixtureReviewedModuleCompositionDescriptor())
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
        for (experiments, operation) in [false, true].flatMap({ experiments in ["batch_norm1d", "batch_norm2d", "instance_norm1d", "instance_norm2d", "group_norm"].map { (experiments, $0) } }) {
            let f = try fixture(normalizationModel: true, operation: operation, verified: false)
            defer { f.host.cancel(); f.workspace.retireWorkflowSession() }
            XCTAssertFalse(experiments ? f.host.runExperiments() : f.host.run(), f.host.statusMessage)
            XCTAssertTrue(f.host.statusMessage.contains(BundledNormalizationComponents.entries[operation]!.definition.title), f.host.statusMessage)
            XCTAssertEqual(f.recorder.counts.0, 0); XCTAssertEqual(f.recorder.counts.1, 0)
            XCTAssertFalse(f.workspace.isBusy); XCTAssertFalse(f.host.isStagingExperiment)
            XCTAssertNil(f.host.queuedRequestID)
            XCTAssertFalse(FileManager.default.fileExists(atPath: f.workspace.workspaceDirectory.appendingPathComponent("workflow-source-staging").path))
        }
    }
    func testDirectRankTwoNormalizationRefusesMissingSupportWithoutAnotherGatedOperator()throws {
        for (experiments,operation) in [false,true].flatMap({experiments in ["batch_norm1d","group_norm"].map {(experiments,$0)}}) {
            let f=try fixture(normalizationModel:true,operation:operation,verified:false,directModel:true)
            defer {f.host.cancel();f.workspace.retireWorkflowSession()}
            XCTAssertFalse(experiments ? f.host.runExperiments():f.host.run(),f.host.statusMessage)
            XCTAssertTrue(f.host.statusMessage.contains(BundledNormalizationComponents.definition(operation).title),f.host.statusMessage)
            XCTAssertEqual(f.recorder.counts.0,0);XCTAssertEqual(f.recorder.counts.1,0)
            XCTAssertFalse(f.workspace.isBusy);XCTAssertNil(f.host.queuedRequestID)
        }
    }
    func testMismatchedDescriptorCannotAdmitAnySelectedNormalizationOperation() throws {
        for operation in ["batch_norm1d", "batch_norm2d", "instance_norm1d", "instance_norm2d", "group_norm"] {
            let f = try fixture(normalizationModel: true, operation: operation, verified: true, malformed: true)
            defer { f.host.cancel(); f.workspace.retireWorkflowSession() }
            XCTAssertFalse(f.workspace.statefulNormalizationRuntimeCapabilities.contains(operation))
            XCTAssertFalse(f.host.run(), f.host.statusMessage)
            XCTAssertTrue(f.host.statusMessage.contains(BundledNormalizationComponents.entries[operation]!.definition.title), f.host.statusMessage)
            XCTAssertEqual(f.recorder.counts.0, 0); XCTAssertEqual(f.recorder.counts.1, 0)
            XCTAssertFalse(f.workspace.isBusy)
        }
    }
    func testVerifiedDescriptorAllowsTrainAndExperimentCaptureWithControlledCancellation() async throws {
        for (experiments, operation) in [false, true].flatMap({ experiments in ["batch_norm1d", "batch_norm2d", "instance_norm1d", "instance_norm2d", "group_norm"].map { (experiments, $0) } }) {
            let f = try fixture(normalizationModel: true, operation: operation, verified: true)
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
        for (experiments, operation) in [false, true].flatMap({ experiments in ["batch_norm1d", "batch_norm2d", "instance_norm1d", "instance_norm2d", "group_norm"].map { (experiments, $0) } }) {
            let f = try fixture(normalizationModel: true, operation: operation, verified: true, controlledCancellation: false)
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
            f.workspace.statefulNormalizationRuntimeCapabilities = .init()
            queue.finish(gate, state: .completed, message: "Release after optional support was revoked")
            try await wait { queue.entries.first(where: { $0.id == queuedID })?.state == .failed && !f.workspace.isBusy }
            let failed = try XCTUnwrap(queue.entries.first { $0.id == queuedID })
            XCTAssertTrue(failed.message.contains(BundledNormalizationComponents.entries[operation]!.definition.title), failed.message)
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
    func testDisconnectedNormalizationDoesNotBlockTheSelectedTrainGraph() async throws {
        for (experiments, operation) in [false, true].flatMap({ experiments in ["batch_norm1d", "batch_norm2d", "instance_norm1d", "instance_norm2d", "group_norm"].map { (experiments, $0) } }) {
            let f = try fixture(normalizationModel: false, operation: operation, verified: false)
            defer { f.host.cancel(); f.workspace.retireWorkflowSession() }
            let compiled = try WorkflowCompiler.compile(f.host.owner.document, train: WorkflowExamples.trainAddress, modelRegistry: f.host.owner.modelRegistry, dataRegistry: f.host.owner.dataRegistry, contractVersion: max(2, WorkflowArchiveCodec.trainingCompilerContractVersion(forWorkflowSchema: f.host.owner.document.schemaVersion)))
            XCTAssertFalse(compiled.model.graph.nodes.contains { $0.type == operation })
            XCTAssertTrue(experiments ? f.host.runExperiments() : f.host.run(), f.host.statusMessage)
            try await waitForCaptureToFinish(f)
        }
    }

}
