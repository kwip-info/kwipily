import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp

final class WorkflowExecutionSessionTests: XCTestCase {
    private func address(_ domain:WorkflowDomain,_ id:String)->WorkflowNodeAddress {.init(domain:domain,localID:id)}
    private func edge(_ a:WorkflowDomain,_ id:String,_ port:String,_ b:WorkflowDomain,_ target:String,_ input:String,_ kind:WorkflowEdgeKind)->WorkflowEdge {
        .init(id:"\(a)-\(id)-\(port)-\(b)-\(target)-\(input)",from:.init(address:address(a,id),port:port),to:.init(address:address(b,target),port:input),kind:kind)
    }
    private func fixture(block:Bool=false)->WorkflowDocument {CurrentExecutionWorkflowTestFixture.make(block:block)}

    private var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    private func dataRegistry() throws -> DataFlowOperatorRegistry {
        try .load(from: repo.appendingPathComponent("runner/dataflow-operators.json"))
    }
    @MainActor private func setup() throws -> (URL, WorkflowAuthoringStore, RuntimeLocation) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let library = ProjectLibrary(directory: root.appendingPathComponent("projects"))
        let source = fixture(block: true)
        _ = try library.saveSource(.workflow(source), id: source.projectID)
        let runtime = try makeRuntime(root.appendingPathComponent("runtime"))
        return (root, try WorkflowAuthoringStore(library: library, projectID: source.projectID,
            modelRegistry: .builtIn(), dataRegistry: dataRegistry()), runtime)
    }
    private func makeRuntime(_ root: URL) throws -> RuntimeLocation {
        for name in [".venv/bin", "runner", "scripts", "Sources/KwipMLCore/Resources"] {
            try FileManager.default.createDirectory(at: root.appendingPathComponent(name), withIntermediateDirectories: true)
        }
        try Data("fixture interpreter; never executed".utf8).write(to: root.appendingPathComponent(".venv/bin/python"))
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: root.appendingPathComponent(".venv/bin/python").path)
        for name in ["runner/main.py", "runner/engine.py", "scripts/runtime_launch.py"] {
            try Data("# source fixture\n".utf8).write(to: root.appendingPathComponent(name))
        }
        try JSONEncoder().encode(ComponentRegistry.builtIn()).write(to: root.appendingPathComponent("Sources/KwipMLCore/Resources/components.json"))
        try JSONEncoder().encode(dataRegistry()).write(to: root.appendingPathComponent("runner/dataflow-operators.json"))
        return try RuntimeLocation.resolve(resources: nil, developmentRoot: root, allowDevelopment: true)
    }
    @MainActor private final class Harness {
        struct Call {
            let arguments:[String]
            let line:(String)->Void
            let exit:(Int32,String)->Void
        }
        var calls:[Call] = []
        var cancellations = 0
        var acquired = 0
        var released = 0
        var available = true
        var outcomes:[WorkflowExecutionOutcome] = []
        var reservation:WorkflowExecutionReservation {
            .init(acquire:{ [self] in guard available else {return nil}; acquired += 1; return UUID() },
                  release:{ [self] _ in released += 1 })
        }
        var launcher:WorkflowWorkerLauncher {
            { [self] _,arguments,_,line,exit in
                calls.append(.init(arguments:arguments,line:line,exit:exit))
                return WorkflowWorkerHandle(cancel:{ [self] in cancellations += 1 })
            }
        }
    }
    @MainActor private func session(_ owner:WorkflowAuthoringStore,_ runtime:RuntimeLocation,_ root:URL,_ harness:Harness) throws -> WorkflowExecutionSession {
        let request = try WorkflowExecutionRequest.capture(from:owner,train:address(.control,"train"),runtime:runtime)
        let source = root.appendingPathComponent("source.csv")
        try Data("a,b,label\n1,2,0\n2,3,1\n3,4,0\n4,5,1\n".utf8).write(to:source)
        let captured = try WorkflowCapturedSource.capture(source)
        return WorkflowExecutionSession(request:request,owner:owner,
            sources:[address(.data,"input"):captured],directory:root.appendingPathComponent("10000000-0000-0000-0000-000000000001"),
            reservation:harness.reservation,launch:harness.launcher,onTerminal:{ harness.outcomes.append($0) })
    }
    @MainActor private func awaitWorker(_ harness:Harness) async throws {
        let deadline=Date().addingTimeInterval(5)
        while harness.calls.isEmpty && Date()<deadline {try await Task.sleep(for:.milliseconds(1))}
        guard !harness.calls.isEmpty else {throw NSError(domain:"TestWorkerTimeout",code:1)}
    }
    @MainActor func testActualPreparationProgressConsumesOnlyItsOwnInputsAndClearsOnCancel() async throws {
        let (root,owner,runtime)=try setup();defer {try? FileManager.default.removeItem(at:root)}
        let harness=Harness(),run=try session(owner,runtime,root,harness)
        XCTAssertTrue(run.start());try await awaitWorker(harness)
        let call=try XCTUnwrap(harness.calls.first)
        XCTAssertTrue(run.canvasProgress.activeNodeAddresses.isEmpty)
        call.line("{\"event\":\"dataflowNodeStarted\",\"nodeID\":\"cast\"}")
        XCTAssertEqual(run.canvasProgress.activeNodeAddresses,[address(.data,"cast")])
        XCTAssertEqual(run.canvasProgress.activeEdgeIDs,["data-input-table-data-cast-table"])
        call.line("{\"event\":\"dataflowPreview\",\"nodeID\":\"cast\",\"outputs\":{},\"execution\":{\"nodeID\":\"cast\",\"status\":\"reused\"}}")
        XCTAssertTrue(run.canvasProgress.activeNodeAddresses.isEmpty)
        XCTAssertEqual(run.canvasProgress.completed[address(.data,"cast")],.reused)
        call.line("{\"event\":\"dataflowNodeStarted\",\"nodeID\":\"unknown\"}")
        XCTAssertTrue(run.canvasProgress.activeNodeAddresses.isEmpty)
        call.line("{\"event\":\"dataflowNodeStarted\",\"nodeID\":\"split\"}")
        run.cancel();XCTAssertTrue(run.canvasProgress.activeNodeAddresses.isEmpty)
        call.line("{\"event\":\"dataflowNodeStarted\",\"nodeID\":\"cast\"}")
        XCTAssertTrue(run.canvasProgress.activeNodeAddresses.isEmpty)
        call.exit(0,"")
    }
    /// Opt-in measurement, not a hardware-independent latency acceptance threshold.
    @MainActor func testBudgetedOrdinaryPreparationUsesPinnedSupervisorAndTimeoutCannotPublishSuccess() async throws {
        for scenario in ["timeout", "cancel", "wrongReady", "memory"] {
        let (temporaryRoot, owner, runtime) = try setup()
        let canonical = try XCTUnwrap(realpath(temporaryRoot.path, nil))
        let root = URL(fileURLWithPath: String(cString: canonical))
        free(canonical)
        defer { try? FileManager.default.removeItem(at: root) }
        let policy = try ResolvedExecutionBudget.resolve(global: .init(maximumRunSeconds: 30), projectMaximumRunSeconds: nil)
        let request = try WorkflowExecutionRequest.capture(from: owner, train: address(.control, "train"), runtime: runtime, executionBudget: policy)
        let source = root.appendingPathComponent("source.csv")
        try Data("a,b,label\n1,2,0\n2,3,1\n".utf8).write(to: source)
        let directory = root.appendingPathComponent("workflow-runs/\(owner.projectID)/\(UUID().uuidString)")
        let harness = Harness()
        let run = WorkflowExecutionSession(request: request, owner: owner,
            sources: [address(.data, "input"): try .capture(source)], directory: directory,
            reservation: harness.reservation, launch: harness.launcher,
            onTerminal: { harness.outcomes.append($0) })
        XCTAssertTrue(run.start())
        try await awaitWorker(harness)
        let call = try XCTUnwrap(harness.calls.first)
        let index = try XCTUnwrap(call.arguments.firstIndex(of: "--kwip-operation-supervise"))
        let folder = URL(fileURLWithPath: call.arguments[index + 1])
        let bytes = try Data(contentsOf: folder.appendingPathComponent("request.json"))
        let envelope = try WorkflowOperationRequest.decode(bytes)
        XCTAssertEqual(envelope.executionBudget, policy)
        XCTAssertEqual(envelope.requestIdentity, request.requestIdentity)
        XCTAssertGreaterThan(try XCTUnwrap(envelope.remainingSeconds), 0)
        XCTAssertLessThanOrEqual(try XCTUnwrap(envelope.remainingSeconds), 30)
        XCTAssertEqual(call.arguments[index + 2], WorkflowArchiveCodec.hash(bytes))
        let ready: [String: Any] = ["event": "operationReady", "operationID": scenario == "wrongReady" ? UUID().uuidString : envelope.operationID,
            "phase": "preparation", "requestSHA256": call.arguments[index + 2], "ownershipSHA256": call.arguments[index + 3]]
        call.line(String(decoding: try JSONSerialization.data(withJSONObject: ready), as: UTF8.self))
        if scenario == "cancel" { run.cancel() }
        call.line("{\"event\":\"dataflowCompleted\",\"outputDir\":\"/untrusted\"}")
        XCTAssertEqual(harness.released, 0)
        XCTAssertTrue(owner.isBusy)
        XCTAssertTrue(harness.outcomes.isEmpty)
        if scenario == "cancel" || scenario == "wrongReady" { XCTAssertEqual(harness.cancellations, 1) }
        if scenario == "memory" { call.line("{\"event\":\"error\",\"message\":\"Run cancelled.\"}") }
        call.exit(scenario == "memory" ? 125 : scenario == "wrongReady" ? 0 : 124, scenario == "memory" ? "The worker exceeded its sampled resident memory budget. Increase the memory budget." : "")
        if scenario == "memory" { XCTAssertTrue(run.statusMessage.contains("sampled resident memory")) }
        XCTAssertEqual(run.phase, scenario == "cancel" ? .cancelled : .failed)
        XCTAssertEqual(harness.calls.count, 1, "Timed-out preparation cannot launch training")
        XCTAssertEqual(harness.released, 1)
        if scenario == "timeout" { XCTAssertTrue(run.statusMessage.contains("30-second")) }
        let outcome = try WorkflowArchiveCodec.decodeOutcome(Data(contentsOf: directory.appendingPathComponent("outcome.json")))
        XCTAssertEqual(outcome.failureCode, scenario == "memory" ? .resourceBudgetFailed : scenario == "timeout" ? .wallTimeLimitExceeded : nil)
        XCTAssertNil(outcome.modelManifestSHA256)
        XCTAssertEqual(outcome.status, scenario == "cancel" ? .cancelled : .failed)
        }
    }

    @MainActor func testActualBudgetedOrdinaryRunSharesAllowanceAcrossPreparationAndTraining() async throws {
        guard let development = ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {
            throw XCTSkip("Set KWIPML_TEST_DEVELOPMENT_ROOT for real supervised training.")
        }
        let (temporaryRoot, owner, _) = try setup()
        let canonical = try XCTUnwrap(realpath(temporaryRoot.path, nil))
        let root = URL(fileURLWithPath: String(cString: canonical))
        free(canonical)
        defer { try? FileManager.default.removeItem(at: root) }
        let runtime = try RuntimeLocation.resolve(resources: nil, developmentRoot: URL(fileURLWithPath: development), allowDevelopment: true)
        let policy = try ResolvedExecutionBudget.resolve(global: .init(maximumRunSeconds: 60), projectMaximumRunSeconds: nil, resources: .init(cpuThreads: 2, maximumResidentBytes: 1_073_741_824))
        let request = try WorkflowExecutionRequest.capture(from: owner, train: address(.control, "train"), runtime: runtime, executionBudget: policy)
        let source = root.appendingPathComponent("source.csv")
        let rows = (0..<64).map { "\($0),\(($0 * 7) % 13),\($0 % 2)" }
        try Data((["a,b,label"] + rows).joined(separator: "\n").utf8).write(to: source)
        let harness = Harness(), done = expectation(description: "Supervised ordinary workflow")
        var outcome: WorkflowExecutionOutcome?
        let directory = root.appendingPathComponent("workflow-runs/\(owner.projectID)/\(UUID().uuidString)")
        let run = WorkflowExecutionSession(request: request, owner: owner,
            sources: [address(.data, "input"): try .capture(source)], directory: directory,
            reservation: harness.reservation, onTerminal: { outcome = $0; done.fulfill() })
        XCTAssertTrue(run.start())
        await fulfillment(of: [done], timeout: 65)
        if owner.isBusy { run.cancel() }
        XCTAssertEqual(run.phase, .completed, run.statusMessage)
        XCTAssertNotNil(outcome?.modelManifestSHA256)
        XCTAssertEqual(run.modelOperationProgress?.completedNodeIDs,Set(request.plan.model.graph.nodes.map(\.id)),"Real validation capture visits every captured model operation")
        XCTAssertNil(run.modelOperationProgress?.activeOrigin,"Terminal completion clears live model activity")
        XCTAssertTrue(run.console.entries.contains {$0.message.contains("modelCaptureCompleted")})
        XCTAssertEqual(harness.released, 1)
        let operations = root.appendingPathComponent("workflow-operations/\(owner.projectID)")
        let folders = try FileManager.default.contentsOfDirectory(at: operations, includingPropertiesForKeys: nil)
        let requests = try folders.map { try WorkflowOperationRequest.decode(Data(contentsOf: $0.appendingPathComponent("request.json"))) }
        XCTAssertEqual(requests.count, 2)
        let preparation = try XCTUnwrap(requests.first { $0.phase == .preparation })
        let training = try XCTUnwrap(requests.first { $0.phase == .training })
        XCTAssertEqual(preparation.executionBudget, policy)
        XCTAssertEqual(training.executionBudget, policy)
        XCTAssertLessThan(try XCTUnwrap(training.remainingSeconds), try XCTUnwrap(preparation.remainingSeconds))
        for folder in folders { XCTAssertTrue(FileManager.default.fileExists(atPath: folder.appendingPathComponent("claim.json").path)) }
        let reference = WorkflowHistoryReference(projectID: owner.projectID, directoryID: try XCTUnwrap(UUID(uuidString: directory.lastPathComponent)))
        guard case .verified = WorkflowHistoryReader.load(reference, root: root) else {
            return XCTFail("Resource-bearing workflow archive must verify independently after completion")
        }

    }

    @MainActor func testNearLimitSourceArchiveResponsiveness() async throws {
        guard ProcessInfo.processInfo.environment["KWIPML_ARCHIVE_BENCHMARK"] == "1" else {throw XCTSkip("Enable the source archive benchmark explicitly")}
        let(root,owner,runtime)=try setup();defer {try? FileManager.default.removeItem(at:root)}
        let request=try WorkflowExecutionRequest.capture(from:owner,train:address(.control,"train"),runtime:runtime)
        let source=root.appendingPathComponent("large.csv")
        // Valid CSV near the ten-megabyte per-source cap; this benchmark stops before parsing/training.
        var bytes=Data("a,b,label\n".utf8)
        bytes.append(Data(String(repeating:"1,2,0\n",count:1_650_000).utf8))
        try bytes.write(to:source)
        let sources=[address(.data,"input"):try WorkflowCapturedSource.capture(source)]
        let baseline=root.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:baseline,withIntermediateDirectories:false)
        let clock=ProcessInfo.processInfo
        let baselineStart=clock.systemUptime
        _=try WorkflowArchiveCaptureWriter.write(request:request,runID:UUID(),sources:sources,directory:baseline,
            modelRegistry:owner.modelRegistry,dataRegistry:owner.dataRegistry)
        let baselineDuration=clock.systemUptime-baselineStart
        let done=expectation(description:"Archive complete at worker launch boundary")
        let directory=root.appendingPathComponent(UUID().uuidString)
        var launches=0,releases=0,heartbeatTimes:[Double]=[],outcome:WorkflowExecutionOutcome?
        let run=WorkflowExecutionSession(request:request,owner:owner,sources:sources,directory:directory,
            reservation:.init(acquire:{UUID()},release:{_ in releases+=1}),
            launch:{_,_,_,_,_ in launches+=1;throw WorkflowPhaseError("Source benchmark worker boundary")},
            onTerminal:{outcome=$0;done.fulfill()})
        let start=clock.systemUptime
        XCTAssertTrue(run.start())
        let returned=clock.systemUptime
        let heartbeat=Task { @MainActor in
            while !Task.isCancelled {
                heartbeatTimes.append(clock.systemUptime)
                try? await Task.sleep(for:.milliseconds(2))
            }
        }
        await fulfillment(of:[done],timeout:20)
        let finished=clock.systemUptime
        heartbeat.cancel()
        XCTAssertEqual(launches,1);XCTAssertEqual(releases,1);XCTAssertFalse(owner.isBusy)
        XCTAssertEqual(outcome?.status,.failed);XCTAssertEqual(outcome?.error,"Source benchmark worker boundary")
        XCTAssertNil(outcome?.recordError)
        let capture=try WorkflowArchiveCodec.decodeCapture(Data(contentsOf:directory.appendingPathComponent("capture.json")))
        XCTAssertEqual(capture.sources.first?.bytes,bytes.count)
        XCTAssertEqual(capture.sources.first?.sha256,WorkflowArchiveCodec.hash(bytes))
        let samples=[start]+heartbeatTimes+[finished]
        let gap=zip(samples,samples.dropFirst()).map {$1-$0}.max() ?? 0
        print(String(format:"SOURCE_ARCHIVE_BENCHMARK bytes=%d synchronous_writer_ms=%.3f start_return_ms=%.3f session_to_worker_ms=%.3f main_actor_callbacks=%d maximum_observed_gap_ms=%.3f",bytes.count,baselineDuration*1000,(returned-start)*1000,(finished-start)*1000,heartbeatTimes.count,gap*1000))
    }

    @MainActor func testCancelBeforeArchivePreparationStartsNoWorkerAndReleasesOnce() async throws {
        let(root,owner,runtime)=try setup();defer {try? FileManager.default.removeItem(at:root)}
        let harness=Harness(),run=try session(owner,runtime,root,harness)
        XCTAssertTrue(run.start());XCTAssertTrue(owner.isBusy)
        XCTAssertTrue(harness.calls.isEmpty,"Archive staging must yield before worker launch")
        run.cancel();run.cancel()
        let deadline=Date().addingTimeInterval(5)
        while harness.outcomes.isEmpty && Date()<deadline {try await Task.sleep(for:.milliseconds(1))}
        XCTAssertEqual(run.phase,.cancelled);XCTAssertFalse(owner.isBusy)
        XCTAssertTrue(harness.calls.isEmpty)
        XCTAssertEqual(harness.acquired,1);XCTAssertEqual(harness.released,1)
        XCTAssertEqual(harness.outcomes.count,1)
    }

    @MainActor func testConsoleBoundsAndCancellationRejectLateWorkerMessages() async throws {
        let(root,owner,runtime)=try setup();defer{try? FileManager.default.removeItem(at:root)}
        let harness=Harness(),run=try session(owner,runtime,root,harness)
        XCTAssertTrue(run.start())
        try await awaitWorker(harness)
        XCTAssertEqual(run.console.entries.first?.phase,.preparing)
        run.cancel()
        let count=run.console.entries.count
        harness.calls[0].line("{\"event\":\"error\",\"message\":\"late ignored error\"}")
        XCTAssertEqual(run.console.entries.count,count)
        harness.calls[0].exit(0,"")
        XCTAssertEqual(run.console.entries.last?.phase,.cancelled)
        let log=WorkflowRunConsole()
        for _ in 0..<510 {log.append(phase:.training,level:.info,message:String(repeating:"x",count:5000))}
        XCTAssertEqual(log.entries.count,500);XCTAssertEqual(log.omittedCount,10)
        XCTAssertTrue(log.entries.allSatisfy {$0.message.count<=4000})
        XCTAssertEqual(Set(log.entries.map(\.id)).count,500)
    }
    @MainActor func testReservationAndCancellationRemainOwnedUntilExitAndNeverLaunchTraining() async throws {
        let(root,owner,runtime) = try setup();defer{try? FileManager.default.removeItem(at:root)}
        let harness = Harness(), run = try session(owner,runtime,root,harness)
        XCTAssertTrue(run.start());XCTAssertTrue(owner.isBusy)
        try await awaitWorker(harness);XCTAssertEqual(harness.calls.count,1)
        XCTAssertTrue(harness.calls[0].arguments.contains("--execute-dataflow"))
        XCTAssertTrue(harness.calls[0].arguments.contains("--retain-partial-dataflow"), "Owned preparation keeps incomplete materialization available after cancellation or owner death")
        XCTAssertEqual(harness.acquired,1);XCTAssertEqual(harness.released,0)
        run.cancel();run.cancel()
        XCTAssertEqual(harness.cancellations,1);XCTAssertTrue(owner.isBusy);XCTAssertTrue(harness.outcomes.isEmpty)
        harness.calls[0].line("{\"event\":\"dataflowCompleted\"}")
        harness.calls[0].exit(0,"")
        XCTAssertEqual(run.phase,.cancelled);XCTAssertFalse(owner.isBusy)
        XCTAssertEqual(harness.calls.count,1);XCTAssertEqual(harness.released,1);XCTAssertEqual(harness.outcomes.count,1)
        XCTAssertEqual(harness.outcomes[0].request.document,run.request.document)
        XCTAssertNil(harness.outcomes[0].modelManifestSHA256)
        harness.calls[0].exit(0,"");XCTAssertEqual(harness.outcomes.count,1);XCTAssertEqual(harness.released,1)
    }
    @MainActor func testFailedPreparationRetainsCapturedIdentityAndCannotPublishSuccessWithoutArtifact() async throws {
        let(root,owner,runtime) = try setup();defer{try? FileManager.default.removeItem(at:root)}
        let harness = Harness(), run = try session(owner,runtime,root,harness)
        XCTAssertTrue(run.start())
        try await awaitWorker(harness)
        harness.calls[0].line("{\"event\":\"dataflowCompleted\",\"outputDir\":\"/untrusted\"}")
        harness.calls[0].exit(0,"")
        XCTAssertEqual(run.phase,.failed);XCTAssertEqual(harness.calls.count,1)
        XCTAssertEqual(run.console.entries.last?.level,.error)
        XCTAssertEqual(run.console.entries.last?.message,run.statusMessage)
        XCTAssertEqual(harness.outcomes.count,1);XCTAssertEqual(harness.outcomes[0].failedPhase,.preparing)
        XCTAssertEqual(harness.outcomes[0].request.lease.projectID,owner.projectID)
        XCTAssertNil(harness.outcomes[0].modelManifestSHA256);XCTAssertFalse(owner.isBusy)
        XCTAssertEqual(harness.released,1)
    }
    @MainActor func testRefusedReservationDoesNotLaunchOrReleaseAnotherOperation() throws {
        let(root,owner,runtime) = try setup();defer{try? FileManager.default.removeItem(at:root)}
        let harness = Harness(), run = try session(owner,runtime,root,harness)
        harness.available = false
        XCTAssertFalse(run.start());XCTAssertTrue(harness.calls.isEmpty)
        XCTAssertEqual(harness.acquired,0);XCTAssertEqual(harness.released,0);XCTAssertFalse(owner.isBusy)
    }
    @MainActor func testChangedCapturedSourceFailsBeforeAnyWorkerStarts() async throws {
        let(root,owner,runtime) = try setup();defer{try? FileManager.default.removeItem(at:root)}
        let harness = Harness(), run = try session(owner,runtime,root,harness)
        try Data("a,b,label\n999,999,0\n".utf8).write(to:root.appendingPathComponent("source.csv"))
        XCTAssertTrue(run.start())
        let deadline=Date().addingTimeInterval(5)
        while harness.outcomes.isEmpty && Date()<deadline {try await Task.sleep(for:.milliseconds(1))}
        XCTAssertEqual(run.phase,.failed);XCTAssertTrue(harness.calls.isEmpty);XCTAssertFalse(owner.isBusy)
        XCTAssertEqual(harness.released,harness.acquired)
    }
    @MainActor func testActualIntegerWorkflowArchivesAndExportsMultidimensionalInputs()async throws {
        try await verifyIntegerWorkflowExport(bundledExample:false)
    }
    @MainActor func testBundledEmbeddingExampleTrainsArchivesAndExports()async throws {
        try await verifyIntegerWorkflowExport(bundledExample:true)
    }
    @MainActor private func verifyIntegerWorkflowExport(bundledExample:Bool)async throws {
        guard let development=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {throw XCTSkip("Enable actual workflow runtime")}
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let runtimeRoot=root.appendingPathComponent("runtime")
        for path in ["scripts","Sources/KwipMLCore/Resources"] {
            try FileManager.default.createDirectory(at:runtimeRoot.appendingPathComponent(path),withIntermediateDirectories:true)
        }
        let developmentRoot=URL(fileURLWithPath:development)
        try FileManager.default.createSymbolicLink(at:runtimeRoot.appendingPathComponent(".venv"),withDestinationURL:developmentRoot.appendingPathComponent(".venv"))
        try FileManager.default.copyItem(at:developmentRoot.appendingPathComponent("runner"),to:runtimeRoot.appendingPathComponent("runner"))
        try FileManager.default.copyItem(at:developmentRoot.appendingPathComponent("scripts/runtime_launch.py"),to:runtimeRoot.appendingPathComponent("scripts/runtime_launch.py"))
        let registry=try ComponentRegistry.builtIn()
        try JSONEncoder().encode(registry).write(to:runtimeRoot.appendingPathComponent("Sources/KwipMLCore/Resources/components.json"))
        let runtime=try RuntimeLocation.resolve(resources:nil,developmentRoot:runtimeRoot,allowDevelopment:true)
        var document=fixture()
        let removed:Set<String>=["fit","training","validation"]
        document.nodes.removeAll {$0.address.domain == .model || ($0.address.domain == .data && removed.contains($0.address.localID))}
        document.edges.removeAll {$0.from.address.domain == .model || $0.to.address.domain == .model || removed.contains($0.from.address.localID) || removed.contains($0.to.address.localID)}
        let cast=try XCTUnwrap(document.nodes.firstIndex {$0.address==address(.data,"cast")})
        document.nodes[cast].payload = .dataOperator(operatorID:"kwip.data.columns",config:CurrentExecutionWorkflowTestFixture.columns([("a","int64"),("b","int64"),("label","int64")]))
        document.edges += [edge(.data,"split","training",.control,"mapping","training",.table),edge(.data,"split","validation",.control,"mapping","validation",.table)]
        let nodes=[GraphNode(id:"source",type:"index_input",config:["shape":.array([1,2])]),
            GraphNode(id:"lookup",type:"embedding",config:Dictionary(uniqueKeysWithValues:EmbeddingComponent.definition.fields.map {($0.key,$0.defaultValue)})),
            GraphNode(id:"flat",type:"flatten",config:["start_dim":1,"end_dim":-1]),
            GraphNode(id:"head",type:"linear",config:["in_features":16,"out_features":2,"bias":true]),GraphNode(id:"output",type:"output")]
        document.nodes += nodes.map {.init(address:address(.model,$0.id),payload:.modelPrimitive(type:$0.type,configurationVersion:2,config:$0.config))}
        document.edges += zip(nodes,nodes.dropFirst()).map {edge(.model,$0.id,"out",.model,$1.id,"in",.tensor)}
        document.edges.append(edge(.model,"output","definition",.control,"model","source",.modelDefinition))
        if bundledExample {document=try WorkflowExamples.preparedEmbeddingClassifier(device:"cpu")}
        let library=ProjectLibrary(directory:root.appendingPathComponent("projects"))
        _=try library.saveSource(.workflow(document),id:document.projectID)
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:registry,dataRegistry:dataRegistry())
        let request=try WorkflowExecutionRequest.capture(from:owner,train:address(.control,"train"),runtime:runtime)
        let source=root.appendingPathComponent("indices.csv")
        let rows=(0..<64).map {"\($0%5),\(($0*3)%7),\($0%2)"}
        if bundledExample {
            try FileManager.default.copyItem(at:developmentRoot.appendingPathComponent("Sources/KwipMLApp/Resources/embedding-observations.csv"),to:source)
        } else {try Data((["a,b,label"]+rows).joined(separator:"\n").utf8).write(to:source)}
        let reference=WorkflowHistoryReference(projectID:document.projectID,directoryID:UUID())
        let directory=root.appendingPathComponent("workflow-runs/"+document.projectID+"/"+reference.directoryID.uuidString)
        let harness=Harness(),finished=expectation(description:"Integer training completes")
        var outcome:WorkflowExecutionOutcome?
        let run=WorkflowExecutionSession(request:request,owner:owner,sources:[bundledExample ? WorkflowExamples.sourceAddress : address(.data,"input"):try WorkflowCapturedSource.capture(source)],directory:directory,reservation:harness.reservation,onTerminal:{outcome=$0;finished.fulfill()})
        XCTAssertTrue(run.start());await fulfillment(of:[finished],timeout:45)
        if owner.isBusy {run.cancel()}
        XCTAssertEqual(run.phase,.completed,"\(run.statusMessage) / \(outcome?.recordError ?? "")")
        let predictions = try XCTUnwrap(run.predictionPreview)
        XCTAssertEqual(predictions.dtype, "int64")
        XCTAssertEqual(predictions.sampleShape, bundledExample ? [2] : [1,2])
        XCTAssertFalse(predictions.rows.isEmpty)
        XCTAssertTrue(predictions.rows.allSatisfy { $0.inputs.count == 2 })
        guard case .verified(let saved)=WorkflowHistoryReader.load(reference,root:root) else {return XCTFail("Integer history did not verify")}
        XCTAssertEqual(saved.modelContent?.predictionPreview, predictions)
        guard case .object(let savedActivations) = saved.modelContent?.activationPreviews else { return XCTFail("Missing saved activation samples") }
        XCTAssertEqual(Set(savedActivations.keys),Set(request.plan.model.graph.nodes.map(\.id)))
        let activationFile = directory.appendingPathComponent("training/artifacts/activations.json")
        let activationBytes = try Data(contentsOf:activationFile)
        try Data("{}".utf8).write(to:activationFile)
        if case .verified = WorkflowHistoryReader.load(reference,root:root) { XCTFail("Corrupt activation artifact must invalidate archived evidence") }
        try activationBytes.write(to:activationFile)

        let predictionFile = directory.appendingPathComponent("training/artifacts/predictions.json")
        let predictionBytes = try Data(contentsOf: predictionFile)
        try Data("{}".utf8).write(to: predictionFile)
        if case .verified = WorkflowHistoryReader.load(reference,root:root) { XCTFail("Corrupt prediction artifact must invalidate archived evidence") }
        try predictionBytes.write(to: predictionFile)
        XCTAssertEqual(saved.plan.featureColumns,bundledExample ? ["signal","context"] : ["a","b"])
        let package=try WorkflowHistoryReader.stageTrainedExport(reference,root:root)
        defer {try? FileManager.default.removeItem(at:package.directory)}
        let contract=try JSONDecoder().decode([String:JSONValue].self,from:Data(contentsOf:package.directory.appendingPathComponent("input-contract.json")))
        guard case .object(let input)=contract["input"] else {return XCTFail("Missing input contract")}
        XCTAssertEqual(input["dtype"],.string("int64"));XCTAssertEqual(input["shape"],.array(bundledExample ? [.string("N"),2] : [.string("N"),1,2]))
        XCTAssertEqual(package.inputContract.input,input)
        XCTAssertTrue(package.inputContract.reviewSummary.contains("int64"))
        let readme=try String(contentsOf:package.directory.appendingPathComponent("README.md"),encoding:.utf8)
        XCTAssertTrue(readme.contains("CPU int64"));XCTAssertTrue(readme.contains(bundledExample ? "[N, 2]" : "[N, 1, 2]"))
        XCTAssertFalse(readme.contains(root.path))
        // A staged export must remain usable without its originating archive or CSV.
        try FileManager.default.removeItem(at:directory)
        try FileManager.default.removeItem(at:source)
        let process=Process(),pipe=Pipe();process.executableURL=runtime.executableURL
        process.arguments=[package.directory.appendingPathComponent("inference.py").path,"--example"]
        process.standardOutput=pipe;process.standardError=pipe
        try process.run();let bytes=pipe.fileHandleForReading.readDataToEndOfFile();process.waitUntilExit()
        XCTAssertEqual(process.terminationStatus,0,String(decoding:bytes,as:UTF8.self))
        let result=try XCTUnwrap(JSONSerialization.jsonObject(with:bytes) as? [String:Any])
        XCTAssertEqual((result["logits"] as? [[Double]])?.count,2)
    }
    @MainActor func testRealSessionRetainsTenThousandEpochEventsWhileCoalescingLivePresentation() async throws {
        guard let development = ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else { throw XCTSkip("Requires actual preparation worker") }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        var document = fixture(block: true)
        let setupIndex = try XCTUnwrap(document.nodes.firstIndex { if case .runSetup = $0.payload { return true }; return false })
        guard case .runSetup(var config) = document.nodes[setupIndex].payload else { return XCTFail("Missing setup") }
        config["epochs"] = .integer(10_000); document.nodes[setupIndex].payload = .runSetup(config: config)
        let library = ProjectLibrary(directory: root.appendingPathComponent("projects"))
        _ = try library.saveSource(.workflow(document), id: document.projectID)
        let runtime = try RuntimeLocation.resolve(resources: nil, developmentRoot: URL(fileURLWithPath: development), allowDevelopment: true)
        let owner = try WorkflowAuthoringStore(library: library, projectID: document.projectID,
            modelRegistry: .builtIn(), dataRegistry: dataRegistry())
        let request = try WorkflowExecutionRequest.capture(from: owner, train: address(.control, "train"), runtime: runtime)
        let source = root.appendingPathComponent("observations.csv")
        try Data((["a,b,label"] + (0..<64).map { "\($0),\(($0 * 7) % 13),\($0 % 2)" }).joined(separator: "\n").utf8).write(to: source)
        let harness = Harness(), ready = expectation(description: "Real preparation verified; training callback installed")
        var trainingLine: ((String) -> Void)?, trainingExit: ((Int32, String) -> Void)?
        let launcher: WorkflowWorkerLauncher = { executable, arguments, directory, line, exit in
            if arguments.contains("--graph") {
                trainingLine = line; trainingExit = exit; ready.fulfill()
                return .init(cancel: {})
            }
            return try WorkflowExecutionSession.launchWorker(executable, arguments, directory, line, exit)
        }
        var outcome: WorkflowExecutionOutcome?
        let run = WorkflowExecutionSession(request: request, owner: owner,
            sources: [address(.data, "input"): try WorkflowCapturedSource.capture(source)],
            directory: root.appendingPathComponent(UUID().uuidString), reservation: harness.reservation,
            launch: launcher, onTerminal: { outcome = $0 })
        var publications = 0
        let observation = run.$liveMetrics.dropFirst().sink { _ in publications += 1 }
        defer { observation.cancel() }
        XCTAssertTrue(run.start())
        await fulfillment(of: [ready], timeout: 20)
        let emit = try XCTUnwrap(trainingLine)
        // Controlled protocol fixture, not claimed as measured model performance.
        func epoch(_ value: Int) {
            emit("{\"event\":\"epoch\",\"epoch\":\(value),\"trainLoss\":\(value).0,\"validationLoss\":0.5,\"accuracy\":0.75}")
        }
        epoch(1)
        try await Task.sleep(for: .milliseconds(250))
        XCTAssertEqual(run.liveMetrics.last?.epoch, 1)
        let firstPublications = publications
        for value in 2...49 { epoch(value) }
        try await Task.sleep(for: .milliseconds(250))
        XCTAssertEqual(run.metrics.count, 49)
        XCTAssertEqual(run.liveMetrics.last?.epoch, 1, "Ineligible progress must not redraw the live chart")
        XCTAssertEqual(publications, firstPublications)
        epoch(50)
        // Arrivals after the eligible point must not leak into its delayed snapshot.
        epoch(51)
        try await Task.sleep(for: .milliseconds(250))
        XCTAssertEqual(run.liveMetrics.last?.epoch, 50)
        XCTAssertEqual(run.metrics.count, 51)
        let burstPublications = publications
        for value in 52...10_000 { epoch(value) }
        XCTAssertEqual(run.metrics.map(\.epoch), Array(1...10_000))
        XCTAssertEqual(run.metrics.map(\.trainLoss), (1...10_000).map(Double.init))
        try await Task.sleep(for: .milliseconds(250))
        XCTAssertLessThanOrEqual(publications - burstPublications, 2)
        XCTAssertEqual(run.liveMetrics.count, 10_000)
        emit("{\"event\":\"error\",\"message\":\"Controlled terminal failure\"}")
        try XCTUnwrap(trainingExit)(1, "Controlled terminal failure")
        XCTAssertEqual(run.phase, .failed)
        XCTAssertEqual(outcome?.error, "Controlled terminal failure")
        XCTAssertNil(outcome?.modelManifestSHA256)
        XCTAssertEqual(run.liveMetrics.map(\.epoch), run.metrics.map(\.epoch))
        let finalCount = publications
        emit("{\"event\":\"epoch\",\"epoch\":10001,\"trainLoss\":1.0,\"validationLoss\":1.0,\"accuracy\":0.5}")
        try await Task.sleep(for: .milliseconds(250))
        XCTAssertEqual(publications, finalCount)
        XCTAssertEqual(run.metrics.count, 10_000)
        XCTAssertFalse(owner.isBusy)
    }

    @MainActor func testOptInActualPreparationToTrainingPublishesMeasuredArtifacts() async throws {
        guard let development = ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {
            throw XCTSkip("Set KWIPML_TEST_DEVELOPMENT_ROOT for actual preparation-to-training execution.")
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let library = ProjectLibrary(directory:root.appendingPathComponent("projects")), document = fixture(block:true)
        _ = try library.saveSource(.workflow(document),id:document.projectID)
        let runtime = try RuntimeLocation.resolve(resources:nil,developmentRoot:URL(fileURLWithPath:development),allowDevelopment:true)
        let model = try JSONDecoder().decode(ComponentRegistry.self,from:Data(contentsOf:runtime.registryURL))
        let data = try DataFlowOperatorRegistry.load(from:runtime.workingDirectoryURL.appendingPathComponent("runner/dataflow-operators.json"))
        let owner = try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:model,dataRegistry:data)
        let request = try WorkflowExecutionRequest.capture(from:owner,train:address(.control,"train"),runtime:runtime)
        let source = root.appendingPathComponent("observations.csv")
        let rows = (0..<64).map { "\($0),\(($0 * 7) % 13),\($0 % 2)" }
        try Data((["a,b,label"]+rows).joined(separator:"\n").utf8).write(to:source)
        let harness = Harness(), finished = expectation(description:"Measured preparation and training terminate")
        var outcome:WorkflowExecutionOutcome?
        let run = WorkflowExecutionSession(request:request,owner:owner,
            sources:[address(.data,"input"):try WorkflowCapturedSource.capture(source)],directory:root.appendingPathComponent("10000000-0000-0000-0000-000000000001"),
            reservation:harness.reservation,onTerminal:{outcome=$0;finished.fulfill()})
        var observedProgress:[WorkflowCanvasProgress]=[]
        let progressObservation=run.$canvasProgress.sink {observedProgress.append($0)}
        defer {progressObservation.cancel()}
        XCTAssertTrue(run.start())
        await fulfillment(of:[finished],timeout:45)
        if owner.isBusy {run.cancel()}
        XCTAssertEqual(run.phase,.completed, "Session status: \(run.statusMessage); outcome: \(outcome?.error ?? "none"); record: \(outcome?.recordError ?? "none")")
        XCTAssertTrue(observedProgress.contains {$0.stage == .preparing && $0.activeNodeAddresses == [self.address(.data,"cast")]})
        XCTAssertTrue(observedProgress.contains {$0.stage == .training && $0.activeNodeAddresses == [request.train]})
        XCTAssertTrue(observedProgress.contains {$0.stage == .verifying && $0.activeNodeAddresses == [request.train]})
        XCTAssertTrue(run.canvasProgress.activeNodeAddresses.isEmpty)
        XCTAssertEqual(run.canvasProgress.completed.count,request.plan.dataflow.nodes.count)
        XCTAssertEqual(run.predictionPreview?.dtype, "float32")
        XCTAssertEqual(run.predictionPreview?.sampleShape, [2])
        let captures = try XCTUnwrap(run.modelCaptures)
        for origin in request.plan.model.origins.values {
            let ports = try XCTUnwrap(captures.ports(instancePath:origin.instancePath,localNodeID:origin.localNodeID))
            guard case .object(let tensor) = ports["out"] else { return XCTFail("Missing verified activation") }
            XCTAssertEqual(tensor["dtype"],.string("float32"))
        }
        XCTAssertNil(captures.ports(instancePath:["unrelated"],localNodeID:"input"))

        XCTAssertEqual(run.metrics.count,3)
        XCTAssertEqual(run.liveMetrics.map(\.epoch), run.metrics.map(\.epoch), "Terminal publication must flush every exact observation")
        XCTAssertEqual(run.liveMetrics.map(\.trainLoss), run.metrics.map(\.trainLoss))
        XCTAssertEqual(run.liveMetrics.map(\.validationLoss), run.metrics.map(\.validationLoss))
        let publishedFinal = run.liveMetrics.map(\.epoch)
        try await Task.sleep(for: .milliseconds(250))
        XCTAssertEqual(run.liveMetrics.map(\.epoch), publishedFinal, "No delayed presentation may replace terminal evidence")
        XCTAssertTrue(run.metrics.allSatisfy { $0.trainLoss.isFinite && $0.validationLoss.isFinite && $0.accuracy.map(\.isFinite) == true })
        let result = try XCTUnwrap(outcome)
        XCTAssertNotNil(result.preparationManifestSHA256);XCTAssertNotNil(result.modelManifestSHA256)
        XCTAssertEqual(result.request.document,document);XCTAssertEqual(result.request.plan.featureColumns,["a","b"])
        XCTAssertEqual(harness.acquired,1);XCTAssertEqual(harness.released,1);XCTAssertFalse(owner.isBusy)
        let training = root.appendingPathComponent("10000000-0000-0000-0000-000000000001/training/artifacts")
        for filename in ["manifest.json","checkpoint.pt","model.py","metrics.json"] {
            XCTAssertTrue(FileManager.default.fileExists(atPath:training.appendingPathComponent(filename).path),filename)
        }
        let manifest = try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:training.appendingPathComponent("manifest.json"))) as? [String:Any])
        XCTAssertEqual(manifest["status"] as? String,"completed")
        XCTAssertEqual((manifest["settings"] as? [String:Any])?["partitionMode"] as? String,"prepared")
        XCTAssertEqual((manifest["settings"] as? [String:Any])?["epochs"] as? Int,3)
        let recorded = try JSONDecoder().decode(GraphDocument.self,from:JSONSerialization.data(withJSONObject:try XCTUnwrap(manifest["graph"])))
        XCTAssertEqual(recorded.nodes,request.plan.model.graph.nodes)
        XCTAssertEqual(recorded.edges,request.plan.model.graph.edges)
        XCTAssertNotNil((manifest["provenance"] as? [String:Any])?["composite"])
        let provenance = try XCTUnwrap(manifest["provenance"] as? [String:Any])
        XCTAssertEqual(provenance["sourceHash"] as? String,try WorkflowCapturedSource.capture(source).sha256)
        // Each balanced 32-row stratum allocates 19/7/6 at 60/20/20:
        // the largest-remainder tie is awarded to Validation before Test.
        XCTAssertEqual(provenance["trainingSamples"] as? Int,38)
        XCTAssertEqual(provenance["validationSamples"] as? Int,14)
        XCTAssertGreaterThan(try Data(contentsOf:training.appendingPathComponent("checkpoint.pt")).count,0)
        let prepared = try XCTUnwrap(provenance["preparation"] as? [String:Any])
        XCTAssertEqual(prepared["manifestSHA256"] as? String,result.preparationManifestSHA256)
        // A History caller must anchor untrusted run children beneath its trusted
        // workspace. Replacing the run directory by an identical-content symlink
        // must fail descriptor traversal, not pass an earlier path precheck.
        let runDirectory = training.deletingLastPathComponent().deletingLastPathComponent()
        let lifecycleExpectation=WorkflowPhaseExpectation(plan:request.plan,modelRegistrySHA256:request.modelRegistrySHA256,
            dataRegistrySHA256:request.dataRegistrySHA256,compilerContractVersion:3)
        let verifiedPreparation=try WorkflowPhaseArtifacts.preparation(expectation:lifecycleExpectation,root:runDirectory)
        let preparationRoot=runDirectory.appendingPathComponent("preparation/artifacts")
        let preparationBytes=try Data(contentsOf:preparationRoot.appendingPathComponent("manifest.json"))
        XCTAssertEqual(WorkflowPhaseArtifacts.hash(preparationBytes),result.preparationManifestSHA256)
        let preparationManifest=try XCTUnwrap(JSONSerialization.jsonObject(with:preparationBytes) as? [String:Any])
        let entries=try XCTUnwrap(preparationManifest["artifacts"] as? [[String:Any]])
        func identities(_ node:String,_ port:String,_ role:String)throws->[String] {
            let entry=try XCTUnwrap(entries.first {$0["nodeID"] as? String == node && $0["port"] as? String == port && $0["kind"] as? String == "table"})
            let bytes=try Data(contentsOf:preparationRoot.appendingPathComponent(try XCTUnwrap(entry["path"] as? String)))
            XCTAssertEqual(WorkflowPhaseArtifacts.hash(bytes),entry["sha256"] as? String)
            let table=try XCTUnwrap(JSONSerialization.jsonObject(with:bytes) as? [String:Any])
            let lineage=try XCTUnwrap(table["lineage"] as? [String:Any])
            XCTAssertEqual(lineage["role"] as? String,role)
            let ids=try XCTUnwrap(lineage["observationIDs"] as? [String])
            XCTAssertEqual(ids.count,try XCTUnwrap(table["rows"] as? [[Any]]).count)
            XCTAssertEqual(Set(ids).count,ids.count)
            return ids
        }
        let sourceIDs=try identities("input","table","unassigned")
        let trainingIDs=try identities("split","training","training")
        let validationIDs=try identities("split","validation","validation")
        let testIDs=try identities("split","test","test")
        XCTAssertEqual(sourceIDs.count,64)
        XCTAssertEqual(trainingIDs.count,38);XCTAssertEqual(validationIDs.count,14);XCTAssertEqual(testIDs.count,12)
        XCTAssertTrue(Set(trainingIDs).isDisjoint(with:validationIDs))
        XCTAssertTrue(Set(trainingIDs).isDisjoint(with:testIDs))
        XCTAssertTrue(Set(validationIDs).isDisjoint(with:testIDs))
        XCTAssertEqual(Set(trainingIDs+validationIDs+testIDs),Set(sourceIDs))
        XCTAssertEqual(provenance["trainingRowIDs"] as? [String],trainingIDs)
        XCTAssertEqual(provenance["validationRowIDs"] as? [String],validationIDs)
        XCTAssertNoThrow(try WorkflowPhaseArtifacts.training(expectation:lifecycleExpectation,root:runDirectory,preparation:verifiedPreparation.selection))
        let manifestURL=training.appendingPathComponent("manifest.json"),originalManifest=try Data(contentsOf:manifestURL)
        var changedManifest=manifest
        var changedLineage=try XCTUnwrap(changedManifest["lineage"] as? [String:Any])
        changedLineage["mode"]="weightsOnlyFineTune";changedManifest["lineage"]=changedLineage
        try JSONSerialization.data(withJSONObject:changedManifest).write(to:manifestURL)
        XCTAssertThrowsError(try WorkflowPhaseArtifacts.training(expectation:lifecycleExpectation,root:runDirectory,preparation:verifiedPreparation.selection))
        try originalManifest.write(to:manifestURL)
        let moved = root.appendingPathComponent("moved-completed-run")
        try FileManager.default.moveItem(at:runDirectory,to:moved)
        try FileManager.default.createSymbolicLink(at:runDirectory,withDestinationURL:moved)
        let expectation = WorkflowPhaseExpectation(request:request)
        XCTAssertNoThrow(try WorkflowPhaseArtifacts.preparation(expectation:expectation,root:runDirectory), "Explicitly trusted root aliases retain existing API behavior")
        XCTAssertThrowsError(try WorkflowPhaseArtifacts.preparation(expectation:expectation,root:runDirectory,
            storageRoot:root,storageParts:[runDirectory.lastPathComponent]))
        let selected = PreparedTrainingSelection(manifestPath:runDirectory.appendingPathComponent("preparation/artifacts/manifest.json").path,
            manifestSHA256:try XCTUnwrap(result.preparationManifestSHA256),graphDigest:try XCTUnwrap(prepared["graphDigest"] as? String),
            trainingNode:request.plan.training.node,trainingPort:request.plan.training.port,
            validationNode:request.plan.validation.node,validationPort:request.plan.validation.port,
            featureColumns:request.plan.featureColumns,targetColumn:try XCTUnwrap(request.plan.targetColumn))
        XCTAssertThrowsError(try WorkflowPhaseArtifacts.training(expectation:expectation,root:runDirectory,preparation:selected,
            storageRoot:root,storageParts:[runDirectory.lastPathComponent]))
    }

    func testReportedOutputAliasesOnlyMatchTheKnownPhysicalDirectory() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at:root) }
        let expected = root.appendingPathComponent("training/artifacts")
        try FileManager.default.createDirectory(at:expected,withIntermediateDirectories:true)
        let alias = root.appendingPathComponent("alias")
        try FileManager.default.createSymbolicLink(at:alias,withDestinationURL:expected)
        XCTAssertTrue(WorkflowPhaseArtifacts.matchesOutputPath(expected.resolvingSymlinksInPath().path,expected:expected))
        XCTAssertTrue(WorkflowPhaseArtifacts.matchesOutputPath(alias.path,expected:expected))
        XCTAssertFalse(WorkflowPhaseArtifacts.matchesOutputPath(root.appendingPathComponent("other/artifacts").path,expected:expected))
        XCTAssertFalse(WorkflowPhaseArtifacts.matchesOutputPath("training/artifacts",expected:expected))
        XCTAssertFalse(WorkflowPhaseArtifacts.matchesOutputPath(nil,expected:expected))
        XCTAssertFalse(WorkflowPhaseArtifacts.matchesOutputPath(expected.path + "\0",expected:expected))
    }

    @MainActor func testAcceptedSessionOwnsItsLifetimeUntilTerminalReleasesReservation() async throws {
        let(root,owner,runtime) = try setup();defer{try? FileManager.default.removeItem(at:root)}
        let harness = Harness()
        var run:WorkflowExecutionSession? = try session(owner,runtime,root,harness)
        weak var retained = run
        XCTAssertTrue(try XCTUnwrap(run).start())
        run = nil
        XCTAssertNotNil(retained,"Accepted resource owner must survive loss of its UI reference")
        XCTAssertTrue(owner.isBusy);XCTAssertEqual(harness.released,0)
        try await awaitWorker(harness)
        harness.calls[0].exit(1,"intentional worker failure")
        XCTAssertEqual(harness.outcomes.count,1);XCTAssertEqual(harness.released,1)
        XCTAssertFalse(owner.isBusy)
        XCTAssertNil(retained,"Terminal callback must release the session's self-retention")
        harness.calls[0].exit(1,"duplicate exit")
        XCTAssertEqual(harness.outcomes.count,1);XCTAssertEqual(harness.released,1)
    }

    @MainActor func testStaleOwnerOrChangedRuntimeCannotStartCapturedRequest() throws {
        let(root,owner,runtime) = try setup();defer{try? FileManager.default.removeItem(at:root)}
        let harness = Harness(), stale = try session(owner,runtime,root,harness)
        XCTAssertNotNil(owner.apply(.rename(address:address(.data,"input"),name:"Changed"),lease:owner.lease()))
        XCTAssertFalse(stale.start());XCTAssertTrue(harness.calls.isEmpty)
        let current = try session(owner,runtime,root,harness)
        try Data("changed implementation".utf8).write(to:runtime.workingDirectoryURL.appendingPathComponent("runner/engine.py"))
        XCTAssertFalse(current.start());XCTAssertTrue(harness.calls.isEmpty);XCTAssertFalse(owner.isBusy)
    }
    @MainActor func testPerformanceCapturePublishesVerifiedColdHistoryAndRefusesTamperedMissingOrUnrequestedEvidence() async throws {
        guard let development = ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {
            throw XCTSkip("Set KWIPML_TEST_DEVELOPMENT_ROOT for real performance capture.")
        }
        let (temporaryRoot, owner, _) = try setup()
        let canonical = try XCTUnwrap(realpath(temporaryRoot.path, nil))
        let root = URL(fileURLWithPath: String(cString: canonical)); free(canonical)
        defer { try? FileManager.default.removeItem(at: root) }
        let runtime = try RuntimeLocation.resolve(resources: nil, developmentRoot: URL(fileURLWithPath: development), allowDevelopment: true)
        let source = root.appendingPathComponent("source.csv")
        let rows = (0..<64).map { "\($0),\(($0 * 7) % 13),\($0 % 2)" }
        try Data((["a,b,label"] + rows).joined(separator: "\n").utf8).write(to: source)
        var baseline: [EpochMetric]?
        for enabled in [false, true] {
            let capture = TrainingSampleCapturePolicy(activationValues: false, predictionRows: false, performanceMeasurements: enabled)
            let request = try WorkflowExecutionRequest.capture(from: owner, train: address(.control,"train"), runtime: runtime,
                executionBudget: ExecutionPreferences.factory.resolve().budget, sampleCapture: capture)
            let id = UUID(), directory = root.appendingPathComponent("workflow-runs/\(owner.projectID)/\(id.uuidString)")
            let harness = Harness(), done = expectation(description: "actual performance \(enabled)")
            let run = WorkflowExecutionSession(request: request, owner: owner, sources: [address(.data,"input"): try .capture(source)],
                directory: directory, reservation: harness.reservation, onTerminal: { _ in done.fulfill() })
            XCTAssertTrue(run.start()); await fulfillment(of: [done], timeout: 45)
            if owner.isBusy { run.cancel() }
            XCTAssertEqual(run.phase, .completed, run.statusMessage)
            let reference = WorkflowHistoryReference(projectID: owner.projectID, directoryID: id)
            guard case .verified(let saved) = WorkflowHistoryReader.load(reference, root: root) else { return XCTFail("Cold performance History refused") }
            if let baseline {
                XCTAssertEqual(run.metrics.map(\.epoch), baseline.map(\.epoch))
                XCTAssertEqual(run.metrics.map(\.trainLoss), baseline.map(\.trainLoss))
                XCTAssertEqual(run.metrics.map(\.validationLoss), baseline.map(\.validationLoss))
                XCTAssertEqual(run.metrics.map(\.accuracy), baseline.map(\.accuracy))
            } else { baseline = run.metrics }
            let file = directory.appendingPathComponent("training/artifacts/performance.json")
            if enabled {
                let profile = try XCTUnwrap(saved.modelContent?.performanceCapture)
                XCTAssertEqual(profile, run.performanceCapture)
                XCTAssertEqual(profile.records.count, request.plan.model.graph.nodes.count)
                XCTAssertTrue(profile.records.allSatisfy { $0.durationNanoseconds != nil })
                XCTAssertTrue(saved.outcome.artifacts.contains { $0.path == "training/artifacts/performance.json" })
                let original = try Data(contentsOf: file)
                try Data("{}".utf8).write(to: file)
                if case .verified = WorkflowHistoryReader.load(reference, root: root) { XCTFail("Tampered performance must refuse") }
                try FileManager.default.removeItem(at: file)
                if case .verified = WorkflowHistoryReader.load(reference, root: root) { XCTFail("Missing performance must refuse") }
                try original.write(to: file)
            } else {
                XCTAssertNil(saved.modelContent?.performanceCapture); XCTAssertNil(run.performanceCapture)
                XCTAssertFalse(FileManager.default.fileExists(atPath: file.path))
                try Data("{}".utf8).write(to: file)
                if case .verified = WorkflowHistoryReader.load(reference, root: root) { XCTFail("Unrequested performance file must refuse") }
                try FileManager.default.removeItem(at: file)
            }
            guard case .verified = WorkflowHistoryReader.load(reference, root: root) else { return XCTFail("Restored baseline must verify") }
        }
    }

    @MainActor func testHistoryOffActuallyOmitsOptionalArtifactsAndColdModelStillVerifies()async throws {
        guard let development=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {throw XCTSkip("Set development runtime for real retention.")}
        let (temporary,owner,_)=try setup();let canonical=try XCTUnwrap(realpath(temporary.path,nil))
        let root=URL(fileURLWithPath:String(cString:canonical));free(canonical);defer {try? FileManager.default.removeItem(at:root)}
        try owner.configureCanvasVersions(root:root);try owner.setRunHistoryPolicy(.none)
        let runtime=try RuntimeLocation.resolve(resources:nil,developmentRoot:URL(fileURLWithPath:development),allowDevelopment:true)
        let source=root.appendingPathComponent("source.csv")
        try Data((["a,b,label"]+(0..<64).map {"\($0),\(($0*7)%13),\($0%2)"}).joined(separator:"\n").utf8).write(to:source)
        let request=try WorkflowExecutionRequest.capture(from:owner,train:address(.control,"train"),runtime:runtime,executionBudget:ExecutionPreferences.factory.resolve().budget,sampleCapture:.init(activationValues:true,predictionRows:true,performanceMeasurements:true))
        XCTAssertEqual(request.sampleCapture?.savesComponentOutputs,false);XCTAssertEqual(request.sampleCapture?.savesDiagnosticDetails,false)
        let inherited=try WorkflowExecutionRequest.capture(from:owner,train:address(.control,"train"),runtime:runtime)
        XCTAssertEqual(inherited.sampleCapture?.savesComponentOutputs,false);XCTAssertEqual(inherited.sampleCapture?.savesDiagnosticDetails,false)
        // Future preference changes cannot alter an accepted request or its immutable decision.
        try owner.setRunHistoryPolicy(.all)
        let id=UUID(),directory=root.appendingPathComponent("workflow-runs/\(owner.projectID)/\(id.uuidString)")
        let harness=Harness(),done=expectation(description:"real capture-off")
        let run=WorkflowExecutionSession(request:request,owner:owner,sources:[address(.data,"input"):try .capture(source)],directory:directory,reservation:harness.reservation,onTerminal:{_ in done.fulfill()})
        XCTAssertTrue(run.start());await fulfillment(of:[done],timeout:45)
        if owner.isBusy {run.cancel()};XCTAssertEqual(run.phase,.completed,run.statusMessage)
        let reference=WorkflowHistoryReference(projectID:owner.projectID,directoryID:id)
        guard case .verified(let saved)=WorkflowHistoryReader.load(reference,root:root) else {return XCTFail("Capture-off cold model failed verification")}
        XCTAssertFalse(try RunHistoryPolicyStore(root:root,projectID:owner.projectID).decision(id:saved.outcome.runID).enabled)
        XCTAssertNotNil(saved.modelContent);XCTAssertFalse(run.metrics.isEmpty)
        XCTAssertNil(saved.modelContent?.performanceCapture);XCTAssertNil(run.performanceCapture)
        for name in ["activations.json","predictions.json","performance.json","gradient-summary.json","derivative-results.json"] {
            XCTAssertFalse(FileManager.default.fileExists(atPath:directory.appendingPathComponent("training/artifacts/"+name).path),name)
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath:directory.appendingPathComponent("preparation/artifacts/previews.json").path))
        // Unrequested files are refused, even if not mentioned in the optional manifest fields.
        try Data("{}".utf8).write(to:directory.appendingPathComponent("training/artifacts/activations.json"))
        if case .verified=WorkflowHistoryReader.load(reference,root:root) {XCTFail("Capture-off cannot admit unexpected samples")}
    }

}
