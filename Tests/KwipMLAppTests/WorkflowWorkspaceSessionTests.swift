import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp

final class WorkflowWorkspaceSessionTests: XCTestCase {
    private func address(_ domain:WorkflowDomain,_ id:String)->WorkflowNodeAddress {.init(domain:domain,localID:id)}
    private func edge(_ a:WorkflowDomain,_ id:String,_ port:String,_ b:WorkflowDomain,_ target:String,_ input:String,_ kind:WorkflowEdgeKind)->WorkflowEdge {
        .init(id:"\(a)-\(id)-\(port)-\(b)-\(target)-\(input)",from:.init(address:address(a,id),port:port),to:.init(address:address(b,target),port:input),kind:kind)
    }
    /// New session captures use current operators; archived legacy integrity has dedicated fixtures.
    private func fixture(block:Bool=false)->WorkflowDocument {CurrentExecutionWorkflowTestFixture.make(block:block)}

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
    private var repo:URL {URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()}
    private func dataRegistry() throws -> DataFlowOperatorRegistry {try .load(from:repo.appendingPathComponent("runner/dataflow-operators.json"))}
    @MainActor private struct Fixture {
        let root:URL
        let workspace:WorkspaceStore
        let owner:WorkflowAuthoringStore
        let host:WorkflowWorkspaceSession
        let csv:URL
    }
    @MainActor private func setup() throws -> Fixture {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let runtime = try makeRuntime(root.appendingPathComponent("runtime"))
        let workspace = WorkspaceStore(rootOverride:runtime.workingDirectoryURL,storageOverride:root.appendingPathComponent("storage"),detectCapabilities:false)
        // This injected worker never executes discovery; install its explicit reviewed fixture contract.
        let contract = try BuiltInCapabilityBridge.executionContract(registry: workspace.registry, runtimeVersion: "2.14.1", availableDevices: [.cpu])
        try workspace.installDiscoveredRuntimeContract(version: "2.14.1", expectedVersion: nil, devices: ["cpu"],
            descriptor: JSONEncoder().encode(contract), tensorCompositionDescriptors: fixtureReviewedCompositionDescriptors())
        let document = fixture(block:true)
        _ = try workspace.projectLibrary.saveSource(.workflow(document),id:document.projectID)
        let owner = try WorkflowAuthoringStore(library:workspace.projectLibrary,projectID:document.projectID,modelRegistry:.builtIn(),dataRegistry:dataRegistry())
        workspace.activeProjectID = document.projectID
        let host = try WorkflowWorkspaceSession(workspace:workspace,owner:owner)
        let csv = root.appendingPathComponent("observations.csv")
        try Data("a,b,label\n1,2,0\n2,3,1\n3,4,0\n4,5,1\n".utf8).write(to:csv)
        XCTAssertTrue(host.bindings.setPath(csv.path,for:address(.data,"input"),lease:owner.lease(),expectedRevision:0))
        return Fixture(root:root,workspace:workspace,owner:owner,host:host,csv:csv)
    }
    @MainActor func testUnactivatedWorkflowRefusesRunExperimentAndEvaluationBeforeCapture() throws {
        let f=try setup();defer{try? FileManager.default.removeItem(at:f.root)}
        f.workspace.workflowSession=f.host
        f.workspace.activeProjectID=f.owner.projectID
        f.workspace.executionAdmission={throw NSError(domain:"test.execution",code:1,userInfo:[NSLocalizedDescriptionKey:"Execution unavailable"])}
        let evaluation=address(.control,"unactivated-evaluate")
        XCTAssertNotNil(f.owner.apply(.insert(.init(address:evaluation,payload:.evaluateTest(config:[:]))),lease:f.owner.lease()))
        let saved=address(.control,"unactivated-saved")
        XCTAssertNotNil(f.owner.apply(.insert(.init(address:saved,payload:.frozenModel(config:[:]))),lease:f.owner.lease()))
        XCTAssertNotNil(f.owner.apply(.connect(from:.init(address:saved,port:"model"),to:.init(address:evaluation,port:"model")),lease:f.owner.lease()))
        let before=f.owner.document
        f.workspace.run()
        XCTAssertTrue(f.host.statusMessage.contains("Execution unavailable"), "The toolbar must show the activation recovery in the workflow status")
        XCTAssertFalse(f.host.run());XCTAssertTrue(f.host.statusMessage.contains("Execution unavailable"))
        XCTAssertFalse(f.host.runExperiments());XCTAssertTrue(f.host.statusMessage.contains("Execution unavailable"))
        XCTAssertFalse(f.host.evaluate(address:evaluation))
        XCTAssertTrue(f.host.statusMessage.contains("Execution unavailable"))
        XCTAssertFalse(f.workspace.isBusy);XCTAssertFalse(f.host.isStagingEvaluation)
        XCTAssertEqual(f.owner.document,before)
        let exported=f.root.appendingPathComponent("workflow.kwipml")
        XCTAssertNoThrow(try f.workspace.exportModelProject(to:exported))
        XCTAssertTrue(FileManager.default.fileExists(atPath:exported.path))
    }
    @MainActor private func awaitBoundary(_ ready:()->Bool) async throws {
        let deadline=Date().addingTimeInterval(5)
        while !ready() && Date()<deadline {try await Task.sleep(for:.milliseconds(1))}
        guard ready() else {throw NSError(domain:"WorkspaceWorkerTimeout",code:1)}
    }
    @MainActor func testRunUsesCanonicalRequestAndRetainsGlobalReservationThroughCancellationExit() async throws {
        let f = try setup();defer{try? FileManager.default.removeItem(at:f.root)}
        let original = f.owner.document, legacy = f.workspace.graph
        var calls = 0,cancels = 0,exit:((Int32,String)->Void)?,arguments:[String] = []
        f.host.executionLaunch = {_,args,_,_,finish in calls += 1;arguments=args;exit=finish;return .init(cancel:{cancels += 1})}
        XCTAssertTrue(f.host.run());try await awaitBoundary {exit != nil}
        XCTAssertEqual(calls,1);XCTAssertTrue(arguments.contains("--kwip-operation-supervise"))
        XCTAssertEqual(f.host.statusMessage,f.host.execution?.statusMessage)
        XCTAssertEqual(f.host.statusMessage,"Preparing captured sources…")
        XCTAssertTrue(f.workspace.isPreparingData);XCTAssertTrue(f.workspace.isBusy);XCTAssertTrue(f.owner.isBusy)
        XCTAssertEqual(f.host.execution?.request.document,original)
        XCTAssertEqual(f.workspace.graph,legacy,"A workflow never installs an expanded mutable graph")
        f.workspace.executionAdmission={throw NSError(domain:"test.execution",code:1,userInfo:[NSLocalizedDescriptionKey:"Execution unavailable"])}
        XCTAssertFalse(f.host.run(),"A later denied start cannot cancel the captured operation")
        XCTAssertEqual(cancels,0);XCTAssertTrue(f.workspace.isBusy);XCTAssertTrue(f.owner.isBusy)
        XCTAssertTrue(f.host.outcomes.isEmpty)
        f.workspace.cancelDataPreparation?()
        XCTAssertEqual(cancels,1);XCTAssertTrue(f.workspace.isPreparingData);XCTAssertTrue(f.host.outcomes.isEmpty)
        XCTAssertEqual(f.host.statusMessage,"Cancelling workflow execution…")
        exit?(0,"")
        XCTAssertFalse(f.workspace.isPreparingData);XCTAssertNil(f.workspace.cancelDataPreparation)
        XCTAssertFalse(f.owner.isBusy);XCTAssertEqual(f.host.outcomes.count,1)
        XCTAssertEqual(f.host.outcomes.first?.status,.cancelled)
        XCTAssertEqual(f.host.outcomes.first?.request.document,original)
    }
    @MainActor func testCapturedPreviewRetainsCancelledPhaseAndRejectsLateWorkerEvents() async throws {
        let f = try setup(); defer { try? FileManager.default.removeItem(at: f.root) }
        f.workspace.workflowSession = f.host
        var emit: ((String) -> Void)?, exit: ((Int32, String) -> Void)?
        f.host.executionLaunch = { _, args, _, line, finish in
            emit = line; exit = finish
            let index = try XCTUnwrap(args.firstIndex(of: "--kwip-operation-supervise"))
            let request = try WorkflowOperationRequest.decode(Data(contentsOf: URL(fileURLWithPath: args[index + 1]).appendingPathComponent("request.json")))
            let ready: [String: Any] = ["event": "operationReady", "operationID": request.operationID,
                "phase": request.phase.rawValue, "requestSHA256": args[index + 2], "ownershipSHA256": args[index + 3]]
            line(String(decoding: try JSONSerialization.data(withJSONObject: ready), as: UTF8.self))
            return .init(cancel: {})
        }
        XCTAssertTrue(f.host.run());try await awaitBoundary {exit != nil}
        let digest = try WorkflowCapturedSource.capture(f.csv).sha256
        emit?(#"{"event":"dataflowNodeStarted","nodeID":"input"}"#)
        // CSV source values remain strings until the downstream Cast stage. Retain the same
        // four-row source and first-row values, with the bounded schema/lineage contract.
        let preview: [String: Any] = ["event": "dataflowPreview", "nodeID": "input",
            "execution": ["nodeID": "input", "status": "executed"],
            "outputs": ["table": ["kind": "table",
                "lineage": ["role": "unassigned", "sourceDigest": digest,
                    "membershipDigest": digest, "dataVersion": digest,
                    "partitionDigest": "", "fittedStateDigests": []],
                "preview": ["partition": "unassigned", "dataVersion": digest, "rowCount": 4,
                    "columns": ["a", "b", "label"].map { ["name": $0, "dtype": "string", "nullable": false] as [String: Any] },
                    "rows": [["1", "2", "0"]], "diagnostics": [], "diagnosticCount": 0]]]]
        emit?(String(decoding: try JSONSerialization.data(withJSONObject: preview), as: UTF8.self))
        let live = try XCTUnwrap(f.host.snapshot(for: address(.data, "input")))
        XCTAssertEqual(live.phase, .preparing)
        XCTAssertTrue(live.matchesCurrentConstruction)
        f.host.cancel(); exit?(1, "cancelled")
        let cancelled = try XCTUnwrap(f.host.snapshot(for: address(.data, "input")))
        XCTAssertEqual(cancelled.phase, .cancelled)
        XCTAssertEqual(cancelled.ports, live.ports)
        emit?(#"{"event":"dataflowPreview","nodeID":"input","outputs":{}}"#)
        XCTAssertEqual(f.host.snapshot(for: address(.data, "input"))?.ports, live.ports)
        XCTAssertEqual(f.host.outcomes.count, 1)
    }
    @MainActor func testMultipleTrainNodesRequireExplicitOrSelectedTerminal() async throws {
        let f = try setup();defer{try? FileManager.default.removeItem(at:f.root)}
        let second = address(.control,"train-2")
        XCTAssertNotNil(f.owner.apply(.insert(.init(address:second,payload:.train(config:[:]))),lease:f.owner.lease()))
        for edge in f.owner.document.edges.filter({$0.to.address == address(.control,"train")}) {
            XCTAssertNotNil(f.owner.apply(.connect(from:edge.from,to:.init(address:second,port:edge.to.port)),lease:f.owner.lease()))
        }
        _ = f.owner.select(.init(),lease:f.owner.lease())
        var launches = 0,exit:((Int32,String)->Void)?
        f.host.executionLaunch = {_,_,_,_,finish in launches += 1;exit=finish;return .init(cancel:{})}
        XCTAssertFalse(f.host.run());XCTAssertEqual(launches,0)
        XCTAssertTrue(f.host.statusMessage.lowercased().contains("train"))
        XCTAssertTrue(f.owner.select(.init(nodeAddresses:[second]),lease:f.owner.lease()))
        XCTAssertTrue(f.host.run());try await awaitBoundary {exit != nil};XCTAssertEqual(f.host.execution?.request.train,second)
        f.host.execution?.cancel();exit?(1,"")
        XCTAssertEqual(launches,1)
        XCTAssertFalse(f.host.run(train:address(.data,"input")));XCTAssertEqual(launches,1)
    }
    @MainActor func testRetireKeepsLateOutcomeBoundToOldProjectAndCannotClearNewReservation() async throws {
        let f = try setup();defer{try? FileManager.default.removeItem(at:f.root)}
        var exit:((Int32,String)->Void)?,cancels = 0
        f.host.executionLaunch = {_,_,_,_,finish in exit=finish;return .init(cancel:{cancels += 1})}
        XCTAssertTrue(f.host.run());try await awaitBoundary {exit != nil}
        let captured = try XCTUnwrap(f.host.execution?.request)
        f.host.retire();XCTAssertTrue(f.host.isRetired);XCTAssertEqual(cancels,1)
        XCTAssertFalse(f.owner.isCurrent(captured.lease))
        let nextID = UUID().uuidString
        f.workspace.activeProjectID = nextID
        var newCancel = 0
        f.workspace.cancelDataPreparation = {newCancel += 1}
        let newReservation = f.workspace.dataPreparationReservationID
        f.workspace.isPreparingData = true
        exit?(0,"")
        XCTAssertEqual(f.workspace.activeProjectID,nextID)
        XCTAssertEqual(f.workspace.dataPreparationReservationID,newReservation)
        XCTAssertTrue(f.workspace.isPreparingData)
        f.workspace.cancelDataPreparation?();XCTAssertEqual(newCancel,1)
        XCTAssertEqual(f.host.outcomes.first?.request.lease.projectID,captured.lease.projectID)
        XCTAssertEqual(f.host.outcomes.first?.status,.cancelled)
        XCTAssertFalse(f.host.run())
    }
    @MainActor func testBindingChangeDuringSnapshotCannotLaunchAndPendingEditCannotBeOmitted() async throws {
        let f = try setup();defer{try? FileManager.default.removeItem(at:f.root)}
        var launches = 0
        f.host.executionLaunch = {_,_,_,_,_ in launches += 1;return .init(cancel:{})}
        let capture = f.host.captureRunSources
        f.host.captureRunSources = { paths, root, permissions, limit in
            let snapshot = try await capture(paths, root, permissions, limit)
            _ = f.host.bindings.setPath("/tmp/changed.csv",for:self.address(.data,"input"),lease:f.owner.lease(),expectedRevision:f.host.bindings.revision)
            return snapshot
        }
        XCTAssertTrue(f.host.run());try await awaitBoundary { !f.workspace.isBusy };XCTAssertEqual(launches,0);XCTAssertFalse(f.workspace.isBusy)
        let node = address(.data,"split")
        _ = f.owner.select(.init(nodeAddresses:[node]),lease:f.owner.lease())
        XCTAssertTrue(f.owner.setDataEditPending(contextID:UUID(),address:node,key:"seed",pending:true,lease:f.owner.lease()))
        XCTAssertFalse(f.host.run());XCTAssertTrue(f.host.statusMessage.lowercased().contains("pending"))
        XCTAssertEqual(launches,0)
    }
    @MainActor func testHostCanBeValidatedBeforeSwitchingActiveProjectButCannotRunYet() throws {
        let f = try setup();defer{try? FileManager.default.removeItem(at:f.root)}
        let previous = UUID().uuidString
        f.workspace.activeProjectID = previous
        let prepared = try WorkflowWorkspaceSession(workspace:f.workspace,owner:f.owner)
        XCTAssertEqual(f.workspace.activeProjectID,previous)
        XCTAssertFalse(prepared.run())
        XCTAssertFalse(f.workspace.isBusy)
    }
    @MainActor func testIdleWorkspaceAndInstalledHostHaveNoOwnershipCycle() throws {
        var fixture:Fixture? = try setup()
        let root = try XCTUnwrap(fixture?.root);defer{try? FileManager.default.removeItem(at:root)}
        weak var workspace = fixture?.workspace
        weak var host = fixture?.host
        fixture?.workspace.workflowSession = fixture?.host
        fixture = nil
        XCTAssertNil(workspace,"An idle installed host/provider must not retain its workspace")
        XCTAssertNil(host)
    }
    @MainActor func testActiveInspectionRetainsWorkspaceOnlyThroughActualExit() async throws {
        var fixture:Fixture? = try setup()
        let root = try XCTUnwrap(fixture?.root);defer{try? FileManager.default.removeItem(at:root)}
        weak var workspace = fixture?.workspace
        weak var host = fixture?.host
        var finish:((Int32,String,SourceInspectionResult?)->Void)?
        fixture?.workspace.workflowSession = fixture?.host
        fixture?.host.sourceProvider.inspectionExecute = {_,_,_,_,_,_,_,exit in finish=exit;return .init(cancel:{})}
        let node = address(.data,"input")
        XCTAssertTrue(try XCTUnwrap(fixture).owner.select(.init(nodeAddresses:[node]),lease:try XCTUnwrap(fixture).owner.lease()))
        XCTAssertTrue(try XCTUnwrap(fixture).host.sourceProvider.inspectSource(for:node,lease:try XCTUnwrap(fixture).owner.lease()))
        fixture = nil
        XCTAssertNotNil(workspace,"Accepted inspection keeps cancellation dependencies alive")
        try await awaitBoundary { finish != nil }
        finish?(1,"intentional failure",nil)
        try await awaitBoundary { workspace == nil }
        XCTAssertNil(workspace);XCTAssertNil(host)
        finish = nil
    }

    @MainActor func testOptInActualHostRunPreservesCapturedBlockAndPreparedProvenance() async throws {
        guard let development = ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {
            throw XCTSkip("Set KWIPML_TEST_DEVELOPMENT_ROOT for actual native host preparation-to-training execution.")
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let workspace = WorkspaceStore(rootOverride:URL(fileURLWithPath:development),storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace)
        let runtime = try XCTUnwrap(workspace.runtime)
        let document = fixture(block:true)
        _ = try workspace.projectLibrary.saveSource(.workflow(document),id:document.projectID)
        let model = try JSONDecoder().decode(ComponentRegistry.self,from:Data(contentsOf:runtime.registryURL))
        let data = try DataFlowOperatorRegistry.load(from:runtime.workingDirectoryURL.appendingPathComponent("runner/dataflow-operators.json"))
        let owner = try WorkflowAuthoringStore(library:workspace.projectLibrary,projectID:document.projectID,modelRegistry:model,dataRegistry:data)
        let host = try WorkflowWorkspaceSession(workspace:workspace,owner:owner)
        workspace.activeProjectID = owner.projectID
        workspace.workflowSession = host
        let csv = root.appendingPathComponent("host-observations.csv")
        let rows = (0..<64).map {"\($0),\(($0 * 7) % 13),\($0 % 2)"}
        try Data((["a,b,label"]+rows).joined(separator:"\n").utf8).write(to:csv)
        XCTAssertTrue(host.bindings.setPath(csv.path,for:address(.data,"input"),lease:owner.lease(),expectedRevision:0))
        let hidden = TrainingSampleCapturePolicy(activationValues: false, predictionRows: false)
        XCTAssertTrue(workspace.setProjectSampleCapture(hidden, projectID: owner.projectID))
        let finished = expectation(description:"Actual host publishes captured terminal outcome")
        let subscription = host.$outcomes.dropFirst().filter{!$0.isEmpty}.prefix(1).sink{_ in finished.fulfill()}
        defer {subscription.cancel()}
        XCTAssertTrue(host.run(),host.statusMessage)
        XCTAssertTrue(workspace.resetProjectSampleCapture(projectID: owner.projectID))
        XCTAssertTrue(workspace.isPreparingData)
        try await awaitBoundary { owner.isBusy || !host.outcomes.isEmpty }
        await fulfillment(of:[finished],timeout:45)
        if workspace.isBusy {host.cancel()}
        let outcome = try XCTUnwrap(host.outcomes.first)
        XCTAssertEqual(outcome.status,.completed,outcome.error ?? host.statusMessage)
        XCTAssertNil(outcome.recordError)
        XCTAssertEqual(outcome.request.document,document)
        XCTAssertEqual(outcome.request.lease.projectID,owner.projectID)
        let reopened = try XCTUnwrap(WorkflowHistoryReader.loadModelContent(
            .init(projectID: owner.projectID, directoryID: try XCTUnwrap(UUID(uuidString: outcome.directory.lastPathComponent))), root: root))
        let retainedPolicy=TrainingSampleCapturePolicy(activationValues:false,predictionRows:false,performanceMeasurements:false,retainComponentOutputs:true,retainDiagnosticDetails:true)
        XCTAssertEqual(reopened.capture.sampleCapture, retainedPolicy)
        XCTAssertEqual(reopened.predictionPreview?.captureStatus, .omittedByPolicy)
        XCTAssertEqual(outcome.request.sampleCapture, retainedPolicy)
        XCTAssertEqual(host.execution?.predictionPreview?.captureStatus, .omittedByPolicy)
        XCTAssertEqual(host.execution?.metrics.count,3)
        XCTAssertTrue(host.execution?.metrics.allSatisfy{$0.trainLoss.isFinite && $0.validationLoss.isFinite && $0.accuracy.map(\.isFinite) == true} == true)
        XCTAssertFalse(workspace.isBusy);XCTAssertFalse(workspace.isPreparingData)
        XCTAssertNil(workspace.cancelDataPreparation);XCTAssertFalse(owner.isBusy)
        let training = outcome.directory.appendingPathComponent("training/artifacts")
        for filename in ["checkpoint.pt","model.py","metrics.json","manifest.json","composite/source.json"] {
            XCTAssertGreaterThan(try Data(contentsOf:training.appendingPathComponent(filename)).count,0,filename)
        }
        let manifest = try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:training.appendingPathComponent("manifest.json"))) as? [String:Any])
        let provenance = try XCTUnwrap(manifest["provenance"] as? [String:Any])
        XCTAssertEqual(provenance["sourceHash"] as? String,try WorkflowCapturedSource.capture(csv).sha256)
        // Two 32-row strata at 60/20/20 allocate 19/7/6 each; twelve rows stay protected.
        XCTAssertEqual(provenance["trainingSamples"] as? Int,38)
        XCTAssertEqual(provenance["validationSamples"] as? Int,14)
        XCTAssertEqual((provenance["preparation"] as? [String:Any])?["manifestSHA256"] as? String,outcome.preparationManifestSHA256)
        XCTAssertNotNil(provenance["composite"])
        XCTAssertEqual((manifest["settings"] as? [String:Any])?["partitionMode"] as? String,"prepared")
        let archived = try JSONDecoder().decode(GraphDocument.self,from:Data(contentsOf:training.appendingPathComponent("graph.json")))
        XCTAssertEqual(archived,outcome.request.plan.model.graph)
        XCTAssertEqual(try workspace.projectLibrary.loadSource(id:owner.projectID),.workflow(document))
        XCTAssertGreaterThan(try Data(contentsOf:outcome.directory.appendingPathComponent("outcome.json")).count,0)
    }

    @MainActor func testDroppedHostIsRetainedOnlyUntilTerminalCallback() async throws {
        let f = try setup();defer{try? FileManager.default.removeItem(at:f.root)}
        var host:WorkflowWorkspaceSession? = try WorkflowWorkspaceSession(workspace:f.workspace,owner:f.owner)
        weak var retained = host
        var exit:((Int32,String)->Void)?
        host?.executionLaunch = {_,_,_,_,finish in exit=finish;return .init(cancel:{})}
        XCTAssertTrue(try XCTUnwrap(host).run());host=nil
        XCTAssertNotNil(retained);XCTAssertTrue(f.workspace.isBusy)
        try await awaitBoundary {exit != nil}
        exit?(1,"intentional worker failure")
        XCTAssertNil(retained);XCTAssertFalse(f.workspace.isBusy)
    }
}
