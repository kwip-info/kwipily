import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp

final class WorkflowHistoryReaderTests: XCTestCase {
    private func address(_ domain:WorkflowDomain,_ id:String)->WorkflowNodeAddress {.init(domain:domain,localID:id)}
    private func edge(_ a:WorkflowDomain,_ id:String,_ port:String,_ b:WorkflowDomain,_ target:String,_ input:String,_ kind:WorkflowEdgeKind)->WorkflowEdge {
        .init(id:"\(a)-\(id)-\(port)-\(b)-\(target)-\(input)",from:.init(address:address(a,id),port:port),to:.init(address:address(b,target),port:input),kind:kind)
    }
    private func historicalFixture(block:Bool=false)->WorkflowDocument {
        let base=ExampleFlows.parallelClassifier
        let source=CompositeExamples.parallelClassifier
        var nodes:[WorkflowNode]=[
            .init(address:address(.data,"input"),payload:.dataOperator(operatorID:"kwip.data.sourceCSV",config:["separator":.string(",")])),
            .init(address:address(.data,"cast"),payload:.dataOperator(operatorID:"kwip.data.cast",config:["columns":.object(["a":.string("float64"),"b":.string("float64"),"label":.string("int64")])])),
            .init(address:address(.data,"split"),payload:.dataOperator(operatorID:"kwip.data.split",config:["seed":.integer(42),"validationPercent":.integer(20),"stratifyColumn":.string("label")])),
            .init(address:address(.data,"fit"),payload:.dataOperator(operatorID:"kwip.data.standardScalerFit",config:["columns":.array([.string("a"),.string("b")])])),
            .init(address:address(.data,"training"),payload:.dataOperator(operatorID:"kwip.data.transform",config:[:])),
            .init(address:address(.data,"validation"),payload:.dataOperator(operatorID:"kwip.data.transform",config:[:])),
            .init(address:address(.control,"model"),payload:.modelDefinition),
            .init(address:address(.control,"mapping"),payload:.trainingData(config:["featureColumns":.array([.string("a"),.string("b")]),"targetColumn":.string("label")])),
            .init(address:address(.control,"setup"),payload:.runSetup(config:["device":.string("cpu"),"epochs":.integer(3),"batchSize":.integer(32),"seed":.integer(17),"learningRate":.number(0.02)])),
            .init(address:address(.control,"train"),payload:.train(config:[:]))]
        let primitives=block ? source.root.nodes : base.nodes
        nodes += primitives.map {.init(address:address(.model,$0.id),name:$0.name,x:$0.x,y:$0.y,payload:.modelPrimitive(type:$0.type,configurationVersion:block ? (source.root.configurationVersion ?? 1) : base.schemaVersion,config:$0.config))}
        if block {nodes += source.root.instances.map {.init(address:address(.model,$0.id),name:$0.name,x:$0.x,y:$0.y,payload:.modelBlock(reference:$0.definition,overrides:$0.overrides))}}
        var edges=[edge(.data,"input","table",.data,"cast","table",.table),edge(.data,"cast","table",.data,"split","table",.table),edge(.data,"split","training",.data,"fit","training",.dataset),edge(.data,"split","training",.data,"training","table",.dataset),edge(.data,"split","validation",.data,"validation","table",.dataset),edge(.data,"fit","state",.data,"training","state",.fittedState),edge(.data,"fit","state",.data,"validation","state",.fittedState),edge(.data,"training","table",.control,"mapping","training",.table),edge(.data,"validation","table",.control,"mapping","validation",.table),edge(.model,"output","definition",.control,"model","source",.modelDefinition),edge(.control,"model","model",.control,"train","model",.modelDefinition),edge(.control,"mapping","data",.control,"train","data",.trainingData),edge(.control,"setup","configuration",.control,"train","configuration",.runConfiguration)]
        edges += (block ? source.root.edges : base.edges).map {.init(id:$0.id,from:.init(address:address(.model,$0.source),port:$0.sourcePort),to:.init(address:address(.model,$0.target),port:$0.targetPort),kind:.tensor)}
        return WorkflowDocument(projectID:UUID().uuidString,revision:7,name:"Measured workflow",nodes:nodes,edges:edges,definitions:block ? source.definitions : [])
    }
    private var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    /// Captured archives retain their original operator contracts. This combined
    /// registry is evidence for replay, never the current execution registry.
    private func historicalDataRegistry() throws -> DataFlowOperatorRegistry {
        var captured = try DataFlowOperatorRegistry.load(from: repo.appendingPathComponent("runner/dataflow-operators.json"))
        let retired = try DataFlowOperatorRegistry.load(from: repo.appendingPathComponent("runner/retired-data-operators-v1.json"))
        XCTAssertEqual(captured.packageID, retired.packageID)
        XCTAssertTrue(Set(captured.operators.map(\.id)).isDisjoint(with: retired.operators.map(\.id)))
        captured.operators += retired.operators
        return captured
    }

    private struct Archive {
        let root: URL
        let directory: URL
        let reference: WorkflowHistoryReference
        let outcome: WorkflowArchiveOutcome
        let document: WorkflowDocument
        let plan: WorkflowExecutionPlan
    }
    private func archive(mode:WorkflowTrainingMode?=nil) throws -> Archive {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        var document = historicalFixture(block: true)
        if let mode {
            document=try WorkflowCodec.migrateToVersion3(document)
            let index=try XCTUnwrap(document.nodes.firstIndex {$0.address == address(.control,"train")})
            document.nodes[index].payload = .train(config:["trainingMode":.string(mode.rawValue)])
        }
        let reference = WorkflowHistoryReference(projectID: document.projectID, directoryID: UUID())
        let directory = root.appendingPathComponent("workflow-runs/\(document.projectID)/\(reference.directoryID.uuidString)")
        try FileManager.default.createDirectory(at: directory.appendingPathComponent("inputs"), withIntermediateDirectories:true)
        let model = try ComponentRegistry.builtIn(), data = try historicalDataRegistry()
        let plan = try WorkflowCompiler.compileHistoricalEvidence(document,train:address(.control,"train"),modelRegistry:model,dataRegistry:data,contractVersion:mode == nil ? 2:3)
        func write(_ path:String,_ bytes:Data) throws -> WorkflowArchiveFile {
            try bytes.write(to:directory.appendingPathComponent(path))
            return .init(path:path,bytes:bytes.count,sha256:WorkflowArchiveCodec.hash(bytes))
        }
        let workflow = try write("workflow.json",WorkflowCodec.encode(document))
        let modelFile = try write("model-registry.json",WorkflowArchiveCodec.encode(model))
        let dataFile = try write("data-registry.json",WorkflowArchiveCodec.encode(data))
        let workerFiles = ["runner/main.py":String(repeating:"a",count:64),"components.json":modelFile.sha256,"runner/dataflow-operators.json":dataFile.sha256]
        let evidence = WorkflowRuntimeEvidence(schemaVersion:1,kind:"development",packageVersions:[:],manifestSHA256:nil,workerFiles:workerFiles,
            workerInventorySHA256:try WorkflowArchiveCodec.registryHash(workerFiles),coverage:"developmentObservedWorkerSourceOnly")
        let runtime = try write("runtime-evidence.json",WorkflowArchiveCodec.encode(evidence))
        let source = try write("inputs/source-0.csv",Data("a,b,label\n1,2,0\n2,3,1\n".utf8))
        let run = UUID().uuidString, request = String(repeating:"b",count:64)
        var parent:WorkflowCheckpointReference?
        var parentFiles:[WorkflowArchiveFile]?
        if let mode,mode != .fresh {
            try FileManager.default.createDirectory(at:directory.appendingPathComponent("continuation-parent"),withIntermediateDirectories:false)
            let checkpoint=Data("fixture checkpoint".utf8)
            let metrics=Data("[{\"epoch\":1,\"trainLoss\":0.5,\"validationLoss\":0.6,\"accuracy\":0.5}]".utf8)
            let manifest:[String:Any]=["schemaVersion":1,"status":"completed","runID":UUID().uuidString,
                "checkpoint":["schemaVersion":1,"completedEpoch":1,"contract":["runtime":["device":"cpu"]],"bytes":checkpoint.count,"sha256":WorkflowArchiveCodec.hash(checkpoint),"metricsSHA256":WorkflowArchiveCodec.hash(metrics)]]
            let manifestFile=try write("continuation-parent/manifest.json",JSONSerialization.data(withJSONObject:manifest))
            parentFiles=[manifestFile,try write("continuation-parent/checkpoint.pt",checkpoint),try write("continuation-parent/metrics.json",metrics)]
            parent = .init(projectID:document.projectID,directoryID:UUID().uuidString,runID:UUID().uuidString,requestIdentity:String(repeating:"c",count:64),modelManifestSHA256:manifestFile.sha256)
        }
        var capture = WorkflowArchiveCapture(projectID:document.projectID,directoryID:reference.directoryID.uuidString,runID:run,
            requestIdentity:request,revision:document.revision,train:plan.train,workflow:workflow,modelRegistry:modelFile,dataRegistry:dataFile,
            modelRegistrySHA256:try WorkflowArchiveCodec.registryHash(model),dataRegistrySHA256:try WorkflowArchiveCodec.registryHash(data),
            runtimeEvidence:runtime,sources:[.init(address:address(.data,"input"),path:source.path,bytes:source.bytes,sha256:source.sha256)],plan:try WorkflowArchivePlan(plan, compilerContractVersion: mode == nil ? 1:3))
        if mode != nil {capture.schemaVersion=3;capture.compilerContractVersion=3;capture.checkpointParent=parent;capture.checkpointFiles=parentFiles}
        let captureFile = try write("capture.json",WorkflowArchiveCodec.encodeCapture(capture))
        let outcome = WorkflowArchiveOutcome(projectID:document.projectID,directoryID:reference.directoryID.uuidString,runID:run,
            requestIdentity:request,status:.failed,terminalPhase:.preparing,startedAt:"2026-10-05T12:00:00.000Z",finishedAt:"2026-10-05T12:00:01.000Z",
            artifacts:[],capture:captureFile,error:"Original preparation error")
        _ = try write("outcome.json",WorkflowArchiveCodec.encodeOutcome(outcome))
        return Archive(root:root,directory:directory,reference:reference,outcome:outcome,document:document,plan:plan)
    }
    private func rewriteCapture(_ archive:Archive,_ edit:(inout [String:Any])->Void) throws {
        let url = archive.directory.appendingPathComponent("capture.json")
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:url)) as? [String:Any])
        edit(&object)
        let bytes = try JSONSerialization.data(withJSONObject:object,options:[.sortedKeys])
        try bytes.write(to:url)
        var outcome = archive.outcome
        outcome.capture = .init(path:"capture.json",bytes:bytes.count,sha256:WorkflowArchiveCodec.hash(bytes))
        try WorkflowArchiveCodec.encodeOutcome(outcome).write(to:archive.directory.appendingPathComponent("outcome.json"))
    }
    func testVersion3ContinuationReplaysCapturedModeAndRejectsInvalidParentRecord()throws {
        for mode in [WorkflowTrainingMode.fresh,.exactResume,.fineTune] {
            let item=try archive(mode:mode);defer{try? FileManager.default.removeItem(at:item.root)}
            guard case .verified(let value)=WorkflowHistoryReader.load(item.reference,root:item.root) else {XCTFail("Version3 capture must replay: \(mode)");continue}
            XCTAssertEqual(value.plan.trainingMode,mode)
            XCTAssertNil(value.modelContent)
            if mode == .fresh {continue}
            let path="continuation-parent/metrics.json"
            let changed=Data("[]".utf8)
            try changed.write(to:item.directory.appendingPathComponent(path))
            try rewriteCapture(item) { object in
                var files=object["checkpointFiles"] as! [[String:Any]]
                let index=files.firstIndex {$0["path"] as? String == path}!
                files[index]["bytes"]=changed.count;files[index]["sha256"]=WorkflowArchiveCodec.hash(changed)
                object["checkpointFiles"]=files
            }
            guard case .unavailable=WorkflowHistoryReader.load(item.reference,root:item.root) else {return XCTFail("Parent manifest must verify metrics independently of capture inventory")}
        }
    }
    func testPureCapturedSourceReplayPreservesFailedStatusAndOriginalError() throws {
        let item = try archive();defer{try? FileManager.default.removeItem(at:item.root)}
        guard case .verified(let value) = WorkflowHistoryReader.load(item.reference,root:item.root) else {return XCTFail("Pinned capture must replay without live owner/runtime")}
        XCTAssertEqual(value.outcome.status,.failed)
        XCTAssertEqual(value.outcome.error,"Original preparation error")
        XCTAssertEqual(value.plan,item.plan)
        // Replay proves the original hash-bound graph, rather than upgrading its
        // retired Cast/Split/Fit/Transform nodes into current authoring tools.
        let capturedBytes = try Data(contentsOf:item.directory.appendingPathComponent("workflow.json"))
        XCTAssertEqual(capturedBytes,try WorkflowCodec.encode(item.document))
        XCTAssertEqual(value.capture.workflow.sha256,WorkflowArchiveCodec.hash(capturedBytes))
        let capturedData = try DataFlowOperatorRegistry.load(from:item.directory.appendingPathComponent("data-registry.json"))
        XCTAssertNotNil(capturedData.operatorDefinition(id:"kwip.data.cast"))
        XCTAssertNotNil(capturedData.operatorDefinition(id:"kwip.data.transform"))
        XCTAssertEqual(value.capture.revision,item.document.revision)
        let listing = try WorkflowHistoryReader.list(root:item.root,projectID:item.document.projectID,limit:100)
        XCTAssertFalse(listing.truncated);XCTAssertEqual(listing.entries.count,1)
        XCTAssertEqual(listing.entries.first?.verification,.recordedV2)
    }
    func testPlanTamperingWithUpdatedPinsStillFailsIndependentCompilation() throws {
        let item = try archive();defer{try? FileManager.default.removeItem(at:item.root)}
        try rewriteCapture(item) { object in
            var plan = object["plan"] as! [String:Any]
            var settings = plan["settings"] as! [String:Any]
            settings["epochs"] = 7;plan["settings"] = settings;object["plan"] = plan
        }
        guard case .unavailable = WorkflowHistoryReader.load(item.reference,root:item.root) else {return XCTFail("Derived plan must be recomputed")}
    }
    func testChangedBytesAndSourceSymlinkAreNotExposedAsVerified() throws {
        let item = try archive();defer{try? FileManager.default.removeItem(at:item.root)}
        let source = item.directory.appendingPathComponent("inputs/source-0.csv")
        let bytes = try Data(contentsOf:source)
        try Data("changed".utf8).write(to:source)
        guard case .unavailable = WorkflowHistoryReader.load(item.reference,root:item.root) else {return XCTFail("Source mutation accepted")}
        let external = item.root.appendingPathComponent("external.csv")
        try bytes.write(to:external);try FileManager.default.removeItem(at:source)
        try FileManager.default.createSymbolicLink(at:source,withDestinationURL:external)
        guard case .unavailable = WorkflowHistoryReader.load(item.reference,root:item.root) else {return XCTFail("Symlink accepted despite matching bytes")}
    }
    func testListingPreservesCorruptIncompleteAndLegacyEntriesWithBoundedEnumeration() throws {
        let item = try archive();defer{try? FileManager.default.removeItem(at:item.root)}
        let parent = item.directory.deletingLastPathComponent()
        for (index,content) in [Data("{bad".utf8),nil].enumerated() {
            let directory = parent.appendingPathComponent(UUID().uuidString)
            try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
            if let content {try content.write(to:directory.appendingPathComponent("outcome.json"))}
            XCTAssertLessThan(index,2)
        }
        let legacyDirectory = parent.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:legacyDirectory,withIntermediateDirectories:true)
        let legacy:[String:Any] = ["schemaVersion":1,"projectID":item.document.projectID,"runID":UUID().uuidString,"requestIdentity":String(repeating:"c",count:64),"status":"cancelled","artifactHashes":[:],"sourceHashes":[:]]
        try JSONSerialization.data(withJSONObject:legacy).write(to:legacyDirectory.appendingPathComponent("outcome.json"))
        let result = try WorkflowHistoryReader.list(root:item.root,projectID:item.document.projectID,limit:100)
        XCTAssertEqual(result.entries.count,4)
        XCTAssertTrue(result.entries.contains{$0.verification == .unavailable})
        XCTAssertTrue(result.entries.contains{$0.verification == .incomplete})
        XCTAssertTrue(result.entries.contains{$0.verification == .legacyV1})
        XCTAssertTrue(try WorkflowHistoryReader.list(root:item.root,projectID:item.document.projectID,limit:2).truncated)
        let reference = WorkflowHistoryReference(projectID:item.document.projectID,directoryID:try XCTUnwrap(UUID(uuidString:legacyDirectory.lastPathComponent)))
        guard case .legacyUnverifiable = WorkflowHistoryReader.load(reference,root:item.root) else {return XCTFail("Legacy receipt must retain explicit evidence limitation")}
    }
    func testForeignIdentityAndMalformedProjectDoNotCrossProjectBoundary() throws {
        let item = try archive();defer{try? FileManager.default.removeItem(at:item.root)}
        XCTAssertThrowsError(try WorkflowHistoryReader.list(root:item.root,projectID:"../escape",limit:100))
        let wrong = WorkflowHistoryReference(projectID:UUID().uuidString,directoryID:item.reference.directoryID)
        guard case .incomplete = WorkflowHistoryReader.load(wrong,root:item.root) else {return XCTFail("Foreign project must not find another project's receipt")}
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with:WorkflowArchiveCodec.encodeOutcome(item.outcome)) as? [String:Any])
        object["projectID"] = UUID().uuidString
        try JSONSerialization.data(withJSONObject:object).write(to:item.directory.appendingPathComponent("outcome.json"))
        guard case .unavailable = WorkflowHistoryReader.load(item.reference,root:item.root) else {return XCTFail("Receipt substitution accepted")}
    }
    func testRunDirectorySymlinkCannotBorrowIdenticalArchiveBytes() throws {
        let item = try archive();defer{try? FileManager.default.removeItem(at:item.root)}
        let moved = item.root.appendingPathComponent("moved-archive")
        try FileManager.default.moveItem(at:item.directory,to:moved)
        try FileManager.default.createSymbolicLink(at:item.directory,withDestinationURL:moved)
        guard case .unavailable = WorkflowHistoryReader.load(item.reference,root:item.root) else { return XCTFail("Run directory symlink must not become a trusted root") }
        let listing = try WorkflowHistoryReader.list(root:item.root,projectID:item.document.projectID,limit:100)
        XCTAssertEqual(listing.entries.first?.verification,.unavailable)
    }

    @MainActor func testOptInActualCompletedArchiveReplaysAfterHostRetirement() async throws {
        guard let development = ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {
            throw XCTSkip("Set KWIPML_TEST_DEVELOPMENT_ROOT for actual native host preparation-to-training execution.")
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let workspace = WorkspaceStore(rootOverride:URL(fileURLWithPath:development),storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace)
        let runtime = try XCTUnwrap(workspace.runtime)
        let document = try WorkflowExamples.preparedParallelClassifier(device:"cpu")
        let operatorIDs = Set(document.nodes.compactMap { node -> String? in
            if case .dataOperator(let id,_) = node.payload {return id};return nil
        })
        XCTAssertTrue(["kwip.data.columns","kwip.data.partition","kwip.data.preprocess"].allSatisfy(operatorIDs.contains))
        XCTAssertTrue(operatorIDs.isDisjoint(with:["kwip.data.cast","kwip.data.split","kwip.data.standardScalerFit","kwip.data.transform"]))
        _ = try workspace.projectLibrary.saveSource(.workflow(document),id:document.projectID)
        let model = try JSONDecoder().decode(ComponentRegistry.self,from:Data(contentsOf:runtime.registryURL))
        let data = try DataFlowOperatorRegistry.load(from:runtime.workingDirectoryURL.appendingPathComponent("runner/dataflow-operators.json"))
        let owner = try WorkflowAuthoringStore(library:workspace.projectLibrary,projectID:document.projectID,modelRegistry:model,dataRegistry:data)
        let host = try WorkflowWorkspaceSession(workspace:workspace,owner:owner)
        workspace.activeProjectID = owner.projectID
        workspace.workflowSession = host
        let csv = root.appendingPathComponent("host-observations.csv")
        let rows = (0..<64).map {"\($0),\(($0 * 7) % 13),\($0 % 2)"}
        try Data((["signal,context,label"]+rows).joined(separator:"\n").utf8).write(to:csv)
        XCTAssertTrue(host.bindings.setPath(csv.path,for:WorkflowExamples.sourceAddress,lease:owner.lease(),expectedRevision:0))
        let finished = expectation(description:"Actual host publishes captured terminal outcome")
        let subscription = host.$outcomes.dropFirst().filter{!$0.isEmpty}.prefix(1).sink{_ in finished.fulfill()}
        defer {subscription.cancel()}
        XCTAssertTrue(host.run(),host.statusMessage)
        XCTAssertTrue(workspace.isPreparingData)
        let captureDeadline = Date().addingTimeInterval(5)
        while !owner.isBusy && host.outcomes.isEmpty && Date() < captureDeadline {
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTAssertTrue(owner.isBusy || !host.outcomes.isEmpty)
        await fulfillment(of:[finished],timeout:45)
        if workspace.isBusy {host.cancel()}
        let outcome = try XCTUnwrap(host.outcomes.first)
        XCTAssertEqual(outcome.status,.completed,outcome.error ?? host.statusMessage)
        XCTAssertNil(outcome.recordError)
        let reference = WorkflowHistoryReference(projectID:document.projectID,directoryID:try XCTUnwrap(UUID(uuidString:outcome.directory.lastPathComponent)))
        host.retire();workspace.workflowSession = nil
        guard case .verified(let result) = WorkflowHistoryReader.load(reference,root:workspace.workspaceDirectory) else {
            return XCTFail("Completed worker archive must independently replay after host retirement")
        }
        XCTAssertEqual(result.outcome.status,.completed)
        XCTAssertEqual(result.plan,outcome.request.plan)
        XCTAssertEqual(result.outcome.modelManifestSHA256,outcome.modelManifestSHA256)
        XCTAssertNotNil(result.outcome.workerRunID)
        let content = try XCTUnwrap(WorkflowHistoryReader.loadModelContent(reference,root:workspace.workspaceDirectory))
        XCTAssertEqual(content.reference,reference)
        XCTAssertEqual(content.outcome.runID,outcome.runID.uuidString)
        XCTAssertEqual(content.capture.plan.modelGraph,outcome.request.plan.model.graph)
        XCTAssertEqual(content.metrics.map(\.epoch),[1,2,3])
        XCTAssertTrue(content.metrics.allSatisfy{$0.trainLoss.isFinite && $0.validationLoss.isFinite && $0.accuracy.map {(0...1).contains($0)} == true})
        XCTAssertEqual(content.code,try String(contentsOf:outcome.directory.appendingPathComponent("training/artifacts/model.py"),encoding:.utf8))
        XCTAssertEqual(result.modelContent?.code,content.code)
        let metricsURL = outcome.directory.appendingPathComponent("training/artifacts/metrics.json")
        let metricsBytes = try Data(contentsOf:metricsURL)
        try Data("[]".utf8).write(to:metricsURL)
        XCTAssertThrowsError(try WorkflowHistoryReader.loadModelContent(reference,root:workspace.workspaceDirectory))
        try metricsBytes.write(to:metricsURL)
        // The accessor validates again; a prior result does not authorize changed bytes.
        let code = outcome.directory.appendingPathComponent("training/artifacts/model.py")
        try Data("# tampered".utf8).write(to:code)
        guard case .unavailable = WorkflowHistoryReader.load(reference,root:workspace.workspaceDirectory) else { return XCTFail("Modified code must invalidate completed archive") }
        XCTAssertThrowsError(try WorkflowHistoryReader.loadModelContent(reference,root:workspace.workspaceDirectory))
    }

    func testVerifiedInterruptedOutcomesHaveNoModelContent() throws {
        let item = try archive();defer{try? FileManager.default.removeItem(at:item.root)}
        XCTAssertNil(try WorkflowHistoryReader.loadModelContent(item.reference,root:item.root))
        let cancelled = WorkflowArchiveOutcome(projectID:item.outcome.projectID,directoryID:item.outcome.directoryID,
            runID:item.outcome.runID,requestIdentity:item.outcome.requestIdentity,status:.cancelled,terminalPhase:.preparing,
            startedAt:item.outcome.startedAt,finishedAt:item.outcome.finishedAt,artifacts:[],capture:item.outcome.capture)
        try WorkflowArchiveCodec.encodeOutcome(cancelled).write(to:item.directory.appendingPathComponent("outcome.json"))
        guard case .verified(let verified) = WorkflowHistoryReader.load(item.reference,root:item.root) else {return XCTFail("Cancelled capture must verify without inventing model evidence")}
        XCTAssertNil(verified.modelContent)
        XCTAssertNil(try WorkflowHistoryReader.loadModelContent(item.reference,root:item.root))
        try Data("changed".utf8).write(to:item.directory.appendingPathComponent("inputs/source-0.csv"))
        XCTAssertThrowsError(try WorkflowHistoryReader.loadModelContent(item.reference,root:item.root))
    }

}
