import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class LossRuntimeAdmissionTests:XCTestCase {
    private var repository:URL {URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()}
    private final class Recorder:@unchecked Sendable {
        private let lock=NSLock();private var count=0
        func capture() {lock.lock();count+=1;lock.unlock()}
        var captures:Int {lock.lock();defer {lock.unlock()};return count}
    }
    private func descriptor(_ id:String)throws->Data {
        let value:[String:Any] = ["version":"1.0.0","adapterID":"kwip.runner.objective."+id,"recipeSchemaVersions":[1],"configurationVersion":1,"predictionDTypes":["float32"],"devices":["cpu","mps"],"predictionRanks":[2,8],"sampleDimensionMaximum":4096,"classMinimum":2,"classMaximum":512,"classAxisMinimum": -7,"classAxisMaximum":7,"excludesBatchClassAxis":true,"reductions":id == "loss_kl_div" ? ["none","mean","sum","batchmean"]:["none","mean","sum"],"scalarizations":["mean","sum"],"targetRoles":id == "loss_ctc" ? ["target","input_lengths","target_lengths"]:id == "loss_triplet_margin" ? []:id == "loss_bce_with_logits" ? ["target","weight","pos_weight"]:["target"],"maximumTensorElements":100000000,"maximumWorkElements":100000000,"recipeMaximumBytes":32768,"fixedWeights":"buffers","mappedWeights":"preparedData","batchBroadcast":false]
        return try JSONSerialization.data(withJSONObject:value)
    }
    private func addr(_ id:String,_ domain:WorkflowDomain = .control)->WorkflowNodeAddress {.init(domain:domain,localID:id)}
    private func mapping(_ id:String)->LossTargetMapping {
        let inputs=id == "loss_triplet_margin" ? ["a","b","c","d","e","f"]:["a","b"]
        if id == "loss_triplet_margin" {return .init(componentID:id,inputColumns:inputs,roles:[],tripletSourceGroups:[["a","b"],["c","d"],["e","f"]])}
        if id == "loss_ctc" {return .init(componentID:id,inputColumns:inputs,roles:[.init(role:"target",columns:["label"],sampleShape:[1],dtype:.int64),.init(role:"input_lengths",columns:["inputLength"],sampleShape:[],dtype:.int64),.init(role:"target_lengths",columns:["targetLength"],sampleShape:[],dtype:.int64)])}
        let indexed=id == "loss_cross_entropy" || id == "loss_nll"
        return .init(componentID:id,inputColumns:inputs,roles:[.init(role:"target",columns:indexed ? ["label"]:["y0","y1","y2"],sampleShape:indexed ? []:[3],dtype:indexed ? .int64:.float32)])
    }
    private func document(_ id:String)throws->WorkflowDocument {
        var doc=try WorkflowExamples.preparedParallelClassifier(device:"cpu");doc.schemaVersion=8
        doc.definitions=[]
        doc.nodes.removeAll {$0.address.domain == .model}
        doc.edges.removeAll {$0.from.address.domain == .model || $0.to.address.domain == .model}
        let inputWidth=id == "loss_triplet_margin" ? 6:2
        let width=(id == "loss_ctc" || id == "loss_triplet_margin") ? 6:3
        doc.nodes += [
            .init(address:addr("input",.model),payload:.modelPrimitive(type:"input",configurationVersion:2,config:["features":.integer(Int64(inputWidth))])),
            .init(address:addr("head",.model),payload:.modelPrimitive(type:"linear",configurationVersion:2,config:["in_features":.integer(Int64(inputWidth)),"out_features":.integer(Int64(width)),"bias":true])),
            .init(address:addr("output",.model),payload:.modelPrimitive(type:"output",configurationVersion:2,config:[:]))]
        func wire(_ source:String,_ target:String) {doc.edges.append(.init(id:source+"-"+target,from:.init(address:addr(source,.model),port:"out"),to:.init(address:addr(target,.model),port:"in"),kind:.tensor))}
        wire("input","head")
        var previous="head"
        if width == 6 {
            let shape:[JSONValue]=id == "loss_ctc" ? [2,3]:[3,2]
            doc.nodes.append(.init(address:addr("shape",.model),payload:.modelPrimitive(type:"reshape",configurationVersion:2,config:["shape":.array(shape)])))
            wire(previous,"shape");previous="shape"
        }
        if ["loss_nll","loss_kl_div","loss_ctc"].contains(id) {
            doc.nodes.append(.init(address:addr("logs",.model),payload:.modelPrimitive(type:"log_softmax",configurationVersion:2,config:["dim":-1])))
            wire(previous,"logs");previous="logs"
        }
        wire(previous,"output")
        doc.edges.append(.init(id:"output-definition",from:.init(address:addr("output",.model),port:"definition"),to:.init(address:addr("model"),port:"source"),kind:.modelDefinition))
        let target=try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(mapping(id)))
        let index=try XCTUnwrap(doc.nodes.firstIndex {$0.address == addr("mapping")})
        doc.nodes[index].payload = .trainingData(config:["lossTargetMapping":target])
        doc.nodes.append(.init(address:addr("objective"),payload:.lossRecipe(type:id,configurationVersion:1,config:[:])))
        doc.edges.append(.init(id:"objective-train",from:.init(address:addr("objective"),port:"out"),to:.init(address:WorkflowExamples.trainAddress,port:"objective"),kind:.objectiveRecipe))
        return doc
    }

    private func fixture(_ id:String,verified:Bool=false,malformed:Bool=false,controlledCancellation:Bool=true)async throws->(WorkspaceStore,WorkflowWorkspaceSession,Recorder) {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        addTeardownBlock {try? FileManager.default.removeItem(at:directory)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:directory,detectCapabilities:false)
        var doc=try document(id)
        let setup=try XCTUnwrap(doc.nodes.firstIndex {if case .runSetup=$0.payload {return true};return false})
        guard case .runSetup(var settings)=doc.nodes[setup].payload else {throw WorkflowArchiveError("Missing setup")}
        settings["experiment"] = .object(["schemaVersion":1,"axes":.array([.object(["field":.string("seed"),"values":.array([7,8])])]),"budget":.object(["maximumRuns":2,"maximumTotalEpochs":6])])
        doc.nodes[setup].payload = .runSetup(config:settings)
        _=try workspace.projectLibrary.saveSource(.workflow(doc),id:doc.projectID)
        XCTAssertTrue(workspace.beginWorkflowProject(id:doc.projectID),workspace.statusMessage)
        let host=try XCTUnwrap(workspace.workflowSession)
        let csv=directory.appendingPathComponent("observations.csv")
        try FileManager.default.copyItem(at:repository.appendingPathComponent("Sources/KwipMLApp/Resources/workflow-observations.csv"),to:csv)
        try workspace.sourceAccess.remember(csv)
        XCTAssertTrue(host.bindings.setPath(csv.path,for:WorkflowExamples.sourceAddress,lease:host.owner.lease(),expectedRevision:host.bindings.revision))
        let base=try JSONEncoder().encode(BuiltInCapabilityBridge.executionContract(registry:workspace.registry,runtimeVersion:"2.14.1",availableDevices:[.cpu]))
        var loss=try XCTUnwrap(JSONSerialization.jsonObject(with:descriptor(id)) as? [String:Any])
        if malformed {loss["version"]=["major":1,"minor":0,"patch":0]}
        let layout:Data=try JSONSerialization.data(withJSONObject:["version":["major":1,"minor":0,"patch":0],"adapterID":"kwip.runner.graph-v2.tensor-layout","graphSchemaVersions":[2],"operations":["reshape","transpose","flatten"],"maximumRank":8,"preservesBatch":true])
        let log:Data=try JSONSerialization.data(withJSONObject:["version":["major":1,"minor":2,"patch":0],"adapterID":"kwip.runner.graph-v2.log_softmax","graphSchemaVersions":[2],"maximumRank":8,"preservesBatch":true,"graphDTypes":["float32"],"maximumTensorElements":100000000,"axisMinimum": -8,"axisMaximum":7,"excludesBatch":true,"outputKind":"logarithmicProbabilities"])
        try workspace.installDiscoveredRuntimeContract(version:"2.14.1",expectedVersion:nil,devices:["cpu"],descriptor:base,tensorLayoutDescriptor:layout,remainingActivationDescriptors:["log_softmax":log],lossDescriptors:verified ? [id:JSONSerialization.data(withJSONObject:loss)]:[:])
        if verified && !malformed {
            // Opening migrates connected legacy recipes to current Train settings.
            // That schema requires proof from the actual selected worker, while this
            // suite still controls exactly which objective descriptors are admitted.
            let selectedObjectives=workspace.lossRuntimeCapabilities
            await verifyRuntimeForIntegration(workspace)
            XCTAssertNotNil(workspace.moduleCompositionDiscoveryEvidence)
            workspace.lossRuntimeCapabilities=selectedObjectives
        }
        let recorder=Recorder()
        if controlledCancellation {
            host.captureRunSources={_,_,_,_ in recorder.capture();throw CancellationError()}
            host.captureExperimentInputs={_,_ in recorder.capture();throw CancellationError()}
        }
        host.executionLaunch={_,_,_,_,_ in XCTFail("No worker launch at admission-only boundary");throw CancellationError()}
        return(workspace,host,recorder)
    }
    func testMissingObjectiveDescriptorRefusesBeforeCaptureAndExperimentReservation()async throws {
        for id in LossConfiguration.componentIDs.sorted() {for experiments in [false,true] {
            let(w,h,r)=try await fixture(id);defer {h.cancel();w.retireWorkflowSession()}
            XCTAssertFalse(experiments ? h.runExperiments():h.run(),id+" "+h.statusMessage)
            XCTAssertTrue(h.statusMessage.lowercased().contains("objective"),h.statusMessage)
            XCTAssertEqual(r.captures,0);XCTAssertFalse(w.isBusy);XCTAssertFalse(h.isStagingExperiment);XCTAssertNil(h.queuedRequestID)
        }}
    }
    func testPartialExactDiscoveryCountAndRefreshFailureRevocation()throws {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:directory)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:directory,detectCapabilities:false)
        let base=try JSONEncoder().encode(BuiltInCapabilityBridge.executionContract(registry:workspace.registry,runtimeVersion:"2.14.1",availableDevices:[.cpu]))
        func install(_ ids:[String])throws {
            let descriptors=try Dictionary(uniqueKeysWithValues:ids.map {($0,try descriptor($0))})
            try workspace.installDiscoveredRuntimeContract(version:"2.14.1",expectedVersion:nil,devices:["cpu"],descriptor:base,lossDescriptors:descriptors)
        }
        try install([]);let count=workspace.verifiedRuntimeComponentCount
        for id in LossConfiguration.componentIDs.sorted() {
            try install([id]);XCTAssertEqual(workspace.lossRuntimeCapabilities.componentIDs,[id]);XCTAssertEqual(workspace.verifiedRuntimeComponentCount,count+1)
        }
        try install(Array(LossConfiguration.componentIDs));XCTAssertEqual(workspace.verifiedRuntimeComponentCount,count+8)
        try workspace.installDiscoveredRuntimeContract(version:"2.14.1",expectedVersion:nil,devices:["cpu"],descriptor:base)
        XCTAssertTrue(workspace.lossRuntimeCapabilities.componentIDs.isEmpty);XCTAssertEqual(workspace.verifiedRuntimeComponentCount,count)
        try install(Array(LossConfiguration.componentIDs))
        XCTAssertThrowsError(try workspace.installDiscoveredRuntimeContract(version:"2.14.1",expectedVersion:nil,devices:["cpu"],descriptor:Data("{}".utf8)))
        XCTAssertTrue(workspace.lossRuntimeCapabilities.componentIDs.isEmpty);XCTAssertEqual(workspace.verifiedRuntimeComponentCount,0)
        try install(Array(LossConfiguration.componentIDs))
        XCTAssertThrowsError(try workspace.installRuntimeContract(version:"bad",expectedVersion:nil))
        XCTAssertTrue(workspace.lossRuntimeCapabilities.componentIDs.isEmpty);XCTAssertEqual(workspace.verifiedRuntimeComponentCount,0)
    }
    func testMalformedStringVersionDescriptorCannotAdmitSelectedObjective()async throws {
        for id in LossConfiguration.componentIDs {
            let(w,h,r)=try await fixture(id,verified:true,malformed:true);defer {h.cancel();w.retireWorkflowSession()}
            XCTAssertFalse(w.lossRuntimeCapabilities.contains(id));XCTAssertFalse(h.run(),h.statusMessage);XCTAssertEqual(r.captures,0)
        }
    }
    func testVerifiedSelectedDescriptorAllowsTrainAndExperimentCapture()async throws {
        for id in LossConfiguration.componentIDs.sorted() {for experiments in [false,true] {
            let(w,h,r)=try await fixture(id,verified:true);defer {h.cancel();w.retireWorkflowSession()}
            XCTAssertTrue(experiments ? h.runExperiments():h.run(),id+" "+h.statusMessage)
            let deadline=Date().addingTimeInterval(5)
            while (r.captures==0 || w.isBusy),Date()<deadline {try await Task.sleep(for:.milliseconds(10))}
            XCTAssertEqual(r.captures,1);XCTAssertFalse(w.isBusy)
        }}
    }
    func testActualWorkerDiscoveryFindsAllEightExactObjectiveDescriptors()async throws {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:directory)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:directory,detectCapabilities:false)
        _=try XCTUnwrap(workspace.runtime,workspace.runtimeError ?? "Actual worker required")
        let ready=await withCheckedContinuation {continuation in workspace.refreshCapabilities {continuation.resume(returning:$0)}}
        XCTAssertTrue(ready,workspace.capabilityStatus);XCTAssertEqual(workspace.lossRuntimeCapabilities.componentIDs,LossConfiguration.componentIDs)
    }

    func testDisconnectedLegacyObjectivesOpenForRepairWithoutLosingStoredSettings()throws {
        for id in LossConfiguration.componentIDs.sorted() {
            let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            defer {try? FileManager.default.removeItem(at:directory)}
            let workspace=WorkspaceStore(rootOverride:repository,storageOverride:directory,detectCapabilities:false)
            var doc=try document("loss_cross_entropy")
            let head=try XCTUnwrap(doc.nodes.firstIndex {$0.address == addr("head",.model)})
            doc.nodes[head].payload = .modelPrimitive(type:"linear",configurationVersion:2,config:["in_features":2,"out_features":2,"bias":true])
            let mapping=try XCTUnwrap(doc.nodes.firstIndex {$0.address == addr("mapping")})
            doc.nodes[mapping].payload = .trainingData(config:["featureColumns":.array([.string("a"),.string("b")]),"targetColumn":.string("label")])
            let objective=try XCTUnwrap(doc.nodes.firstIndex {$0.address == addr("objective")})
            let config=try LossConfiguration(componentID:id,values:["reduction":.string("sum")]).values
            doc.nodes[objective].payload = .lossRecipe(type:id,configurationVersion:1,config:config)
            doc.edges.removeAll {$0.kind == .objectiveRecipe}
            // The historical compiler can still read the old document. Current editing
            // cannot silently discard a configured recipe with no Train destination.
            let plan=try WorkflowCompiler.compileHistoricalEvidence(doc,train:WorkflowExamples.trainAddress,modelRegistry:workspace.registry,dataRegistry:DataFlowOperatorRegistry.load(from:repository.appendingPathComponent("runner/dataflow-operators.json")),contractVersion:7)
            XCTAssertNil(plan.lossRecipe)
            let record=try workspace.projectLibrary.saveSource(.workflow(doc),id:doc.projectID)
            let path=URL(fileURLWithPath:record.path),before=try Data(contentsOf:path)
            let savedBefore=try workspace.projectLibrary.loadSource(id:doc.projectID)
            guard case .workflow(let persisted)=savedBefore,
                  let savedObjective=persisted.nodes.first(where:{$0.address == addr("objective")}),
                  case .lossRecipe(let savedID,let savedVersion,let savedConfig)=savedObjective.payload else {return XCTFail("Saved objective must retain its typed recipe")}
            XCTAssertEqual(savedID,id);XCTAssertEqual(savedVersion,1)
            XCTAssertEqual(savedConfig["reduction"],.string("sum"))
            XCTAssertEqual(try LossConfiguration(componentID:savedID,values:savedConfig).values,config,"JSON round-trip must retain all canonical typed settings")
            XCTAssertTrue(workspace.beginWorkflowProject(id:doc.projectID))
            XCTAssertTrue(workspace.statusMessage.contains("Unconnected legacy training recipe"),workspace.statusMessage)
            XCTAssertTrue(workspace.statusMessage.contains("manual project repair"),workspace.statusMessage)
            let host=try XCTUnwrap(workspace.workflowSession)
            XCTAssertNotNil(host.owner.retiredToolRepairMessage)
            XCTAssertTrue(host.owner.document.nodes.contains {$0.address==addr("objective")})
            XCTAssertFalse(host.run(train:WorkflowExamples.trainAddress),"Unresolved recipe must remain non-executable")
            XCTAssertEqual(try Data(contentsOf:path),before,"Repair opening must preserve the original bytes")
            XCTAssertEqual(try workspace.projectLibrary.loadSource(id:doc.projectID),savedBefore)
        }
    }
    func testQueuedTrainAndExperimentsRecheckRevokedObjectiveSupport()async throws {
        func wait(_ condition:()->Bool)async throws {
            let deadline=Date().addingTimeInterval(10)
            while !condition(),Date()<deadline {try await Task.sleep(for:.milliseconds(10))}
            XCTAssertTrue(condition(),"Timed out at objective queue admission boundary")
        }
        for experiments in [false,true] {
            let(w,h,_)=try await fixture("loss_cross_entropy",verified:true,controlledCancellation:false);defer {h.cancel();w.retireWorkflowSession()}
            let queue=h.executionQueue
            let capture=try WorkflowQueueCapture(projectID:h.owner.projectID,title:"Objective gate",kind:.training,requestIdentities:[String(repeating:"a",count:64)],document:h.owner.document,sourceHashes:[])
            let gate=try queue.enqueue(capture:capture,start:{},cancel:{})
            defer {queue.finish(gate,state:.completed,message:"Cleanup")}
            try await wait {queue.entries.first {$0.id == gate}?.state == .running}
            let accounting=try WorkflowStorageAdmission(workspace:w.workspaceDirectory,accountingRoot:w.applicationDirectory)
            let before=try accounting.snapshot().reservedGrowthBytes
            XCTAssertTrue(experiments ? h.runExperiments():h.run(),h.statusMessage)
            try await wait {h.queuedRequestID != nil}
            let queued=try XCTUnwrap(h.queuedRequestID)
            XCTAssertFalse(try XCTUnwrap(queue.entries.first {$0.id == queued}).capture.sourceHashes.isEmpty)
            w.lossRuntimeCapabilities = .init()
            queue.finish(gate,state:.completed,message:"Objective support revoked")
            try await wait {queue.entries.first {$0.id == queued}?.state == .failed && !w.isBusy}
            let failed=try XCTUnwrap(queue.entries.first {$0.id == queued})
            XCTAssertTrue(failed.message.lowercased().contains("objective"),failed.message)
            XCTAssertNil(h.queuedRequestID);XCTAssertEqual(try accounting.snapshot().reservedGrowthBytes,before)
            let stage=w.workspaceDirectory.appendingPathComponent("workflow-source-staging")
            if FileManager.default.fileExists(atPath:stage.path) {XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath:stage.path).isEmpty)}
        }
    }

}
