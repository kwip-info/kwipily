import Foundation
import XCTest
import KwipMLCore
@testable import KwipMLApp

final class BagWorkflowArchiveTests:XCTestCase {
    private let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    func registry()throws->ComponentRegistry {
        var registry=try ComponentRegistry.builtIn()
        // Independent descriptor fixtures retain the shared field gate: these are
        // real declared controls, not empty templates that bypass configuration validation.
        func choice(_ key:String,_ value:String,_ options:[String])->ComponentField {
            .init(key:key,title:key,kind:.choice,defaultValue:.string(value),min:nil,max:nil,
                  choices:options.map {.init(value:$0,title:$0)})
        }
        func flag(_ key:String)->ComponentField {
            .init(key:key,title:key,kind:.boolean,defaultValue:false,min:nil,max:nil)
        }
        let sourceFields:[ComponentField]=[
            choice("layout","fixed",["fixed","packed"]),
            .init(key:"maximum_items_per_bag",title:"Maximum items",defaultValue:4,min:1,max:128),
            flag("per_sample_weights"),flag("include_last_offset")]
        let tableFields:[ComponentField]=[
            .init(key:"num_embeddings",title:"Table rows",defaultValue:16,min:1,max:100_000_000),
            .init(key:"embedding_dim",title:"Dimensions",defaultValue:4,min:1,max:4096),
            choice("mode","mean",["mean","sum","max"]),
            .init(key:"padding_idx",title:"Padding",defaultValue:.null,min:-100_000_000,max:99_999_999,allowsNull:true),
            .init(key:"max_norm",title:"Maximum norm",kind:.number,defaultValue:.null,min:1.1754943508222875e-38,max:3.4028234663852886e38,allowsNull:true),
            .init(key:"norm_type",title:"Norm order",kind:.number,defaultValue:2,min:1.1754943508222875e-38,max:3.4028234663852886e38),
            flag("scale_grad_by_freq"),flag("sparse"),flag("include_last_offset"),flag("freeze"),
            choice("layout","fixed",["fixed","packed"]),flag("per_sample_weights")]
        registry.components.removeAll {["bag_input","embedding_bag"].contains($0.id)}
        registry.components += [
            .init(id:"bag_input",title:"Bag input",inputs:[],outputs:["indices","offsets","per_sample_weights"],fields:sourceFields),
            .init(id:"embedding_bag",title:"Embedding bag",inputs:["indices","offsets","per_sample_weights"],outputs:["out"],fields:tableFields)]
        return registry
    }
    private func object(_ value:JSONValue)throws->[String:JSONValue] {
        guard case .object(let result)=value else {throw GraphConfigurationError("Expected mapping fixture")};return result
    }
    private func address(_ id:String,_ domain:WorkflowDomain = .control)->WorkflowNodeAddress { .init(domain:domain,localID:id) }
    private func raw(packed:Bool=false,weighted:Bool=false)->JSONValue {
        var v:[String:JSONValue]=["schemaVersion":1,"indicesColumns":.array(["i0","i1"].map(JSONValue.string))]
        if packed {v["lengthColumn"] = .string("length")}
        if weighted {v["weightColumns"] = .array(["w0","w1"].map(JSONValue.string))}
        return .object(v)
    }
    private func source(packed:Bool=false,weighted:Bool=false)throws->BagSourceConfiguration {
        var v=BagSourceConfiguration.defaults;v["maximum_items_per_bag"]=2
        v["layout"] = .string(packed ? "packed":"fixed");v["per_sample_weights"] = .bool(weighted)
        v["include_last_offset"] = .bool(packed)
        return try .init(values:v)
    }
    func fixture(packed:Bool=false,weighted:Bool=false,objective:Bool=false)throws->WorkflowDocument {
        var doc=try WorkflowExamples.preparedEmbeddingClassifier(device:"cpu")
        doc.schemaVersion=12;doc.definitions=[];doc.drafts=[]
        doc.nodes.removeAll {$0.address.domain == .model}
        doc.edges.removeAll {$0.from.address.domain == .model || $0.to.address.domain == .model}
        let source=try source(packed:packed,weighted:weighted)
        var table=EmbeddingBagConfiguration.defaults;table["num_embeddings"]=11;table["embedding_dim"]=2
        // JSON has one numeric token for 2; JSONValue decodes that token as integer.
        // Author the wire-canonical numeric representation so full-document equality
        // remains strict, including exact bag mapping, node and wire ordering.
        table["norm_type"] = .integer(2)
        table["layout"] = .string(source.layout);table["per_sample_weights"] = .bool(weighted)
        table["include_last_offset"] = .bool(packed);table["mode"] = .string(weighted ? "sum":"mean")
        doc.nodes += [.init(address:address("bags",.model),payload:.modelPrimitive(type:"bag_input",configurationVersion:2,config:source.values)),
                      .init(address:address("table",.model),payload:.modelPrimitive(type:"embedding_bag",configurationVersion:2,config:table)),
                      .init(address:address("output",.model),payload:.modelPrimitive(type:"output",configurationVersion:2,config:[:]))]
        func wire(_ from:String,_ out:String,_ to:String,_ input:String) {
            doc.edges.append(.init(id:from+out+to+input,from:.init(address:address(from,.model),port:out),to:.init(address:address(to,.model),port:input),kind:.tensor))
        }
        for port in source.outputPorts {wire("bags",port,"table",port)}
        wire("table","out","output","in")
        doc.edges.append(.init(id:"definition",from:.init(address:address("output",.model),port:"definition"),to:.init(address:address("model"),port:"source"),kind:.modelDefinition))
        var config:[String:JSONValue]=["bagInput":raw(packed:packed,weighted:weighted),"targetColumn":.string("label")]
        if objective {
            let target=LossTargetMapping(componentID:"loss_mse",inputColumns:["i0","i1"],roles:[.init(role:"target",columns:["y0","y1"],sampleShape:[2],dtype:.float32)])
            config=["bagInput":raw(packed:packed,weighted:weighted),"lossTargetMapping":try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(target))]
            doc.nodes.append(.init(address:address("objective"),payload:.lossRecipe(type:"loss_mse",configurationVersion:1,config:[:])))
            doc.edges.append(.init(id:"objective",from:.init(address:address("objective"),port:"out"),to:.init(address:WorkflowExamples.trainAddress,port:"objective"),kind:.objectiveRecipe))
        }
        let index=try XCTUnwrap(doc.nodes.firstIndex {$0.address == address("mapping")})
        doc.nodes[index].payload = .trainingData(config:config)
        return doc
    }
    func compile(_ doc:WorkflowDocument,version:Int=11)throws->WorkflowExecutionPlan {
        let model=try registry()
        let data=try DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json"))
        let retired=doc.nodes.contains {node in
            switch node.payload {case .lossRecipe,.gradientPolicyRecipe:return true;default:return false}
        }
        if retired {
            XCTAssertThrowsError(try WorkflowCompiler.compile(doc,train:WorkflowExamples.trainAddress,modelRegistry:model,dataRegistry:data,contractVersion:version),"Captured retired recipes cannot authorize a new run")
        }
        // Preserve original schema12/compiler11 archival source; no migration.
        return try WorkflowCompiler.compileHistoricalEvidence(doc,train:WorkflowExamples.trainAddress,modelRegistry:model,dataRegistry:data,contractVersion:version)
    }
    private let descriptor = try! JSONDecoder().decode(JSONValue.self, from: Data(#"{"version":"1.0.0","adapterID":"kwip.runner.module-composition","graphSchemaVersions":[2],"planSchemaVersion":1,"modelAuthoringSchemaVersion":2,"compositeSourceSchemaVersion":3,"nativeExpansionContractVersion":2,"workflowSchemaVersion":11,"compilerContractVersion":10,"captureFormatVersion":6,"devices":["cpu","mps"],"modes":["branched-structure","nested-structure","parameter-ownership","train-eval","architecture-export","sequential-structure","ordered-structure","named-structure"],"sourceMembers":["torch.nn.Module","torch.nn.Sequential","torch.nn.ModuleList","torch.nn.ModuleDict"],"containerKinds":["graph","sequential","orderedList","namedMap"],"maximumDefinitions":64,"maximumExpandedNodes":128,"maximumEdges":256,"maximumContainers":129,"maximumChildReferences":256,"maximumBoundaryEndpoints":4096,"maximumDepth":8,"maximumPlanBytes":131072,"maximumSourceBytes":2000000,"maximumModelSourceBytes":2000000,"maximumParameters":200000,"parameterOwnership":"expandedNodeRelativeNativeParameter","stateOwnership":"bijectiveNativePersistentState","sharedParameters":false,"sequentialExecution":"nativeForwardValidatedChain","collectionExecution":"explicitDAG","sourceVerification":"independentCapturedRegistryReExpansion","exactResumeDevices":["cpu"],"fineTuneDevices":["cpu","mps"],"mpsResume":"weightsOnly","portableExport":"selfContainedNativeHierarchy","legacyAbsence":"unchanged"}"#.utf8))

    private let pin=String(repeating:"a",count:64)
    private func json<T:Encodable>(_ value:T)throws->[String:Any] {
        try XCTUnwrap(JSONSerialization.jsonObject(with:WorkflowArchiveCodec.encode(value)) as? [String:Any])
    }
    private func capture(_ plan:WorkflowExecutionPlan,version:Int=12)throws->WorkflowArchiveCapture {
        func file(_ path:String)->WorkflowArchiveFile {.init(path:path,bytes:10,sha256:pin)}
        return .init(schemaVersion:version,compilerContractVersion:version == 12 ? 11:10,
            projectID:plan.projectID,directoryID:UUID().uuidString,runID:UUID().uuidString,requestIdentity:pin,
            revision:plan.revision,train:plan.train,workflow:file("workflow.json"),modelRegistry:file("model-registry.json"),dataRegistry:file("data-registry.json"),
            modelRegistrySHA256:pin,dataRegistrySHA256:pin,runtimeEvidence:file("runtime-evidence.json"),
            sources:plan.sourceAddresses.enumerated().map {.init(address:$0.element,path:"inputs/source-\($0.offset).csv",bytes:10,sha256:pin)},
            plan:try WorkflowArchivePlan(plan,compilerContractVersion:version == 12 ? 11:10),captureFormatVersion:version == 12 ? 7:6,moduleCompositionDescriptor:descriptor)
    }
    private func blockFixture()throws->WorkflowDocument {
        var doc=try fixture(packed:true,weighted:true)
        let tableIndex=try XCTUnwrap(doc.nodes.firstIndex {$0.address == address("table",.model)})
        guard case .modelPrimitive(_,_,let config)=doc.nodes[tableIndex].payload else {throw WorkflowArchiveError("Missing table")}
        let reference=CompositeDefinitionReference("lookup")
        let ports=try source(packed:true,weighted:true).outputPorts
        doc.definitions=[.init(reference:reference,title:"Lookup",body:.init(nodes:[.init(id:"table",type:"embedding_bag",config:config)]),
            inputs:ports.map {.init($0,targets:[.init("table",$0)])},outputs:[.init("out",source:.init("table","out"))],
            composition:.init(kind:.namedMap,children:[.init(localID:"table",key:"lookup")]))]
        doc.nodes[tableIndex].payload = .modelBlock(reference:reference,overrides:[:])
        return doc
    }
    func testCompilerElevenProjectionAndCaptureSevenRetainExactPacketMapping()throws {
        XCTAssertEqual(WorkflowArchiveCodec.trainingCompilerContractVersion(forWorkflowSchema:12),11)
        for packed in [false,true] {for weighted in [false,true] {for objective in [false,true] {
            let plan=try compile(fixture(packed:packed,weighted:weighted,objective:objective))
            let archive=try WorkflowArchivePlan(plan,compilerContractVersion:11)
            let mapping=try XCTUnwrap(json(archive)["bagInput"])
            XCTAssertEqual(try JSONDecoder().decode(JSONValue.self,from:JSONSerialization.data(withJSONObject:mapping)),raw(packed:packed,weighted:weighted))
            let value=try capture(plan),bytes=try WorkflowArchiveCodec.encodeCapture(value),restored=try WorkflowArchiveCodec.decodeCapture(bytes)
            XCTAssertEqual(try WorkflowArchiveCodec.encodeCapture(restored),bytes)
            XCTAssertTrue(try restored.plan.matches(plan,compilerContractVersion:11))
            try restored.plan.validateModelSnapshot(registry:registry(),context:objective ? .explicitObjective:.preparedWorkflow)
            XCTAssertThrowsError(try WorkflowArchivePlan(plan,compilerContractVersion:10))
        }}}
    }
    func testSourceThreePacketProvenanceIsIndependentlyReexpanded()throws {
        let plan=try compile(blockFixture()),archive=try WorkflowArchivePlan(plan,compilerContractVersion:11)
        XCTAssertEqual(archive.modelProvenance?.sourceCompositeDocument.schemaVersion,3)
        XCTAssertEqual(archive.modelProvenance?.nativeExpansionContractVersion,2)
        XCTAssertNotNil(archive.modelGraph.compositionPlan)
        try archive.validateModelSnapshot(registry:registry(),context:.preparedWorkflow)
        XCTAssertNoThrow(try WorkflowArchiveCodec.decodeCapture(WorkflowArchiveCodec.encodeCapture(capture(plan))))
        var bad=try json(archive)
        var mapping=try XCTUnwrap(bad["bagInput"] as? [String:Any]);mapping.removeValue(forKey:"weightColumns");bad["bagInput"]=mapping
        let malformed=try JSONDecoder().decode(WorkflowArchivePlan.self,from:JSONSerialization.data(withJSONObject:bad))
        XCTAssertThrowsError(try malformed.validateModelSnapshot(registry:registry(),context:.preparedWorkflow))
    }
    func testCaptureSevenRejectsMissingNullUnknownMappingsRolesAndVersionPairs()throws {
        let value=try capture(compile(fixture(packed:true,weighted:true)))
        for mode in ["missing","null","unknown","weights","columns","pair","format","descriptor","foreignDescriptor"] {
            var payload=try json(value),plan=try XCTUnwrap(payload["plan"] as? [String:Any])
            var mapping=try XCTUnwrap(plan["bagInput"] as? [String:Any])
            switch mode {
            case "missing":plan.removeValue(forKey:"bagInput")
            case "null":plan["bagInput"]=NSNull()
            case "unknown":mapping["foreign"]=true;plan["bagInput"]=mapping
            case "weights":mapping.removeValue(forKey:"weightColumns");plan["bagInput"]=mapping
            case "columns":plan["featureColumns"]=["foreign","items"]
            case "pair":payload["compilerContractVersion"]=10
            case "format":payload["captureFormatVersion"]=6
            case "descriptor":payload.removeValue(forKey:"moduleCompositionDescriptor")
            default:payload["moduleCompositionDescriptor"]=["version":"1.0.0"]
            }
            payload["plan"]=plan
            XCTAssertThrowsError(try WorkflowArchiveCodec.decodeCapture(JSONSerialization.data(withJSONObject:payload)),mode)
        }
    }
    func testLegacyCaptureSixBytesOmitBagMappingAndCannotAcquireIt()throws {
        var doc=try WorkflowExamples.preparedParallelClassifier(device:"cpu");doc.schemaVersion=11
        let plan=try compile(doc,version:10),value=try capture(plan,version:11),bytes=try WorkflowArchiveCodec.encodeCapture(value)
        XCTAssertNil(try json(value.plan)["bagInput"])
        XCTAssertEqual(try WorkflowArchiveCodec.encodeCapture(WorkflowArchiveCodec.decodeCapture(bytes)),bytes)
        for value:Any in [NSNull(),try JSONSerialization.jsonObject(with:JSONEncoder().encode(raw()))] {
            var payload=try json(WorkflowArchiveCodec.decodeCapture(bytes)),archive=try XCTUnwrap(payload["plan"] as? [String:Any]);archive["bagInput"]=value;payload["plan"]=archive
            XCTAssertThrowsError(try WorkflowArchiveCodec.decodeCapture(JSONSerialization.data(withJSONObject:payload)))
        }
    }
    private func makeRuntime(_ directory:URL)throws->RuntimeLocation {
        for name in [".venv/bin","runner","scripts","Sources/KwipMLCore/Resources"] {try FileManager.default.createDirectory(at:directory.appendingPathComponent(name),withIntermediateDirectories:true)}
        try Data("fixture interpreter; never executed".utf8).write(to:directory.appendingPathComponent(".venv/bin/python"))
        try FileManager.default.setAttributes([.posixPermissions:0o700],ofItemAtPath:directory.appendingPathComponent(".venv/bin/python").path)
        for name in ["runner/main.py","runner/engine.py","scripts/runtime_launch.py"] {try Data("# fixture\n".utf8).write(to:directory.appendingPathComponent(name))}
        try JSONEncoder().encode(registry()).write(to:directory.appendingPathComponent("Sources/KwipMLCore/Resources/components.json"))
        try JSONEncoder().encode(DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json"))).write(to:directory.appendingPathComponent("runner/dataflow-operators.json"))
        return try RuntimeLocation.resolve(resources:nil,developmentRoot:directory,allowDevelopment:true)
    }
    @MainActor func testRequestTwelveRequiresRuntimeQualifiedCompositionEvidence()throws {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:directory)}
        let runtime=try makeRuntime(directory.appendingPathComponent("runtime"))
        let library=ProjectLibrary(directory:directory.appendingPathComponent("projects")),doc=try fixture()
        _ = try library.saveSource(.workflow(doc),id:doc.projectID)
        let owner=try WorkflowAuthoringStore(library:library,projectID:doc.projectID,modelRegistry:registry(),dataRegistry:.load(from:root.appendingPathComponent("runner/dataflow-operators.json")))
        XCTAssertThrowsError(try WorkflowExecutionRequest.capture(from:owner,train:WorkflowExamples.trainAddress,runtime:runtime)) {error in
            XCTAssertTrue(error.localizedDescription.contains("verify native module composition"),error.localizedDescription)
        }
    }
    func testArchiveWriterSelectsSevenAndPinsMappingRegistriesAndRealSourceBytes()throws {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
        defer {try? FileManager.default.removeItem(at:directory)}
        let runtime=try makeRuntime(directory.appendingPathComponent("runtime")),doc=try fixture(packed:true,weighted:true)
        let plan=try compile(doc),registry=try registry(),data=try DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json"))
        let request=WorkflowExecutionRequest(lease:.init(projectID:doc.projectID,sessionID:UUID(),generation:0),document:doc,authoringDocument:doc,experiment:nil,experimentInputs:nil,
            train:plan.train,plan:plan,modelRegistrySHA256:try WorkflowArchiveCodec.registryHash(registry),dataRegistrySHA256:try WorkflowArchiveCodec.registryHash(data),runtime:runtime,
            runtimeEvidence:try WorkflowRuntimeEvidence.capture(runtime),requestIdentity:pin,priorCache:nil,gradientArguments:[],checkpointParent:nil,executionBudget:nil,sampleCapture:nil,preparationCacheLimitBytes:nil,capturedModelRegistry:registry,derivativeCapture:nil,moduleCompositionDescriptor:descriptor)
        let sourceURL=directory.appendingPathComponent("source.csv");try Data("i0,i1,w0,w1,length,label\n1,2,1,1,2,0\n".utf8).write(to:sourceURL)
        let source=try WorkflowCapturedSource.capture(sourceURL),runDirectory=directory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:runDirectory,withIntermediateDirectories:true)
        let result=try WorkflowArchiveCaptureWriter.write(request:request,runID:UUID(),sources:[WorkflowExamples.sourceAddress:source],directory:runDirectory,modelRegistry:registry,dataRegistry:data)
        let bytes=try Data(contentsOf:runDirectory.appendingPathComponent(result.reference.path)),capture=try WorkflowArchiveCodec.decodeCapture(bytes)
        XCTAssertEqual(capture.schemaVersion,12);XCTAssertEqual(capture.compilerContractVersion,11);XCTAssertEqual(capture.captureFormatVersion,7)
        XCTAssertTrue(try capture.plan.matches(plan,compilerContractVersion:11))
        XCTAssertEqual(capture.sources.first?.sha256,source.sha256)
        XCTAssertEqual(capture.moduleCompositionDescriptor,descriptor)
        XCTAssertEqual(capture.modelRegistrySHA256,request.modelRegistrySHA256)
    }
}
