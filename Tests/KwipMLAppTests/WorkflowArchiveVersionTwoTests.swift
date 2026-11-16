import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowArchiveVersionTwoTests: XCTestCase {
    private func address(_ domain:WorkflowDomain,_ id:String)->WorkflowNodeAddress {.init(domain:domain,localID:id)}
    private func edge(_ a:WorkflowDomain,_ id:String,_ port:String,_ b:WorkflowDomain,_ target:String,_ input:String,_ kind:WorkflowEdgeKind)->WorkflowEdge {
        .init(id:"\(a)-\(id)-\(port)-\(b)-\(target)-\(input)",from:.init(address:address(a,id),port:port),to:.init(address:address(b,target),port:input),kind:kind)
    }
    private func fixture(block:Bool=false)->WorkflowDocument {
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
    /// Captured historical descriptors support integrity replay, never new authoring.
    private func dataRegistry() throws -> DataFlowOperatorRegistry {
        var captured=try DataFlowOperatorRegistry.load(from:repo.appendingPathComponent("runner/dataflow-operators.json"))
        let retired=try DataFlowOperatorRegistry.load(from:repo.appendingPathComponent("runner/retired-data-operators-v1.json"))
        XCTAssertEqual(captured.packageID,retired.packageID)
        XCTAssertTrue(Set(captured.operators.map(\.id)).isDisjoint(with:retired.operators.map(\.id)))
        captured.operators += retired.operators
        return captured
    }

    private let digest = String(repeating:"a",count:64)
    private func makeCapture(version:Int,policy:WorkflowPreparationCachePolicy = .off) throws -> WorkflowArchiveCapture {
        var document=fixture(block:true);document.schemaVersion=version
        if version == 2 {
            let index=try XCTUnwrap(document.nodes.firstIndex{$0.address == address(.control,"setup")})
            guard case .runSetup(var config)=document.nodes[index].payload else {throw DataFlowContractError("Missing fixture setup")}
            config["preparationCachePolicy"] = .string(policy.rawValue)
            document.nodes[index].payload = .runSetup(config:config)
        }
        let plan=try WorkflowCompiler.compileHistoricalEvidence(document,train:address(.control,"train"),modelRegistry:.builtIn(),dataRegistry:dataRegistry())
        func file(_ path:String)->WorkflowArchiveFile {.init(path:path,bytes:10,sha256:digest)}
        return WorkflowArchiveCapture(schemaVersion:version,compilerContractVersion:version,projectID:document.projectID,
            directoryID:UUID().uuidString,runID:UUID().uuidString,requestIdentity:digest,revision:document.revision,train:plan.train,
            workflow:file("workflow.json"),modelRegistry:file("model-registry.json"),dataRegistry:file("data-registry.json"),
            modelRegistrySHA256:digest,dataRegistrySHA256:digest,runtimeEvidence:file("runtime-evidence.json"),
            sources:[.init(address:address(.data,"input"),path:"inputs/source-0.csv",bytes:20,sha256:digest)],
            plan:try WorkflowArchivePlan(plan,compilerContractVersion:version),
            priorCache:policy == .reuse ? .init(projectID:document.projectID,directoryID:UUID().uuidString,preparationManifestSHA256:digest):nil)
    }
    private func mutate(_ bytes:Data,_ edit:(inout [String:Any])->Void)throws->Data {
        var object=try XCTUnwrap(JSONSerialization.jsonObject(with:bytes) as? [String:Any]);edit(&object)
        return try JSONSerialization.data(withJSONObject:object,options:[.sortedKeys,.withoutEscapingSlashes])
    }
    func testVersionThreeRequiresExactLifecycleParentAndPreservesLegacyFields()throws {
        for mode in WorkflowTrainingMode.allCases {
            var document=try WorkflowCodec.migrateToVersion3(fixture())
            let index=try XCTUnwrap(document.nodes.firstIndex {$0.address==address(.control,"train")})
            document.nodes[index].payload = .train(config:["trainingMode":.string(mode.rawValue)])
            let plan=try WorkflowCompiler.compileHistoricalEvidence(document,train:address(.control,"train"),modelRegistry:.builtIn(),dataRegistry:dataRegistry(),contractVersion:3)
            let legacy=try makeCapture(version:2)
            var capture=WorkflowArchiveCapture(schemaVersion:3,compilerContractVersion:3,projectID:document.projectID,
                directoryID:legacy.directoryID,runID:legacy.runID,requestIdentity:legacy.requestIdentity,revision:document.revision,
                train:plan.train,workflow:legacy.workflow,modelRegistry:legacy.modelRegistry,dataRegistry:legacy.dataRegistry,
                modelRegistrySHA256:legacy.modelRegistrySHA256,dataRegistrySHA256:legacy.dataRegistrySHA256,
                runtimeEvidence:legacy.runtimeEvidence,sources:legacy.sources,plan:try WorkflowArchivePlan(plan,compilerContractVersion:3))
            let parent=WorkflowCheckpointReference(projectID:document.projectID,directoryID:UUID().uuidString,
                runID:UUID().uuidString,requestIdentity:digest,modelManifestSHA256:digest)
            if mode != .fresh {
                XCTAssertThrowsError(try WorkflowArchiveCodec.encodeCapture(capture))
                capture.checkpointParent=parent
                XCTAssertThrowsError(try WorkflowArchiveCodec.encodeCapture(capture))
                capture.checkpointFiles=WorkflowCheckpointSnapshot.artifactLimits.keys.sorted().map {
                    .init(path:"continuation-parent/"+$0,bytes:10,sha256:digest)
                }
            }
            let bytes=try WorkflowArchiveCodec.encodeCapture(capture)
            XCTAssertEqual(try WorkflowArchiveCodec.decodeCapture(bytes),capture)
            XCTAssertThrowsError(try WorkflowArchiveCodec.decodeCapture(mutate(bytes) { object in
                var plan=object["plan"] as! [String:Any];plan.removeValue(forKey:"trainingMode");object["plan"]=plan
            }))
            if mode != .fresh {
                let complete=capture.checkpointFiles!
                for malformed in [Array(complete.dropLast()),complete+[complete[0]],
                    complete.map {.init(path:$0.path,bytes:100_000_001,sha256:$0.sha256)},
                    complete.map {.init(path:$0.path,bytes:$0.bytes,sha256:String(repeating:"c",count:64))}] as [[WorkflowArchiveFile]] {
                    var bad=capture;bad.checkpointFiles=malformed
                    XCTAssertThrowsError(try WorkflowArchiveCodec.encodeCapture(bad))
                }
                for field in ["projectID","directoryID","runID","requestIdentity","modelManifestSHA256"] {
                    XCTAssertThrowsError(try WorkflowArchiveCodec.decodeCapture(mutate(bytes) {object in
                        var parent=object["checkpointParent"] as! [String:Any];parent[field]="invalid";object["checkpointParent"]=parent
                    }))
                }
                XCTAssertThrowsError(try WorkflowArchiveCodec.decodeCapture(mutate(bytes) {object in
                    var parent=object["checkpointParent"] as! [String:Any];parent["path"]="/private/path";object["checkpointParent"]=parent
                }))
            }
            for version in [1,2] {
                var old=capture;old.schemaVersion=version;old.compilerContractVersion=version
                XCTAssertThrowsError(try WorkflowArchiveCodec.encodeCapture(old))
            }
            if mode == .fresh {
                capture.checkpointParent=parent
                XCTAssertThrowsError(try WorkflowArchiveCodec.encodeCapture(capture))
            } else {
                capture.checkpointParent = .init(projectID:document.projectID,directoryID:capture.directoryID,
                    runID:parent.runID,requestIdentity:digest,modelManifestSHA256:digest)
                XCTAssertThrowsError(try WorkflowArchiveCodec.encodeCapture(capture))
            }
        }
    }
    func testLegacyArchiveCannotDiscardContinuationMode()throws {
        var document=try WorkflowCodec.migrateToVersion3(fixture())
        let index=try XCTUnwrap(document.nodes.firstIndex {$0.address==address(.control,"train")})
        for mode in [WorkflowTrainingMode.exactResume,.fineTune] {
            document.nodes[index].payload = .train(config:["trainingMode":.string(mode.rawValue)])
            let plan=try WorkflowCompiler.compileHistoricalEvidence(document,train:address(.control,"train"),modelRegistry:.builtIn(),dataRegistry:dataRegistry(),contractVersion:3)
            for version in [1,2] {XCTAssertThrowsError(try WorkflowArchivePlan(plan,compilerContractVersion:version))}
        }
    }
    func testVersionOneRetainsExactLegacyKeysAndRoundTripBytes()throws {
        let value=try makeCapture(version:1)
        let bytes=try WorkflowArchiveCodec.encodeCapture(value)
        let object=try XCTUnwrap(JSONSerialization.jsonObject(with:bytes) as? [String:Any])
        XCTAssertNil(object["priorCache"])
        let plan=try XCTUnwrap(object["plan"] as? [String:Any])
        XCTAssertEqual(Set(plan.keys),Set(["projectID","revision","train","dataflow","sourceAddresses","modelGraph","training","validation","featureColumns","targetColumn","settings","modelSemanticSHA256","modelOrigins","modelProvenance","modelAuthoringRevision"]))
        XCTAssertEqual(Set(object.keys),Set(["schemaVersion","compilerContractVersion","projectID","directoryID","runID","requestIdentity","revision","train","workflow","modelRegistry","dataRegistry","modelRegistrySHA256","dataRegistrySHA256","runtimeEvidence","sources","plan"]))
        // Historical archives used JSONEncoder, not JSONSerialization. Freeze the
        // original two Codable shapes and compare the actual historical encoder.
        let legacyValue = try JSONDecoder().decode(LegacyCapture.self, from: bytes)
        let legacyEncoder = JSONEncoder()
        legacyEncoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        let legacyBytes = try legacyEncoder.encode(legacyValue)
        XCTAssertEqual(legacyBytes,bytes)
        XCTAssertEqual(try WorkflowArchiveCodec.encodeCapture(WorkflowArchiveCodec.decodeCapture(legacyBytes)),legacyBytes)
        XCTAssertThrowsError(try WorkflowArchiveCodec.decodeCapture(mutate(bytes){ object in
            var plan=object["plan"] as! [String:Any];plan["preparationCachePolicy"]="off";object["plan"]=plan
        }))
    }
    func testVersionTwoPreservesExplicitPolicyAndOnlyReuseReference()throws {
        for policy in WorkflowPreparationCachePolicy.allCases {
            let value=try makeCapture(version:2,policy:policy)
            let bytes=try WorkflowArchiveCodec.encodeCapture(value)
            let restored=try WorkflowArchiveCodec.decodeCapture(bytes)
            XCTAssertEqual(restored,value)
            XCTAssertEqual(restored.plan.preparationCachePolicy,policy)
            XCTAssertEqual(restored.priorCache != nil,policy == .reuse)
        }
    }
    func testRejectsVersionPairsUnknownVersionAndMissingPolicy()throws {
        let bytes=try WorkflowArchiveCodec.encodeCapture(makeCapture(version:2))
        let edits:[(inout [String:Any])->Void]=[
            {$0["compilerContractVersion"]=1},{$0["schemaVersion"]=1},{$0["compilerContractVersion"]=3},
            {object in var plan=object["plan"] as! [String:Any];plan.removeValue(forKey:"preparationCachePolicy");object["plan"]=plan},
            {object in var plan=object["plan"] as! [String:Any];plan["preparationCachePolicy"]="unknown";object["plan"]=plan}
        ]
        for edit in edits {XCTAssertThrowsError(try WorkflowArchiveCodec.decodeCapture(mutate(bytes,edit)))}
        var document=fixture();document.schemaVersion=2
        let index=try XCTUnwrap(document.nodes.firstIndex{$0.address == address(.control,"setup")})
        document.nodes[index].payload = .runSetup(config:["preparationCachePolicy":.string("materialize")])
        let plan=try WorkflowCompiler.compileHistoricalEvidence(document,train:address(.control,"train"),modelRegistry:.builtIn(),dataRegistry:dataRegistry())
        XCTAssertThrowsError(try WorkflowArchivePlan(plan,compilerContractVersion:1))
        XCTAssertThrowsError(try WorkflowArchivePlan(plan,compilerContractVersion:99))
    }
    func testReferencesAreQualifiedManagedIdentitiesWithNoPrivatePaths()throws {
        let value=try makeCapture(version:2,policy:.reuse),bytes=try WorkflowArchiveCodec.encodeCapture(value)
        let edits:[(inout [String:Any])->Void]=[
            {$0.removeValue(forKey:"priorCache")},
            {object in var ref=object["priorCache"] as! [String:Any];ref["projectID"]=UUID().uuidString;object["priorCache"]=ref},
            {object in var ref=object["priorCache"] as! [String:Any];ref["directoryID"]=value.directoryID;object["priorCache"]=ref},
            {object in var ref=object["priorCache"] as! [String:Any];ref["preparationManifestSHA256"]="bad";object["priorCache"]=ref},
            {object in var ref=object["priorCache"] as! [String:Any];ref["path"]="/private/source";object["priorCache"]=ref},
            {object in var plan=object["plan"] as! [String:Any];plan["preparationCachePolicy"]="off";object["plan"]=plan}
        ]
        for edit in edits {XCTAssertThrowsError(try WorkflowArchiveCodec.decodeCapture(mutate(bytes,edit)))}
    }
    private struct LegacyPlan: Codable {
        let projectID: String
        let revision: UInt64
        let train: WorkflowNodeAddress
        let dataflow: DataFlowDocument
        let sourceAddresses: [WorkflowNodeAddress]
        let modelGraph: GraphDocument
        let training: DataFlowEndpoint
        let validation: DataFlowEndpoint
        let featureColumns: [String]
        let targetColumn: String
        let settings: ExecutionSettings
        let modelSemanticSHA256: String
        let modelOrigins: [WorkflowArchiveOrigin]
        let modelProvenance: CompositeSourceSnapshot?
        let modelAuthoringRevision: UInt64?
    }
    private struct LegacyCapture: Codable {
        var schemaVersion: Int = 1
        var compilerContractVersion: Int = 1
        let projectID: String
        let directoryID: String
        let runID: String
        let requestIdentity: String
        let revision: UInt64
        let train: WorkflowNodeAddress
        let workflow: WorkflowArchiveFile
        let modelRegistry: WorkflowArchiveFile
        let dataRegistry: WorkflowArchiveFile
        let modelRegistrySHA256: String
        let dataRegistrySHA256: String
        let runtimeEvidence: WorkflowArchiveFile
        let sources: [WorkflowArchiveSource]
        let plan: LegacyPlan
    }

}
