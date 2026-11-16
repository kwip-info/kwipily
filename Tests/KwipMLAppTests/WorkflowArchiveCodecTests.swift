import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowArchiveCodecTests: XCTestCase {
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
    private let pin = String(repeating: "a", count: 64)
    private func capture() throws -> WorkflowArchiveCapture {
        let document = fixture(block: true)
        let plan = try WorkflowCompiler.compileHistoricalEvidence(document, train: address(.control,"train"), modelRegistry: .builtIn(), dataRegistry: dataRegistry())
        func file(_ path: String) -> WorkflowArchiveFile { .init(path: path, bytes: 10, sha256: pin) }
        return WorkflowArchiveCapture(projectID: document.projectID, directoryID: UUID().uuidString,
            runID: UUID().uuidString, requestIdentity: pin, revision: document.revision, train: plan.train,
            workflow: file("workflow.json"), modelRegistry: file("model-registry.json"), dataRegistry: file("data-registry.json"),
            modelRegistrySHA256: pin, dataRegistrySHA256: pin, runtimeEvidence: file("runtime-evidence.json"),
            sources: [.init(address: address(.data,"input"), path: "inputs/source-0.csv", bytes: 12, sha256: pin)],
            plan: try WorkflowArchivePlan(plan, compilerContractVersion: 1))
    }
    private func mutate(_ bytes: Data, _ edit: (inout [String:Any])->Void) throws -> Data {
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with: bytes) as? [String:Any])
        edit(&object)
        return try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    }
    func testCaptureRoundTripRetainsDerivedNestedPlanWithoutLiveOwner() throws {
        let value = try capture()
        let bytes = try WorkflowArchiveCodec.encodeCapture(value)
        XCTAssertEqual(try WorkflowArchiveCodec.decodeCapture(bytes), value)
        XCTAssertNotNil(value.plan.modelProvenance)
        XCTAssertFalse(value.plan.modelOrigins.isEmpty)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: bytes) as? [String:Any])
        XCTAssertNil(object["executable"])
        XCTAssertNil(object["workingDirectory"])
        XCTAssertEqual((object["plan"] as? [String:Any])?["projectID"] as? String, value.projectID)
    }
    func testCaptureRejectsUnknownKeysTypesPathsAndIdentityDisagreement() throws {
        let bytes = try WorkflowArchiveCodec.encodeCapture(capture())
        let changes: [(inout [String:Any])->Void] = [
            { $0["extra"] = true }, { $0["schemaVersion"] = true }, { $0["revision"] = -1 },
            { $0["compilerContractVersion"] = 2 }, { $0["projectID"] = UUID().uuidString },
            { $0["requestIdentity"] = String(repeating:"A", count:64) },
            { $0["workflow"] = ["path":"../workflow.json","bytes":10,"sha256":self.pin] },
            { $0["workflow"] = ["path":"workflow.json","bytes":2_000_001,"sha256":self.pin] },
            { $0["sources"] = [["address":["domain":"model","localID":"input"],"path":"inputs/source-0.csv","bytes":12,"sha256":self.pin]] },
            { $0["sources"] = [] }
        ]
        for (index, change) in changes.enumerated() {
            XCTAssertThrowsError(try WorkflowArchiveCodec.decodeCapture(mutate(bytes, change)), "Mutation \(index)")
        }
        XCTAssertThrowsError(try WorkflowArchiveCodec.decodeCapture(Data(repeating: 32,count:8_000_001)))
    }
    func testCancelledAndEarlyFailureRemainTerminalWithoutInventedModelEvidence() throws {
        let project = UUID().uuidString, directory = UUID().uuidString, run = UUID().uuidString
        for status in [WorkflowExecutionPhase.failed, .cancelled] {
            let value = WorkflowArchiveOutcome(projectID:project,directoryID:directory,runID:run,requestIdentity:pin,
                status:status,terminalPhase:.preparing,startedAt:"2026-10-05T12:00:00.000Z",finishedAt:"2026-10-05T12:00:01.000Z",
                artifacts:[],error:status == .failed ? "Source inspection failed":nil)
            let bytes = try WorkflowArchiveCodec.encodeOutcome(value)
            XCTAssertEqual(try WorkflowArchiveCodec.decodeOutcome(bytes),value)
            XCTAssertThrowsError(try WorkflowArchiveCodec.decodeOutcome(mutate(bytes) { $0["status"] = "completed" }))
            XCTAssertThrowsError(try WorkflowArchiveCodec.decodeOutcome(mutate(bytes) { $0["status"] = "training" }))
            XCTAssertThrowsError(try WorkflowArchiveCodec.decodeOutcome(mutate(bytes) { $0["modelManifestSHA256"] = self.pin }))
            XCTAssertThrowsError(try WorkflowArchiveCodec.decodeOutcome(mutate(bytes) { $0["finishedAt"] = "2026-10-05T11:59:59.000Z" }))
        }
    }
    func testCompletedOutcomeRequiresExactManifestPinsAndBoundedInventory() throws {
        let files = ["preparation/artifacts/manifest.json", "training/artifacts/manifest.json", "training/artifacts/prediction-table.json"].map {
            WorkflowArchiveFile(path:$0,bytes:20,sha256:pin)
        }
        let value = WorkflowArchiveOutcome(projectID:UUID().uuidString,directoryID:UUID().uuidString,runID:UUID().uuidString,
            requestIdentity:pin,status:.completed,terminalPhase:.training,
            startedAt:"2026-10-05T12:00:00.000Z",finishedAt:"2026-10-05T12:00:01.000Z",artifacts:files,
            capture:.init(path:"capture.json",bytes:100,sha256:pin),preparationManifestSHA256:pin,
            modelManifestSHA256:pin,workerRunID:UUID().uuidString)
        let bytes = try WorkflowArchiveCodec.encodeOutcome(value)
        XCTAssertEqual(try WorkflowArchiveCodec.decodeOutcome(bytes),value)
        let changes: [(inout [String:Any])->Void] = [
            { $0.removeValue(forKey:"capture") }, { $0.removeValue(forKey:"workerRunID") },
            { $0["modelManifestSHA256"] = String(repeating:"b",count:64) },
            { $0["error"] = "Cannot be completed" }, { $0["workerRunID"] = NSNull() },
            { $0["artifacts"] = Array(repeating:["path":"training/artifacts/model.py","bytes":20,"sha256":self.pin],count:321) },
            { $0["artifacts"] = [["path":"training/artifacts/../../secret","bytes":20,"sha256":self.pin]] }
        ]
        for change in changes { XCTAssertThrowsError(try WorkflowArchiveCodec.decodeOutcome(mutate(bytes,change))) }
    }
    func testCaptureBudgetProvenanceRoundTripsWithoutChangingAuthoringOrCompilerVersion()throws {
        var value=try capture()
        let legacyBytes=try WorkflowArchiveCodec.encodeCapture(value)
        let legacy=try WorkflowArchiveCodec.decodeCapture(legacyBytes)
        XCTAssertNil(legacy.captureFormatVersion)
        XCTAssertNil(legacy.executionBudget)
        XCTAssertEqual(try WorkflowArchiveCodec.encodeCapture(legacy),legacyBytes)
        let legacyObject=try XCTUnwrap(JSONSerialization.jsonObject(with:legacyBytes) as? [String:Any])
        XCTAssertNil(legacyObject["captureFormatVersion"])
        XCTAssertNil(legacyObject["executionBudget"])
        let sourceVersion=value.schemaVersion,compiler=value.compilerContractVersion
        let policy=try ResolvedExecutionBudget.resolve(global:.init(maximumRunSeconds:3600),projectMaximumRunSeconds:120,capabilityMaximumSeconds:90)
        value.captureFormatVersion=2
        value.executionBudget=policy
        let bytes=try WorkflowArchiveCodec.encodeCapture(value)
        let decoded=try WorkflowArchiveCodec.decodeCapture(bytes)
        XCTAssertEqual(decoded,value)
        XCTAssertEqual(decoded.schemaVersion,sourceVersion)
        XCTAssertEqual(decoded.compilerContractVersion,compiler)
        XCTAssertEqual(decoded.executionBudget?.requestedSeconds,120)
        XCTAssertEqual(decoded.executionBudget?.maximumRunSeconds,90)
        XCTAssertEqual(decoded.executionBudget?.source,.project)
    }

    func testCaptureBudgetRejectsPartialUnsupportedAndForgedProvenance()throws {
        var value=try capture()
        value.captureFormatVersion=2
        value.executionBudget=try .resolve(global:.init(maximumRunSeconds:3600),projectMaximumRunSeconds:120,capabilityMaximumSeconds:90)
        let bytes=try WorkflowArchiveCodec.encodeCapture(value)
        let changes:[(inout [String:Any])->Void]=[
            {$0.removeValue(forKey:"captureFormatVersion")},
            {$0.removeValue(forKey:"executionBudget")},
            {$0["captureFormatVersion"]=1},{$0["captureFormatVersion"]=3},
            {$0["captureFormatVersion"]=true},{$0["captureFormatVersion"]=NSNull()},
            {$0["executionBudget"]=NSNull()},
            {var budget=$0["executionBudget"] as! [String:Any];budget["maximumRunSeconds"]=120;$0["executionBudget"]=budget},
            {var budget=$0["executionBudget"] as! [String:Any];budget["requestedSeconds"] = -1;$0["executionBudget"]=budget},
            {var budget=$0["executionBudget"] as! [String:Any];budget["extra"]=true;$0["executionBudget"]=budget},
            {var budget=$0["executionBudget"] as! [String:Any];budget.removeValue(forKey:"source");$0["executionBudget"]=budget}
        ]
        for (index,change) in changes.enumerated() {
            XCTAssertThrowsError(try WorkflowArchiveCodec.decodeCapture(mutate(bytes,change)),"Budget mutation \(index)")
        }
        value.captureFormatVersion=nil
        XCTAssertThrowsError(try WorkflowArchiveCodec.encodeCapture(value))
        value.captureFormatVersion=2;value.executionBudget=nil
        XCTAssertThrowsError(try WorkflowArchiveCodec.encodeCapture(value))
    }

}
