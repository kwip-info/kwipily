import XCTest
@testable import KwipMLCore

final class WorkflowDocumentTests: XCTestCase {
    private let id="11111111-1111-1111-1111-111111111111"
    private func address(_ domain:WorkflowDomain,_ id:String)->WorkflowNodeAddress {.init(domain:domain,localID:id)}
    func testMixedCanonicalRoundTripAndQualifiedIdentities()throws {
        let data=address(.data,"input"),model=address(.model,"input"),setup=address(.control,"setup")
        let document=WorkflowDocument(projectID:id,nodes:[
            .init(address:data,name:"Source",x:1,y:2,payload:.dataOperator(operatorID:"kwip.data.sourceCSV",config:["separator":.string(";")])),
            .init(address:model,name:"Features",x:3,y:4,payload:.modelPrimitive(type:"input",configurationVersion:1,config:["features":2])),
            .init(address:setup,payload:.runSetup(config:["seed":.integer(9007199254740993)]))
        ],edges:[.init(id:"incomplete",from:.init(address:data,port:"table"),to:.init(address:address(.control,"missing"),port:"training"),kind:.table)])
        let encoded=try WorkflowCodec.encode(document)
        XCTAssertEqual(try WorkflowCodec.decode(encoded),document)
        XCTAssertEqual(document.nodes[0].address.localID,document.nodes[1].address.localID)
        XCTAssertNotEqual(document.nodes[0].address,document.nodes[1].address)
        let text=String(decoding:encoded,as:UTF8.self)
        XCTAssertTrue(text.contains("workflowProject"));XCTAssertFalse(text.contains("sourceBindings"))
        XCTAssertTrue(text.contains("9007199254740993"))
    }
    func testModelMigrationKeepsLegacyConfigurationAndExecutionRemainder()throws {
        var graph=ExampleFlows.parallelClassifier;graph.schemaVersion=1
        for i in graph.nodes.indices {graph.nodes[i].config.removeValue(forKey:"bias")}
        graph.execution=ExecutionSettings(datasetCSV:"/private/source.csv",seed:123)
        let before=graph
        let result=try WorkflowCodec.migrateModel(.legacyFlat(graph),projectID:id)
        XCTAssertEqual(result.retainedExecution,graph.execution)
        XCTAssertTrue(result.sourceBindings.isEmpty)
        XCTAssertEqual(result.document.nodes.map(\.address.localID),graph.nodes.map(\.id))
        XCTAssertEqual(result.document.edges.map(\.id),graph.edges.map(\.id))
        for (node,original) in zip(result.document.nodes,graph.nodes) {
            XCTAssertEqual(node.name,original.name);XCTAssertEqual(node.x,original.x)
            guard case .modelPrimitive(let type,let version,let config)=node.payload else{return XCTFail("Expected primitive")}
            XCTAssertEqual(type,original.type);XCTAssertEqual(version,1);XCTAssertEqual(config,original.config)
        }
        let text=String(decoding:try WorkflowCodec.encode(result.document),as:UTF8.self)
        XCTAssertFalse(text.contains("/private/source.csv"));XCTAssertFalse(text.contains("preparedTraining"));XCTAssertFalse(text.contains("runSetup"))
        XCTAssertEqual(graph,before)
    }
    func testDataMigrationRetainsPrivateBindingsOutsidePortableSource()throws {
        let source=DataFlowProject(name:"Private",graph:.init(nodes:[.init(id:"csv",operatorID:"kwip.data.sourceCSV",config:["separator":.string(",")])]),editor:["csv":.init(name:"My data",x:40,y:70)],sourceBindings:["csv":"/private/source.csv"])
        let result=try WorkflowCodec.migrateData(source,projectID:id)
        XCTAssertEqual(result.sourceBindings[address(.data,"csv")],"/private/source.csv")
        XCTAssertNil(result.retainedExecution)
        XCTAssertEqual(result.document.nodes[0].name,"My data");XCTAssertEqual(result.document.nodes[0].x,40)
        XCTAssertFalse(String(decoding:try WorkflowCodec.encode(result.document),as:UTF8.self).contains("/private/"))
        var unknown=result.document
        unknown.nodes.append(.init(address:address(.data,"future"),payload:.dataOperator(operatorID:"future.unresolved",config:["custom":.array([1,true])])) )
        XCTAssertNoThrow(try WorkflowCodec.encode(unknown))
    }
    func testConnectedDataMigrationRequiresAuthoritativePortRegistry()throws {
        let unknown=DataFlowProject(graph:.init(nodes:[
            .init(id:"future",operatorID:"future.stateProducer"),
            .init(id:"consumer",operatorID:"future.stateConsumer")
        ],edges:[.init(from:.init(node:"future",port:"state"),to:.init(node:"consumer",port:"state"))]))
        XCTAssertThrowsError(try WorkflowCodec.migrateData(unknown,projectID:id),"Migration cannot guess that an unknown state output is a table")
        let known=DataFlowProject(graph:.init(nodes:[
            .init(id:"split",operatorID:"kwip.data.split"),
            .init(id:"fit",operatorID:"kwip.data.standardScalerFit"),
            .init(id:"transform",operatorID:"kwip.data.transform")
        ],edges:[
            .init(from:.init(node:"split",port:"training"),to:.init(node:"fit",port:"training")),
            .init(from:.init(node:"fit",port:"state"),to:.init(node:"transform",port:"state"))
        ]))
        XCTAssertThrowsError(try WorkflowCodec.migrateData(known,projectID:id),"Even familiar names require the captured authoritative registry rather than a duplicated operator switch")
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        // This fixture proves decoding of an original captured legacy contract, not current execution.
        var registry=try DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/retired-data-operators-v1.json"))
        let migrated=try WorkflowCodec.migrateData(known,projectID:id,registry:registry)
        XCTAssertEqual(migrated.document.edges.map(\.kind),[.dataset,.fittedState])
        XCTAssertEqual(migrated.document.edges.map(\.id),known.graph.edges.map(\.id))
        XCTAssertThrowsError(try WorkflowCodec.migrateData(unknown,projectID:id,registry:registry))
        // A renamed operator with the same authoritative port contract still classifies correctly.
        let index=try XCTUnwrap(registry.operators.firstIndex{$0.id=="kwip.data.standardScalerFit"})
        registry.operators[index].id="future.fittedTransformProducer"
        var renamed=known;renamed.graph.nodes[1].operatorID="future.fittedTransformProducer"
        XCTAssertEqual(try WorkflowCodec.migrateData(renamed,projectID:id,registry:registry).document.edges.map(\.kind),[.dataset,.fittedState])
        registry.operators[index].outputs[0].stateKind="optimizer"
        XCTAssertThrowsError(try WorkflowCodec.migrateData(renamed,projectID:id,registry:registry))
    }
    func testCompositeDraftMigrationPreservesPinsAndIncompleteBody()throws {
        var model=ModelAuthoringDocument(projectID:id,revision:4,source:CompositeExamples.parallelClassifier)
        model=try ModelBlockEditing.beginDraft(model,path:["features"])
        var proposed=model.drafts[0].proposed;proposed.body.edges=[]
        model=try ModelBlockEditing.updateDraft(model,proposed:proposed,expectedBaseRevision:4,path:["features"])
        let migrated=try WorkflowCodec.migrateModel(.blockModel(model),projectID:id).document
        XCTAssertEqual(migrated.revision,4);XCTAssertEqual(migrated.definitions,model.source.definitions);XCTAssertEqual(migrated.drafts,model.drafts)
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(migrated)),migrated)
        XCTAssertThrowsError(try WorkflowCodec.migrateModel(.blockModel(model),projectID:UUID().uuidString))
        var invalid=migrated;invalid.drafts[0].baseRevision=3
        XCTAssertThrowsError(try WorkflowCodec.encode(invalid))
    }
    func testStrictTaggedStorageRejectsUnknownFieldsDomainMismatchPrivateSettingsAndDuplicates()throws {
        let node=WorkflowNode(address:address(.data,"csv"),payload:.dataOperator(operatorID:"kwip.data.sourceCSV",config:[:]))
        let valid=WorkflowDocument(projectID:id,nodes:[node])
        var object=try XCTUnwrap(JSONSerialization.jsonObject(with:WorkflowCodec.encode(valid)) as? [String:Any])
        object["unexpected"]=true
        XCTAssertThrowsError(try WorkflowCodec.decode(JSONSerialization.data(withJSONObject:object)))
        object.removeValue(forKey:"unexpected");object["kind"]="modelProject"
        XCTAssertThrowsError(try WorkflowCodec.decode(JSONSerialization.data(withJSONObject:object)))
        var privateSource=valid;privateSource.nodes[0].payload = .dataOperator(operatorID:"kwip.data.sourceCSV",config:["path":.string("/private/source.csv")])
        XCTAssertThrowsError(try WorkflowCodec.encode(privateSource))
        var wrong=valid;wrong.nodes[0].address.domain = .model
        XCTAssertThrowsError(try WorkflowCodec.encode(wrong))
        var duplicate=valid;duplicate.nodes.append(node)
        XCTAssertThrowsError(try WorkflowCodec.encode(duplicate))
        let setup=WorkflowNode(address:address(.control,"run"),payload:.runSetup(config:["preparedTraining":.object(["manifestPath":.string("/private/run")])]))
        XCTAssertThrowsError(try WorkflowCodec.encode(.init(projectID:id,nodes:[setup])))
        var raw=try XCTUnwrap(JSONSerialization.jsonObject(with:WorkflowCodec.encode(valid)) as? [String:Any])
        var nodes=try XCTUnwrap(raw["nodes"] as? [[String:Any]])
        var payload=try XCTUnwrap(nodes[0]["payload"] as? [String:Any]);payload["futureField"]=true;nodes[0]["payload"]=payload;raw["nodes"]=nodes
        XCTAssertThrowsError(try WorkflowCodec.decode(JSONSerialization.data(withJSONObject:raw)))
    }
    func testBoundsFiniteConfigurationAndDuplicateTargets()throws {
        var document=WorkflowDocument(projectID:id)
        document.nodes=(0..<129).map{.init(address:address(.model,"n\($0)"),payload:.modelPrimitive(type:"relu",configurationVersion:2,config:[:]))}
        XCTAssertThrowsError(try WorkflowCodec.encode(document))
        document.nodes=(0..<257).map{.init(address:address(.control,"n\($0)"),payload:.train(config:[:]))}
        XCTAssertThrowsError(try WorkflowCodec.encode(document))
        document.nodes=[.init(address:address(.model,"x"),x:.infinity,payload:.modelPrimitive(type:"relu",configurationVersion:2,config:[:]))]
        XCTAssertThrowsError(try WorkflowCodec.encode(document))
        document.nodes=[]
        let a=WorkflowEndpoint(address:address(.data,"missing-a"),port:"out"),b=WorkflowEndpoint(address:address(.control,"missing-b"),port:"in")
        document.edges=[.init(id:"a",from:a,to:b,kind:.table),.init(id:"b",from:a,to:b,kind:.dataset)]
        XCTAssertThrowsError(try WorkflowCodec.encode(document))
        XCTAssertThrowsError(try WorkflowCodec.decode(Data(repeating:32,count:WorkflowCodec.maximumBytes+1)))
    }
}
