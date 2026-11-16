import XCTest
@testable import KwipMLCore

final class EmbeddingBagPublicationTests:XCTestCase {
    func testPublicExampleUsesWholeBagMappingAndNeverScalesCategoryIDs()throws {
        let document=try WorkflowExamples.preparedEmbeddingBagClassifier(device:"cpu")
        XCTAssertTrue(document.name.localizedCaseInsensitiveContains("bag"))
        let primitives=document.nodes.compactMap {node -> String? in
            if case .modelPrimitive(let type,_,_)=node.payload {return type};return nil
        }
        XCTAssertEqual(Set(primitives),["bag_input","embedding_bag","linear","output"])
        let mapping=try XCTUnwrap(document.nodes.first {$0.address.localID == "mapping"})
        guard case .trainingData(let config)=mapping.payload else {return XCTFail("Expected training mapping")}
        XCTAssertNil(config["featureColumns"])
        XCTAssertEqual(config["targetColumn"],.string("label"))
        guard case .object(let packet)=config["bagInput"] else {return XCTFail("Expected whole-bag input mapping")}
        XCTAssertEqual(packet["schemaVersion"],1)
        XCTAssertEqual(packet["indicesColumns"],.array(["id_a","id_b","id_c"].map(JSONValue.string)))
        let operators=document.nodes.compactMap {node -> String? in
            if case .dataOperator(let id,_)=node.payload {return id};return nil
        }
        XCTAssertEqual(Set(operators),["kwip.data.sourceCSV","kwip.data.columns","kwip.data.partition"])
        XCTAssertFalse(document.edges.contains {$0.kind == .fittedState})
        try WorkflowCodec.validate(document)
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let plan=try WorkflowCompiler.compile(document,train:WorkflowExamples.trainAddress,modelRegistry:.builtIn(),dataRegistry:.load(from:root.appendingPathComponent("runner/dataflow-operators.json")),contractVersion:13)
        XCTAssertEqual(plan.bagInput?.indicesColumns,["id_a","id_b","id_c"])
        XCTAssertEqual(plan.targetColumn,"label")
        XCTAssertEqual(plan.model.graph.nodes.filter {$0.type == "bag_input"}.count,1)
        XCTAssertTrue(try BuiltInCapabilityBridge.validate(graph:plan.model.graph,registry:.builtIn(),runtimeVersion:"2.10.0",context:.preparedWorkflow).isValid)
    }
    func testPublishedBagDeclarationsKeepPhysicalRolesAndIndependentLearningIdentity()throws {
        let registry=try ComponentRegistry.builtIn()
        for id in ["bag_input","embedding_bag"] {
            let definition=try XCTUnwrap(registry.component(type:id))
            let declaration=try EmbeddingBagCapabilityBridge.declaration(definition)
            XCTAssertEqual(declaration.adapterID,"kwip.runner.graph-v2."+id)
            XCTAssertEqual(Set(declaration.learning.lessonIDs),Set(EmbeddingBagLearningTask.allCases.map {"assessment."+$0.rawValue}))
            XCTAssertEqual(declaration.learning.exampleIDs,["prepared-embedding-bag-classifier"])
            let ids=try XCTUnwrap(declaration.ports.first {$0.id == "indices"})
            XCTAssertEqual(ids.schema.tensor?.dtype,.int64)
            XCTAssertNotNil(ids.schema.tensor?.leadingAxes)
            XCTAssertFalse(ids.schema.tensor?.gradients ?? true)
            var wrong=definition;wrong.outputs.append("invented")
            XCTAssertThrowsError(try EmbeddingBagCapabilityBridge.declaration(wrong))
        }
        let package=try BuiltInCapabilityBridge.package(registry:registry,runtimeVersion:"2.10.0")
        XCTAssertTrue(CapabilityContractValidator.validate(package).isValid)
    }
    func testExampleRefusesInvalidIdentityAndDevice() {
        XCTAssertThrowsError(try WorkflowExamples.preparedEmbeddingBagClassifier(projectID:"not-a-uuid"))
        XCTAssertThrowsError(try WorkflowExamples.preparedEmbeddingBagClassifier(device:"cuda"))
    }
    func testUnspecializedCoordinatesAndWrongPacketRolesStillRefuse()throws {
        let registry=try ComponentRegistry.builtIn()
        let source=try EmbeddingBagCapabilityBridge.declaration(XCTUnwrap(registry.component(type:"bag_input")))
        let table=try EmbeddingBagCapabilityBridge.declaration(XCTUnwrap(registry.component(type:"embedding_bag")))
        let generic=CapabilityContractValidator.connection(from:.init(source,instanceID:"source",portID:"indices"),to:.init(table,instanceID:"table",portID:"indices"),phase:.forward)
        XCTAssertFalse(generic.isValid)
        XCTAssertTrue(generic.issues.contains {$0.code == "shape.unresolved"})
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let document=try WorkflowExamples.preparedEmbeddingBagClassifier(device:"cpu")
        var graph=try WorkflowCompiler.compile(document,train:WorkflowExamples.trainAddress,modelRegistry:registry,dataRegistry:.load(from:root.appendingPathComponent("runner/dataflow-operators.json")),contractVersion:13).model.graph
        let index=try XCTUnwrap(graph.edges.firstIndex {$0.targetPort == "indices"})
        graph.edges[index].sourcePort="offsets"
        XCTAssertFalse(BuiltInCapabilityBridge.validate(graph:graph,registry:registry,runtimeVersion:"2.10.0",context:.preparedWorkflow).isValid)
    }
}
