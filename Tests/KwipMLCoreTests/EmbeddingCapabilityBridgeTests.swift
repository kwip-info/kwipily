import XCTest
@testable import KwipMLCore

final class EmbeddingCapabilityBridgeTests:XCTestCase {
    func testLookupDeclarationHasIntegerInputFloatOutputAndOwnedParameters()throws {
        let registry=try EmbeddingGraphTests().registry()
        let declarations=try BuiltInCapabilityBridge.declarations(registry:registry)
        let source=try XCTUnwrap(declarations.first {$0.id=="index_input"})
        let lookup=try XCTUnwrap(declarations.first {$0.id=="embedding"})
        XCTAssertEqual(source.ports.count,1)
        XCTAssertEqual(source.ports[0].schema.tensor?.dtype,.int64)
        XCTAssertFalse(source.ports[0].schema.tensor?.gradients ?? true)
        XCTAssertEqual(lookup.ports.map {$0.schema.tensor?.dtype},[.int64,.float32])
        XCTAssertEqual(lookup.lifecycle.state.map(\.kind),[.parameters])
        XCTAssertEqual(lookup.learning.lessonIDs, EmbeddingLearningTask.allCases.map { "assessment." + $0.rawValue })
        XCTAssertEqual(lookup.learning.exampleIDs, ["prepared-embedding-classifier"])
        XCTAssertTrue(CapabilityContractValidator.validate(lookup).isValid)
        var changed=registry
        let index=try XCTUnwrap(changed.components.firstIndex {$0.id=="embedding"})
        changed.components[index].fields[0].max=100
        XCTAssertThrowsError(try BuiltInCapabilityBridge.declarations(registry:changed))
    }
    func testIntegerLayoutSpecializationAndFloatConsumerRejection()throws {
        let registry=try EmbeddingGraphTests().registry()
        var graph=EmbeddingGraphTests().graph()
        graph.nodes.insert(GraphNode(id:"shape",type:"reshape",config:["shape":.array([1,2])]),at:1)
        graph.edges=zip(graph.nodes,graph.nodes.dropFirst()).map {GraphEdge(source:$0.id,target:$1.id)}
        let valid=BuiltInCapabilityBridge.validate(graph:graph,registry:registry,runtimeVersion:"2.14.1")
        XCTAssertTrue(valid.isValid,"\(valid.issues)")
        graph.nodes[1].type="relu";graph.nodes[1].config=[:]
        let invalid=BuiltInCapabilityBridge.validate(graph:graph,registry:registry,runtimeVersion:"2.14.1")
        XCTAssertFalse(invalid.isValid)
        XCTAssertTrue(invalid.issues.contains {$0.code=="connection.dtype"})
    }
}
