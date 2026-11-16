import XCTest
@testable import KwipMLCore

final class EmbeddingGraphTests:XCTestCase {
    func registry()throws->ComponentRegistry {
        let value=try ComponentRegistry.builtIn()
        return value
    }
    func graph()->GraphDocument {
        let nodes=[GraphNode(id:"source",type:"index_input",config:["shape":.array([2])]),
                   GraphNode(id:"lookup",type:"embedding",config:["num_embeddings":5,"embedding_dim":3,"padding_idx":.null,"max_norm":.null,"norm_type":2,"scale_grad_by_freq":false,"sparse":false,"freeze":false]),
                   GraphNode(id:"flat",type:"flatten",config:["start_dim":1,"end_dim":-1]),
                   GraphNode(id:"head",type:"linear",config:["in_features":6,"out_features":2,"bias":true]),
                   GraphNode(id:"output",type:"output")]
        return GraphDocument(nodes:nodes,edges:zip(nodes,nodes.dropFirst()).map {GraphEdge(source:$0.id,target:$1.id)})
    }
    func testIndexLookupContractsAndCanonicalConfiguration()throws {
        let registry=try registry();var flow=graph()
        flow.nodes[0].config["shape"] = .array([.number(2)])
        flow.nodes[1].config["padding_idx"] = -1
        let result=GraphValidator.validate(flow,registry:registry)
        XCTAssertTrue(result.isValid,"\(result.issues)")
        XCTAssertEqual(result.tensorContracts["source"]?.dtype,.int64)
        XCTAssertEqual(result.tensorContracts["lookup"],try TypedTensorContract(sample:[2,3],dtype:.float32))
        XCTAssertEqual(result.tensorLayouts["flat"]?.sample.dimensions,[6])
        XCTAssertEqual(try GraphMigration.effectiveConfiguration(of:flow.nodes[1],schemaVersion:2,registry:registry)["padding_idx"],4)
        XCTAssertThrowsError(try GraphMigration.effectiveConfiguration(of:flow.nodes[0],schemaVersion:1,registry:registry))
    }
    func testIntegerTransformPreservesTypeAndInvalidFloatingConsumerStopsInference()throws {
        var flow=graph();flow.nodes.remove(at:1)
        flow.nodes[2].config["in_features"]=2
        flow.edges=zip(flow.nodes,flow.nodes.dropFirst()).map {GraphEdge(source:$0.id,target:$1.id)}
        let result=GraphValidator.validate(flow,registry:try registry())
        XCTAssertEqual(result.tensorContracts["flat"]?.dtype,.int64)
        XCTAssertNil(result.tensorContracts["head"])
        XCTAssertTrue(result.issues.contains {$0.nodeID=="head" && $0.message.contains("float")})
    }
    func testFloatingLookupAndMalformedSourceRejected()throws {
        var flow=graph();flow.nodes[0].type="input";flow.nodes[0].config=["features":2]
        let result=GraphValidator.validate(flow,registry:try registry())
        XCTAssertNil(result.tensorContracts["lookup"])
        XCTAssertTrue(result.issues.contains {$0.nodeID=="lookup" && $0.message.contains("indices")})
        for shape:JSONValue in [.array([]),.array([true]),.array([.number(1.5)]),.array([0]),.array([100_000_000,2])] {
            var invalid=graph();invalid.nodes[0].config["shape"]=shape
            XCTAssertFalse(GraphValidator.validate(invalid,registry:try registry()).isValid)
        }
    }
}
