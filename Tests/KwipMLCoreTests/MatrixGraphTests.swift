import XCTest
@testable import KwipMLCore

final class MatrixGraphTests:XCTestCase {
    func registry()throws->ComponentRegistry {
        var result=try ComponentRegistry.builtIn()
        result.components.removeAll {$0.id == "matmul"}
        result.components.append(MatrixMultiplyComponent.definition)
        return result
    }
    func graph()->GraphDocument {
        let nodes:[GraphNode]=[
            .init(id:"input",type:"input",config:["features":2]),
            .init(id:"left",type:"linear",config:["in_features":2,"out_features":24]),
            .init(id:"right",type:"linear",config:["in_features":2,"out_features":20]),
            .init(id:"a",type:"reshape",config:["shape":.array([2,3,4])]),
            .init(id:"b",type:"reshape",config:["shape":.array([4,5])]),
            .init(id:"product",type:"matmul"),
            .init(id:"flat",type:"flatten",config:["start_dim":1,"end_dim":-1]),
            .init(id:"head",type:"linear",config:["in_features":30,"out_features":2]),
            .init(id:"output",type:"output")]
        let links=[("input","left","in"),("input","right","in"),("left","a","in"),("right","b","in"),("a","product","a"),("b","product","b"),("product","flat","in"),("flat","head","in"),("head","output","in")]
        return GraphDocument(nodes:nodes,edges:links.map {GraphEdge(source:$0.0,target:$0.1,targetPort:$0.2)})
    }
    func testMatrixShapeFlowsThroughClassifierAndInvalidatesDescendants()throws {
        let registry=try registry()
        var flow=graph()
        let valid=GraphValidator.validate(flow,registry:registry)
        XCTAssertTrue(valid.isValid,"\(valid.issues)")
        XCTAssertEqual(valid.tensorLayouts["product"],try BatchedTensorLayout(sample:[2,3,5]))
        XCTAssertEqual(valid.tensorContracts["product"]?.dtype,.float32)
        XCTAssertNil(valid.widths["product"])
        XCTAssertEqual(valid.widths["flat"],30)
        flow.nodes[4].config["shape"] = .array([5,4])
        let invalid=GraphValidator.validate(flow,registry:registry)
        XCTAssertFalse(invalid.isValid)
        XCTAssertTrue(invalid.issues.contains {$0.nodeID == "product" && $0.message.contains("must match")})
        XCTAssertNil(invalid.tensorLayouts["product"])
        XCTAssertNil(invalid.tensorLayouts["output"])
    }
    func testConfigurationAndSemanticRoundTrip()throws {
        let registry=try registry()
        let node=GraphNode(id:"product",type:"matmul")
        XCTAssertEqual(try GraphMigration.effectiveConfiguration(of:node,schemaVersion:2,registry:registry),[:])
        XCTAssertThrowsError(try GraphMigration.effectiveConfiguration(of:node,schemaVersion:1,registry:registry))
        var invalid=node;invalid.config["transpose"] = true
        XCTAssertThrowsError(try GraphMigration.effectiveConfiguration(of:invalid,schemaVersion:2,registry:registry))
        let flow=graph()
        let restored=try JSONDecoder().decode(GraphDocument.self,from:JSONEncoder().encode(flow))
        XCTAssertEqual(try GraphSemantics.canonicalData(flow,registry:registry),try GraphSemantics.canonicalData(restored,registry:registry))
    }
}
