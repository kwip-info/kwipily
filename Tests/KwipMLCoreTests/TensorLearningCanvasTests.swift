import XCTest
@testable import KwipMLCore

final class TensorLearningCanvasTests:XCTestCase {
    func testPaletteDefaultsSerializeAsLessonSubmissionAndShapesFollowConnections() throws {
        let registry=TensorLearningCanvas.registry
        XCTAssertEqual(Set(registry.components.map(\.id)),["input","output","reshape","transpose","flatten"])
        let nodes=registry.components.filter {$0.id != "reshape"}.map {component in
            GraphNode(id:component.id,type:component.id,config:Dictionary(uniqueKeysWithValues:component.fields.map {($0.key,$0.defaultValue)}))
        }
        let graph=GraphDocument(nodes:nodes,edges:[GraphEdge(source:"input",target:"transpose"),GraphEdge(source:"transpose",target:"flatten"),GraphEdge(source:"flatten",target:"output")])
        XCTAssertNoThrow(try TensorLearningSubmission(graph:graph))
        let result=TensorLearningCanvas.inspect(graph)
        XCTAssertTrue(result.isValid,result.issues.map(\.message).joined())
        XCTAssertEqual(result.tensorLayouts["input"]?.sample.dimensions,[2,3])
        XCTAssertEqual(result.tensorLayouts["transpose"]?.sample.dimensions,[3,2])
        XCTAssertEqual(result.tensorLayouts["output"]?.sample.dimensions,[6])
    }
    func testIncompleteAndImpossibleShapesDoNotFabricateOutputs() {
        var graph=GraphDocument(nodes:[GraphNode(id:"i",type:"input"),GraphNode(id:"r",type:"reshape",config:["shape":.array([4,4])]),GraphNode(id:"o",type:"output")],edges:[GraphEdge(source:"i",target:"r"),GraphEdge(source:"r",target:"o")])
        let invalid=TensorLearningCanvas.inspect(graph)
        XCTAssertFalse(invalid.isValid)
        XCTAssertNotNil(invalid.tensorLayouts["i"])
        XCTAssertNil(invalid.tensorLayouts["r"]);XCTAssertNil(invalid.tensorLayouts["o"])
        graph.edges=[]
        let incomplete=TensorLearningCanvas.inspect(graph)
        XCTAssertFalse(incomplete.isValid)
        XCTAssertNil(incomplete.tensorLayouts["r"])
    }
    func testWrongButExecutableConstructionIsNotGradedByCanvas() {
        let graph=GraphDocument(nodes:[GraphNode(id:"i",type:"input"),GraphNode(id:"r",type:"reshape",config:["shape":.array([3,2])]),GraphNode(id:"o",type:"output")],edges:[GraphEdge(source:"i",target:"r"),GraphEdge(source:"r",target:"o")])
        XCTAssertTrue(TensorLearningCanvas.inspect(graph).isValid)
        XCTAssertEqual(TensorLearningCanvas.inspect(graph).tensorLayouts["o"]?.sample.dimensions,[3,2])
    }
}
