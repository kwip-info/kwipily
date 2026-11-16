import XCTest
@testable import KwipMLCore

final class EmbeddingLearningCanvasTests:XCTestCase {
    func testPaletteAndTypedShapePropagation()throws {
        let registry=TensorLearningCanvas.registry(operation:EmbeddingLearningTask.gather.rawValue)
        XCTAssertNotNil(registry.component(type:"embedding"))
        let graph=GraphDocument(nodes:[.init(id:"i",type:"input"),.init(id:"r",type:"reshape",config:["shape":.array([2,2])]),.init(id:"e",type:"embedding",config:try EmbeddingLearningTask.gather.configuration.values),.init(id:"o",type:"output")],edges:[.init(source:"i",target:"r"),.init(source:"r",target:"e"),.init(source:"e",target:"o")])
        let result=TensorLearningCanvas.inspect(graph,allowEmbedding:true)
        XCTAssertTrue(result.isValid,result.issues.map(\.message).joined())
        XCTAssertEqual(result.tensorContracts["i"]?.dtype,.int64)
        XCTAssertEqual(result.tensorContracts["r"]?.dtype,.int64)
        XCTAssertEqual(result.tensorContracts["o"]?.dtype,.float64)
        XCTAssertEqual(result.tensorLayouts["o"]?.sample.dimensions,[2,2,3])
        XCTAssertFalse(TensorLearningCanvas.inspect(graph).isValid)
    }
    func testIncompletePathDiagnosticsAreNotDuplicated() {
        let result=TensorLearningCanvas.inspect(GraphDocument(),allowEmbedding:true)
        XCTAssertFalse(result.isValid)
        XCTAssertEqual(result.issues.count,Set(result.issues.map(\.message)).count)
    }
    func testMissingAndRepeatedLookupAreInvalid()throws {
        let noLookup=GraphDocument(nodes:[.init(id:"i",type:"input"),.init(id:"o",type:"output")],edges:[.init(source:"i",target:"o")])
        XCTAssertFalse(TensorLearningCanvas.inspect(noLookup,allowEmbedding:true).isValid)
        let graph=GraphDocument(nodes:[.init(id:"i",type:"input"),.init(id:"a",type:"embedding",config:try EmbeddingLearningTask.gather.configuration.values),.init(id:"b",type:"embedding",config:try EmbeddingLearningTask.gather.configuration.values),.init(id:"o",type:"output")],edges:[.init(source:"i",target:"a"),.init(source:"a",target:"b"),.init(source:"b",target:"o")])
        let result=TensorLearningCanvas.inspect(graph,allowEmbedding:true)
        XCTAssertFalse(result.isValid)
        XCTAssertNil(result.tensorContracts["b"])
    }
}
