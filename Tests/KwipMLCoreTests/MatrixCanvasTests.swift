import XCTest
@testable import KwipMLCore

final class MatrixCanvasTests:XCTestCase {
    func testRepresentativeShapesAreInferredByNamedOperandPort()throws {
        let graph=MatrixSubmissionTests.graph()
        for task in MatrixLearningTask.allCases {
            let registry=TensorLearningCanvas.registry(operation:task.rawValue)
            XCTAssertEqual(registry.component(type:"matmul"),MatrixMultiplyComponent.definition)
            let result=TensorLearningCanvas.inspect(graph,matrixTask:task)
            XCTAssertTrue(result.isValid,"\(result.issues)")
            XCTAssertEqual(result.tensorLayouts["o"]?.sample.dimensions,task == .contraction ? [2,4]:[2,2,4])
            XCTAssertEqual(result.tensorContracts["o"]?.dtype,.float64)
            var reversed=graph;reversed.edges[0].targetPort="b";reversed.edges[1].targetPort="a"
            let invalid=TensorLearningCanvas.inspect(reversed,matrixTask:task)
            XCTAssertFalse(invalid.isValid)
            XCTAssertNil(invalid.tensorLayouts["o"])
        }
    }
}
