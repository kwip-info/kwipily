import XCTest
@testable import KwipMLCore

final class BilinearCanvasTests: XCTestCase {
    func testRepresentativeShapesKeepParametersUnbatchedInBothModes() throws {
        for task in BilinearLearningTask.allCases {
            let graph = BilinearLearningSubmissionTests.graph(bias: task == .withBias)
            let projection = BilinearLearningCanvas.inspect(graph, task: task)
            XCTAssertTrue(projection.validation.issues.isEmpty, "\(projection.validation.issues)")
            XCTAssertEqual(projection.shapeDescription("left"), "[N, 2]")
            XCTAssertEqual(projection.shapeDescription("right"), "[N, 3]")
            XCTAssertEqual(projection.shapeDescription("weight"), "[2, 2, 3]")
            XCTAssertEqual(projection.shapes["weight"]?.leadingAxis, .sharedParameter(2))
            XCTAssertEqual(projection.shapes["left"]?.leadingAxis, .observations)
            XCTAssertEqual(projection.shapeDescription("result"), "[N, 2]")
            XCTAssertTrue(projection.validation.tensorContracts.isEmpty)
            if task == .withBias { XCTAssertEqual(projection.shapeDescription("bias"), "[2]") }
            let component = try XCTUnwrap(BilinearLearningCanvas.registry(task: task).components.first { $0.id == "bilinear" })
            XCTAssertEqual(component.inputs, task == .withBias ? ["a", "b", "weight", "bias"] : ["a", "b", "weight"])
            XCTAssertEqual(component.fields.map(\.key), ["bias"])
            XCTAssertEqual(component.fields.first?.defaultValue, .bool(task == .withBias))
        }
    }

    func testAuthoredParameterTransformChangesVisibleShapeAndDiagnosesMismatch() {
        var graph = BilinearLearningSubmissionTests.graph()
        graph.nodes.append(.init(id: "transpose", type: "transpose", config: ["dim0": 1, "dim1": 2]))
        graph.edges[2].source = "transpose"
        graph.edges.append(.init(source: "weight", target: "transpose"))
        let projection = BilinearLearningCanvas.inspect(graph, task: .withBias)
        XCTAssertEqual(projection.shapeDescription("transpose"), "[2, 3, 2]")
        XCTAssertEqual(projection.shapes["transpose"]?.leadingAxis, .sharedParameter(2))
        XCTAssertTrue(projection.validation.issues.contains { $0.nodeID == "contract" })
        XCTAssertNil(projection.shapeDescription("result"))
    }

    func testObservationTransformsPreserveSymbolicBatchAndOutputAuthorship() {
        var graph = BilinearLearningSubmissionTests.graph(bias: false)
        graph.nodes.append(.init(id: "reshape", type: "reshape", config: ["shape": .array([1, 2])]))
        graph.edges[3].target = "reshape"
        graph.edges.append(.init(source: "reshape", target: "result"))
        let projection = BilinearLearningCanvas.inspect(graph, task: .withoutBias)
        XCTAssertTrue(projection.validation.issues.isEmpty)
        XCTAssertEqual(projection.shapeDescription("result"), "[N, 1, 2]")
    }

    func testBiasTransformTaskMismatchAndIncompleteGraphsAreHonestDiagnostics() {
        var graph = BilinearLearningSubmissionTests.graph()
        graph.nodes.append(.init(id: "flat", type: "flatten", config: ["start_dim": 1, "end_dim": -1]))
        graph.edges[4].source = "flat"
        graph.edges.append(.init(source: "bias", target: "flat"))
        XCTAssertTrue(BilinearLearningCanvas.inspect(graph, task: .withBias).validation.issues.contains { $0.nodeID == "flat" })
        XCTAssertFalse(BilinearLearningCanvas.inspect(BilinearLearningSubmissionTests.graph(), task: .withoutBias).validation.issues.isEmpty)
        graph = BilinearLearningSubmissionTests.graph(bias: false)
        graph.edges.removeFirst()
        let partial = BilinearLearningCanvas.inspect(graph, task: .withoutBias)
        XCTAssertFalse(partial.validation.issues.isEmpty)
        XCTAssertEqual(partial.shapeDescription("left"), "[N, 2]")
        XCTAssertNil(partial.shapeDescription("contract"))
    }
}
