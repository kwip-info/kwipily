import XCTest
import KwipMLCore
@testable import KwipMLApp
@MainActor final class TensorOperationLessonMethodTests:XCTestCase {
    func testRechoosingCurrentLessonMethodNeverResetsAuthoredValues()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let session=TensorLearningDraftSession(repository:.init(directory:root))
        XCTAssertTrue(session.create(projectID:UUID().uuidString,operation:"chunk-uneven-parts",mode:.guided))
        let canvas=TensorLearningCanvasAdapter(session:session)
        canvas.addNode(type:"chunk",x:0,y:0)
        let id=try XCTUnwrap(canvas.graph.nodes.first?.id)
        XCTAssertTrue(canvas.configure(nodeID:id,values:["dim":2,"chunks":6]))
        let source=try XCTUnwrap(canvas.graph.nodes.first)
        XCTAssertTrue(canvas.setTensorOperation("chunk",nodeID:id,expectedType:"chunk",expected:source.config))
        XCTAssertEqual(canvas.graph.nodes.first,source)
    }
}
