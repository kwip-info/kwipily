import XCTest
import KwipMLCore
@testable import KwipMLApp

final class MatrixCanvasAdapterTests:XCTestCase {
    @MainActor func testBuildsTwoBoundariesAndMatrixWithSharedCanvasCommands()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let session=TensorLearningDraftSession(repository:.init(directory:root))
        XCTAssertTrue(session.create(projectID:UUID().uuidString,operation:MatrixLearningTask.broadcast.rawValue,mode:.practice))
        let canvas=TensorLearningCanvasAdapter(session:session)
        for type in ["input","input","matmul","output"] {canvas.addNode(type:type,x:0,y:0)}
        let nodes=canvas.graph.nodes
        XCTAssertEqual(nodes.count,4)
        guard nodes.count == 4 else {return}
        XCTAssertEqual(nodes[0].config["fixture"],.string("left"))
        XCTAssertEqual(nodes[1].config["fixture"],.string("right"))
        XCTAssertFalse(canvas.canAddType("input"))
        canvas.connect(source:nodes[0].id,sourcePort:"out",target:nodes[2].id,targetPort:"a")
        canvas.connect(source:nodes[1].id,sourcePort:"out",target:nodes[2].id,targetPort:"b")
        canvas.connect(source:nodes[2].id,sourcePort:"out",target:nodes[3].id,targetPort:"in")
        XCTAssertTrue(canvas.validation.isValid,"\(canvas.validation.issues)")
        XCTAssertNoThrow(try session.draft!.submissionData())
        XCTAssertEqual(canvas.validation.tensorLayouts[nodes[3].id]?.sample.dimensions,[2,2,4])
    }
}
