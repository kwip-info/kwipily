import XCTest
import KwipMLCore
@testable import KwipMLApp
@MainActor final class ReshapeLessonEditingTests:XCTestCase {
    func testSharedPlacementCanReachFlattenWithExactLessonInputAndUndo()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let session=TensorLearningDraftSession(repository:.init(directory:root))
        XCTAssertTrue(session.create(projectID:UUID().uuidString,operation:"flatten",mode:.guided))
        let canvas=TensorLearningCanvasAdapter(session:session)
        for type in ["input","reshape","output"] {canvas.addNode(type:type,x:0,y:0)}
        let nodes=canvas.graph.nodes;let shape=nodes[1]
        canvas.connect(source:nodes[0].id,sourcePort:"out",target:shape.id,targetPort:"in")
        canvas.connect(source:shape.id,sourcePort:"out",target:nodes[2].id,targetPort:"in")
        canvas.selectNode(shape.id,extending:false)
        XCTAssertTrue(canvas.setReshape(.flatten,nodeID:shape.id,expectedType:shape.type,expected:shape.config),canvas.statusMessage)
        XCTAssertEqual(canvas.graph.nodes[1].type,"flatten");XCTAssertEqual(canvas.graph.edges.count,2)
        XCTAssertEqual(session.draft?.operation,"flatten")
        canvas.undo();XCTAssertEqual(canvas.graph.nodes[1],shape)
        XCTAssertFalse(canvas.setReshape(.flatten,nodeID:shape.id,expectedType:"flatten",expected:shape.config))
    }
}
