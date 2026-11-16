import XCTest
import KwipMLCore
@testable import KwipMLApp

final class EmbeddingLearningInspectorTests:XCTestCase {
    @MainActor func testOptionalSettingsApplyUndoAndRejectRetiredContext()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let session=TensorLearningDraftSession(repository:.init(directory:root))
        XCTAssertTrue(session.create(projectID:UUID().uuidString,operation:"embedding-gather",mode:.guided))
        let canvas=TensorLearningCanvasAdapter(session:session)
        canvas.addNode(type:"embedding",x:0,y:0)
        let id=try XCTUnwrap(canvas.graph.nodes.first?.id)
        XCTAssertTrue(canvas.setOptionalEnabled(nodeID:id,field:"max_norm",enabled:true))
        XCTAssertNotEqual(canvas.graph.nodes[0].config["max_norm"],.null)
        XCTAssertTrue(canvas.setPending(nodeID:id,field:"max_norm",text:"0.5"))
        XCTAssertTrue(canvas.applyPending(nodeID:id,field:"max_norm"))
        XCTAssertEqual(canvas.graph.nodes[0].config["max_norm"]?.doubleValue,0.5)
        XCTAssertTrue(canvas.setOptionalEnabled(nodeID:id,field:"max_norm",enabled:false))
        XCTAssertEqual(canvas.graph.nodes[0].config["max_norm"],.null)
        canvas.undo();XCTAssertEqual(canvas.graph.nodes[0].config["max_norm"]?.doubleValue,0.5)
        XCTAssertFalse(canvas.setOptionalEnabled(nodeID:id,field:"embedding_dim",enabled:false))
        XCTAssertTrue(session.create(projectID:UUID().uuidString,operation:"transpose",mode:.guided))
        XCTAssertFalse(canvas.setOptionalEnabled(nodeID:id,field:"max_norm",enabled:true))
    }
}
