import XCTest
import KwipMLCore
@testable import KwipMLApp

final class LayerNormCanvasTests:XCTestCase {
    @MainActor func testParametersDeletionDuplicationAndUndoShareOneHistory() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let session=TensorLearningDraftSession(repository:.init(directory:root))
        XCTAssertTrue(session.create(projectID:UUID().uuidString,operation:LayerNormLearningTask.affine.rawValue,mode:.practice))
        let canvas=TensorLearningCanvasAdapter(session:session)
        for type in ["input","layer_norm","output"] {canvas.addNode(type:type,x:0,y:0)}
        XCTAssertEqual(canvas.graph.nodes.count,3)
        guard canvas.graph.nodes.count == 3 else {return}
        let nodes=canvas.graph.nodes,id=nodes[1].id
        var config=nodes[1].config;config["normalized_shape"] = .array([3])
        XCTAssertTrue(canvas.configure(nodeID:id,values:config))
        canvas.connect(source:nodes[0].id,sourcePort:"out",target:id,targetPort:"in")
        canvas.connect(source:id,sourcePort:"out",target:nodes[2].id,targetPort:"in")
        XCTAssertTrue(canvas.validation.isValid)
        let parameters=["weight":LearningTensor(shape:[3],values:[0.5,-1,2]),"bias":LearningTensor(shape:[3],values:[1,-0.5,0.25])]
        XCTAssertTrue(canvas.setNormalizationParameters(nodeID:id,values:parameters))
        XCTAssertTrue(canvas.setGradientTarget(id))
        let before=try session.draft!.submissionData()
        var disabled=config;disabled["elementwise_affine"]=false
        XCTAssertTrue(canvas.configure(nodeID:id,values:disabled))
        XCTAssertNil(session.draft?.parameters?[id]);XCTAssertNil(session.draft?.gradientNodeID)
        XCTAssertNoThrow(try session.draft!.submissionData())
        canvas.undo();XCTAssertEqual(try session.draft!.submissionData(),before)
        canvas.selectNode(id,extending:false);canvas.duplicateSelection()
        let copied=try XCTUnwrap(canvas.selectedNodeIDs.first)
        XCTAssertNotEqual(copied,id)
        XCTAssertEqual(session.draft?.parameters?[copied],parameters)
        canvas.undo();XCTAssertEqual(try session.draft!.submissionData(),before)
        canvas.selectNode(id,extending:false);canvas.deleteSelection()
        XCTAssertNil(session.draft?.parameters?[id]);XCTAssertNil(session.draft?.gradientNodeID)
        canvas.undo();XCTAssertEqual(try session.draft!.submissionData(),before)
        canvas.redo();XCTAssertNil(session.draft?.parameters?[id])
        canvas.undo()
        XCTAssertTrue(canvas.setPending(nodeID:id,field:"eps",text:"1e-3"))
        XCTAssertTrue(canvas.applyPending(nodeID:id,field:"eps"))
        XCTAssertEqual(canvas.graph.nodes[1].config["eps"],.number(0.001))
        XCTAssertTrue(session.create(projectID:UUID().uuidString,operation:"reshape",mode:.guided))
        XCTAssertFalse(canvas.setNormalizationParameters(nodeID:id,values:parameters))
        XCTAssertNil(canvas.registry.component(type:"layer_norm"))
    }
}
