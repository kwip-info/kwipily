import XCTest
import KwipMLCore
@testable import KwipMLApp

final class RemainingActivationCanvasTests:XCTestCase {
    @MainActor func testChannelSlopesUsePersistedNumericEditorUndoAndConfigurationOwnership()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let session=TensorLearningDraftSession(repository:.init(directory:root))
        let project=UUID().uuidString
        XCTAssertTrue(session.create(projectID:project,operation:RemainingActivationLearningTask.channelSlopes.rawValue,mode:.guided))
        let canvas=TensorLearningCanvasAdapter(session:session)
        for type in ["input","prelu","output"] {canvas.addNode(type:type,x:0,y:0)}
        let nodes=canvas.graph.nodes
        XCTAssertEqual(nodes.count,3);guard nodes.count == 3 else {return}
        let slope=nodes[1].id
        canvas.connect(source:nodes[0].id,sourcePort:"out",target:slope,targetPort:"in")
        canvas.connect(source:slope,sourcePort:"out",target:nodes[2].id,targetPort:"in")
        canvas.selectNode(slope)
        XCTAssertTrue(canvas.configure(nodeID:slope,values:["num_parameters":2,"init":0.25]))
        XCTAssertTrue(canvas.initializeNormalizationParameters(nodeID:slope))
        XCTAssertEqual(session.draft?.parameters?[slope]?["weight"]?.values,[0.25,0.25])
        XCTAssertTrue(canvas.setGradientTarget(slope))
        for (index,text) in ["0.1","0.5"].enumerated() {
            XCTAssertTrue(canvas.setPendingParameter(nodeID:slope,name:"weight",index:index,text:text))
            XCTAssertTrue(canvas.applyPendingParameter(nodeID:slope,name:"weight",index:index))
        }
        XCTAssertEqual(session.draft?.parameters?[slope]?["weight"]?.values,[0.1,0.5])
        XCTAssertTrue(canvas.validation.isValid)
        XCTAssertNoThrow(try session.draft!.submissionData())
        XCTAssertTrue(canvas.configure(nodeID:slope,values:["num_parameters":1,"init":0.25]))
        XCTAssertNil(session.draft?.parameters?[slope],"A changed slope count requires explicit matching state")
        canvas.undo()
        XCTAssertEqual(session.draft?.parameters?[slope]?["weight"]?.values,[0.1,0.5])
        XCTAssertTrue(session.save());XCTAssertTrue(session.open(projectID:project))
        let reopened=TensorLearningCanvasAdapter(session:session)
        XCTAssertEqual(session.draft?.gradientNodeID,slope)
        reopened.selectNode(slope);reopened.deleteSelection()
        XCTAssertNil(session.draft?.parameters?[slope]);XCTAssertNil(session.draft?.gradientNodeID)
        reopened.undo()
        XCTAssertEqual(session.draft?.parameters?[slope]?["weight"]?.values,[0.1,0.5])
        XCTAssertFalse(canvas.initializeNormalizationParameters(nodeID:slope),"An old session cannot mutate reopened parameters")
    }
}
