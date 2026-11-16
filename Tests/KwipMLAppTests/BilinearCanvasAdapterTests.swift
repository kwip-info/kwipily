import XCTest
import KwipMLCore
@testable import KwipMLApp

final class BilinearCanvasAdapterTests: XCTestCase {
    @MainActor func testBuildsBothModesWithHonestParameterAxesAndUndo() throws {
        for task in BilinearLearningTask.allCases {
            let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            defer {try? FileManager.default.removeItem(at:root)}
            let session=TensorLearningDraftSession(repository:.init(directory:root))
            XCTAssertTrue(session.create(projectID:UUID().uuidString,operation:task.rawValue,mode:.practice))
            let canvas=TensorLearningCanvasAdapter(session:session)
            let roles=["left","right","weight"]+(task == .withBias ? ["bias"]:[])
            for _ in roles {canvas.addNode(type:"input",x:0,y:0)}
            canvas.addNode(type:"bilinear",x:300,y:100)
            canvas.addNode(type:"output",x:600,y:100)
            let nodes=canvas.graph.nodes
            XCTAssertEqual(nodes.count,roles.count+2)
            guard nodes.count == roles.count+2 else {continue}
            XCTAssertFalse(canvas.canAddType("input"))
            XCTAssertFalse(canvas.canAddType("bilinear"))
            XCTAssertTrue(canvas.canvasPortDescription(nodeID:nodes[roles.count].id,port:"weight").contains("Shared parameter"))
            XCTAssertFalse(canvas.canvasPortDescription(nodeID:nodes[roles.count].id,port:"weight").contains("Independent observations"))
            let ports=["a","b","weight"]+(task == .withBias ? ["bias"]:[])
            for index in roles.indices {
                XCTAssertEqual(nodes[index].config["fixture"],.string(roles[index]))
                canvas.connect(source:nodes[index].id,sourcePort:"out",target:nodes[roles.count].id,targetPort:ports[index])
            }
            canvas.connect(source:nodes[roles.count].id,sourcePort:"out",target:nodes.last!.id,targetPort:"in")
            XCTAssertTrue(canvas.validation.isValid,"\(canvas.validation.issues)")
            XCTAssertNoThrow(try session.draft!.submissionData())
            XCTAssertFalse(canvas.canvasSummary(nodes[2]).contains("N"))
            XCTAssertTrue(canvas.canvasSummary(nodes[0]).contains("N"))
            XCTAssertTrue(canvas.canvasPortDescription(nodeID:nodes[2].id,port:"out").contains("Shared"))
            XCTAssertFalse(canvas.configure(nodeID:nodes[roles.count].id,values:["bias":.bool(task != .withBias)]))
            let complete=canvas.graph
            canvas.undo()
            XCTAssertThrowsError(try session.draft!.submissionData())
            canvas.redo()
            XCTAssertEqual(canvas.graph,complete)
            XCTAssertTrue(session.save())
            XCTAssertTrue(session.open(projectID:session.draft!.projectID))
            XCTAssertEqual(session.draft?.graph,complete)
        }
    }
}
