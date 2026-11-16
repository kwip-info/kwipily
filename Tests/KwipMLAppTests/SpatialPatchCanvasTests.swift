import XCTest
import KwipMLCore
@testable import KwipMLApp
func spatialPatchLessonGraph(_ task:SpatialPatchLearningTask)->GraphDocument {
    .init(name:"Patch practice",nodes:[.init(id:"x",type:"input"),.init(id:"op",type:task.componentID,config:task.configuration),.init(id:"out",type:"output")],edges:[.init(id:"e0",source:"x",sourcePort:"out",target:"op",targetPort:"in"),.init(id:"e1",source:"op",sourcePort:"out",target:"out",targetPort:"in")])
}
final class SpatialPatchCanvasTests:XCTestCase {
    @MainActor func testFourTasksUseSuppliedSourceTypedPairedDefaultsAndInferredShapes()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:root)}
        for task in SpatialPatchLearningTask.allCases {
            let session=TensorLearningDraftSession(repository:.init(directory:root))
            XCTAssertTrue(session.create(projectID:UUID().uuidString,operation:task.rawValue,mode:.guided))
            let canvas=TensorLearningCanvasAdapter(session:session)
            canvas.addNode(type:"input",x:0,y:0)
            XCTAssertEqual(canvas.graph.nodes.first?.config,[:])
            XCTAssertEqual(canvas.graph.nodes.first?.name,"Supplied patch data")
            XCTAssertFalse(canvas.canAddType("input"))
            canvas.addNode(type:task.componentID,x:100,y:0)
            XCTAssertEqual(canvas.graph.nodes.last?.config["kernel_size"],.array([2,2]))
            XCTAssertEqual(canvas.graph.nodes.last?.config["stride"],.array([1,1]))
            XCTAssertTrue(session.update {$0.graph=spatialPatchLessonGraph(task)})
            XCTAssertTrue(canvas.validation.isValid,canvas.validation.issues.map(\.message).joined(separator:"; "))
            XCTAssertEqual(canvas.validation.tensorLayouts["x"]?.sample.dimensions,task.sampleShape)
            XCTAssertEqual(canvas.validation.tensorLayouts["out"]?.sample.dimensions,task.outputSampleShape)
            XCTAssertTrue(canvas.previewWeightLabel.contains("measured"))
            XCTAssertTrue(canvas.canvasPortDescription(nodeID:"op",port:"out").contains(canvas.validation.tensorLayouts["out"]?.displayShape ?? "Missing output layout"))
        }
    }
    @MainActor func testPairedDraftEditsApplyDiscardUndoAndColdPersistenceWithoutSilentRepair()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:root)}
        let session=TensorLearningDraftSession(repository:.init(directory:root))
        XCTAssertTrue(session.create(projectID:UUID().uuidString,operation:SpatialPatchLearningTask.foldHoles.rawValue,mode:.practice))
        XCTAssertTrue(session.update {$0.graph=spatialPatchLessonGraph(.foldHoles)})
        let canvas=TensorLearningCanvasAdapter(session:session)
        XCTAssertTrue(canvas.setPending(nodeID:"op",field:"output_size",text:"6, 6"))
        XCTAssertEqual(canvas.graph.nodes.first {$0.id == "op"}?.config["output_size"],.array([5,5]))
        XCTAssertTrue(canvas.applyPending(nodeID:"op",field:"output_size"))
        XCTAssertEqual(canvas.graph.nodes.first {$0.id == "op"}?.config["output_size"],.array([6,6]))
        canvas.undo()
        XCTAssertEqual(canvas.graph.nodes.first {$0.id == "op"}?.config["output_size"],.array([5,5]))
        canvas.discardPending(nodeID:"op",field:"output_size")
        XCTAssertNil(canvas.pendingText(nodeID:"op",field:"output_size"))
        XCTAssertTrue(canvas.setPending(nodeID:"op",field:"kernel_size",text:"2"))
        XCTAssertFalse(canvas.applyPending(nodeID:"op",field:"kernel_size"))
        XCTAssertEqual(canvas.pendingText(nodeID:"op",field:"kernel_size"),"2")
        let decoded=try TensorLearningDraft.decode(session.draft!.encoded())
        XCTAssertEqual(decoded.pendingFields,session.draft!.pendingFields)
        XCTAssertThrowsError(try decoded.submissionData())
        canvas.discardPending(nodeID:"op",field:"kernel_size")
        XCTAssertTrue(canvas.validation.isValid)
    }
}
