import XCTest
import KwipMLCore
@testable import KwipMLApp

final class TensorLearningCanvasAdapterTests:XCTestCase {
    @MainActor func testSoftmaxConstructionUsesTaskPaletteAndRetiresItsValidation()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let session=TensorLearningDraftSession(repository:.init(directory:root))
        XCTAssertTrue(session.create(projectID:UUID().uuidString,operation:"softmax-last-axis",mode:.practice))
        let canvas=TensorLearningCanvasAdapter(session:session)
        for type in ["input","softmax","output"] {canvas.addNode(type:type,x:0,y:0)}
        let nodes=canvas.graph.nodes
        XCTAssertEqual(nodes.count,3)
        guard nodes.count == 3 else {return}
        canvas.connect(source:nodes[0].id,sourcePort:"out",target:nodes[1].id,targetPort:"in")
        canvas.connect(source:nodes[1].id,sourcePort:"out",target:nodes[2].id,targetPort:"in")
        XCTAssertTrue(canvas.validation.isValid)
        XCTAssertEqual(canvas.canvasPortDescriptor(nodeID:nodes[2].id,port:"in",direction:.input).declared,["[N, 2, 3]","float64"])
        XCTAssertNoThrow(try session.draft!.submission())
        XCTAssertTrue(session.create(projectID:UUID().uuidString,operation:"transpose",mode:.practice))
        XCTAssertNil(canvas.registry.component(type:"softmax"))
        XCTAssertFalse(canvas.canAddType("softmax"))
        XCTAssertFalse(canvas.validation.isValid)
    }

    @MainActor func testViewportPersistsAcrossReopenWithoutChangingSubmissionOrLeakingAcrossSessions() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let session=TensorLearningDraftSession(repository:.init(directory:root))
        let id=UUID().uuidString
        XCTAssertTrue(session.create(projectID:id,operation:"transpose",mode:.guided))
        let canvas=TensorLearningCanvasAdapter(session:session)
        for type in ["input","transpose","output"] {canvas.addNode(type:type,x:0,y:0)}
        let nodes=canvas.graph.nodes
        canvas.connect(source:nodes[0].id,sourcePort:"out",target:nodes[1].id,targetPort:"in")
        canvas.connect(source:nodes[1].id,sourcePort:"out",target:nodes[2].id,targetPort:"in")
        let submission=try session.draft!.submission().encoded()
        let view=CanvasViewState(zoom:1.2,scrollX:120,scrollY:230,selectedNodeIDs:[nodes[1].id],presentation:.init(flowDirection:.topToBottom,nodeDetail:.small))
        canvas.canvasViewState=view
        XCTAssertTrue(session.save())
        XCTAssertTrue(session.open(projectID:id))
        let reopened=TensorLearningCanvasAdapter(session:session)
        XCTAssertEqual(reopened.canvasViewState,view)
        XCTAssertEqual(try session.draft!.submission().encoded(),submission)
        canvas.canvasViewState=CanvasViewState(zoom:0.2)
        XCTAssertEqual(reopened.canvasViewState,view)
        var invalid=view;invalid.zoom=20
        reopened.canvasViewState=invalid
        XCTAssertEqual(reopened.canvasViewState,view)
    }
    @MainActor func testBuildConnectUndoAndStaleProjectRefusal() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let session=TensorLearningDraftSession(repository:.init(directory:root))
        XCTAssertTrue(session.create(projectID:UUID().uuidString,operation:"transpose",mode:.practice))
        let canvas=TensorLearningCanvasAdapter(session:session)
        canvas.addNode(type:"input",x:0,y:0);canvas.addNode(type:"transpose",x:200,y:0);canvas.addNode(type:"output",x:400,y:0)
        let nodes=canvas.graph.nodes
        XCTAssertEqual(nodes.count,3)
        XCTAssertEqual(nodes.map(\.name),["Input","Transpose","Model Output"])
        canvas.pendingSourceID=nodes[0].id
        canvas.connect(source:nodes[0].id,sourcePort:"out",target:nodes[1].id,targetPort:"in")
        canvas.connect(source:nodes[1].id,sourcePort:"out",target:nodes[2].id,targetPort:"in")
        XCTAssertNil(canvas.pendingSourceID)
        XCTAssertTrue(canvas.validation.isValid)
        XCTAssertEqual(canvas.canvasPortDescriptor(nodeID:nodes[2].id,port:"in",direction:.input).declared,["[N, 3, 2]","float64"])
        XCTAssertTrue(canvas.canvasPortDescriptor(nodeID:nodes[2].id,port:"in",direction:.input).measured.isEmpty)
        canvas.undo();XCTAssertEqual(canvas.graph.edges.count,1)
        canvas.redo();XCTAssertEqual(canvas.graph.edges.count,2)
        canvas.selectNode(nodes[1].id);canvas.moveNodes(ids:[nodes[1].id],dx:50,dy:10)
        XCTAssertEqual(canvas.graph.nodes[1].x,250)
        canvas.undo();XCTAssertEqual(canvas.graph.nodes[1].x,200)
        let edge=try XCTUnwrap(canvas.graph.edges.first)
        canvas.insertNode(type:"reshape",edgeID:edge.id,x:100,y:0)
        XCTAssertEqual(canvas.graph.nodes.count,4)
        XCTAssertEqual(canvas.graph.edges.count,3)
        canvas.undo();XCTAssertEqual(canvas.graph.nodes.count,3)
        XCTAssertFalse(canvas.configure(nodeID:nodes[1].id,values:["dim0":0,"dim1":1]))
        XCTAssertEqual(canvas.graph.nodes[1].config["dim0"],1)
        let old=canvas.graph
        canvas.connect(source:nodes[1].id,sourcePort:"out",target:nodes[1].id,targetPort:"in")
        XCTAssertEqual(canvas.graph,old)
        XCTAssertTrue(session.create(projectID:UUID().uuidString,operation:"flatten",mode:.guided))
        canvas.addNode(type:"input",x:0,y:0);canvas.undo()
        XCTAssertTrue(session.draft!.graph.nodes.isEmpty)
        XCTAssertFalse(canvas.historyContextIsActive)
        let fresh=TensorLearningCanvasAdapter(session:session)
        let currentID=try XCTUnwrap(session.draft?.projectID)
        XCTAssertTrue(session.open(projectID:currentID))
        fresh.addNode(type:"input",x:0,y:0)
        XCTAssertTrue(session.draft!.graph.nodes.isEmpty)
        XCTAssertFalse(fresh.historyContextIsActive)
    }
}

extension TensorLearningCanvasAdapterTests {
    @MainActor func testPendingConfigurationSurvivesInvalidApplyAndClearsOnlyOnSuccess() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let session=TensorLearningDraftSession(repository:.init(directory:root))
        XCTAssertTrue(session.create(projectID:UUID().uuidString,operation:"reshape",mode:.practice))
        let canvas=TensorLearningCanvasAdapter(session:session)
        canvas.addNode(type:"reshape",x:100,y:100)
        let id=try XCTUnwrap(canvas.graph.nodes.first?.id)
        XCTAssertTrue(canvas.setPending(nodeID:id,field:"shape",text:"3, -"))
        XCTAssertFalse(canvas.applyPending(nodeID:id,field:"shape"))
        XCTAssertEqual(canvas.pendingText(nodeID:id,field:"shape"),"3, -")
        XCTAssertEqual(canvas.graph.nodes[0].config["shape"],.array([-1]))
        XCTAssertTrue(canvas.setPending(nodeID:id,field:"shape",text:"3, 2"))
        XCTAssertTrue(canvas.applyPending(nodeID:id,field:"shape"))
        XCTAssertNil(canvas.pendingText(nodeID:id,field:"shape"))
        XCTAssertEqual(canvas.graph.nodes[0].config["shape"],.array([3,2]))
        // Native TextField can echo the normalized display after Return/focus loss.
        XCTAssertTrue(canvas.setPending(nodeID:id,field:"shape",text:"3, 2"))
        XCTAssertNil(canvas.pendingText(nodeID:id,field:"shape"))
        canvas.undo();XCTAssertEqual(canvas.graph.nodes[0].config["shape"],.array([-1]))
        XCTAssertFalse(canvas.setPending(nodeID:id,field:"unknown",text:"2"))
        XCTAssertTrue(canvas.setPending(nodeID:id,field:"shape",text:"2, -"))
        canvas.selectNode(id);canvas.deleteSelection()
        XCTAssertTrue(session.draft!.pendingFields.isEmpty)
        canvas.undo()
        XCTAssertEqual(canvas.pendingText(nodeID:id,field:"shape"),"2, -")
    }
}

extension TensorLearningCanvasAdapterTests {
    @MainActor func testValidationReadsFollowConfigurationUndoAndSessionRetirement() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let session=TensorLearningDraftSession(repository:.init(directory:root))
        XCTAssertTrue(session.create(projectID:UUID().uuidString,operation:"reshape",mode:.practice))
        let canvas=TensorLearningCanvasAdapter(session:session)
        for type in ["input","reshape","output"] {canvas.addNode(type:type,x:0,y:0)}
        let nodes=canvas.graph.nodes
        canvas.connect(source:nodes[0].id,sourcePort:"out",target:nodes[1].id,targetPort:"in")
        canvas.connect(source:nodes[1].id,sourcePort:"out",target:nodes[2].id,targetPort:"in")
        XCTAssertTrue(canvas.validation.isValid)
        XCTAssertTrue(canvas.configure(nodeID:nodes[1].id,values:["shape":.array([4,4])]))
        XCTAssertFalse(canvas.validation.isValid)
        canvas.undo();XCTAssertTrue(canvas.validation.isValid)
        XCTAssertTrue(session.create(projectID:UUID().uuidString,operation:"flatten",mode:.guided))
        XCTAssertFalse(canvas.validation.isValid)
        XCTAssertTrue(canvas.validation.tensorLayouts.isEmpty)
    }
    @MainActor func testRepeatedValidationDiagnostic() throws {
        guard ProcessInfo.processInfo.environment["KWIPML_CANVAS_BENCHMARK"] == "1" else {throw XCTSkip("Enable tensor validation timing explicitly")}
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let session=TensorLearningDraftSession(repository:.init(directory:root))
        XCTAssertTrue(session.create(projectID:UUID().uuidString,operation:"transpose",mode:.practice))
        var nodes=[GraphNode(id:"0",type:"input")]
        for i in 1...30 {nodes.append(GraphNode(id:String(i),type:"transpose",config:["dim0":1,"dim1":2]))}
        nodes.append(GraphNode(id:"31",type:"output"))
        let edges=(1...31).map {GraphEdge(source:String($0-1),target:String($0))}
        XCTAssertTrue(session.update {$0.graph=GraphDocument(nodes:nodes,edges:edges)})
        let canvas=TensorLearningCanvasAdapter(session:session)
        let start=ProcessInfo.processInfo.systemUptime
        for _ in 0..<256 {XCTAssertTrue(canvas.validation.isValid)}
        print(String(format:"TENSOR_VALIDATION_BENCHMARK nodes=32 reads=256 elapsed_ms=%.3f",(ProcessInfo.processInfo.systemUptime-start)*1000))
    }
}
