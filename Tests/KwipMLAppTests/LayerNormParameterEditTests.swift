import XCTest
import KwipMLCore
@testable import KwipMLApp

final class LayerNormParameterEditTests:XCTestCase {
    @MainActor func testRetainedConfigurationBindingReadsCommittedGraphAndUndo() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let session=TensorLearningDraftSession(repository:.init(directory:root))
        XCTAssertTrue(session.create(projectID:UUID().uuidString,operation:LayerNormLearningTask.affine.rawValue,mode:.guided))
        let canvas=TensorLearningCanvasAdapter(session:session)
        canvas.addNode(type:"layer_norm",x:0,y:0)
        let id=try XCTUnwrap(canvas.graph.nodes.first?.id)
        let binding=canvas.configurationTextBinding(nodeID:id,field:"normalized_shape")
        XCTAssertEqual(binding.wrappedValue,"2")
        binding.wrappedValue="3"
        XCTAssertTrue(canvas.applyPending(nodeID:id,field:"normalized_shape"))
        XCTAssertEqual(binding.wrappedValue,"3","A retained native field must read committed state, not its original node snapshot")
        binding.wrappedValue=binding.wrappedValue
        XCTAssertNil(canvas.pendingText(nodeID:id,field:"normalized_shape"),"A focus-change echo must not revive the old configuration")
        canvas.undo()
        XCTAssertEqual(binding.wrappedValue,"3","Undo restores the pending edit that preceded apply")
        canvas.discardPending(nodeID:id,field:"normalized_shape")
        XCTAssertEqual(binding.wrappedValue,"2")
        canvas.redo()
        canvas.discardPending(nodeID:id,field:"normalized_shape")
        XCTAssertEqual(binding.wrappedValue,"3")
    }
    @MainActor func testRetainedParameterBindingTracksCommitResetAndUndo() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let session=TensorLearningDraftSession(repository:.init(directory:root))
        XCTAssertTrue(session.create(projectID:UUID().uuidString,operation:LayerNormLearningTask.affine.rawValue,mode:.guided))
        let canvas=TensorLearningCanvasAdapter(session:session)
        canvas.addNode(type:"layer_norm",x:0,y:0)
        let id=try XCTUnwrap(canvas.graph.nodes.first?.id)
        XCTAssertTrue(canvas.initializeNormalizationParameters(nodeID:id))
        let binding=canvas.parameterTextBinding(nodeID:id,name:"weight",index:0)
        XCTAssertEqual(binding.wrappedValue,"1.0")
        binding.wrappedValue="0.25"
        XCTAssertTrue(canvas.applyPendingParameter(nodeID:id,name:"weight",index:0))
        XCTAssertEqual(binding.wrappedValue,"0.25")
        binding.wrappedValue=binding.wrappedValue
        XCTAssertNil(canvas.pendingParameter(nodeID:id,name:"weight",index:0))
        XCTAssertTrue(canvas.initializeNormalizationParameters(nodeID:id))
        XCTAssertEqual(binding.wrappedValue,"1.0")
        canvas.undo()
        XCTAssertEqual(binding.wrappedValue,"0.25")
        canvas.redo()
        XCTAssertEqual(binding.wrappedValue,"1.0")
        let absent=canvas.parameterTextBinding(nodeID:id,name:"weight",index:99)
        XCTAssertEqual(absent.wrappedValue,"")
        absent.wrappedValue="5"
        XCTAssertNil(canvas.pendingParameter(nodeID:id,name:"weight",index:99))
        canvas.addNode(type:"output",x:200,y:0)
        binding.wrappedValue="7"
        XCTAssertNil(canvas.pendingParameter(nodeID:id,name:"weight",index:0),"A retained field from an unselected inspector must not edit the prior component")
    }
    @MainActor func testPendingCellPersistsRejectsInvalidAndCommitsWithUndo() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let session=TensorLearningDraftSession(repository:.init(directory:root))
        let project=UUID().uuidString
        XCTAssertTrue(session.create(projectID:project,operation:LayerNormLearningTask.affine.rawValue,mode:.guided))
        let canvas=TensorLearningCanvasAdapter(session:session);canvas.addNode(type:"layer_norm",x:0,y:0)
        let id=try XCTUnwrap(canvas.graph.nodes.first?.id)
        XCTAssertTrue(canvas.initializeNormalizationParameters(nodeID:id))
        XCTAssertEqual(session.draft?.parameters?[id]?["weight"]?.values,[1,1])
        XCTAssertTrue(canvas.setPendingParameter(nodeID:id,name:"weight",index:0,text:"1e-"))
        XCTAssertFalse(canvas.applyPendingParameter(nodeID:id,name:"weight",index:0))
        XCTAssertTrue(session.save())
        let reopened=TensorLearningDraftSession(repository:.init(directory:root));XCTAssertTrue(reopened.open(projectID:project))
        let editor=TensorLearningCanvasAdapter(session:reopened)
        XCTAssertEqual(editor.pendingParameter(nodeID:id,name:"weight",index:0),"1e-")
        XCTAssertTrue(editor.setPendingParameter(nodeID:id,name:"weight",index:0,text:"2.5e-1"))
        XCTAssertTrue(editor.applyPendingParameter(nodeID:id,name:"weight",index:0))
        XCTAssertEqual(reopened.draft?.parameters?[id]?["weight"]?.values,[0.25,1])
        XCTAssertNil(editor.pendingParameter(nodeID:id,name:"weight",index:0))
        editor.undo()
        XCTAssertEqual(reopened.draft?.parameters?[id]?["weight"]?.values,[1,1])
        XCTAssertEqual(editor.pendingParameter(nodeID:id,name:"weight",index:0),"2.5e-1")
        editor.redo();XCTAssertEqual(reopened.draft?.parameters?[id]?["weight"]?.values,[0.25,1])
        XCTAssertFalse(editor.setPendingParameter(nodeID:id,name:"weight",index:99,text:"1"))
        XCTAssertTrue(editor.setPendingParameter(nodeID:id,name:"bias",index:0,text:"nan"))
        XCTAssertFalse(editor.applyPendingParameter(nodeID:id,name:"bias",index:0))
        editor.discardPendingParameter(nodeID:id,name:"bias",index:0)
        XCTAssertNil(editor.pendingParameter(nodeID:id,name:"bias",index:0))
        XCTAssertTrue(editor.setPendingParameter(nodeID:id,name:"weight",index:0,text:"unfinished"))
        XCTAssertTrue(editor.initializeNormalizationParameters(nodeID:id))
        XCTAssertNil(editor.pendingParameter(nodeID:id,name:"weight",index:0))
        editor.undo()
        XCTAssertEqual(editor.pendingParameter(nodeID:id,name:"weight",index:0),"unfinished")
        editor.discardPendingParameter(nodeID:id,name:"weight",index:0)
        XCTAssertTrue(editor.setPendingParameter(nodeID:id,name:"bias",index:0,text:"unfinished"))
        var config=editor.graph.nodes[0].config;config["bias"]=false
        XCTAssertTrue(editor.configure(nodeID:id,values:config))
        XCTAssertNil(editor.pendingParameter(nodeID:id,name:"bias",index:0))
        editor.undo();XCTAssertEqual(editor.pendingParameter(nodeID:id,name:"bias",index:0),"unfinished")
    }
}
