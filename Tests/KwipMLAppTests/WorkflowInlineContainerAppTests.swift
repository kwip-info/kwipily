import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowInlineContainerAppTests:XCTestCase {
    private let rootAddress=WorkflowNodeAddress(domain:.model,localID:"features")
    @MainActor private func fixture()throws->(URL,WorkflowAuthoringStore,WorkflowCanvasAdapter) {
        let directory=try canonicalBackendTemporaryDirectory()
        let library=ProjectLibrary(directory:directory.appendingPathComponent("projects"))
        let document=try WorkflowTrainedModelSource.promoteAuthoring(WorkflowExamples.preparedParallelClassifier(device:"cpu"))
        _=try library.saveSource(.workflow(document))
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
        let adapter=try WorkflowCanvasAdapter(owner:owner,modelRegistry:owner.modelRegistry,dataRegistry:owner.dataRegistry)
        addTeardownBlock {try? FileManager.default.removeItem(at:directory);UserDefaults.standard.removeObject(forKey:"workflow.inlineContainers."+document.projectID)}
        return(directory,owner,adapter)
    }
    @MainActor func testCollapseAndChildLayoutLeaveSourceAndScientificFreshnessUnchanged()throws {
        let(_,owner,canvas)=try fixture();defer {canvas.invalidate();owner.invalidate()}
        let root=try XCTUnwrap(canvas.inlineContainers.first),before=owner.document,bytes=try WorkflowCodec.encode(before)
        XCTAssertEqual(root.childNodeIDs.count,5);XCTAssertTrue(root.isExpanded)
        let child=try XCTUnwrap(canvas.graph.nodes.first {$0.id==root.childNodeIDs[0]})
        canvas.moveNodes(ids:[child.id],dx:8,dy:6)
        XCTAssertEqual(try WorkflowCodec.encode(owner.document),bytes)
        XCTAssertTrue(owner.document.matchesCapturedConstruction(before))
        let moved=try XCTUnwrap(canvas.graph.nodes.first {$0.id==child.id});XCTAssertEqual(moved.x,child.x+8,accuracy:0.01)
        canvas.toggleInlineContainer(root.id)
        XCTAssertFalse(try XCTUnwrap(canvas.inlineContainers.first).isExpanded)
        XCTAssertFalse(canvas.graph.nodes.contains {$0.id==child.id})
        XCTAssertEqual(try WorkflowCodec.encode(owner.document),bytes)
        canvas.toggleInlineContainer(root.id)
        XCTAssertEqual(try XCTUnwrap(canvas.graph.nodes.first {$0.id==child.id}).x,moved.x,accuracy:0.01)
        let reopened=try WorkflowCanvasAdapter(owner:owner,modelRegistry:owner.modelRegistry,dataRegistry:owner.dataRegistry);defer {reopened.invalidate()}
        XCTAssertEqual(try XCTUnwrap(reopened.graph.nodes.first {$0.id==child.id}).x,moved.x,accuracy:0.01)
    }
    @MainActor func testChildInspectorUsesActualConfigurationAndPreservesSelectionAcrossOwnCommit()throws {
        let(_,owner,canvas)=try fixture();defer {canvas.invalidate();owner.invalidate()}
        for frame in canvas.inlineContainers {for child in frame.childNodeIDs {
            XCTAssertTrue(canvas.canvasNodeFeedback(for:child).issueMessages.isEmpty,"Internal wires and exposed boundary inputs are connected.")
        }}
        let address=WorkflowInlineContainer.address(root:rootAddress,childID:"branch-a")
        let id=try XCTUnwrap(canvas.projection.displayIDByAddress[address]);canvas.selectNode(id,extending:false)
        let context=try WorkflowModelNodeContext(owner:owner,address:address)
        XCTAssertEqual(context.configuration["out_features"],8);XCTAssertTrue(context.inlineBoundFields.contains("out_features"))
        XCTAssertTrue(context.rename("Editable inside"),context.statusMessage)
        XCTAssertTrue(context.isCurrent);XCTAssertEqual(owner.selection.nodeAddresses,[address])
        XCTAssertEqual(owner.authoringNode(at:address)?.name,"Editable inside")
        XCTAssertTrue(context.setField("bias",value:false,expected:context.configuration),context.statusMessage)
        XCTAssertEqual(context.configuration["bias"],false)
        XCTAssertEqual(WorkflowResultsScope.resolve(selection:owner.selection,document:owner.document),.component(address))
    }
    @MainActor func testDragBoundChildOutAndBackPreservesConnectionsAndUndoableSource()throws {
        let(_,owner,canvas)=try fixture();defer {canvas.invalidate();owner.invalidate()}
        let before=owner.document,address=WorkflowInlineContainer.address(root:rootAddress,childID:"branch-a")
        let id=try XCTUnwrap(canvas.projection.displayIDByAddress[address]),child=try XCTUnwrap(canvas.graph.nodes.first {$0.id==id})
        let parent=try XCTUnwrap(owner.document.nodes.first {$0.address==rootAddress})
        canvas.moveNodes(ids:[id],dx:parent.x-400-child.x,dy:0)
        let moved=try XCTUnwrap(owner.document.nodes.first {node in !before.nodes.contains(where:{$0.address==node.address})})
        guard case .modelPrimitive(let type,_,let config)=moved.payload else {return XCTFail("Moved member missing")}
        XCTAssertEqual(type,"linear");XCTAssertEqual(config["out_features"],8)
        XCTAssertTrue(owner.document.edges.contains {$0.from.address==moved.address && $0.to.address==rootAddress})
        _=try WorkflowCompiler.compile(owner.document,train:WorkflowExamples.trainAddress,modelRegistry:owner.modelRegistry,dataRegistry:owner.dataRegistry,contractVersion:15)
        canvas.undo();var restored=owner.document;restored.revision=before.revision;XCTAssertEqual(restored,before)
        canvas.redo();XCTAssertNotEqual(owner.document,before)
        let movedID=try XCTUnwrap(canvas.projection.displayIDByAddress[moved.address])
        canvas.moveNodes(ids:[movedID],dx:parent.x+70-moved.x,dy:parent.y+100-moved.y)
        XCTAssertFalse(owner.document.nodes.contains {$0.address==moved.address})
        XCTAssertTrue(try WorkflowInlineContainer.definition(in:owner.document,root:rootAddress).body.nodes.contains {$0.id==moved.address.localID})
        _=try WorkflowCompiler.compile(owner.document,train:WorkflowExamples.trainAddress,modelRegistry:owner.modelRegistry,dataRegistry:owner.dataRegistry,contractVersion:15)
    }
    @MainActor func testInternalWireRemoveReconnectAndCollapseSelectionAreCanonical()throws {
        let(_,owner,canvas)=try fixture();defer {canvas.invalidate();owner.invalidate()}
        let edgeID=WorkflowInlineContainerProjection.internalEdgeID(root:rootAddress,edgeID:"e3")
        let display=try XCTUnwrap(canvas.projection.displayIDByEdgeID[edgeID]);canvas.selectEdge(display)
        if case .connection=WorkflowResultsScope.resolve(selection:owner.selection,document:owner.document) {} else {XCTFail("Internal wire needs connection results")}
        let root=try XCTUnwrap(canvas.projection.displayIDByAddress[rootAddress]);canvas.toggleInlineContainer(root)
        XCTAssertEqual(owner.selection.nodeAddresses,[rootAddress]);XCTAssertTrue(owner.selection.edgeIDs.isEmpty)
        canvas.toggleInlineContainer(root);canvas.selectEdge(try XCTUnwrap(canvas.projection.displayIDByEdgeID[edgeID]));canvas.deleteSelection()
        let from=try XCTUnwrap(canvas.projection.displayIDByAddress[WorkflowInlineContainer.address(root:rootAddress,childID:"branch-a")])
        let to=try XCTUnwrap(canvas.projection.displayIDByAddress[WorkflowInlineContainer.address(root:rootAddress,childID:"activation-a")])
        let target=try XCTUnwrap(canvas.graph.nodes.first {$0.id==to})
        XCTAssertTrue(canvas.canConnect(source:from,sourcePort:"out",target:target,port:"in"),canvas.statusMessage)
        canvas.connect(source:from,sourcePort:"out",target:to,targetPort:"in")
        XCTAssertTrue(try WorkflowInlineContainer.definition(in:owner.document,root:rootAddress).body.edges.contains {$0.source=="branch-a" && $0.target=="activation-a"})
    }
    @MainActor func testPalettePlacesExpandedContainerWithoutOpeningAChildWindow()throws {
        let(_,owner,canvas)=try fixture();defer {canvas.invalidate();owner.invalidate()}
        let entry=try XCTUnwrap(canvas.creationEntries.first {$0.component.title=="Parallel features"})
        let ribbon=canvas.creationEntries.map {ComponentRibbonEntry(id:$0.id,title:$0.component.title,shortTitle:$0.component.title,purpose:"",category:$0.component.category,typeIcon:$0.iconType)}
        XCTAssertTrue(ComponentRibbonCatalog.visible(ribbon,scope:.common,query:"").contains {$0.id==entry.id})
        var opened=0;canvas.onOpenBlock={_,_ in opened+=1}
        let before=owner.document.nodes.count;canvas.addNode(type:entry.id,x:1800,y:900)
        XCTAssertEqual(owner.document.nodes.count,before+1)
        let added=try XCTUnwrap(owner.document.nodes.last),id=try XCTUnwrap(canvas.projection.displayIDByAddress[added.address])
        XCTAssertTrue(canvas.inlineContainers.contains {$0.id==id && $0.isExpanded && $0.childNodeIDs.count==5})
        canvas.openBlock(id);XCTAssertEqual(opened,0);XCTAssertFalse(try XCTUnwrap(canvas.inlineContainers.first {$0.id==id}).isExpanded)
    }
    @MainActor func testRemovingEveryMemberIsAtomicAndActionable()throws {
        let(_,owner,canvas)=try fixture();defer {canvas.invalidate();owner.invalidate()}
        let frame=try XCTUnwrap(canvas.inlineContainers.first),before=owner.document
        canvas.clearSelection()
        for id in frame.childNodeIDs {canvas.selectNode(id,extending:true)}
        canvas.deleteSelection()
        XCTAssertEqual(owner.document,before)
        XCTAssertEqual(canvas.inlineContainers.first?.childNodeIDs.count,5)
        XCTAssertTrue(canvas.statusMessage.contains("at least one component"),canvas.statusMessage)
    }
}
