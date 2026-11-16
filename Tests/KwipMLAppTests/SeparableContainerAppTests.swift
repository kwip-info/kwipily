import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class SeparableContainerAppTests:XCTestCase {
    private let root=WorkflowNodeAddress(domain:.model,localID:"convolution")
    private func fixture()throws->(WorkflowAuthoringStore,WorkflowCanvasAdapter) {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let library=ProjectLibrary(directory:directory)
        let document=try WorkflowTrainedModelSource.promoteAuthoring(WorkflowExamples.preparedSeparableConv2dClassifier(device:"cpu"))
        _=try library.saveSource(.workflow(document))
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
        let canvas=try WorkflowCanvasAdapter(owner:owner,modelRegistry:owner.modelRegistry,dataRegistry:owner.dataRegistry)
        addTeardownBlock {try? FileManager.default.removeItem(at:directory);UserDefaults.standard.removeObject(forKey:"workflow.inlineContainers."+document.projectID)}
        return(owner,canvas)
    }
    func testSameSeparableStagesExpandCollapseAndLayoutNeverRewriteSource()throws {
        let(owner,canvas)=try fixture();defer {canvas.invalidate();owner.invalidate()}
        let before=owner.document,frame=try XCTUnwrap(canvas.inlineContainers.first)
        XCTAssertTrue(frame.isExpanded);XCTAssertEqual(frame.childNodeIDs.count,2)
        var opened=0;canvas.onOpenBlock={_,_ in opened+=1}
        canvas.openBlock(frame.id);XCTAssertEqual(opened,0);XCTAssertFalse(try XCTUnwrap(canvas.inlineContainers.first).isExpanded)
        canvas.toggleInlineContainer(frame.id)
        let child=try XCTUnwrap(canvas.graph.nodes.first {$0.id==frame.childNodeIDs[0]})
        canvas.moveNodes(ids:[child.id],dx:5,dy:5);XCTAssertEqual(owner.document,before)
        XCTAssertEqual(ComponentVisualFamily.resolve(type:"separable_conv2d"),.layers)
    }
    func testInterfaceContextIsAtomicUndoableAndRefusesStaleOrUnselectedOwner()throws {
        let(owner,canvas)=try fixture();defer {canvas.invalidate();owner.invalidate()}
        let before=owner.document
        XCTAssertTrue(owner.select(.init(nodeAddresses:[root]),lease:owner.lease()))
        let context=try WorkflowModelNodeContext(owner:owner,address:root)
        let definition=try XCTUnwrap(context.definition)
        XCTAssertTrue(context.isInlineContainer)
        XCTAssertEqual(context.containerPortChoices.outputs.count,2)
        context.setNumericPending(origin:context.numericOrigin(key:"container-interface"),key:"container-interface",pending:true)
        XCTAssertTrue(owner.hasPendingInspectorEdits)
        var outputs=definition.outputs;outputs.append(.init("spatial",source:.init("depthwise","out")))
        XCTAssertTrue(context.setContainerInterface(inputs:definition.inputs,outputs:outputs,expected:definition),context.statusMessage)
        context.setNumericPending(origin:context.numericOrigin(key:"container-interface"),key:"container-interface",pending:false)
        XCTAssertEqual(owner.document.edges,before.edges)
        XCTAssertFalse(context.setContainerInterface(inputs:definition.inputs,outputs:outputs,expected:definition))
        XCTAssertTrue(owner.undo(lease:owner.lease()));XCTAssertEqual(owner.document.nodes,before.nodes)
        XCTAssertTrue(owner.select(.init(),lease:owner.lease()))
        XCTAssertFalse(context.setContainerInterface(inputs:definition.inputs,outputs:outputs,expected:definition))
    }
    func testMovePointwiseStageOutAndBackRetainsExecutableTopology()throws {
        let(owner,canvas)=try fixture();defer {canvas.invalidate();owner.invalidate()}
        let before=owner.document,address=WorkflowInlineContainer.address(root:root,childID:"pointwise")
        let id=try XCTUnwrap(canvas.projection.displayIDByAddress[address]),child=try XCTUnwrap(canvas.graph.nodes.first {$0.id==id}),parent=try XCTUnwrap(owner.document.nodes.first {$0.address==root})
        canvas.moveNodes(ids:[id],dx:parent.x-400-child.x,dy:0)
        let moved=try XCTUnwrap(owner.document.nodes.first {node in !before.nodes.contains(where:{$0.address==node.address})})
        _=try WorkflowCompiler.compile(owner.document,train:WorkflowExamples.trainAddress,modelRegistry:owner.modelRegistry,dataRegistry:owner.dataRegistry,contractVersion:15)
        let movedID=try XCTUnwrap(canvas.projection.displayIDByAddress[moved.address])
        let frame=try XCTUnwrap(canvas.inlineContainers.first)
        let geometry=CanvasGeometrySnapshot(registry:canvas.registry,presentation:canvas.canvasViewState.presentation,graph:canvas.graph,validation:canvas.validation)
        let origin=geometry.projection.project(CGPoint(x:parent.x,y:parent.y))
        let target=geometry.projection.inverse(CGPoint(x:origin.x+frame.size.width/2,y:origin.y+frame.size.height/2))
        canvas.moveNodes(ids:[movedID],dx:target.x-75-moved.x,dy:target.y-40-moved.y)
        XCTAssertFalse(owner.document.nodes.contains {$0.address==moved.address},canvas.statusMessage)
        _=try WorkflowCompiler.compile(owner.document,train:WorkflowExamples.trainAddress,modelRegistry:owner.modelRegistry,dataRegistry:owner.dataRegistry,contractVersion:15)
    }
    func testFrameGrowsNearEdgeAndDoesNotShrinkUntilExplicitFit()throws {
        let(owner,canvas)=try fixture();defer {canvas.invalidate();owner.invalidate()}
        let before=try WorkflowCodec.encode(owner.document),frame=try XCTUnwrap(canvas.inlineContainers.first)
        let child=try XCTUnwrap(canvas.graph.nodes.first {$0.id==frame.childNodeIDs[0]}),parent=try XCTUnwrap(owner.document.nodes.first {$0.address==root})
        let geometry=CanvasGeometrySnapshot(registry:canvas.registry,presentation:canvas.canvasViewState.presentation,graph:canvas.graph,validation:canvas.validation)
        let origin=geometry.projection.project(CGPoint(x:parent.x,y:parent.y))
        let target=geometry.projection.inverse(CGPoint(x:origin.x+frame.size.width+40,y:origin.y+frame.size.height/2))
        canvas.moveNodes(ids:[child.id],dx:target.x-75-child.x,dy:target.y-40-child.y)
        XCTAssertEqual(owner.document.nodes.count,try WorkflowCodec.decode(before).nodes.count,"Near-edge move stays a member")
        let grown=try XCTUnwrap(canvas.inlineContainers.first);XCTAssertGreaterThan(grown.size.width,frame.size.width)
        let moved=try XCTUnwrap(canvas.graph.nodes.first {$0.id==child.id})
        canvas.moveNodes(ids:[child.id],dx:child.x-moved.x,dy:child.y-moved.y)
        XCTAssertEqual(try XCTUnwrap(canvas.inlineContainers.first).size.width,grown.size.width,accuracy:0.000001)
        XCTAssertEqual(try XCTUnwrap(canvas.inlineContainers.first).size.height,grown.size.height,accuracy:0.000001)
        canvas.fitInlineContainer(frame.id)
        XCTAssertLessThan(try XCTUnwrap(canvas.inlineContainers.first).size.width,grown.size.width)
        XCTAssertEqual(try WorkflowCodec.encode(owner.document),before,"Resizing and child layout are presentation only")
    }
    func testPreviousPresentationWithoutSizeStateRemainsReadable()throws {
        let old=Data(#"{"collapsed":["group"],"positions":{"member":{"x":3,"y":4}}}"#.utf8)
        let state=try JSONDecoder().decode(WorkflowInlinePresentation.self,from:old)
        XCTAssertEqual(state.collapsed,["group"]);XCTAssertEqual(state.positions["member"]?.x,3);XCTAssertNil(state.sizes)
    }

}
