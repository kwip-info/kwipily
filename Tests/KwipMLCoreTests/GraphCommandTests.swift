import XCTest
@testable import KwipMLCore

final class GraphCommandTests:XCTestCase {
    func apply(_ command:GraphCommand,_ state:GraphEditorState=GraphEditorState(document:ExampleFlows.parallelClassifier))throws->GraphCommandResult {
        var next=0
        return try GraphCommandEngine.apply(command,to:state,registry:ComponentRegistry.builtIn(),makeID:{next+=1;return "new-\(next)"})
    }
    func testSelectionReplaceAddToggleAndRemoveAreDocumentNeutral() throws {
        let start=GraphEditorState(document:ExampleFlows.parallelClassifier)
        let first=try apply(.select(GraphSelection(nodeIDs:["input"]),mode:.replace),start)
        XCTAssertEqual(first.after.selection.nodeIDs,["input"]);XCTAssertFalse(first.changesDocument)
        let second=try apply(.select(GraphSelection(nodeIDs:["branch-a"],edgeIDs:["e1"]),mode:.add),first.after)
        XCTAssertEqual(second.after.selection.nodeIDs,["input","branch-a"])
        let third=try apply(.select(GraphSelection(nodeIDs:["input","branch-b"]),mode:.toggle),second.after)
        XCTAssertEqual(third.after.selection.nodeIDs,["branch-a","branch-b"])
        let fourth=try apply(.select(GraphSelection(nodeIDs:["branch-a"],edgeIDs:["e1"]),mode:.remove),third.after)
        XCTAssertEqual(fourth.after.selection.nodeIDs,["branch-b"]);XCTAssertTrue(fourth.after.selection.edgeIDs.isEmpty)
        XCTAssertEqual(fourth.after.document,start.document)
        XCTAssertThrowsError(try apply(.select(GraphSelection(nodeIDs:["missing"]),mode:.replace),start))
    }
    func testDuplicateRemapsInternalEdgesAndLeavesBoundaryDisconnected() throws {
        let start=GraphEditorState(document:ExampleFlows.parallelClassifier)
        let result=try apply(.duplicate(nodeIDs:["branch-a","activation-a"],offset:GraphOffset(x:40,y:90),semantics:.independent),start)
        XCTAssertEqual(result.after.document.nodes.count,10);XCTAssertEqual(result.after.document.edges.count,9)
        let branchID=try XCTUnwrap(result.copies.nodes["branch-a"])
        let activationID=try XCTUnwrap(result.copies.nodes["activation-a"])
        let branch=try XCTUnwrap(result.after.document.nodes.first {$0.id == branchID})
        XCTAssertEqual(branch.config,start.document.nodes[1].config);XCTAssertEqual(branch.x,start.document.nodes[1].x+40)
        XCTAssertEqual(branch.y,start.document.nodes[1].y+90);XCTAssertEqual(branch.name,start.document.nodes[1].name)
        let cloned=try XCTUnwrap(result.after.document.edges.first {$0.id == result.copies.edges["e3"]})
        XCTAssertEqual(cloned.source,branchID);XCTAssertEqual(cloned.target,activationID)
        XCTAssertFalse(result.after.document.edges.contains {$0.source == "input" && $0.target == branchID})
        XCTAssertFalse(result.after.document.edges.contains {$0.source == activationID && $0.target == "merge"})
        XCTAssertEqual(result.after.selection.nodeIDs,[branchID,activationID])
        var independentlyEdited=result.after.document
        let index=try XCTUnwrap(independentlyEdited.nodes.firstIndex {$0.id == branchID})
        independentlyEdited.nodes[index].config["out_features"]=12
        XCTAssertEqual(independentlyEdited.nodes.first {$0.id == "branch-a"}?.config["out_features"],8)
        XCTAssertThrowsError(try apply(.duplicate(nodeIDs:["branch-a"],offset:GraphOffset(),semantics:.sharedParameters),start))
    }
    func testDeleteRemovesIncidentWiresWithoutRepairAndUndoRestoresIdentity() throws {
        let start=GraphEditorState(document:ExampleFlows.parallelClassifier,selection:GraphSelection(nodeIDs:["branch-a"],edgeIDs:["e3"]))
        let result=try apply(.delete(GraphSelection(nodeIDs:["branch-a"])),start)
        XCTAssertEqual(result.after.document.nodes.count,7)
        XCTAssertEqual(result.delta.removedEdgeIDs,["e1","e3"])
        XCTAssertFalse(result.after.document.edges.contains {$0.source == "input" && $0.target == "activation-a"})
        XCTAssertTrue(result.after.selection.nodeIDs.isEmpty);XCTAssertTrue(result.after.selection.edgeIDs.isEmpty)
        var history=EditHistory<GraphEditorState>()
        history.record(before:result.before,after:result.after)
        XCTAssertEqual(history.undo(current:result.after),start)
        XCTAssertEqual(history.redo(current:start),result.after)
    }
    func testReconnectKeepsEdgeIdentityAndRejectsOccupiedCycleOrShape() throws {
        let original=GraphEditorState(document:ExampleFlows.parallelClassifier)
        let result=try apply(.reconnect(edgeID:"e5",source:GraphPortAddress("activation-b","out"),target:GraphPortAddress("merge","a")),original)
        XCTAssertEqual(result.after.document.edges.first {$0.id == "e5"}?.source,"activation-b")
        XCTAssertEqual(result.after.document.edges.count,8);XCTAssertEqual(result.delta.updatedEdgeIDs,["e5"])
        XCTAssertThrowsError(try apply(.reconnect(edgeID:"e5",source:GraphPortAddress("activation-a","out"),target:GraphPortAddress("merge","b")),original))
        XCTAssertThrowsError(try apply(.reconnect(edgeID:"e1",source:GraphPortAddress("activation-a","out"),target:GraphPortAddress("branch-a","in")),original))
        XCTAssertThrowsError(try apply(.reconnect(edgeID:"e8",source:GraphPortAddress("activation-a","out"),target:GraphPortAddress("output","in")),original))
        XCTAssertEqual(original.document,ExampleFlows.parallelClassifier)
    }
    func testInsertOnWireIsExplicitShapeCheckedAndReversible() throws {
        let original=GraphEditorState(document:ExampleFlows.parallelClassifier)
        let node=GraphNode(id:"new-layer",type:"linear",name:"Extra projection",x:50,y:60,config:["in_features":2,"out_features":2])
        let result=try apply(.insertOnWire(edgeID:"e1",node:node,inputPort:"in",outputPort:"out"),original)
        XCTAssertEqual(result.after.document.edges.first {$0.id == "e1"}?.target,node.id)
        XCTAssertTrue(result.after.document.edges.contains {$0.source == node.id && $0.target == "branch-a"})
        XCTAssertEqual(result.delta.insertedNodeIDs,[node.id]);XCTAssertEqual(result.delta.insertedEdgeIDs.count,1)
        XCTAssertEqual(result.delta.updatedEdgeIDs,["e1"])
        XCTAssertTrue(GraphValidator.validate(result.after.document,registry:try .builtIn()).isValid)
        var wrong=node;wrong.config["out_features"]=3
        XCTAssertThrowsError(try apply(.insertOnWire(edgeID:"e1",node:wrong,inputPort:"in",outputPort:"out"),original))
        XCTAssertEqual(original.document,ExampleFlows.parallelClassifier)
    }
    func testIncompleteDraftMayConnectButKnownMismatchCannot() throws {
        let graph=GraphDocument(nodes:[GraphNode(id:"input",type:"input",config:["features":2]),GraphNode(id:"linear",type:"linear",config:["in_features":2,"out_features":8])])
        let connected=try apply(.connect(source:GraphPortAddress("input","out"),target:GraphPortAddress("linear","in")),GraphEditorState(document:graph))
        XCTAssertEqual(connected.after.document.edges.count,1)
        XCTAssertThrowsError(try apply(.connect(source:GraphPortAddress("input","out"),target:GraphPortAddress("linear","in")),connected.after))
        var wrong=graph;wrong.nodes[1].config["in_features"]=9
        XCTAssertThrowsError(try apply(.connect(source:GraphPortAddress("input","out"),target:GraphPortAddress("linear","in")),GraphEditorState(document:wrong)))
    }
    func testIdentityCollisionsAndInvalidPositionsRejectWithoutMutation() throws {
        let state=GraphEditorState(document:ExampleFlows.parallelClassifier)
        XCTAssertThrowsError(try GraphCommandEngine.apply(.duplicate(nodeIDs:["input"],offset:GraphOffset(),semantics:.independent),to:state,registry:.builtIn(),makeID:{"input"}))
        XCTAssertThrowsError(try apply(.duplicate(nodeIDs:["input"],offset:GraphOffset(x:.infinity,y:0),semantics:.independent),state))
        XCTAssertThrowsError(try apply(.insertOnWire(edgeID:"e1",node:GraphNode(id:"input",type:"relu"),inputPort:"in",outputPort:"out"),state))
        XCTAssertEqual(state.document,ExampleFlows.parallelClassifier)
    }
}


extension GraphCommandTests {
    func testGroupMovePreservesRelativeLayoutAndUndoIsSingleValueChange() throws {
        let before=GraphEditorState(document:ExampleFlows.parallelClassifier,selection:GraphSelection(nodeIDs:["branch-a","activation-a"]))
        let result=try apply(.move(nodeIDs:before.selection.nodeIDs,offset:GraphOffset(x:20,y:40)),before)
        for node in result.after.document.nodes {
            let original=try XCTUnwrap(before.document.nodes.first {$0.id == node.id})
            XCTAssertEqual(node.x,original.x+(before.selection.nodeIDs.contains(node.id) ? 20 : 0))
            XCTAssertEqual(node.y,original.y+(before.selection.nodeIDs.contains(node.id) ? 40 : 0))
        }
        XCTAssertEqual(result.after.selection,before.selection)
        XCTAssertEqual(result.after.document.edges,before.document.edges)
        var history=EditHistory<GraphEditorState>();history.record(before:before,after:result.after)
        XCTAssertEqual(history.undo(current:result.after),before)
        XCTAssertThrowsError(try apply(.move(nodeIDs:["branch-a"],offset:GraphOffset(x:.nan,y:0)),before))
    }
}
