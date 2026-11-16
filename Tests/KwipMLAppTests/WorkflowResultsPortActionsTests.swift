import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowResultsPortActionsTests:XCTestCase {
    private let input=WorkflowNodeAddress(domain:.model,localID:"input")
    private let layer=WorkflowNodeAddress(domain:.model,localID:"layer")
    private let other=WorkflowNodeAddress(domain:.model,localID:"other")
    private func fixture()throws->(URL,WorkflowAuthoringStore,WorkflowCanvasAdapter) {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let models=try ComponentRegistry.builtIn(),data=try DataFlowOperatorRegistry.load(from:repository.appendingPathComponent("runner/dataflow-operators.json"))
        let doc=WorkflowDocument(projectID:UUID().uuidString,nodes:[
            .init(address:input,payload:.modelPrimitive(type:"input",configurationVersion:1,config:["features":2])),
            .init(address:layer,payload:.modelPrimitive(type:"linear",configurationVersion:2,config:["in_features":2,"out_features":2,"bias":true])),
            .init(address:other,payload:.modelPrimitive(type:"linear",configurationVersion:2,config:["in_features":2,"out_features":2,"bias":true]))])
        let library=ProjectLibrary(directory:root);_=try library.saveSource(.workflow(doc))
        let owner=try WorkflowAuthoringStore(library:library,projectID:doc.projectID,modelRegistry:models,dataRegistry:data)
        let canvas=try WorkflowCanvasAdapter(owner:owner,modelRegistry:models,dataRegistry:data)
        _=owner.select(.init(nodeAddresses:[layer]),lease:owner.lease())
        return(root,owner,canvas)
    }
    func testConnectReconnectRemoveAreCanonicalUndoableAndLeaseBound()throws {
        let(root,owner,canvas)=try fixture();defer {canvas.invalidate();try? FileManager.default.removeItem(at:root)}
        let from=WorkflowEndpoint(address:input,port:"out"),to=WorkflowEndpoint(address:layer,port:"in")
        let original=WorkflowResultsPortActions(owner:owner,canvas:canvas,scope:.component(layer))
        XCTAssertTrue(original.connect(from:from,to:to))
        let edge=try XCTUnwrap(owner.document.edges.first)
        XCTAssertFalse(original.remove(edge),"A committed edit invalidates an old menu lease")
        _=owner.select(.init(edgeIDs:[edge.id]),lease:owner.lease())
        let edit=WorkflowResultsPortActions(owner:owner,canvas:canvas,scope:.connection(edge))
        XCTAssertTrue(edit.reconnect(edge,from:from,to:.init(address:other,port:"in")))
        XCTAssertEqual(owner.document.edges.first?.to.address,other)
        canvas.undo();XCTAssertEqual(owner.document.edges.first?.to,to)
        let restored=try XCTUnwrap(owner.document.edges.first)
        _=owner.select(.init(edgeIDs:[restored.id]),lease:owner.lease())
        XCTAssertTrue(WorkflowResultsPortActions(owner:owner,canvas:canvas,scope:.connection(restored)).remove(restored))
        XCTAssertTrue(owner.document.edges.isEmpty)
        canvas.undo();XCTAssertEqual(owner.document.edges.count,1)
    }
    func testInspectionEligibilityRejectsUndeclaredAndUnrelatedPorts()throws {
        let(root,owner,canvas)=try fixture();defer {canvas.invalidate();try? FileManager.default.removeItem(at:root)}
        let action=WorkflowResultsPortActions(owner:owner,canvas:canvas,scope:.component(layer))
        XCTAssertTrue(action.canInspect(.init(address:layer,port:"in"),direction:.input))
        XCTAssertFalse(action.canInspect(.init(address:layer,port:"saved-missing"),direction:.input))
        XCTAssertFalse(action.canInspect(.init(address:other,port:"in"),direction:.input))
        XCTAssertFalse(action.inspect(.init(address:layer,port:"saved-missing"),direction:.input))
        XCTAssertNil(canvas.currentPortInspection)
    }
    func testPendingInspectorEditsBlockMutationButInspectionRemainsReadOnly()throws {
        let(root,owner,canvas)=try fixture();defer {canvas.invalidate();try? FileManager.default.removeItem(at:root)}
        let context=UUID(),endpoint=WorkflowEndpoint(address:layer,port:"in")
        let action=WorkflowResultsPortActions(owner:owner,canvas:canvas,scope:.component(layer))
        XCTAssertTrue(owner.setInspectorEditPending(contextID:context,address:layer,key:"in_features",pending:true,lease:owner.lease()))
        XCTAssertFalse(action.canEdit)
        XCTAssertFalse(action.connect(from:.init(address:input,port:"out"),to:endpoint))
        XCTAssertTrue(action.inspect(endpoint,direction:.input))
        XCTAssertEqual(canvas.currentPortInspection?.endpoint,endpoint)
        XCTAssertTrue(owner.document.edges.isEmpty)
        owner.clearInspectorEdits(contextID:context)
        owner.setExecutionBusy(true)
        XCTAssertTrue(action.inspect(endpoint,direction:.input),"Inspection does not acquire authoring authority")
        XCTAssertFalse(action.canEdit)
        owner.setExecutionBusy(false)
        XCTAssertTrue(action.canEdit)
    }
    func testStaleSelectionBusyAndUndeclaredPortsCannotMutate()throws {
        let(root,owner,canvas)=try fixture();defer {canvas.invalidate();try? FileManager.default.removeItem(at:root)}
        let from=WorkflowEndpoint(address:input,port:"out"),to=WorkflowEndpoint(address:layer,port:"in")
        let action=WorkflowResultsPortActions(owner:owner,canvas:canvas,scope:.component(layer))
        _=owner.select(.init(nodeAddresses:[other]),lease:owner.lease())
        XCTAssertFalse(action.connect(from:from,to:to))
        _=owner.select(.init(nodeAddresses:[layer]),lease:owner.lease())
        owner.setExecutionBusy(true);XCTAssertFalse(action.connect(from:from,to:to));owner.setExecutionBusy(false)
        XCTAssertFalse(action.connect(from:.init(address:input,port:"saved-missing"),to:to))
        XCTAssertFalse(action.connect(from:from,to:.init(address:other,port:"in")),"Cannot edit unrelated endpoints through a component menu")
        XCTAssertTrue(owner.document.edges.isEmpty)
        XCTAssertTrue(action.connect(from:from,to:to))
        _=owner.select(.init(nodeAddresses:[layer]),lease:owner.lease())
        let fresh=WorkflowResultsPortActions(owner:owner,canvas:canvas,scope:.component(layer))
        XCTAssertFalse(fresh.connect(from:from,to:to),"Canonical cardinality/duplicate rules still apply")
        XCTAssertEqual(owner.document.edges.count,1)
    }
}
