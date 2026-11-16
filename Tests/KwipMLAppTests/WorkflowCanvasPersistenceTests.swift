import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowCanvasPersistenceTests:XCTestCase {
    private let data=WorkflowNodeAddress(domain:.data,localID:"same")
    private let model=WorkflowNodeAddress(domain:.model,localID:"same")
    private var repo:URL {URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()}
    private func setup()throws->(WorkspaceStore,WorkflowWorkspaceSession) {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        addTeardownBlock {try? FileManager.default.removeItem(at:directory)}
        let workspace=WorkspaceStore(rootOverride:repo,storageOverride:directory,detectCapabilities:false)
        let document=WorkflowDocument(projectID:UUID().uuidString,nodes:[
            .init(address:data,payload:.dataOperator(operatorID:"kwip.data.sourceCSV",config:["separator":.string(",")])),
            .init(address:model,payload:.modelPrimitive(type:"linear",configurationVersion:2,config:["in_features":2,"out_features":2,"bias":true]))])
        _=try workspace.projectLibrary.saveSource(.workflow(document),id:document.projectID)
        workspace.activeProjectID=document.projectID
        let host=try makeHost(workspace,projectID:document.projectID)
        workspace.workflowSession=host
        return(workspace,host)
    }
    private func makeHost(_ workspace:WorkspaceStore,projectID:String)throws->WorkflowWorkspaceSession {
        let owner=try WorkflowAuthoringStore(library:workspace.projectLibrary,projectID:projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
        return try WorkflowWorkspaceSession(workspace:workspace,owner:owner)
    }
    func testCameraPacketsDoNotInvalidateWholeWorkspaceButFlushLatestIntentImmediately() throws {
        let(workspace,host)=try setup()
        let persistence=workspace.installWorkflowCanvasPersistence(host)
        var canvasChanges=0,workspaceChanges=0
        let a=host.canvas.objectWillChange.sink {_ in canvasChanges+=1}
        let b=workspace.objectWillChange.sink {_ in workspaceChanges+=1}
        defer {a.cancel();b.cancel();persistence.cancel()}
        let document=host.owner.document
        for index in 1...200 {
            var next=host.canvas.canvasViewState
            next.zoom=0.2+Double(index)/1000;next.scrollX=Double(index*3);next.scrollY=Double(index*2)
            host.canvas.canvasViewState=next
        }
        XCTAssertEqual(canvasChanges,0,"Camera owns its local repaint; it must not invalidate every inspector/result/lesson")
        XCTAssertEqual(workspaceChanges,0,"Sidecar scheduling must not republish camera packets into the whole application")
        XCTAssertEqual(workspace.captureWorkspaceView().modelCanvas.zoom,0.4)
        XCTAssertEqual(workspace.captureWorkspaceView().modelCanvas.scrollX,600)
        XCTAssertTrue(workspace.flushWorkflowCanvasBeforeTransition())
        let disk=try XCTUnwrap(workspace.arrangementLibrary.projectState(projectID:host.owner.projectID))
        XCTAssertEqual(disk.modelCanvas,workspace.captureWorkspaceView().modelCanvas,"No debounce may lose the final camera before immediate close")
        XCTAssertEqual(host.owner.document,document)
        host.canvas.canvasViewState.overviewVisible.toggle()
        XCTAssertGreaterThan(canvasChanges,0,"Actual presentation controls still repaint")
    }

    func testSignedWorldPlacementUndoRedoAndReopenPreserveOtherNodes()throws {
        let(workspace,host)=try setup()
        let original=host.owner.document.nodes
        let entry=try XCTUnwrap(host.canvas.creationEntries.first {$0.iconType == "kwip.data.sourceCSV"})
        host.canvas.addNode(type:entry.id,x:-400,y:-280)
        let inserted=try XCTUnwrap(host.owner.document.nodes.first { !original.map(\.address).contains($0.address) })
        XCTAssertEqual(inserted.x,-400);XCTAssertEqual(inserted.y,-280)
        XCTAssertEqual(host.owner.document.nodes.filter {$0.address != inserted.address},original)
        host.canvas.undo()
        XCTAssertEqual(host.owner.document.nodes,original)
        host.canvas.redo()
        let reopened=try makeHost(workspace,projectID:host.owner.projectID)
        XCTAssertEqual(reopened.owner.document.nodes,host.owner.document.nodes)
        XCTAssertEqual(reopened.owner.document.nodes.first {$0.address == inserted.address}?.x,-400)
    }
    func testCaptureAndImmediateFlushReopenPreserveQualifiedSelectionAndPresentation()throws {
        let(workspace,host)=try setup()
        let observation=workspace.installWorkflowCanvasPersistence(host)
        defer {observation.cancel()}
        let original=host.owner.document
        _=host.owner.select(.init(nodeAddresses:[data,model]),lease:host.owner.lease())
        host.canvas.canvasViewState = .init(zoom:0.37,scrollX:170,scrollY:93,presentation:.init(flowDirection:.topToBottom,nodeDetail:.large))
        workspace.activeSection="History"
        let captured=workspace.captureWorkspaceView()
        XCTAssertEqual(captured.modelCanvas.selectedNodeIDs,Set([try XCTUnwrap(host.canvas.projection.displayIDByAddress[data]),try XCTUnwrap(host.canvas.projection.displayIDByAddress[model])]))
        XCTAssertEqual(captured.modelCanvas.zoom,0.37)
        XCTAssertTrue(workspace.flushWorkflowCanvasBeforeTransition())
        let disk=try XCTUnwrap(workspace.arrangementLibrary.projectState(projectID:host.owner.projectID))
        XCTAssertEqual(disk,captured)
        host.retire()
        let reopened=try makeHost(workspace,projectID:host.owner.projectID)
        workspace.workflowSession=reopened
        workspace.installWorkspaceView(disk)
        let restoredObservation=workspace.installWorkflowCanvasPersistence(reopened)
        defer {restoredObservation.cancel()}
        XCTAssertEqual(reopened.canvas.canvasViewState,captured.modelCanvas)
        XCTAssertEqual(reopened.owner.selection.nodeAddresses,[data,model])
        XCTAssertEqual(reopened.owner.document,original,"Presentation does not edit source or revision.")
    }
    func testApplyArrangementRestoresOpenCanvasAndRejectsOldRevisionCallback()throws {
        let(workspace,host)=try setup()
        let observation=workspace.installWorkflowCanvasPersistence(host);defer {observation.cancel()}
        let previousRevision=host.canvas.canvasViewRevision
        var state=workspace.captureWorkspaceView()
        state.modelCanvas = .init(zoom:1.1,scrollX:40,scrollY:50,selectedNodeIDs:[try XCTUnwrap(host.canvas.projection.displayIDByAddress[model]),"invalid-render-id"],presentation:.init(flowDirection:.topToBottom,nodeDetail:.small))
        workspace.installWorkspaceView(state)
        XCTAssertNotEqual(host.canvas.canvasViewRevision,previousRevision)
        XCTAssertEqual(host.owner.selection.nodeAddresses,[model])
        XCTAssertEqual(host.canvas.canvasViewState.selectedNodeIDs,[try XCTUnwrap(host.canvas.projection.displayIDByAddress[model])])
        let installed=workspace.captureWorkspaceView()
        workspace.recordWorkflowCanvasState(.init(zoom:0.2),host:host,revision:previousRevision)
        XCTAssertEqual(workspace.captureWorkspaceView(),installed)
    }
    func testFlushFailureVetoRetainsSessionAndLateOldSessionCannotWriteNewWorkspace()throws {
        let(workspace,host)=try setup()
        let observation=workspace.installWorkflowCanvasPersistence(host);defer {observation.cancel()}
        XCTAssertTrue(workspace.flushWorkflowCanvasBeforeTransition())
        let file=workspace.workspaceDirectory.appendingPathComponent("workspace-arrangements/projects/\(host.owner.projectID).json")
        let prior=try Data(contentsOf:file)
        try FileManager.default.removeItem(at:file)
        try FileManager.default.createDirectory(at:file,withIntermediateDirectories:false)
        host.canvas.canvasViewState.zoom=0.4
        XCTAssertFalse(workspace.flushWorkflowCanvasBeforeTransition())
        XCTAssertTrue(workspace.workflowSession === host);XCTAssertFalse(host.isRetired)
        XCTAssertNotNil(workspace.arrangementError)
        XCTAssertFalse(workspace.beginProject(ExampleFlows.parallelClassifier),"A real project transition must honor the failed layout flush.")
        XCTAssertTrue(workspace.workflowSession === host)
        XCTAssertEqual(workspace.activeProjectID,host.owner.projectID)
        XCTAssertFalse(host.isRetired)
        try FileManager.default.removeItem(at:file);try prior.write(to:file,options:.atomic)
        let replacement=try makeHost(workspace,projectID:host.owner.projectID)
        workspace.workflowSession=replacement
        let original=workspace.captureWorkspaceView()
        workspace.recordWorkflowCanvasState(.init(zoom:0.12),host:host,revision:host.canvas.canvasViewRevision)
        XCTAssertEqual(workspace.captureWorkspaceView(),original)
        XCTAssertEqual(try Data(contentsOf:file),prior)
    }
    func testWorkflowSelectionBoundsAndLongReversibleIDsRetainSidecarByteLimit()throws {
        var state=WorkspaceViewState(activeSection:"History")
        state.modelCanvas.selectedNodeIDs=Set((0..<256).map {"node-\($0)"})
        state.modelCanvas.selectedEdgeIDs=Set((0..<512).map {"edge-\($0)"})
        XCTAssertNoThrow(try state.validate())
        state.modelCanvas.selectedNodeIDs.insert("overflow")
        XCTAssertThrowsError(try state.validate())
        state.modelCanvas.selectedNodeIDs.remove("overflow")
        state.modelCanvas.selectedEdgeIDs.insert("overflow")
        XCTAssertThrowsError(try state.validate())
        let raw=String(repeating:"\0",count:1024)
        let display=try WorkflowCanvasIDs.displayEdgeID(raw)
        XCTAssertEqual(try WorkflowCanvasIDs.edgeID(from:display),raw)
        state.modelCanvas.selectedEdgeIDs=[display]
        XCTAssertNoThrow(try state.validate())
        let(workspace,host)=try setup()
        try workspace.arrangementLibrary.saveProjectState(state,projectID:host.owner.projectID)
        XCTAssertEqual(try workspace.arrangementLibrary.projectState(projectID:host.owner.projectID),state)
        state.modelCanvas.selectedEdgeIDs=[String(repeating:"x",count:16385)]
        XCTAssertThrowsError(try state.validate())
        state.modelCanvas.selectedEdgeIDs=Set(try (0..<512).map {index in
            try WorkflowCanvasIDs.displayEdgeID(String(repeating:"\0",count:1000)+String(index))
        })
        XCTAssertNoThrow(try state.validate())
        XCTAssertThrowsError(try workspace.arrangementLibrary.saveProjectState(state,projectID:host.owner.projectID),"Bounded individual identities do not bypass the overall one-megabyte sidecar limit.")
    }

}
