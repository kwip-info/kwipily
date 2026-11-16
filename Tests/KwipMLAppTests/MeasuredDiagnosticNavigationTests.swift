import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class MeasuredDiagnosticNavigationTests:XCTestCase {
    func testHistoryNavigationSurvivesCanvasAppearanceAndSelectsInspector()throws {
        let root=try canonicalBackendTemporaryDirectory();defer {try? FileManager.default.removeItem(at:root)}
        let workspace=makeWorkspace(root)
        XCTAssertTrue(workspace.loadPreparedWorkflowExample(device:"cpu"),workspace.statusMessage)
        let host=try XCTUnwrap(workspace.workflowSession)
        workspace.activeSection="History";workspace.showInspector=false
        var saved=host.canvas.canvasViewState
        saved.zoom=0.65;saved.scrollX=120;saved.scrollY=40
        saved.selectedNodeIDs=[];saved.selectedEdgeIDs=[];host.canvas.canvasViewState=saved
        let source=host.owner.document
        let hash=WorkflowArchiveCodec.hash(try WorkflowCodec.encode(source))
        try MeasuredDiagnosticNavigation.show(WorkflowExamples.trainAddress,workflowSHA256:hash,host:host,workspace:workspace)
        XCTAssertEqual(workspace.activeSection,"Build");XCTAssertTrue(workspace.showInspector)
        let display=try XCTUnwrap(host.canvas.projection.displayIDByAddress[WorkflowExamples.trainAddress])
        XCTAssertEqual(host.canvas.canvasViewState.selectedNodeIDs,[display],"Hidden canvas restore must receive the requested selection")
        XCTAssertEqual(host.canvas.canvasViewState.zoom,saved.zoom)
        XCTAssertEqual(host.canvas.canvasViewState.scrollX,saved.scrollX)
        XCTAssertEqual(host.canvas.canvasViewState.scrollY,saved.scrollY)
        // Mirror GraphCanvasView.restoreViewport: remount uses the saved view
        // state and clears canonical selection before restoring valid display IDs.
        let restored=CanvasViewportProjection.restored(host.canvas.canvasViewState,nodes:Set(host.canvas.graph.nodes.map(\.id)),edges:Set(host.canvas.graph.edges.map(\.id)))
        host.canvas.clearSelection()
        for id in restored.selectedNodeIDs.sorted() {host.canvas.selectNode(id,extending:true)}
        if restored.selectedNodeIDs.isEmpty,let edge=restored.selectedEdgeIDs.sorted().first {host.canvas.selectEdge(edge)}
        XCTAssertEqual(host.owner.selection.nodeAddresses,[WorkflowExamples.trainAddress])
        let inspector=WorkflowInspectorSession(owner:host.owner,sources:host.sourceProvider,outputs:host)
        let appearance=inspector.activate();defer {inspector.deactivate(appearance:appearance)}
        XCTAssertEqual(inspector.selection?.address,WorkflowExamples.trainAddress)
        XCTAssertTrue(inspector.selection?.isCurrent == true)
        XCTAssertEqual(host.owner.document,source,"Navigation must not edit the model")
    }
    func testStaleCaptureCannotChangeSectionSelectionOrSavedViewport()throws {
        let root=try canonicalBackendTemporaryDirectory();defer {try? FileManager.default.removeItem(at:root)}
        let workspace=makeWorkspace(root)
        XCTAssertTrue(workspace.loadPreparedWorkflowExample(device:"cpu"))
        let host=try XCTUnwrap(workspace.workflowSession)
        workspace.activeSection="History"
        let hash=WorkflowArchiveCodec.hash(try WorkflowCodec.encode(host.owner.document))
        XCTAssertTrue(host.owner.rename("Edited since capture",lease:host.owner.lease()))
        let selection=host.owner.selection,view=host.canvas.canvasViewState
        XCTAssertThrowsError(try MeasuredDiagnosticNavigation.show(WorkflowExamples.trainAddress,workflowSHA256:hash,host:host,workspace:workspace))
        XCTAssertEqual(workspace.activeSection,"History")
        XCTAssertEqual(host.owner.selection,selection);XCTAssertEqual(host.canvas.canvasViewState,view)
    }
    private func makeWorkspace(_ root:URL)->WorkspaceStore {
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        return WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
    }
}
