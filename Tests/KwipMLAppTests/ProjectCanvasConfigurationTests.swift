import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp

@MainActor final class ProjectCanvasConfigurationTests:XCTestCase {
    private var repo:URL { URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    private func workspace() -> WorkspaceStore {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        addTeardownBlock {try? FileManager.default.removeItem(at:directory)}
        return WorkspaceStore(rootOverride:repo,storageOverride:directory,detectCapabilities:false)
    }
    func testLegacyPresentationChangesPreserveGraphViewportAndRejectOldEditor() {
        let workspace=workspace(),original=workspace.graph
        let revision=workspace.canvasViewRevision
        workspace.modelCanvasViewState.zoom=0.4
        XCTAssertTrue(workspace.configureProjectCanvas(projectID:workspace.activeProjectID,revision:revision) {
            $0.presentation = .init(flowDirection:.topToBottom,nodeDetail:.small);$0.overviewVisible=false
        })
        XCTAssertEqual(workspace.graph,original)
        XCTAssertEqual(workspace.modelCanvasViewState.zoom,0.4)
        XCTAssertFalse(workspace.modelCanvasViewState.overviewVisible)
        workspace.canvasViewRevision=UUID()
        XCTAssertFalse(workspace.configureProjectCanvas(projectID:workspace.activeProjectID,revision:revision) {$0.overviewVisible=true})
        XCTAssertFalse(workspace.modelCanvasViewState.overviewVisible)
    }
    func testUniversalUpdatesLiveAdapterAndSidecarWithoutEditingSource() throws {
        let workspace=workspace()
        let document=WorkflowDocument(projectID:UUID().uuidString,nodes:[])
        _=try workspace.projectLibrary.saveSource(.workflow(document),id:document.projectID)
        workspace.activeProjectID=document.projectID
        let owner=try WorkflowAuthoringStore(library:workspace.projectLibrary,projectID:document.projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
        let host=try WorkflowWorkspaceSession(workspace:workspace,owner:owner)
        workspace.workflowSession=host
        let observation=workspace.installWorkflowCanvasPersistence(host);defer {observation.cancel()}
        let revision=host.canvas.canvasViewRevision
        XCTAssertTrue(workspace.configureProjectCanvas(projectID:document.projectID,revision:revision) {$0.presentation.nodeDetail = .large;$0.overviewVisible=false})
        XCTAssertEqual(host.owner.document,document)
        XCTAssertEqual(host.canvas.canvasViewState.presentation.nodeDetail,.large)
        XCTAssertTrue(workspace.flushWorkflowCanvasBeforeTransition())
        XCTAssertFalse(try XCTUnwrap(workspace.arrangementLibrary.projectState(projectID:document.projectID)).modelCanvas.overviewVisible)
        workspace.activeProjectID=UUID().uuidString
        XCTAssertFalse(workspace.configureProjectCanvas(projectID:document.projectID,revision:revision) {$0.overviewVisible=true})
        XCTAssertFalse(host.canvas.canvasViewState.overviewVisible)
    }
}
