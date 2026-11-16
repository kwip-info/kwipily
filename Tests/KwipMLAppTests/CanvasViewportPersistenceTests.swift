import XCTest
import KwipMLCore
@testable import KwipMLApp

final class CanvasViewportPersistenceTests:XCTestCase {
    @MainActor func testRequestedFitSurvivesContextSwitchBeforeNativeAcknowledgement() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:root,detectCapabilities:false)
        XCTAssertTrue(workspace.saveManagedProject())
        let editor=CompositeWorkspaceStore(workspace:workspace)
        let persistence=CanvasViewportPersistence()
        let revision=editor.canvasViewRevision
        let oldToken=persistence.beginCapture()
        XCTAssertTrue(persistence.request(store:editor,revision:revision,zoom:0.46,origin:.zero))
        // Native scrolling has not acknowledged the request when the user leaves root.
        editor.openDefinition(instanceID:"features")
        XCTAssertFalse(persistence.capture(store:editor,revision:revision,token:oldToken,zoom:0.65,origin:CGPoint(x:110,y:77)))
        editor.save()
        let replacement=WorkspaceStore(rootOverride:root,detectCapabilities:false)
        XCTAssertTrue(replacement.beginProject(workspace.graph,id:workspace.activeProjectID))
        let reopened=CompositeWorkspaceStore(workspace:replacement)
        reopened.showRoot()
        XCTAssertEqual(reopened.canvasViewState.zoom,0.46)
        XCTAssertEqual(reopened.canvasViewState.scrollX,0)
        XCTAssertEqual(reopened.canvasViewState.scrollY,0)
    }
    @MainActor func testNewRequestRejectsOldSameContextCallbackAndPreservesPresentation() {
        let workspace=WorkspaceStore(rootOverride:FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString),detectCapabilities:false)
        workspace.modelCanvasViewState.presentation = CanvasPresentationSettings(flowDirection:.topToBottom,nodeDetail:.large)
        let persistence=CanvasViewportPersistence(),revision=workspace.canvasViewRevision
        let old=persistence.beginCapture()
        XCTAssertTrue(persistence.request(store:workspace,revision:revision,zoom:0.25,origin:CGPoint(x:17,y:23)))
        XCTAssertEqual(workspace.canvasViewState.zoom,0.25)
        XCTAssertEqual(workspace.canvasViewState.scrollY,23)
        XCTAssertFalse(persistence.capture(store:workspace,revision:revision,token:old,zoom:0.71,origin:CGPoint(x:80,y:90)))
        let fresh=persistence.beginCapture()
        XCTAssertTrue(persistence.capture(store:workspace,revision:revision,token:fresh,zoom:0.25,origin:CGPoint(x:15,y:20)))
        XCTAssertEqual(workspace.canvasViewState.zoom,0.25)
        XCTAssertEqual(workspace.canvasViewState.scrollX,15)
        XCTAssertEqual(workspace.canvasViewState.presentation.nodeDetail,.large)
        XCTAssertEqual(workspace.canvasViewState.presentation.flowDirection,.topToBottom)
    }
}
