import XCTest
import KwipMLCore
@testable import KwipMLApp

final class CompositeCanvasPersistenceTests:XCTestCase {
    @MainActor func testContextProjectAndArrangementIdentityProtectCanvasState() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer { try? FileManager.default.removeItem(at:root) }
        let workspace=WorkspaceStore(rootOverride:root,detectCapabilities:false);XCTAssertTrue(workspace.saveManagedProject())
        let id=try XCTUnwrap(workspace.activeProjectID)
        let editor=CompositeWorkspaceStore(workspace:workspace)
        var view=CanvasViewState(zoom:1.1,scrollX:31,scrollY:17);view.presentation=CanvasPresentationSettings(flowDirection:.topToBottom,nodeDetail:.large)
        editor.canvasViewState=view
        let oldKey=editor.contextKey,oldRevision=editor.canvasViewRevision
        editor.openDefinition(instanceID:"features");XCTAssertEqual(editor.canvasViewState,CanvasViewState())
        editor.recordCanvasViewState(CanvasViewState(zoom:0.3),contextKey:oldKey,revision:oldRevision)
        XCTAssertEqual(editor.canvasViewState,CanvasViewState())
        var detail=CanvasViewState(zoom:0.9);detail.presentation.nodeDetail = .small;editor.canvasViewState=detail
        editor.showRoot();XCTAssertEqual(editor.canvasViewState,view)
        editor.save()
        let replacement=WorkspaceStore(rootOverride:root,detectCapabilities:false)
        XCTAssertTrue(replacement.beginProject(workspace.graph,id:id))
        let reopened=CompositeWorkspaceStore(workspace:replacement)
        XCTAssertEqual(reopened.canvasViewState,view)
        reopened.openDefinition(instanceID:"features");XCTAssertEqual(reopened.canvasViewState,detail)
        let old=reopened.canvasViewRevision;var arrangement=replacement.captureWorkspaceView()
        arrangement.compositeCanvasStates[reopened.contextKey]=view;replacement.installWorkspaceView(arrangement)
        reopened.recordCanvasViewState(detail,contextKey:reopened.contextKey,revision:old)
        XCTAssertEqual(reopened.canvasViewState,view)
        replacement.beginProject(.init(name: "Untitled flow"));reopened.canvasViewState=detail
        XCTAssertTrue(replacement.compositeCanvasStates.isEmpty)
    }
}

extension CompositeCanvasPersistenceTests {
    @MainActor func testPublishedRevisionsHaveIndependentPresentationWithoutChangingArchitecture() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer { try? FileManager.default.removeItem(at:root) }
        let workspace=WorkspaceStore(rootOverride:root,detectCapabilities:false);XCTAssertTrue(workspace.saveManagedProject())
        let editor=CompositeWorkspaceStore(workspace:workspace)
        let original=editor.project
        editor.openDefinition(instanceID:"features")
        var old=CanvasViewState();old.presentation.nodeDetail = .small;editor.canvasViewState=old
        XCTAssertEqual(editor.project,original)
        let oldKey=editor.contextKey,oldRevision=editor.canvasViewRevision
        editor.beginRevision();try editor.publishRevision()
        XCTAssertNotEqual(editor.contextKey,oldKey)
        XCTAssertEqual(editor.canvasViewState,CanvasViewState())
        editor.recordCanvasViewState(old,contextKey:oldKey,revision:oldRevision)
        XCTAssertEqual(editor.canvasViewState,CanvasViewState())
        XCTAssertEqual(workspace.compositeCanvasStates[oldKey],old)
        editor.showRoot();editor.openDefinition(instanceID:"features")
        XCTAssertEqual(editor.contextKey,oldKey);XCTAssertEqual(editor.canvasViewState,old)
    }
}
