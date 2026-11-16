import XCTest
import KwipMLCore
@testable import KwipMLApp
final class PreparationSessionTests:XCTestCase {
    @MainActor func testTabAndLauncherReturnKeepSameProjectEditorState() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer{try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:root,detectCapabilities:false)
        workspace.activeProjectID=UUID().uuidString
        let session=PreparationSession();session.attach(workspace)
        let initial=session.preparation(for:workspace)
        initial.selectNode("split");initial.previews=["split":["fixture":"retained state"]]
        let before=initial.project;initial.project.name="Edited"
        initial.history.record(before:before,after:initial.project)
        session.showPreparation=true
        workspace.activeSection="Run"
        workspace.activeSection="Projects"
        workspace.activeSection="Data"
        let returned=session.preparation(for:workspace)
        XCTAssertTrue(initial === returned)
        XCTAssertEqual(returned.selectedNodeID,"split")
        XCTAssertNotNil(returned.previews["split"])
        XCTAssertTrue(returned.history.canUndo)
        XCTAssertTrue(session.showPreparation)
    }
    @MainActor func testProjectTransitionClosesAndDropsPriorContextWithoutRetainCycle() {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer{try? FileManager.default.removeItem(at:root)}
        var workspace:WorkspaceStore?=WorkspaceStore(rootOverride:root,detectCapabilities:false)
        weak var weakWorkspace=workspace
        let session=PreparationSession();session.attach(workspace!)
        let firstID=UUID().uuidString;workspace!.activeProjectID=firstID
        let first=session.preparation(for:workspace!);session.showPreparation=true
        first.previews=["old":["fixture":"private"]]
        workspace!.activeProjectID=UUID().uuidString
        XCTAssertTrue(first.inspectionClosed)
        let second=session.preparation(for:workspace!)
        XCTAssertFalse(first === second);XCTAssertTrue(second.previews.isEmpty)
        XCTAssertFalse(session.showPreparation)
        workspace!.activeProjectID=firstID
        XCTAssertTrue(session.showPreparation)
        XCTAssertFalse(first === session.preparation(for:workspace!))
        session.close();workspace=nil
        // Existing explicit references to child stores retain their parent; releasing the session itself adds no cycle.
        XCTAssertNotNil(weakWorkspace)
    }
    @MainActor func testClosingSessionReleasesWorkspaceWithoutCycle() {
        var workspace:WorkspaceStore?=WorkspaceStore(rootOverride:FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString),detectCapabilities:false)
        weak var weakWorkspace=workspace
        let session=PreparationSession();_ = session.preparation(for:workspace!)
        session.close();workspace=nil
        XCTAssertNil(weakWorkspace)
    }

}
