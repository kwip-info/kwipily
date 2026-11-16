import XCTest
import KwipMLCore
@testable import KwipMLApp

final class ModelChildCanvasPersistenceTests: XCTestCase {
    @MainActor private func fixture(_ root: URL) throws -> WorkspaceStore {
        let workspace = WorkspaceStore(storageOverride: root, detectCapabilities: false)
        let id = UUID().uuidString
        _ = try workspace.projectLibrary.saveDocument(.blockModel(.init(projectID: id, source: CompositeExamples.parallelClassifier)))
        XCTAssertTrue(workspace.beginOwnedProject(id: id))
        return workspace
    }
    @MainActor func testCloseRestartAndPrivateApplyRetainPathPresentationWithoutSourceMutation() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = try fixture(root), id = try XCTUnwrap(workspace.activeProjectID)
        let session = try XCTUnwrap(workspace.modelChildWindows?.session)
        XCTAssertTrue(session.open(path: ["features"]))
        let editor = try XCTUnwrap(session.editor), owner = try XCTUnwrap(workspace.modelAuthoring)
        let before = owner.document
        let state = CanvasViewState(zoom: 0.37, scrollX: 141, scrollY: 98, selectedNodeIDs: ["branch-a"], presentation: .init(flowDirection: .topToBottom, nodeDetail: .large))
        editor.restoreCanvasViewState(state)
        XCTAssertTrue(session.close())
        XCTAssertEqual(owner.document, before)
        XCTAssertTrue(session.open(path: ["features"]))
        XCTAssertEqual(session.editor?.capturedCanvasViewState(), state)
        XCTAssertTrue(session.apply())
        XCTAssertTrue(session.open(path: ["features"]))
        XCTAssertEqual(session.editor?.capturedCanvasViewState(), state)
        XCTAssertTrue(session.close())
        let reopened = WorkspaceStore(storageOverride: root, detectCapabilities: false)
        XCTAssertTrue(reopened.beginOwnedProject(id: id))
        let fresh = try XCTUnwrap(reopened.modelChildWindows?.session)
        XCTAssertTrue(fresh.open(path: ["features"]))
        XCTAssertEqual(fresh.editor?.capturedCanvasViewState(), state)
    }
    @MainActor func testNamedArrangementRestoresOpenChildAndRejectsStaleCallbacksAcrossProjects() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = try fixture(root), session = try XCTUnwrap(workspace.modelChildWindows?.session)
        XCTAssertTrue(session.open(path: ["features"]))
        let editor = try XCTUnwrap(session.editor)
        let first = CanvasViewState(zoom: 0.32, scrollX: 12, scrollY: 24)
        editor.canvasViewState = first
        let named = try workspace.arrangementLibrary.save(name: "Child", state: workspace.captureWorkspaceView())
        let oldCallback = editor.onCanvasViewChange
        editor.canvasViewState = .init(zoom: 0.9)
        workspace.applyWorkspaceArrangement(named)
        XCTAssertEqual(editor.capturedCanvasViewState(), first)
        oldCallback?(.init(zoom: 1.4))
        XCTAssertEqual(workspace.captureWorkspaceView().blockCanvasStates, named.state.blockCanvasStates)
        let secondID = UUID().uuidString
        _ = try workspace.projectLibrary.saveDocument(.blockModel(.init(projectID: secondID, source: CompositeExamples.parallelClassifier)))
        XCTAssertTrue(workspace.beginOwnedProject(id: secondID))
        oldCallback?(.init(zoom: 1.2))
        XCTAssertTrue(workspace.blockCanvasStates.isEmpty)
        XCTAssertTrue(workspace.modelChildWindows!.session.open(path: ["features"]))
        XCTAssertEqual(workspace.modelChildWindows!.session.editor?.canvasViewState.presentation, workspace.appPreferences.canvasDefaults)
    }
    @MainActor func testFailedCloseFlushKeepsEditorAndSurfacesError() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = try fixture(root), session = try XCTUnwrap(workspace.modelChildWindows?.session)
        XCTAssertTrue(session.open(path: ["features"]))
        let editor = try XCTUnwrap(session.editor)
        editor.canvasViewState = .init(zoom: 0.42)
        let file = root.appendingPathComponent("workspace-arrangements/projects/\(workspace.activeProjectID!).json")
        try FileManager.default.removeItem(at: file)
        try FileManager.default.createDirectory(at: file, withIntermediateDirectories: false)
        XCTAssertFalse(session.close())
        XCTAssertTrue(editor === session.editor)
        XCTAssertNotNil(session.error)
        XCTAssertNotNil(workspace.arrangementError)
    }
}
