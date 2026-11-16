import XCTest
import KwipMLCore
@testable import KwipMLApp

final class AppCanvasDefaultsTests: XCTestCase {
    @MainActor func testDefaultsApplyOnlyToNewWorkspacesAndResetPreservesSavedProjects() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:root,detectCapabilities:false)
        workspace.beginProject(.init(name: "Untitled flow"))
        let first=try XCTUnwrap(workspace.activeProjectID)
        let original=workspace.modelCanvasViewState
        let chosen=CanvasPresentationSettings(flowDirection:.topToBottom,nodeDetail:.small)
        XCTAssertTrue(workspace.setCanvasDefaults(chosen))
        XCTAssertEqual(workspace.modelCanvasViewState,original)
        workspace.beginProject(.init(name: "Untitled flow"))
        XCTAssertEqual(workspace.modelCanvasViewState.presentation,chosen)
        XCTAssertEqual(workspace.dataCanvasViewState.presentation,chosen)
        let second=try XCTUnwrap(workspace.activeProjectID)
        let graph=workspace.graph
        XCTAssertTrue(workspace.resetCanvasDefaults())
        XCTAssertEqual(workspace.modelCanvasViewState.presentation,chosen)
        XCTAssertEqual(workspace.graph,graph)
        workspace.beginProject(.init(name: "Untitled flow"))
        XCTAssertEqual(workspace.modelCanvasViewState.presentation,CanvasPresentationSettings())
        workspace.openManagedProject(try XCTUnwrap(workspace.projects.first{$0.id==second}))
        XCTAssertEqual(workspace.modelCanvasViewState.presentation,chosen)
        workspace.openManagedProject(try XCTUnwrap(workspace.projects.first{$0.id==first}))
        XCTAssertEqual(workspace.modelCanvasViewState,original)
        XCTAssertTrue(workspace.setCanvasDefaults(chosen))
        let reopened=WorkspaceStore(rootOverride:root,detectCapabilities:false)
        reopened.beginProject(.init(name: "Untitled flow"))
        XCTAssertEqual(reopened.modelCanvasViewState.presentation,chosen)
    }
    @MainActor func testInvalidPreferencesArePreservedAndFailedSaveDoesNotApply() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let storage=root.appendingPathComponent(".local")
        try FileManager.default.createDirectory(at:storage,withIntermediateDirectories:true)
        let file=storage.appendingPathComponent("app-preferences.json")
        let original=Data("invalid preferences".utf8)
        try original.write(to:file)
        let workspace=WorkspaceStore(rootOverride:root,detectCapabilities:false)
        XCTAssertNotNil(workspace.appPreferencesError)
        XCTAssertEqual(try Data(contentsOf:file),original)
        XCTAssertEqual(workspace.appPreferences.canvasDefaults,CanvasPresentationSettings())
        try FileManager.default.removeItem(at:file)
        try FileManager.default.createDirectory(at:file,withIntermediateDirectories:false)
        XCTAssertFalse(workspace.setCanvasDefaults(.init(flowDirection:.topToBottom,nodeDetail:.large)))
        XCTAssertEqual(workspace.appPreferences.canvasDefaults,CanvasPresentationSettings())
        XCTAssertNotNil(workspace.appPreferencesError)
    }

}
