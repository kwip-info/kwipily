import XCTest
import KwipMLCore
@testable import KwipMLApp

final class AppAppearanceActionsTests:XCTestCase {
    private var repo:URL {URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()}

    @MainActor func testAppearancePersistsAcrossRelaunchAndResetPreservesOtherPreferencesAndWorkflow()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:repo,storageOverride:root,detectCapabilities:false)
        XCTAssertTrue(workspace.loadPreparedWorkflowExample(device:"cpu"),workspace.statusMessage)
        let host=try XCTUnwrap(workspace.workflowSession)
        XCTAssertTrue(host.owner.select(.init(nodeAddresses:[WorkflowExamples.sourceAddress]),lease:host.owner.lease()))
        let original=host.owner.document,selection=host.owner.selection,lease=host.owner.lease()
        let graph=workspace.graph,project=workspace.activeProjectID
        let canvas=CanvasPresentationSettings(flowDirection:.topToBottom,nodeDetail:.small)
        let shortcut=ShortcutChord(key:"k",shift:true)
        XCTAssertTrue(workspace.setCanvasDefaults(canvas))
        XCTAssertTrue(workspace.setShortcut(shortcut,for:.learningLibrary))
        let appearance=AppearancePreferences(mode:.system,palette:.teal,transparency:.subtle)
        XCTAssertTrue(workspace.setAppearance(appearance))
        XCTAssertEqual(workspace.appearanceState.preferences,appearance)
        XCTAssertEqual(workspace.appPreferences.appearance,appearance)
        XCTAssertNil(workspace.appPreferencesError)
        XCTAssertTrue(workspace.workflowSession === host)
        XCTAssertTrue(host.owner.isCurrent(lease))
        XCTAssertEqual(host.owner.document,original)
        XCTAssertEqual(host.owner.selection,selection)
        XCTAssertEqual(workspace.graph,graph)
        XCTAssertEqual(workspace.activeProjectID,project)
        let reopened=WorkspaceStore(rootOverride:repo,storageOverride:root,detectCapabilities:false)
        XCTAssertEqual(reopened.appPreferences.appearance,appearance)
        XCTAssertEqual(reopened.appPreferences.canvasDefaults,canvas)
        XCTAssertEqual(reopened.shortcut(for:.learningLibrary),shortcut)
        XCTAssertTrue(workspace.resetAppearance())
        XCTAssertEqual(workspace.appearanceState.preferences,.factory)
        XCTAssertEqual(workspace.appPreferences.appearance,.factory)
        XCTAssertEqual(workspace.appPreferences.canvasDefaults,canvas)
        XCTAssertEqual(workspace.shortcut(for:.learningLibrary),shortcut)
        XCTAssertTrue(workspace.workflowSession === host)
        XCTAssertTrue(host.owner.isCurrent(lease))
        XCTAssertEqual(host.owner.document,original)
        XCTAssertEqual(host.owner.selection,selection)
        XCTAssertEqual(workspace.graph,graph)
        XCTAssertEqual(try workspace.projectLibrary.loadSource(id:host.owner.projectID),.workflow(original))
    }

    @MainActor func testFailedAppearanceSaveAndResetPreserveLiveStateAndStoredEvidence()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:repo,storageOverride:root,detectCapabilities:false)
        XCTAssertTrue(workspace.loadPreparedWorkflowExample(device:"cpu"),workspace.statusMessage)
        XCTAssertTrue(workspace.setAppearance(.init(mode:.light,palette:.teal,transparency:.opaque)))
        let host=try XCTUnwrap(workspace.workflowSession)
        XCTAssertTrue(host.owner.select(.init(nodeAddresses:[WorkflowExamples.sourceAddress]),lease:host.owner.lease()))
        let original=host.owner.document,selection=host.owner.selection,lease=host.owner.lease()
        let graph=workspace.graph,live=workspace.appPreferences
        let file=workspace.appPreferenceStore.fileURL
        try FileManager.default.removeItem(at:file)
        try FileManager.default.createDirectory(at:file,withIntermediateDirectories:false)
        let sentinel=file.appendingPathComponent("preserve.txt")
        try Data("external directory sentinel".utf8).write(to:sentinel)
        XCTAssertFalse(workspace.setAppearance(.init(mode:.dark,palette:.rose,transparency:.subtle)))
        XCTAssertFalse(workspace.resetAppearance())
        XCTAssertNotNil(workspace.appPreferencesError)
        XCTAssertEqual(workspace.appPreferences,live)
        XCTAssertTrue(workspace.workflowSession === host)
        XCTAssertTrue(host.owner.isCurrent(lease))
        XCTAssertEqual(host.owner.document,original)
        XCTAssertEqual(host.owner.selection,selection)
        XCTAssertEqual(workspace.graph,graph)
        XCTAssertEqual(try Data(contentsOf:sentinel),Data("external directory sentinel".utf8))
    }
}
