import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkspaceToolbarTests: XCTestCase {
    func testRunConfigurationRoutesOnlyCurrentControlNodesAndRevealsSelection() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let workspace = WorkspaceStore(rootOverride: repo, storageOverride: root, detectCapabilities: false)
        workspace.newProject()
        let host = try XCTUnwrap(workspace.workflowSession)
        let setup = WorkflowNodeAddress(domain: .control, localID: "setup")
        let train = WorkflowNodeAddress(domain: .control, localID: "train")
        XCTAssertNil(host.owner.apply(.insert(.init(address:setup,payload:.runSetup(config:[:]))),lease:host.owner.lease()),"Retired Run setup cannot be newly placed")
        XCTAssertNotNil(host.owner.apply(.insert(.init(address:train,name:"First path",payload:.train(config:["runSetup":.object([:])]))),lease:host.owner.lease()))
        XCTAssertEqual(workspace.runConfigurationNodes.map(\.address),[train])
        workspace.activeSection = "Code"
        XCTAssertTrue(workspace.revealRunConfiguration(train))
        XCTAssertEqual(workspace.activeSection, "Build")
        XCTAssertTrue(workspace.showInspector)
        XCTAssertEqual(host.owner.selection.nodeAddresses, [train])
        XCTAssertNotNil(host.canvas.canvasRevealRequest)
        let revision = host.owner.document.revision
        XCTAssertFalse(workspace.revealRunConfiguration(.init(domain: .control, localID: "missing")))
        XCTAssertEqual(host.owner.document.revision, revision, "Navigation must not edit computation")
        host.owner.setExecutionBusy(true)
        XCTAssertFalse(workspace.revealRunConfiguration(train))
        XCTAssertEqual(host.owner.selection.nodeAddresses, [train])
    }
    func testFeedbackToggleMergesLatestPreferencesAcrossProjectWindows() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let workspace = WorkspaceStore(rootOverride: repo, storageOverride: root, detectCapabilities: false)
        var latest = try workspace.appPreferenceStore.load()
        latest.runFeedback = .init(showStatusIcon: false, notifyCompleted: true, notifyFailed: false)
        try workspace.appPreferenceStore.save(latest)
        runFeedbackBinding(store: workspace, path: \.notifyFailed).wrappedValue = true
        let saved = try workspace.appPreferenceStore.load().effectiveRunFeedback
        XCTAssertFalse(saved.showStatusIcon)
        XCTAssertTrue(saved.notifyCompleted, "Toggling one field must preserve another window’s changes")
        XCTAssertTrue(saved.notifyFailed)
    }
    func testToolbarPresentationKeepsCanonicalSectionIDs() {
        XCTAssertEqual(WorkspaceToolbarSection.title("Settings"), "Configure")
        XCTAssertEqual(WorkspaceToolbarSection.title("History"), "History")
        XCTAssertEqual(WorkspaceToolbarSection.symbol("Build"), "point.3.connected.trianglepath.dotted")
        XCTAssertEqual(WorkspaceToolbarSection.symbol("Code"), "chevron.left.forwardslash.chevron.right")
    }
}
