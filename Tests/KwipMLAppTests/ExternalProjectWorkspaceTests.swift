import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class ExternalProjectWorkspaceTests: XCTestCase {
    func testCopiesIsolatePrivateStateWhileSharingAppPreferencesAndPermissions() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let source = try WorkflowExamples.preparedSeparableConv2dClassifier()
        let plan = try FlowDocumentMigration.plan(source)
        let firstURL = try FlowDocumentStore.commitMigration(plan, to: directory.appendingPathComponent("first"))
        let secondURL = try FlowDocumentStore.commitMigration(plan, to: directory.appendingPathComponent("second"))
        let application = directory.appendingPathComponent("app")
        let a = WorkspaceStore(rootOverride: root, storageOverride: application, detectCapabilities: false,
            externalLocation: .init(id: UUID(), projectID: source.projectID, rootURL: firstURL))
        let b = WorkspaceStore(rootOverride: root, storageOverride: application, detectCapabilities: false,
            externalLocation: .init(id: UUID(), projectID: source.projectID, rootURL: secondURL))
        XCTAssertNotEqual(a.workspaceDirectory, b.workspaceDirectory)
        XCTAssertEqual(a.applicationDirectory, application)
        XCTAssertEqual(a.sourceAccess.directory, b.sourceAccess.directory)
        XCTAssertEqual(a.sourceAccess.directory, application.appendingPathComponent("source-permissions"))
        XCTAssertTrue(a.beginWorkflowProject(id: source.projectID))
        XCTAssertTrue(b.beginWorkflowProject(id: source.projectID))
        XCTAssertEqual(a.workflowSession?.owner.flowRootURL, firstURL)
        XCTAssertEqual(b.workflowSession?.owner.flowRootURL, secondURL)
        XCTAssertEqual(a.projectURL, firstURL)
        XCTAssertEqual(b.projectURL, secondURL)
        var defaults = a.appPreferences.canvasDefaults; defaults.flowDirection = .topToBottom
        XCTAssertTrue(a.setCanvasDefaults(defaults))
        b.loadAppPreferences()
        XCTAssertEqual(b.appPreferences.canvasDefaults, defaults)
        let owner = try XCTUnwrap(a.workflowSession?.owner)
        XCTAssertTrue(owner.rename("Only first copy", lease: owner.lease()))
        XCTAssertEqual(b.workflowSession?.owner.document.name, source.name)
        XCTAssertEqual(try FlowDocumentStore.snapshot(at: secondURL).workflow().name, source.name)
        a.retireWorkflowSession(); b.retireWorkflowSession()
    }
}
