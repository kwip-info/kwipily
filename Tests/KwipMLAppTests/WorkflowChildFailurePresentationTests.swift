import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowChildFailurePresentationTests: XCTestCase {
    func testActualPrivateCopyAndOpenCallbacksExposeCurrentFailureAndClearAfterRetry() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let workspace = WorkspaceStore(rootOverride: repo, storageOverride: directory, detectCapabilities: false)
        let source = try WorkflowExamples.preparedSeparableConv2dClassifier()
        _ = try workspace.projectLibrary.saveSource(.workflow(source))
        _ = try workspace.projectLibrary.activateFileBackedWorkflow(id: source.projectID)
        XCTAssertTrue(workspace.beginWorkflowProject(id: source.projectID), workspace.statusMessage)
        let host = try XCTUnwrap(workspace.workflowSession), children = try XCTUnwrap(workspace.workflowChildren)
        children.present = { session, path in session.open(path: path) }
        let root = WorkflowNodeAddress(domain: .model, localID: "convolution")
        let url = try XCTUnwrap(host.owner.flowRootURL), saved = try Data(contentsOf: url)
        XCTAssertTrue(workspace.setAuthoringRetention(try .init(maximumFiles: 1, maximumBytes: 1)))
        try XCTUnwrap(host.canvas.onEditPrivateBlock)(root, host.owner.lease())
        XCTAssertTrue(host.canvas.footerStatusMessage.contains("Settings"), host.canvas.footerStatusMessage)
        XCTAssertTrue(host.canvas.footerStatusMessage.contains("limit"), host.canvas.footerStatusMessage)
        XCTAssertTrue(host.owner.document.drafts.isEmpty)
        XCTAssertEqual(try Data(contentsOf: url), saved)
        XCTAssertTrue(workspace.setAuthoringRetention(.factory))
        try XCTUnwrap(host.canvas.onEditPrivateBlock)(root, host.owner.lease())
        XCTAssertEqual(host.owner.document.drafts.count, 1)
        XCTAssertEqual(host.canvas.footerStatusMessage, host.canvas.statusMessage)
        XCTAssertFalse(host.canvas.footerStatusMessage.contains("authoring-evidence limit"))
        XCTAssertTrue(try XCTUnwrap(children.session).discard())
        children.presentFlow = { _, _ in false }
        try XCTUnwrap(host.canvas.onOpenBlock)(root, host.owner.lease())
        XCTAssertEqual(host.canvas.footerStatusMessage, try XCTUnwrap(children.error))
        children.presentFlow = { _, _ in true }
        try XCTUnwrap(host.canvas.onOpenBlock)(root, host.owner.lease())
        XCTAssertEqual(host.canvas.footerStatusMessage, host.canvas.statusMessage)
        workspace.retireWorkflowSession()
    }
}
