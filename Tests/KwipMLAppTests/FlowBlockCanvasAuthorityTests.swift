import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class FlowBlockCanvasAuthorityTests: XCTestCase {
    func testExistingCanvasEditsIndependentFileAndRoutesUndo() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let plan = try FlowDocumentMigration.plan(WorkflowExamples.preparedSeparableConv2dClassifier())
        let root = try FlowDocumentStore.commitMigration(plan, to: directory.appendingPathComponent("Project"))
        let child = root.deletingLastPathComponent().appendingPathComponent(try XCTUnwrap(plan.subflows.keys.first))
        let owner = try FlowDocumentOwner(url: child)
        let authority = try FlowBlockCanvasAuthority(owner: owner)
        let canvas = try ModelCanvasAdapter(owner: authority, path: authority.path)
        XCTAssertFalse(canvas.isBusy); XCTAssertTrue(canvas.editingDraft)
        let stale = authority.lease(path: authority.path)
        let original = try XCTUnwrap(canvas.activeDefinition)
        var edited = original; edited.title = "Peer canvas"
        XCTAssertTrue(authority.updateDraft(edited, lease: stale))
        XCTAssertTrue(owner.isDirty); XCTAssertTrue(canvas.canUndo)
        XCTAssertFalse(authority.updateDraft(original, lease: stale))
        canvas.undo(); XCTAssertEqual(canvas.activeDefinition, original)
        XCTAssertTrue(canvas.canRedo); canvas.redo()
        XCTAssertEqual(canvas.activeDefinition?.title, "Peer canvas")
        XCTAssertTrue(authority.applyDraft(lease: authority.lease(path: authority.path)))
        XCTAssertFalse(owner.isDirty)
        XCTAssertEqual(try FlowDocumentStore.snapshot(at: root), try plan.snapshot())
        XCTAssertEqual(try FlowDocumentOwner(url: child).document, owner.document)
    }
}
