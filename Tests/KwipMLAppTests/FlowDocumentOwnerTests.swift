import XCTest
@testable import KwipMLApp
import KwipMLCore

@MainActor final class FlowDocumentOwnerTests: XCTestCase {
    func testUnsavedContentSurvivesOwnerRecreation() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let plan = try FlowDocumentMigration.plan(WorkflowExamples.preparedSeparableConv2dClassifier())
        let root = try FlowDocumentStore.commitMigration(plan, to: directory.appendingPathComponent("Project"))
        let url = root.deletingLastPathComponent().appendingPathComponent(try XCTUnwrap(plan.subflows.keys.first))
        let original = try Data(contentsOf: url)
        var interrupted: FlowDocumentOwner? = try FlowDocumentOwner(url: url)
        XCTAssertTrue(interrupted!.edit { doc in
            if case .modelBlock(var block) = doc.payload { block.title = "Recovered unsaved edit"; doc.payload = .modelBlock(block) }
        })
        let pending = interrupted!.document
        interrupted = nil
        let recovered = try FlowDocumentOwner(url: url)
        XCTAssertEqual(recovered.document, pending)
        XCTAssertTrue(recovered.isDirty)
        XCTAssertEqual(try Data(contentsOf: url), original)
        XCTAssertTrue(recovered.save())
        let reopened = try FlowDocumentOwner(url: url)
        XCTAssertEqual(reopened.document, recovered.document)
        XCTAssertFalse(reopened.isDirty)
    }
    func testTwoRestoredOwnersCannotOverwriteEachOthersRecovery() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let plan = try FlowDocumentMigration.plan(WorkflowExamples.preparedSeparableConv2dClassifier())
        let root = try FlowDocumentStore.commitMigration(plan, to: directory.appendingPathComponent("Project"))
        let url = root.deletingLastPathComponent().appendingPathComponent(try XCTUnwrap(plan.subflows.keys.first))
        let initial = try FlowDocumentOwner(url: url)
        func rename(_ owner: FlowDocumentOwner, _ title: String) -> Bool {
            owner.edit { doc in if case .modelBlock(var block) = doc.payload { block.title = title; doc.payload = .modelBlock(block) } }
        }
        XCTAssertTrue(rename(initial, "Pending"))
        let first = try FlowDocumentOwner(url: url), stale = try FlowDocumentOwner(url: url)
        XCTAssertTrue(rename(first, "More recent pending content"))
        let before = stale.document
        XCTAssertFalse(rename(stale, "Cannot overwrite recovery"))
        XCTAssertEqual(stale.document, before)
        XCTAssertFalse(stale.revert())
        XCTAssertEqual(try FlowDocumentOwner(url: url).document, first.document)
    }
    func testIndependentEditsUndoSaveAndReopen() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let plan = try FlowDocumentMigration.plan(WorkflowExamples.preparedSeparableConv2dClassifier())
        let root = try FlowDocumentStore.commitMigration(plan, to: directory.appendingPathComponent("Project"))
        let childURL = root.deletingLastPathComponent().appendingPathComponent(try XCTUnwrap(plan.subflows.keys.first))
        let parent = try FlowDocumentOwner(url: root), child = try FlowDocumentOwner(url: childURL)
        let original = child.document
        XCTAssertTrue(child.edit { document in
            if case .modelBlock(var block) = document.payload { block.title = "Independent edit"; document.payload = .modelBlock(block) }
        })
        XCTAssertTrue(child.isDirty); XCTAssertFalse(parent.isDirty)
        XCTAssertTrue(child.undo()); XCTAssertEqual(child.document, original); XCTAssertFalse(child.isDirty)
        XCTAssertTrue(child.redo()); XCTAssertTrue(child.isDirty)
        XCTAssertTrue(child.save()); XCTAssertFalse(child.isDirty)
        XCTAssertEqual(child.document.revision, original.revision + 1)
        XCTAssertEqual(try FlowDocumentOwner(url: childURL).document, child.document)
        XCTAssertTrue(child.undo()); XCTAssertTrue(child.isDirty)
        XCTAssertTrue(child.save()); XCTAssertEqual(child.document.revision, original.revision + 2)
        XCTAssertEqual(try FlowDocumentStore.snapshot(at: root), try plan.snapshot())
    }
    func testConflictingOwnerSavePreservesUnsavedEditsAndRejectsIdentityMutation() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let plan = try FlowDocumentMigration.plan(WorkflowExamples.preparedSeparableConv2dClassifier())
        let root = try FlowDocumentStore.commitMigration(plan, to: directory.appendingPathComponent("Project"))
        let childURL = root.deletingLastPathComponent().appendingPathComponent(try XCTUnwrap(plan.subflows.keys.first))
        let first = try FlowDocumentOwner(url: childURL), second = try FlowDocumentOwner(url: childURL)
        func rename(_ owner: FlowDocumentOwner, _ title: String) {
            XCTAssertTrue(owner.edit { doc in if case .modelBlock(var block) = doc.payload { block.title = title; doc.payload = .modelBlock(block) } })
        }
        rename(first, "First"); rename(second, "Second")
        XCTAssertTrue(first.save()); XCTAssertFalse(second.save())
        XCTAssertTrue(second.isDirty); XCTAssertNotNil(second.error)
        let before = second.document
        XCTAssertFalse(second.edit { $0.id = UUID() }); XCTAssertEqual(second.document, before)
        XCTAssertEqual(try FlowDocumentOwner(url: childURL).document, first.document)
    }
}
