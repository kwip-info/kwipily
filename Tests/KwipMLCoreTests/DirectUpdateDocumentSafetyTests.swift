import XCTest
@testable import KwipMLCore

final class DirectUpdateDocumentSafetyTests: XCTestCase {
    func testLegacyWorkflowSaveAlsoRefusesAProjectWrittenByAFutureVersion() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let library = ProjectLibrary(directory: directory)
        let old = try WorkflowExamples.preparedSeparableConv2dClassifier()
        let record = try library.saveSource(.workflow(old))
        let url = URL(fileURLWithPath: record.path)
        var future = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        future["schemaVersion"] = 99
        let bytes = try JSONSerialization.data(withJSONObject: future, options: [.sortedKeys])
        try bytes.write(to: url)
        XCTAssertThrowsError(try library.saveSource(.workflow(old), id: old.projectID))
        XCTAssertEqual(try Data(contentsOf: url), bytes)
    }

    func testOlderClientCannotOverwriteFutureSchemaWithItsPreviouslyLoadedRevision() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let plan = try FlowDocumentMigration.plan(WorkflowExamples.preparedSeparableConv2dClassifier())
        let url = try FlowDocumentStore.commitMigration(plan, to: directory.appendingPathComponent("Project"))
        let old = try FlowDocumentStore.readHead(at: url)
        let expected = try FlowDocumentCodec.reference(old)
        var future = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        future["schemaVersion"] = 99
        future["newRequiredMeaning"] = "must not be discarded by an old application"
        let bytes = try JSONSerialization.data(withJSONObject: future, options: [.sortedKeys])
        try bytes.write(to: url)
        var edited = old
        edited.revision += 1
        if case .workflow(var workflow) = edited.payload {
            workflow.revision = edited.revision
            workflow.name = "Old client's edit"
            edited.payload = .workflow(workflow)
        }
        XCTAssertThrowsError(try FlowDocumentStore.readHead(at: url))
        XCTAssertThrowsError(try FlowDocumentStore.save(edited, at: url, expected: expected))
        XCTAssertEqual(try Data(contentsOf: url), bytes, "Downgrading the binary must never downgrade a future document")
    }

    func testInterruptedSaveKeepsOriginalAndSuccessfulRetryRetainsPinnedRecoveryRevision() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let plan = try FlowDocumentMigration.plan(WorkflowExamples.preparedSeparableConv2dClassifier())
        let url = try FlowDocumentStore.commitMigration(plan, to: directory.appendingPathComponent("Project"))
        let old = try FlowDocumentStore.readHead(at: url)
        let before = try Data(contentsOf: url)
        let expected = try FlowDocumentCodec.reference(old)
        var edited = old
        edited.revision += 1
        if case .workflow(var workflow) = edited.payload {
            workflow.revision = edited.revision
            workflow.name = "Updated project"
            edited.payload = .workflow(workflow)
        }
        enum Interrupted: Error { case save }
        XCTAssertThrowsError(try FlowDocumentStore.save(edited, at: url, expected: expected, beforeCommit: { _ in throw Interrupted.save }))
        XCTAssertEqual(try Data(contentsOf: url), before)
        try FlowDocumentStore.save(edited, at: url, expected: expected)
        XCTAssertEqual(try FlowDocumentStore.readHead(at: url), edited)
        let archive = url.deletingLastPathComponent().appendingPathComponent(".revisions")
            .appendingPathComponent(old.id.uuidString).appendingPathComponent(expected.sha256 + ".kwipflow")
        XCTAssertEqual(try FlowDocumentCodec.decode(Data(contentsOf: archive)), old)
        XCTAssertThrowsError(try FlowDocumentStore.save(edited, at: url, expected: expected))
        XCTAssertEqual(try FlowDocumentStore.readHead(at: url), edited)
    }
}
