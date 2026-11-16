import XCTest
@testable import KwipMLCore

final class FlowDocumentRecoveryTests: XCTestCase {
    func testInterruptedEditsRecoverWithoutChangingSavedFile() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let plan = try FlowDocumentMigration.plan(WorkflowExamples.preparedSeparableConv2dClassifier())
        let root = try FlowDocumentStore.commitMigration(plan, to: directory.appendingPathComponent("Project"))
        let url = root.deletingLastPathComponent().appendingPathComponent(try XCTUnwrap(plan.subflows.keys.first))
        let bytes = try Data(contentsOf: url), original = try FlowDocumentCodec.decode(bytes)
        var edited = original
        if case .modelBlock(var block) = edited.payload { block.title = "Recovered construction"; edited.payload = .modelBlock(block) }
        let first = try FlowDocumentRecoveryStore.write(edited, base: original, beside: url, sessionID: UUID(), supersedes: [])
        XCTAssertEqual(try Data(contentsOf: url), bytes)
        XCTAssertEqual(try FlowDocumentRecoveryStore.records(beside: url, documentID: original.id), [first])
        XCTAssertEqual(try first.document(), edited)
        let continued = try FlowDocumentRecoveryStore.write(edited, base: original, beside: url, sessionID: UUID(), supersedes: [first.sessionID])
        XCTAssertEqual(try FlowDocumentRecoveryStore.records(beside: url, documentID: original.id), [continued])
    }
    func testStaleRecoveryWriterCannotOverwriteNewerPendingContent() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let original = try FlowDocumentMigration.plan(WorkflowExamples.preparedMaxPool3dClassifier()).root
        let url = directory.appendingPathComponent("Main.kwipflow"), sessionID = UUID()
        let first = try FlowDocumentRecoveryStore.write(original, base: original, beside: url, sessionID: sessionID, supersedes: [])
        var edited = original
        if case .workflow(var flow) = edited.payload { flow.name = "Newer unsaved content"; edited.payload = .workflow(flow) }
        let next = try FlowDocumentRecoveryStore.write(edited, base: original, beside: url, sessionID: sessionID, supersedes: [], replacing: first)
        XCTAssertThrowsError(try FlowDocumentRecoveryStore.write(original, base: original, beside: url, sessionID: sessionID, supersedes: [], replacing: first))
        XCTAssertEqual(try FlowDocumentRecoveryStore.records(beside: url, documentID: original.id), [next])
    }
    func testConcurrentRecoveryBranchesRemainDistinct() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let original = try FlowDocumentMigration.plan(WorkflowExamples.preparedMaxPool3dClassifier()).root
        let url = directory.appendingPathComponent("Main.kwipflow")
        let first = try FlowDocumentRecoveryStore.write(original, base: original, beside: url, sessionID: UUID(), supersedes: [])
        let second = try FlowDocumentRecoveryStore.write(original, base: original, beside: url, sessionID: UUID(), supersedes: [])
        XCTAssertEqual(Set(try FlowDocumentRecoveryStore.records(beside: url, documentID: original.id).map(\.sessionID)), [first.sessionID, second.sessionID])
        try FlowDocumentRecoveryStore.remove(first, beside: url)
        XCTAssertEqual(try FlowDocumentRecoveryStore.records(beside: url, documentID: original.id), [second])
    }
}
