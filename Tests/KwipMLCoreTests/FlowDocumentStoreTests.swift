import XCTest
@testable import KwipMLCore

final class FlowDocumentStoreTests: XCTestCase {
    func testMigrationCommitsIndependentFilesAndReloadsExactGraph() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let source = try WorkflowExamples.preparedSeparableConv2dClassifier()
        let original = directory.appendingPathComponent("original.kwipml")
        let bytes = try WorkflowCodec.encode(source)
        try bytes.write(to: original)
        let plan = try FlowDocumentMigration.plan(source)
        let destination = directory.appendingPathComponent("Converted")
        let root = try FlowDocumentStore.commitMigration(plan, to: destination)
        XCTAssertEqual(try Data(contentsOf: original), bytes)
        XCTAssertEqual(try FlowDocumentStore.snapshot(at: root).workflow(), source)
        for path in plan.subflows.keys { XCTAssertTrue(FileManager.default.fileExists(atPath: destination.appendingPathComponent(path).path)) }
        XCTAssertThrowsError(try FlowDocumentStore.commitMigration(plan, to: destination))
        XCTAssertEqual(try FlowDocumentStore.snapshot(at: root).workflow(), source)
    }
    func testInterruptedStagingNeverPublishesPartialDirectory() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let plan = try FlowDocumentMigration.plan(WorkflowExamples.preparedSeparableConv2dClassifier())
        let destination = directory.appendingPathComponent("Converted")
        enum Failure: Error { case interrupted }
        XCTAssertThrowsError(try FlowDocumentStore.commitMigration(plan, to: destination, afterWrite: { _ in throw Failure.interrupted }))
        XCTAssertFalse(FileManager.default.fileExists(atPath: destination.path))
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: directory.path), [])
    }
    func testModifiedAndSymlinkedDependencyRefuse() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let plan = try FlowDocumentMigration.plan(WorkflowExamples.preparedSeparableConv2dClassifier())
        let destination = directory.appendingPathComponent("Converted")
        let root = try FlowDocumentStore.commitMigration(plan, to: destination)
        let path = try XCTUnwrap(plan.subflows.keys.first)
        let child = destination.appendingPathComponent(path)
        let bytes = try Data(contentsOf: child)
        var modified = try FlowDocumentCodec.decode(bytes)
        modified.revision += 1
        try FlowDocumentCodec.encode(modified).write(to: child)
        XCTAssertThrowsError(try FlowDocumentStore.snapshot(at: root))
        let external = directory.appendingPathComponent("external.kwipflow")
        try bytes.write(to: external)
        try FileManager.default.removeItem(at: child)
        try FileManager.default.createSymbolicLink(at: child, withDestinationURL: external)
        XCTAssertThrowsError(try FlowDocumentStore.snapshot(at: root))
    }
}
