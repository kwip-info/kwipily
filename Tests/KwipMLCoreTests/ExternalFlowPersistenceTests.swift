import XCTest
@testable import KwipMLCore

final class ExternalFlowPersistenceTests: XCTestCase {
    func testExternalRootEditsExactFileWithoutTouchingSameIdentityOriginal() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let original = ProjectLibrary(directory: root.appendingPathComponent("original"))
        let source = try WorkflowExamples.preparedSeparableConv2dClassifier()
        _ = try original.saveSource(.workflow(source))
        let originalURL = try original.documentURL(id: source.projectID)
        let originalBytes = try Data(contentsOf: originalURL)
        let file = try FlowDocumentStore.commitMigration(FlowDocumentMigration.plan(source), to: root.appendingPathComponent("export"))
        let library = ProjectLibrary(directory: root.appendingPathComponent("private"), externalFlowRoot: file, projectID: source.projectID)
        let owner = try WorkflowSourcePersistence(library: library, projectID: source.projectID)
        let stale = try WorkflowSourcePersistence(library: library, projectID: source.projectID)
        let pins = owner.state.flowDocument?.dependencies
        var edited = owner.state.document; edited.name = "External copy"; edited.revision += 1
        _ = try owner.save(edited)
        XCTAssertEqual(try FlowDocumentStore.snapshot(at: file).workflow(), edited)
        XCTAssertEqual(owner.state.flowDocument?.dependencies, pins)
        XCTAssertEqual(try Data(contentsOf: originalURL), originalBytes)
        XCTAssertEqual(try library.listSources().map(\.path), [file.path])
        XCTAssertThrowsError(try stale.save(edited))
        XCTAssertThrowsError(try library.loadSource(id: UUID().uuidString))
        XCTAssertThrowsError(try library.saveSource(.workflow(source)))
        XCTAssertEqual(try library.activateFileBackedWorkflow(id: source.projectID).path, file.path)
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("private").path))
        try FileManager.default.removeItem(at: file)
        XCTAssertThrowsError(try library.loadSource(id: source.projectID))
    }
}
