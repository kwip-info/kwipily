import XCTest
@testable import KwipMLCore

final class ModelProjectLibraryTests: XCTestCase {
    func testImportBlockReidentifiesParentAndDraftWithoutOverwritingOriginal() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let library = ProjectLibrary(directory: root)
        let original = try ModelBlockEditing.beginDraft(ModelAuthoringDocument(projectID: UUID().uuidString, revision: 7, source: CompositeExamples.parallelClassifier), path: ["features"])
        let saved = try library.saveDocument(.blockModel(original))
        let url = URL(fileURLWithPath: saved.path), bytes = try Data(contentsOf: url)
        let imported = try library.importModelDocument(from: url)
        XCTAssertNotEqual(imported.id, saved.id)
        XCTAssertEqual(try Data(contentsOf: url), bytes)
        guard case .blockModel(let copy) = try library.loadDocument(id: imported.id) else { return XCTFail("Import flattened source") }
        var expected = original; expected.projectID = imported.id
        for i in expected.drafts.indices { expected.drafts[i].projectID = imported.id }
        XCTAssertEqual(copy, expected)
        XCTAssertEqual(Set(try library.listDocuments().map(\.id)), [saved.id, imported.id])
        XCTAssertThrowsError(try library.importDocument(from: url))
    }

    func testFlatImportPreservesExactBytesAndLegacySchema() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let url = root.appendingPathComponent("portable.kwipml")
        let bytes = Data("{ \"schemaVersion\" : 1, \"name\" : \"Raw flat\", \"nodes\" : [], \"edges\" : [] }\n".utf8)
        try bytes.write(to: url)
        let library = ProjectLibrary(directory: root.appendingPathComponent("managed"))
        let first = try library.importModelDocument(from: url)
        let second = try library.importModelDocument(from: url)
        XCTAssertNotEqual(first.id, second.id)
        XCTAssertEqual(try Data(contentsOf: URL(fileURLWithPath: first.path)), bytes)
        XCTAssertEqual(try Data(contentsOf: URL(fileURLWithPath: second.path)), bytes)
        XCTAssertEqual(try Data(contentsOf: url), bytes)
        XCTAssertEqual(try library.load(id: first.id).schemaVersion, 1)
    }

    func testImportRejectsMalformedEnvelopeAndUnsafeInputWithoutCreatingProjects() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let url = root.appendingPathComponent("portable.kwipml")
        let library = ProjectLibrary(directory: root.appendingPathComponent("managed"))
        var invalid = ModelAuthoringDocument(projectID: "not-a-uuid", source: CompositeExamples.parallelClassifier)
        try JSONEncoder().encode(invalid).write(to: url)
        XCTAssertThrowsError(try library.importModelDocument(from: url))
        invalid.projectID = UUID().uuidString; invalid.kind = "unknown"
        try JSONEncoder().encode(invalid).write(to: url)
        XCTAssertThrowsError(try library.importModelDocument(from: url))
        try Data(repeating: 32, count: ModelProjectCodec.maximumBytes + 1).write(to: url)
        XCTAssertThrowsError(try library.importModelDocument(from: url))
        let link = root.appendingPathComponent("link.kwipml")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: url)
        XCTAssertThrowsError(try library.importModelDocument(from: link))
        XCTAssertThrowsError(try library.importModelDocument(from: root))
        XCTAssertTrue(try library.listDocuments().isEmpty)
    }
    func testDocumentAwareHistoryCanExplicitlyRestoreOriginalFlatFormat() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let library=ProjectLibrary(directory:root)
        let flat=try library.save(ExampleFlows.parallelClassifier)
        let before=try Data(contentsOf:library.documentURL(id:flat.id))
        let nested=ModelProjectDocument.blockModel(ModelAuthoringDocument(projectID:flat.id,source:CompositeExamples.parallelClassifier))
        _=try library.saveDocument(nested,id:flat.id)
        XCTAssertThrowsError(try library.save(ExampleFlows.parallelClassifier,id:flat.id))
        _=try library.saveDocument(.legacyFlat(ExampleFlows.parallelClassifier),id:flat.id)
        XCTAssertEqual(try Data(contentsOf:library.documentURL(id:flat.id)),before)
        XCTAssertEqual(try library.load(id:flat.id),ExampleFlows.parallelClassifier)
    }
    func testDualFormatRecentsAndLegacyAPIsNeverFlattenBlockSource() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let library=ProjectLibrary(directory:root)
        let flat=try library.save(ExampleFlows.parallelClassifier)
        let flatURL=try library.documentURL(id:flat.id)
        let flatBytes=try Data(contentsOf:flatURL)
        let id=UUID().uuidString
        let nested=ModelProjectDocument.blockModel(ModelAuthoringDocument(projectID:id,source:CompositeExamples.parallelClassifier))
        let saved=try library.saveDocument(nested,id:id)
        XCTAssertEqual(saved.id,id)
        XCTAssertEqual(try library.loadDocument(id:id),nested)
        XCTAssertEqual(try library.loadDocument(id:flat.id),.legacyFlat(ExampleFlows.parallelClassifier))
        XCTAssertEqual(try library.listDocuments().map(\.id),[id,flat.id])
        XCTAssertEqual(try library.list().map(\.id),[flat.id])
        XCTAssertThrowsError(try library.load(id:id))
        let before=try Data(contentsOf:library.documentURL(id:id))
        XCTAssertThrowsError(try library.save(ExampleFlows.parallelClassifier,id:id))
        XCTAssertEqual(try Data(contentsOf:library.documentURL(id:id)),before)
        XCTAssertEqual(try Data(contentsOf:flatURL),flatBytes)
    }
    func testIdentityMismatchAndCorruptionNeverOverwriteOrMasqueradeAsFlat() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let library=ProjectLibrary(directory:root)
        let id=UUID().uuidString,other=UUID().uuidString
        let nested=ModelProjectDocument.blockModel(ModelAuthoringDocument(projectID:id,source:CompositeExamples.parallelClassifier))
        XCTAssertThrowsError(try library.saveDocument(nested,id:other))
        XCTAssertFalse(FileManager.default.fileExists(atPath:try library.documentURL(id:other).path))
        _=try library.saveDocument(nested,id:id)
        let url=try library.documentURL(id:id)
        try FileManager.default.copyItem(at:url,to:library.documentURL(id:other))
        XCTAssertThrowsError(try library.loadDocument(id:other))
        XCTAssertEqual(try library.listDocuments().map(\.id),[id])
        let invalid=Data(#"{"kind":"modelProject","schemaVersion":1,"name":"Wrong","nodes":[],"edges":[]}"#.utf8)
        try invalid.write(to:url)
        XCTAssertThrowsError(try library.loadDocument(id:id))
        XCTAssertThrowsError(try library.load(id:id))
        XCTAssertEqual(try library.listDocuments(),[])
        XCTAssertEqual(try Data(contentsOf:url),invalid)
    }
}
