import XCTest
@testable import KwipMLCore

final class WorkflowProjectLibraryTests: XCTestCase {
    private func directory() -> URL { FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString) }

    func testMixedSourceListRoundTripAndOldModelAPIIsolation() throws {
        let root=directory(); defer {try? FileManager.default.removeItem(at:root)}
        let library=ProjectLibrary(directory:root)
        let flat=try library.saveSource(.legacyFlat(ExampleFlows.parallelClassifier))
        let model=ModelAuthoringDocument(projectID:UUID().uuidString,source:CompositeExamples.parallelClassifier)
        let block=try library.saveSource(.blockModel(model))
        let workflow=WorkflowDocument(projectID:UUID().uuidString,name:"Universal")
        let mixed=try library.saveSource(.workflow(workflow))
        XCTAssertEqual(try library.loadSource(id:mixed.id),.workflow(workflow))
        XCTAssertEqual(try library.loadSource(id:block.id),.blockModel(model))
        XCTAssertEqual(try library.loadSource(id:flat.id),.legacyFlat(ExampleFlows.parallelClassifier))
        XCTAssertEqual(try library.listSources().map(\.id),[mixed.id,block.id,flat.id])
        XCTAssertEqual(Set(try library.listDocuments().map(\.id)),[block.id,flat.id])
        XCTAssertEqual(try library.list().map(\.id),[flat.id])
        XCTAssertThrowsError(try library.loadDocument(id:mixed.id))
    }

    func testModelSaveCannotDestroyWorkflowAndCrossFamilySourceSaveIsExplicitlyRejected() throws {
        let root=directory(); defer {try? FileManager.default.removeItem(at:root)}
        let library=ProjectLibrary(directory:root), id=UUID().uuidString
        let workflow=WorkflowDocument(projectID:id,name:"Keep source")
        let record=try library.saveSource(.workflow(workflow))
        let url=URL(fileURLWithPath:record.path), before=try Data(contentsOf:url)
        XCTAssertThrowsError(try library.saveDocument(.legacyFlat(ExampleFlows.parallelClassifier),id:id))
        XCTAssertThrowsError(try library.saveDocument(.blockModel(.init(projectID:id,source:CompositeExamples.parallelClassifier)),id:id))
        XCTAssertThrowsError(try library.save(ExampleFlows.parallelClassifier,id:id))
        XCTAssertThrowsError(try library.saveSource(.legacyFlat(ExampleFlows.parallelClassifier),id:id))
        XCTAssertEqual(try Data(contentsOf:url),before)
        let flat=try library.save(ExampleFlows.parallelClassifier)
        XCTAssertThrowsError(try library.saveSource(.workflow(.init(projectID:flat.id)),id:flat.id))
        XCTAssertEqual(try library.load(id:flat.id),ExampleFlows.parallelClassifier)
    }

    func testWorkflowImportRebasesOnlyParentAndDraftIdentity() throws {
        let root=directory(); defer {try? FileManager.default.removeItem(at:root)}
        let library=ProjectLibrary(directory:root)
        let model=try ModelBlockEditing.beginDraft(.init(projectID:UUID().uuidString,revision:8,source:CompositeExamples.parallelClassifier),path:["features"])
        let workflow=try WorkflowCodec.migrateModel(.blockModel(model),projectID:model.projectID).document
        let original=try library.saveSource(.workflow(workflow))
        let url=URL(fileURLWithPath:original.path), before=try Data(contentsOf:url)
        let imported=try library.importSource(from:url)
        var expected=workflow; expected.projectID=imported.id
        for index in expected.drafts.indices {expected.drafts[index].projectID=imported.id}
        XCTAssertNotEqual(imported.id,original.id)
        XCTAssertEqual(try library.loadSource(id:imported.id),.workflow(expected))
        XCTAssertEqual(try Data(contentsOf:url),before)
        XCTAssertFalse(String(decoding:try ProjectSourceCodec.encode(.workflow(expected)),as:UTF8.self).contains("preparedData"))
    }

    func testIdentityMalformedTaggedAndUnsafeFilesFailWithoutOverwrite() throws {
        let root=directory(); defer {try? FileManager.default.removeItem(at:root)}
        let library=ProjectLibrary(directory:root), id=UUID().uuidString, other=UUID().uuidString
        let workflow=WorkflowDocument(projectID:id)
        XCTAssertThrowsError(try library.saveSource(.workflow(workflow),id:other))
        _=try library.saveSource(.workflow(workflow))
        try FileManager.default.copyItem(at:library.documentURL(id:id),to:library.documentURL(id:other))
        XCTAssertThrowsError(try library.loadSource(id:other))
        XCTAssertEqual(try library.listSources().map(\.id),[id])
        let bytes=Data(#"{"kind":"future","schemaVersion":1,"name":"Fake flat","nodes":[],"edges":[]}"#.utf8)
        XCTAssertThrowsError(try ProjectSourceCodec.decode(bytes))
        let url=try library.documentURL(id:id);try bytes.write(to:url)
        XCTAssertThrowsError(try library.saveDocument(.legacyFlat(ExampleFlows.parallelClassifier),id:id))
        XCTAssertEqual(try Data(contentsOf:url),bytes)
        let link=try library.documentURL(id:UUID().uuidString)
        try FileManager.default.createSymbolicLink(at:link,withDestinationURL:url)
        XCTAssertThrowsError(try library.importSource(from:link))
        XCTAssertThrowsError(try library.saveSource(.workflow(.init(projectID:link.deletingPathExtension().lastPathComponent))))
        XCTAssertThrowsError(try library.importSource(from:root))
    }

    func testLegacySourceImportKeepsRawBytesAndOversizedInputDoesNotCreateProject() throws {
        let root=directory(); defer {try? FileManager.default.removeItem(at:root)}
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        let source=root.appendingPathComponent("portable.kwipml")
        let bytes=Data("{ \"schemaVersion\":1, \"name\":\"Legacy\", \"nodes\":[], \"edges\":[] }\n".utf8)
        try bytes.write(to:source)
        let library=ProjectLibrary(directory:root.appendingPathComponent("managed"))
        let record=try library.importSource(from:source)
        XCTAssertEqual(try Data(contentsOf:URL(fileURLWithPath:record.path)),bytes)
        try Data(repeating:32,count:2_000_001).write(to:source)
        XCTAssertThrowsError(try library.importSource(from:source))
        XCTAssertEqual(try library.listSources().count,1)
    }
}
