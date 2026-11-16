import XCTest
@testable import KwipMLCore

final class WorkflowSourcePersistenceTests: XCTestCase {
    func testCutoverKeepsOriginalAndMakesMainAuthoritative() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at:root) }
        let library = ProjectLibrary(directory:root)
        let source = try WorkflowExamples.preparedSeparableConv2dClassifier()
        _ = try library.saveSource(.workflow(source))
        let originalURL = try library.documentURL(id:source.projectID)
        let original = try Data(contentsOf:originalURL)
        let record = try library.activateFileBackedWorkflow(id:source.projectID)
        XCTAssertEqual(record.id,source.projectID)
        XCTAssertEqual(URL(fileURLWithPath:record.path).lastPathComponent,"Main.kwipflow")
        let storage = try WorkflowSourcePersistence(library:library,projectID:source.projectID)
        let envelope = try XCTUnwrap(storage.state.flowDocument)
        var edited = storage.state.document; edited.name = "Authoritative name"; edited.revision += 1
        _ = try storage.save(edited)
        XCTAssertEqual(storage.state.flowDocument?.dependencies,envelope.dependencies)
        XCTAssertEqual(try library.loadSource(id:source.projectID),.workflow(edited))
        XCTAssertEqual(try Data(contentsOf:originalURL),original)
        XCTAssertEqual(try library.listSources().filter{$0.id == source.projectID}.map(\.path),[record.path])
        XCTAssertThrowsError(try library.saveSource(.workflow(source)))
        try FileManager.default.removeItem(atPath:record.path)
        XCTAssertThrowsError(try library.loadSource(id:source.projectID),"A durable cutover must never fall back to old bytes")
        XCTAssertTrue(try library.listSources().contains{$0.id == source.projectID},"Unavailable projects must remain discoverable")
        try FileManager.default.createSymbolicLink(atPath:record.path,withDestinationPath:originalURL.path)
        XCTAssertThrowsError(try library.loadSource(id:source.projectID),"Main symlinks must not become a fallback")
        try FileManager.default.removeItem(at:URL(fileURLWithPath:record.path).deletingLastPathComponent())
        XCTAssertThrowsError(try library.loadSource(id:source.projectID),"Missing whole flow directory must never reactivate legacy source")
        XCTAssertEqual(try library.listSources().filter{$0.id == source.projectID}.map(\.path),[record.path])
    }
    func testOrdinarySaveCannotChangePinnedDefinitionsAndCASRejectsStaleWriter() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at:root) }
        let library = ProjectLibrary(directory:root)
        let source = try WorkflowExamples.preparedSeparableConv2dClassifier()
        _ = try library.saveSource(.workflow(source)); _ = try library.activateFileBackedWorkflow(id:source.projectID)
        let first = try WorkflowSourcePersistence(library:library,projectID:source.projectID)
        let stale = try WorkflowSourcePersistence(library:library,projectID:source.projectID)
        var changed = first.state.document; changed.revision += 1; changed.definitions[0].title = "Silent rewrite"
        XCTAssertThrowsError(try first.save(changed))
        changed = first.state.document; changed.revision += 1; changed.name = "Saved"
        _ = try first.save(changed)
        XCTAssertThrowsError(try stale.save(changed))
        XCTAssertEqual(stale.state.document,source)
    }
    func testExplicitChildLayerPolicyAdoptionPromotesParentAndPreservesOldPins()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let library=ProjectLibrary(directory:root)
        let source=try WorkflowTrainRecipes.migrateAuthoringSchema(WorkflowExamples.preparedParallelClassifier())
        _=try library.saveSource(.workflow(source));_=try library.activateFileBackedWorkflow(id:source.projectID)
        let storage=try WorkflowSourcePersistence(library:library,projectID:source.projectID)
        let original=storage.state
        let envelope=try XCTUnwrap(original.flowDocument)
        let dependency=try XCTUnwrap(envelope.dependencies.first)
        let file=try XCTUnwrap(storage.fileURL).deletingLastPathComponent().appendingPathComponent(dependency.relativePath)
        var child=try FlowDocumentStore.readHead(at:file)
        let old=try FlowDocumentCodec.reference(child)
        guard case .modelBlock(var definition)=child.payload else {return XCTFail("Missing Block")}
        let index=try XCTUnwrap(definition.body.nodes.firstIndex {$0.type=="linear"})
        definition.body.nodes[index].parameterSettings=["weight":.object(["trainability":.string("frozen")])]
        child.revision += 1;child.payload = .modelBlock(definition)
        try FlowDocumentStore.save(child,at:file,expected:old)
        XCTAssertEqual(try WorkflowSourcePersistence(library:library,projectID:source.projectID).state,original,"New child head cannot silently change the parent pin")
        let proposal=try storage.prepareAdoption(definition:dependency.definition,expectedHead:FlowDocumentCodec.reference(child))
        XCTAssertEqual(proposal.after.document.schemaVersion,15)
        XCTAssertEqual(proposal.after.document.modelSourceVersions,original.document.modelSourceVersions)
        _=try storage.commitAdoption(proposal)
        XCTAssertEqual(try library.loadSource(id:source.projectID),.workflow(proposal.after.document))
        XCTAssertNotEqual(storage.state.flowDocument?.dependencies,envelope.dependencies)
    }

}
