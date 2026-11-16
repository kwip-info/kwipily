import XCTest
@testable import KwipMLCore

final class WorkflowDependencyAdoptionTests: XCTestCase {
    private func setup() throws -> (URL, WorkflowSourcePersistence, FlowDocumentDependency, URL) {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let library = ProjectLibrary(directory:directory)
        let source = try WorkflowExamples.preparedSeparableConv2dClassifier()
        _ = try library.saveSource(.workflow(source))
        _ = try library.activateFileBackedWorkflow(id:source.projectID)
        let storage = try WorkflowSourcePersistence(library:library,projectID:source.projectID)
        let dependency = try XCTUnwrap(storage.state.flowDocument?.dependencies.first)
        let childURL = try XCTUnwrap(storage.fileURL).deletingLastPathComponent().appendingPathComponent(dependency.relativePath)
        return (directory, storage, dependency, childURL)
    }
    private func advance(_ url: URL, title: String) throws -> FlowDocumentRevision {
        var child = try FlowDocumentCodec.decode(Data(contentsOf:url))
        let old = try FlowDocumentCodec.reference(child)
        child.revision += 1
        if case .modelBlock(var block) = child.payload { block.title = title; child.payload = .modelBlock(block) }
        try FlowDocumentStore.save(child,at:url,expected:old)
        return try FlowDocumentCodec.reference(child)
    }
    func testPrepareDoesNotWriteCommitAdoptsAndUndoCanRestoreExactOldPins() throws {
        let (directory,storage,dependency,childURL) = try setup()
        defer { try? FileManager.default.removeItem(at:directory) }
        let before = storage.state
        let rootURL = try XCTUnwrap(storage.fileURL), bytes = try Data(contentsOf:rootURL)
        let head = try advance(childURL,title:"Reviewed new child")
        XCTAssertEqual(try storage.adoptionHead(definition:dependency.definition),head)
        let proposal = try storage.prepareAdoption(definition:dependency.definition,expectedHead:head)
        XCTAssertEqual(proposal.before,before)
        XCTAssertEqual(proposal.after.document.definitions.first?.title,"Reviewed new child")
        XCTAssertEqual(storage.state,before)
        XCTAssertEqual(try Data(contentsOf:rootURL),bytes)
        _ = try storage.commitAdoption(proposal)
        XCTAssertEqual(storage.state,proposal.after)
        XCTAssertEqual(storage.state.flowDocument?.dependencies.first?.revision,head)
        var undone = before.document; undone.revision = storage.state.document.revision + 1
        _ = try storage.save(undone,restoring:before.flowDocument)
        XCTAssertEqual(storage.state.flowDocument?.dependencies.first?.revision,dependency.revision)
        XCTAssertEqual(storage.state.document.definitions,before.document.definitions)
    }
    func testStaleChildOrParentCannotCommitReviewedProposal() throws {
        let (directory,storage,dependency,childURL) = try setup()
        defer { try? FileManager.default.removeItem(at:directory) }
        let head = try advance(childURL,title:"First")
        let proposal = try storage.prepareAdoption(definition:dependency.definition,expectedHead:head)
        _ = try advance(childURL,title:"Second")
        XCTAssertThrowsError(try storage.prepareAdoption(definition:dependency.definition,expectedHead:head))
        XCTAssertThrowsError(try storage.commitAdoption(proposal))
        XCTAssertEqual(storage.state,proposal.before)
        let latest = try storage.adoptionHead(definition:dependency.definition)
        let fresh = try storage.prepareAdoption(definition:dependency.definition,expectedHead:latest)
        var edit = storage.state.document; edit.revision += 1; edit.name = "Parent changed"
        _ = try storage.save(edit)
        XCTAssertThrowsError(try storage.commitAdoption(fresh))
        XCTAssertEqual(storage.state.document,edit)
    }
    func testWrongIdentityDefinitionAndUnchangedHeadRefuseWithoutWrites() throws {
        let (directory,storage,dependency,childURL) = try setup()
        defer { try? FileManager.default.removeItem(at:directory) }
        let before = storage.state
        XCTAssertThrowsError(try storage.prepareAdoption(definition:dependency.definition,expectedHead:dependency.revision))
        var child = try FlowDocumentCodec.decode(Data(contentsOf:childURL))
        child.id = UUID()
        try FlowDocumentCodec.encode(child).write(to:childURL)
        XCTAssertThrowsError(try storage.adoptionHead(definition:dependency.definition))
        child.id = dependency.revision.documentID
        if case .modelBlock(var block) = child.payload { block.reference.id = "different-definition"; child.payload = .modelBlock(block) }
        try FlowDocumentCodec.encode(child).write(to:childURL)
        XCTAssertThrowsError(try storage.adoptionHead(definition:dependency.definition))
        XCTAssertEqual(storage.state,before)
    }
}
