import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp

@MainActor final class ModelCanvasRootAuthorityTests: XCTestCase {
    func testRootCommandsUseDeclaredAuthorityInsteadOfConcreteStoreType() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let library = ProjectLibrary(directory: directory)
        let source = try ModuleCompositionLearningExamples.defaultSource(task: .orderName)
        let id = UUID().uuidString
        _ = try library.saveDocument(.blockModel(.init(projectID: id, source: source, schemaVersion: 2)), id: id)
        let backing = try ModelAuthoringStore(library: library, projectID: id)
        let authority = BorrowedRootAuthority(backing)
        let canvas = try ModelCanvasAdapter(owner: authority)
        let before = try XCTUnwrap(canvas.graph.nodes.first { $0.id == "input" })
        canvas.moveNodes(ids: ["input"], dx: 35, dy: 20)
        let after = try XCTUnwrap(canvas.graph.nodes.first { $0.id == "input" })
        XCTAssertEqual(after.x, before.x + 35)
        XCTAssertEqual(after.y, before.y + 20)
        XCTAssertEqual(try library.loadDocument(id: id), backing.document)
        canvas.connect(source: "input", sourcePort: "out", target: "sequence", targetPort: "in")
        XCTAssertTrue(canvas.graph.edges.contains { $0.source == "input" && $0.target == "sequence" })
        authority.allowRoot = false
        XCTAssertThrowsError(try ModelCanvasAdapter(owner: authority))
    }
}

/// Same existing owner, borrowed through its protocol rather than its concrete type.
@MainActor private final class BorrowedRootAuthority: ModelCanvasAuthority {
    let backing: ModelAuthoringStore
    var allowRoot = true
    init(_ backing: ModelAuthoringStore) { self.backing = backing }
    var projectID: String { backing.projectID }
    var sessionID: UUID { backing.sessionID }
    var isBusy: Bool { backing.isBusy }
    var error: String? { backing.error }
    var allowsRootCanvas: Bool { allowRoot }
    var canvasDocument: ModelProjectDocument { backing.document }
    var canvasChanges: AnyPublisher<Void, Never> { backing.canvasChanges }
    var canvasDocuments: AnyPublisher<ModelProjectDocument, Never> { backing.canvasDocuments }
    func modelCanvasRegistry() throws -> ComponentRegistry { try backing.modelCanvasRegistry() }
    func lease(path: [String]) -> ModelEditorLease { backing.lease(path: path) }
    func beginDraft(path: [String], lease: ModelEditorLease) -> Bool { backing.beginDraft(path: path, lease: lease) }
    func updateDraft(_ proposed: CompositeDefinition, lease: ModelEditorLease) -> Bool { backing.updateDraft(proposed, lease: lease) }
    func editDraftScope(_ mutation: (inout CompositeGraph) -> Void, lease: ModelEditorLease) -> Bool { backing.editDraftScope(mutation, lease: lease) }
    func applyDraftCommand(_ command: GraphCommand, selection: GraphSelection, lease: ModelEditorLease) -> GraphCommandResult? { backing.applyDraftCommand(command, selection: selection, lease: lease) }
    func applyDraft(lease: ModelEditorLease) -> Bool { backing.applyDraft(lease: lease) }
    func discardDraft(lease: ModelEditorLease) -> Bool { backing.discardDraft(lease: lease) }
    func applyRootCommand(_ command: GraphCommand, selection: GraphSelection, lease: ModelEditorLease) -> GraphCommandResult? { backing.applyRootCommand(command, selection: selection, lease: lease) }
    func editRootScope(_ mutation: (inout CompositeGraph) -> Void, lease: ModelEditorLease) -> Bool { backing.editRootScope(mutation, lease: lease) }
}
