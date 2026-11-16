import Foundation
import Combine
import KwipMLCore

/// Projects a standalone block file into the shared block editor without borrowing a parent's authority.
@MainActor final class FlowBlockCanvasAuthority: ModelCanvasAuthority {
    let documentOwner: FlowDocumentOwner
    static let canvasID = "document"
    var path: [String] { [Self.canvasID] }
    private let dependencies: [CompositeDefinition]
    private var observation: AnyCancellable?
    private var generation: UInt64 = 0
    private var localError: String?
    var projectID: String { documentOwner.document.id.uuidString }
    var sessionID: UUID { documentOwner.sessionID }
    var isBusy: Bool { false }
    var error: String? { localError ?? documentOwner.error }
    var allowsRootCanvas: Bool { false }
    var isFileBacked: Bool { true }
    var editStatus: String { documentOwner.isDirty ? "Unsaved changes in this subflow document." : "Subflow document saved. Callers retain their pinned revision." }

    init(owner: FlowDocumentOwner) throws {
        guard case .modelBlock = owner.document.payload else { throw FlowDocumentError.invalidDocument }
        documentOwner = owner
        let snapshot = try FlowDocumentStore.snapshot(at: owner.url)
        dependencies = snapshot.documents.filter { $0.id != owner.document.id }.compactMap {
            if case .modelBlock(let definition) = $0.payload { return definition }; return nil
        }
        observation = owner.objectWillChange.sink { [weak self] _ in self?.generation &+= 1 }
    }
    var canvasDocument: ModelProjectDocument { projection(documentOwner.document) }
    var canvasChanges: AnyPublisher<Void, Never> { documentOwner.objectWillChange.eraseToAnyPublisher() }
    var canvasDocuments: AnyPublisher<ModelProjectDocument, Never> {
        documentOwner.$document.map { [unowned self] in self.projection($0) }.eraseToAnyPublisher()
    }
    private func projection(_ document: FlowDocument) -> ModelProjectDocument {
        guard case .modelBlock(let block) = document.payload else { preconditionFailure("Block owner changed document kind") }
        let root = CompositeGraph(instances: [.init(id: Self.canvasID, definition: block.reference, name: block.title)])
        return .blockModel(.init(projectID: document.id.uuidString, revision: document.revision,
            source: .init(name: block.title, root: root, definitions: [block] + dependencies),
            drafts: [.init(projectID: document.id.uuidString, instancePath: path, baseReference: block.reference,
                           baseRevision: document.revision, proposed: block)]))
    }
    func modelCanvasRegistry() throws -> ComponentRegistry { try .builtIn() }
    func lease(path: [String]) -> ModelEditorLease {
        .init(projectID: projectID, sessionID: sessionID, editRevision: generation, instancePath: path)
    }
    private func check(_ token: ModelEditorLease) -> Bool {
        guard token == lease(path: path) else { localError = "This command belongs to an older document state."; return false }
        localError = nil; return true
    }
    func beginDraft(path: [String], lease: ModelEditorLease) -> Bool { path == self.path && check(lease) }
    func updateDraft(_ proposed: CompositeDefinition, lease: ModelEditorLease) -> Bool {
        guard check(lease), case .modelBlock(let current) = documentOwner.document.payload,
              proposed.reference == current.reference else { return false }
        return documentOwner.edit { $0.payload = .modelBlock(proposed) }
    }
    func editDraftScope(_ mutation: (inout CompositeGraph) -> Void, lease: ModelEditorLease) -> Bool {
        guard check(lease), case .modelBlock(var block) = documentOwner.document.payload else { return false }
        mutation(&block.body)
        return updateDraft(block, lease: lease)
    }
    func applyDraftCommand(_ command: GraphCommand, selection: GraphSelection, lease: ModelEditorLease) -> GraphCommandResult? {
        guard check(lease), case .modelBlock(var block) = documentOwner.document.payload else { return nil }
        do {
            let result = try ModelScopeEditing.apply(command, to: block.body, definitions: [block] + dependencies,
                                                     name: block.title, selection: selection, registry: modelCanvasRegistry())
            block.body = result.scope
            return updateDraft(block, lease: lease) ? result.command : nil
        } catch { localError = error.localizedDescription; return nil }
    }
    func applyDraft(lease: ModelEditorLease) -> Bool { check(lease) && documentOwner.save() }
    func discardDraft(lease: ModelEditorLease) -> Bool {
        guard check(lease) else { return false }; return documentOwner.revert()
    }
    func scopeCanUndo(path: [String]) -> Bool { path == self.path && documentOwner.canUndo }
    func scopeCanRedo(path: [String]) -> Bool { path == self.path && documentOwner.canRedo }
    func scopeUndo(lease: ModelEditorLease) -> Bool { check(lease) && documentOwner.undo() }
    func scopeRedo(lease: ModelEditorLease) -> Bool { check(lease) && documentOwner.redo() }
}
