import Foundation
import Combine
import KwipMLCore

/// One qualified root's borrowed child authority. There is no model store, mutable
/// model copy, independent history, or save destination behind this adapter.
@MainActor final class WorkflowBlockCanvasAuthority: ObservableObject, ModelCanvasAuthority {
    let owner: WorkflowAuthoringStore
    let root: WorkflowNodeAddress
    private var observation: AnyCancellable?
    @Published private var localError: String?
    var projectID: String { owner.projectID }
    var sessionID: UUID { owner.sessionID }
    var isBusy: Bool { owner.isBusy || !owner.isCurrent(owner.lease()) }
    var error: String? { localError ?? owner.error }
    var allowsRootCanvas: Bool { false }
    var canvasChanges: AnyPublisher<Void, Never> { objectWillChange.eraseToAnyPublisher() }
    var canvasDocument: ModelProjectDocument { Self.presentation(owner.document) }
    var canvasDocuments: AnyPublisher<ModelProjectDocument, Never> {
        owner.$document.map { Self.presentation($0) }.eraseToAnyPublisher()
    }
    func modelCanvasRegistry() throws -> ComponentRegistry { owner.modelRegistry }

    init(owner: WorkflowAuthoringStore, root: WorkflowNodeAddress) throws {
        guard owner.isCurrent(owner.lease()), root.domain == .model,
              let node = owner.document.nodes.first(where: { $0.address == root }),
              case .modelBlock(let reference, _) = node.payload,
              owner.document.definitions.contains(where: { $0.reference == reference }) else {
            throw CompositeContractError(.missingDefinition, "Open an existing model-domain Block with an available pinned definition.")
        }
        self.owner = owner; self.root = root
        observation = owner.objectWillChange.sink { [weak self] _ in self?.objectWillChange.send() }
    }
    func lease(path: [String]) -> ModelEditorLease {
        let token = owner.lease()
        return .init(projectID: token.projectID, sessionID: token.sessionID,
                     editRevision: token.generation, instancePath: path)
    }
    @discardableResult func beginDraft(path: [String], lease token: ModelEditorLease) -> Bool {
        perform {
            let ownerToken = try checkedOwner(token)
            guard token.instancePath.isEmpty else { throw failure("Opening a child requires a current parent lease.") }
            try checkPath(path)
            return owner.beginDraft(root: root, nestedPath: Array(path.dropFirst()), lease: ownerToken)
        }
    }
    @discardableResult func updateDraft(_ proposed: CompositeDefinition, lease token: ModelEditorLease) -> Bool {
        perform { owner.updateDraft(proposed, lease: try checkedBlock(token)) }
    }
    @discardableResult func discardDraft(lease token: ModelEditorLease) -> Bool {
        perform { owner.discardDraft(lease: try checkedBlock(token)) }
    }
    @discardableResult func applyDraft(lease token: ModelEditorLease) -> Bool {
        perform { owner.applyDraft(lease: try checkedBlock(token)) }
    }
    @discardableResult func editDraftScope(_ mutation: (inout CompositeGraph) -> Void, lease token: ModelEditorLease) -> Bool {
        perform {
            let lease = try checkedBlock(token)
            var proposed = try proposal(path: token.instancePath)
            let before = proposed.body
            mutation(&proposed.body)
            guard before.configurationVersion == proposed.body.configurationVersion,
                  before.instances.count == proposed.body.instances.count,
                  Set(proposed.body.instances.map(\.id)).count == proposed.body.instances.count,
                  before.instances.allSatisfy({ old in proposed.body.instances.contains { $0.id == old.id && $0.definition == old.definition } }) else {
                throw failure("Use graph commands for structural Block edits; field edits must preserve schema and pinned instance identities.")
            }
            return owner.updateDraft(proposed, lease: lease)
        }
    }
    @discardableResult func applyDraftCommand(_ command: GraphCommand, selection: GraphSelection,
                                              lease token: ModelEditorLease) -> GraphCommandResult? {
        do {
            let lease = try checkedBlock(token)
            var proposed = try proposal(path: token.instancePath)
            let result = try ModelScopeEditing.apply(command, to: proposed.body,
                definitions: owner.document.definitions, name: proposed.title,
                selection: selection, registry: owner.modelRegistry)
            proposed.body = result.scope
            guard owner.updateDraft(proposed, lease: lease) else { localError = nil; return nil }
            localError = nil
            return result.command
        } catch { localError = error.localizedDescription; return nil }
    }
    private func checkedOwner(_ token: ModelEditorLease) throws -> WorkflowEditorLease {
        // Preserve the supplied generation; substituting owner.lease() here would revive stale callbacks.
        let lease = WorkflowEditorLease(projectID: token.projectID, sessionID: token.sessionID, generation: token.editRevision)
        guard owner.isCurrent(lease), !owner.isBusy else { throw failure("This Block callback belongs to an earlier workflow state, or execution is active. Reopen the editor before applying it.") }
        return lease
    }
    private func checkedBlock(_ token: ModelEditorLease) throws -> WorkflowBlockLease {
        let ownerToken = try checkedOwner(token)
        try checkPath(token.instancePath)
        _ = try proposal(path: token.instancePath)
        return .init(owner: ownerToken, root: root, nestedPath: Array(token.instancePath.dropFirst()))
    }
    private func checkPath(_ path: [String]) throws {
        guard (1...8).contains(path.count), path.first == root.localID else {
            throw failure("This child authority belongs to a different qualified root Block. Resolve the current draft before opening another root.")
        }
    }
    private func proposal(path: [String]) throws -> CompositeDefinition {
        guard let draft = owner.document.drafts.first, draft.projectID == owner.projectID,
              draft.baseRevision == owner.document.revision, draft.instancePath == path else {
            throw failure("This child editor does not own the current persisted workflow draft.")
        }
        return draft.proposed
    }
    private func perform(_ action: () throws -> Bool) -> Bool {
        do { let result = try action(); localError = nil; return result }
        catch { localError = error.localizedDescription; return false }
    }
    private func failure(_ message: String) -> CompositeContractError { .init(.immutableRevision, message) }

    private static func presentation(_ document: WorkflowDocument) -> ModelProjectDocument {
        // Only the Block-root lookup needed by breadcrumbs/path traversal is projected.
        // Mixed data/control nodes and primitive versions remain exclusively in the owner.
        let instances = document.nodes.compactMap { node -> CompositeInstance? in
            guard node.address.domain == .model, case .modelBlock(let reference, let overrides) = node.payload else { return nil }
            return .init(id: node.address.localID, definition: reference, name: node.name,
                         x: node.x, y: node.y, overrides: overrides)
        }
        let source = CompositeProjectDocument(schemaVersion: document.schemaVersion >= 11 ? 3 : 2,
            name: document.name, root: .init(instances: instances), definitions: document.definitions)
        var drafts = document.drafts
        for index in drafts.indices where drafts[index].migrationOrigin != nil {
            drafts[index].migrationOrigin = .model(modelSchemaVersion: 1, sourceSchemaVersion: 2)
        }
        return .blockModel(.init(projectID: document.projectID, revision: document.revision, source: source,
            drafts: drafts, schemaVersion: document.schemaVersion >= 11 ? 2 : 1))
    }
}
