import Foundation
import Combine
import KwipMLCore

/// Child editing authority borrowed by the shared model-scope UI. A canvas document is
/// read-only presentation input; it must never be exported, executed, or saved by the UI.
@MainActor protocol ModelCanvasAuthority: AnyObject {
    var projectID: String { get }
    var sessionID: UUID { get }
    var isBusy: Bool { get }
    var error: String? { get }
    var allowsRootCanvas: Bool { get }
    var isFileBacked: Bool { get }
    var editStatus: String { get }
    func applyRootCommand(_ command: GraphCommand, selection: GraphSelection, lease: ModelEditorLease) -> GraphCommandResult?
    func editRootScope(_ mutation: (inout CompositeGraph) -> Void, lease: ModelEditorLease) -> Bool
    func scopeCanUndo(path: [String]) -> Bool
    func scopeCanRedo(path: [String]) -> Bool
    func scopeUndo(lease: ModelEditorLease) -> Bool
    func scopeRedo(lease: ModelEditorLease) -> Bool
    var canvasDocument: ModelProjectDocument { get }
    var canvasChanges: AnyPublisher<Void, Never> { get }
    var canvasDocuments: AnyPublisher<ModelProjectDocument, Never> { get }
    func modelCanvasRegistry() throws -> ComponentRegistry
    func lease(path: [String]) -> ModelEditorLease
    func beginDraft(path: [String], lease: ModelEditorLease) -> Bool
    func updateDraft(_ proposed: CompositeDefinition, lease: ModelEditorLease) -> Bool
    func editDraftScope(_ mutation: (inout CompositeGraph) -> Void, lease: ModelEditorLease) -> Bool
    func applyDraftCommand(_ command: GraphCommand, selection: GraphSelection, lease: ModelEditorLease) -> GraphCommandResult?
    func applyDraft(lease: ModelEditorLease) -> Bool
    func discardDraft(lease: ModelEditorLease) -> Bool
}

extension ModelCanvasAuthority {
    func applyRootCommand(_ command: GraphCommand, selection: GraphSelection, lease: ModelEditorLease) -> GraphCommandResult? { nil }
    func editRootScope(_ mutation: (inout CompositeGraph) -> Void, lease: ModelEditorLease) -> Bool { false }
    var isFileBacked: Bool { false }
    var editStatus: String { "Saved with parent project." }
    func scopeCanUndo(path: [String]) -> Bool { false }
    func scopeCanRedo(path: [String]) -> Bool { false }
    func scopeUndo(lease: ModelEditorLease) -> Bool { false }
    func scopeRedo(lease: ModelEditorLease) -> Bool { false }
    func lease() -> ModelEditorLease { lease(path: []) }
}

extension ModelAuthoringStore: ModelCanvasAuthority {
    var allowsRootCanvas: Bool { true }
    func scopeCanUndo(path: [String]) -> Bool { path.isEmpty && canUndo }
    func scopeCanRedo(path: [String]) -> Bool { path.isEmpty && canRedo }
    func scopeUndo(lease: ModelEditorLease) -> Bool { undo(lease: lease) }
    func scopeRedo(lease: ModelEditorLease) -> Bool { redo(lease: lease) }
    var canvasDocument: ModelProjectDocument { document }
    var canvasChanges: AnyPublisher<Void, Never> { objectWillChange.eraseToAnyPublisher() }
    var canvasDocuments: AnyPublisher<ModelProjectDocument, Never> { $document.eraseToAnyPublisher() }
    func modelCanvasRegistry() throws -> ComponentRegistry { try .builtIn() }
}
