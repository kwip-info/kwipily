import Foundation
import Combine
import KwipMLCore

/// Presentation lifetime for one canonical workflow's selected-node inspector.
/// Contexts retain their editing identity across their own accepted transactions;
/// external changes retire them and are reconciled after the owner's publication.
@MainActor final class WorkflowInspectorSession: ObservableObject {
    @MainActor enum Selection {
        case data(WorkflowDataNodeContext)
        case model(WorkflowModelNodeContext)
        case control(WorkflowControlNodeContext)

        var address: WorkflowNodeAddress {
            switch self {
            case .data(let context): return context.address
            case .model(let context): return context.authority.address
            case .control(let context): return context.authority.address
            }
        }
        var isCurrent: Bool {
            switch self {
            case .data(let context): return context.isCurrent
            case .model(let context): return context.isCurrent
            case .control(let context): return context.isCurrent
            }
        }
        var identity: String {
            switch self {
            case .data(let context): return context.contextIdentity
            case .model(let context): return context.contextIdentity
            case .control(let context): return context.contextIdentity
            }
        }
        fileprivate var changes: AnyPublisher<Void, Never> {
            switch self {
            case .data(let context): return context.objectWillChange.eraseToAnyPublisher()
            case .model(let context): return context.objectWillChange.eraseToAnyPublisher()
            case .control(let context): return context.objectWillChange.eraseToAnyPublisher()
            }
        }
        func retire() {
            switch self {
            case .data(let context): context.retire()
            case .model(let context): context.retire()
            case .control(let context): context.retire()
            }
        }
    }

    let owner: WorkflowAuthoringStore
    private let sources: any WorkflowDataSources
    private let outputs: (any WorkflowDataOutputs)?
    private let onEditPrivateBlock: ((WorkflowNodeAddress, WorkflowEditorLease) -> Void)?
    private let onOpenBlock: ((WorkflowNodeAddress, WorkflowEditorLease) -> Void)?
    @Published private(set) var selection: Selection?
    @Published private(set) var error: String?
    @Published private(set) var isVisible = false
    private var appearance: UUID?
    private var queuedReconciliation: UUID?
    private var isReconciling = false
    private var ownerObservation: AnyCancellable?
    private var contextObservation: AnyCancellable?

    init(owner: WorkflowAuthoringStore, sources: any WorkflowDataSources,
         outputs: (any WorkflowDataOutputs)? = nil,
         onOpenBlock: ((WorkflowNodeAddress, WorkflowEditorLease) -> Void)? = nil,
         onEditPrivateBlock: ((WorkflowNodeAddress, WorkflowEditorLease) -> Void)? = nil) {
        self.owner = owner
        self.sources = sources
        self.outputs = outputs
        self.onOpenBlock = onOpenBlock
        self.onEditPrivateBlock = onEditPrivateBlock
        ownerObservation = owner.objectWillChange.sink { [weak self] _ in
            self?.scheduleReconciliation()
        }
    }

    /// A replacement appearance receives an independent token even when SwiftUI
    /// has not yet cancelled the outgoing appearance's task.
    @discardableResult func activate() -> UUID {
        let token = UUID()
        appearance = token
        queuedReconciliation = nil
        if !isVisible { isVisible = true }
        reconcile()
        return token
    }

    func deactivate(appearance token: UUID) {
        guard appearance == token else { return }
        appearance = nil
        queuedReconciliation = nil
        isVisible = false
        clearSelection()
        error = nil
    }

    /// Used directly by the view's `.task(id: showInspector)`. The cancellation
    /// handler wakes a suspended stream, not a polling timer. The local token is
    /// never read from mutable view State by an obsolete onDisappear callback.
    func runAppearanceLifetime() async {
        guard !Task.isCancelled else { return }
        let token = activate()
        defer { deactivate(appearance: token) }
        let (stream, continuation) = AsyncStream<Void>.makeStream()
        await withTaskCancellationHandler {
            for await _ in stream { }
        } onCancel: {
            continuation.finish()
        }
    }

    func reconcile() {
        guard isVisible, appearance != nil, !isReconciling else { return }
        isReconciling = true
        defer { isReconciling = false }
        guard owner.isCurrent(owner.lease()) else {
            clearSelection()
            error = "This workflow editor is no longer active. Reopen the workspace."
            return
        }
        guard sources.projectID == owner.projectID, sources.sessionID == owner.sessionID,
              outputs == nil || (outputs?.projectID == owner.projectID && outputs?.sessionID == owner.sessionID) else {
            clearSelection()
            error = "The inspector and source service do not belong to this workflow session. Reopen the workspace."
            return
        }
        guard owner.selection.nodeAddresses.count == 1, owner.selection.edgeIDs.isEmpty,
              let address = owner.selection.nodeAddresses.first else {
            clearSelection()
            error = nil
            return
        }
        if let selection, selection.address == address, selection.isCurrent {
            if error != nil { error = nil }
            return
        }
        clearSelection()
        do {
            let next: Selection
            switch address.domain {
            case .data:
                next = .data(try WorkflowDataNodeContext(owner: owner, address: address, sources: sources, outputs: outputs))
            case .model where owner.document.nodes.first(where:{$0.address==address}).map(WorkflowModelBoundary.isDefinition)==true:
                next = .control(try WorkflowControlNodeContext(owner:owner,address:address))
            case .model:
                next = .model(try WorkflowModelNodeContext(owner: owner, address: address, onOpenBlock: onOpenBlock, onEditPrivateBlock: onEditPrivateBlock))
            case .control:
                next = .control(try WorkflowControlNodeContext(owner: owner, address: address))
            }
            selection = next
            error = nil
            contextObservation = next.changes.sink { [weak self] _ in
                self?.scheduleReconciliation()
            }
        } catch {
            self.error = error.localizedDescription
        }
    }

    private func clearSelection() {
        // Unsubscribe before retiring: retirement itself publishes and clears
        // owner-owned pending tokens. Neither event should renew a hidden editor.
        contextObservation = nil
        let previous = selection
        if previous != nil { selection = nil }
        previous?.retire()
    }

    private func scheduleReconciliation() {
        guard isVisible, let appearance, !isReconciling, queuedReconciliation == nil else { return }
        let request = UUID()
        queuedReconciliation = request
        Task { @MainActor [weak self] in
            // objectWillChange/@Published precede assignment. Own transactions
            // also renew their editing lease after publishing the document.
            await Task.yield()
            guard let self, self.isVisible, self.appearance == appearance,
                  self.queuedReconciliation == request else { return }
            self.queuedReconciliation = nil
            self.reconcile()
        }
    }
}
