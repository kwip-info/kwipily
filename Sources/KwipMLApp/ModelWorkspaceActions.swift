import Foundation
import Combine
import KwipMLCore

extension WorkspaceStore {
    var modelSourceName: String {
        if let host = workflowSession { return host.owner.document.name }
        switch modelAuthoring?.document {
        case .legacyFlat(let graph): return graph.name
        case .blockModel(let model): return model.source.name
        case nil: return legacyGraph.name
        }
    }
    var modelSourceExecution: ExecutionSettings? {
        switch modelAuthoring?.document {
        case .legacyFlat(let graph): return graph.execution
        case .blockModel(let model): return model.source.execution
        case nil: return legacyGraph.execution
        }
    }
    var modelExecutionError: String? {
        guard let owner = modelAuthoring else { return nil }
        do { _ = try owner.executionSnapshot(); return nil }
        catch { return error.localizedDescription }
    }
    func synchronizeModelBusy() { modelAuthoring?.setExecutionBusy(isBusy) }
    func retireModelOwner() {
        modelChildWindows?.invalidate(); modelChildWindows = nil
        modelCanvas?.invalidate(); modelCanvas = nil
        modelOwnerObservation = nil
        modelAuthoring?.setExecutionBusy(true)
        modelAuthoring = nil
    }
    @discardableResult func beginOwnedProject(id: String) -> Bool {
        guard !isBusy else { return false }
        if modelAuthoring?.projectID == id { return true }
        do {
            guard !learningProjects.hasLessonMarker(projectID: id) else {
                throw CompositeContractError(.invalidBoundary, "Open this learning project through its lesson controller.")
            }
            if hasUnsavedChanges && !saveManagedProject() { return false }
            let owner = try ModelAuthoringStore(library: projectLibrary, projectID: id)
            let adapter = try ModelCanvasAdapter(owner: owner, presentation: modelCanvasViewState.presentation)
            let children = ModelChildWindowController(session: ModelChildEditorSession(owner: owner), appearance: appearanceState)
            guard flushWorkflowCanvasBeforeTransition(), flushChildCanvasBeforeTransition(), learningProjects.beforeProjectTransition() else { return false }
            persistWorkspaceView()
            retireWorkflowSession()
            retireModelOwner()
            modelAuthoring = owner; modelCanvas = adapter; modelChildWindows = children
            children.onDraftResolved = { [weak self, weak owner, weak children] in
                guard let self, let owner, let children, self.modelAuthoring === owner,
                      self.modelChildWindows === children, self.activeProjectID == owner.projectID else { return }
                self.focusOwningProject?()
            }
            modelOwnerObservation = owner.objectWillChange.sink { [weak self] _ in self?.objectWillChange.send() }
            adapter.onOpenBlock = { [weak self] path in
                guard let self else { return }
                if self.modelChildWindows?.open(path: path) != true { self.statusMessage = self.modelChildWindows?.session.error ?? "Block could not be opened." }
            }
            adapter.onCanvasViewChange = { [weak self, weak adapter] state in
                guard let self, self.modelCanvas === adapter else { return }
                self.modelCanvasViewState = state
            }
            // This empty legacy value is never an authoritative copy of an expanded model.
            installProjectSession(GraphDocument(name: modelSourceName), id: id, owned: true)
            adapter.canvasViewState = modelCanvasViewState
            children.session.presentation = modelCanvasViewState.presentation
            bindChildCanvasPersistence(children.session)
            synchronizeModelBusy()
            return true
        } catch { statusMessage = "Open failed: \(error.localizedDescription)"; return false }
    }
    func reportModelEdit(_ accepted: Bool) {
        if accepted { refreshProjects(); persistWorkspaceView(); statusMessage = "Saved locally · \(modelSourceName)"; saveError = nil }
        else { statusMessage = modelAuthoring?.error ?? "The model change was not applied." }
    }
    func exportModelProject(to url: URL) throws {
        if let owner = workflowSession?.owner {
            try ProjectSourceCodec.encode(.workflow(owner.document)).write(to: url, options: .atomic)
            return
        }
        let document: ModelProjectDocument
        if let owner = modelAuthoring { document = owner.document }
        else { document = .legacyFlat(try GraphMigration.toCurrent(legacyGraph, registry: registry)) }
        try ModelProjectCodec.encode(document).write(to: url, options: .atomic)
    }
}
