import Foundation
import KwipMLCore

extension WorkspaceStore {
    func bindChildCanvasPersistence(_ session: ModelChildEditorSession,isCurrent: (() -> Bool)? = nil) {
        // Capture the actual authority, not merely its project UUID. A replacement
        // owner for the same project cannot inherit an old child callback.
        let authority=session.owner
        session.configurePresentation = { [weak self, weak session] editor in
            guard let self, let session, self.ownsChildCanvasAuthority(authority), session.owner === authority, isCurrent?() != false,
                  self.activeProjectID == session.owner.projectID, session.editor === editor,
                  let key = try? ModelChildCanvasKey.make(path: editor.path) else { return }
            editor.restoreCanvasViewState(self.blockCanvasStates[key] ?? CanvasViewState(presentation: self.appPreferences.canvasDefaults))
            let revision = self.canvasViewRevision
            editor.onCanvasViewChange = { [weak self, weak session, weak editor] state in
                guard let self, let session, let editor, session.editor === editor,
                      self.ownsChildCanvasAuthority(authority), session.owner === authority, isCurrent?() != false, self.activeProjectID == session.owner.projectID,
                      self.canvasViewRevision == revision, !self.isRestoringArrangement else { return }
                do { try self.recordChildCanvasState(state, key: key) }
                catch { self.arrangementError = error.localizedDescription }
            }
        }
        session.persistPresentation = { [weak self, weak session] in
            guard let self, let session, self.ownsChildCanvasAuthority(authority), session.owner === authority, isCurrent?() != false,
                  self.activeProjectID == session.owner.projectID else {
                throw WorkspaceArrangementError("This Block editor belongs to a previous project session.")
            }
            do {
                if let editor = session.editor {
                    let key = try ModelChildCanvasKey.make(path: editor.path)
                    try self.recordChildCanvasState(editor.capturedCanvasViewState(), key: key)
                }
                self.arrangementSaveTask?.cancel(); self.arrangementSaveTask = nil
                try self.arrangementLibrary.saveProjectState(self.captureWorkspaceView(), projectID: session.owner.projectID)
                self.arrangementError = nil
            } catch { self.arrangementError = error.localizedDescription; throw error }
        }
        session.restorePresentation()
    }

    func ownsChildCanvasAuthority(_ authority:any ModelCanvasAuthority)->Bool {
        guard activeProjectID == authority.projectID else {return false}
        if let borrowed=authority as? WorkflowBlockCanvasAuthority {
            guard let host=workflowSession else {return false}
            return !host.isRetired && host.owner === borrowed.owner && host.owner.isCurrent(host.owner.lease())
        }
        return modelAuthoring === authority
    }

    private func recordChildCanvasState(_ state: CanvasViewState, key: String) throws {
        var proposed = captureWorkspaceView()
        proposed.blockCanvasStates[key] = state
        try proposed.validate()
        if blockCanvasStates[key] != state { blockCanvasStates[key] = state }
    }

    @discardableResult func flushChildCanvasBeforeTransition() -> Bool {
        guard workflowChildren?.flushPresentation() != false else {
            statusMessage = workflowChildren?.error ?? "Block layout could not be saved. The current project remains open."
            return false
        }
        guard modelChildWindows?.session.flushPresentation() != false else {
            statusMessage = modelChildWindows?.session.error ?? "Block layout could not be saved. The current project remains open."
            return false
        }
        return true
    }
}
