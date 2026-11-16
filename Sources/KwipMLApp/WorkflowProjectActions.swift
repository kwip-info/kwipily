import Foundation
import Combine
import KwipMLCore

extension WorkspaceStore {
    func retireWorkflowSession() {
        workflowHistory?.invalidate(); workflowHistory = nil
        workflowChildren?.invalidate(); workflowChildren = nil
        workflowOwnerObservation = nil
        workflowCanvasObservation = nil
        workflowSession?.retire()
        workflowSession = nil
    }

    @discardableResult func beginWorkflowProject(id: String, forceReload: Bool = false) -> Bool {
        guard !isBusy else { return false }
        if workflowSession?.owner.projectID == id && !forceReload { return true }
        do {
            guard !learningProjects.hasLessonMarker(projectID: id) else {
                throw DataFlowContractError("Open this learning project through its lesson controller.")
            }
            if hasUnsavedChanges && !saveManagedProject() { return false }
            let operators = try workflowDataRegistry()
            let owner = try WorkflowAuthoringStore(library: projectLibrary, projectID: id, modelRegistry: registry, dataRegistry: operators)
            let migratedRetiredTools = try owner.migrateRetiredToolsForAuthoring()
            // Validate source and private bindings before retiring the current editor.
            let host = try WorkflowWorkspaceSession(workspace: self, owner: owner)
            guard flushWorkflowCanvasBeforeTransition(), flushChildCanvasBeforeTransition(), learningProjects.beforeProjectTransition() else { return false }
            persistWorkspaceView()
            retireWorkflowSession(); retireModelOwner()
            workflowSession = host
            // Observe the canonical owner only. The host's source provider already
            // observes workspace busy state; forwarding the host would recurse.
            workflowOwnerObservation = owner.objectWillChange.sink { [weak self] _ in self?.objectWillChange.send() }
            installProjectSession(GraphDocument(name: owner.document.name), id: id, owned: true)
            workflowHistory = WorkflowHistoryStore(workspace: self, host: host)
            activeSection = "Build"
            workflowCanvasObservation = installWorkflowCanvasPersistence(host)
            let children = WorkflowChildWorkspace(workspace: self, host: host)
            workflowChildren = children
            host.canvas.observeChildRequests(children)
            host.canvas.onOpenBlock = { [weak self, weak host, weak children] address, lease in
                guard let self, let host, let children, self.workflowSession === host else { return }
                _ = children.open(address: address, lease: lease)
            }
            host.canvas.onEditPrivateBlock = { [weak self, weak host, weak children] address, lease in
                guard let self, let host, let children, self.workflowSession === host else { return }
                _ = children.openPrivateCopy(address: address, lease: lease)
            }
            host.reloadGenerationRecords()
            statusMessage = "Opened workflow · \(owner.document.name)" + (migratedRetiredTools ? " · Retired tools upgraded; recovery copy saved." : "") + (owner.retiredToolRepairMessage.map { " · " + $0 } ?? "")
            return true
        } catch { statusMessage = "Open failed: \(error.localizedDescription)"; return false }
    }

    func workflowDataRegistry() throws -> DataFlowOperatorRegistry {
        let directory: URL
        if let runtime {
            directory = runtime.kind == .bundled ? runtime.workingDirectoryURL.appendingPathComponent("runner") : runtime.entryURL.deletingLastPathComponent()
        } else { directory = projectRoot.appendingPathComponent("runner") }
        return try DataFlowOperatorRegistry.load(from: directory.appendingPathComponent("dataflow-operators.json"))
    }

    func reportWorkflowEdit(_ accepted: Bool) {
        if accepted {
            refreshProjects(); persistWorkspaceView()
            statusMessage = "Saved locally · \(modelSourceName)"; saveError = nil
        } else { statusMessage = workflowSession?.owner.error ?? "The workflow change was not applied." }
    }
}
