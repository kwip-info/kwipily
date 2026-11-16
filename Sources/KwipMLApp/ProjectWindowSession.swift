import SwiftUI
import Combine

/// A root file owns its editing, learning, output, and preparation state for its whole window lifetime.
@MainActor final class ProjectWindowSession: ObservableObject {
    let workspace: WorkspaceStore
    let affine: AffineLearningStore
    let normalization: NormalizationLearningStore
    let tensor: TensorLearningStore
    let preparation = PreparationSession()
    private var observation: AnyCancellable?
    init(workspace: WorkspaceStore) {
        self.workspace = workspace
        affine = AffineLearningStore(workspace: workspace)
        normalization = NormalizationLearningStore(workspace: workspace)
        tensor = TensorLearningStore(workspace: workspace)
        observation = workspace.objectWillChange.sink { [weak self] _ in self?.objectWillChange.send() }
    }
    @discardableResult func save() -> Bool {
        guard workspace.permitsManagedProjectSave() else { return false }
        if tensor.isActive { return tensor.save() }
        if normalization.isActive { return normalization.save() }
        if affine.isActive { return affine.save() }
        return workspace.saveManagedProject()
    }
    func prepareToClose() -> Bool {
        guard workspace.permitsManagedProjectSave() else { return false }
        guard !workspace.isBusy else {
            workspace.statusMessage = "Stop the active operation before closing this project."
            return false
        }
        guard workspace.flushWorkflowCanvasBeforeTransition(), workspace.flushChildCanvasBeforeTransition(),
              workspace.learningProjects.flushForProjectTransition() else { return false }
        if workspace.hasUnsavedChanges && !save() { return false }
        workspace.persistWorkspaceView()
        return true
    }
    func retire() {
        workspace.learningProjects.retireProjectContext()
        workspace.retireWorkflowSession(); workspace.retireModelOwner()
        workspace.externalFolderLease = nil
    }
}
