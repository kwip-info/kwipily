import Foundation
import KwipMLCore

struct LearningProjectContext: Equatable {
    enum Kind: Equatable { case affine, normalization, tensor }
    enum RestoreState: Equatable { case ready, failed }
    let projectID:String
    let kind:Kind
    let restoreState:RestoreState
}

/// One synchronous owner of lesson project transitions and assessment cancellation.
/// Task controllers retain task-specific drafts, graph editing, execution and receipts.
@MainActor final class LearningProjectCoordinator {
    private weak var workspace:WorkspaceStore?
    private weak var affine:AffineLearningStore?
    private weak var normalization:NormalizationLearningStore?
    private weak var tensor:TensorLearningStore?
    private var assessmentToken:UUID?
    private var assessmentCancellation:(() -> Void)?
    let tensorDraftSession:TensorLearningDraftSession
    init(workspace:WorkspaceStore) {
        self.workspace=workspace
        tensorDraftSession=TensorLearningDraftSession(repository:TensorLearningDraftRepository(
            directory:workspace.workspaceDirectory.appendingPathComponent("learning/projects")))
    }
    @discardableResult func attachAffine(_ session:AffineLearningStore) -> Bool {
        guard affine !== session else { return true }
        guard assessmentToken == nil else { return false }
        affine=session
        return true
    }
    func owns(_ session:AffineLearningStore) -> Bool { affine === session }
    func activateAffine(projectID:String) {
        guard workspace?.activeProjectID == projectID else { return }
        normalization?.clearProjectContext()
        workspace?.updateLearningContext(.init(projectID:projectID,kind:.affine,restoreState:.ready))
    }
    @discardableResult func attachNormalization(_ session:NormalizationLearningStore) -> Bool {
        guard normalization !== session else {return true}
        guard assessmentToken == nil else {return false}
        normalization=session
        return true
    }
    func owns(_ session:NormalizationLearningStore) -> Bool {normalization === session}
    func activateNormalization(projectID:String) {
        guard workspace?.activeProjectID == projectID else {return}
        affine?.clearProjectContext()
        workspace?.updateLearningContext(.init(projectID:projectID,kind:.normalization,restoreState:.ready))
    }
    @discardableResult func attachTensor(_ session:TensorLearningStore)->Bool {
        guard tensor !== session else {return true}
        guard assessmentToken == nil, !tensorDraftSession.hasUnsavedChanges else {return false}
        tensor=session;return true
    }
    func owns(_ session:TensorLearningStore)->Bool {tensor === session}
    private func marker(_ id:String,kind:LearningProjectContext.Kind)->URL? {
        let filename:String
        switch kind {
        case .affine:filename="draft.json"
        case .normalization:filename="normalization.json"
        case .tensor:filename="tensor.json"
        }
        return workspace?.workspaceDirectory.appendingPathComponent("learning/projects/\(id)/\(filename)")
    }
    private func markerKinds(_ id:String)->[LearningProjectContext.Kind] {
        [.affine,.normalization,.tensor].filter {markerExists(marker(id,kind:$0))}
    }
    private func markerExists(_ url:URL?) -> Bool {
        guard let url else {return false}
        // Include directories and dangling symbolic links: malformed markers remain protected.
        return FileManager.default.fileExists(atPath:url.path) || (try? FileManager.default.attributesOfItem(atPath:url.path)) != nil
    }
    func hasLessonMarker(projectID:String)->Bool {
        !markerKinds(projectID).isEmpty
    }
    func beforeProjectTransition() -> Bool {
        guard flushForProjectTransition() else { return false }
        retireProjectContext()
        return true
    }
    /// Persistence preparation can be repeated or followed by another window vetoing quit.
    func flushForProjectTransition() -> Bool {
        guard let workspace, !workspace.isBusy else { return false }
        // Marker conflicts can change the displayed lesson kind while local edits remain.
        // Flush the owned session independently of that presentation state.
        guard tensorDraftSession.flushBeforeTransition() else {
            workspace.statusMessage=tensorDraftSession.errorMessage ?? "Tensor lesson save failed."
            return false
        }
        switch workspace.lessonContext?.kind {
        case .normalization:
            if let normalization,!normalization.persistForProjectTransition() {workspace.statusMessage=normalization.statusMessage;return false}
        case .affine:
            if let affine,!affine.persistForProjectTransition() {workspace.statusMessage=affine.statusMessage;return false}
        case .tensor, nil: break
        }
        return true
    }
    func retireProjectContext() {
        affine?.clearProjectContext();normalization?.clearProjectContext()
        tensor?.clearProjectContext()
        tensorDraftSession.discardLocalChanges()
        workspace?.updateLearningContext(nil)
    }
    func restoreCurrentProject() {
        guard let workspace,!workspace.isBusy else { return }
        guard let id=workspace.activeProjectID else {
            affine?.clearProjectContext();normalization?.clearProjectContext();workspace.updateLearningContext(nil);return
        }
        let kinds=markerKinds(id)
        if kinds.count > 1 {
            affine?.clearProjectContext();normalization?.clearProjectContext()
            workspace.updateLearningContext(.init(projectID:id,kind:.normalization,restoreState:.failed))
            workspace.activeSection="Learn"
            workspace.statusMessage="Multiple lesson markers exist for this project. Resolve the conflicting lesson files before reopening; ordinary execution is disabled."
            return
        }
        guard let kind=kinds.first else {
            // Preserve the active newly-created draft before its first atomic save.
            if affine?.draft?.projectID != id && normalization?.draft?.projectID != id {
                affine?.clearProjectContext();normalization?.clearProjectContext();workspace.updateLearningContext(nil)
            }
            return
        }
        // Protection precedes parsing: corrupt lesson files never become ordinary runs.
        workspace.updateLearningContext(.init(projectID:id,kind:kind,restoreState:.failed))
        if kind == .tensor {
            affine?.clearProjectContext();normalization?.clearProjectContext()
            workspace.activeSection="Learn"
            if tensorDraftSession.draft?.projectID != id && !tensorDraftSession.open(projectID:id) {
                workspace.statusMessage="Tensor lesson could not be restored: " + (tensorDraftSession.errorMessage ?? "Unknown storage error.")
                return
            }
            guard let tensor else {
                workspace.statusMessage="Tensor lesson editor is not available in this build. Saved lesson files are preserved; ordinary execution is disabled."
                return
            }
            workspace.updateLearningContext(.init(projectID:id,kind:.tensor,restoreState:.ready))
            tensor.restoreEditor()
            workspace.statusMessage=tensor.statusMessage
            return
        }
        if kind == .normalization {
            affine?.clearProjectContext()
            guard let normalization else {workspace.statusMessage="Normalization lesson controller unavailable; ordinary execution is disabled.";return}
            normalization.restoreProjectContext()
            if normalization.isRestoreFailure {workspace.statusMessage=normalization.statusMessage}
            workspace.updateLearningContext(.init(projectID:id,kind:kind,restoreState:normalization.isRestoreFailure || normalization.draft?.projectID != id ? .failed:.ready))
        } else {
            normalization?.clearProjectContext()
            guard let affine else {workspace.statusMessage="Lesson controller unavailable; ordinary execution is disabled.";return}
            affine.restoreProjectContext()
            if affine.isRestoreFailure {workspace.statusMessage=affine.statusMessage}
            workspace.updateLearningContext(.init(projectID:id,kind:kind,restoreState:affine.isRestoreFailure || affine.draft?.projectID != id ? .failed:.ready))
        }
    }
    func beginAssessment(projectID:String,cancel:@escaping () -> Void,onAdmissionFailure:((String)->Void)?=nil) -> UUID? {
        guard let workspace,!workspace.isBusy,workspace.activeProjectID == projectID,
              workspace.lessonContext?.projectID == projectID,workspace.lessonContext?.restoreState == .ready else { return nil }
        guard workspace.allowExecution() else {onAdmissionFailure?(workspace.statusMessage);return nil}
        let token=UUID();assessmentToken=token;assessmentCancellation=cancel
        workspace.setLearningAssessmentActive(true)
        return token
    }
    func finishAssessment(token:UUID) {
        guard token == assessmentToken else { return }
        assessmentToken=nil;assessmentCancellation=nil;workspace?.setLearningAssessmentActive(false)
    }
    func cancelAssessment() { assessmentCancellation?() }
}
