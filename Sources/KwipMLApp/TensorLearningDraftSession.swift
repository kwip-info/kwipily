import Foundation
import Combine
import KwipMLCore

/// The editable source owner. Assessment evidence is deliberately kept outside this session.
/// A failed save retains local edits and vetoes navigation until retry or explicit discard.
@MainActor final class TensorLearningDraftSession:ObservableObject {
    @Published private(set) var identity=UUID()
    @Published private(set) var draft:TensorLearningDraft?
    @Published private(set) var hasUnsavedChanges=false
    @Published private(set) var errorMessage:String?
    private var revision:String?
    private let repository:TensorLearningDraftRepository
    init(repository:TensorLearningDraftRepository) {self.repository=repository}

    @discardableResult func create(projectID:String,operation:String,mode:LearningMode)->Bool {
        guard flushBeforeTransition() else {return false}
        let candidate=TensorLearningDraft(projectID:projectID,operation:operation,mode:mode)
        do {
            let saved=try repository.save(candidate,expectedRevision:nil)
            identity=UUID();draft=candidate;revision=saved;hasUnsavedChanges=false;errorMessage=nil
            return true
        } catch {errorMessage=error.localizedDescription;return false}
    }
    @discardableResult func open(projectID:String)->Bool {
        guard flushBeforeTransition() else {return false}
        // Never leave the preceding draft attached to a failed incoming project.
        identity=UUID();draft=nil;revision=nil;hasUnsavedChanges=false
        do {
            let saved=try repository.load(projectID:projectID)
            draft=saved.draft;revision=saved.revision;errorMessage=nil
            return true
        } catch {errorMessage=error.localizedDescription;return false}
    }
    @discardableResult func update(_ change:(inout TensorLearningDraft)->Void)->Bool {
        guard var candidate=draft else {return false}
        change(&candidate)
        do {
            guard candidate.projectID == draft?.projectID,candidate.operation == draft?.operation else {
                throw LearningContractError("Create a separate lesson to change its identity or operation.")
            }
            try candidate.validate()
            if candidate != draft {draft=candidate;hasUnsavedChanges=true}
            errorMessage=nil;return true
        } catch {errorMessage=error.localizedDescription;return false}
    }
    @discardableResult func save()->Bool {
        guard let draft else {return false}
        do {
            revision=try repository.save(draft,expectedRevision:revision)
            hasUnsavedChanges=false;errorMessage=nil;return true
        } catch {errorMessage=error.localizedDescription;return false}
    }
    @discardableResult func flushBeforeTransition()->Bool {
        !hasUnsavedChanges || save()
    }
    /// Explicit abandonment only; callers must not use this to bypass a failed transition.
    func discardLocalChanges() {
        identity=UUID();draft=nil;revision=nil;hasUnsavedChanges=false;errorMessage=nil
    }
}
