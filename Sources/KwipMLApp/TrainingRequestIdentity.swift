import Foundation

/// Captured request identity is separate from the last verified completed checkpoint.
struct TrainingRequestIdentity: Equatable {
    let runID: String
    let mode: String
    let parentRunID: String?
    init(runID: String, continuation: CheckpointContinuation?) {
        self.runID = runID
        self.mode = continuation.map { $0.mode == .exactResume ? "exactResume" : "weightsOnlyFineTune" } ?? "newTraining"
        self.parentRunID = continuation?.parent.runID
    }
    var description: String { "Current request: \(mode)" + (parentRunID.map { " · parent \($0.prefix(8))" } ?? "") }
}

extension WorkspaceStore {
    func refreshVerifiedCheckpointParent() {
        guard lastCompleted, let directory = outputDirectory else { return }
        do { verifiedCheckpointParent = try checkpointParent(at: directory); checkpointPresentationError = nil }
        catch { checkpointPresentationError = "Completed run cannot be used as a checkpoint parent: \(error.localizedDescription)" }
    }
}
