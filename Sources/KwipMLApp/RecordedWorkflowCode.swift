import Foundation
import KwipMLCore

/// Reads retained, verified code only. Never invokes Python or a live workflow generator.
enum RecordedWorkflowCode {
    struct Choice: Identifiable { let id: String; let label: String }
    struct Snapshot {
        let files: [String: String]
        let originLabel: String
        let inferenceOnly: Bool
    }
    enum Record {
        case training(WorkflowHistoryReference)
        case preparation(WorkflowDataPreparationHistory.Entry)
        case backend(BackendFitHistoryStore.Entry)
    }
    @MainActor static func choices(workspace: WorkspaceStore) -> [Choice] {
        guard let host = workspace.workflowSession else { return [] }
        let training = (workspace.workflowHistory?.listing.entries ?? []).filter { $0.verification == .recordedV2 }.map {
            Choice(id: "training:" + $0.reference.directoryID.uuidString,
                   label: "Training · " + ($0.finishedAt ?? "Recorded run"))
        }
        let preparation = host.dataPreparationHistory.entries.map {
            Choice(id: "preparation:" + $0.id, label: "Preparation · " + $0.completedAt.formatted(date: .abbreviated, time: .shortened))
        }
        let backend = host.backendHistory.entries.map {
            Choice(id: "backend:" + $0.id, label: $0.backendID + " · " + $0.completedAt.formatted(date: .abbreviated, time: .shortened))
        }
        return training + preparation + backend
    }
    @MainActor static func load(workspace: WorkspaceStore, selection: String) async throws -> Snapshot {
        guard let host = workspace.workflowSession else { throw unavailable() }
        let parts = selection.split(separator: ":", omittingEmptySubsequences: false)
        guard parts.count == 2, let id = UUID(uuidString: String(parts[1])), id.uuidString == parts[1] else { throw unavailable() }
        let record: Record
        switch parts[0] {
        case "training": record = .training(.init(projectID: host.owner.projectID, directoryID: id))
        case "preparation":
            guard let entry = host.dataPreparationHistory.entries.first(where: { $0.id == id.uuidString }) else { throw unavailable() }
            record = .preparation(entry)
        case "backend":
            guard let entry = host.backendHistory.entries.first(where: { $0.id == id.uuidString }) else { throw unavailable() }
            record = .backend(entry)
        default: throw unavailable()
        }
        return try await load(root: workspace.workspaceDirectory, projectID: host.owner.projectID, record: record)
    }
    @MainActor static func load(root: URL, projectID: String, record: Record) async throws -> Snapshot {
        guard UUID(uuidString: projectID)?.uuidString == projectID else { throw unavailable() }
        try Task.checkCancellation()
        switch record {
        case .training(let reference):
            guard reference.projectID == projectID else { throw unavailable() }
            return try await WorkflowHistoryVerificationWork.run {
                try Task.checkCancellation()
                guard case .verified(let value) = WorkflowHistoryReader.load(reference, root: root) else { throw unavailable() }
                let base = ["workflow-runs", projectID, reference.directoryID.uuidString]
                let files = try readPinnedSources(root: root, base: base, artifacts: value.outcome.artifacts)
                guard !files.isEmpty else { throw WorkflowArchiveError("This recorded run has no retained Python source.") }
                return Snapshot(files: files, originLabel: "Recorded training run", inferenceOnly: false)
            }
        case .preparation(let entry):
            let verified = try await WorkflowDataPreparationHistory(root: root, projectID: projectID).verify(entry)
            return try await Task.detached(priority: .userInitiated) {
                let files = try readPinnedSources(root: root, base: ["workflow-operations", projectID, verified.id, "artifacts"], artifacts: verified.artifacts)
                guard !files.isEmpty else { throw WorkflowArchiveError("This preparation has no retained Python source.") }
                return Snapshot(files: files, originLabel: "Recorded preparation", inferenceOnly: false)
            }.value
        case .backend(let entry):
            let verified = try await BackendFitHistoryStore(root: root, projectID: projectID).verify(entry)
            return try await Task.detached(priority: .userInitiated) {
                // Backend archives retain portable inference, not a reproducible training script.
                let artifacts = verified.artifacts.filter { $0.path.hasPrefix("portable/") }
                let files = try readPinnedSources(root: root, base: ["workflow-operations", projectID, verified.id, "artifacts"], artifacts: artifacts)
                guard !files.isEmpty else { throw WorkflowArchiveError("This backend run has no retained inference source.") }
                return Snapshot(files: files, originLabel: "Recorded backend · inference only", inferenceOnly: true)
            }.value
        }
    }
    /// Rechecks the actual bytes after authoritative history verification. Caller-selected
    /// paths never replace descriptor-relative reads or the receipt's content hashes.
    static func readPinnedSources(root: URL, base: [String], artifacts: [WorkflowArchiveFile]) throws -> [String: String] {
        var files: [String: String] = [:], total = 0
        for file in artifacts where file.path.hasSuffix(".py") {
            try Task.checkCancellation()
            guard files.count < 128, file.bytes >= 0, file.bytes <= 2_000_000,
                  file.bytes <= 8_000_000 - total, files[file.path] == nil else { throw unavailable() }
            let parts = try WorkflowArchiveCodec.pathParts(file.path)
            let bytes = try DiagnosticFileIO.read(root: root, parts: base + parts, limit: 2_000_000)
            guard bytes.count == file.bytes, WorkflowArchiveCodec.hash(bytes) == file.sha256,
                  let source = String(data: bytes, encoding: .utf8) else { throw unavailable() }
            files[file.path] = source; total += bytes.count
        }
        try Task.checkCancellation()
        return files
    }
    private static func unavailable() -> WorkflowArchiveError {
        WorkflowArchiveError("Recorded code is unavailable or its retained evidence changed. Refresh History and select a verified record.")
    }
}
